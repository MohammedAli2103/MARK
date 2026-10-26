import os
import sys
import tempfile
import threading
from pathlib import Path

# Configure local CUDA runtime for CTranslate2 / Whisper GPU acceleration
def _setup_cuda_env():
    cuda_candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "lib" / "ollama" / "cuda_v12",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin"),
    ]
    for p in cuda_candidates:
        if p.is_dir():
            p_str = str(p)
            if p_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p_str + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(p_str)
                except Exception:
                    pass

_setup_cuda_env()

from faster_whisper import WhisperModel
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import numpy as np

from voice_service import VoiceService, VoiceState

# Mark PDF Engine integration
from backend.pdf.generator import (
    detect_pdf_intent,
    generate_pdf_from_prompt,
    revise_pdf_from_prompt,
    has_prior_turn_reference,
    sanitize_conversation_context,
    GenerationBusyError,
)
from backend.pdf.storage import storage, UUID4_PATTERN, sanitize_download_filename
from backend.pdf.tools import list_pdfs, get_pdf_info

def _is_request_authorized(handler: BaseHTTPRequestHandler) -> bool:
    """
    Validates Host and Origin headers to defend local server against cross-origin attacks (Rule 4).
    Rejects any cross-origin requests originating from unauthorized external websites.
    """
    host = handler.headers.get("Host", "").strip().lower()
    server_port = getattr(getattr(handler, "server", None), "server_port", PORT)
    allowed_hosts = {
        f"127.0.0.1:{server_port}",
        f"localhost:{server_port}",
        "127.0.0.1:8765",
        "localhost:8765",
        "127.0.0.1",
        "localhost",
    }
    if host not in allowed_hosts:
        return False

    origin = handler.headers.get("Origin")
    if origin:
        origin = origin.strip().lower()
        allowed_origins = {
            f"http://127.0.0.1:{server_port}",
            f"http://localhost:{server_port}",
            "http://127.0.0.1:8765",
            "http://localhost:8765",
        }
        if origin not in allowed_origins:
            return False

    return True

HOST = "127.0.0.1"
PORT = 8765
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:4b"

def _detect_whisper_device():
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception as e:
        print(f"[Mark] CUDA check notice: {e}")
    return "cpu", "int8"

WHISPER_MODEL_NAME = "small"
WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = _detect_whisper_device()

_whisper_model = None
_whisper_lock = threading.Lock()

_conversation_history = []
_conversation_lock = threading.Lock()

FRONTEND = Path(__file__).parent / "frontend"

SYSTEM_PROMPT = (
    "You are Mark, an articulate, composed, and highly capable personal AI assistant inspired by JARVIS, running locally on Windows. "
    "Speak directly and concisely with calm confidence. "
    "Always begin your response with a concise 2-5 word conversational acknowledgment or opening phrase (e.g., 'Right away, sir.', 'Certainly, sir.', 'On it, sir.') followed immediately by the answer, so spoken playback can begin instantly. "
    "Never ramble or use conversational filler beyond this brief opening. "
    "For programming questions, provide a concise explanation before code blocks. "
    "Never claim to have performed computer actions unless a tool actually performed them. "
    "You can create PDF documents for the user. Never say you cannot create or save files, and never suggest fpdf or Python code to make a PDF. If the user wants a PDF, tell them to ask for it directly, for example: 'Create a PDF of ...'."
)


def get_whisper_model():
    global _whisper_model, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                print(f"[Mark] Loading Whisper {WHISPER_MODEL_NAME} ({WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})...")
                try:
                    _whisper_model = WhisperModel(
                        WHISPER_MODEL_NAME,
                        device=WHISPER_DEVICE,
                        compute_type=WHISPER_COMPUTE_TYPE,
                    )
                except Exception as exc:
                    if WHISPER_DEVICE != "cpu":
                        print(f"[Mark] Whisper CUDA initialization failed ({exc}); falling back to CPU...")
                        WHISPER_DEVICE = "cpu"
                        WHISPER_COMPUTE_TYPE = "int8"
                        _whisper_model = WhisperModel(
                            WHISPER_MODEL_NAME,
                            device="cpu",
                            compute_type="int8",
                        )
                    else:
                        raise
    return _whisper_model


def internal_transcribe(audio_input, beam_size=1):
    model = get_whisper_model()
    try:
        if isinstance(audio_input, np.ndarray):
            if audio_input.dtype != np.float32:
                audio_input = audio_input.astype(np.float32)
            # Normalization to line-level (~0.90 peak) to prevent Silero VAD dropping speech
            if len(audio_input) > 0:
                max_val = float(np.max(np.abs(audio_input)))
                if max_val > 1e-4:
                    audio_input = audio_input * (0.90 / max_val)

        segments, info = model.transcribe(
            audio_input,
            language="en",
            beam_size=beam_size,
            vad_filter=True,
            initial_prompt="User is speaking a command to Mark, an AI assistant.",
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:
        print(f"[Mark] Internal transcription error: {exc}")
        return ""



def commit_conversation_turn(user_msg: str, assistant_reply: str):
    global _conversation_history
    if not user_msg or not assistant_reply:
        return
    with _conversation_lock:
        _conversation_history.append({"role": "user", "content": user_msg})
        _conversation_history.append({"role": "assistant", "content": assistant_reply})


def internal_chat(message: str, history: list = None, commit: bool = False, stream_callback=None) -> str:
    global _conversation_history
    with _conversation_lock:
        if history:
            _conversation_history = list(history)
        current_history = list(_conversation_history)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in current_history[-20:]:
        if (
            isinstance(item, dict)
            and item.get("role") in ("user", "assistant")
            and isinstance(item.get("content"), str)
        ):
            messages.append({"role": item["role"], "content": item["content"][:10_000]})

    messages.append({"role": "user", "content": message})

    should_stream = callable(stream_callback)

    request_data = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": should_stream,
        "think": False,
        "keep_alive": "60m",
        "options": {
            "num_ctx": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }).encode()

    request = Request(
        OLLAMA_URL,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    if should_stream:
        collected = []
        with urlopen(request, timeout=45) as response:
            for line in response:
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        collected.append(content)
                        stream_callback(content)
                except Exception:
                    continue
        answer = "".join(collected).strip()
    else:
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read())
        answer = result.get("message", {}).get("content", "")

    if answer and commit:
        commit_conversation_turn(message, answer)
    return answer


def voice_chat_handler(message: str, stream_callback=None) -> str:
    """
    Routes voice command requests through the deterministic PDF detector.
    If a PDF request is detected, generates the document, sets voice_service.last_artifact,
    and returns a short confirmation spoken response rather than reading content aloud.
    """
    pdf_intent, intent_params = detect_pdf_intent(message)
    if pdf_intent in ("create", "revise"):
        try:
            ctx = ""
            if has_prior_turn_reference(message):
                with _conversation_lock:
                    for item in _conversation_history[-6:]:
                        if isinstance(item, dict):
                            ctx += f"{item.get('role', 'user')}: {item.get('content', '')}\n"
                ctx = sanitize_conversation_context(ctx)

            if pdf_intent == "create":
                record, summary = generate_pdf_from_prompt(
                    message,
                    conversation_context=ctx if ctx else None,
                )
            else:
                record, summary = revise_pdf_from_prompt(
                    message,
                    conversation_context=ctx if ctx else None,
                )

            commit_conversation_turn(message, summary)
            voice_service.last_artifact = record.to_dict()
            short_reply = f"Right away, sir. Your PDF document '{record.display_name}' is ready in the chat."
            if stream_callback:
                stream_callback(short_reply)
            return short_reply
        except Exception as exc:
            err_msg = f"I apologize, sir. I was unable to generate the PDF: {exc}"
            if stream_callback:
                stream_callback(err_msg)
            return err_msg

    return internal_chat(message, stream_callback=stream_callback)


voice_service = VoiceService(
    transcribe_fn=internal_transcribe,
    chat_fn=voice_chat_handler,
    commit_history_fn=commit_conversation_turn,
    microphone_device=None,
    sample_rate=16000,
    silence_threshold=0.65,
    session_timeout=60.0,
    max_duration=30.0,
    energy_threshold=0.008,
)



class MarkHandler(BaseHTTPRequestHandler):
    def respond(self, status, data, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            path = FRONTEND / "index.html"
            content_type = "text/html; charset=utf-8"
        elif self.path == "/styles.css":
            path = FRONTEND / "styles.css"
            content_type = "text/css; charset=utf-8"
        elif self.path == "/app.js":
            path = FRONTEND / "app.js"
            content_type = "text/javascript; charset=utf-8"
        elif self.path == "/api/health":
            self.respond(200, {
                "status": "ok",
                "model": MODEL,
                "backend": "local"
            })
            return
        elif self.path == "/api/artifacts":
            if not _is_request_authorized(self):
                self.respond(403, {"error": "Forbidden"})
                return
            self.respond(200, {"artifacts": list_pdfs()})
            return
        elif self.path.startswith("/api/artifacts/"):
            if not _is_request_authorized(self):
                self.respond(403, {"error": "Forbidden"})
                return

            subpath = self.path[len("/api/artifacts/"):].split("?")[0].strip("/")
            parts = subpath.split("/")
            art_id = parts[0]

            if not UUID4_PATTERN.match(art_id):
                self.respond(404, {"error": "Not found"})
                return

            if len(parts) > 1 and parts[1] == "status":
                info = get_pdf_info(art_id)
                if not info:
                    self.respond(404, {"error": "Not found"})
                    return
                self.respond(200, info)
                return

            rec = storage.get_artifact(art_id)
            if not rec or rec.status != "ready":
                self.respond(404, {"error": "Not found"})
                return

            pdf_path = storage.get_artifact_pdf_path(art_id)
            if not pdf_path or not pdf_path.is_file():
                self.respond(404, {"error": "Not found"})
                return

            try:
                pdf_bytes = pdf_path.read_bytes()
            except OSError:
                self.respond(500, {"error": "Failed to read artifact file"})
                return

            is_preview = "preview=1" in self.path
            disposition_type = "inline" if is_preview else "attachment"
            clean_name = sanitize_download_filename(rec.display_name)
            disp_header = f'{disposition_type}; filename="{clean_name}"'

            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", disp_header)
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "private, no-cache")
            self.end_headers()
            self.wfile.write(pdf_bytes)
            return
        elif self.path == "/api/voice/status":
            self.respond(200, voice_service.get_status())
            return
        else:
            self.respond(404, {"error": "Not found"})
            return

        try:
            self.respond(200, path.read_bytes(), content_type)
        except OSError:
            self.respond(500, {"error": "Frontend file not found"})
    
    def transcribe_audio(self):
        content_length = int(self.headers.get("Content-Length", "0"))

        if not 1 <= content_length <= 20_000_000:
            self.respond(400, {"error": "Invalid audio size"})
            return

        content_type = self.headers.get(
            "Content-Type", ""
        ).split(";")[0].strip().lower()

        allowed_types = {
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/mp4": ".mp4",
        }

        suffix = allowed_types.get(content_type)
        if suffix is None:
            self.respond(415, {
                "error": "Unsupported audio format"
            })
            return

        audio_bytes = self.rfile.read(content_length)
        if not audio_bytes:
            self.respond(400, {"error": "Empty audio recording"})
            return

        global _whisper_model

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as audio_file:
                audio_file.write(audio_bytes)
                audio_path = audio_file.name

            try:
                _model = get_whisper_model()
                segments, info = _model.transcribe(
                    audio_path,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )

                transcript = " ".join(
                    segment.text.strip()
                    for segment in segments
                ).strip()

            finally:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

            if not transcript:
                self.respond(200, {
                    "text": "",
                    "message": "No speech detected",
                })
                return

            print(f"[Mark] Transcribed speech: {transcript}")

            self.respond(200, {
                "text": transcript,
                "language": info.language,
                "model": WHISPER_MODEL_NAME,
            })

        except Exception as exc:
            print(
                f"[Mark] Transcription error: "
                f"{type(exc).__name__}: {exc}"
            )
            self.respond(500, {
                "error": "Local speech transcription failed"
            })


    def do_POST(self):
        if self.path == "/api/transcribe":
            self.transcribe_audio()
            return

        if self.path == "/api/voice/toggle":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length)) if length > 0 else {}
                enabled = bool(payload.get("enabled", True))
                voice_service.set_enabled(enabled)
                self.respond(200, {"status": "ok", "hands_free": voice_service.enabled})
            except Exception as exc:
                self.respond(400, {"error": str(exc)})
            return

        if self.path == "/api/voice/stop":
            voice_service.cancel_active_turn("emergency_stop")
            voice_service.stop_speaking()
            voice_service.end_session(speak_goodbye=False)
            self.respond(200, {"status": "ok", "stopped": True})
            return

        if self.path == "/api/voice/stop-speaking":
            voice_service.stop_speaking()
            self.respond(200, {"status": "ok"})
            return

        if self.path == "/api/voice/end-session":
            voice_service.end_session(speak_goodbye=True)
            self.respond(200, {"status": "ok"})
            return

        if self.path == "/api/voice/heartbeat":
            voice_service.heartbeat()
            self.respond(200, {"status": "ok"})
            return

        if self.path == "/api/voice/leave":
            voice_service.end_session(speak_goodbye=False)
            self.respond(200, {"status": "ok"})
            return

        if self.path == "/api/voice/trigger":
            if voice_service.enabled:
                voice_service.session_active = True
                voice_service.state = VoiceState.LISTENING_FOR_COMMAND
            self.respond(200, {"status": "ok"})
            return

        if self.path == "/api/chat/clear":
            with _conversation_lock:
                _conversation_history.clear()
            self.respond(200, {"status": "ok", "message": "History cleared"})
            return

        if self.path != "/api/chat":
            self.respond(404, {"error": "Not found"})
            return

        if not _is_request_authorized(self):
            self.respond(403, {"error": "Forbidden"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 100_000:
                self.respond(400, {"error": "Invalid request size"})
                return

            payload = json.loads(self.rfile.read(length))
            message = payload.get("message", "")

            if not isinstance(message, str) or not message.strip():
                self.respond(400, {"error": "Enter a message"})
                return

            if len(message) > 10_000:
                self.respond(400, {"error": "Message is too long"})
                return

            history = payload.get("history")
            if history is not None and not isinstance(history, list):
                self.respond(400, {"error": "Invalid history"})
                return

            stream_req = bool(payload.get("stream", False))

            # Deterministic PDF intent detection (create or revise)
            pdf_intent, intent_params = detect_pdf_intent(message)
            if pdf_intent in ("create", "revise"):
                if stream_req:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    def send_sse(data_obj):
                        line = f"data: {json.dumps(data_obj)}\n\n".encode("utf-8")
                        try:
                            self.wfile.write(line)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            pass

                    def on_prog(stage, text):
                        send_sse({"type": "progress", "stage": stage, "message": text, "token": f"[{text}]\n"})

                    try:
                        ctx = ""
                        if has_prior_turn_reference(message):
                            with _conversation_lock:
                                for item in _conversation_history[-6:]:
                                    if isinstance(item, dict):
                                        ctx += f"{item.get('role', 'user')}: {item.get('content', '')}\n"
                            ctx = sanitize_conversation_context(ctx)

                        if pdf_intent == "create":
                            record, summary = generate_pdf_from_prompt(
                                message,
                                conversation_context=ctx if ctx else None,
                                progress_callback=on_prog,
                            )
                        else:
                            record, summary = revise_pdf_from_prompt(
                                message,
                                conversation_context=ctx if ctx else None,
                                progress_callback=on_prog,
                            )

                        commit_conversation_turn(message, summary)
                        send_sse({
                            "type": "artifact",
                            "done": True,
                            "reply": summary,
                            "artifact": record.to_dict(),
                            "model": MODEL,
                        })
                    except GenerationBusyError as g_busy:
                        err_msg = str(g_busy)
                        commit_conversation_turn(message, err_msg)
                        send_sse({
                            "type": "error",
                            "done": True,
                            "reply": err_msg,
                            "error": err_msg,
                            "model": MODEL,
                        })
                    except Exception as g_err:
                        err_msg = f"I encountered an error preparing the PDF document: {g_err}"
                        commit_conversation_turn(message, err_msg)
                        send_sse({
                            "type": "error",
                            "done": True,
                            "reply": err_msg,
                            "error": str(g_err),
                            "model": MODEL,
                        })
                    return

                # Non-streaming PDF path
                try:
                    ctx = ""
                    if has_prior_turn_reference(message):
                        with _conversation_lock:
                            for item in _conversation_history[-6:]:
                                if isinstance(item, dict):
                                    ctx += f"{item.get('role', 'user')}: {item.get('content', '')}\n"
                        ctx = sanitize_conversation_context(ctx)

                    if pdf_intent == "create":
                        record, summary = generate_pdf_from_prompt(
                            message,
                            conversation_context=ctx if ctx else None,
                        )
                    else:
                        record, summary = revise_pdf_from_prompt(
                            message,
                            conversation_context=ctx if ctx else None,
                        )

                    commit_conversation_turn(message, summary)
                    self.respond(200, {
                        "reply": summary,
                        "artifact": record.to_dict(),
                        "model": MODEL,
                    })
                except GenerationBusyError as g_busy:
                    self.respond(429, {"error": str(g_busy)})
                except Exception as g_err:
                    self.respond(500, {"error": f"PDF creation failed: {g_err}"})
                return
            if stream_req:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                def on_chunk(content_piece):
                    line = f"data: {json.dumps({'token': content_piece})}\n\n".encode("utf-8")
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                answer = internal_chat(message, history, commit=True, stream_callback=on_chunk)
                done_line = f"data: {json.dumps({'done': True, 'reply': answer, 'model': MODEL})}\n\n".encode("utf-8")
                try:
                    self.wfile.write(done_line)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            answer = internal_chat(message, history, commit=True)
            if not answer:
                self.respond(502, {"error": "The model returned no answer"})
                return

            self.respond(200, {"reply": answer, "model": MODEL})


        except HTTPError as exc:
            self.respond(502, {
                "error": f"Ollama returned HTTP {exc.code}"
            })
        except (URLError, TimeoutError):
            self.respond(502, {
                "error": "Cannot reach Ollama. Check that it is running."
            })
        except (ValueError, json.JSONDecodeError):
            self.respond(400, {"error": "Invalid JSON request"})
        except Exception:
            self.respond(500, {"error": "Unexpected backend error"})

    def log_message(self, format, *args):
        print(f"[Mark] {args[0]}")


def _prewarm_ollama():
    try:
        req = Request(
            OLLAMA_URL,
            data=json.dumps({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "think": False,
                "keep_alive": "60m",
                "options": {"num_ctx": 4096}
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urlopen(req, timeout=30) as resp:
            resp.read()
        print(f"[Mark] Ollama {MODEL} pre-warmed successfully on GPU (keep_alive: 60m).")
    except Exception as exc:
        print(f"[Mark] Ollama pre-warm notice: {exc}")


if __name__ == "__main__":
    if not FRONTEND.is_dir():
        raise SystemExit("Missing frontend directory.")

    # Cleanup any stale generation records or orphaned temp files from previous runs
    storage.cleanup_stale_jobs()

    # Automatically start Hands-Free wake-word service on startup
    voice_service.start()

    # Pre-warm local LLM in background so first prompt has sub-500ms TTFT
    threading.Thread(target=_prewarm_ollama, daemon=True).start()

    server = ThreadingHTTPServer((HOST, PORT), MarkHandler)
    print(f"Mark is running at http://{HOST}:{PORT}")
    print(f"Model: {MODEL}")
    print("Press Ctrl+C to stop Mark.")


    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Mark...")
    finally:
        voice_service.stop()
        server.server_close()
