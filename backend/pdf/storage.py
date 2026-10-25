"""
Thread-Safe Atomic Artifact Storage & Registry for Mark PDF Engine.
Enforces opaque UUIDs, strict containment within ARTIFACTS_DIR, path sanitization,
atomic write-replace, revision history, and startup cleanup.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Dict, List, Optional

from config.pdf_config import ARTIFACTS_DIR
from backend.pdf.schema import DocumentModel

# Windows reserved device names
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

# UUID4 validation pattern
UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def sanitize_download_filename(title: str, max_len: int = 80) -> str:
    """
    Sanitizes human-readable title for safe usage in Content-Disposition header.
    Strips path separators, control characters, Windows reserved names, trailing/leading dots/spaces.
    """
    if not title or not isinstance(title, str):
        title = "document"

    # Normalize unicode and strip path separators / control chars
    title = title.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
    title = title.replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")

    # Allow alphanumeric, spaces, dashes, underscores, periods
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\. ]", "", title).strip()

    # Eliminate path traversal sequences like '..'
    cleaned = re.sub(r"\.{2,}", "_", cleaned)

    # Replace reserved Windows device names
    cleaned = re.sub(r"\b(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\b", r"doc_\1", cleaned, flags=re.IGNORECASE)

    # Collapse repeated whitespace or underscores
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"_+", "_", cleaned).strip()

    # Strip leading and trailing dots, spaces, dashes (prevents hidden/traversal names)
    cleaned = cleaned.strip(". -_")

    # Check for empty or dots-only name
    if not cleaned or re.match(r"^\.+$", cleaned):
        cleaned = "document"

    # Check for Windows reserved names as stem
    base_stem = cleaned.split(".")[0].upper()
    if base_stem in WINDOWS_RESERVED_NAMES:
        cleaned = f"doc_{cleaned}"

    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip(". -_")

    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"

    return cleaned


@dataclass
class ArtifactRecord:
    artifact_id: str
    kind: str = "pdf"
    display_name: str = "document.pdf"
    size_bytes: int = 0
    page_count: int = 0
    sha256: str = ""
    created_at: str = ""
    version: int = 1
    parent_artifact_id: Optional[str] = None
    status: str = "generating"  # generating, ready, failed
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    source_json_file: str = ""
    pdf_file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArtifactStorage:
    """
    Manages filesystem storage and registry indexing for generated artifacts.
    Guarantees thread-safe atomic operations and path confinement.
    """
    def __init__(self, root_dir: Path = ARTIFACTS_DIR):
        self.root_dir = root_dir.resolve()
        self.registry_file = self.root_dir / "registry.json"
        self._lock = threading.RLock()
        self._ensure_dir()

    def _ensure_dir(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            self._write_registry({})

    def _read_registry(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            if not self.registry_file.exists():
                return {}
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

    def _write_registry(self, data: Dict[str, Dict[str, Any]]):
        with self._lock:
            tmp_path = self.root_dir / f"registry_{os.getpid()}_{time.time_ns()}.tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, self.registry_file)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    def check_disk_space(self, required_bytes: int = 50 * 1024 * 1024) -> bool:
        """Verify at least required_bytes (default 50 MB) free space on target disk."""
        try:
            usage = shutil.disk_usage(self.root_dir)
            return usage.free >= required_bytes
        except Exception:
            return True

    def verify_path_containment(self, target_path: Path) -> Path:
        """
        Verify target path is strictly contained within ARTIFACTS_DIR.
        Raises ValueError if path traversal is attempted.
        """
        resolved = target_path.resolve()
        try:
            resolved.relative_to(self.root_dir)
        except ValueError:
            raise ValueError(f"Security error: Path '{resolved}' escapes artifact root directory '{self.root_dir}'.")
        return resolved

    def register_generating(
        self,
        artifact_id: str,
        display_name: str,
        version: int = 1,
        parent_artifact_id: Optional[str] = None,
    ) -> ArtifactRecord:
        """Register an artifact with status 'generating'."""
        if not UUID4_PATTERN.match(artifact_id):
            raise ValueError(f"Invalid artifact ID format: {artifact_id}. Must be UUID4.")

        clean_name = sanitize_download_filename(display_name)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            kind="pdf",
            display_name=clean_name,
            version=version,
            parent_artifact_id=parent_artifact_id,
            status="generating",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            pdf_file=f"{artifact_id}.pdf",
            source_json_file=f"{artifact_id}.json",
        )

        with self._lock:
            reg = self._read_registry()
            reg[artifact_id] = record.to_dict()
            self._write_registry(reg)

        return record

    def commit_ready_artifact(
        self,
        artifact_id: str,
        pdf_bytes: bytes,
        source_doc: DocumentModel,
        page_count: int,
        warnings: Optional[List[str]] = None,
    ) -> ArtifactRecord:
        """
        Atomically writes PDF and source document JSON to disk, computes SHA256,
        and marks artifact as 'ready' in registry.
        """
        if not UUID4_PATTERN.match(artifact_id):
            raise ValueError(f"Invalid artifact ID format: {artifact_id}.")

        if not self.check_disk_space():
            raise IOError("Insufficient disk space on artifact storage volume.")

        target_pdf = self.verify_path_containment(self.root_dir / f"{artifact_id}.pdf")
        target_json = self.verify_path_containment(self.root_dir / f"{artifact_id}.json")

        # Atomic write via temp file
        tmp_pdf = self.root_dir / f"{artifact_id}_{time.time_ns()}.pdf.tmp"
        tmp_json = self.root_dir / f"{artifact_id}_{time.time_ns()}.json.tmp"

        sha256_hash = hashlib.sha256(pdf_bytes).hexdigest()

        try:
            with open(tmp_pdf, "wb") as pf:
                pf.write(pdf_bytes)
            with open(tmp_json, "w", encoding="utf-8") as jf:
                json.dump(source_doc.to_dict(), jf, indent=2)

            os.replace(tmp_pdf, target_pdf)
            os.replace(tmp_json, target_json)
        except Exception:
            if tmp_pdf.exists():
                try: tmp_pdf.unlink()
                except OSError: pass
            if tmp_json.exists():
                try: tmp_json.unlink()
                except OSError: pass
            raise

        with self._lock:
            reg = self._read_registry()
            cur = reg.get(artifact_id, {})
            cur.update({
                "status": "ready",
                "size_bytes": len(pdf_bytes),
                "page_count": page_count,
                "sha256": sha256_hash,
                "warnings": warnings or [],
                "error_message": None,
            })
            reg[artifact_id] = cur
            self._write_registry(reg)
            return ArtifactRecord(**cur)

    def mark_failed(self, artifact_id: str, error_message: str):
        """Mark artifact as failed and clean up partial files."""
        with self._lock:
            reg = self._read_registry()
            cur = reg.get(artifact_id, {"artifact_id": artifact_id})
            cur["status"] = "failed"
            cur["error_message"] = error_message
            reg[artifact_id] = cur
            self._write_registry(reg)

        # Remove partial files if any
        pdf_path = self.root_dir / f"{artifact_id}.pdf"
        json_path = self.root_dir / f"{artifact_id}.json"
        for p in (pdf_path, json_path):
            if p.exists():
                try: p.unlink()
                except OSError: pass

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Retrieve artifact record by opaque UUID."""
        if not artifact_id or not UUID4_PATTERN.match(artifact_id):
            return None
        reg = self._read_registry()
        data = reg.get(artifact_id)
        if data:
            return ArtifactRecord(**data)
        return None

    def get_artifact_pdf_path(self, artifact_id: str) -> Optional[Path]:
        """Retrieve verified path to PDF file if status is ready."""
        rec = self.get_artifact(artifact_id)
        if not rec or rec.status != "ready":
            return None
        pdf_path = self.verify_path_containment(self.root_dir / f"{artifact_id}.pdf")
        if pdf_path.is_file():
            return pdf_path
        return None

    def get_artifact_source_doc(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored structured JSON document for an artifact."""
        rec = self.get_artifact(artifact_id)
        if not rec:
            return None
        json_path = self.verify_path_containment(self.root_dir / f"{artifact_id}.json")
        if json_path.is_file():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def list_artifacts(self, limit: int = 50) -> List[ArtifactRecord]:
        """List recent artifacts (metadata only)."""
        reg = self._read_registry()
        records = [ArtifactRecord(**data) for data in reg.values()]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    def cleanup_stale_jobs(self, max_stale_seconds: float = 300.0):
        """
        Cleanup stale 'generating' jobs and orphaned .tmp files at startup.
        Never deletes finished 'ready' artifacts.
        """
        with self._lock:
            # 1. Clean orphaned temp files
            for tmp in self.root_dir.glob("*.tmp"):
                try:
                    tmp.unlink()
                except OSError:
                    pass

            # 2. Mark stale generating records as failed
            reg = self._read_registry()
            changed = False
            for art_id, data in reg.items():
                if data.get("status") == "generating":
                    data["status"] = "failed"
                    data["error_message"] = "Server restarted while document was generating."
                    changed = True
            if changed:
                self._write_registry(reg)


# Global storage instance
storage = ArtifactStorage()
