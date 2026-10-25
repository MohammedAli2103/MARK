/**
 * MARK — Holo Table V3 Production HUD Controller
 * Autonomous neural interface connecting real backend services:
 * - Ollama (Qwen 3.5 4B)
 * - Faster-Whisper (speech transcription)
 * - Kokoro-82M (neural voice synthesis)
 * - Continuous Hands-Free Session Manager
 */

(() => {
  'use strict';

  /* ════════════════════════════════════════════════════════════
     1. SELECTORS & DOM REFS
     ════════════════════════════════════════════════════════════ */
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const NS = 'http://www.w3.org/2000/svg';

  const els = {
    live: $('#live'),
    connPill: $('#conn'),
    connTxt: $('#connTxt'),
    hfBtn: $('#hf'),
    clearBtn: $('#clear'),

    // User Column
    youTag: $('#youTag'),
    youCount: $('#youCount'),
    youStream: $('#youStream'),
    liveCaptureCard: $('#liveCaptureCard'),
    captureTitle: $('#captureTitle'),
    liveTranscript: $('#liveTranscript'),
    liveTent: $('#liveTent'),
    stopListen: $('#stopListen'),
    cancelListen: $('#cancelListen'),

    // Stage SVG Elements
    holo: $('#holo'),
    floor: $('#floor'),
    padrings: $('#padrings'),
    parts: $('#parts'),
    bars: $('#bars'),
    lat: $('#lat'),
    meridians: $('#meridians'),
    mesh: $('#mesh'),
    spikes: $('#spikes'),
    think: $('#think'),
    prog: $('#prog'),
    nodes: $('#nodes'),
    label: $('#label'),
    sub: $('#sub'),
    pipeStt: $('#pipeStt'),
    pipeLlm: $('#pipeLlm'),
    pipeTts: $('#pipeTts'),
    pipeSttText: $('#pipeSttText'),
    pipeLlmText: $('#pipeLlmText'),
    pipeTtsText: $('#pipeTtsText'),

    // Mark Column
    markTag: $('#markTag'),
    markMeta: $('#markMeta'),
    markStream: $('#markStream'),
    liveThinkingCard: $('#liveThinkingCard'),
    liveSpeakingCard: $('#liveSpeakingCard'),
    speakingStatusText: $('#speakingStatusText'),
    liveOfflineCard: $('#liveOfflineCard'),
    reconnect: $('#reconnect'),
    interrupt: $('#interrupt'),
    stopThink: $('#stopThink'),
    task: $('#task'),
    perm: $('#perm'),
    permInfo: $('#permInfo'),
    permAllowOnce: $('#permAllowOnce'),
    permAllowTask: $('#permAllowTask'),
    permDeny: $('#permDeny'),

    // Footer Composer
    form: $('#form'),
    msg: $('#msg'),
    mic: $('#mic'),
  };

  const mk = (n, a = {}, p) => {
    const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]);
    if (p) p.appendChild(e);
    return e;
  };

  const reduced = matchMedia('(prefers-reduced-motion: reduce)');

  function formatTime() {
    const now = new Date();
    return now.toTimeString().slice(0, 5);
  }

  /* ════════════════════════════════════════════════════════════
     2. BACKEND API ADAPTER
     ════════════════════════════════════════════════════════════ */
  const Backend = {
    isOnline: false,
    model: 'Qwen 3.5 4B',

    async checkHealth(signal) {
      try {
        const res = await fetch('/api/health', { signal, cache: 'no-store' });
        if (!res.ok) throw new Error(`Health check returned HTTP ${res.status}`);
        const data = await res.json();
        this.isOnline = true;
        if (data.model) this.model = data.model;
        return true;
      } catch {
        this.isOnline = false;
        return false;
      }
    },

    async ask(message, history = [], onChunk = null, signal) {
      const isStreaming = typeof onChunk === 'function';
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history, stream: isStreaming }),
        signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }

      if (isStreaming && res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let fullReply = '';
        let artifactData = null;
        let buf = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n\n');
          buf = lines.pop() || '';

          for (const line of lines) {
            const clean = line.replace(/^data:\s*/, '').trim();
            if (!clean) continue;
            try {
              const parsed = jsonParseSafe(clean);
              if (parsed.type === 'progress') {
                onChunk('', fullReply, { type: 'progress', message: parsed.message, stage: parsed.stage });
              } else if (parsed.type === 'artifact') {
                fullReply = parsed.reply || fullReply;
                artifactData = parsed.artifact;
                onChunk('', fullReply, { type: 'artifact', artifact: parsed.artifact });
              } else if (parsed.type === 'error') {
                fullReply = parsed.reply || parsed.error || fullReply;
                onChunk('', fullReply, { type: 'error', error: parsed.error });
              } else if (parsed.token) {
                fullReply += parsed.token;
                onChunk(parsed.token, fullReply);
              } else if (parsed.done && parsed.reply) {
                fullReply = parsed.reply;
                if (parsed.artifact) artifactData = parsed.artifact;
              }
            } catch {}
          }
        }
        return { reply: fullReply, artifact: artifactData };
      } else {
        const data = await res.json();
        return { reply: data.reply || '', artifact: data.artifact || null };
      }
    },

    async getVoiceStatus(signal) {
      const res = await fetch('/api/voice/status', { signal, cache: 'no-store' });
      if (!res.ok) throw new Error(`Voice status returned HTTP ${res.status}`);
      return await res.json();
    },

    async toggleHandsFree(enabled, signal) {
      const res = await fetch('/api/voice/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
        signal,
      });
      if (!res.ok) throw new Error('Toggle failed');
      return await res.json();
    },

    async triggerVoice(signal) {
      try {
        await fetch('/api/voice/trigger', { method: 'POST', signal });
      } catch {}
    },

    async stopVoice(signal) {
      try {
        await fetch('/api/voice/stop', { method: 'POST', signal });
      } catch {}
    },

    async stopSpeaking(signal) {
      try {
        await fetch('/api/voice/stop-speaking', { method: 'POST', signal });
      } catch {}
    },

    async clearChat(signal) {
      try {
        await fetch('/api/chat/clear', { method: 'POST', signal });
      } catch {}
    },
  };

  function jsonParseSafe(str) {
    try {
      return JSON.parse(str);
    } catch {
      return {};
    }
  }

  /* ════════════════════════════════════════════════════════════
     3. STATE MACHINE
     ════════════════════════════════════════════════════════════ */
  const STATES = {
    idle: {
      label: 'STANDBY',
      sub: 'Waiting for “Hey Mark”',
      stage: null,
      live: 'Mark is on standby',
    },
    listening: {
      label: 'LISTENING',
      sub: 'Listening for your command',
      stage: 'stt',
      live: 'Mark is listening',
    },
    transcribing: {
      label: 'PROCESSING',
      sub: 'Decoding acoustic speech stream',
      stage: 'stt',
      live: 'Mark is transcribing speech',
    },
    thinking: {
      label: 'PROCESSING',
      sub: 'Local neural inference processing',
      stage: 'llm',
      live: 'Mark is processing response',
    },
    speaking: {
      label: 'SPEAKING',
      sub: 'Kokoro neural audio playback · Space to interrupt',
      stage: 'tts',
      live: 'Mark is speaking',
    },
    permission: {
      label: 'AWAITING PERMISSION',
      sub: 'Mark is waiting for authorization',
      stage: null,
      live: 'Authorization required',
    },
    offline: {
      label: 'OFFLINE',
      sub: 'Backend disconnected: 127.0.0.1:8765 unreachable',
      stage: null,
      live: 'Backend service offline',
    },
    error: {
      label: 'ERROR',
      sub: 'Hardware or service error',
      stage: null,
      live: 'Mark encountered an error',
    },
  };

  let currentState = 'idle';

  function setState(s, customSub) {
    if (!STATES[s]) s = 'idle';
    currentState = s;
    const cfg = STATES[s];

    document.body.dataset.state = s;
    if (els.label) els.label.textContent = cfg.label;
    if (els.sub) els.sub.textContent = customSub || cfg.sub;
    if (els.live) els.live.textContent = cfg.live;

    // Pipeline active badges
    $$('.pipe li').forEach(li => {
      li.classList.toggle('on', li.dataset.stage === cfg.stage);
    });

    // Mic button active indicator
    if (els.mic) {
      els.mic.setAttribute('aria-label', s === 'listening' ? 'Stop listening' : 'Start listening');
    }

    // Input state
    const isOff = s === 'offline';
    if (els.msg) {
      els.msg.disabled = isOff;
      els.msg.placeholder = isOff
        ? 'Backend disconnected: reconnect to send'
        : 'Say “Hey Mark”, or type here…';
    }

    // Header connection text
    if (els.connTxt) {
      els.connTxt.textContent = isOff
        ? 'BACKEND DISCONNECTED'
        : `BACKEND CONNECTED // ${Backend.model.toUpperCase()}`;
    }

    // Live operational card toggles
    if (els.liveCaptureCard) els.liveCaptureCard.hidden = s !== 'listening' && s !== 'transcribing';
    if (els.liveThinkingCard) els.liveThinkingCard.hidden = s !== 'thinking';
    if (els.liveSpeakingCard) els.liveSpeakingCard.hidden = s !== 'speaking';
    if (els.liveOfflineCard) els.liveOfflineCard.hidden = s !== 'offline';

    syncVisualizer();
    draw(performance.now());
  }

  /* ════════════════════════════════════════════════════════════
     4. HOLOGRAPHIC SCENE (SVG Construction)
     ════════════════════════════════════════════════════════════ */
  const CX = 500, SY = 250, R = 140;

  function buildScene() {
    // 1. Perspective Floor Grid
    [334, 342, 356, 378, 410, 452, 508, 580, 672].forEach(y => {
      mk('path', { d: `M0 ${y}H1000`, class: 'fl' }, els.floor);
    });
    for (let i = -7; i <= 7; i++) {
      mk('path', { d: `M500 330L${500 + i * 150} 700`, class: 'fl' }, els.floor);
    }

    // 2. Pad Rings
    const spin = mk('g', { class: 'pad-spin' }, els.padrings);
    mk('ellipse', { cx: 500, cy: 520, rx: 300, ry: 70, class: 'ring', 'stroke-opacity': '.75', 'stroke-width': '2' }, spin);
    mk('ellipse', { cx: 500, cy: 520, rx: 268, ry: 60, class: 'ring', 'stroke-opacity': '.5', 'stroke-width': '6', 'stroke-dasharray': '1 10' }, spin);
    mk('ellipse', { cx: 500, cy: 520, rx: 205, ry: 45, class: 'ring', 'stroke-opacity': '.35' }, els.padrings);
    mk('ellipse', { cx: 500, cy: 520, rx: 92, ry: 20, class: 'ring', 'stroke-opacity': '.85', 'stroke-width': '2' }, els.padrings);

    // 3. Ascending Energy Particles
    for (let i = 1; i <= 36; i++) {
      const c = mk('circle', {
        cx: 400 + (i * 97) % 200,
        cy: 512,
        r: (1 + (i % 3) * 0.6).toFixed(1),
        class: 'pt',
      }, els.parts);
      c.style.animationDuration = `${4 + (i % 5)}s`;
      c.style.animationDelay = `${(-i * 0.41).toFixed(2)}s`;
    }

    // 4. Core Sphere Latitude Rings
    [-100, -60, -20, 20, 60, 100].forEach(dy => {
      const rx = Math.sqrt(R * R - dy * dy);
      mk('ellipse', {
        cx: CX,
        cy: SY + dy,
        rx: rx.toFixed(1),
        ry: (rx / 5).toFixed(1),
        class: 'ring',
        'stroke-opacity': '.5',
        'stroke-width': '1.1',
      }, els.lat);
    });

    // 5. Breathing Meridians
    for (let i = 0; i < 6; i++) {
      const e = mk('ellipse', { cx: CX, cy: SY, rx: R, ry: R, class: 'mer' }, els.meridians);
      e.style.animationDelay = `${(-i * 16 / 6).toFixed(2)}s`;
    }

    // 6. 3D Fibonacci Neural Node Mesh
    const N = 44, t = 0.35, P = [], rr = R - 8;
    for (let i = 0; i < N; i++) {
      const y = 1 - 2 * (i + 0.5) / N;
      const r = Math.sqrt(1 - y * y);
      const th = i * 2.399963;
      const x = r * Math.cos(th);
      const z = r * Math.sin(th);
      P.push({
        x,
        y: y * Math.cos(t) - z * Math.sin(t),
        z: y * Math.sin(t) + z * Math.cos(t),
      });
    }
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const a = P[i], b = P[j];
        if (Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) < 0.5) {
          mk('line', {
            x1: CX + rr * a.x,
            y1: SY + rr * a.y,
            x2: CX + rr * b.x,
            y2: SY + rr * b.y,
            'stroke-opacity': (0.1 + 0.4 * ((a.z + b.z) / 2 + 1) / 2).toFixed(2),
            class: 'acc',
            'stroke-width': '1',
          }, els.mesh);
        }
      }
    }
    P.forEach((p, i) => {
      mk('circle', {
        cx: CX + rr * p.x,
        cy: SY + rr * p.y,
        r: (1.4 + 2 * (p.z + 1) / 2).toFixed(1),
        'fill-opacity': (0.25 + 0.7 * (p.z + 1) / 2).toFixed(2),
        class: 'accf' + (i % 7 === 0 ? ' tw' : ''),
      }, els.mesh);
    });

    // 7. Thinking Mode Segmented Arcs
    const C1 = 2 * Math.PI * (R + 34), C2 = 2 * Math.PI * (R + 52);
    const r1 = mk('g', { class: 'rot' }, els.think);
    mk('circle', {
      cx: CX, cy: SY, r: R + 34, class: 'ring',
      'stroke-width': '3', 'stroke-linecap': 'round',
      'stroke-dasharray': `${C1 * 0.17} ${C1}`,
    }, r1);
    mk('circle', {
      cx: CX, cy: SY, r: R + 34, class: 'ring',
      'stroke-width': '3', 'stroke-linecap': 'round',
      'stroke-dasharray': `${C1 * 0.07} ${C1}`,
      'stroke-dashoffset': -C1 * 0.5,
    }, r1);

    const r2 = mk('g', { class: 'rot2' }, els.think);
    mk('circle', {
      cx: CX, cy: SY, r: R + 52, class: 'ring',
      'stroke-width': '2', 'stroke-opacity': '.7', 'stroke-linecap': 'round',
      'stroke-dasharray': `${C2 * 0.11} ${C2}`,
    }, r2);

    // 8. Pipeline Arc Nodes
    [150, 90, 30].forEach(a => {
      const x = 500 + 300 * Math.cos(a * Math.PI / 180);
      const y = 520 + 70 * Math.sin(a * Math.PI / 180);
      mk('circle', {
        cx: x.toFixed(1), cy: y.toFixed(1), r: 7,
        fill: '#0A1220', class: 'ring', 'stroke-width': '2',
      }, els.nodes);
    });
  }

  /* ════════════════════════════════════════════════════════════
     5. AUDIO DYNAMICS & WAVEFORM RENDERING
     ════════════════════════════════════════════════════════════ */
  let amp = 0, raf = 0;

  function levelFor(t) {
    if (reduced.matches) {
      return (currentState === 'listening' || currentState === 'speaking') ? 0.35 : 0;
    }
    if (currentState === 'listening' || currentState === 'transcribing') {
      return 0.44 + 0.35 * Math.sin(t / 170) * Math.sin(t / 430 + 1);
    }
    if (currentState === 'speaking') {
      return 0.48 + 0.38 * Math.abs(Math.sin(t / 140)) * Math.abs(Math.sin(t / 390 + 0.5));
    }
    return 0;
  }

  function draw(t) {
    amp += (levelFor(t) - amp) * 0.25;
    const sA = (currentState === 'listening' || currentState === 'transcribing') ? 50 : currentState === 'speaking' ? 36 : 0;
    const bA = (currentState === 'listening' || currentState === 'transcribing') ? 44 : currentState === 'speaking' ? 30 : 0;
    const ph = t / 260;

    let d = '';
    if (sA) {
      for (let i = 0; i < 132; i++) {
        const a = 2 * Math.PI * i / 132;
        const h = sA * amp * (0.2 + 0.8 * (Math.sin(i * 1.7 + ph) * Math.sin(i * 0.53 + ph * 2) + 1) / 2);
        const r0 = 152;
        d += `M${(CX + r0 * Math.cos(a)).toFixed(1)} ${(SY + r0 * Math.sin(a)).toFixed(1)}L${(CX + (r0 + h) * Math.cos(a)).toFixed(1)} ${(SY + (r0 + h) * Math.sin(a)).toFixed(1)}`;
      }
    }
    if (els.spikes) els.spikes.setAttribute('d', d);

    let b = '';
    if (bA) {
      for (let i = 0; i < 150; i++) {
        const a = 2 * Math.PI * i / 150;
        const h = bA * amp * (0.15 + 0.85 * (Math.sin(i * 1.3 + ph) * Math.sin(i * 0.37 + ph * 3) + 1) / 2);
        const x = 500 + 322 * Math.cos(a);
        const y = 520 + 76 * Math.sin(a);
        const s = Math.sin(a);
        b += `M${x.toFixed(1)} ${y.toFixed(1)}L${x.toFixed(1)} ${(y - h * (0.5 + 0.5 * s * s)).toFixed(1)}`;
      }
    }
    if (els.bars) els.bars.setAttribute('d', b);
  }

  function loop(t) {
    draw(t);
    raf = requestAnimationFrame(loop);
  }

  function syncVisualizer() {
    const active = (currentState === 'listening' || currentState === 'transcribing' || currentState === 'speaking')
                   && !reduced.matches && !document.hidden;
    if (active && !raf) raf = requestAnimationFrame(loop);
    if (!active && raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
  }

  document.addEventListener('visibilitychange', syncVisualizer);
  reduced.addEventListener('change', () => {
    syncVisualizer();
    draw(performance.now());
  });

  /* ════════════════════════════════════════════════════════════
     6. CONVERSATION RENDERING & HISTORY STREAMS
     ════════════════════════════════════════════════════════════ */
  const conversationHistory = [];
  let userTurnCount = 0;

  function maybeScrollToBottom(streamEl) {
    if (!streamEl) return;
    const isNearBottom = streamEl.scrollHeight - streamEl.scrollTop - streamEl.clientHeight < 160;
    if (isNearBottom) {
      streamEl.scrollTo({ top: streamEl.scrollHeight, behavior: 'smooth' });
    }
  }

  function formatContent(text) {
    const wrap = document.createElement('div');
    wrap.className = 'turn-content';

    const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
    let lastIdx = 0;
    let match;

    if (codeBlockRegex.test(text)) {
      codeBlockRegex.lastIndex = 0;
      while ((match = codeBlockRegex.exec(text)) !== null) {
        const preText = text.substring(lastIdx, match.index);
        if (preText.trim()) {
          const p = document.createElement('div');
          p.style.marginBottom = '8px';
          p.textContent = preText;
          wrap.appendChild(p);
        }

        const lang = match[1] || 'CODE';
        const codeText = match[2];

        const codeBox = document.createElement('div');
        codeBox.className = 'code-wrap';

        const head = document.createElement('div');
        head.className = 'code-header';
        head.innerHTML = `<span>${lang.toUpperCase()}</span>`;

        const copyBtn = document.createElement('button');
        copyBtn.className = 'code-copy';
        copyBtn.type = 'button';
        copyBtn.textContent = 'COPY';
        copyBtn.onclick = () => {
          navigator.clipboard.writeText(codeText).then(() => {
            copyBtn.textContent = 'COPIED';
            setTimeout(() => (copyBtn.textContent = 'COPY'), 2000);
          }).catch(() => {});
        };
        head.appendChild(copyBtn);

        const pre = document.createElement('pre');
        const codeEl = document.createElement('code');
        codeEl.textContent = codeText;
        pre.appendChild(codeEl);

        codeBox.append(head, pre);
        wrap.appendChild(codeBox);

        lastIdx = match.index + match[0].length;
      }

      const rest = text.substring(lastIdx);
      if (rest.trim()) {
        const p = document.createElement('div');
        p.textContent = rest;
        wrap.appendChild(p);
      }
    } else {
      // Inline formatting for **bold** and `code`
      const tokens = text.match(/`[^`]+`|\*\*[^*]+\*\*|\S+|\s+/g) || [];
      tokens.forEach(tok => {
        if (tok.startsWith('`') && tok.endsWith('`') && tok.length > 2) {
          const c = document.createElement('code');
          c.textContent = tok.slice(1, -1);
          wrap.appendChild(c);
        } else if (tok.startsWith('**') && tok.endsWith('**') && tok.length > 4) {
          const b = document.createElement('strong');
          b.textContent = tok.slice(2, -2);
          wrap.appendChild(b);
        } else {
          wrap.appendChild(document.createTextNode(tok));
        }
      });
    }

    return wrap;
  }

  function appendUserTurn(turnId, text, origin = 'VOICE') {
    if (!els.youStream || !text.trim()) return;

    // Check if this turn was already rendered
    const existing = els.youStream.querySelector(`[data-turn-id="${turnId}"]`);
    if (existing) return;

    userTurnCount++;
    if (els.youCount) els.youCount.textContent = `${userTurnCount} TURN${userTurnCount === 1 ? '' : 'S'}`;

    const card = document.createElement('div');
    card.className = 'turn-bubble user-bubble';
    card.dataset.turnId = String(turnId);

    const header = document.createElement('div');
    header.className = 'turn-header';
    header.innerHTML = `
      <span class="turn-origin">${origin}</span>
      <span class="turn-timestamp">${formatTime()}</span>
    `;

    const body = document.createElement('div');
    body.className = 'turn-content you-text';
    body.textContent = text.trim();

    card.append(header, body);
    els.youStream.appendChild(card);
    maybeScrollToBottom(els.youStream);
  }

  function createArtifactCard(artifact) {
    if (!artifact) return null;
    const card = document.createElement('div');
    card.className = 'artifact-card';
    card.dataset.artifactId = artifact.artifact_id || '';
    const st = (artifact.status || 'ready').toLowerCase();
    card.dataset.status = st;

    const displayName = artifact.display_name || 'Document.pdf';

    // Top row: Type Badge + Sanitized Title (textContent: Rule 5/7)
    const header = document.createElement('div');
    header.className = 'artifact-header';

    const badge = document.createElement('span');
    badge.className = 'artifact-badge';
    badge.textContent = 'PDF';

    const titleEl = document.createElement('div');
    titleEl.className = 'artifact-title';
    titleEl.textContent = displayName;

    header.append(badge, titleEl);

    // Meta row: version, pages, size, status (textContent: Rule 5/7)
    const meta = document.createElement('div');
    meta.className = 'artifact-meta';

    if (artifact.version) {
      const versionTag = document.createElement('span');
      versionTag.className = 'artifact-tag';
      versionTag.textContent = `v${artifact.version}`;
      meta.append(versionTag);
    }

    if (artifact.page_count) {
      const pages = artifact.page_count;
      const pagesTag = document.createElement('span');
      pagesTag.className = 'artifact-tag';
      pagesTag.textContent = `${pages} PAGE${pages === 1 ? '' : 'S'}`;
      meta.append(pagesTag);
    }

    if (artifact.size_bytes) {
      const sizeTag = document.createElement('span');
      sizeTag.className = 'artifact-tag';
      sizeTag.textContent = `${Math.round(artifact.size_bytes / 1024)} KB`;
      meta.append(sizeTag);
    }

    const statusTag = document.createElement('span');
    statusTag.className = `artifact-status status-${st}`;
    statusTag.textContent = st.toUpperCase();
    meta.append(statusTag);

    // Actions & content
    const actions = document.createElement('div');
    actions.className = 'artifact-actions';

    const errorBanner = document.createElement('div');
    errorBanner.className = 'artifact-download-error';
    errorBanner.textContent = 'Download failed, retry';
    errorBanner.hidden = true;

    if (st === 'ready' && artifact.artifact_id) {
      card.classList.add('clickable');
      card.setAttribute('role', 'link');
      card.setAttribute('tabindex', '0');
      card.setAttribute('aria-label', `Download ${displayName}`);

      const downloadUrl = `/api/artifacts/${encodeURIComponent(artifact.artifact_id)}?download=1`;
      card.setAttribute('href', downloadUrl);

      // Real same-origin anchor for standard browser download behavior
      const dlLink = document.createElement('a');
      dlLink.className = 'btn primary btn-sm artifact-btn artifact-download-btn';
      dlLink.href = downloadUrl;
      dlLink.setAttribute('download', displayName);
      dlLink.textContent = '⬇ Download';
      dlLink.setAttribute('aria-label', `Download ${displayName}`);

      // Secondary Preview button opening ?preview=1 inline
      const prevLink = document.createElement('a');
      prevLink.className = 'btn plain btn-sm artifact-btn artifact-preview-btn';
      prevLink.href = `/api/artifacts/${encodeURIComponent(artifact.artifact_id)}?preview=1`;
      prevLink.target = '_blank';
      prevLink.rel = 'noopener noreferrer';
      prevLink.textContent = '👁 Preview';
      prevLink.setAttribute('aria-label', `Preview ${displayName} in new tab`);

      prevLink.addEventListener('click', (e) => {
        e.stopPropagation();
      });

      actions.append(dlLink, prevLink);

      const triggerDownload = () => {
        try {
          dlLink.click();
        } catch (err) {
          errorBanner.hidden = false;
        }
      };

      // Clicking anywhere on the card starts download
      card.addEventListener('click', (e) => {
        if (e.target.closest('.artifact-preview-btn')) return;
        triggerDownload();
      });

      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          if (e.target.closest('.artifact-preview-btn')) return;
          e.preventDefault();
          triggerDownload();
        }
      });

      errorBanner.addEventListener('click', (e) => {
        e.stopPropagation();
        triggerDownload();
      });
    } else if (st === 'failed') {
      const errEl = document.createElement('div');
      errEl.className = 'artifact-error';
      errEl.textContent = artifact.error_message || 'Generation failed.';
      actions.append(errEl);
    } else if (st === 'generating') {
      const progEl = document.createElement('div');
      progEl.className = 'artifact-progress';
      progEl.textContent = artifact.progress_message || 'Generating document...';
      actions.append(progEl);
    }

    card.append(header, meta, actions, errorBanner);
    return card;
  }

  function appendAssistantTurn(turnId, text, origin = 'VOICE', artifact = null) {
    if (!els.markStream || !text.trim()) return;

    let card = els.markStream.querySelector(`[data-turn-id="${turnId}"]`);
    if (!card) {
      card = document.createElement('div');
      card.className = 'turn-bubble mark-bubble';
      card.dataset.turnId = String(turnId);

      const header = document.createElement('div');
      header.className = 'turn-header';
      header.innerHTML = `
        <span class="turn-origin">MARK // ${Backend.model.toUpperCase()}</span>
        <span class="turn-timestamp">${formatTime()}</span>
      `;

      const contentHolder = document.createElement('div');
      contentHolder.className = 'turn-content mark-text';
      contentHolder.appendChild(formatContent(text));

      card.append(header, contentHolder);
      if (artifact) {
        card.appendChild(createArtifactCard(artifact));
      }
      els.markStream.appendChild(card);
    } else {
      const contentHolder = card.querySelector('.turn-content.mark-text');
      if (contentHolder) {
        contentHolder.replaceChildren(formatContent(text));
      }
      if (artifact) {
        const existingCard = card.querySelector('.artifact-card');
        if (existingCard) existingCard.remove();
        card.appendChild(createArtifactCard(artifact));
      }
    }

    maybeScrollToBottom(els.markStream);
    return card;
  }

  /* ════════════════════════════════════════════════════════════
     7. TYPED VS VOICE WORKFLOW SEPARATION
     ════════════════════════════════════════════════════════════ */
  let activeTextAbort = null;

  async function handleTypedSubmit(text) {
    if (!text.trim() || currentState === 'offline') return;
    const cleanText = text.trim();

    if (activeTextAbort) activeTextAbort.abort();
    activeTextAbort = new AbortController();
    const sig = activeTextAbort.signal;

    const turnId = `text_${Date.now()}`;
    appendUserTurn(turnId, cleanText, 'TEXT');
    conversationHistory.push({ role: 'user', content: cleanText });

    setState('thinking', 'Neural inference processing');

    let assistantTurnCard = null;

    try {
      const result = await Backend.ask(
        cleanText,
        conversationHistory,
        (token, accumulated, extra) => {
          if (els.liveThinkingCard) els.liveThinkingCard.hidden = true;
          if (extra && extra.type === 'progress') {
            setState('thinking', extra.message);
            if (!assistantTurnCard) {
              assistantTurnCard = appendAssistantTurn(turnId, accumulated || 'Right away, sir. Generating document...', 'TEXT');
            }
            let card = assistantTurnCard.querySelector('.artifact-card');
            if (!card) {
              card = createArtifactCard({ status: 'generating', progress_message: extra.message, display_name: 'Document.pdf' });
              assistantTurnCard.appendChild(card);
            } else {
              const progEl = card.querySelector('.artifact-progress');
              if (progEl) progEl.textContent = extra.message;
            }
          } else if (extra && extra.type === 'artifact' && assistantTurnCard) {
            const existingCard = assistantTurnCard.querySelector('.artifact-card');
            if (existingCard) existingCard.remove();
            assistantTurnCard.appendChild(createArtifactCard(extra.artifact));
          } else if (extra && extra.type === 'error' && assistantTurnCard) {
            const existingCard = assistantTurnCard.querySelector('.artifact-card');
            if (existingCard) existingCard.remove();
            assistantTurnCard.appendChild(createArtifactCard({ status: 'failed', error_message: extra.error || 'Generation failed.' }));
          }
          if (!assistantTurnCard) {
            assistantTurnCard = appendAssistantTurn(turnId, accumulated, 'TEXT');
          } else {
            const contentHolder = assistantTurnCard.querySelector('.turn-content.mark-text');
            if (contentHolder) contentHolder.replaceChildren(formatContent(accumulated));
            maybeScrollToBottom(els.markStream);
          }
        },
        sig
      );

      const replyText = (result && typeof result === 'object') ? result.reply : result;
      const artifact = (result && typeof result === 'object') ? result.artifact : null;

      if (!assistantTurnCard) {
        appendAssistantTurn(turnId, replyText, 'TEXT', artifact);
      } else {
        const contentHolder = assistantTurnCard.querySelector('.turn-content.mark-text');
        if (contentHolder) contentHolder.replaceChildren(formatContent(replyText));
        if (artifact) {
          const existingCard = assistantTurnCard.querySelector('.artifact-card');
          if (existingCard) existingCard.remove();
          assistantTurnCard.appendChild(createArtifactCard(artifact));
        }
      }
      conversationHistory.push({ role: 'assistant', content: replyText });

      // Return immediately to IDLE. NEVER trigger TTS for typed input!
      setState('idle', 'Local intelligence online');
    } catch (err) {
      if (err.name !== 'AbortError') {
        if (!assistantTurnCard) {
          appendAssistantTurn(turnId, `I encountered an error processing that: ${err.message}`, 'TEXT');
        } else {
          const existingCard = assistantTurnCard.querySelector('.artifact-card');
          if (existingCard) existingCard.remove();
          assistantTurnCard.appendChild(createArtifactCard({ status: 'failed', error_message: err.message }));
        }
        setState('error', err.message);
      }
    } finally {
      if (els.liveThinkingCard) els.liveThinkingCard.hidden = true;
      activeTextAbort = null;
      if (els.msg) els.msg.focus();
    }
  }

  /* ════════════════════════════════════════════════════════════
     8. VOICE TELEMETRY POLLING & LIFECYCLE
     ════════════════════════════════════════════════════════════ */
  let lastSeenVoiceTurnId = 0;
  let handsFreeActive = true;

  async function pollVoiceTelemetry() {
    try {
      const data = await Backend.getVoiceStatus();

      if (!Backend.isOnline) {
        Backend.isOnline = true;
        if (els.liveOfflineCard) els.liveOfflineCard.hidden = true;
        if (els.connPill) els.connPill.classList.remove('offline');
      }

      handsFreeActive = !!data.hands_free;
      if (els.hfBtn) {
        els.hfBtn.setAttribute('aria-pressed', handsFreeActive ? 'true' : 'false');
      }

      const vState = data.state;

      // Real state machine synchronization
      if (currentState !== 'permission') {
        if (vState === 'SPEAKING') {
          if (currentState !== 'speaking') setState('speaking');
        } else if (vState === 'THINKING' || vState === 'PROCESSING') {
          if (currentState !== 'thinking') setState('thinking');
        } else if (vState === 'LISTENING_FOR_COMMAND' || vState === 'LISTENING TO COMMAND' || vState === 'LISTENING') {
          if (currentState !== 'listening') {
            setState('listening');
            if (els.captureTitle) els.captureTitle.textContent = 'LISTENING FOR COMMAND';
          }
        } else if (vState === 'TRANSCRIBING') {
          if (currentState !== 'transcribing') {
            setState('transcribing');
            if (els.captureTitle) els.captureTitle.textContent = 'PROCESSING SPEECH';
          }
        } else if (vState === 'GREETING' || vState === 'HEY MARK DETECTED') {
          if (currentState !== 'speaking') setState('speaking', 'Greeting · Active conversation');
        } else if (vState === 'WAKE_WORD_LISTENING' || vState === 'STANDBY') {
          if (currentState !== 'idle' && !activeTextAbort) {
            setState('idle', 'Waiting for “Hey Mark”');
          }
        } else if (vState === 'DISABLED') {
          if (currentState !== 'idle' && !activeTextAbort) {
            setState('idle', 'Hands-free off · Standby');
          }
        } else if (vState === 'MICROPHONE ERROR') {
          setState('error', data.error || 'Microphone unavailable');
        }
      }

      // Synchronize authoritative voice turn into conversation feeds in real time
      if (typeof data.turn_id === 'number' && data.turn_id > lastSeenVoiceTurnId) {
        const vTurnId = `voice_${data.turn_id}`;

        // 1. Immediately render user transcript the moment Whisper finishes
        if (data.last_transcript) {
          appendUserTurn(vTurnId, data.last_transcript, 'VOICE');
        }

        // 2. Stream assistant reply progressively during THINKING and SPEAKING
        const streamReply = data.partial_reply || data.last_reply;
        if (streamReply) {
          if (els.liveThinkingCard) els.liveThinkingCard.hidden = true;
          appendAssistantTurn(vTurnId, streamReply, 'VOICE', data.last_artifact);
        }

        // 3. Spoken sentence highlighting
        if (data.current_spoken_sentence && (data.turn_status === 'SPEAKING' || vState === 'SPEAKING')) {
          const card = els.markStream ? els.markStream.querySelector(`[data-turn-id="${vTurnId}"]`) : null;
          if (card) {
            const markBody = card.querySelector('.turn-content.mark-text');
            if (markBody) {
              markBody.dataset.speakingSentence = data.current_spoken_sentence;
            }
          }
        }

        // 4. Finalize completed turn
        if (data.turn_status === 'COMPLETE') {
          if (els.liveThinkingCard) els.liveThinkingCard.hidden = true;
          lastSeenVoiceTurnId = data.turn_id;
          if (data.last_transcript && data.last_reply) {
            conversationHistory.push({ role: 'user', content: data.last_transcript });
            conversationHistory.push({ role: 'assistant', content: data.last_reply });
          }
          if (data.last_reply) {
            appendAssistantTurn(vTurnId, data.last_reply, 'VOICE', data.last_artifact);
          }

          // Return automatically to LISTENING if hands-free is enabled!
          if (data.hands_free && data.session_active) {
            setState('listening', 'Listening for next command...');
          } else if (data.hands_free) {
            setState('idle', 'Waiting for “Hey Mark”');
          } else {
            setState('idle', 'Hands-free off · Standby');
          }
        } else if (data.turn_status === 'CANCELLED') {
          if (els.liveThinkingCard) els.liveThinkingCard.hidden = true;
          lastSeenVoiceTurnId = data.turn_id;
        }
      }
    } catch {
      if (Backend.isOnline) {
        Backend.isOnline = false;
        setState('offline');
      }
    }
  }

  /* ════════════════════════════════════════════════════════════
     9. EVENT WIRING & KEYBOARD SHORTCUTS
     ════════════════════════════════════════════════════════════ */
  function interruptAll() {
    if (activeTextAbort) {
      activeTextAbort.abort();
      activeTextAbort = null;
    }
    Backend.stopVoice();
    setState('idle', 'Interrupted · Standby');
  }

  // Form submit (typed text)
  if (els.form) {
    els.form.addEventListener('submit', e => {
      e.preventDefault();
      const v = els.msg ? els.msg.value : '';
      if (els.msg) {
        els.msg.value = '';
        els.msg.style.height = 'auto';
      }
      handleTypedSubmit(v);
    });
  }

  // Textarea auto-resize and Enter/Shift+Enter
  if (els.msg) {
    els.msg.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const v = els.msg.value;
        els.msg.value = '';
        els.msg.style.height = 'auto';
        handleTypedSubmit(v);
      }
    });

    els.msg.addEventListener('input', () => {
      els.msg.style.height = 'auto';
      els.msg.style.height = Math.min(els.msg.scrollHeight, 140) + 'px';
    });
  }

  // Mic Button Action
  if (els.mic) {
    els.mic.addEventListener('click', async () => {
      if (currentState === 'listening') {
        interruptAll();
      } else {
        if (!handsFreeActive) {
          try {
            await Backend.toggleHandsFree(true);
            handsFreeActive = true;
            if (els.hfBtn) els.hfBtn.setAttribute('aria-pressed', 'true');
          } catch {}
        }
        await Backend.triggerVoice();
        setState('listening', 'Microphone active');
      }
    });
  }

  if (els.stopListen) {
    els.stopListen.addEventListener('click', interruptAll);
  }

  if (els.cancelListen) {
    els.cancelListen.addEventListener('click', interruptAll);
  }

  if (els.interrupt) {
    els.interrupt.addEventListener('click', interruptAll);
  }

  if (els.stopThink) {
    els.stopThink.addEventListener('click', interruptAll);
  }

  if (els.reconnect) {
    els.reconnect.addEventListener('click', async () => {
      setState('idle', 'Reconnecting to backend...');
      const ok = await Backend.checkHealth();
      if (ok) {
        setState('idle', 'Backend restored');
      } else {
        setState('offline');
      }
    });
  }

  // Hands-free Toggle Button
  if (els.hfBtn) {
    els.hfBtn.addEventListener('click', async () => {
      const next = !handsFreeActive;
      els.hfBtn.disabled = true;
      try {
        if (!next) await Backend.stopVoice();
        const res = await Backend.toggleHandsFree(next);
        handsFreeActive = !!res.hands_free;
        els.hfBtn.setAttribute('aria-pressed', handsFreeActive ? 'true' : 'false');
        if (!handsFreeActive) {
          setState('idle', 'Hands-free disabled');
        } else {
          setState('idle', 'Waiting for “Hey Mark”');
        }
      } catch (e) {
        console.error('Hands-free toggle error:', e);
      } finally {
        els.hfBtn.disabled = false;
      }
    });
  }

  // Clear Log Button
  if (els.clearBtn) {
    els.clearBtn.addEventListener('click', async () => {
      interruptAll();
      if (els.youStream) els.youStream.replaceChildren();
      if (els.markStream) els.markStream.replaceChildren();
      conversationHistory.length = 0;
      userTurnCount = 0;
      if (els.youCount) els.youCount.textContent = '0 TURNS';
      await Backend.clearChat();
      setState('idle', 'Conversation history cleared');
      if (els.msg) els.msg.focus();
    });
  }

  // Global Keyboard Shortcuts
  document.addEventListener('keydown', e => {
    const isTyping = /^(INPUT|TEXTAREA)$/.test(e.target.tagName);

    if (e.key === 'Escape') {
      interruptAll();
    } else if (e.key === ' ' && currentState === 'speaking' && !isTyping) {
      e.preventDefault();
      interruptAll();
    } else if (e.key === '/' && !isTyping) {
      e.preventDefault();
      if (els.msg) els.msg.focus();
    }
  });

  // Client Unload Beacons
  window.addEventListener('pagehide', () => {
    try { navigator.sendBeacon('/api/voice/leave'); } catch {}
  });
  window.addEventListener('beforeunload', () => {
    try { navigator.sendBeacon('/api/voice/leave'); } catch {}
  });

  /* ════════════════════════════════════════════════════════════
     10. BOOTSTRAP & INITIALIZATION
     ════════════════════════════════════════════════════════════ */
  buildScene();
  setState('idle');

  // Verify backend health
  Backend.checkHealth().then(ok => {
    if (ok) {
      setState('idle');
      if (els.pipeLlmText) els.pipeLlmText.textContent = Backend.model.toUpperCase();
    } else {
      setState('offline');
    }
  }).catch(() => setState('offline'));

  // Adaptive polling: fast (160ms) during active voice/text turns, relaxed (650ms) during idle
  let pollTimer = null;
  async function pollLoop() {
    await pollVoiceTelemetry();
    const isActive = (currentState === 'listening' || currentState === 'transcribing' || currentState === 'thinking' || currentState === 'speaking');
    const delay = isActive ? 160 : 650;
    pollTimer = setTimeout(pollLoop, delay);
  }
  pollLoop();
})();
