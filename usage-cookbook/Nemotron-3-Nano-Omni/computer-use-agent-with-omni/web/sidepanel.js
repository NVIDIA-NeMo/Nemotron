// Computer Use Agent with Omni - side panel logic
// Connects to the FastAPI backend for desktop status, agent control, and SSE events.
// Uses KasmVNC iframe for live desktop viewing.

const $ = (id) => document.getElementById(id);
const btnRun = $("btn-run");
const btnStop = $("btn-stop");
const btnRestartDesktop = $("btn-restart-desktop");
const envStatus = $("env-status");
const agentStatus = $("agent-status");
const envBadge = $("env-badge");
const log = $("event-log");
const placeholder = $("vnc-placeholder");
const vncFrame = $("vnc-frame");
const instruction = $("instruction");

let state = {
  jobId: null,
  eventSource: null,
  desktopRestarting: false,
};

// ── Desktop status ─────────────────────────────────────────────────────────

async function checkDesktop() {
  if (state.desktopRestarting) return false;
  try {
    const r = await fetch("/health");
    const j = await r.json();
    btnRestartDesktop.disabled = false;
    if (j.desktop === "ready") {
      envStatus.textContent = `Ready — ${j.model}`;
      setBadge("ready", "ready");
      btnRun.disabled = false;

      showVnc(j);
      return true;
    } else {
      envStatus.textContent = "Desktop not ready — run: docker compose up -d";
      setBadge("offline", "offline");
      btnRun.disabled = true;
      return false;
    }
  } catch (e) {
    envStatus.textContent = `Cannot reach server: ${e.message}`;
    setBadge("offline", "offline");
    btnRun.disabled = true;
    btnRestartDesktop.disabled = true;
    return false;
  }
}

function showVnc(health, force = false) {
  const wasHidden = vncFrame.hidden;
  const params = new URLSearchParams({
    autoconnect: "1",
    resize: "scale",
    reconnect: "1",
    reconnect_delay: "1500",
    path: "vnc/websockify",
    password: health.vnc_password || "password",
    kasmvnc_mode_preference: "image",
  });
  const vncUrl = `/vnc/vnc.html?${params.toString()}`;
  placeholder.hidden = true;
  vncFrame.hidden = false;
  if (force || wasHidden || vncFrame.src === "about:blank") {
    vncFrame.src = vncUrl;
  }
}

function showPlaceholder(message) {
  placeholder.innerHTML = `<p>${escape(message)}</p>`;
  placeholder.hidden = false;
  vncFrame.hidden = true;
  vncFrame.src = "about:blank";
}

async function restartDesktop() {
  state.desktopRestarting = true;
  envStatus.textContent = "restarting desktop container…";
  setBadge("running", "restarting");
  btnRestartDesktop.disabled = true;
  btnRun.disabled = true;
  btnStop.disabled = true;
  showPlaceholder("Restarting desktop container...");

  try {
    const r = await fetch("/env/restart", { method: "POST" });
    if (!r.ok) {
      const text = await r.text();
      throw new Error(text);
    }
    state.desktopRestarting = false;
    const ready = await checkDesktop();
    if (!ready) {
      throw new Error("desktop restart finished, but health check is not ready");
    }
    const health = await (await fetch("/health")).json();
    showVnc(health, true);
  } catch (e) {
    state.desktopRestarting = false;
    envStatus.textContent = `restart failed: ${e.message}`;
    setBadge("error", "error");
    btnRestartDesktop.disabled = false;
  }
}

// ── Agent lifecycle ────────────────────────────────────────────────────────

async function runAgent() {
  if (!instruction.value.trim()) return;
  log.innerHTML = "";
  agentStatus.textContent = "starting…";
  setBadge("running", "running");
  btnRun.disabled = true;
  btnStop.disabled = false;

  try {
    const r = await fetch("/agent/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        instruction: instruction.value.trim(),
      }),
    });
    if (!r.ok) {
      const text = await r.text();
      throw new Error(text);
    }
    const j = await r.json();
    state.jobId = j.job_id;
    agentStatus.textContent = `running (${j.job_id})`;
    streamEvents(j.job_id);
  } catch (e) {
    agentStatus.textContent = `start failed: ${e.message}`;
    setBadge("error", "error");
    btnRun.disabled = false;
    btnStop.disabled = true;
  }
}

function streamEvents(jobId) {
  if (state.eventSource) state.eventSource.close();
  const es = new EventSource(`/agent/${jobId}/events`);
  state.eventSource = es;
  es.onmessage = (msg) => {
    if (!msg.data) return;
    let ev;
    try { ev = JSON.parse(msg.data); } catch { return; }
    appendEvent(ev);
    if (["done", "failed", "error", "stopping", "stopped"].includes(ev.kind)) {
      agentStatus.textContent = ev.kind;
      if (ev.kind === "done") setBadge("ready", "done ✓");
      else if (ev.kind === "failed") setBadge("error", "failed");
      else setBadge("ready", ev.kind);
    }
    if (ev.kind === "finished") {
      es.close();
      btnRun.disabled = false;
      btnStop.disabled = true;
    }
  };
  es.onerror = () => {
    agentStatus.textContent = "stream disconnected";
  };
}

async function stopAgent() {
  if (!state.jobId) return;
  agentStatus.textContent = "stopping…";
  btnStop.disabled = true;
  try {
    const r = await fetch(`/agent/${state.jobId}/stop`, { method: "POST" });
    if (!r.ok) {
      const text = await r.text();
      throw new Error(text);
    }
  } catch (e) {
    agentStatus.textContent = `stop failed: ${e.message}`;
    btnStop.disabled = false;
  }
}

// ── UI helpers ─────────────────────────────────────────────────────────────

function setBadge(cls, text) {
  envBadge.className = "badge " + cls;
  envBadge.textContent = text;
}

// Per-step DOM nodes and buffered text for live streaming
const liveStep = {
  step: null,
  container: null,
  thought: null,
  content: null,
  pendingReasoning: "",
  pendingContent: "",
  flushScheduled: false,
};

function ensureLiveStep(step) {
  if (liveStep.step === step && liveStep.container && liveStep.container.isConnected) {
    return;
  }
  const div = document.createElement("div");
  div.className = "event";
  div.innerHTML = `<span class="tag step">step ${step}</span>` +
    `<div class="thought live-thought"></div>` +
    `<div class="action live-content"></div>`;
  log.appendChild(div);
  liveStep.step = step;
  liveStep.container = div;
  liveStep.thought = div.querySelector(".live-thought");
  liveStep.content = div.querySelector(".live-content");
  log.scrollTop = log.scrollHeight;
}

function scheduleLiveFlush() {
  if (liveStep.flushScheduled) return;
  liveStep.flushScheduled = true;
  requestAnimationFrame(() => {
    liveStep.flushScheduled = false;
    if (!liveStep.container || !liveStep.container.isConnected) {
      liveStep.pendingReasoning = "";
      liveStep.pendingContent = "";
      return;
    }
    if (liveStep.pendingReasoning) {
      if (!liveStep.thought.textContent) {
        liveStep.thought.textContent = "reasoning\n";
      }
      liveStep.thought.textContent += liveStep.pendingReasoning;
      liveStep.pendingReasoning = "";
    }
    if (liveStep.pendingContent) {
      if (!liveStep.content.textContent) {
        liveStep.content.textContent = "response\n";
      }
      liveStep.content.textContent += liveStep.pendingContent;
      liveStep.pendingContent = "";
    }
    log.scrollTop = log.scrollHeight;
  });
}

function appendEvent(ev) {
  // Streaming deltas
  if (ev.kind === "thought_delta") {
    ensureLiveStep(ev.step);
    if (ev.reasoning) {
      liveStep.pendingReasoning += ev.reasoning;
    }
    if (ev.content) {
      liveStep.pendingContent += ev.content;
    }
    if (ev.reasoning || ev.content) {
      agentStatus.textContent = `streaming step ${ev.step}…`;
      scheduleLiveFlush();
    }
    return;
  }

  const div = document.createElement("div");
  div.className = "event";
  let body;
  switch (ev.kind) {
    case "started":
      body = `<span class="tag">▶ started</span>${escape(ev.instruction)}`;
      break;
    case "screen_size":
      body = `<span class="tag">screen</span>${ev.width}×${ev.height}`;
      break;
    case "step_started":
      liveStep.step = null;
      liveStep.container = null;
      liveStep.pendingReasoning = "";
      liveStep.pendingContent = "";
      liveStep.flushScheduled = false;
      agentStatus.textContent = `step ${ev.step}: waiting for model…`;
      body = `<span class="tag step">step ${ev.step}</span>`;
      break;
    case "thought":
      if (liveStep.step === ev.step && liveStep.container) {
        if (ev.action) {
          const a = document.createElement("div");
          a.className = "action";
          a.textContent = "→ " + ev.action;
          liveStep.container.appendChild(a);
        }
        if (ev.code) {
          const c = document.createElement("div");
          c.className = "code";
          c.textContent = ev.code;
          liveStep.container.appendChild(c);
        }
        log.scrollTop = log.scrollHeight;
        return;
      }
      body = `<span class="tag step">step ${ev.step}</span>` +
             (ev.thought ? `<div class="thought">💭 ${escape(ev.thought)}</div>` : "") +
             (ev.action ? `<div class="action">→ ${escape(ev.action)}</div>` : "") +
             (ev.code ? `<div class="code">${escape(ev.code)}</div>` : "");
      break;
    case "executed":
      body = `<span class="tag">⚡ executed</span>` +
             (ev.output ? `<div class="code">${escape(ev.output)}</div>` : "");
      break;
    case "execute_error":
      body = `<span class="tag err">⚠ exec err</span><div class="err">${escape(ev.message)}</div>`;
      break;
    case "wait":
      body = `<span class="tag">⏳ wait</span>${ev.seconds}s`;
      break;
    case "done":
      body = `<span class="tag" style="color:var(--green)">✓ DONE</span>`;
      break;
    case "failed":
      body = `<span class="tag err">✗ FAILED</span>${ev.reason || ""}`;
      break;
    case "stopped":
      body = `<span class="tag">■ stopped</span>`;
      break;
    case "stopping":
      body = `<span class="tag">■ stopping</span>`;
      break;
    case "error":
      body = `<span class="tag err">⚠ error</span><div class="err">${escape(ev.message || "")}</div>`;
      break;
    case "finished":
      body = `<span class="tag">— finished (${ev.status}) —</span>`;
      break;
    default:
      body = `<span class="tag">${escape(ev.kind)}</span>${escape(JSON.stringify(ev))}`;
  }
  div.innerHTML = body;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function escape(s) {
  if (s === undefined || s === null) return "";
  return String(s)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

// ── Wire up ────────────────────────────────────────────────────────────────

btnRun.addEventListener("click", runAgent);
btnStop.addEventListener("click", stopAgent);
btnRestartDesktop.addEventListener("click", restartDesktop);

// Initial check + periodic polling
checkDesktop();
setInterval(checkDesktop, 10000);
