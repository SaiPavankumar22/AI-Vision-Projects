const state = {
  sessionId: "",
  stream: null,
  monitorTimer: null,
  ws: null,
};

const $ = (id) => document.getElementById(id);
const video = $("video");

function logEvent(message) {
  const log = $("eventLog");
  const ts = new Date().toLocaleTimeString();
  log.textContent = `[${ts}] ${message}\n` + log.textContent;
}

function setAlert(level, message) {
  const banner = $("alertBanner");
  banner.textContent = message;
  banner.className = "alert " + level.toLowerCase();
}

function updateDetectionUI(data) {
  $("sessionId").textContent = data.session_id ?? "-";
  $("headDirection").textContent = data.head_direction ?? "-";
  $("bodyPosition").textContent = data.body_position ?? "-";
  $("facesCount").textContent = String(data.faces_count ?? "-");
  $("alertLevel").textContent = data.alert_level ?? "-";
  $("violationCount").textContent = String(data.violation_count ?? 0);
  $("latencyMs").textContent = data.metrics?.processing_ms ?? "-";
  setAlert(data.alert_level || "OK", data.alert_message || "No issues detected");
}

async function startCamera() {
  if (state.stream) return;
  state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = state.stream;
}

function stopCamera() {
  if (!state.stream) return;
  state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
}

async function captureFrameBlob() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
}

async function sendFrameRest() {
  const backend = $("backendUrl").value.trim();
  const blob = await captureFrameBlob();
  if (!blob) return;

  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");
  formData.append("session_id", state.sessionId);

  const res = await fetch(`${backend}/api/v1/analyze-frame`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let detail = `Analyze failed with status ${res.status}`;
    try {
      const data = await res.json();
      if (data && data.detail) detail = data.detail;
    } catch (_) {}
    throw new Error(detail);
  }

  const data = await res.json();
  state.sessionId = data.session_id;
  updateDetectionUI(data);
  if (data.alert_level !== "OK") {
    logEvent(`${data.alert_level}: ${data.alert_message}`);
  }
}

async function startRestLoop() {
  $("connectionStatus").textContent = "Monitoring via REST";
  state.monitorTimer = setInterval(async () => {
    try {
      await sendFrameRest();
    } catch (err) {
      $("connectionStatus").textContent = `REST error: ${err.message}`;
      logEvent(err.message);
    }
  }, 300);
}

async function startWsLoop() {
  const backendHttp = $("backendUrl").value.trim();
  const wsBase = backendHttp.replace("http://", "ws://").replace("https://", "wss://");
  const url = `${wsBase}/api/v1/ws/analyze?session_id=${encodeURIComponent(state.sessionId)}`;
  state.ws = new WebSocket(url);

  state.ws.onopen = () => {
    $("connectionStatus").textContent = "Monitoring via WebSocket";
    state.monitorTimer = setInterval(async () => {
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
      const blob = await captureFrameBlob();
      if (!blob) return;
      const arr = await blob.arrayBuffer();
      state.ws.send(arr);
    }, 300);
  };

  state.ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    state.sessionId = data.session_id;
    updateDetectionUI(data);
    if (data.alert_level !== "OK") {
      logEvent(`${data.alert_level}: ${data.alert_message}`);
    }
  };

  state.ws.onerror = () => {
    $("connectionStatus").textContent = "WebSocket error";
    logEvent("WebSocket error");
  };

  state.ws.onclose = () => {
    $("connectionStatus").textContent = "WebSocket closed";
  };
}

function stopMonitoring() {
  if (state.monitorTimer) {
    clearInterval(state.monitorTimer);
    state.monitorTimer = null;
  }
  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }
  stopCamera();
  $("startBtn").disabled = false;
  $("stopBtn").disabled = true;
  $("connectionStatus").textContent = "Stopped";
  logEvent("Monitoring stopped");
}

async function startMonitoring() {
  await startCamera();
  if (!state.sessionId) {
    state.sessionId = crypto.randomUUID();
  }

  const mode = $("modeSelect").value;
  if (mode === "ws") {
    await startWsLoop();
  } else {
    await startRestLoop();
  }
  $("startBtn").disabled = true;
  $("stopBtn").disabled = false;
  logEvent(`Monitoring started in ${mode.toUpperCase()} mode`);
}

const startBtn = $("startBtn");
const stopBtn = $("stopBtn");

if (startBtn) {
  startBtn.addEventListener("click", async () => {
    try {
      await startMonitoring();
    } catch (err) {
      $("connectionStatus").textContent = err.message;
      logEvent(err.message);
    }
  });
}

if (stopBtn) {
  stopBtn.addEventListener("click", () => {
    stopMonitoring();
  });
}
