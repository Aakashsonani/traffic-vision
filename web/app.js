"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  file: null, source: "upload", isVideo: false, stream: null, objectURL: null,
  result: null, original: null, showingOriginal: false, ready: false,
  busy: false, running: false, generation: 0, controller: null, history: [], catalogue: [],
};
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g,
  (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
const show = (id, visible = true) => $(id).classList.toggle("hidden", !visible);
let toastTimer;
function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  show("toast");
  toastTimer = setTimeout(() => show("toast", false), 6500);
}

function setStatus(message) {
  $("preview-status").innerHTML = `<span class="tiny-dot"></span>${escapeHTML(message)}`;
}

function syncControls() {
  $("analyze-button").disabled = !state.ready || state.busy || state.running
    || !(state.file || state.stream);
  $("analyze-button").querySelector("span").textContent = state.isVideo ? "Scan video"
    : state.stream ? "Start live recognition" : "Run recognition";
  show("stop-button", state.running || state.busy);
  $("enable-camera").disabled = !!state.stream;
  $("threshold").disabled = state.busy || state.running;
  $("scan-mode").disabled = state.busy || state.running;
  $("detail-mode").disabled = state.busy || state.running || $("scan-mode").value === "signal";
}

function stopScan() {
  state.running = false;
  state.generation++;
  state.controller?.abort();
  state.controller = null;
  state.busy = false;
  $("source-video").pause();
  $("source-video").controls = state.isVideo;
  show("busy-overlay", false);
  if (state.file || state.stream) setStatus("Ready to scan");
  syncControls();
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  $("source-video").srcObject = null;
  $("source-video").classList.remove("camera");
  $("enable-camera").textContent = "Enable camera";
  syncControls();
}

function clearResult() {
  state.result = null;
  state.history = [];
  state.showingOriginal = false;
  $("result-count").textContent = "0";
  $("inference-time").textContent = "READY WHEN YOU ARE";
  $("toggle-original").disabled = true;
  $("toggle-original").textContent = "Show original";
  $("download-image").disabled = true;
  $("download-json").disabled = true;
  $("results").innerHTML = '<div class="empty-results"><span>—</span><p>No observations yet.<small>Run recognition to see labels, scores, and sign meanings.</small></p></div>';
}

function resetInput() {
  stopScan();
  stopCamera();
  if (state.objectURL) URL.revokeObjectURL(state.objectURL);
  state.objectURL = null;
  state.file = null;
  state.original = null;
  state.isVideo = false;
  $("source-video").removeAttribute("src");
  $("source-video").load();
  $("file-input").value = "";
  $("preview-image").removeAttribute("src");
  show("preview-image", false);
  show("source-video", false);
  show("empty-preview");
  show("selected-file", false);
  $("image-info").textContent = "Your input stays on this device";
  setStatus("Waiting for input");
  clearResult();
  syncControls();
}

function selectSource(source) {
  if (state.source === source) return;
  resetInput();
  state.source = source;
  document.querySelectorAll("[data-source]").forEach((button) => {
    const active = button.dataset.source === source;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  show("upload-area", source === "upload");
  show("camera-area", source === "camera");
}

function selectFile(file) {
  if (!file) return;
  const isVideo = file.type.startsWith("video/") || /\.(mp4|webm)$/i.test(file.name);
  const isImage = /^image\/(jpeg|png|webp|bmp)$/.test(file.type)
    || /\.(jpe?g|png|webp|bmp)$/i.test(file.name);
  if (!isImage && !isVideo) return toast("Choose a JPG, PNG, WebP, BMP, MP4 or WebM file.");
  if (file.size > (isVideo ? 500 : 16) * 1024 * 1024) {
    return toast(isVideo ? "Choose a video smaller than 500 MB." : "Choose an image smaller than 16 MB.");
  }
  if (state.source !== "upload") selectSource("upload");
  resetInput();
  state.file = file;
  state.isVideo = isVideo;
  state.objectURL = URL.createObjectURL(file);
  state.original = state.objectURL;
  $("file-name").textContent = file.name;
  show("selected-file");
  show("empty-preview", false);
  if (isVideo) {
    $("source-video").src = state.objectURL;
    $("source-video").controls = true;
    show("source-video");
    $("image-info").textContent = "Video · one frame per second, processed in sequence";
  } else {
    $("preview-image").src = state.objectURL;
    show("preview-image");
    $("image-info").textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  }
  setStatus("Ready to scan");
  syncControls();
}

async function enableCamera() {
  if (!navigator.mediaDevices?.getUserMedia) return toast("Camera access requires localhost in a supported browser.");
  const generation = state.generation;
  $("enable-camera").disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment",
      width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    if (generation !== state.generation || state.source !== "camera") {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    state.stream = stream;
    const video = $("source-video");
    video.srcObject = stream;
    video.controls = false;
    video.classList.add("camera");
    await video.play();
    show("empty-preview", false);
    show("preview-image", false);
    show("source-video");
    $("enable-camera").textContent = "Camera enabled";
    $("image-info").textContent = "Live camera · recognition speed depends on your CPU";
    setStatus("Camera ready");
  } catch (error) {
    stopCamera();
    toast(error.name === "NotAllowedError" ? "Camera permission was denied. Allow camera access in your browser."
      : "Could not open the camera. Check that it is connected and not in use.");
  } finally { syncControls(); }
}

function waitForMedia(video) {
  if (video.readyState >= 2) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { cleanup(); reject(new Error("The browser could not decode this video.")); }, 10000);
    function cleanup() { clearTimeout(timeout); video.removeEventListener("loadeddata", loaded); video.removeEventListener("error", failed); }
    function loaded() { cleanup(); resolve(); }
    function failed() { cleanup(); reject(new Error("This video codec is not supported. Try an MP4 with H.264 video.")); }
    video.addEventListener("loadeddata", loaded, { once: true });
    video.addEventListener("error", failed, { once: true });
  });
}

async function captureFrame() {
  const video = $("source-video");
  await waitForMedia(video);
  const canvas = $("capture-canvas");
  const scale = Math.min(1, 1920 / video.videoWidth);
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", .92));
  if (!blob) throw new Error("Could not capture the video frame.");
  return blob;
}

function seekVideo(time) {
  const video = $("source-video");
  if (Math.abs(video.currentTime - time) < .01) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { cleanup(); reject(new Error("Could not seek this video frame.")); }, 10000);
    function cleanup() { clearTimeout(timeout); video.removeEventListener("seeked", done); video.removeEventListener("error", failed); }
    function done() { cleanup(); resolve(); }
    function failed() { cleanup(); reject(new Error("The browser could not read this video.")); }
    video.addEventListener("seeked", done, { once: true });
    video.addEventListener("error", failed, { once: true });
    video.currentTime = time;
  });
}

async function requestRecognition(blob, generation, timestamp = null) {
  if (generation !== state.generation) return null;
  const data = new FormData();
  data.append("file", blob, state.isVideo || state.stream ? "frame.jpg" : state.file.name);
  data.append("mode", $("scan-mode").value);
  data.append("threshold", Number($("threshold").value) / 100);
  data.append("detail", $("detail-mode").checked && $("scan-mode").value === "scene");
  state.controller = new AbortController();
  const response = await fetch("/api/analyze", { method: "POST", body: data, signal: state.controller.signal });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Recognition request failed.");
  if (generation !== state.generation) return null;
  result.timestamp_seconds = timestamp;
  const { annotated_image, ...record } = result;
  if (timestamp !== null) {
    state.history.push(record);
    if (state.history.length > 1000) state.history.shift();
  }
  state.result = result;
  renderResult(result);
  return result;
}

async function runRecognition() {
  if (state.busy || state.running || !state.ready || !(state.file || state.stream)) return;
  const generation = ++state.generation;
  state.busy = true;
  state.running = state.isVideo || !!state.stream;
  state.history = [];
  syncControls();
  show("busy-overlay");
  setStatus("Recognition in progress");
  try {
    if (!state.running) {
      await requestRecognition(state.file, generation);
    } else {
      const video = $("source-video");
      await waitForMedia(video);
      const duration = state.isVideo ? video.duration : Infinity;
      if (state.isVideo && (!Number.isFinite(duration) || duration <= 0)) throw new Error("Video duration is unavailable. Try another MP4.");
      video.controls = false;
      if (state.stream) await video.play();
      // Videos are sampled at one-second intervals; no frames are uploaded as a whole file.
      let timestamp = state.isVideo ? 0 : performance.now() / 1000;
      while (state.running && generation === state.generation && timestamp < duration) {
        if (state.isVideo) await seekVideo(timestamp);
        if (generation !== state.generation) break;
        const frame = await captureFrame();
        const result = await requestRecognition(frame, generation,
          state.isVideo ? timestamp : performance.now() / 1000 - timestamp);
        show("busy-overlay", false);
        if (!result) break;
        if (state.isVideo) {
          $("image-info").textContent = `Video scan · ${timestamp.toFixed(0)} / ${duration.toFixed(0)} s · ${state.history.length} frames`;
          timestamp += 1;
        } else {
          await new Promise((resolve) => setTimeout(resolve, 150));
        }
        if (state.history.length >= 1000) { toast("Scan limit reached: 1,000 frames. Export the results before continuing."); break; }
      }
    }
  } catch (error) {
    if (error.name !== "AbortError" && generation === state.generation) {
      toast(error.message || "Recognition failed. Check the local server.");
      setStatus("Recognition could not finish");
    }
  } finally {
    if (generation === state.generation) {
      state.running = false;
      state.busy = false;
      state.controller = null;
      $("source-video").controls = state.isVideo;
      show("busy-overlay", false);
      syncControls();
    }
  }
}

function renderResult(result) {
  state.showingOriginal = false;
  $("preview-image").src = result.annotated_image;
  $("preview-image").alt = "Recognition output with labelled bounding boxes";
  show("preview-image");
  show("source-video", false);
  show("empty-preview", false);
  $("result-count").textContent = String(result.count);
  $("inference-time").textContent = `${result.elapsed_ms.toLocaleString()} MS · CPU`;
  $("image-info").textContent = `${result.width} × ${result.height} px · ${result.mode === "signal" ? "Close-up signal" : "Full scene"}`;
  setStatus(result.count ? "Recognition complete" : "No confident observations");
  $("results").innerHTML = result.detections.length ? result.detections.map((item, index) => `
    <div class="observation"><span class="obs-index">${String(index + 1).padStart(2, "0")}</span>
    <div class="obs-body"><div class="obs-title"><span class="obs-dot ${escapeHTML(item.color || "sign")}"></span>${escapeHTML(item.label)}${item.status === "uncertain" ? ' <span class="meta">UNCERTAIN</span>' : ""}</div>
    <p>${escapeHTML(item.meaning)}</p>${item.alternatives ? `<p class="alternatives">Also predicted: ${escapeHTML(item.alternatives.join(", "))}</p>` : ""}</div>
    <span class="obs-score ${item.status === "uncertain" ? "uncertain" : ""}" title="Model score; not measured accuracy">${Math.round(item.score * 100)}%</span></div>`).join("")
    : '<div class="empty-results"><span>—</span><p>No confident observations.<small>Try a clearer image, a close-up signal, or a lower score threshold.</small></p></div>';
  $("download-image").disabled = false;
  $("download-json").disabled = false;
  $("toggle-original").disabled = !state.original || state.isVideo || !!state.stream;
  $("toggle-original").textContent = "Show original";
}

function downloadURL(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function signGlyph(sign) {
  const symbols = { turn_left: "↰", turn_right: "↱", straight: "↑", keep_left: "↙", keep_right: "↘",
    straight_left: "↰ ↑", straight_right: "↑ ↱", roundabout: "↻", stop: "STOP", yield: "▽",
    no_entry: "−", no_left: "↰", no_right: "↱", no_u_turn: "↶", u_turn: "↶", speed_60: "60",
    no_parking: "P̸", parking: "P", no_stopping: "×", signals_ahead: "⋮", left_right: "↔",
    curve_left: "↶", curve_right: "↷", bus_stop: "BUS", children: "!", caution: "!" };
  return symbols[sign.id] || (sign.category === "Warning" ? "△" : sign.category === "Prohibition" ? "⊘" : "◇");
}

function renderCatalogue() {
  const search = $("search-signs").value.trim().toLowerCase();
  const category = $("category-filter").value;
  const filtered = state.catalogue.filter((sign) => (category === "all" || category === sign.category)
    && `${sign.label} ${sign.meaning}`.toLowerCase().includes(search));
  $("library-count").textContent = `${filtered.length} SIGNS`;
  $("sign-grid").innerHTML = filtered.length ? filtered.map((sign) => `<article class="sign-card" data-category="${escapeHTML(sign.category)}"><div class="sign-glyph" aria-hidden="true">${escapeHTML(signGlyph(sign))}</div><span class="meta">${escapeHTML(sign.category.toUpperCase())}</span><h3>${escapeHTML(sign.label)}</h3><p>${escapeHTML(sign.meaning)}</p></article>`).join("")
    : '<p class="category-empty">No signs match this search.</p>';
}

async function loadEvaluation() {
  try {
    const report = await (await fetch("/api/evaluation")).json();
    const evaluated = report.status === "evaluated";
    const metric = (value) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
    $("evaluation-content").innerHTML = `<div class="metric-grid"><div class="panel metric-card"><span class="meta">PRECISION</span><strong>${evaluated ? metric(report.precision) : "—"}</strong><p>${evaluated ? "Correct accepted observations" : "Target: at least 95%"}</p></div><div class="panel metric-card"><span class="meta">RECALL</span><strong>${evaluated ? metric(report.recall) : "—"}</strong><p>${evaluated ? "Labelled objects recognised" : "Target: at least 90%"}</p></div><div class="panel metric-card"><span class="meta">TEST IMAGES</span><strong>${evaluated ? Number(report.images) : "0"}</strong><p>${evaluated ? escapeHTML(report.dataset_name || "Local dataset") : "Representative dataset required"}</p></div></div>
      <div class="eval-note">${evaluated ? `Evaluation: ${escapeHTML(report.scope)}. Exact class match at IoU ≥ ${Number(report.iou_threshold)}. ${report.passed_targets ? "Overall targets met on this dataset." : "Overall targets have not been met on this dataset."}` : "Accuracy has not been measured on local road scenes. The targets above are goals, not achieved results."}</div>
      ${evaluated ? `<div class="panel scope-panel"><h2>Results by class</h2><table class="eval-table"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>Support</th></tr></thead><tbody>${report.per_class.map((row) => `<tr><td>${escapeHTML(row.label)}</td><td>${metric(row.precision)}</td><td>${metric(row.recall)}</td><td>${Number(row.support)}</td></tr>`).join("")}</tbody></table></div>` : ""}`;
  } catch { $("evaluation-content").innerHTML = '<div class="alert">Could not load the evaluation report. Check the local server.</div>'; }
}

function navigate(page) {
  const pages = { analyzer: "Recognition studio", library: "Sign catalogue", evaluation: "Evaluation", about: "Project notes" };
  if (!pages[page]) page = "analyzer";
  if (page !== "analyzer" && (state.running || state.stream)) { stopScan(); stopCamera(); }
  document.querySelectorAll(".page").forEach((node) => node.classList.toggle("active", node.id === `page-${page}`));
  document.querySelectorAll("[data-page]").forEach((node) => {
    const active = node.dataset.page === page;
    node.classList.toggle("active", active);
    if (active) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current");
  });
  $("breadcrumb").textContent = pages[page];
  if (page === "evaluation") loadEvaluation();
}

async function init() {
  document.querySelectorAll("[data-page]").forEach((button) => button.addEventListener("click", () => {
    location.hash = button.dataset.page;
  }));
  window.addEventListener("hashchange", () => navigate(location.hash.slice(1)));
  navigate(location.hash.slice(1) || "analyzer");
  document.querySelectorAll("[data-source]").forEach((button) => button.addEventListener("click", () => selectSource(button.dataset.source)));
  $("file-input").addEventListener("change", (event) => selectFile(event.target.files[0]));
  $("clear-file").addEventListener("click", resetInput);
  $("dropzone").addEventListener("dragover", (event) => { event.preventDefault(); $("dropzone").classList.add("dragover"); });
  $("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("dragover"));
  $("dropzone").addEventListener("drop", (event) => { event.preventDefault(); $("dropzone").classList.remove("dragover"); selectFile(event.dataTransfer.files[0]); });
  $("enable-camera").addEventListener("click", enableCamera);
  $("analyze-button").addEventListener("click", runRecognition);
  $("stop-button").addEventListener("click", stopScan);
  $("threshold").addEventListener("input", () => { $("threshold-value").textContent = `${$("threshold").value}%`; });
  $("scan-mode").addEventListener("change", () => {
    const signal = $("scan-mode").value === "signal";
    $("mode-help").textContent = signal ? "Use a close-up of one signal housing to read its lamps and arrows." : "Find signs and signal housings, then read individual lamps.";
    syncControls();
  });
  $("toggle-original").addEventListener("click", () => {
    if (!state.result || !state.original) return;
    state.showingOriginal = !state.showingOriginal;
    $("preview-image").src = state.showingOriginal ? state.original : state.result.annotated_image;
    $("preview-image").alt = state.showingOriginal ? "Original input image" : "Recognition output with labelled bounding boxes";
    $("toggle-original").textContent = state.showingOriginal ? "Show detections" : "Show original";
  });
  $("download-image").addEventListener("click", () => { if (state.result) downloadURL(state.result.annotated_image, "traffic-vision-result.jpg"); });
  $("download-json").addEventListener("click", () => {
    if (!state.result) return;
    const { annotated_image, ...record } = state.result;
    const report = { generated_at: new Date().toISOString(), source: state.file?.name || "camera",
      ...record, ...(state.history.length ? { frames: state.history, frame_sampling_seconds: state.isVideo ? 1 : null } : {}) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    downloadURL(url, "traffic-vision-observations.json");
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  $("search-signs").addEventListener("input", renderCatalogue);
  $("category-filter").addEventListener("change", renderCatalogue);
  $("source-video").addEventListener("error", () => { if (state.file) toast("Video could not be decoded. Try an MP4 with H.264 video."); });
  $("preview-image").addEventListener("error", () => { if (state.file) toast("Could not display this image. Try a JPG or PNG."); });
  window.addEventListener("pagehide", () => { stopScan(); stopCamera(); });
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Health check failed");
    const health = await response.json();
    state.ready = health.ready;
    $("model-status").textContent = health.ready ? "Models ready · CPU" : "Models need setup";
    $("model-dot").classList.toggle("warn", !health.ready);
    show("setup-alert", !health.ready);
    $("nav-count").textContent = String(health.classes);
    $("class-caption").textContent = `${health.classes} distinct sign meanings`;
    syncControls();
  } catch { $("model-status").textContent = "Server unavailable"; toast("Could not reach the local Python server."); }
  try {
    const data = await (await fetch("/api/catalogue")).json();
    state.catalogue = data.signs;
    renderCatalogue();
  } catch { toast("The sign catalogue could not be loaded."); }
  try {
    const samples = await (await fetch("/api/samples")).json();
    $("samples").innerHTML = samples.map((sample, index) => `<button class="sample-card" data-sample="${index}" title="${escapeHTML(sample.description)}"><img src="/samples/${encodeURIComponent(sample.file)}" alt="${escapeHTML(sample.label)}"><span>${escapeHTML(sample.label)}</span></button>`).join("");
    if (!samples.length) $("samples").innerHTML = '<p class="field-help">Upload an image to begin.</p>';
    document.querySelectorAll("[data-sample]").forEach((button) => button.addEventListener("click", async () => {
      try {
        const sample = samples[Number(button.dataset.sample)];
        const response = await fetch(`/samples/${encodeURIComponent(sample.file)}`);
        if (!response.ok) throw new Error("Could not load this example.");
        const blob = await response.blob();
        selectFile(new File([blob], sample.file, { type: blob.type }));
        $("scan-mode").value = sample.mode || "scene";
        $("scan-mode").dispatchEvent(new Event("change"));
        toast(sample.description);
      } catch (error) { toast(error.message); }
    }));
  } catch { $("samples").innerHTML = '<p class="field-help">Examples could not be loaded.</p>'; }
}

init();
