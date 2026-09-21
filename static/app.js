/**
 * ArtGate Forensic Verifier - Client Controller
 * Digital Forensic Laboratory Workbench Engine
 */

const state = {
  currentMode: "upload", // "upload" | "camera"
  selectedFile: null,
  capturedBase64: null,
  mediaStream: null,
  facingMode: "user",
  lastAnalysisResult: null,
  history: []
};

const el = {
  statusBadge: document.getElementById("statusBadge"),
  statusText: document.getElementById("statusText"),
  viewportStatusText: document.getElementById("viewportStatusText"),
  tabUpload: document.getElementById("tabUpload"),
  tabCamera: document.getElementById("tabCamera"),
  viewUpload: document.getElementById("viewUpload"),
  viewCamera: document.getElementById("viewCamera"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  dropzoneEmpty: document.getElementById("dropzoneEmpty"),
  dropzonePreview: document.getElementById("dropzonePreview"),
  previewImage: document.getElementById("previewImage"),
  previewFilename: document.getElementById("previewFilename"),
  previewMetaDimensions: document.getElementById("previewMetaDimensions"),
  previewMetaSize: document.getElementById("previewMetaSize"),
  btnRemoveFile: document.getElementById("btnRemoveFile"),
  cameraVideo: document.getElementById("cameraVideo"),
  cameraCanvas: document.getElementById("cameraCanvas"),
  cameraResolution: document.getElementById("cameraResolution"),
  cameraSnapshotPreview: document.getElementById("cameraSnapshotPreview"),
  capturedFrameImg: document.getElementById("capturedFrameImg"),
  btnSwitchCamera: document.getElementById("btnSwitchCamera"),
  btnCaptureAnalyze: document.getElementById("btnCaptureAnalyze"),
  btnRetakeCamera: document.getElementById("btnRetakeCamera"),
  btnAnalyze: document.getElementById("btnAnalyze"),
  analyzeBtnLabel: document.getElementById("analyzeBtnLabel"),
  analysisProgress: document.getElementById("analysisProgress"),
  verdictCard: document.getElementById("verdictCard"),
  verdictTag: document.getElementById("verdictTag"),
  metaTimestamp: document.getElementById("metaTimestamp"),
  verdictTitle: document.getElementById("verdictTitle"),
  verdictDesc: document.getElementById("verdictDesc"),
  verdictConfidence: document.getElementById("verdictConfidence"),
  probRealVal: document.getElementById("probRealVal"),
  probAiVal: document.getElementById("probAiVal"),
  metaLatency: document.getElementById("metaLatency"),
  metaDevice: document.getElementById("metaDevice"),
  gatingSpatial: document.getElementById("gatingSpatial"),
  gatingWavelet: document.getElementById("gatingWavelet"),
  gatingClip: document.getElementById("gatingClip"),
  imgGradCAM: document.getElementById("imgGradCAM"),
  imgSubbandLL: document.getElementById("imgSubbandLL"),
  imgSubbandLH: document.getElementById("imgSubbandLH"),
  imgSubbandHL: document.getElementById("imgSubbandHL"),
  imgSubbandHH: document.getElementById("imgSubbandHH"),
  imgFFTSpectrum: document.getElementById("imgFFTSpectrum"),
  energyLL: document.getElementById("energyLL"),
  energyLH: document.getElementById("energyLH"),
  energyHL: document.getElementById("energyHL"),
  energyHH: document.getElementById("energyHH"),
  highFreqRatio: document.getElementById("highFreqRatio"),
  historyTableBody: document.getElementById("historyTableBody"),
  btnClearHistory: document.getElementById("btnClearHistory"),
  btnExportJson: document.getElementById("btnExportJson"),
  inspectorModal: document.getElementById("inspectorModal"),
  modalTitle: document.getElementById("modalTitle"),
  modalDesc: document.getElementById("modalDesc"),
  modalImg: document.getElementById("modalImg")
};

// ---------------------------------------------------------------------
// 1. INITIALIZATION & ENGINE HEALTH
// ---------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initEventListeners();
  loadSessionHistory();
});

async function initHealthCheck() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Server not responding");
    const data = await res.json();
    el.statusBadge.className = "status-indicator status-ready";
    el.statusText.textContent = `ONLINE [${data.device.toUpperCase()}]`;
  } catch (err) {
    el.statusBadge.className = "status-indicator";
    el.statusText.textContent = "OFFLINE";
    console.warn("Backend not responding:", err);
  }
}


// ---------------------------------------------------------------------
// 3. TAB CONTROLS & EVENT LISTENERS
// ---------------------------------------------------------------------
function initEventListeners() {
  // Tabs
  el.tabUpload.addEventListener("click", () => switchTab("upload"));
  el.tabCamera.addEventListener("click", () => switchTab("camera"));

  // Dropzone
  el.dropzone.addEventListener("click", (e) => {
    if (e.target !== el.btnRemoveFile && !el.btnRemoveFile.contains(e.target)) {
      el.fileInput.click();
    }
  });

  el.fileInput.addEventListener("change", handleFileInput);

  el.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    el.dropzone.classList.add("dragover");
  });

  el.dropzone.addEventListener("dragleave", () => {
    el.dropzone.classList.remove("dragover");
  });

  el.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    el.dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      loadSelectedFile(e.dataTransfer.files[0]);
    }
  });

  el.btnRemoveFile.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUploadState();
  });

  // Camera Actions
  el.btnSwitchCamera.addEventListener("click", flipCamera);
  el.btnCaptureAnalyze.addEventListener("click", captureAndAnalyze);
  el.btnRetakeCamera.addEventListener("click", retakeCamera);

  // Analyze Action
  el.btnAnalyze.addEventListener("click", executeAnalysis);

  // Export Raw Telemetry JSON
  el.btnExportJson.addEventListener("click", exportRawTelemetry);

  // Clear Audit Trail
  el.btnClearHistory.addEventListener("click", clearSessionHistory);

  // Global ESC to close modals
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeInspectorModal();
    }
  });
}

function switchTab(mode) {
  state.currentMode = mode;

  if (mode === "upload") {
    el.tabUpload.classList.add("active");
    el.tabUpload.setAttribute("aria-selected", "true");
    el.tabCamera.classList.remove("active");
    el.tabCamera.setAttribute("aria-selected", "false");
    el.viewUpload.classList.add("active");
    el.viewCamera.classList.remove("active");
    el.viewportStatusText.textContent = state.selectedFile ? "File loaded in viewport" : "Awaiting input source";
    stopCameraStream();
  } else if (mode === "camera") {
    el.tabCamera.classList.add("active");
    el.tabCamera.setAttribute("aria-selected", "true");
    el.tabUpload.classList.remove("active");
    el.tabUpload.setAttribute("aria-selected", "false");
    el.viewCamera.classList.add("active");
    el.viewUpload.classList.remove("active");
    el.viewportStatusText.textContent = "Hardware optical capture stream";
    startCameraStream();
  }
}

async function startCameraStream() {
  stopCameraStream();
  try {
    const constraints = {
      video: {
        facingMode: state.facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };
    state.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    el.cameraVideo.srcObject = state.mediaStream;

    el.cameraVideo.onloadedmetadata = () => {
      el.cameraResolution.textContent = `${el.cameraVideo.videoWidth} x ${el.cameraVideo.videoHeight}`;
    };

    el.cameraSnapshotPreview.style.display = "none";
    el.btnRetakeCamera.style.display = "none";
    el.btnCaptureAnalyze.style.display = "inline-block";
  } catch (err) {
    console.error("Camera access error:", err);
    alert("Camera permission denied or camera device unavailable.");
    switchTab("upload");
  }
}

function stopCameraStream() {
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach(track => track.stop());
    state.mediaStream = null;
  }
}

function flipCamera() {
  state.facingMode = state.facingMode === "user" ? "environment" : "user";
  startCameraStream();
}

function captureAndAnalyze() {
  if (!el.cameraVideo.videoWidth) {
    alert("Camera optical sensor initializing.");
    return;
  }

  const canvas = el.cameraCanvas;
  canvas.width = el.cameraVideo.videoWidth;
  canvas.height = el.cameraVideo.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(el.cameraVideo, 0, 0, canvas.width, canvas.height);

  const base64Data = canvas.toDataURL("image/jpeg", 0.94);
  state.capturedBase64 = base64Data;
  state.selectedFile = null;

  el.capturedFrameImg.src = base64Data;
  el.cameraSnapshotPreview.style.display = "flex";
  el.btnRetakeCamera.style.display = "inline-block";
  el.previewFilename.textContent = `Capture_${new Date().toISOString().slice(11,19)}.jpg`;
  el.previewMetaDimensions.textContent = `${canvas.width} x ${canvas.height}`;
  el.previewMetaSize.textContent = "Optical Capture";

  executeAnalysis();
}

function retakeCamera() {
  state.capturedBase64 = null;
  el.cameraSnapshotPreview.style.display = "none";
  el.btnRetakeCamera.style.display = "none";
}

// ---------------------------------------------------------------------
// 4. FILE INPUT & PREVIEW
// ---------------------------------------------------------------------
function handleFileInput(e) {
  if (e.target.files && e.target.files.length > 0) {
    loadSelectedFile(e.target.files[0]);
  }
}

function loadSelectedFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Invalid input format. Supported: JPEG, PNG, WebP, BMP.");
    return;
  }

  state.currentMode = "upload";
  state.selectedFile = file;
  state.capturedBase64 = null;

  const reader = new FileReader();
  reader.onload = (e) => {
    el.previewImage.src = e.target.result;
    el.dropzoneEmpty.style.display = "none";
    el.dropzonePreview.style.display = "flex";
    el.previewFilename.textContent = file.name;
    el.btnRemoveFile.style.display = "inline";
    el.viewportStatusText.textContent = `Loaded: ${file.name}`;

    const tmpImg = new Image();
    tmpImg.onload = () => {
      el.previewMetaDimensions.textContent = `${tmpImg.width} x ${tmpImg.height}`;
    };
    tmpImg.src = e.target.result;

    const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
    el.previewMetaSize.textContent = `${sizeInMB} MB`;
  };
  reader.readAsDataURL(file);
}

function resetUploadState() {
  state.selectedFile = null;
  el.fileInput.value = "";
  el.dropzoneEmpty.style.display = "flex";
  el.dropzonePreview.style.display = "none";
  el.previewFilename.textContent = "No file loaded";
  el.previewMetaDimensions.textContent = "--";
  el.previewMetaSize.textContent = "--";
  el.btnRemoveFile.style.display = "none";
  el.viewportStatusText.textContent = "Awaiting input source";
}

// ---------------------------------------------------------------------
// 5. DIAGNOSTIC INFERENCE EXECUTION
// ---------------------------------------------------------------------
async function executeAnalysis() {
  if (!state.selectedFile && !state.capturedBase64) {
    alert("No evidence loaded. Please select an image or capture a webcam frame.");
    return;
  }

  setAnalyzingState(true);

  try {
    let response;

    if (state.capturedBase64) {
      response = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: state.capturedBase64, is_webcam: true })
      });
    } else if (state.selectedFile) {
      const formData = new FormData();
      formData.append("image", state.selectedFile);
      response = await fetch("/api/detect", {
        method: "POST",
        body: formData
      });
    }

    const rawText = await response.text();

    let data = null;
    try {
      data = JSON.parse(rawText);
    } catch (_) {
      data = null;
    }

    if (!response.ok) {
      const errMsg = (data && data.error)
        ? data.error
        : `Server Error (${response.status}: ${response.statusText || "Internal Error"}). The server may be busy or reloading.`;
      throw new Error(errMsg);
    }

    if (!data) {
      throw new Error("Invalid response format received from server.");
    }

    state.lastAnalysisResult = data;
    renderDiagnosticReport(data);

  } catch (err) {
    console.error("Diagnostic execution failed:", err);
    alert(`Diagnostic Error: ${err.message}`);
  } finally {
    setAnalyzingState(false);
  }
}

function setAnalyzingState(isAnalyzing) {
  el.btnAnalyze.disabled = isAnalyzing;
  if (isAnalyzing) {
    el.analyzeBtnLabel.textContent = "Executing Signal Analysis...";
    el.analysisProgress.style.display = "block";
  } else {
    el.analyzeBtnLabel.textContent = "Execute Diagnostic Scan";
    el.analysisProgress.style.display = "none";
  }
}

// ---------------------------------------------------------------------
// 6. RENDER DIAGNOSTIC REPORT
// ---------------------------------------------------------------------
function renderDiagnosticReport(data) {
  const isReal = data.verdict_code === "REAL";
  const confidence = data.confidence;
  const now = new Date().toLocaleTimeString();

  // 1. Verdict Card
  el.metaTimestamp.textContent = now;
  if (isReal) {
    el.verdictCard.className = "verdict-box real";
    el.verdictTag.textContent = "STATUS: VERIFIED AUTHENTIC PHOTO";
    el.verdictTitle.textContent = "Authentic Photographic Baseline";
    el.verdictDesc.textContent = "Signal decomposition confirms organic spectral falloff, expected sensor noise entropy in HH subbands, and coherent spatial boundaries.";
  } else {
    el.verdictCard.className = "verdict-box ai";
    el.verdictTag.textContent = "STATUS: SYNTHETIC ARTIFACTS DETECTED";
    el.verdictTitle.textContent = "Synthetic / AI-Generated Specimen";
    el.verdictDesc.textContent = "Detected characteristic generative artifacts: high-frequency spectral attenuation, latent upsampling checkerboard spikes, and synthetic texture repetition.";
  }

  // 2. Metrics Table
  el.verdictConfidence.textContent = `${confidence.toFixed(1)}%`;
  el.probRealVal.textContent = `${data.probabilities.real}%`;
  el.probAiVal.textContent = `${data.probabilities.ai_generated}%`;
  el.metaLatency.textContent = `${data.metadata.inference_time_ms} ms`;
  el.metaDevice.textContent = data.metadata.device.toUpperCase();

  // 3. Gating Table
  el.gatingSpatial.textContent = `${data.gating_weights.fingerprint || data.gating_weights.spatial}%`;
  el.gatingWavelet.textContent = `${data.gating_weights.wavelet}%`;
  el.gatingClip.textContent = `${data.gating_weights.clip}%`;

  // 4. Grad-CAM, Wavelet Subbands & FFT
  if (data.forensics.gradcam_heatmap && el.imgGradCAM) {
    el.imgGradCAM.src = data.forensics.gradcam_heatmap;
  }

  const sub = data.forensics.wavelet_subbands;
  el.imgSubbandLL.src = sub.ll_approx;
  el.imgSubbandLH.src = sub.lh_horizontal;
  el.imgSubbandHL.src = sub.hl_vertical;
  el.imgSubbandHH.src = sub.hh_diagonal;
  el.imgFFTSpectrum.src = data.forensics.fft_spectrum;

  const met = data.forensics.wavelet_metrics;
  el.energyLL.textContent = met.energy_ll.toLocaleString();
  el.energyLH.textContent = met.energy_lh;
  el.energyHL.textContent = met.energy_hl;
  el.energyHH.textContent = met.energy_hh;
  el.highFreqRatio.textContent = `${met.high_freq_ratio}%`;

  // 5. Append to Audit Trail
  const specimenName = state.selectedFile 
    ? state.selectedFile.name 
    : "Optical Capture";

  appendAuditRecord({
    time: now,
    specimen: specimenName,
    dimensions: data.metadata.dimensions,
    verdict: data.prediction,
    isReal: isReal,
    confidence: `${confidence.toFixed(1)}%`
  });
}

// ---------------------------------------------------------------------
// 7. SESSION AUDIT TRAIL
// ---------------------------------------------------------------------
function appendAuditRecord(record) {
  state.history.unshift(record);
  if (state.history.length > 15) state.history.pop();
  saveSessionHistory();
  renderAuditTable();
}

function renderAuditTable() {
  if (state.history.length === 0) {
    el.historyTableBody.innerHTML = '<tr><td colspan="5" class="empty-cell">No evidentiary scans recorded in this session.</td></tr>';
    return;
  }

  el.historyTableBody.innerHTML = "";
  state.history.forEach(item => {
    const row = document.createElement("tr");
    const statusClass = item.isReal ? "status-cell-real" : "status-cell-ai";
    row.innerHTML = `
      <td class="mono">${item.time}</td>
      <td>${item.specimen}</td>
      <td class="mono">${item.dimensions}</td>
      <td class="${statusClass}">${item.verdict}</td>
      <td class="mono bold">${item.confidence}</td>
    `;
    el.historyTableBody.appendChild(row);
  });
}

function saveSessionHistory() {
  try {
    sessionStorage.setItem("artgate_forensic_audit", JSON.stringify(state.history));
  } catch (e) {}
}

function loadSessionHistory() {
  try {
    const raw = sessionStorage.getItem("artgate_forensic_audit");
    if (raw) {
      state.history = JSON.parse(raw);
      renderAuditTable();
    }
  } catch (e) {}
}

function clearSessionHistory() {
  state.history = [];
  saveSessionHistory();
  renderAuditTable();
}

function exportRawTelemetry() {
  if (!state.lastAnalysisResult) {
    alert("No active analysis result to export. Please run a scan first.");
    return;
  }
  const jsonStr = JSON.stringify(state.lastAnalysisResult, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ArtGate_Forensic_Telemetry_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------
// 8. MODALS
// ---------------------------------------------------------------------
function openInspectorModal(title, desc, imgSrc) {
  el.modalTitle.textContent = title;
  el.modalDesc.textContent = desc;
  el.modalImg.src = imgSrc;
  el.inspectorModal.style.display = "flex";
}

function closeInspectorModal() {
  el.inspectorModal.style.display = "none";
}
