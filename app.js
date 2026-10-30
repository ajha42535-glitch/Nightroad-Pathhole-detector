// NightRoad Guardian frontend — connected to the local FastAPI + YOLO backend.
const API_BASE_URL = "http://127.0.0.1:8000";
const $ = (selector) => document.querySelector(selector);
const imageInput = $("#imageInput");
const preview = $("#preview");
const fileName = $("#fileName");
const resultState = $("#resultState");
const cameraDialog = $("#cameraDialog");
const cameraVideo = $("#cameraVideo");
const cameraMessage = $("#cameraMessage");
const toast = $("#toast");
let selectedFile = null;
let cameraStream = null;
let coords = null;
let toastTimer = null;
let latestDetection = null;
let inspectionTotal = 0;
let potholeTotal = 0;
let localReports = [];

function notify(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3500);
}
function setStatus(message, online = false) {
  const status = $("#backendStatus");
  status.textContent = message;
  const dot = status.parentElement?.querySelector("i");
  if (dot) dot.style.background = online ? "#19e5a5" : "#ffbd59";
}
function safeText(value) {
  return String(value ?? "").replace(/[&<>\"']/g, (ch) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}
function resetResult() {
  latestDetection = null;
  $("#potholeCount").textContent = "—";
  $("#confidence").textContent = "—";
  resultState.innerHTML = '<div class="checkmark">✓</div><h3>Ready for your image</h3><p>Choose a road image and AI detection will start automatically.</p>';
  $("#analysisHint").textContent = "Waiting for AI analysis";
  const resultStatus = $(".results-panel .status span");
  if (resultStatus) resultStatus.textContent = "Ready";
}
function showPreview(file) {
  if (!file || !file.type.startsWith("image/")) {
    notify("Please choose a valid image file (JPG, PNG or WEBP).");
    return false;
  }
  if (file.size > 15 * 1024 * 1024) {
    notify("Image is too large. Please choose an image under 15 MB.");
    return false;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  preview.replaceChildren();
  const img = document.createElement("img");
  img.src = url;
  img.alt = "Selected road image preview";
  img.onload = () => {
    $("#resolution").textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
    URL.revokeObjectURL(url);
  };
  preview.appendChild(img);
  fileName.textContent = file.name || "Camera capture";
  resetResult();
  notify("Image loaded. Starting AI detection…");
  return true;
}
imageInput.addEventListener("change", () => {
  if (showPreview(imageInput.files[0])) detectSelectedImage();
});
$("#clearBtn").addEventListener("click", () => {
  selectedFile = null;
  latestDetection = null;
  imageInput.value = "";
  preview.innerHTML = '<div class="empty-state"><div class="empty-icon">▧</div><b>Your road image will appear here</b><p>Upload an image to get started.</p></div>';
  fileName.textContent = "No image selected";
  $("#resolution").textContent = "—";
  resetResult();
});

// Real AI request: sends the selected image to POST /api/detect.
async function detectSelectedImage() {
  if (!selectedFile) {
    notify("First upload a road image or capture one with the camera.");
    return;
  }
  const button = $("#analyzeBtn");
  const oldText = button.textContent;
  button.disabled = true;
  button.textContent = "Analysing…";
  $("#analysisHint").textContent = "Sending image to YOLO model…";
  resultState.innerHTML = '<div class="checkmark">…</div><h3>AI is analysing your image</h3><p>Please wait. This may take a little time on the first run.</p>';
  try {
    const form = new FormData();
    form.append("file", selectedFile, selectedFile.name || "road-image.jpg");
    if (coords) {
      form.append("latitude", String(coords.lat));
      form.append("longitude", String(coords.lon));
    }
    const response = await fetch(`${API_BASE_URL}/api/detect`, { method: "POST", body: form });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Backend error (${response.status})`);

    const detections = Array.isArray(data.detections) ? data.detections : [];
    const count = Number(data.pothole_count ?? data.detection_count ?? detections.length);
    const confidenceValues = detections.map(d => Number(d.confidence)).filter(Number.isFinite);
    const avgConfidence = Number(data.average_confidence);
    const confidence = Number.isFinite(avgConfidence) && avgConfidence > 0
      ? avgConfidence
      : (confidenceValues.length ? confidenceValues.reduce((a,b) => a+b, 0) / confidenceValues.length : 0);
    latestDetection = { ...data, pothole_count: count, detections };
    inspectionTotal++;
    potholeTotal += count;
    $("#potholeCount").textContent = String(count);
    $("#confidence").textContent = confidence > 0 ? `${Math.round(confidence * 100)}%` : "—";
    $("#analysisHint").textContent = `Model: ${data.model_name || "YOLO"}`;
    const resultStatus = $(".results-panel .status span");
    if (resultStatus) resultStatus.textContent = "AI result";

    if (data.annotated_image) {
      const img = document.createElement("img");
      img.src = data.annotated_image;
      img.alt = "YOLO pothole detection result with bounding boxes";
      preview.replaceChildren(img);
    }
    const labels = detections.slice(0, 8).map(d => `${safeText(d.label ?? d.class_name ?? `Class ${d.class_id ?? "?"}`)} (${Math.round(Number(d.confidence || 0) * 100)}%)`);
    const details = labels.length ? `<p class="detection-labels">${labels.join(" · ")}</p>` : "";
    resultState.innerHTML = `<div class="checkmark">${count ? "!" : "✓"}</div><h3>${count ? `${count} possible pothole${count === 1 ? "" : "s"} detected` : "No potholes detected"}</h3><p>${count ? "Review the AI-marked boxes carefully. Predictions can be incorrect." : "The model returned zero detections for this image. This does not prove the road is damage-free."}</p>${details}`;
    notify("AI analysis completed. Review the marked image and results.");
  } catch (error) {
    resultState.innerHTML = `<div class="checkmark">!</div><h3>AI analysis failed</h3><p>${safeText(error.message || error)}</p>`;
    $("#analysisHint").textContent = "Detection failed";
    notify("Could not complete AI detection. Check the backend error shown in Results.");
  } finally {
    button.disabled = false;
    button.textContent = oldText;
  }
}
$("#analyzeBtn").addEventListener("click", detectSelectedImage);

$("#locationBtn").addEventListener("click", () => {
  if (!navigator.geolocation) { notify("Location is not supported by this browser."); return; }
  $("#locationLabel").textContent = "Requesting permission…";
  navigator.geolocation.getCurrentPosition(position => {
    coords = { lat: position.coords.latitude, lon: position.coords.longitude };
    const pretty = `${coords.lat.toFixed(5)}, ${coords.lon.toFixed(5)}`;
    $("#locationLabel").textContent = pretty;
    $("#coords").textContent = `Your coordinates: ${pretty}`;
    notify("Location added to this inspection.");
  }, () => {
    $("#locationLabel").textContent = "Location permission unavailable";
    notify("Could not get location. You can still analyse an image without it.");
  }, { enableHighAccuracy: true, timeout: 10000 });
});
$("#cameraBtn").addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) { notify("Camera needs a modern browser and a secure page (HTTPS or localhost)."); return; }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    cameraVideo.srcObject = cameraStream;
    cameraMessage.textContent = "Camera is active. Capture a road image when ready.";
    cameraDialog.showModal();
  } catch { notify("Camera could not open. Check browser permission."); }
});
function stopCamera() {
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = null;
  cameraVideo.srcObject = null;
}
$("#stopCamera").addEventListener("click", stopCamera);
$("#closeCamera").addEventListener("click", () => { stopCamera(); cameraDialog.close(); });
cameraDialog.addEventListener("close", stopCamera);
$("#captureBtn").addEventListener("click", () => {
  if (!cameraVideo.videoWidth) { notify("Camera is not ready yet."); return; }
  const canvas = document.createElement("canvas");
  canvas.width = cameraVideo.videoWidth; canvas.height = cameraVideo.videoHeight;
  canvas.getContext("2d").drawImage(cameraVideo, 0, 0);
  canvas.toBlob(blob => {
    if (blob) showPreview(new File([blob], `road-capture-${Date.now()}.jpg`, { type: "image/jpeg" }));
    stopCamera(); cameraDialog.close();
  }, "image/jpeg", .92);
});

// Local draft reports are clearly labelled as local; the current API does not save reports.
$("#reportForm").addEventListener("submit", event => {
  event.preventDefault();
  const name = $("#reportName").value.trim();
  if (!name) { notify("Enter a road or area name first."); $("#reportName").focus(); return; }
  const row = document.createElement("div"); row.className = "report-row";
  const pin = document.createElement("span"); pin.className = "pin"; pin.textContent = "⌖";
  const details = document.createElement("div");
  const title = document.createElement("b"); title.textContent = name;
  const sub = document.createElement("small"); sub.textContent = coords ? `${coords.lat.toFixed(4)}, ${coords.lon.toFixed(4)} · Local draft` : "Saved in this browser session · Local draft";
  details.append(title, sub);
  const pill = document.createElement("span"); pill.className = "pill pending"; pill.textContent = "Draft";
  row.append(pin, details, pill); $("#reportList").prepend(row);
  $("#reportName").value = ""; localReports.push(name);
  notify("Local draft added. It has not been sent to a server.");
});
$("#viewReports").addEventListener("click", () => { $("#reports").scrollIntoView({ behavior: "smooth", block: "center" }); });
document.querySelectorAll(".nav-link").forEach(link => link.addEventListener("click", () => {
  document.querySelectorAll(".nav-link").forEach(item => item.classList.remove("active")); link.classList.add("active");
}));

async function checkBackend() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) throw new Error("Health check failed");
    const data = await response.json();
    setStatus(data.model_configured ? "AI model configured" : "Backend online · model missing", true);
    $("#analysisHint").textContent = data.model_configured ? "AI model configured — ready to test" : "Backend is online, but model weights are missing";
  } catch {
    setStatus("Backend offline", false);
    $("#analysisHint").textContent = "Start the Python backend to enable AI detection";
  }
}
checkBackend();
