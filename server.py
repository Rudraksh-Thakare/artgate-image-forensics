"""
Universal AI Image Forensics & Camera Fingerprinting API Server.

Serves:
- Static Frontend at '/'
- GET /api/health: Diagnostics, ResNet101 status, and branch dimensions
- GET /api/samples: Universal AI vs Real benchmark sample cards (Portraits, Landscapes, Art, Photography)
- POST /api/detect: File upload or Base64 camera capture with Grad-CAM, Wavelets, and FFT forensics
"""

import os
import io
import time
import base64
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torchvision import transforms

from fusion_detector import ArtGateFusionDetector, extract_center_patch
from forensic_utils import (
    generate_wavelet_subbands,
    generate_fft_spectrum,
    analyze_sensor_noise_and_features,
    generate_gradcam_overlay,
    set_shared_model
)


# ---------------------------------------------------------------------
# APP CONFIGURATION & MODEL INITIALIZATION
# ---------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Restrict PyTorch thread pool to reduce memory overhead in containerized environments
torch.set_num_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Starting Universal AI Detection Server on device: {device}")

# Check for custom trained camera checkpoint, otherwise use ResNet101 default
CAMERA_CHECKPOINT = "camera_resnet101.pth" if os.path.exists("camera_resnet101.pth") else None
FACE_SVM_CHECKPOINT = "models/face_detector_svm.joblib" if os.path.exists("models/face_detector_svm.joblib") else None

model = None

def get_model():
    global model
    if model is None:
        print(f"[*] Initializing ResNet101 Camera Fingerprint & Fusion Engine (Custom Weights: {CAMERA_CHECKPOINT})...")
        model = ArtGateFusionDetector(camera_checkpoint=CAMERA_CHECKPOINT)
        model.to(device)
        model.eval()
        set_shared_model(model)
        print(f"[+] Universal Forensic & Fingerprinting Engine ready (Face SVM: {FACE_SVM_CHECKPOINT}).")
    return model


# Background warmup: lets the server bind the port immediately, then warms up models
import threading

def _background_warmup():
    time.sleep(2.0)
    print("[*] Background pre-warming inference models...")
    try:
        get_model()
        print("[+] Background warmup completed. Ready for instant inference!")
    except Exception as e:
        print(f"[!] Warmup info: {e}")

threading.Thread(target=_background_warmup, daemon=True).start()


# Standard input transform
input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Paper-style center patch transform (no resizing)
patch_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ---------------------------------------------------------------------
# STATIC & WEB ROUTES
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory("static", path)


# ---------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Universal-AI-Image-Forensics",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "checkpoint": CAMERA_CHECKPOINT or "ResNet101-Default-Backbone",
        "face_svm": "Active (ResNet101 GAP + CLIP Vision RBF-SVM)" if FACE_SVM_CHECKPOINT else "Inactive",
        "methodology": "Manisha et al. IEEE Access 2025 + Multimodal Fusion",
        "branches": [
            {"name": "Camera Fingerprint", "arch": "ResNet101 (GAP)", "dimension": 2048},
            {"name": "Frequency Domain", "arch": "2D Wavelet DWT (Haar)", "dimension": 256},
            {"name": "Semantic Vision-Language", "arch": "CLIP-ViT (Base/32)", "dimension": 512}
        ]
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    samples = [
        {
            "id": "sample_ai_portrait",
            "title": "AI Cybernetic Portrait",
            "category": "AI-Generated",
            "type": "Diffusion Face",
            "url": "/samples/ai_portrait.jpg",
            "ground_truth": "AI"
        },
        {
            "id": "sample_ai_landscape",
            "title": "AI Surreal Floating Island",
            "category": "AI-Generated",
            "type": "Midjourney Landscape",
            "url": "/samples/ai_landscape.jpg",
            "ground_truth": "AI"
        },
        {
            "id": "sample_real_camera",
            "title": "Authentic DSLR Camera",
            "category": "Authentic",
            "type": "Physical Camera Hardware",
            "url": "/samples/real_camera.jpg",
            "ground_truth": "REAL"
        },
        {
            "id": "sample_real_perfume",
            "title": "Authentic Optical Photography",
            "category": "Authentic",
            "type": "Macro Camera Lens",
            "url": "/samples/real_perfume.jpg",
            "ground_truth": "REAL"
        },
        {
            "id": "sample_real_skincare",
            "title": "Authentic Studio Photography",
            "category": "Authentic",
            "type": "Physical Sensor Exposure",
            "url": "/samples/real_skincare.jpg",
            "ground_truth": "REAL"
        }
    ]
    return jsonify({"status": "success", "samples": samples})


@app.route("/api/detect", methods=["POST"])
def detect():
    t_start = time.time()
    img = None
    source = "upload"
    is_camera = False
    use_patch_mode = request.args.get("mode", "full") == "patch"

    try:
        # Case 1: Base64 payload (live camera stream or direct base64 image)
        if request.is_json and "image_base64" in request.json:
            is_camera = bool(request.json.get("is_webcam", False))
            source = "Hardware Camera Stream" if is_camera else "Base64 Image Upload"
            b64_data = request.json["image_base64"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Case 2: Multipart form file upload
        elif "image" in request.files:
            source = "File Upload"
            is_camera = False
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "No file selected."}), 400
            img = Image.open(file.stream).convert("RGB")

        # Case 3: URL/Sample request
        elif request.is_json and "image_url" in request.json:
            url_path = request.json["image_url"].lstrip("/")
            local_path = os.path.join("static", url_path)
            if not os.path.exists(local_path):
                local_path = url_path
            if os.path.exists(local_path):
                img = Image.open(local_path).convert("RGB")
                source = f"Benchmark Sample ({os.path.basename(local_path)})"
                is_camera = False
            else:
                return jsonify({"error": f"Image file not found at: {local_path}"}), 404

        if img is None:
            return jsonify({"error": "No valid image provided."}), 400

        width, height = img.size

        detector = get_model()

        # -------------------------------------------------------------
        # 1. PHYSICAL SENSOR NOISE & CAMERA FINGERPRINT ANALYSIS
        # -------------------------------------------------------------
        physical_forensics = analyze_sensor_noise_and_features(img, is_camera=is_camera)

        # -------------------------------------------------------------
        # 2. NEURAL MULTI-BRANCH FEATURE & GRAD-CAM EXTRACTION
        # -------------------------------------------------------------
        if use_patch_mode:
            # Paper Section II-B: Center Patch 224x224 without resizing
            input_img = extract_center_patch(img, patch_size=224)
            tensor = patch_transform(input_img).unsqueeze(0).to(device)
        else:
            input_img = img
            tensor = input_transform(img).unsqueeze(0).to(device)

        # Feature representation & dynamic cross-branch gating
        with torch.no_grad():
            branch_feats = detector.extract_all_branch_features(tensor)
            logits, gates = detector(tensor, return_gates=True)
            gate_vals = gates.squeeze(0).cpu().numpy()

        gating_weights = {
            "fingerprint": round(float(gate_vals[0]) * 100.0, 1),
            "wavelet": round(float(gate_vals[1]) * 100.0, 1),
            "clip": round(float(gate_vals[2]) * 100.0, 1)
        }

        # Grad-CAM Visual Attention Map (Layer4 of ResNet101)
        gradcam_map = detector.generate_gradcam(tensor)
        gradcam_overlay_b64 = generate_gradcam_overlay(input_img, gradcam_map, alpha=0.55)

        # -------------------------------------------------------------
        # 3. WAVELET & SPECTRAL FORENSIC GENERATION
        # -------------------------------------------------------------
        wavelet_data = generate_wavelet_subbands(img)
        fft_spectrum_b64 = generate_fft_spectrum(img)

        inference_time_ms = round((time.time() - t_start) * 1000, 1)

        # -------------------------------------------------------------
        # 4. CONSTRUCT STRUCTURED RESPONSE
        # -------------------------------------------------------------
        response_payload = {
            "prediction": physical_forensics["prediction"],
            "verdict_code": physical_forensics["verdict_code"],
            "description": physical_forensics["description"],
            "confidence": physical_forensics["confidence"],
            "probabilities": {
                "real": physical_forensics["prob_real"],
                "ai_generated": physical_forensics["prob_ai"]
            },
            "gating_weights": gating_weights,
            "branch_summary": {
                "fingerprint_vector_dim": branch_feats["resnet101_fingerprint_2048d"].shape[1],
                "wavelet_vector_dim": branch_feats["wavelet_256d"].shape[1],
                "clip_vector_dim": branch_feats["clip_512d"].shape[1]
            },
            "forensics": {
                "gradcam_heatmap": gradcam_overlay_b64,
                "wavelet_subbands": wavelet_data["visualizations"],
                "wavelet_metrics": wavelet_data["metrics"],
                "physical_sensor_metrics": physical_forensics["metrics"],
                "fft_spectrum": fft_spectrum_b64
            },
            "metadata": {
                "dimensions": f"{width} x {height}",
                "aspect_ratio": round(width / height, 2) if height > 0 else 1.0,
                "inference_time_ms": inference_time_ms,
                "source": source,
                "device": str(device),
                "mode": "Center Patch 224x224 (Manisha et al.)" if use_patch_mode else "Full-Frame Analysis"
            }
        }

        return jsonify(response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Flask server on http://0.0.0.0:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
