# ArtGate: Universal AI Image Forensics & Camera Fingerprinting

An advanced multi-branch AI forensic detection system capable of discerning synthetic AI-generated images (portraits, AI-generated professional headshots, text-to-image diffusion, Midjourney, DALL-E, StyleGAN, FLUX) from authentic physical camera photographs.

---

## Quick Start (For Running on Another Laptop)

### Prerequisites
- **Python 3.10 to 3.12+** installed on the computer.
- When installing Python, make sure to check **"Add Python to PATH"**.

---

### Step-by-Step Setup

#### Method 1: One-Click Run (Windows)
1. Copy or extract the project folder to the laptop.
2. Double-click **`run.bat`**.
3. It will automatically verify Python, install any missing dependencies from `requirements.txt`, start the forensic server, and open `http://localhost:5000` in your web browser.

---

#### Method 2: Manual Setup (Windows / Mac / Linux)

1. **Open Terminal / Command Prompt** in the project folder:
   ```bash
   cd "path/to/image detection"
   ```

2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the API & Web Server**:
   ```bash
   python server.py
   ```

4. **Access the Workbench**:
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

---

## Essential Files to Share

When sharing this project with your friend (e.g., via ZIP file, USB drive, or Google Drive), make sure the following files and folders are included:

```
image detection/
├── models/
│   └── face_detector_svm.joblib    <-- Trained multi-generator face & headshot SVM
├── static/
│   ├── index.html                  <-- Modern ArtGate Web UI
│   ├── app.js                      <-- Interactive frontend logic
│   ├── style.css                   <-- Styling & glassmorphism theme
│   └── samples/                    <-- Built-in reference benchmark images
├── forensic_utils.py               <-- Signal processing, noise variance, CLIP & SVM
├── fusion_detector.py              <-- ResNet-101 GAP, Wavelets, and Grad-CAM engine
├── server.py                       <-- Flask API backend server
├── requirements.txt                <-- Python package dependencies
├── run.bat                         <-- One-click Windows launcher script
└── README.md                       <-- Instructions
```

> **Tip**: You can exclude `__pycache__` and `.git` folders to keep the ZIP file small and clean!

---

## How It Works
- **Physical Sensor Noise**: Measures Poisson-Gaussian CMOS optical sensor shot noise floor vs. hyper-smooth latent diffusion skin textures.
- **Wavelet Frequency Decomposition**: 2D Haar Discrete Wavelet Transform (LL, LH, HL, HH subbands).
- **Deep Camera Fingerprint**: ResNet-101 Global Average Pooling (GAP) 2048-dimensional features.
- **Grad-CAM Attention Map**: ResNet-101 Layer 4 spatial activation heatmaps.
- **Multimodal Semantic Verification**: Vision-Language zero-shot prompt ensemble.
- **Calibrated Multi-Generator SVM**: Calibrated on real photographs (NVIDIA FFHQ) and synthetic faces (StyleGAN + SDXL Diffusion).
