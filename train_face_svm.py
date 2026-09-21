"""
Universal AI Image Forensics: Face & Headshot Classifier Trainer (Manisha et al. IEEE Access 2025 + Multimodal Fusion).

Extracts 2048-d ResNet-101 GAP features + 512-d CLIP vision embeddings from:
- Authentic camera portraits (NVIDIA FFHQ)
- Synthetic GAN faces (StyleGAN)
- Synthetic Latent Diffusion faces (Stable Diffusion XL / SDXL)
trains an RBF-kernel Support Vector Machine (SVM) with probability calibration,
and exports the model to 'models/face_detector_svm.joblib'.
"""

import os
import io
import time
import joblib
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from huggingface_hub import hf_hub_download


def extract_features_batch(images, fe_resnet, res_transform, clip_model, clip_proc, device):
    """
    Extracts concatenated ResNet-101 GAP (2048d) + CLIP vision (512d) normalized feature vectors.
    """
    features = []
    with torch.no_grad():
        for img in images:
            # 1. ResNet-101 2048-d GAP feature
            t_res = res_transform(img).unsqueeze(0).to(device)
            f_res = fe_resnet(t_res).squeeze().cpu().numpy()
            norm_res = np.linalg.norm(f_res) + 1e-8
            f_res = f_res / norm_res

            # 2. CLIP 512-d Vision feature
            t_clip = clip_proc(images=img, return_tensors="pt").to(device)
            out_clip = clip_model.get_image_features(**t_clip)
            f_clip = out_clip.pooler_output.squeeze().cpu().numpy()
            norm_clip = np.linalg.norm(f_clip) + 1e-8
            f_clip = f_clip / norm_clip

            # Fused 2560-d vector
            f_fused = np.concatenate([f_res, f_clip])
            features.append(f_fused)

    return np.array(features)


def main():
    print("=" * 70)
    print("UNIVERSAL AI FORENSICS: MULTI-GENERATOR FACE SVM TRAINER")
    print("Generators: NVIDIA FFHQ (Real) vs StyleGAN (GAN) + SDXL (Diffusion)")
    print("Methodology: Manisha et al. (IEEE Access 2025) + Multimodal Fusion")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # 1. Initialize Backbones
    print("[*] Loading ResNet-101 Feature Extractor...")
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    fe_resnet = torch.nn.Sequential(*modules).to(device).eval()

    res_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("[*] Loading CLIP Vision Backbone (openai/clip-vit-base-patch32)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 2. Load Datasets
    print("[*] Loading Real FFHQ & StyleGAN faces (TheKernel01/140k-Real-and-Fake-Faces)...")
    f_ffhq = hf_hub_download(
        repo_id="TheKernel01/140k-Real-and-Fake-Faces",
        filename="data/test-00000-of-00002.parquet",
        repo_type="dataset"
    )
    df_140k = pd.read_parquet(f_ffhq)

    print("[*] Loading SDXL Diffusion faces (bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces)...")
    f_sdxl = hf_hub_download(
        repo_id="bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces",
        filename="base_transforms/train-00000-of-00008.parquet",
        repo_type="dataset"
    )
    df_sdxl = pd.read_parquet(f_sdxl)

    n_real = 200
    n_stylegan = 100
    n_sdxl = 100

    print(f"[*] Sampling: {n_real} Real FFHQ, {n_stylegan} StyleGAN fakes, {n_sdxl} SDXL Diffusion fakes...")
    real_df = df_140k[df_140k["label"] == 0].head(n_real)
    stylegan_df = df_140k[df_140k["label"] == 1].head(n_stylegan)
    sdxl_df = df_sdxl.head(n_sdxl)

    real_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in real_df["image"].apply(lambda x: x["bytes"])]
    fake_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in stylegan_df["image"].apply(lambda x: x["bytes"])]
    fake_images += [Image.open(io.BytesIO(b)).convert("RGB") for b in sdxl_df["image"].apply(lambda x: x["bytes"])]

    # 3. Feature Extraction
    print(f"[*] Extracting features for {len(real_images)} Real faces...")
    t0 = time.time()
    X_real = extract_features_batch(real_images, fe_resnet, res_transform, clip_model, clip_proc, device)
    y_real = np.zeros(len(X_real), dtype=int)
    print(f"[+] Real features extracted in {time.time() - t0:.1f}s. Shape: {X_real.shape}")

    print(f"[*] Extracting features for {len(fake_images)} Fake faces (GAN + Diffusion)...")
    t1 = time.time()
    X_fake = extract_features_batch(fake_images, fe_resnet, res_transform, clip_model, clip_proc, device)
    y_fake = np.ones(len(X_fake), dtype=int)
    print(f"[+] Fake features extracted in {time.time() - t1:.1f}s. Shape: {X_fake.shape}")

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([y_real, y_fake])
    print(f"[+] Total Dataset: {X.shape[0]} specimens (200 Real, 200 Fake), {X.shape[1]} feature dimensions.")

    # 4. Stratified Cross-Validation
    print("[*] Running 5-Fold Stratified Cross-Validation on RBF-kernel SVM...")
    clf_pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.5, probability=True, random_state=42))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf_pipeline, X, y, cv=cv, scoring="accuracy")

    print(f"[+] 5-Fold Cross-Validation Scores: {[round(s * 100, 2) for s in scores]}%")
    print(f"[+] Mean Model Accuracy: {np.mean(scores) * 100:.2f}% (+/- {np.std(scores) * 100:.2f}%)")

    # 5. Fit Final Calibrated Model on Full Training Set
    print("[*] Training final calibrated SVM model on all samples...")
    clf_pipeline.fit(X, y)

    # 6. Save Model Checkpoint
    os.makedirs("models", exist_ok=True)
    out_model_path = os.path.join("models", "face_detector_svm.joblib")
    joblib.dump(clf_pipeline, out_model_path)
    print(f"[SUCCESS] Trained model successfully saved to: {out_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
