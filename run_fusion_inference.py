"""
Full multimodal inference script using the ArtGate Fusion Detector.
Takes any input image, extracts:
1. Spatial features via the ImageMALL MobileNetV2 checkpoint
2. Frequency-domain features via 2D Wavelet DWT (LH, HL, HH detail subbands)
3. Vision-Language semantic features via CLIP-ViT
4. Fuses them using adaptive gating and outputs the AI-Generated vs Real prediction.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from fusion_detector import ArtGateFusionDetector


def evaluate_image(image_path: str, checkpoint_path: str = "ProScan/electronics_model.pth"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("      ArtGate Multi-Branch Fusion AI Detection Inference")
    print("=" * 65)
    print(f"[*] Target Image:     {image_path}")
    print(f"[*] Checkpoint:       {checkpoint_path}")
    print(f"[*] Compute Device:   {device}")

    # Standard 224x224 RGB image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # Instantiate ArtGate fusion model
    model = ArtGateFusionDetector(imagemall_checkpoint=checkpoint_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # 1. Inspect individual branch feature representations
        branch_feats = model.extract_all_branch_features(tensor)
        # 2. Forward pass through fusion network
        logits, gates = model(tensor, return_gates=True)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    gate_weights = gates.squeeze(0).cpu().tolist()
    raw_logits = logits.squeeze(0).cpu().tolist()

    print("\n" + "-" * 65)
    print("                BRANCH FEATURE EXTRACTION")
    print("-" * 65)
    print(f" [1] Spatial Branch (MobileNetV2): {tuple(branch_feats['spatial_1280d'].shape)} vector")
    print(f"     ImageMALL legacy logits:      {branch_feats['imagemall_legacy_logits'].squeeze(0).cpu().tolist()}")
    print(f" [2] Frequency Branch (Wavelet):   {tuple(branch_feats['wavelet_256d'].shape)} vector")
    print(f" [3] Semantic Branch (CLIP-ViT):   {tuple(branch_feats['clip_512d'].shape)} vector")

    print("\n" + "-" * 65)
    print("             CROSS-BRANCH ADAPTIVE GATING")
    print("-" * 65)
    print(f"  * Spatial Weight  (w_spatial): {gate_weights[0]*100:.1f}%")
    print(f"  * Wavelet Weight  (w_wavelet): {gate_weights[1]*100:.1f}%")
    print(f"  * CLIP-ViT Weight (w_clip):    {gate_weights[2]*100:.1f}%")

    print("\n" + "=" * 65)
    print("                    DETECTION RESULTS")
    print("=" * 65)
    print(f" Raw Fusion Logits: {raw_logits}")
    print(f" Probability Real:         {probs[0]*100:.2f}%")
    print(f" Probability AI-Generated: {probs[1]*100:.2f}%")

    pred_label = "Real" if probs[0] > probs[1] else "AI-Generated"
    confidence = max(probs[0], probs[1]) * 100
    print(f"\n >>> FINAL PREDICTION: {pred_label} ({confidence:.2f}% confidence) <<<")
    print("=" * 65)

    return {
        "raw_logits": raw_logits,
        "prob_real": probs[0],
        "prob_ai": probs[1],
        "gates": gate_weights,
        "branch_features": branch_feats
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal ArtGate AI Image Detection Inference")
    parser.add_argument(
        "--image",
        type=str,
        default="ProScan/Dataset/electronics/Original/daniel-korpai-DDEOxavdeIk-unsplash.jpg",
        help="Path to image file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ProScan/electronics_model.pth",
        help="Path to ImageMALL .pth checkpoint"
    )
    args = parser.parse_args()

    evaluate_image(args.image, args.checkpoint)
