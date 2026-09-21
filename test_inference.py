"""
Inference script for the extracted ImageMALL (ProScan) MobileNetV2 models.
Supports loading beauty_model.pth, electronics_model.pth, fragrance_model.pth,
or category_model.pth and running inference on an arbitrary test image.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image


def load_model(checkpoint_path: str, num_classes: int = 2, device: torch.device = None):
    """
    Instantiates MobileNetV2 architecture matching the training script,
    replaces classifier[1] with Linear(1280, num_classes), and loads state_dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Build base MobileNetV2
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def get_transform(mode: str = "train"):
    """
    Returns image transformation pipeline.
    - 'train': Matches train_model.py (Normalize mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    - 'app': Matches app.py (Resize + ToTensor without normalization)
    - 'imagenet': Standard torchvision ImageNet normalization
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    elif mode == "app":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif mode == "imagenet":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unknown transform mode: {mode}")


def run_inference(image_path: str, checkpoint_path: str, transform_mode: str = "train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running inference on device: {device}")
    print(f"[*] Loading model from: {checkpoint_path}")
    print(f"[*] Target image: {image_path}")
    print(f"[*] Transform mode: {transform_mode}")

    is_category_model = "category" in os.path.basename(checkpoint_path).lower()
    num_classes = 3 if is_category_model else 2

    model = load_model(checkpoint_path, num_classes=num_classes, device=device)
    transform = get_transform(transform_mode)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)

    raw_logits_list = logits.squeeze(0).cpu().tolist()
    probs_list = probabilities.squeeze(0).cpu().tolist()

    print("\n" + "=" * 50)
    print("                 INFERENCE RESULTS")
    print("=" * 50)
    print(f"Raw Logits:       {raw_logits_list}")
    print(f"Probabilities:    {probs_list}")

    if is_category_model:
        class_names = ["beauty", "electronics", "fragrance"]
        for idx, name in enumerate(class_names):
            print(f" - {name.capitalize()}: {probs_list[idx]*100:.2f}%")
        pred_idx = int(torch.argmax(probabilities, dim=1).item())
        print(f"\nPredicted Category: {class_names[pred_idx]} ({probs_list[pred_idx]*100:.2f}%)")
    else:
        # Based on Dataset/ class sorting:
        # beauty: Fake (0), Original (1)
        # electronics: Fake (0), Original (1)
        # fragrance: fragrance (0), Original (1)
        class_names = ["Fake / Class 0", "Original / Class 1"]
        print(f" - Class 0 (Fake):     {probs_list[0]*100:.2f}% (Logit: {raw_logits_list[0]:.4f})")
        print(f" - Class 1 (Original): {probs_list[1]*100:.2f}% (Logit: {raw_logits_list[1]:.4f})")
        pred_idx = int(torch.argmax(probabilities, dim=1).item())
        print(f"\nPredicted Label: {class_names[pred_idx]} (Confidence: {probs_list[pred_idx]*100:.2f}%)")

    print("=" * 50)
    return logits, probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with ImageMALL MobileNetV2 model")
    parser.add_argument(
        "--image",
        type=str,
        default="ProScan/Dataset/electronics/Original/cosmin-ursea-0QAe85hi_Mw-unsplash.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ProScan/electronics_model.pth",
        help="Path to .pth checkpoint file"
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="train",
        choices=["train", "app", "imagenet"],
        help="Preprocessing transform to apply"
    )
    args = parser.parse_args()

    run_inference(args.image, args.checkpoint, args.transform)
