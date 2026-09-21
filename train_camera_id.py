"""
Camera Model Identification Pre-Training Script (Manisha et al. IEEE Access 2025).

Implements Section II-A:
1. Loads images/video I-frames captured by distinct physical camera models
   (e.g., from QUFVD, Dresden, VISION, or custom smartphone folders).
2. Resizes frames to 224x224 (introducing a low-pass filtering effect that
   eliminates fragile high-frequency details and guides ResNet101 to learn
   robust mid-frequency camera-specific sensor fingerprints).
3. Trains ResNet101 on multi-class camera attribution.
4. Exports the trained weights to 'camera_resnet101.pth' for use as a frozen
   2048-dimensional camera fingerprint extractor.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet101, ResNet101_Weights


def train_camera_model_identifier(
    data_dir: str,
    output_path: str = "camera_resnet101.pth",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4
):
    if not os.path.exists(data_dir):
        print(f"[!] Error: Dataset directory '{data_dir}' not found.")
        print("    Please provide a directory structured by camera model subfolders:")
        print("    dataset/")
        print("      ├── camera_model_1/")
        print("      ├── camera_model_2/")
        print("      └── ...")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("   ResNet101 Camera Identification Pre-training (IEEE Access 2025)")
    print("=" * 70)
    print(f"[*] Data Directory: {data_dir}")
    print(f"[*] Target Output:  {output_path}")
    print(f"[*] Compute Device: {device}")

    # Section II-A: Resizing to 224x224 introduces a low-pass filter effect
    # suppressing fragile high-frequency noise and emphasizing mid-frequency sensor fingerprints
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    classes = dataset.classes
    num_classes = len(classes)
    print(f"[*] Detected {num_classes} camera models: {classes}")

    if num_classes < 2:
        print("[!] Need at least 2 camera model classes to pre-train camera identification.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Initialize ResNet101
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100.0
        print(f"[Epoch {epoch:02d}/{epochs:02d}] Loss: {epoch_loss:.4f} | Camera-ID Accuracy: {epoch_acc:.2f}%")

    # Save checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "camera_classes": classes,
        "feature_dim": 2048
    }, output_path)
    print(f"[+] Camera Identification weights saved successfully to: {output_path}")
    print("[+] Model can now be used as the 2048-d frozen camera fingerprint extractor!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain ResNet101 for Camera Model Attribution")
    parser.add_argument("--data", type=str, default="camera_dataset", help="Path to camera dataset folder")
    parser.add_argument("--output", type=str, default="camera_resnet101.pth", help="Checkpoint output file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_camera_model_identifier(
        data_dir=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
