"""
Camera Fingerprint SVM Evaluation Pipeline (Manisha et al. IEEE Access 2025).

Implements Section II-B, II-C, and III:
1. Extracts 224x224 central patches without resizing to preserve raw sensor fingerprints.
2. Extracts 2048-dimensional Global Average Pooling (GAP) feature vectors via ResNet101.
3. Fits a Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel.
4. Evaluates binary accuracy, ROC AUC, and classification report.
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from fusion_detector import ResNet101FingerprintBranch, extract_center_patch
from torchvision import transforms


def extract_dataset_fingerprints(dataset_dir: str, feature_extractor, device: torch.device):
    """
    Expects dataset_dir to contain:
      dataset_dir/
        ├── real/  (Authentic physical camera photos)
        └── fake/  (AI-synthesized images: Midjourney, Stable Diffusion, DALL-E, etc.)
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    features = []
    labels = []

    classes = {"real": 0, "fake": 1}
    for class_name, label in classes.items():
        folder = os.path.join(dataset_dir, class_name)
        if not os.path.exists(folder):
            continue

        file_list = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        print(f"[*] Extracting fingerprints for {len(file_list)} '{class_name}' images...")

        for fname in file_list:
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert("RGB")
                # Section II-B: Center Patch Cropping (224x224) WITHOUT resizing
                patch = extract_center_patch(img, patch_size=224)
                tensor = preprocess(patch).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = feature_extractor(tensor).squeeze(0).cpu().numpy()
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"  [!] Skipped {fname}: {e}")

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def run_svm_evaluation(data_dir: str, checkpoint_path: str = None, test_size: float = 0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Camera Fingerprint RBF-SVM Classification (Manisha et al. 2025)")
    print("=" * 70)

    # Initialize frozen ResNet101 feature extractor
    extractor = ResNet101FingerprintBranch(checkpoint_path=checkpoint_path, freeze_backbone=True)
    extractor.to(device)
    extractor.eval()

    X, y = extract_dataset_fingerprints(data_dir, extractor, device)
    if len(X) == 0:
        print("[!] No images found. Ensure directory has 'real' and 'fake' subfolders.")
        return

    print(f"\n[+] Total samples extracted: {len(X)} (2048 dimensions each)")
    print(f"    - Real camera images: {(y == 0).sum()}")
    print(f"    - AI-synthesized images: {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Section II-C: SVM with Radial Basis Function (RBF) kernel
    print("\n[*] Training RBF-kernel SVM classifier...")
    svm = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42)
    svm.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm.predict(X_test)
    y_probs = svm.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "-" * 70)
    print("                  EVALUATION METRICS")
    print("-" * 70)
    print(f"  * Classification Accuracy: {acc * 100:.2f}%")
    print(f"  * ROC AUC Score:           {auc:.4f}")
    print(f"  * Confusion Matrix:\n{cm}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real Camera", "AI-Synthesized"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Camera Fingerprint RBF-SVM on Real vs AI images")
    parser.add_argument("--data", type=str, default="benchmark_data", help="Directory with real/ and fake/ subfolders")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom camera_resnet101.pth (optional)")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split ratio (default 25%)")
    args = parser.parse_args()

    run_svm_evaluation(data_dir=args.data, checkpoint_path=args.checkpoint, test_size=args.test_size)
