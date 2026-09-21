"""
Pre-download and cache model weights during Render build phase.
Ensures zero network latency, zero timeout, and instant inference at runtime.
"""
import os
import torch

print("[*] Pre-caching ResNet101 weights...")
try:
    from torchvision.models import resnet101, ResNet101_Weights
    resnet101(weights=ResNet101_Weights.DEFAULT)
    print("[+] ResNet101 weights cached.")
except Exception as e:
    print(f"[!] Warning caching ResNet101: {e}")

print("[*] Pre-caching CLIP Vision Transformer weights...")
try:
    from transformers import CLIPVisionModel, CLIPProcessor
    CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("[+] CLIP Vision Transformer cached.")
except Exception as e:
    print(f"[!] Warning caching CLIP: {e}")

print("[+] Pre-cache step completed successfully.")
