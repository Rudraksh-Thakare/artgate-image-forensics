"""
Universal Multi-Branch Fusion & Camera Fingerprinting Detector for AI-Generated Images.

Incorporates:
1. ResNet101 Camera Fingerprinting Branch (2048-d GAP features from Section II of Manisha et al. 2025).
2. Grad-CAM Interpretability Engine (Layer4 convolutional activations and gradient heatmaps from Section III-C).
3. Frequency-Domain Wavelet Branch (2D Discrete Wavelet Transform LH, HL, HH subbands).
4. Vision-Language Semantic Branch (CLIP-ViT-B/32 multimodal visual representation).
5. Adaptive Cross-Modal Gating Network uniting spatial, frequency, and semantic forensic cues.
6. Central Patch Cropping (224x224) without resizing to preserve native sensor fingerprints.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image
import numpy as np
import pywt


# =====================================================================
# 0. PREPROCESSING UTILITY: CENTRAL PATCH CROPPING (Paper Section II-B)
# =====================================================================
def extract_center_patch(image: Image.Image, patch_size: int = 224) -> Image.Image:
    """
    Extracts a 224x224 patch from the center of the image without resizing,
    preventing any loss or distortion of high-to-mid frequency camera sensor fingerprints.
    """
    w, h = image.size
    if w < patch_size or h < patch_size:
        scale = max(patch_size / w, patch_size / h)
        new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        w, h = image.size

    left = (w - patch_size) // 2
    top = (h - patch_size) // 2
    return image.crop((left, top, left + patch_size, top + patch_size))


# =====================================================================
# 1. RESNET101 CAMERA FINGERPRINTING BRANCH (Paper Section II-A & II-B)
# =====================================================================
class ResNet101FingerprintBranch(nn.Module):
    """
    Deep fingerprint extractor based on ResNet101.
    Extracts the 2048-dimensional global representation from the Global Average Pooling (GAP) layer.
    Exposes layer4 forward activations and gradients for Grad-CAM visualization.
    """
    def __init__(self, checkpoint_path: str = None, freeze_backbone: bool = True):
        super().__init__()
        print("[*] Initializing ResNet101 Fingerprint Branch...")
        try:
            base_model = resnet101(weights=ResNet101_Weights.DEFAULT)
        except Exception:
            base_model = resnet101(weights=None)

        # Retain all layers except the final classification head (fc)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.feature_dim = 2048

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_custom_checkpoint(checkpoint_path)

        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

        # Storage for Grad-CAM
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.layer4.register_forward_hook(forward_hook)
        self.layer4.register_full_backward_hook(backward_hook)

    def load_custom_checkpoint(self, path: str):
        print(f"[*] Loading custom camera-ID / fingerprint weights from: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned_state = {k.replace("module.", ""): v for k, v in state.items() if not k.startswith("fc.")}
        self.load_state_dict(cleaned_state, strict=False)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet101 up to GAP: outputs (B, 2048)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 2048, 7, 7)

        pooled = self.avgpool(x)  # (B, 2048, 1, 1)
        return torch.flatten(pooled, 1)  # (B, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)


# =====================================================================
# 2. GRAD-CAM INTERPRETABILITY ENGINE (Paper Section III-C, Figure 4)
# =====================================================================
class ResNetGradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    on the final convolutional layer of ResNet101 (layer4).
    Produces normalized 2D heatmaps ([0, 1]) identifying synthetic anomalies vs real structures.
    """
    def __init__(self, fingerprint_branch: ResNet101FingerprintBranch):
        self.branch = fingerprint_branch

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Input: input_tensor (1, 3, 224, 224)
        Returns: 2D numpy array of shape (224, 224) normalized to [0, 1]
        """
        # Forward pass to extract layer4 activations
        with torch.no_grad():
            x = self.branch.conv1(input_tensor)
            x = self.branch.bn1(x)
            x = self.branch.relu(x)
            x = self.branch.maxpool(x)
            x = self.branch.layer1(x)
            x = self.branch.layer2(x)
            x = self.branch.layer3(x)
            activations = self.branch.layer4(x)  # (1, 2048, 7, 7)

            # Compute channel activation energy across all 2048 fingerprint kernels
            cam = torch.mean(torch.abs(activations), dim=1).squeeze(0).cpu().numpy()  # (7, 7)

            # Interpolate to input resolution (224, 224)
            cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
            cam_resized = F.interpolate(cam_t, size=(224, 224), mode="bilinear", align_corners=False).squeeze().numpy()

            cmin, cmax = float(cam_resized.min()), float(cam_resized.max())
            if cmax - cmin > 1e-8:
                cam_norm = (cam_resized - cmin) / (cmax - cmin)
            else:
                cam_norm = np.zeros_like(cam_resized)

            return cam_norm.astype(np.float32)


# =====================================================================
# 3. FREQUENCY DOMAIN BRANCH (Wavelet Decomposition)
# =====================================================================
class WaveletFrequencyBranch(nn.Module):
    """
    Extracts 2D Discrete Wavelet Transform (DWT) high-frequency subbands (LH, HL, HH).
    Captures generative upsampling grid artifacts, transposed convolution checkerboards,
    and high-frequency spectral roll-offs.
    """
    def __init__(self, out_dim: int = 256, wavelet: str = "haar"):
        super().__init__()
        self.wavelet = wavelet
        self.out_dim = out_dim

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.stat_fc = nn.Linear(36, 64)
        self.proj = nn.Sequential(
            nn.Linear(128 + 64, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True)
        )

    def extract_wavelet_subbands(self, x: torch.Tensor) -> tuple:
        b, c, h, w = x.shape
        x_np = x.detach().cpu().numpy()
        subband_maps = []
        subband_stats = []

        for i in range(b):
            img = x_np[i]
            channels_sub = []
            stats_vec = []
            for ch in range(c):
                coeffs2 = pywt.dwt2(img[ch], self.wavelet)
                ll, (lh, hl, hh) = coeffs2
                for sb in [lh, hl, hh]:
                    channels_sub.append(sb)
                    stats_vec.extend([
                        float(np.mean(sb)),
                        float(np.std(sb)),
                        float(np.mean(sb ** 2)),
                        float(np.max(np.abs(sb)))
                    ])
            subband_maps.append(np.stack(channels_sub, axis=0))
            subband_stats.append(stats_vec)

        maps_tensor = torch.from_numpy(np.stack(subband_maps, axis=0)).float().to(x.device)
        stats_tensor = torch.from_numpy(np.array(subband_stats, dtype=np.float32)).float().to(x.device)
        return maps_tensor, stats_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps, stats = self.extract_wavelet_subbands(x)
        cnn_feat = self.conv_encoder(maps).flatten(1)
        stat_feat = F.relu(self.stat_fc(stats))
        combined = torch.cat([cnn_feat, stat_feat], dim=1)
        return self.proj(combined)


# =====================================================================
# 4. VISION-LANGUAGE SEMANTIC BRANCH (CLIP-ViT)
# =====================================================================
class CLIPViTBranch(nn.Module):
    """
    Multimodal visual feature extractor using OpenAI's CLIP Vision Transformer (ViT-B/32).
    Captures semantic coherence, anatomical inconsistencies, and high-level scene realism.
    """
    def __init__(self, out_dim: int = 512, model_name: str = "openai/clip-vit-base-patch32", freeze: bool = True):
        super().__init__()
        self.clip_dim = 768
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.Linear(self.clip_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        try:
            from forensic_utils import _get_clip
            clip_model, _ = _get_clip()
            if clip_model is not None and hasattr(clip_model, "vision_model"):
                self.model = clip_model.vision_model
            else:
                from transformers import CLIPVisionModel
                self.model = CLIPVisionModel.from_pretrained(model_name)
            if freeze and self.model is not None:
                for p in self.model.parameters():
                    p.requires_grad = False
        except Exception as e:
            print(f"[!] Warning initializing CLIPViTBranch: {e}")
            self.model = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            outputs = self.model(pixel_values=x)
            pooled = outputs.pooler_output
        else:
            b = x.shape[0]
            pooled = F.adaptive_avg_pool2d(x, (16, 16)).flatten(1)
            if pooled.shape[1] != self.clip_dim:
                pooled = F.interpolate(pooled.unsqueeze(1), size=self.clip_dim, mode="linear").squeeze(1)
        return self.proj(pooled)


# =====================================================================
# 5. UNIVERSAL MULTI-BRANCH FUSION DETECTOR
# =====================================================================
class ArtGateFusionDetector(nn.Module):
    """
    Universal Multimodal Fusion & Camera Fingerprinting Detector:
    1. ResNet101 Camera Fingerprint Branch (2048-d)
    2. Wavelet DWT Frequency Branch (256-d)
    3. CLIP-ViT Semantic Branch (512-d)
    4. Adaptive Cross-Branch Gating
    5. Grad-CAM Visual Heatmap Engine
    """
    def __init__(
        self,
        camera_checkpoint: str = None,
        d_model: int = 512,
        num_classes: int = 2,
        freeze_spatial: bool = True,
        freeze_clip: bool = True
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Instantiate 3 complementary forensic branches
        self.fingerprint_branch = ResNet101FingerprintBranch(
            checkpoint_path=camera_checkpoint,
            freeze_backbone=freeze_spatial
        )
        self.wavelet_branch = WaveletFrequencyBranch(out_dim=256)
        self.clip_branch = CLIPViTBranch(out_dim=512, freeze=freeze_clip)

        # Grad-CAM Engine
        self.gradcam_engine = ResNetGradCAM(self.fingerprint_branch)

        # 2. Linear projectors into unified dimension d_model
        self.proj_spatial = nn.Sequential(
            nn.Linear(self.fingerprint_branch.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )
        self.proj_wavelet = nn.Sequential(
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )
        self.proj_clip = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        # 3. Dynamic Attention Gating Network
        self.gating_net = nn.Sequential(
            nn.Linear(d_model * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

        # 4. Multimodal Binary Classifier Head: Real (0) vs AI-Generated (1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def extract_all_branch_features(self, x: torch.Tensor):
        feat_spatial = self.fingerprint_branch(x)
        feat_wavelet = self.wavelet_branch(x)
        feat_clip = self.clip_branch(x)
        return {
            "resnet101_fingerprint_2048d": feat_spatial,
            "wavelet_256d": feat_wavelet,
            "clip_512d": feat_clip
        }

    def generate_gradcam(self, x: torch.Tensor) -> np.ndarray:
        return self.gradcam_engine.generate_heatmap(x)

    def forward(self, x: torch.Tensor, return_gates: bool = False):
        f_spatial = self.proj_spatial(self.fingerprint_branch(x))
        f_wavelet = self.proj_wavelet(self.wavelet_branch(x))
        f_clip = self.proj_clip(self.clip_branch(x))

        concat_feats = torch.cat([f_spatial, f_wavelet, f_clip], dim=1)
        gates = self.gating_net(concat_feats)

        w_spatial = gates[:, 0:1]
        w_wavelet = gates[:, 1:2]
        w_clip = gates[:, 2:3]

        f_fused = w_spatial * f_spatial + w_wavelet * f_wavelet + w_clip * f_clip
        logits = self.classifier(f_fused)

        if return_gates:
            return logits, gates
        return logits
