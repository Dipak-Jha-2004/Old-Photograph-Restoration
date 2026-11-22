"""
Black-and-White Photo Colorization Module for Old Photograph Restoration.

Integrates pretrained deep-learning colorization models:
- ECCV 2016 ("Colorful Image Colorization", Richard Zhang et al.)
- SIGGRAPH 2017 ("Real-Time User-Guided Image Colorization", Richard Zhang et al.)

Features:
- CPU inference with automatic device detection (CPU fallback)
- Preserves 100% of original photo lightness and detail by operating in CIELAB color space
  (predicts ab chrominance and fuses with native L channel)
- Model caching to avoid redundant weight deserialization
- Robust exception and memory handling
"""

import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import color

from utils.model_downloader import download_model, is_model_available


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "pretrained"

# Cache dictionary: (model_type, device_str) -> instantiated model
_COLORIZER_CACHE = {}


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------------------------------------------------------
# Neural Network Architectures (ECCV16 & SIGGRAPH17)
# -------------------------------------------------------------------------

class BaseColor(nn.Module):
    """Base class for Lab color space normalization and unnormalization."""
    def __init__(self):
        super(BaseColor, self).__init__()
        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0

    def normalize_l(self, in_l: torch.Tensor) -> torch.Tensor:
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l: torch.Tensor) -> torch.Tensor:
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        return in_ab * self.ab_norm


class ECCVGenerator(BaseColor):
    """ECCV 2016 Colorful Image Colorization feed-forward network."""
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        # Conv 1
        model1 = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        ]

        # Conv 2
        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]

        # Conv 3
        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]

        # Conv 4
        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Conv 5 (Dilated)
        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Conv 6 (Dilated)
        model6 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Conv 7
        model7 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Conv 8 (Upsampling)
        model8 = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, input_l: torch.Tensor) -> torch.Tensor:
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))


class SIGGRAPHGenerator(BaseColor):
    """SIGGRAPH 2017 Real-Time Colorization network with U-Net skip connections."""
    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529):
        super(SIGGRAPHGenerator, self).__init__()

        # Conv 1
        model1 = [
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        ]

        # Conv 2
        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]

        # Conv 3
        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]

        # Conv 4
        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Conv 5
        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Conv 6
        model6 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Conv 7
        model7 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Conv 8
        model8up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        model3short8 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]

        model8 = [
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]

        # Conv 9
        model9up = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        model2short9 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)]

        model9 = [
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]

        # Conv 10
        model10up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        model1short10 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]

        model10 = [
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
        ]

        # classification output
        model_class = [
            nn.Conv2d(256, classes, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
        ]

        model_out = [
            nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
            nn.Tanh(),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)
        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

    def forward(self, input_A: torch.Tensor, input_B: Optional[torch.Tensor] = None, mask_B: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_B is None:
            input_B = torch.cat((input_A * 0, input_A * 0), dim=1)
        if mask_B is None:
            mask_B = input_A * 0

        conv1_2 = self.model1(torch.cat((self.normalize_l(input_A), self.normalize_ab(input_B), mask_B), dim=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        return self.unnormalize_ab(out_reg)


# -------------------------------------------------------------------------
# Loader and Inference Engine
# -------------------------------------------------------------------------

def get_colorizer(model_type: str = "eccv16") -> nn.Module:
    """
    Retrieves or instantiates a cached colorizer model with pretrained weights.

    Args:
        model_type: 'eccv16' (Colorful Image Colorization) or 'siggraph17' (Real-time priors)

    Returns:
        Evaluated PyTorch model on the active execution device (CPU fallback).
    """
    global _COLORIZER_CACHE

    model_type = model_type.lower()
    if model_type not in ("eccv16", "siggraph17"):
        raise ValueError(f"Unknown colorization model type: '{model_type}'. Choose 'eccv16' or 'siggraph17'.")

    device = get_device()
    cache_key = (model_type, str(device))
    if cache_key in _COLORIZER_CACHE:
        return _COLORIZER_CACHE[cache_key]

    if model_type == "eccv16":
        model_key = "colorization_eccv16"
        filename = "colorization_release_v2-9b330a0b.pth"
        generator_cls = ECCVGenerator
    else:
        model_key = "colorization_siggraph17"
        filename = "siggraph17-df00044c.pth"
        generator_cls = SIGGRAPHGenerator

    model_path = MODELS_DIR / filename
    if not model_path.exists():
        if is_model_available(model_key):
            model_path = MODELS_DIR / filename
        else:
            print(f"[COLORIZATION] Model weights missing at {model_path}. Downloading...")
            model_path = download_model(model_key)

    print(f"[COLORIZATION] Loading {model_type.upper()} weights from {model_path.name} on {device}...")
    model = generator_cls()
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    _COLORIZER_CACHE[cache_key] = model
    return model


def colorize_image(
    image_rgb: np.ndarray,
    model_type: str = "eccv16",
    render_factor: int = 256,
    color_saturation: float = 1.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Colorizes a photograph using deep learning while strictly preserving native sharpness.

    Workflow:
    1. Converts image to CIELAB space and isolates the full-resolution L (Lightness) channel.
    2. Resizes L to render_factor (typically 256) and feeds it to the neural network.
    3. The network predicts ab chrominance distributions.
    4. Upscales predicted ab channels back to the original image dimensions using bilinear interpolation.
    5. Re-combines full-resolution original L channel with upscaled ab channels and converts back to RGB.

    Args:
        image_rgb: Input RGB uint8 NumPy array (H, W, 3).
        model_type: 'eccv16' (rich chromaticity) or 'siggraph17' (natural tones).
        render_factor: Resolution fed to colorizer network (default 256).
        color_saturation: Scale factor applied to ab chromaticity (1.0 = normal, 1.2 = vibrant).

    Returns:
        (colorized_rgb_image, metadata_dict)
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray):
        raise ValueError("Invalid image input: expected a valid NumPy array.")

    # Convert grayscale (2D) to 3D RGB representation if necessary
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 1:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    elif image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Invalid image channels: expected 3-channel RGB, got shape {image_rgb.shape}")

    t_start = time.perf_counter()
    device = get_device()
    h_orig, w_orig = image_rgb.shape[:2]

    try:
        model = get_colorizer(model_type=model_type)
    except Exception as e:
        return image_rgb.copy(), {
            "success": False,
            "error": f"Failed to load colorization model: {str(e)}",
            "model_type": model_type,
            "dimensions": (w_orig, h_orig),
            "execution_time_ms": 0.0,
            "device": str(device)
        }

    try:
        # Preprocessing in CIELAB space using skimage for high-precision floating point conversion
        img_rgb_float = image_rgb.astype(np.float32) / 255.0
        img_lab_orig = color.rgb2lab(img_rgb_float)
        img_l_orig = img_lab_orig[:, :, 0]  # Full-resolution lightness [0, 100]

        # Resized L channel for network input
        img_rgb_rs = cv2.resize(image_rgb, (render_factor, render_factor), interpolation=cv2.INTER_AREA)
        img_lab_rs = color.rgb2lab(img_rgb_rs.astype(np.float32) / 255.0)
        img_l_rs = img_lab_rs[:, :, 0]

        # Convert to PyTorch tensors (1, 1, H, W)
        tens_orig_l = torch.from_numpy(img_l_orig).float().unsqueeze(0).unsqueeze(0).to(device)
        tens_rs_l = torch.from_numpy(img_l_rs).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out_ab = model(tens_rs_l)

        # Scale saturation if customized
        if color_saturation != 1.0:
            out_ab = out_ab * float(color_saturation)

        # Upscale predicted ab channels back to original image dimensions via bilinear interpolation
        out_ab_orig = F.interpolate(out_ab, size=(h_orig, w_orig), mode="bilinear", align_corners=True)

        # Concatenate original full-res L with upscaled predicted ab channels
        out_lab = torch.cat((tens_orig_l, out_ab_orig), dim=1)
        out_lab_np = out_lab.squeeze(0).cpu().numpy().transpose((1, 2, 0))

        # Convert back from Lab to RGB space
        colorized_float = color.lab2rgb(out_lab_np)
        colorized_rgb = np.clip(colorized_float * 255.0, 0, 255).astype(np.uint8)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return image_rgb.copy(), {
                "success": False,
                "error": "Insufficient RAM/VRAM during colorization. Try reducing render factor.",
                "model_type": model_type,
                "dimensions": (w_orig, h_orig),
                "execution_time_ms": 0.0,
                "device": str(device)
            }
        else:
            return image_rgb.copy(), {
                "success": False,
                "error": f"Colorization inference error: {str(e)}",
                "model_type": model_type,
                "dimensions": (w_orig, h_orig),
                "execution_time_ms": 0.0,
                "device": str(device)
            }
    except Exception as e:
        return image_rgb.copy(), {
            "success": False,
            "error": f"Unexpected colorization error: {str(e)}",
            "model_type": model_type,
            "dimensions": (w_orig, h_orig),
            "execution_time_ms": 0.0,
            "device": str(device)
        }

    exec_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    metadata = {
        "success": True,
        "error": None,
        "model_type": model_type,
        "render_factor": render_factor,
        "color_saturation": color_saturation,
        "dimensions": (w_orig, h_orig),
        "execution_time_ms": exec_time_ms,
        "device": str(device)
    }

    return colorized_rgb, metadata
