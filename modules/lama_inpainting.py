"""
Pretrained LaMa (Large Mask Inpainting) Module for Old Photographs.

Uses Samsung AI's resolution-robust LaMa with Fast Fourier Convolutions (FFCs)
to seamlessly repair large scratches, cracks, tears, and stains.
Includes automatic CPU fallback to OpenCV Inpainting (Telea / Navier-Stokes)
and guarantees 100% bit-exact preservation of all undamaged pixels.
"""

import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "pretrained"
LAMA_WEIGHTS_PATH = MODELS_DIR / "big-lama.pt"

# Global model cache to avoid re-instantiating on every execution
_LAMA_MODEL = None
_CACHED_DEVICE = None


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_lama_available() -> bool:
    """Checks whether the pretrained big-lama.pt model exists."""
    return LAMA_WEIGHTS_PATH.exists() and LAMA_WEIGHTS_PATH.stat().st_size > 50 * 1024 * 1024


def load_lama_model(device: Optional[torch.device] = None):
    """Loads and caches the pretrained LaMa TorchScript model."""
    global _LAMA_MODEL, _CACHED_DEVICE

    if device is None:
        device = get_device()

    if _LAMA_MODEL is not None and _CACHED_DEVICE == device:
        return _LAMA_MODEL

    if not is_lama_available():
        raise FileNotFoundError(
            f"Pretrained LaMa weights not found at {LAMA_WEIGHTS_PATH}. "
            f"Please ensure big-lama.pt (~196 MB) is placed in models/pretrained/."
        )

    print(f"[LaMa] Loading TorchScript model from {LAMA_WEIGHTS_PATH.name} on {device}...")
    os.environ["LAMA_MODEL"] = str(LAMA_WEIGHTS_PATH)
    
    from simple_lama_inpainting import SimpleLama
    model = SimpleLama(device=device)

    _LAMA_MODEL = model
    _CACHED_DEVICE = device
    return _LAMA_MODEL


def inpaint_with_lama(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Inpaints damaged regions using the pretrained LaMa deep neural network.
    
    Args:
        image_rgb: Source uint8 RGB image array (H, W, 3).
        mask: Binary damage mask (H, W) or (H, W, 1) where > 0 indicates damaged pixels.
        device: PyTorch compute device (CPU or CUDA).
        
    Returns:
        (inpainted_rgb, metadata_dict)
    """
    if image_rgb is None or mask is None:
        raise ValueError("Image and mask must not be None.")

    orig_h, orig_w = image_rgb.shape[:2]

    # Ensure mask is 2D uint8
    if mask.ndim == 3:
        mask_2d = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) if mask.shape[2] == 3 else mask[:, :, 0]
    else:
        mask_2d = mask.copy()

    # Binarize mask
    binary_mask = (mask_2d > 0).astype(np.uint8) * 255

    # If mask is empty, return original immediately
    if np.count_nonzero(binary_mask) == 0:
        return image_rgb.copy(), {
            "success": True,
            "method": "lama",
            "execution_time_ms": 0.0,
            "pixels_repaired": 0,
            "fallback_used": False
        }

    t0 = time.perf_counter()

    try:
        lama = load_lama_model(device=device)
        pil_img = Image.fromarray(image_rgb)
        pil_mask = Image.fromarray(binary_mask)

        # Run LaMa inference
        raw_result = lama(pil_img, pil_mask)
        raw_np = np.array(raw_result)

        # Unpad / crop back to exact original dimensions
        inpainted_cropped = raw_np[:orig_h, :orig_w]

        # GUARANTEE 100% PRESERVATION OF UNDAMAGED PIXELS WITH FEATHERED BOUNDARY:
        # Feather mask boundary with 1px Gaussian blur for seamless blending at patch edges
        mask_f = (binary_mask > 0).astype(np.float32)
        mask_feathered = cv2.GaussianBlur(mask_f, (5, 5), 1.0)[:, :, None]
        final_rgb = np.clip(
            mask_feathered * inpainted_cropped.astype(np.float32) + (1.0 - mask_feathered) * image_rgb.astype(np.float32),
            0,
            255
        ).astype(np.uint8)
        # Guarantee 100% bit-exact original pixels wherever mask is completely zero
        final_rgb[binary_mask == 0] = image_rgb[binary_mask == 0]

        exec_time = round((time.perf_counter() - t0) * 1000, 2)

        return final_rgb, {
            "success": True,
            "method": "lama",
            "execution_time_ms": exec_time,
            "pixels_repaired": int(np.count_nonzero(binary_mask)),
            "fallback_used": False
        }

    except Exception as e:
        print(f"[WARNING] LaMa inpainting encountered an issue: {e}. Falling back to OpenCV Telea...")
        # Fallback to OpenCV Telea
        t_fb = time.perf_counter()
        inpainted_cv = cv2.inpaint(image_rgb, binary_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        final_rgb = np.where(binary_mask[:, :, None] > 0, inpainted_cv, image_rgb)
        exec_time = round((time.perf_counter() - t_fb) * 1000, 2)

        return final_rgb, {
            "success": True,
            "method": "opencv_telea_fallback",
            "execution_time_ms": exec_time,
            "pixels_repaired": int(np.count_nonzero(binary_mask)),
            "fallback_used": True,
            "error": str(e)
        }


def inpaint_photo(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    method: str = "lama",
    inpaint_radius: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Unified Inpainting Orchestrator.
    Supports:
    - 'lama': Pretrained Large Mask Inpainting (recommended for vintage photos)
    - 'telea': OpenCV Fast Marching (fast, lightweight)
    - 'ns': OpenCV Navier-Stokes (fluid dynamics)
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return image_rgb.copy(), {
            "success": True,
            "method": method,
            "execution_time_ms": 0.0,
            "pixels_repaired": 0,
            "fallback_used": False
        }

    # Normalize mask
    if mask.ndim == 3:
        mask_2d = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) if mask.shape[2] == 3 else mask[:, :, 0]
    else:
        mask_2d = mask.copy()
    binary_mask = (mask_2d > 0).astype(np.uint8) * 255

    method_lower = method.lower()
    if method_lower == "lama" and is_lama_available():
        return inpaint_with_lama(image_rgb, binary_mask, device=device)
    else:
        # OpenCV inpainting
        t0 = time.perf_counter()
        flag = cv2.INPAINT_TELEA if method_lower in ["telea", "lama"] else cv2.INPAINT_NS
        inpainted = cv2.inpaint(image_rgb, binary_mask, inpaintRadius=max(1, inpaint_radius), flags=flag)
        # Preserve undamaged pixels strictly
        final_rgb = np.where(binary_mask[:, :, None] > 0, inpainted, image_rgb)
        exec_time = round((time.perf_counter() - t0) * 1000, 2)

        return final_rgb, {
            "success": True,
            "method": f"opencv_{'telea' if flag == cv2.INPAINT_TELEA else 'ns'}",
            "execution_time_ms": exec_time,
            "pixels_repaired": int(np.count_nonzero(binary_mask)),
            "fallback_used": (method_lower == "lama" and not is_lama_available())
        }
