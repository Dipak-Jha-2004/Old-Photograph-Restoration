"""
Real-ESRGAN Super-Resolution Module for Old Photograph Enhancement.

Provides high-fidelity 2x and 4x neural upscaling with:
- Tiled inference (default tile=400) to prevent CPU memory spikes and Out-Of-Memory errors
- CPU fallback configuration (device=cpu, half=False)
- Model caching to avoid re-instantiation overhead
- Support for official RealESRGAN_x4plus and RealESRGAN_x2plus checkpoints
"""

import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np
import torch

from utils.model_downloader import download_model, is_model_available


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "pretrained"

# Cache for loaded upscaler instances: (scale, tile, device_str) -> upscaler
_UPSCALER_CACHE = {}


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_real_esrgan_upscaler(
    scale: int = 4,
    tile: int = 400,
    tile_pad: int = 10,
    pre_pad: int = 0
):
    """
    Initializes or retrieves a cached RealESRGANer instance.
    
    Args:
        scale: Upscaling factor, either 2 or 4.
        tile: Tile size for cropped tiled inference (400 recommended for CPU memory safety).
        tile_pad: Padding around each tile to avoid seam artifacts.
        pre_pad: Pre-padding size at borders.
        
    Returns:
        Instantiated RealESRGANer object.
    """
    global _UPSCALER_CACHE

    if scale not in (2, 4):
        raise ValueError(f"Unsupported scale factor: {scale}. Real-ESRGAN supports 2x and 4x upscaling.")

    model_key = "realesrgan_x4plus" if scale == 4 else "realesrgan_x2plus"
    model_filename = "RealESRGAN_x4plus.pth" if scale == 4 else "RealESRGAN_x2plus.pth"
    model_path = MODELS_DIR / model_filename

    # Ensure weights exist locally
    if not model_path.exists():
        if is_model_available(model_key):
            model_path = MODELS_DIR / model_filename
        else:
            print(f"[INFO] Real-ESRGAN weights not found at {model_path}. Downloading...")
            model_path = download_model(model_key)

    device = get_device()
    # Half precision (FP16) is only supported on CUDA devices; must be False on CPU
    use_half = (device.type == "cuda")

    cache_key = (scale, tile, str(device), use_half)
    if cache_key in _UPSCALER_CACHE:
        return _UPSCALER_CACHE[cache_key]

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError as e:
        raise ImportError(
            f"Failed to import RealESRGANer or RRDBNet. Please check installed packages. Error: {e}"
        )

    # RRDBNet architecture configuration
    # num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )

    print(f"[REAL-ESRGAN] Initializing {scale}x upscaler on {device} (tile={tile}, half={use_half})...")
    upscaler = RealESRGANer(
        scale=scale,
        model_path=str(model_path),
        model=model,
        tile=int(tile),
        tile_pad=int(tile_pad),
        pre_pad=int(pre_pad),
        half=use_half,
        device=device
    )

    _UPSCALER_CACHE[cache_key] = upscaler
    return upscaler


def upscale_image(
    image_rgb: np.ndarray,
    scale: int = 4,
    tile: int = 400
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Upscales an RGB photograph using Real-ESRGAN.
    
    Args:
        image_rgb: Standard RGB uint8 NumPy array (H, W, 3).
        scale: Upscale factor, either 2 or 4.
        tile: Tile size for memory-safe tiled inference (default 400).
        
    Returns:
        (upscaled_rgb_image, metadata_dict)
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray):
        raise ValueError("Invalid image input: expected a valid NumPy array.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Invalid image channels: expected 3-channel RGB, got shape {image_rgb.shape}")

    t_start = time.perf_counter()
    device = get_device()
    h_in, w_in = image_rgb.shape[:2]

    try:
        upscaler = get_real_esrgan_upscaler(scale=scale, tile=tile)
    except Exception as e:
        return image_rgb.copy(), {
            "success": False,
            "error": f"Failed to initialize Real-ESRGAN model: {str(e)}",
            "scale": 1,
            "input_shape": (w_in, h_in),
            "output_shape": (w_in, h_in),
            "execution_time_ms": 0.0,
            "device": str(device)
        }

    # Real-ESRGAN expects BGR uint8 format
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        with torch.no_grad():
            output_bgr, _ = upscaler.enhance(img_bgr, outscale=scale)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Memory safeguard: retry with smaller tile size if possible
            if tile > 200:
                print(f"[WARN] Memory limit reached with tile={tile}. Retrying with tile=200...")
                upscaler = get_real_esrgan_upscaler(scale=scale, tile=200)
                with torch.no_grad():
                    output_bgr, _ = upscaler.enhance(img_bgr, outscale=scale)
            else:
                return image_rgb.copy(), {
                    "success": False,
                    "error": "Insufficient RAM/VRAM for super-resolution. Please try a smaller image or 2x upscale.",
                    "scale": 1,
                    "input_shape": (w_in, h_in),
                    "output_shape": (w_in, h_in),
                    "execution_time_ms": 0.0,
                    "device": str(device)
                }
        else:
            return image_rgb.copy(), {
                "success": False,
                "error": f"Real-ESRGAN inference error: {str(e)}",
                "scale": 1,
                "input_shape": (w_in, h_in),
                "output_shape": (w_in, h_in),
                "execution_time_ms": 0.0,
                "device": str(device)
            }

    exec_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    h_out, w_out = output_rgb.shape[:2]

    metadata = {
        "success": True,
        "error": None,
        "scale": scale,
        "tile": tile,
        "device": str(device),
        "input_shape": (w_in, h_in),
        "output_shape": (w_out, h_out),
        "execution_time_ms": exec_time_ms
    }

    return output_rgb, metadata
