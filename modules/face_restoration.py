"""
GFPGAN Face Restoration Module for Old and Damaged Photographs.

Restores blurred, faded, low-resolution, and cracked faces using Tencent ARC Lab's
GFPGAN (Generative Facial Prior GAN) with full CPU fallback support.
"""

import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np
import torch

from utils.model_downloader import download_model, is_model_available


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "pretrained"
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "GFPGANv1.4.pth"

# Global model cache to avoid re-instantiating the heavy model on every call
_GFPGAN_RESTORER = None
_CACHED_PARAMS = None


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_face_restorer(
    model_path: Optional[Path] = None,
    upscale: int = 1,
    arch: str = "clean",
    channel_multiplier: int = 2
):
    """
    Loads and caches the official GFPGANer engine.
    
    Args:
        model_path: Path to GFPGANv1.4.pth. If None, checks default pretrained dir.
        upscale: Target overall image upscale factor (default 1 = preserve size).
        arch: GFPGAN architecture version ('clean' for v1.4).
        channel_multiplier: Channel multiplier for StyleGAN generator (2 for v1.4).
        
    Returns:
        Instantiated GFPGANer object.
    """
    global _GFPGAN_RESTORER, _CACHED_PARAMS

    if model_path is None:
        model_path = DEFAULT_WEIGHTS_PATH

    model_path = Path(model_path)
    if not model_path.exists():
        if is_model_available("gfpgan_v1.4"):
            model_path = DEFAULT_WEIGHTS_PATH
        else:
            # Attempt automated download
            print(f"[INFO] GFPGAN weights not found at {model_path}. Downloading...")
            model_path = download_model("gfpgan_v1.4")

    device = get_device()
    current_params = (str(model_path), upscale, arch, channel_multiplier, str(device))

    # Return cached instance if parameters match
    if _GFPGAN_RESTORER is not None and _CACHED_PARAMS == current_params:
        return _GFPGAN_RESTORER

    # Import GFPGANer
    try:
        from gfpgan import GFPGANer
    except ImportError as e:
        raise ImportError(
            f"GFPGAN package is not installed or failed to import. Error: {e}"
        )

    print(f"[GFPGAN] Initializing GFPGANer on device: {device} | Weights: {model_path.name}")
    restorer = GFPGANer(
        model_path=str(model_path),
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None,  # Dedicated to face restoration; background upsampler is handled by Real-ESRGAN in Stage 3
        device=device
    )

    _GFPGAN_RESTORER = restorer
    _CACHED_PARAMS = current_params
    return _GFPGAN_RESTORER


def restore_faces(
    image_rgb: np.ndarray,
    fidelity_weight: float = 0.5,
    upscale: int = 1,
    only_center_face: bool = False,
    model_path: Optional[Path] = None,
    face_blend_ratio: float = 0.80
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Restores degraded faces in an RGB photograph using GFPGAN.
    
    Args:
        image_rgb: Standard RGB uint8 image array (H, W, 3).
        fidelity_weight: Float between 0.0 and 1.0.
            - 0.0: Strict fidelity to original face (fewer hallucinations).
            - 0.5: Balanced fidelity and generative enhancement (recommended).
            - 1.0: Maximum generative restoration and high-frequency sharpness.
        upscale: Integer scaling factor (1 = keep original image dimensions).
        only_center_face: If True, only restores the most prominent center face.
        model_path: Optional custom path to GFPGAN weights.
        face_blend_ratio: Float between 0.0 and 1.0 (default 0.80). Blends the
            GFPGAN-enhanced output with the original photograph to preserve
            authentic family member likeness, original film grain, and avoid
            artificial or waxy AI-generated face appearances.
        
    Returns:
        (restored_rgb_image, metadata_dict)
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray):
        raise ValueError("Invalid image input: expected a valid NumPy array.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Invalid image channels: expected 3-channel RGB, got shape {image_rgb.shape}")

    t_start = time.perf_counter()

    # Get device and restorer engine
    device = get_device()
    try:
        restorer = get_face_restorer(model_path=model_path, upscale=upscale)
    except Exception as e:
        return image_rgb.copy(), {
            "success": False,
            "error": f"Failed to initialize GFPGAN model: {str(e)}",
            "faces_detected": 0,
            "cropped_faces": [],
            "restored_faces": [],
            "execution_time_ms": 0.0
        }

    # GFPGAN expects BGR uint8
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        # Run inference
        with torch.no_grad():
            cropped_faces, restored_faces, restored_bgr = restorer.enhance(
                img_bgr,
                has_aligned=False,
                only_center_face=only_center_face,
                paste_back=True,
                weight=float(fidelity_weight)
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Memory safeguard
            return image_rgb.copy(), {
                "success": False,
                "error": "Insufficient RAM/VRAM to process face restoration on this image size. Try reducing resolution.",
                "faces_detected": 0,
                "cropped_faces": [],
                "restored_faces": [],
                "execution_time_ms": 0.0
            }
        else:
            return image_rgb.copy(), {
                "success": False,
                "error": f"Inference runtime error: {str(e)}",
                "faces_detected": 0,
                "cropped_faces": [],
                "restored_faces": [],
                "execution_time_ms": 0.0
            }

    exec_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # Convert restored BGR back to standard RGB
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

    # Natural identity preservation: blend restored facial details with authentic input texture
    if 0.0 < face_blend_ratio < 1.0 and restored_rgb.shape == image_rgb.shape and len(restored_faces) > 0:
        alpha = float(face_blend_ratio)
        restored_rgb = np.clip(
            alpha * restored_rgb.astype(np.float32) + (1.0 - alpha) * image_rgb.astype(np.float32),
            0,
            255
        ).astype(np.uint8)

    # Convert cropped face crops to RGB for UI inspection
    cropped_faces_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in cropped_faces] if cropped_faces else []
    restored_faces_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in restored_faces] if restored_faces else []

    metadata = {
        "success": True,
        "error": None,
        "faces_detected": len(restored_faces),
        "fidelity_weight": fidelity_weight,
        "face_blend_ratio": face_blend_ratio,
        "device": str(device),
        "execution_time_ms": exec_time_ms,
        "cropped_faces": cropped_faces_rgb,
        "restored_faces": restored_faces_rgb
    }

    return restored_rgb, metadata

