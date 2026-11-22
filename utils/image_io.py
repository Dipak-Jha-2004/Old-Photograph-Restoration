"""
Image I/O and Validation Utilities for Old Photograph Restoration.
"""

import io
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image_file(file_obj_or_path: Union[str, Path, io.BytesIO]) -> Tuple[bool, str]:
    """
    Validates whether an input file or byte stream is a valid, readable image.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        if isinstance(file_obj_or_path, (str, Path)):
            path = Path(file_obj_or_path)
            if not path.exists():
                return False, f"File does not exist: {path}"
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                return False, f"Unsupported file extension: {path.suffix}. Allowed: JPG, JPEG, PNG."
            with Image.open(path) as img:
                img.verify()
        else:
            # BytesIO or stream
            file_obj_or_path.seek(0)
            with Image.open(file_obj_or_path) as img:
                img.verify()
            file_obj_or_path.seek(0)
        return True, ""
    except Exception as e:
        return False, f"Invalid or corrupt image: {str(e)}"


def load_image(
    source: Union[str, Path, io.BytesIO, bytes, Image.Image],
    max_dimension: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Loads an image from various sources into a standard RGB uint8 NumPy array.
    Automatically handles orientation (EXIF) and converts RGBA/Grayscale to RGB.
    
    Args:
        source: File path, BytesIO, raw bytes, or PIL Image.
        max_dimension: Optional max width/height to resize while preserving aspect ratio.
        
    Returns:
        (image_rgb_numpy, metadata_dict)
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, io.BytesIO):
        source.seek(0)
        img = Image.open(source)
    elif isinstance(source, Image.Image):
        img = source.copy()
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")

    # Auto-orient based on EXIF tag
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    original_mode = img.mode
    original_size = img.size  # (width, height)

    # Convert to RGB cleanly
    if img.mode == "RGBA":
        # Create white background for alpha channel
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Downscale if max_dimension is provided and exceeded
    if max_dimension is not None:
        w, h = img.size
        if max(w, h) > max_dimension:
            scale = max_dimension / float(max(w, h))
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img_np = np.array(img, dtype=np.uint8)
    
    metadata = {
        "original_mode": original_mode,
        "original_size": original_size,
        "current_shape": img_np.shape,
        "channels": img_np.shape[2] if img_np.ndim == 3 else 1
    }
    
    return img_np, metadata


def save_image(
    image_rgb: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95
) -> Path:
    """
    Saves an RGB NumPy array to disk with high quality.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8))
    
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        pil_img.save(path, format="JPEG", quality=quality, subsampling=0)
    elif path.suffix.lower() == ".png":
        pil_img.save(path, format="PNG", compress_level=3)
    else:
        # Default to PNG if unspecified
        path = path.with_suffix(".png")
        pil_img.save(path, format="PNG")
        
    return path


def image_to_bytes(image_rgb: np.ndarray, format: str = "PNG", quality: int = 95) -> bytes:
    """
    Encodes an RGB NumPy array into image bytes for UI downloads.
    """
    pil_img = Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    fmt = format.upper()
    if fmt in ["JPG", "JPEG"]:
        pil_img.save(buffer, format="JPEG", quality=quality, subsampling=0)
    else:
        pil_img.save(buffer, format="PNG", compress_level=3)
    return buffer.getvalue()


def to_pil(image_rgb: np.ndarray) -> Image.Image:
    """Converts an RGB NumPy array to PIL Image."""
    return Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8))

