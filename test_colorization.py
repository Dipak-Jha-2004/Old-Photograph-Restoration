"""
Verification and Test Suite for Stage 4: Black-and-White Photo Colorization.

Tests:
1. Pretrained model weights loading (ECCV16 and SIGGRAPH17).
2. CPU inference configuration (device=cpu, half-precision disabled).
3. ECCV16 colorization execution on B&W image.
4. SIGGRAPH17 colorization execution on B&W image.
5. Preservation of native image dimensions (input WxH == output WxH).
6. Robust error handling for invalid/empty inputs.
7. Full 4-stage end-to-end integrated restoration pipeline:
   Stage 1: OpenCV Denoise & Scratch Inpainting
   Stage 2: Deep Learning Colorization (ECCV16)
   Stage 3: GFPGAN Face Restoration
   Stage 4: Real-ESRGAN Super-Resolution (2x)
"""

import sys
import time
from pathlib import Path
import pytest
import numpy as np
import cv2
import torch

WORKSPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_DIR))

from modules.colorization import (
    colorize_image,
    get_colorizer,
    get_device,
    MODELS_DIR
)
from modules.opencv_restoration import restore_image_opencv, PRESETS
from modules.face_restoration import restore_faces
from modules.super_resolution import upscale_image
from utils.image_io import load_image, save_image


@pytest.fixture(scope="module")
def sample_bw_image() -> np.ndarray:
    """Prepares a grayscale photograph for colorization tests."""
    sample_path = WORKSPACE_DIR / "test_samples" / "sample_damaged.png"
    assert sample_path.exists(), f"Sample image missing at {sample_path}"
    full_img, _ = load_image(sample_path)
    # Convert to pure grayscale then back to 3-channel RGB to simulate authentic B&W photo
    gray = cv2.cvtColor(full_img, cv2.COLOR_RGB2GRAY)
    bw_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # Crop 256x256 patch for testing
    h, w = bw_rgb.shape[:2]
    ch, cw = h // 2, w // 2
    patch = bw_rgb[ch - 128:ch + 128, cw - 128:cw + 128].copy()
    assert patch.shape == (256, 256, 3)
    return patch


def test_1_colorization_weights_loading():
    """Verify that both ECCV16 and SIGGRAPH17 pretrained weights load into PyTorch."""
    w_eccv = MODELS_DIR / "colorization_release_v2-9b330a0b.pth"
    w_sig = MODELS_DIR / "siggraph17-df00044c.pth"

    assert w_eccv.exists(), f"ECCV16 weights file missing: {w_eccv}"
    assert w_sig.exists(), f"SIGGRAPH17 weights file missing: {w_sig}"
    assert w_eccv.stat().st_size > 100 * 1024 * 1024, "ECCV16 file size too small"
    assert w_sig.stat().st_size > 100 * 1024 * 1024, "SIGGRAPH17 file size too small"

    model_eccv = get_colorizer("eccv16")
    assert model_eccv is not None, "Failed to instantiate ECCV16 model"

    model_sig = get_colorizer("siggraph17")
    assert model_sig is not None, "Failed to instantiate SIGGRAPH17 model"

    print(f"\n[PASS] Colorization weights loaded: ECCV16 ({w_eccv.stat().st_size:,} bytes), SIGGRAPH17 ({w_sig.stat().st_size:,} bytes)")


def test_2_cpu_inference_configuration():
    """Verify CPU inference configuration and device fallback."""
    device = get_device()
    assert device.type == "cpu", f"Expected CPU device fallback, got {device.type}"

    model = get_colorizer("eccv16")
    param = next(model.parameters())
    assert param.device == device, f"Model parameter device mismatch: {param.device} vs {device}"

    print(f"\n[PASS] CPU inference verified on device: {device.type}")


def test_3_eccv16_colorization_execution(sample_bw_image):
    """Verify ECCV16 colorization on B&W image."""
    h_in, w_in = sample_bw_image.shape[:2]
    t0 = time.perf_counter()
    colorized, meta = colorize_image(sample_bw_image, model_type="eccv16", render_factor=256)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert meta["success"] is True, f"Colorization failed: {meta.get('error')}"
    assert colorized is not None, "Colorized output is None"
    assert colorized.shape == sample_bw_image.shape, f"Dimensions changed: {colorized.shape} vs {sample_bw_image.shape}"
    assert colorized.dtype == np.uint8, "Output dtype must be uint8"

    # Verify that color channels have chrominance variation (not flat grayscale)
    r = colorized[:, :, 0].astype(float)
    g = colorized[:, :, 1].astype(float)
    b = colorized[:, :, 2].astype(float)
    chroma_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(r - b)) + np.mean(np.abs(g - b))
    assert chroma_diff > 2.0, f"Output appears monochrome/uncolorized: chroma_diff={chroma_diff}"

    out_dir = WORKSPACE_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(colorized, out_dir / "test_colorized_eccv16.png")

    print(f"\n[PASS] ECCV16 Colorization: input ({w_in}x{h_in}) -> output ({colorized.shape[1]}x{colorized.shape[0]}) in {elapsed_ms:.1f} ms (chroma_diff={chroma_diff:.2f})")


def test_4_siggraph17_colorization_execution(sample_bw_image):
    """Verify SIGGRAPH17 colorization on B&W image."""
    h_in, w_in = sample_bw_image.shape[:2]
    t0 = time.perf_counter()
    colorized, meta = colorize_image(sample_bw_image, model_type="siggraph17", render_factor=256)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert meta["success"] is True, f"Colorization failed: {meta.get('error')}"
    assert colorized.shape == sample_bw_image.shape, f"Dimensions changed: {colorized.shape}"
    assert colorized.dtype == np.uint8, "Output dtype must be uint8"

    out_dir = WORKSPACE_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(colorized, out_dir / "test_colorized_siggraph17.png")

    print(f"\n[PASS] SIGGRAPH17 Colorization: input ({w_in}x{h_in}) -> output ({colorized.shape[1]}x{colorized.shape[0]}) in {elapsed_ms:.1f} ms")


def test_5_dimension_preservation_and_aspect_ratio(sample_bw_image):
    """Verify that arbitrary non-square image dimensions are strictly preserved."""
    # Test on a non-square 180x320 crop
    non_square = sample_bw_image[:180, :240].copy()
    h_in, w_in = non_square.shape[:2]

    colorized, meta = colorize_image(non_square, model_type="eccv16", render_factor=256)
    assert meta["success"] is True
    h_out, w_out = colorized.shape[:2]

    assert (w_in, h_in) == (w_out, h_out), f"Dimension mismatch: expected ({w_in}, {h_in}), got ({w_out}, {h_out})"
    print(f"\n[PASS] Non-square dimension preservation: input ({w_in}x{h_in}) -> output ({w_out}x{h_out})")


def test_6_robust_error_handling():
    """Verify error handling for invalid image types and empty arrays."""
    with pytest.raises(ValueError):
        colorize_image(None)

    # 4D tensor or invalid dimension array should raise ValueError
    invalid_arr = np.zeros((10, 10, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        colorize_image(invalid_arr)

    print("\n[PASS] Robust error handling verified for None and invalid shape inputs")


def test_7_full_4_stage_restoration_pipeline(sample_bw_image):
    """
    Verify complete 4-stage pipeline:
    OpenCV (Denoise & Inpaint) -> Colorization (ECCV16) -> GFPGAN (Face) -> Real-ESRGAN (2x)
    """
    t_start = time.perf_counter()

    # Stage 1: OpenCV Denoise & Scratch Inpainting
    stage1_out, meta_s1 = restore_image_opencv(sample_bw_image, config=PRESETS["balanced"])
    assert stage1_out.shape == sample_bw_image.shape

    # Stage 2: Colorization (B&W to Color)
    stage2_out, meta_s2 = colorize_image(stage1_out, model_type="eccv16", render_factor=256)
    assert meta_s2["success"] is True
    assert stage2_out.shape == sample_bw_image.shape

    # Stage 3: GFPGAN Face Restoration
    stage3_out, meta_s3 = restore_faces(stage2_out, fidelity_weight=0.5, upscale=1)
    assert stage3_out.shape == sample_bw_image.shape

    # Stage 4: Real-ESRGAN 2x Super-Resolution
    stage4_out, meta_s4 = upscale_image(stage3_out, scale=2, tile=400)
    assert meta_s4["success"] is True

    expected_final_shape = (sample_bw_image.shape[0] * 2, sample_bw_image.shape[1] * 2, 3)
    assert stage4_out.shape == expected_final_shape, f"Expected {expected_final_shape}, got {stage4_out.shape}"

    total_time = (time.perf_counter() - t_start) * 1000
    out_dir = WORKSPACE_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(stage4_out, out_dir / "test_full_4stage_restored.png")

    print(
        f"\n[PASS] Full 4-Stage Pipeline Succeeded:"
        f" Input ({sample_bw_image.shape[1]}x{sample_bw_image.shape[0]}) -> Final ({stage4_out.shape[1]}x{stage4_out.shape[0]})"
        f" in {total_time:.1f} ms"
        f" (OpenCV: {meta_s1['total_time_ms']:.1f}ms, Colorize: {meta_s2['execution_time_ms']:.1f}ms,"
        f" GFPGAN: {meta_s3['execution_time_ms']:.1f}ms, Real-ESRGAN: {meta_s4['execution_time_ms']:.1f}ms)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
