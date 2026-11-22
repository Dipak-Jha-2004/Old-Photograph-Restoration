"""
Verification and Test Suite for Stage 3: Real-ESRGAN Super-Resolution.

Tests:
1. Model weights load successfully (RealESRGAN_x2plus.pth and RealESRGAN_x4plus.pth).
2. CPU inference works (device=cpu, half=False, no CUDA dependency).
3. 2x upscaling works (high fidelity super-resolution).
4. 4x upscaling works (ultra resolution super-resolution).
5. Output image dimensions are correct (2x -> 2*W, 2*H; 4x -> 4*W, 4*H).
6. Tiled inference works without memory errors (tile=400 memory safeguard).
7. Complete 3-stage restoration pipeline works (OpenCV + GFPGAN + Real-ESRGAN).
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

from modules.super_resolution import (
    upscale_image,
    get_device,
    get_real_esrgan_upscaler,
    MODELS_DIR
)
from modules.face_restoration import restore_faces
from modules.opencv_restoration import restore_image_opencv, PRESETS
from utils.image_io import load_image, save_image


@pytest.fixture(scope="module")
def test_patch() -> np.ndarray:
    """Prepares a deterministic 128x128 RGB test patch from damaged sample."""
    sample_path = WORKSPACE_DIR / "test_samples" / "sample_damaged.png"
    assert sample_path.exists(), f"Sample image missing at {sample_path}"
    full_img, _ = load_image(sample_path)
    h, w = full_img.shape[:2]
    ch, cw = h // 2, w // 2
    # 128x128 crop gives fast CPU turnaround while testing real photographic textures
    patch = full_img[ch - 64:ch + 64, cw - 64:cw + 64].copy()
    assert patch.shape == (128, 128, 3)
    return patch


def test_1_model_weights_loading():
    """Verify that both 2x and 4x Real-ESRGAN pretrained weights load successfully."""
    w2 = MODELS_DIR / "RealESRGAN_x2plus.pth"
    w4 = MODELS_DIR / "RealESRGAN_x4plus.pth"
    assert w2.exists(), f"Weights file missing: {w2}"
    assert w4.exists(), f"Weights file missing: {w4}"
    assert w2.stat().st_size > 50 * 1024 * 1024, f"File size too small: {w2.stat().st_size} bytes"
    assert w4.stat().st_size > 50 * 1024 * 1024, f"File size too small: {w4.stat().st_size} bytes"

    # Instantiate upscalers to verify weights can be deserialized and loaded into RRDBNet
    upscaler_2x = get_real_esrgan_upscaler(scale=2, tile=400)
    assert upscaler_2x is not None, "Failed to instantiate 2x upscaler"

    upscaler_4x = get_real_esrgan_upscaler(scale=4, tile=400)
    assert upscaler_4x is not None, "Failed to instantiate 4x upscaler"

    print(f"\n[PASS] Model weights loaded: 2x ({w2.stat().st_size:,} bytes), 4x ({w4.stat().st_size:,} bytes)")


def test_2_cpu_inference_configuration():
    """Verify CPU inference configuration: device is CPU and half-precision is disabled."""
    device = get_device()
    assert device.type == "cpu", f"Expected CPU device fallback, got {device.type}"

    upscaler = get_real_esrgan_upscaler(scale=2, tile=400)
    assert upscaler.device == device, f"Upscaler device mismatch: {upscaler.device} vs {device}"
    assert upscaler.half is False, f"half-precision must be False on CPU, got {upscaler.half}"

    print(f"\n[PASS] CPU inference configured: device={device.type}, half={upscaler.half}")


def test_3_2x_upscaling_execution(test_patch):
    """Verify 2x super-resolution upscaling execution."""
    h_in, w_in = test_patch.shape[:2]
    t0 = time.perf_counter()
    upscaled, meta = upscale_image(test_patch, scale=2, tile=400)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert meta["success"] is True, f"2x Upscale failed: {meta.get('error')}"
    assert upscaled is not None, "Upscaled image is None"
    assert isinstance(upscaled, np.ndarray), "Output must be a numpy ndarray"
    assert upscaled.dtype == np.uint8, "Output dtype must be uint8"

    h_out, w_out = upscaled.shape[:2]
    print(f"\n[PASS] 2x Upscaling: input ({w_in}x{h_in}) -> output ({w_out}x{h_out}) in {elapsed_ms:.1f} ms")


def test_4_4x_upscaling_execution(test_patch):
    """Verify 4x super-resolution upscaling execution."""
    h_in, w_in = test_patch.shape[:2]
    t0 = time.perf_counter()
    upscaled, meta = upscale_image(test_patch, scale=4, tile=400)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert meta["success"] is True, f"4x Upscale failed: {meta.get('error')}"
    assert upscaled is not None, "Upscaled image is None"
    assert isinstance(upscaled, np.ndarray), "Output must be a numpy ndarray"
    assert upscaled.dtype == np.uint8, "Output dtype must be uint8"

    h_out, w_out = upscaled.shape[:2]
    print(f"\n[PASS] 4x Upscaling: input ({w_in}x{h_in}) -> output ({w_out}x{h_out}) in {elapsed_ms:.1f} ms")


def test_5_output_image_dimensions(test_patch):
    """Verify that output image dimensions strictly equal (scale * W, scale * H, 3)."""
    h_in, w_in = test_patch.shape[:2]

    # Verify 2x
    out_2x, _ = upscale_image(test_patch, scale=2, tile=400)
    expected_2x = (h_in * 2, w_in * 2, 3)
    assert out_2x.shape == expected_2x, f"2x shape mismatch: expected {expected_2x}, got {out_2x.shape}"

    # Verify 4x
    out_4x, _ = upscale_image(test_patch, scale=4, tile=400)
    expected_4x = (h_in * 4, w_in * 4, 3)
    assert out_4x.shape == expected_4x, f"4x shape mismatch: expected {expected_4x}, got {out_4x.shape}"

    print(
        f"\n[PASS] Dimensions verified:"
        f" Input ({w_in}x{h_in}) | 2x -> ({out_2x.shape[1]}x{out_2x.shape[0]}) | 4x -> ({out_4x.shape[1]}x{out_4x.shape[0]})"
    )


def test_6_tiled_inference_memory_safe(test_patch):
    """Verify tiled inference executes cleanly with tile=400 without memory exhaustion."""
    out, meta = upscale_image(test_patch, scale=2, tile=400)
    assert meta["success"] is True, f"Tiled inference failed: {meta.get('error')}"
    assert meta["tile"] == 400, f"Expected tile=400, got {meta.get('tile')}"
    assert out.shape == (test_patch.shape[0] * 2, test_patch.shape[1] * 2, 3)

    print(f"\n[PASS] Tiled inference (tile=400): Output dimensions ({out.shape[1]}x{out.shape[0]}), memory safe")


def test_7_complete_3_stage_restoration_pipeline(test_patch):
    """Verify full end-to-end integration: OpenCV -> GFPGAN -> Real-ESRGAN."""
    t_start = time.perf_counter()

    # Stage 1: OpenCV Restoration (Denoising, Scratch Inpainting, CLAHE, Sharpening)
    stage1_out, meta_s1 = restore_image_opencv(test_patch, config=PRESETS["balanced"])
    assert stage1_out.shape == test_patch.shape, f"Stage 1 shape changed: {stage1_out.shape}"

    # Stage 2: GFPGAN Face Restoration
    stage2_out, meta_s2 = restore_faces(stage1_out, fidelity_weight=0.5, upscale=1)
    assert stage2_out.shape == test_patch.shape, f"Stage 2 shape changed: {stage2_out.shape}"

    # Stage 3: Real-ESRGAN 2x Super-Resolution
    stage3_out, meta_s3 = upscale_image(stage2_out, scale=2, tile=400)
    assert meta_s3["success"] is True, f"Stage 3 failed: {meta_s3.get('error')}"

    expected_shape = (test_patch.shape[0] * 2, test_patch.shape[1] * 2, 3)
    assert stage3_out.shape == expected_shape, f"Final pipeline shape mismatch: expected {expected_shape}, got {stage3_out.shape}"

    total_time = (time.perf_counter() - t_start) * 1000
    out_dir = WORKSPACE_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(stage3_out, out_dir / "test_full_pipeline_restored.png")

    print(
        f"\n[PASS] Complete 3-Stage Pipeline:"
        f" Input ({test_patch.shape[1]}x{test_patch.shape[0]}) -> Final Output ({stage3_out.shape[1]}x{stage3_out.shape[0]})"
        f" in {total_time:.1f} ms"
        f" (Stage 1: {meta_s1['total_time_ms']:.1f}ms, Stage 2: {meta_s2['execution_time_ms']:.1f}ms, Stage 3: {meta_s3['execution_time_ms']:.1f}ms)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

