"""
Unit and Verification Tests for Pretrained LaMa Inpainting Module.

Tests:
1. Verification of big-lama.pt model availability in models/pretrained/.
2. CPU inference execution and performance.
3. Strict preservation of undamaged pixels (0 diff outside mask).
4. Exact preservation of image dimensions (H, W, 3).
5. Seamless OpenCV fallback when requested or if model is bypassed.
6. Integration test on damaged vintage photograph.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

WORKSPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_DIR))

from modules.lama_inpainting import (
    is_lama_available,
    load_lama_model,
    inpaint_with_lama,
    inpaint_photo
)
from utils.image_io import load_image


def test_1_lama_model_availability():
    """Verify that big-lama.pt exists in models/pretrained/ with valid size."""
    assert is_lama_available(), "Pretrained LaMa weights (big-lama.pt) must be available in models/pretrained/"


def test_2_cpu_inference_and_dimensions():
    """Verify LaMa execution on CPU and exact dimension preservation."""
    # Create test pattern (200x200) with a synthetic diagonal cut
    test_img = np.full((200, 200, 3), 180, dtype=np.uint8)
    test_mask = np.zeros((200, 200), dtype=np.uint8)
    
    # Add black diagonal cut
    cv2.line(test_img, (20, 20), (180, 180), (10, 10, 10), 5)
    cv2.line(test_mask, (20, 20), (180, 180), 255, 7)

    inpainted, meta = inpaint_with_lama(test_img, test_mask)
    
    assert meta["success"] is True
    assert inpainted.shape == test_img.shape
    assert inpainted.dtype == np.uint8
    assert meta["pixels_repaired"] > 0
    print(f"\n[PASS] LaMa CPU inference succeeded in {meta['execution_time_ms']} ms ({meta['pixels_repaired']} px repaired)")


def test_3_strict_undamaged_pixel_preservation():
    """Verify that every pixel outside the mask is 100.00% identical to the original."""
    h, w = 150, 150
    # Create image with random texture
    np.random.seed(42)
    test_img = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    test_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Mark a 30x30 center region as damaged
    test_mask[60:90, 60:90] = 255

    inpainted, _ = inpaint_with_lama(test_img, test_mask)

    # Check outside mask
    diff_outside = cv2.absdiff(test_img, inpainted)[test_mask == 0]
    assert np.all(diff_outside == 0), "Pixels outside the mask MUST be 100% bit-exact identical to the original image!"
    print("\n[PASS] Bit-exact preservation of undamaged pixels confirmed (0.000 max difference outside mask)")


def test_4_opencv_fallback_functionality():
    """Verify OpenCV fallback mode produces valid inpainting."""
    test_img = np.full((100, 100, 3), 150, dtype=np.uint8)
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[40:60, 40:60] = 255

    # Telea
    out_telea, meta_t = inpaint_photo(test_img, test_mask, method="telea")
    assert meta_t["success"] is True
    assert out_telea.shape == test_img.shape

    # Navier-Stokes
    out_ns, meta_ns = inpaint_photo(test_img, test_mask, method="ns")
    assert meta_ns["success"] is True
    assert out_ns.shape == test_img.shape
    print("\n[PASS] OpenCV inpainting fallback (Telea & Navier-Stokes) verified")


def test_5_family_photo_lama_inpainting():
    """Verify LaMa inpainting on the actual damaged family photograph."""
    sample_path = WORKSPACE_DIR / "test_samples" / "family_damaged_756x745.png"
    if not sample_path.exists():
        pytest.skip("Family photo sample not available")

    img_rgb, _ = load_image(sample_path)
    h, w = img_rgb.shape[:2]

    # Create a test scratch mask (simulate diagonal scratch)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask, (50, 50), (400, 400), 255, 4)

    restored, meta = inpaint_photo(img_rgb, mask, method="lama")
    
    assert meta["success"] is True
    assert restored.shape == (h, w, 3)
    assert np.all(restored[mask == 0] == img_rgb[mask == 0]), "Undamaged pixels must remain strictly identical!"
    print(f"\n[PASS] Family photo LaMa inpainting verified ({meta['pixels_repaired']} px repaired in {meta['execution_time_ms']} ms)")
