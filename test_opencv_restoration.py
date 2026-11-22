"""
Verification and Test Script for OpenCV Restoration Module.
Creates a synthetic degraded photograph (with noise, scratch cracks, fading, and color cast)
and runs the entire restoration pipeline to verify correct functionality and performance.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add workspace to sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_DIR))

from modules.opencv_restoration import (
    RestorationConfig,
    PRESETS,
    restore_image_opencv,
    remove_noise,
    detect_scratches,
    repair_scratches,
    apply_clahe,
    adjust_brightness_contrast,
    balance_colors_gray_world,
    unsharp_mask
)
from utils.image_io import save_image, load_image, validate_image_file


def create_synthetic_damaged_photo(width: int = 512, height: int = 512) -> np.ndarray:
    """
    Creates a realistic degraded photograph pattern with:
    1. Base portrait-like structure (gradient + circular tones)
    2. Aged sepia/yellow color cast
    3. Low contrast / fading
    4. Film grain / Gaussian noise
    5. Bright scratches & dark crease lines
    """
    # 1. Base image (smooth portrait lighting)
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    base_luma = 180 - (dist_from_center / float(np.hypot(center_x, center_y))) * 110
    base_luma = np.clip(base_luma, 40, 220).astype(np.uint8)

    # Convert to 3-channel
    img = np.dstack([base_luma, base_luma, base_luma]).astype(np.float32)

    # Add geometric facial/portrait contours
    cv2.circle(img, (center_x, center_y - 30), 80, (140, 140, 140), -1)
    cv2.ellipse(img, (center_x, center_y + 110), (120, 90), 0, 0, 360, (90, 90, 90), -1)

    # 2. Add sepia tone
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ]).T
    img = np.dot(img, sepia_filter)
    img = np.clip(img, 0, 255)

    # 3. Add fading (compress dynamic range towards midtones)
    img = img * 0.65 + 45.0

    # 4. Add film grain & noise
    noise = np.random.normal(0, 12, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # 5. Add realistic scratches and cracks
    # Bright scratches (white cuts)
    cv2.line(img, (60, 40), (280, 480), (245, 245, 245), 2)
    cv2.line(img, (320, 80), (460, 380), (250, 250, 250), 1)
    # Dark crease lines
    cv2.line(img, (140, 490), (410, 100), (25, 20, 15), 2)
    # Dust specks
    for _ in range(35):
        rx, ry = np.random.randint(20, width - 20), np.random.randint(20, height - 20)
        color = (245, 245, 245) if np.random.rand() > 0.5 else (20, 20, 20)
        cv2.circle(img, (rx, ry), np.random.randint(1, 3), color, -1)

    return img.astype(np.uint8)


def run_tests():
    print("=" * 60)
    print("TEST SUITE: OpenCV Restoration Module Verification")
    print("=" * 60)

    # 1. Generate damaged test image
    test_dir = WORKSPACE_DIR / "test_samples"
    out_dir = WORKSPACE_DIR / "outputs"
    test_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    damaged_img = create_synthetic_damaged_photo(512, 512)
    sample_path = test_dir / "sample_damaged.png"
    save_image(damaged_img, sample_path)
    print(f"[OK] Generated synthetic damaged test photo at: {sample_path}")

    # 2. Test Image IO & Validation
    is_valid, err = validate_image_file(sample_path)
    assert is_valid, f"Validation failed: {err}"
    loaded_img, meta = load_image(sample_path)
    assert loaded_img.shape == (512, 512, 3), f"Unexpected loaded shape: {loaded_img.shape}"
    assert loaded_img.dtype == np.uint8, "Loaded image must be uint8"
    print(f"[OK] Image I/O validation passed: {meta}")

    # 3. Test Denoising
    denoised_fastnl = remove_noise(loaded_img, method="fastnl", strength=7.0)
    assert denoised_fastnl.shape == loaded_img.shape
    save_image(denoised_fastnl, out_dir / "test_denoised_fastnl.png")
    print("[OK] remove_noise (FastNL) succeeded.")

    denoised_bilateral = remove_noise(loaded_img, method="bilateral", strength=7.0)
    assert denoised_bilateral.shape == loaded_img.shape
    print("[OK] remove_noise (Bilateral) succeeded.")

    # 4. Test Scratch Detection and Inpainting
    mask = detect_scratches(loaded_img, threshold=15, kernel_size=7)
    assert mask.shape == (512, 512)
    defects_found = np.count_nonzero(mask)
    assert defects_found > 0, "Scratch detection should find synthetic scratches and dust specks!"
    save_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), out_dir / "test_scratch_mask.png")
    print(f"[OK] detect_scratches identified {defects_found} damaged pixels.")

    inpainted_telea, _ = repair_scratches(loaded_img, mask=mask, inpaint_method="telea")
    assert inpainted_telea.shape == loaded_img.shape
    save_image(inpainted_telea, out_dir / "test_inpainted_telea.png")
    print("[OK] repair_scratches (Telea Fast Marching) succeeded.")

    inpainted_ns, _ = repair_scratches(loaded_img, mask=mask, inpaint_method="ns")
    assert inpainted_ns.shape == loaded_img.shape
    save_image(inpainted_ns, out_dir / "test_inpainted_ns.png")
    print("[OK] repair_scratches (Navier-Stokes) succeeded.")

    # 5. Test Contrast & Tone (CLAHE, Brightness, Gamma)
    clahe_img = apply_clahe(loaded_img, clip_limit=2.0, grid_size=8)
    assert clahe_img.shape == loaded_img.shape
    save_image(clahe_img, out_dir / "test_clahe.png")
    print("[OK] apply_clahe (LAB color space) succeeded.")

    tone_img = adjust_brightness_contrast(loaded_img, brightness=10, contrast=1.2, gamma=1.1)
    assert tone_img.shape == loaded_img.shape
    print("[OK] adjust_brightness_contrast succeeded.")

    wb_img = balance_colors_gray_world(loaded_img)
    assert wb_img.shape == loaded_img.shape
    save_image(wb_img, out_dir / "test_white_balance.png")
    print("[OK] balance_colors_gray_world succeeded.")

    # 6. Test Unsharp Masking
    sharp_img = unsharp_mask(loaded_img, amount=0.6, radius=1.0)
    assert sharp_img.shape == loaded_img.shape
    save_image(sharp_img, out_dir / "test_sharpened.png")
    print("[OK] unsharp_mask succeeded.")

    # 7. Test Full Orchestrator Pipeline with Presets
    for preset_name, config in PRESETS.items():
        restored, metadata = restore_image_opencv(loaded_img, config=config)
        assert restored.shape == loaded_img.shape
        assert restored.dtype == np.uint8
        out_preset_path = out_dir / f"test_restored_{preset_name}.png"
        save_image(restored, out_preset_path)
        print(f"[OK] Full Pipeline Preset '{preset_name}': {metadata['total_time_ms']} ms | Timings: {metadata['timings_ms']}")

    print("=" * 60)
    print("ALL TESTS PASSED! OpenCV Restoration engine is fully verified.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

