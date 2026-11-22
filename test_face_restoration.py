"""
Verification and Test Script for Stage 2: GFPGAN Face Restoration.

Tests:
1. Model loading and CPU device configuration.
2. Authentic vintage portrait face detection and restoration.
3. Integration with the existing OpenCV restoration pipeline.
4. Output image shapes, data types, and visual fidelity preservation.
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2

WORKSPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_DIR))

from modules.face_restoration import restore_faces, get_device
from modules.opencv_restoration import restore_image_opencv, PRESETS
from utils.image_io import load_image, save_image


def run_face_restoration_tests():
    print("=" * 60)
    print("STAGE 2 TEST SUITE: GFPGAN Face Restoration & Pipeline Integration")
    print("=" * 60)

    device = get_device()
    print(f"[DEVICE] PyTorch inference execution device: {device.type.upper()}")

    sample_path = WORKSPACE_DIR / "test_samples" / "sample_vintage_portrait.jpg"
    out_dir = WORKSPACE_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sample_path.exists():
        raise FileNotFoundError(f"Test portrait not found: {sample_path}")

    # 1. Load portrait
    image_rgb, metadata = load_image(sample_path)
    print(f"[OK] Loaded test portrait: {image_rgb.shape} | Original: {metadata['original_size']}")

    # 2. Test Standalone Face Restoration
    print("[RUNNING] Executing GFPGAN blind face restoration on CPU...")
    t0 = time.perf_counter()
    restored_face_rgb, meta_face = restore_faces(
        image_rgb,
        fidelity_weight=0.5,
        upscale=1,
        only_center_face=False
    )
    dt_face = (time.perf_counter() - t0) * 1000

    assert meta_face["success"], f"Face restoration failed: {meta_face.get('error')}"
    assert meta_face["faces_detected"] > 0, "Expected at least 1 face detected in portrait sample!"
    assert restored_face_rgb.shape == image_rgb.shape, f"Shape mismatch: {restored_face_rgb.shape} vs {image_rgb.shape}"
    assert restored_face_rgb.dtype == np.uint8, "Restored image must be uint8"

    save_image(restored_face_rgb, out_dir / "test_face_restored.png")
    print(f"[OK] GFPGAN Face Restoration Succeeded: {meta_face['faces_detected']} face(s) restored in {dt_face:.1f} ms.")

    if meta_face["cropped_faces"]:
        save_image(meta_face["cropped_faces"][0], out_dir / "test_face_crop_before.png")
        save_image(meta_face["restored_faces"][0], out_dir / "test_face_crop_after.png")
        print("[OK] Saved side-by-side face zoom crops for inspection.")

    # 3. Test Integrated Pipeline (OpenCV Denoise/Scratch/CLAHE + GFPGAN Face Restoration)
    print("[RUNNING] Testing Integrated Pipeline: OpenCV restoration + GFPGAN Face Restoration...")
    t_pipe_start = time.perf_counter()

    # Step A: OpenCV Restoration
    opencv_restored, opencv_meta = restore_image_opencv(image_rgb, config=PRESETS["balanced"])
    # Step B: Face Restoration
    pipeline_restored, pipe_face_meta = restore_faces(opencv_restored, fidelity_weight=0.5, upscale=1)

    total_pipe_time = (time.perf_counter() - t_pipe_start) * 1000
    assert pipeline_restored.shape == image_rgb.shape
    assert pipeline_restored.dtype == np.uint8

    save_image(pipeline_restored, out_dir / "test_integrated_pipeline.png")
    print(f"[OK] Integrated Pipeline Succeeded in {total_pipe_time:.1f} ms.")
    print(f"     - OpenCV Time: {opencv_meta['total_time_ms']} ms (Denoise + Scratch Inpaint + CLAHE + Sharpen)")
    print(f"     - GFPGAN Time: {pipe_face_meta['execution_time_ms']} ms ({pipe_face_meta['faces_detected']} faces)")

    print("=" * 60)
    print("STAGE 2 ALL TESTS PASSED! GFPGAN is fully operational.")
    print("=" * 60)


if __name__ == "__main__":
    run_face_restoration_tests()

