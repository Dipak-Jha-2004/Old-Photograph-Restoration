"""
OpenCV-based Restoration Module for Old Photographs.

Provides modular, robust algorithms for:
1. Advanced Denoising (Fast Non-Local Means & Bilateral Filtering)
2. Scratch, Crack, and Tear Detection & Inpainting (Navier-Stokes / Fast Marching)
3. Dynamic Range & Contrast Recovery (CLAHE in LAB color space, Gamma correction, Brightness/Contrast)
4. Unsharp Masking & Detail Enhancement
5. Color Cast / White-Balance Normalization (Gray-World Assumption)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
import cv2
import numpy as np


@dataclass
class RestorationConfig:
    # Noise reduction
    enable_denoise: bool = True
    denoise_method: str = "bilateral"  # "bilateral" (edge-preserving, keeps grain) or "fastnl"
    denoise_strength: float = 2.0   # Filter strength h (2.0 preserves photographic grain & textures)
    denoise_color_strength: float = 2.0  # Filter strength for color

    # Scratch & Damage Repair
    enable_scratch_repair: bool = True
    scratch_threshold: int = 24      # Sensitivity threshold for scratch detection (15-40)
    scratch_kernel_size: int = 15    # Structuring element size for directional line kernels
    scratch_min_length: int = 12     # Minimum span for elongated cracks/scratches
    scratch_min_elongation: float = 2.0  # Min aspect ratio for crack contours
    protect_faces: bool = True       # Protect facial features (eyes, mouth) from scratch inpainting
    inpaint_radius: int = 3          # Inpainting radius (2-4 recommended)
    inpaint_method: str = "telea"    # "telea" or "ns" (Navier-Stokes)
    inpaint_engine: str = "lama"     # "lama" (Deep Learning), "telea", or "ns"
    detect_cracks: bool = True       # Detect elongated fissures, scratches, and cracks
    detect_dust: bool = True         # Detect isolated pinpoint specks and dust
    detect_stains: bool = False      # Detect broad water stains/blemishes
    mask_dilation: int = 0           # Mask expansion pixels (0 = exact, avoids bloat)
    feather_edges: bool = True       # Seamless feathered blending along inpaint boundary

    # Contrast & Brightness Enhancement
    enable_contrast_enhancement: bool = True
    enable_clahe: bool = True
    clahe_clip_limit: float = 1.2
    clahe_grid_size: int = 8
    brightness: int = 0              # -100 to 100
    contrast: float = 1.0            # 0.5 to 2.0
    gamma: float = 1.0               # 0.5 to 2.0

    # Color & White Balance
    enable_white_balance: bool = False  # Normalizes aged yellow/brown cast (keep False for sepia)

    # Sharpness
    enable_sharpening: bool = True
    sharpen_amount: float = 0.25     # Subtle USM (0.2 to 0.4 avoids halo artifacts)
    sharpen_radius: float = 1.0      # Gaussian blur sigma
    sharpen_threshold: float = 3.0   # Threshold to avoid amplifying flat noise/grain


# Predefined configurations
PRESETS = {
    "natural": RestorationConfig(
        enable_denoise=True,
        denoise_method="bilateral",
        denoise_strength=2.0,
        denoise_color_strength=2.0,
        enable_scratch_repair=True,
        scratch_threshold=24,
        scratch_kernel_size=15,
        scratch_min_length=12,
        scratch_min_elongation=2.0,
        protect_faces=True,
        inpaint_radius=3,
        inpaint_method="telea",
        inpaint_engine="lama",
        detect_cracks=True,
        detect_dust=True,
        detect_stains=False,
        mask_dilation=0,
        feather_edges=True,
        enable_contrast_enhancement=True,
        clahe_clip_limit=1.2,
        enable_white_balance=False,
        enable_sharpening=True,
        sharpen_amount=0.25,
        sharpen_threshold=3.0
    ),
    "like_new": RestorationConfig(
        enable_denoise=True,
        denoise_method="bilateral",
        denoise_strength=2.5,
        denoise_color_strength=2.5,
        enable_scratch_repair=True,
        scratch_threshold=22,
        scratch_kernel_size=15,
        scratch_min_length=10,
        scratch_min_elongation=1.8,
        protect_faces=True,
        inpaint_radius=3,
        inpaint_method="telea",
        inpaint_engine="lama",
        detect_cracks=True,
        detect_dust=True,
        detect_stains=False,
        mask_dilation=0,
        feather_edges=True,
        enable_contrast_enhancement=True,
        clahe_clip_limit=1.5,
        enable_white_balance=False,
        enable_sharpening=True,
        sharpen_amount=0.35,
        sharpen_threshold=2.5
    ),
    "gentle": RestorationConfig(
        enable_denoise=True,
        denoise_method="bilateral",
        denoise_strength=1.5,
        denoise_color_strength=1.5,
        enable_scratch_repair=False,
        enable_contrast_enhancement=True,
        clahe_clip_limit=1.1,
        enable_sharpening=True,
        sharpen_amount=0.2,
        sharpen_threshold=3.0
    ),
    "balanced": RestorationConfig(
        enable_denoise=True,
        denoise_method="bilateral",
        denoise_strength=2.2,
        denoise_color_strength=2.2,
        enable_scratch_repair=True,
        scratch_threshold=24,
        scratch_kernel_size=15,
        scratch_min_length=12,
        scratch_min_elongation=2.0,
        protect_faces=True,
        inpaint_radius=3,
        inpaint_method="telea",
        inpaint_engine="lama",
        detect_cracks=True,
        detect_dust=True,
        detect_stains=False,
        mask_dilation=0,
        feather_edges=True,
        enable_contrast_enhancement=True,
        clahe_clip_limit=1.3,
        enable_sharpening=True,
        sharpen_amount=0.3,
        sharpen_threshold=2.5
    ),
    "deep_repair": RestorationConfig(
        enable_denoise=True,
        denoise_method="fastnl",
        denoise_strength=4.0,
        denoise_color_strength=4.0,
        enable_scratch_repair=True,
        scratch_threshold=18,
        scratch_kernel_size=17,
        scratch_min_length=8,
        scratch_min_elongation=1.6,
        protect_faces=True,
        inpaint_radius=4,
        inpaint_method="ns",
        inpaint_engine="lama",
        detect_cracks=True,
        detect_dust=True,
        detect_stains=True,
        mask_dilation=1,
        feather_edges=True,
        enable_contrast_enhancement=True,
        clahe_clip_limit=2.0,
        enable_white_balance=False,
        enable_sharpening=True,
        sharpen_amount=0.45,
        sharpen_threshold=2.0
    )
}


def remove_noise(
    image_rgb: np.ndarray,
    method: str = "fastnl",
    strength: float = 7.0,
    color_strength: float = 7.0
) -> np.ndarray:
    """
    Denoises the photograph while preserving edge sharpness.
    """
    if strength <= 0.0:
        return image_rgb.copy()

    if method == "bilateral":
        # Bilateral filter in OpenCV expects uint8 BGR/RGB
        d = int(np.clip(strength, 3, 15))
        sigma_color = float(strength * 6.0)
        sigma_space = float(strength * 4.0)
        return cv2.bilateralFilter(image_rgb, d, sigma_color, sigma_space)
    else:
        # Fast Non-Local Means Denoising
        h = float(strength)
        h_color = float(color_strength)
        # Check if color or monochrome (all channels equal)
        is_monochrome = (
            np.all(image_rgb[:, :, 0] == image_rgb[:, :, 1]) and
            np.all(image_rgb[:, :, 1] == image_rgb[:, :, 2])
        )
        if is_monochrome:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            denoised_gray = cv2.fastNlMeansDenoising(
                gray, None, h=h, templateWindowSize=7, searchWindowSize=21
            )
            return cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2RGB)
        else:
            return cv2.fastNlMeansDenoisingColored(
                image_rgb, None, h=h, hColor=h_color, templateWindowSize=7, searchWindowSize=21
            )


def detect_scratches(
    image_rgb: np.ndarray,
    threshold: int = 24,
    kernel_size: int = 15,
    min_elongation: float = 2.0,
    min_length: int = 12,
    protect_faces: bool = True,
    detect_cracks: bool = True,
    detect_dust: bool = True,
    detect_stains: bool = False,
    mask_dilation: int = 0
) -> np.ndarray:
    """
    Precision Damage Detection for Old Photographs.
    
    Distinguishes genuine cracks, scratches, and dust from natural textures:
    - Directional line filtering (8 angles) detects elongated cracks & fissures.
    - Local density texture suppression prevents mistaking clothing patterns,
      floral prints, hair, curtains, or furniture for damage.
    - Facial sensitive shielding protects natural eye, nose, and mouth features.
    
    Returns:
        Binary mask (uint8, 0 or 255) where 255 indicates damaged areas to inpaint.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    total_pixels = h * w
    final_mask = np.zeros_like(gray)

    # Multi-scale background estimation
    bg = cv2.medianBlur(gray, 15)
    diff_dark = cv2.subtract(bg, gray)
    diff_bright = cv2.subtract(gray, bg)

    # 1. Detect facial regions for protective shielding
    face_boxes = []
    if protect_faces:
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                face_boxes = faces
        except Exception:
            face_boxes = []

    # 2. Elongated Cracks & Scratches (Directional Morphology)
    if detect_cracks:
        angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
        k_len = max(9, int(kernel_size))
        line_responses = []
        for a in angles:
            rad = np.deg2rad(a)
            dx = int(round((k_len / 2) * np.cos(rad)))
            dy = int(round((k_len / 2) * np.sin(rad)))
            k = np.zeros((k_len, k_len), dtype=np.uint8)
            cv2.line(k, (k_len // 2 - dx, k_len // 2 - dy), (k_len // 2 + dx, k_len // 2 + dy), 1, 1)
            resp_dark = cv2.morphologyEx(diff_dark, cv2.MORPH_OPEN, k)
            resp_bright = cv2.morphologyEx(diff_bright, cv2.MORPH_OPEN, k)
            line_responses.append(cv2.max(resp_dark, resp_bright))

        max_line = np.max(line_responses, axis=0)
        _, crack_raw = cv2.threshold(max_line, threshold, 255, cv2.THRESH_BINARY)

        num_c, labels_c, stats_c, centroids_c = cv2.connectedComponentsWithStats(crack_raw, connectivity=8)
        for i in range(1, num_c):
            area = stats_c[i, cv2.CC_STAT_AREA]
            if area < 5:
                continue
            bw = stats_c[i, cv2.CC_STAT_WIDTH]
            bh = stats_c[i, cv2.CC_STAT_HEIGHT]
            bbox_area = max(1, bw * bh)
            density = area / float(bbox_area)
            if area > (total_pixels * 0.05) and density > 0.40:
                continue

            cx, cy = centroids_c[i]
            in_eye_mouth = False
            for (fx, fy, fw, fh) in face_boxes:
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    rx = (cx - fx) / float(fw)
                    ry = (cy - fy) / float(fh)
                    if (0.30 <= ry <= 0.55 and (0.18 <= rx <= 0.46 or 0.54 <= rx <= 0.82)) or (0.68 <= ry <= 0.90 and 0.28 <= rx <= 0.72):
                        in_eye_mouth = True
                    break

            length = max(bw, bh)
            width = max(1, min(bw, bh))
            elongation = length / float(width)
            avg_thickness = area / float(length)

            if in_eye_mouth:
                # In eye or mouth core, strictly require severe crack crossing through
                if elongation >= 3.0 and length >= 20 and avg_thickness <= 4.0:
                    final_mask[labels_c == i] = 255
            else:
                # Cracks must be elongated and thin, or long continuous tears
                if (elongation >= min_elongation and length >= min_length and avg_thickness <= 6.5) or (length >= 24 and elongation >= 1.7):
                    final_mask[labels_c == i] = 255

    # 3. Isolated Dust & Pinpoint Specks
    if detect_dust:
        # Require higher contrast for isolated specks to prevent triggering on film grain
        _, spot_raw = cv2.threshold(diff_dark, threshold + 8, 255, cv2.THRESH_BINARY)
        _, b_spot_raw = cv2.threshold(diff_bright, threshold + 10, 255, cv2.THRESH_BINARY)
        spot_candidates = cv2.bitwise_or(spot_raw, b_spot_raw)

        # Local density suppression: if a region is dense with candidate dots, it is
        # textured fabric (floral dress, tweed, lace), wallpaper, or hair, NOT dust!
        k_dense = np.ones((31, 31), dtype=np.float32) / (31 * 31)
        spot_density = cv2.filter2D((spot_candidates > 0).astype(np.float32), -1, k_dense)
        isolated_spots = spot_candidates & ((spot_density < 0.06) * 255).astype(np.uint8)

        num_s, labels_s, stats_s, centroids_s = cv2.connectedComponentsWithStats(isolated_spots, connectivity=8)
        for i in range(1, num_s):
            area = stats_s[i, cv2.CC_STAT_AREA]
            if 2 <= area <= 22:
                cx, cy = centroids_s[i]
                in_face = False
                for (fx, fy, fw, fh) in face_boxes:
                    if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                        in_face = True
                        break
                if not in_face:
                    final_mask[labels_s == i] = 255

    # 4. Large Stains / Paper Tears (Optional)
    if detect_stains:
        blur_large = cv2.GaussianBlur(gray, (51, 51), 0)
        stain_diff = cv2.absdiff(gray, blur_large)
        _, stain_raw = cv2.threshold(stain_diff, threshold + 14, 255, cv2.THRESH_BINARY)
        num_st, labels_st, stats_st, _ = cv2.connectedComponentsWithStats(stain_raw, connectivity=8)
        for i in range(1, num_st):
            area = stats_st[i, cv2.CC_STAT_AREA]
            if 60 <= area <= (total_pixels * 0.03):
                final_mask[labels_st == i] = 255

    # Controlled dilation only if requested (default 0 keeps mask precise)
    if mask_dilation > 0:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mask_dilation + 1, 2 * mask_dilation + 1))
        final_mask = cv2.dilate(final_mask, k_dil, iterations=1)

    return final_mask


def repair_scratches(
    image_rgb: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: int = 24,
    kernel_size: int = 15,
    inpaint_radius: int = 3,
    inpaint_method: str = "telea",
    min_elongation: float = 2.0,
    min_length: int = 12,
    protect_faces: bool = True,
    inpaint_engine: str = "lama",
    detect_cracks: bool = True,
    detect_dust: bool = True,
    detect_stains: bool = False,
    mask_dilation: int = 0,
    feather_edges: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Repairs detected damage using LaMa deep inpainting with OpenCV fallback,
    strictly preserving 100% of original pixels outside the damage mask.
    
    Returns:
        (inpainted_rgb_image, damage_mask)
    """
    if mask is None:
        mask = detect_scratches(
            image_rgb,
            threshold=threshold,
            kernel_size=kernel_size,
            min_elongation=min_elongation,
            min_length=min_length,
            protect_faces=protect_faces,
            detect_cracks=detect_cracks,
            detect_dust=detect_dust,
            detect_stains=detect_stains,
            mask_dilation=mask_dilation
        )

    # If mask has no damaged pixels, return original image untouched
    if np.count_nonzero(mask) == 0:
        return image_rgb.copy(), mask

    inpainted = None
    # 1. Try LaMa inpainting if requested
    if inpaint_engine.lower() == "lama":
        try:
            from modules.lama_inpainting import inpaint_photo
            res, meta = inpaint_photo(image_rgb, mask=mask, method="lama", inpaint_radius=inpaint_radius)
            if meta.get("success", False):
                inpainted = res
        except Exception:
            inpainted = None

    # 2. OpenCV inpainting fallback
    if inpainted is None:
        flag = cv2.INPAINT_TELEA if inpaint_method.lower() == "telea" else cv2.INPAINT_NS
        inpainted = cv2.inpaint(image_rgb, mask, inpaintRadius=max(1, inpaint_radius), flags=flag)

    # 3. Localized Restoration Compositing:
    # Blend inpainted regions strictly where mask > 0 with seamless edge feathering
    if feather_edges and np.count_nonzero(mask) > 0:
        mask_f = (mask > 0).astype(np.float32)
        mask_feathered = cv2.GaussianBlur(mask_f, (5, 5), 1.0)[:, :, None]
        final_rgb = np.clip(
            mask_feathered * inpainted.astype(np.float32) + (1.0 - mask_feathered) * image_rgb.astype(np.float32),
            0,
            255
        ).astype(np.uint8)
        # Guarantee 100% exact original pixels wherever mask is completely zero
        exact_clean = (mask == 0)
        final_rgb[exact_clean] = image_rgb[exact_clean]
    else:
        final_rgb = np.where(mask[:, :, None] > 0, inpainted, image_rgb)

    return final_rgb, mask


def apply_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization in LAB color space.
    Operating on L (luminance) channel avoids color shifts and unnatural saturation.
    """
    if clip_limit <= 0.0:
        return image_rgb.copy()

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(grid_size), int(grid_size))
    )
    cl = clahe.apply(l)
    
    merged_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)


def adjust_brightness_contrast(
    image_rgb: np.ndarray,
    brightness: int = 0,
    contrast: float = 1.0,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Adjusts brightness (-100 to 100), contrast multiplier (0.5 to 2.0),
    and non-linear gamma (0.5 to 2.0).
    """
    img = image_rgb.astype(np.float32)

    # 1. Contrast and Brightness
    if contrast != 1.0 or brightness != 0:
        img = img * contrast + brightness
        img = np.clip(img, 0, 255)

    # 2. Gamma correction
    if abs(gamma - 1.0) > 0.01 and gamma > 0:
        inv_gamma = 1.0 / gamma
        img = 255.0 * ((img / 255.0) ** inv_gamma)
        img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


def balance_colors_gray_world(image_rgb: np.ndarray) -> np.ndarray:
    """
    Applies Gray-World White Balance algorithm to remove sepia/yellowing casts.
    """
    img = image_rgb.astype(np.float32)
    avg_r = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_b = np.mean(img[:, :, 2])
    
    if avg_r == 0 or avg_g == 0 or avg_b == 0:
        return image_rgb.copy()
        
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    
    scale_r = avg_gray / avg_r
    scale_g = avg_gray / avg_g
    scale_b = avg_gray / avg_b
    
    # Scale channels
    img[:, :, 0] = np.clip(img[:, :, 0] * scale_r, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * scale_g, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * scale_b, 0, 255)
    
    return img.astype(np.uint8)


def unsharp_mask(
    image_rgb: np.ndarray,
    amount: float = 0.5,
    radius: float = 1.0,
    threshold: int = 0
) -> np.ndarray:
    """
    Enhances sharpness and edge details using Unsharp Masking (USM).
    Formula: Sharp = Original + Amount * (Original - Blurred)
    """
    if amount <= 0.0:
        return image_rgb.copy()

    # Calculate kernel size based on radius/sigma
    k_size = int(2 * np.ceil(2 * radius) + 1)
    blurred = cv2.GaussianBlur(image_rgb, (k_size, k_size), sigmaX=radius, sigmaY=radius)
    
    # Detail / High-frequency component
    diff = image_rgb.astype(np.float32) - blurred.astype(np.float32)
    
    # Optional threshold to prevent amplifying flat noise
    if threshold > 0:
        low_contrast_mask = np.abs(diff) < threshold
        diff[low_contrast_mask] = 0.0

    sharpened = image_rgb.astype(np.float32) + amount * diff
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def restore_image_opencv(
    image_rgb: np.ndarray,
    config: Optional[RestorationConfig] = None,
    custom_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Full OpenCV Restoration Pipeline for old photographs.
    
    Pipeline Steps:
    1. Color / White Balance normalization (if enabled)
    2. Edge-preserving Denoising (Fast Non-Local Means / Bilateral)
    3. Scratch & Tear Detection and Inpainting (Navier-Stokes / Telea)
    4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    5. Brightness, Contrast & Gamma Adjustment
    6. Unsharp Masking Detail Enhancement
    
    Returns:
        (restored_rgb_image, execution_metadata_dict)
    """
    if config is None:
        config = PRESETS["balanced"]

    t_start = time.perf_counter()
    timings = {}
    current = image_rgb.copy()
    damage_mask = None

    # Step 1: Color cast correction
    if config.enable_white_balance:
        t0 = time.perf_counter()
        current = balance_colors_gray_world(current)
        timings["white_balance"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 2: Denoise
    if config.enable_denoise and config.denoise_strength > 0:
        t0 = time.perf_counter()
        current = remove_noise(
            current,
            method=config.denoise_method,
            strength=config.denoise_strength,
            color_strength=config.denoise_color_strength
        )
        timings["denoise"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 3: Scratch & Crack Inpainting
    if config.enable_scratch_repair:
        t0 = time.perf_counter()
        current, damage_mask = repair_scratches(
            current,
            mask=custom_mask,
            threshold=config.scratch_threshold,
            kernel_size=config.scratch_kernel_size,
            inpaint_radius=config.inpaint_radius,
            inpaint_method=config.inpaint_method,
            min_elongation=config.scratch_min_elongation,
            min_length=getattr(config, "scratch_min_length", 12),
            protect_faces=getattr(config, "protect_faces", True),
            inpaint_engine=getattr(config, "inpaint_engine", "lama"),
            detect_cracks=getattr(config, "detect_cracks", True),
            detect_dust=getattr(config, "detect_dust", True),
            detect_stains=getattr(config, "detect_stains", False),
            mask_dilation=getattr(config, "mask_dilation", 0),
            feather_edges=getattr(config, "feather_edges", True)
        )
        timings["scratch_repair"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 4: CLAHE Contrast Recovery
    if config.enable_contrast_enhancement and config.enable_clahe:
        t0 = time.perf_counter()
        current = apply_clahe(
            current,
            clip_limit=config.clahe_clip_limit,
            grid_size=config.clahe_grid_size
        )
        timings["clahe"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 5: Brightness, Contrast & Gamma
    if config.enable_contrast_enhancement and (
        config.brightness != 0 or config.contrast != 1.0 or config.gamma != 1.0
    ):
        t0 = time.perf_counter()
        current = adjust_brightness_contrast(
            current,
            brightness=config.brightness,
            contrast=config.contrast,
            gamma=config.gamma
        )
        timings["tone_adjust"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 6: Unsharp Masking
    if config.enable_sharpening and config.sharpen_amount > 0:
        t0 = time.perf_counter()
        current = unsharp_mask(
            current,
            amount=config.sharpen_amount,
            radius=config.sharpen_radius,
            threshold=getattr(config, "sharpen_threshold", 2.0)
        )
        timings["sharpen"] = round((time.perf_counter() - t0) * 1000, 2)

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    metadata = {
        "total_time_ms": total_time_ms,
        "timings_ms": timings,
        "detected_defects": int(np.count_nonzero(damage_mask)) if damage_mask is not None else 0,
        "damage_mask": damage_mask
    }

    return current, metadata

