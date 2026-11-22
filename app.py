"""
Old Photograph Restoration and Enhancement Application
AI & Computer Vision Powered Pipeline
"""

import io
import time
from datetime import datetime
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Internal modules
from utils.image_io import (
    load_image,
    save_image,
    image_to_bytes,
    validate_image_file,
    SUPPORTED_EXTENSIONS
)
from modules.opencv_restoration import (
    RestorationConfig,
    PRESETS,
    restore_image_opencv,
    detect_scratches
)
from modules.lama_inpainting import (
    inpaint_photo,
    is_lama_available
)
from modules.face_restoration import (
    restore_faces,
    get_device
)
from modules.super_resolution import (
    upscale_image
)
from modules.colorization import (
    colorize_image
)
import base64
import streamlit.elements.image as st_image

# Streamlit >= 1.60 compatibility patch for streamlit_drawable_canvas
if not hasattr(st_image, "image_to_url"):
    def _compat_image_to_url(image, width=None, clamp=True, channels="RGB", output_format="PNG", image_id=""):
        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        return str(image)
    st_image.image_to_url = _compat_image_to_url

try:
    import streamlit_drawable_canvas
    if hasattr(streamlit_drawable_canvas, "st_image") and not hasattr(streamlit_drawable_canvas.st_image, "image_to_url"):
        streamlit_drawable_canvas.st_image.image_to_url = _compat_image_to_url
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    st_canvas = None
    CANVAS_AVAILABLE = False

# App configuration
st.set_page_config(
    page_title="AI Old Photo Restorer",
    page_icon="🕰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Custom CSS
CUSTOM_CSS = """
<style>
    /* Dark Theme Accent Styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #262c3a 100%);
        border: 1px solid #3d4455;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #9aa0a6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #64b5f6;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        border-bottom: 2px solid #29b6f6;
        padding-bottom: 6px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1976d2 0%, #0288d1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1565c0 0%, #0277bd 100%);
        box-shadow: 0 4px 12px rgba(2, 136, 209, 0.4);
        transform: translateY(-1px);
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        background-color: #2e7d32;
        color: white;
        margin-left: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_restoration_config():
    """Builds restoration configuration from sidebar user controls."""
    st.sidebar.markdown("### ⚙️ Restoration Settings")

    mode = st.sidebar.radio(
        "Restoration Mode",
        options=["Automatic Preset", "Custom Tuning"],
        help="Select 'Automatic Preset' for one-click curated restoration or 'Custom Tuning' to fine-tune individual processing stages."
    )

    if mode == "Automatic Preset":
        preset_choice = st.sidebar.selectbox(
            "Select Preset Profile",
            options=[
                "🍃 Natural Authentic Restoration (Recommended - Preserves Original Details)",
                "🌟 Like-New Studio Enhancement",
                "Balanced Restoration",
                "Gentle Refresh",
                "Deep Damage Recovery"
            ],
            index=0,
            help=(
                "- **🍃 Natural Authentic Restoration**: Precision crack & dust inpainting, preserves 100% of undamaged textures, clothing patterns, furniture, and facial expressions without over-smoothing or artificial AI artifacts.\n"
                "- **🌟 Like-New Studio Enhancement**: Balanced inpainting with optional gentle face restoration and 2x super-resolution.\n"
                "- **Balanced Restoration**: High quality scratch removal, CLAHE contrast, balanced edge-preserving denoising.\n"
                "- **Gentle Refresh**: Mild denoising, slight contrast & clarity boost.\n"
                "- **Deep Damage Recovery**: Heavy crack inpainting for severely torn or cracked photographs."
            )
        )
        preset_key_map = {
            "🍃 Natural Authentic Restoration (Recommended - Preserves Original Details)": "natural",
            "🌟 Like-New Studio Enhancement": "like_new",
            "Gentle Refresh": "gentle",
            "Balanced Restoration": "balanced",
            "Deep Damage Recovery": "deep_repair"
        }
        config = PRESETS[preset_key_map[preset_choice]]
        is_like_new = (preset_choice == "🌟 Like-New Studio Enhancement")
    else:
        is_like_new = False
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 1. Noise Reduction")
        enable_denoise = st.sidebar.checkbox("Enable Denoising", value=True)
        denoise_method = st.sidebar.selectbox(
            "Filter Algorithm",
            options=["bilateral", "fastnl"],
            format_func=lambda x: "Bilateral Filter (Edge-Preserving - Recommended, Keeps Grain)" if x == "bilateral" else "Fast Non-Local Means"
        )
        denoise_strength = st.sidebar.slider("Denoise Strength (h)", 0.5, 10.0, 2.0, 0.5) if enable_denoise else 0.0

        st.sidebar.markdown("#### 2. Scratch & Crack Repair")
        enable_scratch = st.sidebar.checkbox("Enable Inpainting", value=True)
        inpaint_engine = st.sidebar.selectbox(
            "Inpaint Engine",
            options=["lama", "telea", "ns"],
            format_func=lambda x: "Pretrained LaMa (Deep Learning - Recommended)" if x == "lama" else ("Telea (OpenCV Fast Marching)" if x == "telea" else "Navier-Stokes (OpenCV Fluid Dynamics)"),
            index=0,
            help="Pretrained LaMa neural network fills missing regions with high structural coherence. Telea and Navier-Stokes provide fast local interpolation."
        ) if enable_scratch else "lama"
        scratch_threshold = st.sidebar.slider(
            "Scratch Sensitivity Threshold", 14, 40, 24, 2,
            help="Higher threshold = more conservative (only obvious cracks); lower threshold = detects fainter scratches."
        ) if enable_scratch else 24
        inpaint_method = "telea"
        if inpaint_engine != "lama" and enable_scratch:
            inpaint_method = inpaint_engine
        inpaint_radius = st.sidebar.slider("Inpainting Radius", 1, 6, 3, 1) if enable_scratch else 3
        protect_faces = st.sidebar.checkbox(
            "Protect Facial Features",
            value=True,
            help="Shields eyes, nose, and mouth contours from being mistaken for cracks or scratches."
        ) if enable_scratch else True

        st.sidebar.markdown("#### 3. Tone & Contrast")
        enable_clahe = st.sidebar.checkbox("Adaptive Contrast (CLAHE in LAB)", value=True)
        clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 0.5, 4.0, 1.2, 0.1) if enable_clahe else 0.0
        brightness = st.sidebar.slider("Brightness Offset", -50, 50, 0, 1)
        contrast = st.sidebar.slider("Contrast Scale", 0.5, 2.0, 1.0, 0.05)
        gamma = st.sidebar.slider("Gamma Correction", 0.5, 2.0, 1.0, 0.05)
        enable_wb = st.sidebar.checkbox("Fix Yellow/Sepia Cast (Gray-World)", value=False)

        st.sidebar.markdown("#### 4. Sharpness")
        enable_sharpen = st.sidebar.checkbox("Unsharp Masking (USM)", value=True)
        sharpen_amount = st.sidebar.slider("Sharpen Amount", 0.1, 1.0, 0.25, 0.05) if enable_sharpen else 0.0

        config = RestorationConfig(
            enable_denoise=enable_denoise,
            denoise_method=denoise_method,
            denoise_strength=denoise_strength,
            denoise_color_strength=denoise_strength,
            enable_scratch_repair=enable_scratch,
            scratch_threshold=scratch_threshold,
            inpaint_radius=inpaint_radius,
            inpaint_method=inpaint_method,
            inpaint_engine=inpaint_engine,
            protect_faces=protect_faces,
            enable_contrast_enhancement=True,
            enable_clahe=enable_clahe,
            clahe_clip_limit=clahe_clip,
            brightness=brightness,
            contrast=contrast,
            gamma=gamma,
            enable_white_balance=enable_wb,
            enable_sharpening=enable_sharpen,
            sharpen_amount=sharpen_amount
        )

    # Stage 2: Deep Learning Face Restoration Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Deep Learning Face Restoration")
    enable_gfpgan = st.sidebar.checkbox(
        "Enable GFPGAN Face Enhancement",
        value=is_like_new,
        help="Optional: enhances facial features in vintage portraits using generative facial priors. Uncheck to preserve 100% authentic original faces."
    )

    face_fidelity = 0.50 if is_like_new else 0.45
    face_blend = 0.80 if is_like_new else 0.70
    only_center = False
    if enable_gfpgan:
        face_fidelity = st.sidebar.slider(
            "Facial Fidelity Weight",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Lower values (0.35-0.50) preserve true facial structure and expressions; higher values generate artificial features."
        )
        face_blend = st.sidebar.slider(
            "Authentic Likeness Blend",
            min_value=0.40,
            max_value=1.0,
            value=0.70,
            step=0.05,
            help="Blends enhanced clarity with authentic photograph skin texture (0.70 avoids plastic doll look)."
        )
        only_center = st.sidebar.checkbox(
            "Only Center Face",
            value=False,
            help="If checked, only the primary central face is restored."
        )

    # Stage 3: Deep Learning Super-Resolution (Real-ESRGAN) Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 AI Super-Resolution (Real-ESRGAN)")
    enable_sr = st.sidebar.checkbox(
        "Enable Real-ESRGAN Upscaling",
        value=False,
        help="Optional: upscales image resolution by 2x or 4x with sharp edge detail reconstruction."
    )

    sr_scale = 2
    sr_tile = 400
    if enable_sr:
        sr_scale = st.sidebar.radio(
            "Upscale Factor",
            options=[2, 4],
            format_func=lambda x: f"{x}x (High Clarity)" if x == 2 else f"{x}x (Ultra Resolution)",
            index=0,
            help="2x is faster and ideal for high-resolution scans; 4x provides maximum magnification and detail reconstruction."
        )
        sr_tile = st.sidebar.slider(
            "Tiling Window Size",
            min_value=200,
            max_value=600,
            value=400,
            step=50,
            help="Splits large images into tiles to prevent CPU memory exhaustion. 400 is the recommended sweet spot."
        )

    # Stage 4: Deep Learning Photo Colorization (B&W to Color) Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 AI Colorization (B&W to Color)")
    enable_colorization = st.sidebar.checkbox(
        "Enable Photo Colorization",
        value=False,
        help="Transforms black-and-white or faded monochrome historical photographs into natural colors using deep neural networks."
    )

    color_model = "eccv16"
    color_saturation = 1.0
    if enable_colorization:
        color_model = st.sidebar.selectbox(
            "Colorization Model",
            options=["eccv16", "siggraph17"],
            format_func=lambda x: "ECCV 2016 (Vibrant & Rich)" if x == "eccv16" else "SIGGRAPH 2017 (Natural & Balanced)",
            index=0,
            help="ECCV 2016 provides rich vivid tones; SIGGRAPH 2017 provides subtle, natural skin and earth tones."
        )
        color_saturation = st.sidebar.slider(
            "Color Saturation",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.05,
            help="Adjust color vibrancy: 1.0 is default, values above 1.0 enhance color warmth and saturation."
        )

    return (
        config,
        enable_gfpgan,
        face_fidelity,
        face_blend,
        only_center,
        enable_sr,
        sr_scale,
        sr_tile,
        enable_colorization,
        color_model,
        color_saturation
    )


def main():
    device = get_device()
    dev_label = "CUDA" if device.type == "cuda" else "CPU"

    # Header
    st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem;">🕰️ Old Photograph Restoration & Enhancement</h1>
                <p style="color: #9aa0a6; margin: 5px 0 0 0; font-size: 1rem;">
                    Restore faded, noisy, scratched, and damaged historical photographs with computer vision & AI.
                </p>
            </div>
            <div>
                <span class="badge">OpenCV 4.11</span>
                <span class="badge" style="background-color: #00897b;">LaMa Inpainting</span>
                <span class="badge" style="background-color: #6a1b9a;">GFPGAN v1.4</span>
                <span class="badge" style="background-color: #e65100;">Real-ESRGAN</span>
                <span class="badge" style="background-color: #c2185b;">Colorization</span>
                <span class="badge" style="background-color: #0288d1;">{dev_label} Active</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Configuration sidebar
    (
        config,
        enable_gfpgan,
        face_fidelity,
        face_blend,
        only_center,
        enable_sr,
        sr_scale,
        sr_tile,
        enable_colorization,
        color_model,
        color_saturation
    ) = get_restoration_config()

    # Image upload area
    st.markdown('<div class="section-header">📁 Step 1: Upload Photograph</div>', unsafe_allow_html=True)
    
    col_upload, col_sample_fam, col_sample1, col_sample2, col_sample3 = st.columns([2.0, 1.1, 1, 1, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload an old photograph (JPG, JPEG, or PNG)",
            type=["jpg", "jpeg", "png"],
            help="High-resolution scans or photos of vintage prints produce the best results."
        )
    with col_sample_fam:
        st.markdown("<p style='font-size: 0.85rem; color: #9aa0a6; margin-top: 15px;'>Family Portrait:</p>", unsafe_allow_html=True)
        use_sample_family = st.button("👨‍👩‍👧‍👦 Family Photo", help="Loads the damaged family photograph (756x745).")
    with col_sample1:
        st.markdown("<p style='font-size: 0.85rem; color: #9aa0a6; margin-top: 15px;'>Test Scratches:</p>", unsafe_allow_html=True)
        use_sample_damage = st.button("🧪 Scratched", help="Loads a synthetic test photo with heavy scratches, grain, and fading.")
    with col_sample2:
        st.markdown("<p style='font-size: 0.85rem; color: #9aa0a6; margin-top: 15px;'>Test Portrait:</p>", unsafe_allow_html=True)
        use_sample_portrait = st.button("👤 Portrait", help="Loads an authentic vintage degraded portrait to test GFPGAN face restoration.")
    with col_sample3:
        st.markdown("<p style='font-size: 0.85rem; color: #9aa0a6; margin-top: 15px;'>Test B&W:</p>", unsafe_allow_html=True)
        use_sample_bw = st.button("🎨 B&W Photo", help="Loads an authentic black-and-white historical photo to test AI colorization.")

    if "active_sample" not in st.session_state:
        st.session_state["active_sample"] = None
    if "restored_image" not in st.session_state:
        st.session_state["restored_image"] = None
    if "exec_meta" not in st.session_state:
        st.session_state["exec_meta"] = None
    if "color_meta" not in st.session_state:
        st.session_state["color_meta"] = None
    if "gfpgan_meta" not in st.session_state:
        st.session_state["gfpgan_meta"] = None
    if "sr_meta" not in st.session_state:
        st.session_state["sr_meta"] = None
    if "diff_stats" not in st.session_state:
        st.session_state["diff_stats"] = None
    if "active_image_id" not in st.session_state:
        st.session_state["active_image_id"] = None

    image_source = None
    if uploaded_file is not None:
        is_valid, err_msg = validate_image_file(uploaded_file)
        if not is_valid:
            st.error(f"❌ {err_msg}")
            return
        image_source = uploaded_file
        st.session_state["active_sample"] = None
    elif use_sample_family:
        st.session_state["active_sample"] = "family"
    elif use_sample_damage:
        st.session_state["active_sample"] = "damaged"
    elif use_sample_portrait:
        st.session_state["active_sample"] = "portrait"
    elif use_sample_bw:
        st.session_state["active_sample"] = "bw"

    if image_source is None and st.session_state.get("active_sample") is not None:
        sample_key = st.session_state["active_sample"]
        if sample_key == "family":
            sample_family_path = Path(__file__).resolve().parent / "test_samples" / "family_damaged_756x745.png"
            if sample_family_path.exists():
                image_source = sample_family_path
        elif sample_key == "damaged":
            sample_path = Path(__file__).resolve().parent / "test_samples" / "sample_damaged.png"
            if sample_path.exists():
                image_source = sample_path
        elif sample_key == "portrait":
            sample_portrait_path = Path(__file__).resolve().parent / "test_samples" / "sample_vintage_portrait.jpg"
            if sample_portrait_path.exists():
                image_source = sample_portrait_path
        elif sample_key == "bw":
            sample_bw_path = Path(__file__).resolve().parent / "test_samples" / "sample_bw_vintage.png"
            if sample_bw_path.exists():
                image_source = sample_bw_path

    if image_source is None:
        st.info("👆 Please upload an image or click one of the quick test buttons above to begin restoration.")
        return

    # Check if active image changed; if so, clear previous restoration
    curr_id = str(image_source) if isinstance(image_source, Path) else getattr(image_source, "name", "uploaded")
    if st.session_state.get("active_image_id") != curr_id:
        st.session_state["active_image_id"] = curr_id
        st.session_state["restored_image"] = None
        st.session_state["exec_meta"] = None
        st.session_state["color_meta"] = None
        st.session_state["gfpgan_meta"] = None
        st.session_state["sr_meta"] = None
        st.session_state["diff_stats"] = None

    # Load image
    try:
        image_rgb, metadata = load_image(image_source)
    except Exception as e:
        st.error(f"❌ Failed to load image: {str(e)}")
        return

    # Step 1.5: Precision Damage Detection & Live Mask Preview
    st.markdown('<div class="section-header">🔍 Step 1.5: Damage Detection & Live Mask Inspection</div>', unsafe_allow_html=True)
    st.markdown(
        "Inspect the detected damage map below. The AI targets **only** physical cracks, scratches, and dust. "
        "Original clothing patterns, faces, furniture, and film grain are preserved."
    )

    col_sens1, col_sens2, col_sens3 = st.columns([1.2, 1, 1])
    with col_sens1:
        mask_sens_val = st.slider(
            "Damage Detection Sensitivity",
            min_value=14,
            max_value=36,
            value=int(config.scratch_threshold),
            step=2,
            help="Higher threshold = more conservative (only prominent cracks); lower threshold = detects fainter scratches."
        )
    with col_sens2:
        chk_cracks = st.checkbox("Detect Cracks & Scratches", value=True, help="Elongated tears and creases across the photograph.")
        chk_dust = st.checkbox("Detect Dust & Pinpoint Specks", value=True, help="Isolated small spots on smooth regions.")
    with col_sens3:
        chk_stains = st.checkbox("Detect Stains / Paper Tears", value=False, help="Broad water marks or frayed paper edges.")
        chk_faces = st.checkbox("Protect Facial Features", value=True, help="Shields eyes, nose, and mouth contours.")

    # Calculate initial automatic damage mask
    auto_mask = detect_scratches(
        image_rgb,
        threshold=mask_sens_val,
        kernel_size=config.scratch_kernel_size,
        min_elongation=config.scratch_min_elongation,
        min_length=config.scratch_min_length,
        protect_faces=chk_faces,
        detect_cracks=chk_cracks,
        detect_dust=chk_dust,
        detect_stains=chk_stains,
        mask_dilation=0
    )

    defect_count = int(np.count_nonzero(auto_mask))
    total_px = image_rgb.shape[0] * image_rgb.shape[1]
    defect_pct = (defect_count / float(total_px)) * 100.0

    # Display live damage mask preview side-by-side with original
    col_prev1, col_prev2 = st.columns(2)
    with col_prev1:
        st.markdown(f"**📷 Original Photograph** ({image_rgb.shape[1]} × {image_rgb.shape[0]})")
        st.image(image_rgb, use_container_width=True)
    with col_prev2:
        st.markdown(f"**🔍 Detected Damage Map** ({defect_count:,} pixels / {defect_pct:.2f}% of image)")
        # Highlight damage overlay on photo for intuitive inspection
        overlay = image_rgb.copy()
        overlay[auto_mask > 0] = [255, 45, 45]  # vibrant red highlight
        mask_display = cv2.addWeighted(image_rgb, 0.65, overlay, 0.35, 0)
        st.image(mask_display, caption="Red overlay = damaged pixels targeted for inpainting", use_container_width=True)

    # Optional Manual Brush Correction for Missed Scratches or False Positives
    active_mask = auto_mask.copy()
    if config.enable_scratch_repair:
        with st.expander("🖌️ Fine-Tune Mask with Interactive Brush (Optional)", expanded=False):
            st.markdown(
                "Draw with the manual brush to add any missed cracks, or customize the damage selection."
            )
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                brush_size = st.slider("Manual Brush Size (px)", min_value=2, max_value=40, value=8, step=2)
            with col_b2:
                mask_mode = st.radio(
                    "Mask Selection Mode",
                    options=["Automatic Detection (AI)", "Automatic + Manual Brush", "Manual Brush Only"],
                    index=0,
                    horizontal=True,
                    help="Automatic: AI & morphological detector. Auto + Brush: combines detector with your brush strokes. Manual Only: repairs only areas painted by your brush."
                )

            manual_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
            if mask_mode in ["Automatic + Manual Brush", "Manual Brush Only"]:
                if CANVAS_AVAILABLE and st_canvas is not None:
                    try:
                        pil_bg = Image.fromarray(image_rgb)
                        max_canvas_w = 750
                        scale_ratio = min(1.0, max_canvas_w / float(image_rgb.shape[1]))
                        c_w = int(image_rgb.shape[1] * scale_ratio)
                        c_h = int(image_rgb.shape[0] * scale_ratio)

                        canvas_result = st_canvas(
                            fill_color="rgba(255, 255, 255, 0.0)",
                            stroke_width=brush_size,
                            stroke_color="#FFFFFF",
                            background_image=pil_bg.resize((c_w, c_h)) if scale_ratio < 1.0 else pil_bg,
                            update_streamlit=True,
                            height=c_h,
                            width=c_w,
                            drawing_mode="freedraw",
                            key="manual_damage_canvas"
                        )

                        if canvas_result is not None and canvas_result.image_data is not None:
                            drawn = (canvas_result.image_data[:, :, 3] > 0).astype(np.uint8) * 255
                            if np.count_nonzero(drawn) > 0:
                                if (c_w, c_h) != (image_rgb.shape[1], image_rgb.shape[0]):
                                    manual_mask = cv2.resize(drawn, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                                else:
                                    manual_mask = drawn
                    except Exception as e:
                        st.warning(f"Interactive canvas notice: {e}. Falling back to automatic AI detection.")
                else:
                    st.info("Interactive canvas library is unavailable. Using automatic AI detection.")

            if mask_mode == "Manual Brush Only":
                active_mask = manual_mask
            elif mask_mode == "Automatic + Manual Brush" and np.count_nonzero(manual_mask) > 0:
                active_mask = cv2.bitwise_or(auto_mask, manual_mask)

    # Step 1.6: Prominent Action Button to Trigger Restoration
    st.markdown('<div class="section-header">🚀 Step 2: Restore Photograph</div>', unsafe_allow_html=True)
    st.markdown(
        "Click **⚡ Restore Photograph** below to execute inpainting on the detected damage mask and generate the clean photograph."
    )

    col_btn_run, col_btn_reset = st.columns([3, 1])
    with col_btn_run:
        restore_btn_clicked = st.button(
            "⚡ Restore Photograph",
            type="primary",
            use_container_width=True,
            help="Run AI inpainting (LaMa / OpenCV) and deep learning models to repair detected damage"
        )
    with col_btn_reset:
        if st.button("🔄 Clear Restoration", use_container_width=True, help="Clear restored photograph and re-detect"):
            st.session_state["restored_image"] = None
            st.session_state["exec_meta"] = None
            st.session_state["color_meta"] = None
            st.session_state["gfpgan_meta"] = None
            st.session_state["sr_meta"] = None
            st.session_state["diff_stats"] = None
            st.rerun()

    # When user clicks the Restore Photograph button, run pipeline
    if restore_btn_clicked:
        with st.spinner("Restoring photograph with AI inpainting & deep learning..."):
            progress_bar = st.progress(0, text="Initializing restoration pipeline...")
            time.sleep(0.05)
            
            # 1. OpenCV & Deep Learning Inpainting Stage
            progress_bar.progress(20, text=f"Inpainting damage with {config.inpaint_engine.upper()} engine...")
            restored_rgb, exec_meta = restore_image_opencv(image_rgb, config=config, custom_mask=active_mask)
            progress_bar.progress(45, text="Inpainting complete. Preserving undamaged details...")

            # 2. Deep Learning Colorization Stage (ECCV16 / SIGGRAPH17)
            color_meta = {"success": False, "execution_time_ms": 0.0}
            if enable_colorization:
                progress_bar.progress(55, text=f"Running {color_model.upper()} AI Colorization on {dev_label}...")
                try:
                    restored_rgb, color_meta = colorize_image(
                        restored_rgb,
                        model_type=color_model,
                        render_factor=256,
                        color_saturation=color_saturation
                    )
                    if not color_meta.get("success", False):
                        st.warning(f"⚠️ Colorization notice: {color_meta.get('error')}")
                except Exception as e:
                    st.warning(f"⚠️ Colorization encountered an issue: {str(e)}")

            # 3. Deep Learning Face Restoration Stage (GFPGAN)
            gfpgan_meta = {"faces_detected": 0, "execution_time_ms": 0.0, "cropped_faces": [], "restored_faces": []}
            if enable_gfpgan:
                progress_bar.progress(70, text=f"Running GFPGAN v1.4 Face Restoration on {dev_label}...")
                try:
                    restored_rgb, gfpgan_meta = restore_faces(
                        restored_rgb,
                        fidelity_weight=face_fidelity,
                        upscale=1,
                        only_center_face=only_center,
                        face_blend_ratio=face_blend
                    )
                    if not gfpgan_meta.get("success", False):
                        st.warning(f"⚠️ Face restoration notice: {gfpgan_meta.get('error')}")
                except Exception as e:
                    st.warning(f"⚠️ Face restoration encountered an issue: {str(e)}")

            # 4. Deep Learning Super-Resolution Stage (Real-ESRGAN)
            sr_meta = {
                "success": False,
                "execution_time_ms": 0.0,
                "scale": 1,
                "output_shape": (restored_rgb.shape[1], restored_rgb.shape[0])
            }
            if enable_sr:
                progress_bar.progress(88, text=f"Running Real-ESRGAN {sr_scale}x Super-Resolution on {dev_label} (tile={sr_tile})...")
                try:
                    restored_rgb, sr_meta = upscale_image(
                        restored_rgb,
                        scale=sr_scale,
                        tile=sr_tile
                    )
                    if not sr_meta.get("success", False):
                        st.warning(f"⚠️ Super-resolution notice: {sr_meta.get('error')}")
                except Exception as e:
                    st.warning(f"⚠️ Super-resolution encountered an issue: {str(e)}")

            progress_bar.progress(100, text="Restoration complete!")
            time.sleep(0.1)
            progress_bar.empty()

            # Calculate verification metrics proving output is genuinely restored
            orig_h, orig_w = image_rgb.shape[:2]
            res_h, res_w = restored_rgb.shape[:2]
            if (res_w, res_h) != (orig_w, orig_h):
                res_down = cv2.resize(restored_rgb, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
            else:
                res_down = restored_rgb

            diff = cv2.absdiff(image_rgb, res_down)
            mask_bool = active_mask > 0
            if np.count_nonzero(mask_bool) > 0:
                max_diff_mask = int(np.max(diff[mask_bool]))
                mean_diff_mask = float(np.mean(diff[mask_bool]))
            else:
                max_diff_mask = 0
                mean_diff_mask = 0.0

            clean_bool = active_mask == 0
            diff_outside = diff[clean_bool]
            if len(diff_outside) > 0:
                mean_diff_clean = float(np.mean(diff_outside))
            else:
                mean_diff_clean = 0.0

            diff_stats = {
                "max_diff_mask": max_diff_mask,
                "mean_diff_mask": round(mean_diff_mask, 1),
                "mean_diff_clean": round(mean_diff_clean, 2),
                "repaired_pixels": int(np.count_nonzero(mask_bool)),
                "diff_map": cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            }

            # Cache in session state
            st.session_state["restored_image"] = restored_rgb
            st.session_state["exec_meta"] = exec_meta
            st.session_state["color_meta"] = color_meta
            st.session_state["gfpgan_meta"] = gfpgan_meta
            st.session_state["sr_meta"] = sr_meta
            st.session_state["diff_stats"] = diff_stats
            st.toast("Restoration successfully completed!", icon="✨")

    # Step 3: Processing & Results Display
    st.markdown('<div class="section-header">✨ Step 3: Restoration Results & Inspection</div>', unsafe_allow_html=True)

    if st.session_state.get("restored_image") is not None:
        restored_rgb = st.session_state["restored_image"]
        exec_meta = st.session_state["exec_meta"]
        color_meta = st.session_state.get("color_meta") or {}
        gfpgan_meta = st.session_state.get("gfpgan_meta") or {}
        sr_meta = st.session_state.get("sr_meta") or {}
        diff_stats = st.session_state.get("diff_stats") or {}

        # Display Metrics Cards
        total_time = (
            exec_meta['total_time_ms']
            + color_meta.get('execution_time_ms', 0.0)
            + gfpgan_meta.get('execution_time_ms', 0.0)
            + sr_meta.get('execution_time_ms', 0.0)
        )
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Input Size</div>
                    <div class="metric-value">{metadata['original_size'][0]} × {metadata['original_size'][1]}</div>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            out_w, out_h = restored_rgb.shape[1], restored_rgb.shape[0]
            scale_tag = f" ({sr_scale}x)" if enable_sr and sr_meta.get("success", False) else ""
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Output Size{scale_tag}</div>
                    <div class="metric-value" style="color: #4caf50;">{out_w} × {out_h}</div>
                </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Latency</div>
                    <div class="metric-value">{total_time:.1f} ms</div>
                </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Defects Repaired</div>
                    <div class="metric-value">{exec_meta['detected_defects']:,}</div>
                </div>
            """, unsafe_allow_html=True)
        with m5:
            faces_count = gfpgan_meta.get('faces_detected', 0)
            face_label = f"{faces_count} Restored" if faces_count > 0 else ("None" if enable_gfpgan else "Off")
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Faces Restored</div>
                    <div class="metric-value" style="color: #ab47bc;">{face_label}</div>
                </div>
            """, unsafe_allow_html=True)
        with m6:
            if enable_colorization and color_meta.get("success", False):
                color_label = color_model.upper()
                color_card_color = "#e91e63"
            elif enable_colorization:
                color_label = "Failed"
                color_card_color = "#e53935"
            else:
                color_label = "Off"
                color_card_color = "#9aa0a6"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Colorization</div>
                    <div class="metric-value" style="color: {color_card_color};">{color_label}</div>
                </div>
            """, unsafe_allow_html=True)

        # Tabs for Comprehensive Inspection
        tab_side, tab_compare, tab_verify = st.tabs([
            "📊 Side-by-Side (Original | Damage Mask | Restored)",
            "🔍 Interactive Comparison (Before vs After)",
            "🛡️ Restoration Verification & Difference Map"
        ])

        with tab_side:
            has_damage_mask = exec_meta.get("damage_mask") is not None
            if has_damage_mask:
                col_orig, col_mask, col_rest = st.columns(3)
                with col_orig:
                    st.markdown(f"**📷 Original Photograph** ({metadata['original_size'][0]} × {metadata['original_size'][1]})")
                    st.image(image_rgb, use_container_width=True)
                with col_mask:
                    defects_ct = exec_meta.get("detected_defects", 0)
                    st.markdown(f"**🔍 Detected Damage Mask** ({defects_ct:,} px)")
                    st.image(
                        exec_meta["damage_mask"],
                        caption="Inpainting Mask (White = cracks, scratches & dust)",
                        use_container_width=True
                    )
                with col_rest:
                    st.markdown(f"**✨ Restored Photograph** ({restored_rgb.shape[1]} × {restored_rgb.shape[0]})")
                    st.image(restored_rgb, use_container_width=True)
            else:
                col_orig, col_rest = st.columns(2)
                with col_orig:
                    st.markdown(f"**📷 Original Photograph** ({metadata['original_size'][0]} × {metadata['original_size'][1]})")
                    st.image(image_rgb, use_container_width=True)
                with col_rest:
                    st.markdown(f"**✨ Restored Photograph** ({restored_rgb.shape[1]} × {restored_rgb.shape[0]})")
                    st.image(restored_rgb, use_container_width=True)

        with tab_compare:
            cmp_mode = st.radio("Display View", options=["✨ Restored Photograph", "📷 Original Photograph", "👥 Side-by-Side Zoom"], horizontal=True)
            if cmp_mode == "✨ Restored Photograph":
                st.image(restored_rgb, caption=f"Restored Photograph ({restored_rgb.shape[1]}×{restored_rgb.shape[0]})", use_container_width=True)
            elif cmp_mode == "📷 Original Photograph":
                st.image(image_rgb, caption=f"Original Photograph ({image_rgb.shape[1]}×{image_rgb.shape[0]})", use_container_width=True)
            else:
                cz_col1, cz_col2 = st.columns(2)
                with cz_col1:
                    st.markdown("**Original Photograph (Full)**")
                    st.image(image_rgb, use_container_width=True)
                with cz_col2:
                    st.markdown("**Restored Photograph (Full)**")
                    st.image(restored_rgb, use_container_width=True)

        with tab_verify:
            st.markdown("### 🛡️ Pixel Modification Verification")
            st.markdown(
                "This verification confirms that the restoration engine **actually repaired damaged pixels** "
                "and **did not simply return the original image**."
            )
            v1, v2, v3, v4 = st.columns(4)
            with v1:
                st.metric("Pixels Repaired", f"{diff_stats.get('repaired_pixels', 0):,} px")
            with v2:
                st.metric("Peak Inpainting Delta", f"{diff_stats.get('max_diff_mask', 0)} / 255")
            with v3:
                st.metric("Avg Crack Pixel Change", f"{diff_stats.get('mean_diff_mask', 0.0):.1f} levels")
            with v4:
                is_changed = diff_stats.get("max_diff_mask", 0) > 0
                st.metric("Output Differs From Original?", "YES (Repaired)" if is_changed else "NO CHANGE")

            if "diff_map" in diff_stats:
                st.markdown("**Repaired Pixel Difference Map (Brighter = larger inpainting correction):**")
                diff_display = cv2.equalizeHist(diff_stats["diff_map"]) if np.max(diff_stats["diff_map"]) > 0 else diff_stats["diff_map"]
                st.image(diff_display, caption="Difference Map showing where scratches and cracks were repaired", use_container_width=True)

        # Face Zoom Inspection (if faces were detected and restored)
        if gfpgan_meta.get("faces_detected", 0) > 0 and gfpgan_meta.get("cropped_faces"):
            st.markdown("### 🔍 Restored Facial Close-ups")
            num_faces = len(gfpgan_meta["cropped_faces"])
            cols = st.columns(min(4, num_faces * 2))
            for idx in range(min(2, num_faces)):
                col_b = cols[idx * 2]
                col_a = cols[idx * 2 + 1]
                with col_b:
                    st.image(
                        gfpgan_meta["cropped_faces"][idx],
                        caption=f"Face #{idx + 1} - Before (Original / Scratched)",
                        use_container_width=True
                    )
                with col_a:
                    st.image(
                        gfpgan_meta["restored_faces"][idx],
                        caption=f"Face #{idx + 1} - After (GFPGAN Restored)",
                        use_container_width=True
                    )

        # Diagnostic tools
        with st.expander("🔍 Diagnostic Inspection: Detected Scratch & Damage Mask"):
            if exec_meta.get("damage_mask") is not None:
                mask_disp = exec_meta["damage_mask"]
                st.image(mask_disp, caption="Binary Inpainting Mask (White = Detected Scratches/Tears/Cracks)", use_container_width=True)
            else:
                st.info("Scratch repair was disabled or no scratches detected.")

        with st.expander("⏱️ Pipeline Breakdown Details"):
            timings_breakdown = dict(exec_meta.get("timings_ms", {}))
            if enable_colorization and color_meta.get("success", False):
                timings_breakdown[f"colorization_{color_model}_ms"] = color_meta.get("execution_time_ms", 0.0)
            if enable_gfpgan:
                timings_breakdown["gfpgan_face_inference_ms"] = gfpgan_meta.get("execution_time_ms", 0.0)
            if enable_sr and sr_meta.get("success", False):
                timings_breakdown[f"realesrgan_{sr_scale}x_upscale_ms"] = sr_meta.get("execution_time_ms", 0.0)
            st.json(timings_breakdown)

        # Step 4: Download Section
        st.markdown('<div class="section-header">💾 Step 4: Download Restored Photograph</div>', unsafe_allow_html=True)
        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            download_format = st.selectbox("Export Format", options=["PNG", "JPG"], index=0)
        with d_col2:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "png" if download_format == "PNG" else "jpg"
            mime = "image/png" if download_format == "PNG" else "image/jpeg"
            file_bytes = image_to_bytes(restored_rgb, format=download_format)
            scale_suffix = f"_{sr_scale}x" if (enable_sr and sr_meta.get("success", False)) else ""
            color_suffix = "_colorized" if (enable_colorization and color_meta.get("success", False)) else ""
            
            st.download_button(
                label=f"⬇️ Download Restored Photograph ({download_format} - {restored_rgb.shape[1]}×{restored_rgb.shape[0]})",
                data=file_bytes,
                file_name=f"restored_photo{color_suffix}{scale_suffix}_{timestamp}.{ext}",
                mime=mime,
                use_container_width=True
            )

            if exec_meta.get("damage_mask") is not None and np.count_nonzero(exec_meta["damage_mask"]) > 0:
                mask_rgb = cv2.cvtColor(exec_meta["damage_mask"], cv2.COLOR_GRAY2RGB)
                mask_bytes = image_to_bytes(mask_rgb, format="PNG")
                st.download_button(
                    label="⬇️ Download Damage Mask (PNG)",
                    data=mask_bytes,
                    file_name=f"damage_mask_{timestamp}.png",
                    mime="image/png",
                    use_container_width=True
                )
    else:
        st.info("👆 Damage mask detected above! Click the **⚡ Restore Photograph** button above to repair the damaged areas and generate the restored image.")


if __name__ == "__main__":
    main()
