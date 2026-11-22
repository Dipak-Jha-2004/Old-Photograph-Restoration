"""
Pretrained Deep Learning Model Downloader and Cache Manager.

Handles downloading and verifying official pretrained weights:
- LaMa Inpainting: big-lama.pt (~196 MB)
- GFPGAN v1.4: GFPGANv1.4.pth (~332 MB)
- Real-ESRGAN: RealESRGAN_x4plus.pth & RealESRGAN_x2plus.pth (~64 MB each)
- Colorization: ECCV16 (~129 MB) & SIGGRAPH17 (~138 MB)
- Face Detection / Parsing: detection_Resnet50_Final.pth (~104 MB) & parsing_parsenet.pth (~81 MB)
"""

import argparse
from pathlib import Path
import sys
import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models" / "pretrained"
GFPGAN_WEIGHTS_DIR = ROOT_DIR / "gfpgan" / "weights"

MODEL_REGISTRY = {
    "big_lama": {
        "filename": "big-lama.pt",
        "dest_dir": MODELS_DIR,
        "url": "https://github.com/enesmsvn/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
        "size_mb": 196.3,
        "expected_bytes": 205803670,
        "description": "Resolution-robust Large Mask Inpainting (LaMa) model with Fast Fourier Convolutions"
    },
    "gfpgan_v1.4": {
        "filename": "GFPGANv1.4.pth",
        "dest_dir": MODELS_DIR,
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "size_mb": 332.5,
        "expected_bytes": 348632874,
        "description": "GFPGAN v1.4 official blind face restoration model by Tencent ARC Lab"
    },
    "realesrgan_x4plus": {
        "filename": "RealESRGAN_x4plus.pth",
        "dest_dir": MODELS_DIR,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "size_mb": 63.9,
        "expected_bytes": 67040989,
        "description": "Real-ESRGAN x4plus official 4x super-resolution model"
    },
    "realesrgan_x2plus": {
        "filename": "RealESRGAN_x2plus.pth",
        "dest_dir": MODELS_DIR,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "size_mb": 63.9,
        "expected_bytes": 67061725,
        "description": "Real-ESRGAN x2plus official 2x super-resolution model"
    },
    "colorization_eccv16": {
        "filename": "colorization_release_v2-9b330a0b.pth",
        "dest_dir": MODELS_DIR,
        "url": "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
        "size_mb": 128.8,
        "expected_bytes": 128976165,
        "description": "ECCV 2016 Colorful Image Colorization model by Richard Zhang et al."
    },
    "colorization_siggraph17": {
        "filename": "siggraph17-df00044c.pth",
        "dest_dir": MODELS_DIR,
        "url": "https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth",
        "size_mb": 137.9,
        "expected_bytes": 136787426,
        "description": "SIGGRAPH 2017 Real-Time Image Colorization model by Richard Zhang et al."
    },
    "facexlib_detection": {
        "filename": "detection_Resnet50_Final.pth",
        "dest_dir": GFPGAN_WEIGHTS_DIR,
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "size_mb": 104.4,
        "expected_bytes": 109497761,
        "description": "ResNet50 face detector checkpoint used by facexlib/GFPGAN"
    },
    "facexlib_parsing": {
        "filename": "parsing_parsenet.pth",
        "dest_dir": GFPGAN_WEIGHTS_DIR,
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "size_mb": 81.4,
        "expected_bytes": 85331193,
        "description": "ParseNet facial parsing checkpoint used by facexlib/GFPGAN"
    }
}


def download_model(model_key: str, force: bool = False) -> Path:
    """
    Downloads model weights with streaming progress.
    
    Args:
        model_key: Key in MODEL_REGISTRY
        force: If True, re-downloads even if file exists.
        
    Returns:
        Path to the downloaded model weights file.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: '{model_key}'. Registered: {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[model_key]
    dest_dir = info.get("dest_dir", MODELS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / info["filename"]

    if target_path.exists() and not force:
        file_size = target_path.stat().st_size
        if file_size >= info.get("expected_bytes", 1000000) * 0.95:
            print(f"[CACHE] Model '{info['filename']}' already exists ({file_size / (1024*1024):.1f} MB). Skipping.")
            return target_path
        else:
            print(f"[WARN] Incomplete model file detected ({file_size} bytes). Re-downloading...")

    url = info["url"]
    print("=" * 70)
    print(f"DOWNLOADING: {info['filename']}")
    print(f"Description: {info['description']}")
    print(f"File Size:   ~{info['size_mb']} MB")
    print(f"Destination: {target_path}")
    print("=" * 70)

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB chunk

    temp_path = target_path.with_suffix(".tmp")
    with open(temp_path, "wb") as f, tqdm(
        desc=info["filename"],
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    if temp_path.exists():
        temp_path.replace(target_path)

    final_size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Download completed: {target_path.name} ({final_size_mb:.2f} MB)")
    return target_path


def is_model_available(model_key: str) -> bool:
    """Checks if a model file exists and is intact."""
    if model_key not in MODEL_REGISTRY:
        return False
    info = MODEL_REGISTRY[model_key]
    dest_dir = info.get("dest_dir", MODELS_DIR)
    target_path = dest_dir / info["filename"]
    return target_path.exists() and target_path.stat().st_size > 10000000


def download_all_models(force: bool = False):
    """Downloads all models in the registry."""
    print("Starting download of all required model weights...")
    for key in MODEL_REGISTRY:
        try:
            download_model(key, force=force)
        except Exception as e:
            print(f"[ERROR] Failed downloading {key}: {e}")


def list_models():
    """Lists all registered models and their cache status."""
    print(f"{'Model Key':<24} {'Filename':<35} {'Size':<10} {'Status'}")
    print("-" * 80)
    for key, info in MODEL_REGISTRY.items():
        avail = is_model_available(key)
        status = "Cached" if avail else "Missing"
        print(f"{key:<24} {info['filename']:<35} ~{info['size_mb']} MB   {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained model weights for Old Photo Restoration")
    parser.add_argument("--all", action="store_true", help="Download all model weights")
    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_REGISTRY.keys()), help="Download specific model")
    parser.add_argument("--list", action="store_true", help="List model status")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.all:
        download_all_models(force=args.force)
    elif args.model:
        download_model(args.model, force=args.force)
    else:
        list_models()
        print("\nUsage:")
        print("  python utils/model_downloader.py --all           # Download all weights")
        print("  python utils/model_downloader.py --model <name>  # Download specific model")
