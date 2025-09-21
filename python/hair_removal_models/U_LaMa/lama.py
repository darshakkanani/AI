import os
import sys
import subprocess
from pathlib import Path
import shutil
import zipfile
import cv2
import torch
from typing import Optional

# LaMa repo and weights management
BASE_DIR = Path(__file__).resolve().parent
LAMA_DIR = BASE_DIR / "lama"
WEIGHTS_DIR = BASE_DIR / "big-lama"
WEIGHTS_ZIP = BASE_DIR / "big-lama.zip"
HF_URL = "https://huggingface.co/saic-mdal/lama/resolve/main/big-lama.zip"


def ensure_lama_repo_and_weights():
    """Clone LaMa repo and download weights if missing."""
    if not LAMA_DIR.exists():
        subprocess.run(["git", "clone", "https://github.com/advimman/lama.git", str(LAMA_DIR)], check=True)
    if not WEIGHTS_DIR.exists():
        # download zip
        import requests
        with requests.get(HF_URL, stream=True) as r:
            r.raise_for_status()
            with open(WEIGHTS_ZIP, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # unzip
        with zipfile.ZipFile(WEIGHTS_ZIP, 'r') as zf:
            zf.extractall(BASE_DIR)
        WEIGHTS_ZIP.unlink(missing_ok=True)

    # add lama repo to sys.path
    if str(LAMA_DIR) not in sys.path:
        sys.path.append(str(LAMA_DIR))


def load_lama_predictor(device: Optional[str] = None):
    ensure_lama_repo_and_weights()
    from saicinpainting.evaluation.predictor import load_predictor
    from omegaconf import OmegaConf

    cfg_path = LAMA_DIR / "configs/prediction/default.yaml"
    ckpt_path = WEIGHTS_DIR
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = load_predictor(str(cfg_path), str(ckpt_path), device=device)
    return predictor


def inpaint_with_lama(img_path: str, mask_path: str, output_path: str = "final_output.png") -> str:
    predictor = load_lama_predictor()
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    # LaMa expects RGB and 0/255 mask
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if mask.shape[:2] != img_rgb.shape[:2]:
        mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    result = predictor({
        'image': img_rgb,
        'mask': mask
    })['inpainted']  # returns RGB

    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LaMa hair inpainting")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("mask", help="Path to binary hair mask image (0/255)")
    parser.add_argument("-o", "--output", default="final_output.png", help="Path to save inpainted result")
    args = parser.parse_args()

    out = inpaint_with_lama(args.input, args.mask, args.output)
    print(f"[INFO] Hair removed image saved as {out}")


if __name__ == "__main__":
    main()
