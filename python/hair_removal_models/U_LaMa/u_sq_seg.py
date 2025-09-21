import os
import sys
import subprocess
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import gdown

# Paths
BASE_DIR = Path(__file__).resolve().parent
U2NET_DIR = BASE_DIR / "U-2-Net"
U2NET_WEIGHTS = BASE_DIR / "u2net.pth"      # full-size model
U2NETP_WEIGHTS = BASE_DIR / "u2netp.pth"     # small model
# Google Drive IDs from original repo
U2NET_WEIGHTS_ID = "1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy"   # u2net.pth
U2NETP_WEIGHTS_ID = "1H-yKMFVYv-nyBEWf0r6U-8bU4ZCOZBNz"  # u2netp.pth


def ensure_u2net_repo():
    if not U2NET_DIR.exists():
        subprocess.run(["git", "clone", "https://github.com/xuebinqin/U-2-Net.git", str(U2NET_DIR)], check=True)
    if str(U2NET_DIR) not in sys.path:
        sys.path.append(str(U2NET_DIR))


def download_weights_if_missing():
    if not U2NET_WEIGHTS.exists():
        url = f"https://drive.google.com/uc?id={U2NET_WEIGHTS_ID}"
        gdown.download(url, str(U2NET_WEIGHTS), quiet=False)
    # don't always download u2netp; only if needed


def load_u2net_any():
    """Try loading U2NET with u2net.pth; if size mismatch, fall back to U2NETP with u2netp.pth."""
    ensure_u2net_repo()
    download_weights_if_missing()
    # Import models after repo available
    from model import U2NET, U2NETP

    # Try full U2NET first
    net = U2NET(3, 1)
    try:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(str(U2NET_WEIGHTS)))
            net.cuda()
        else:
            net.load_state_dict(torch.load(str(U2NET_WEIGHTS), map_location="cpu"))
        net.eval()
        return net
    except RuntimeError as e:
        # Likely size mismatch -> try U2NETP
        print(f"[WARN] Could not load U2NET with u2net.pth ({e}). Trying U2NETP...")
        if not U2NETP_WEIGHTS.exists():
            url = f"https://drive.google.com/uc?id={U2NETP_WEIGHTS_ID}"
            gdown.download(url, str(U2NETP_WEIGHTS), quiet=False)
        netp = U2NETP(3, 1)
        if torch.cuda.is_available():
            netp.load_state_dict(torch.load(str(U2NETP_WEIGHTS)))
            netp.cuda()
        else:
            netp.load_state_dict(torch.load(str(U2NETP_WEIGHTS), map_location="cpu"))
        netp.eval()
        return netp


def segment_hair(img_path, output_mask="hair_mask.png", input_size=320, threshold=0.5):
    """Run U²-Net (or U²-NetP) to generate a binary hair mask for an input image path."""
    net = load_u2net_any()

    image = Image.open(img_path).convert("RGB")
    w0, h0 = image.size
    im = np.array(image).astype(np.float32) / 255.0
    im_res = cv2.resize(im, (input_size, input_size), interpolation=cv2.INTER_AREA)
    im_res = im_res.transpose((2, 0, 1))  # CHW
    tens = torch.from_numpy(im_res).unsqueeze(0).float()

    if torch.cuda.is_available():
        tens = tens.cuda()
        net = net.cuda()

    with torch.no_grad():
        d1, *_ = net(tens)
        pred = d1[:, 0, :, :]
        pred = torch.sigmoid(pred)  # ensure probabilities
        mask_small = (pred > threshold).float().cpu().numpy().squeeze()

    mask = (cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8) * 255
    cv2.imwrite(output_mask, mask)
    return output_mask


def main():
    import argparse
    parser = argparse.ArgumentParser(description="U²-Net Hair Segmentation")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("-o", "--output", default="hair_mask.png", help="Path to save hair mask (PNG)")
    parser.add_argument("--size", type=int, default=320, help="Input size for network (default: 320)")
    parser.add_argument("--thr", type=float, default=0.5, help="Threshold for mask binarization (default: 0.5)")
    args = parser.parse_args()

    out = segment_hair(args.input, args.output, input_size=args.size, threshold=args.thr)
    print(f"[INFO] Hair mask saved at: {out}")


if __name__ == "__main__":
    main()
