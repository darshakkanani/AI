#!/usr/bin/env python3
"""
DHR Hair Removal (Black-Hat + Inpainting)
Follows the exact 4-step DHR algorithm:
1. Convert RGB -> grayscale
2. Apply morphological Black-Hat
3. Create mask for inpainting
4. Inpaint original image using mask

Usage:
    python dhr_hair_removal.py input.jpg -o output.png --show

Author: (generated)
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

def create_blackhat(enhanced_gray, kernel_size=(17, 17)):
    """
    Apply Black-Hat morphological transform to highlight dark hair on bright skin.
    kernel_size: tuple(width, height) -> typically elongated (e.g., (17,1) or (17,17))
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    blackhat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def build_mask_from_blackhat(blackhat, min_area=40, closing_iter=2, dilate_iter=2):
    """
    Convert blackhat result to binary mask suitable for inpainting.
    Steps:
      - Normalize and threshold (Otsu)
      - Morphological closing to join hair strands
      - Remove small components (noise)
      - Dilate to ensure full coverage for inpainting
    """
    # Normalize & threshold using Otsu
    # Ensure blackhat is uint8
    bh = blackhat.copy()
    if bh.dtype != np.uint8:
        bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to join broken hair segments
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=closing_iter)

    # Remove small connected components (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    # Slight dilation so inpainting covers whole hair width
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=dilate_iter)

    return cleaned

def dhr_remove_hair(img_bgr,
                    kernel_size=(17, 17),
                    clahe_clip=3.0,
                    clahe_grid=(8,8),
                    min_area=40,
                    closing_iter=2,
                    dilate_iter=2,
                    inpaint_radius=5,
                    inpaint_method='telea'):
    """
    Run DHR hair removal pipeline on a BGR image and return (result_bgr, mask, blackhat)
    Parameters:
      - kernel_size: morphological kernel size for black-hat. Tune (17,1) or (17,17) as needed.
      - clahe_clip, clahe_grid: for CLAHE on grayscale
      - min_area: remove components smaller than this (pixels)
      - closing_iter, dilate_iter: morphological iterations
      - inpaint_radius: passed to cv2.inpaint
      - inpaint_method: 'telea' or 'ns' (Navier-Stokes)
    """
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Improve contrast (helps black-hat highlight hairs)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    # 2) Apply Black-Hat transform
    blackhat = create_blackhat(enhanced, kernel_size=kernel_size)

    # 3) Create mask for inpainting
    mask = build_mask_from_blackhat(blackhat,
                                    min_area=min_area,
                                    closing_iter=closing_iter,
                                    dilate_iter=dilate_iter)

    # 4) Inpaint on original color image
    flags = cv2.INPAINT_TELEA if inpaint_method.lower().startswith('t') else cv2.INPAINT_NS
    inpainted = cv2.inpaint(img_bgr, mask, inpaint_radius, flags)

    return inpainted, mask, blackhat, enhanced

def parse_args(argv):
    p = argparse.ArgumentParser(description="DHR Hair Removal (Black-Hat + Inpainting)")
    p.add_argument("input", help="Input image path")
    p.add_argument("-o", "--output", help="Output image path (default: <input>_dhr.png)")
    p.add_argument("--mask", help="Save mask path (optional)", default=None)
    p.add_argument("--kernel", type=int, nargs=2, metavar=('W','H'), default=[17,17],
                   help="Black-hat kernel size W H (default 17 17). Try (17 1) to emphasize linear hairs.")
    p.add_argument("--inpaint-radius", type=int, default=5, help="Inpaint radius (default 5)")
    p.add_argument("--inpaint-method", choices=['telea','ns'], default='telea', help="Inpainting method")
    p.add_argument("--min-area", type=int, default=40, help="Minimum component area to keep in mask")
    p.add_argument("--show", action="store_true", help="Show intermediate steps (press any key to close)")
    p.add_argument("--save-intermediate", action="store_true", help="Save blackhat and enhanced grayscale (next to output)")
    return p.parse_args(argv)

def main(argv):
    args = parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"Error: input file {inp} not found")
        return 2

    out_path = args.output
    if not out_path:
        out_path = str(inp.parent / f"{inp.stem}_dhr{inp.suffix}")

    img = cv2.imread(str(inp))
    result, mask, blackhat, enhanced = dhr_remove_hair(
        img_bgr=img,
        kernel_size=tuple(args.kernel),
        inpaint_radius=args.inpaint_radius,
        inpaint_method=args.inpaint_method,
        min_area=args.min_area
    )

    # Save outputs
    cv2.imwrite(out_path, result)
    print(f"Saved inpainted image to: {out_path}")

    mask_path = args.mask or str(Path(out_path).with_name(Path(out_path).stem + "_mask.png"))
    cv2.imwrite(mask_path, mask)
    print(f"Saved mask to: {mask_path}")

    if args.save_intermediate:
        blackhat_path = str(Path(out_path).with_name(Path(out_path).stem + "_blackhat.png"))
        enhanced_path = str(Path(out_path).with_name(Path(out_path).stem + "_enhanced_gray.png"))
        cv2.imwrite(blackhat_path, blackhat)
        cv2.imwrite(enhanced_path, enhanced)
        print(f"Saved blackhat -> {blackhat_path}")
        print(f"Saved enhanced gray -> {enhanced_path}")

    if args.show:
        # Resize for display convenience if very large
        def disp(name, img_show):
            h = img_show.shape[0]
            max_h = 600
            if h > max_h:
                scale = max_h / h
                w = int(img_show.shape[1] * scale)
                return cv2.resize(img_show, (w, max_h))
            return img_show

        cv2.imshow("Original", disp("Original", img))
        cv2.imshow("Enhanced Gray (CLAHE)", disp("Enhanced", enhanced))
        cv2.imshow("Blackhat", disp("Blackhat", blackhat))
        cv2.imshow("Mask for Inpainting", disp("Mask", mask))
        cv2.imshow("Inpainted Result", disp("Result", result))
        print("Press any key in an image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
