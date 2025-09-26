#!/usr/bin/env python3
"""
Enhanced Hair Removal Model
- Multi-scale black-hat + edge-based detection
- Adaptive mask refinement with connected components
- Progressive inpainting and smoothing
"""

import cv2
import numpy as np
from pathlib import Path
import argparse, os

def enhanced_hair_removal(image_path, output_path=None, show_steps=False):
    print(f"Processing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # --------------------------
    # STEP 1: Multi-scale black-hat hair detection
    # --------------------------
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (9,1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1,9)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (17,1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1,17)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)),
    ]
    blackhat_results = [cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, k) for k in kernels]
    blackhat_combined = np.maximum.reduce(blackhat_results)

    # --------------------------
    # STEP 2: Edge-based refinement (captures faint hairs)
    # --------------------------
    edges = cv2.Canny(enhanced, 30, 100)
    edges = cv2.dilate(edges, None, iterations=1)
    combined = cv2.addWeighted(blackhat_combined, 0.7, edges, 0.3, 0)

    # --------------------------
    # STEP 3: Mask creation + refinement
    # --------------------------
    _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only connected components (remove noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    refined_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 20:  # keep only real strands
            refined_mask[labels == i] = 255

    # Dilate for stronger coverage
    refined_mask = cv2.dilate(refined_mask, kernel, iterations=2)

    # --------------------------
    # STEP 4: Progressive Inpainting
    # --------------------------
    inpaint_telea = cv2.inpaint(img, refined_mask, 9, cv2.INPAINT_TELEA)
    inpaint_ns = cv2.inpaint(inpaint_telea, refined_mask, 11, cv2.INPAINT_NS)

    # Optional smoothing to remove harsh transitions
    final = cv2.bilateralFilter(inpaint_ns, 9, 50, 50)

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, final)
        cv2.imwrite(str(Path(output_path).with_name(Path(output_path).stem+"_mask.png")), refined_mask)

    # Show steps
    if show_steps:
        cv2.imshow("Original", img)
        cv2.imshow("Enhanced Gray", enhanced)
        cv2.imshow("Blackhat", blackhat_combined)
        cv2.imshow("Edges", edges)
        cv2.imshow("Mask", refined_mask)
        cv2.imshow("Result", final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Enhanced hair removal completed!")
    return final

def main():
    parser = argparse.ArgumentParser(description="Enhanced Hair Removal")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--display", action="store_true", help="Show steps")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("Error: File does not exist")
        return

    if not args.output:
        inp = Path(args.input)
        args.output = str(inp.parent / f"{inp.stem}_hairfree{inp.suffix}")

    enhanced_hair_removal(args.input, args.output, args.display)

if __name__ == "__main__":
    main()
