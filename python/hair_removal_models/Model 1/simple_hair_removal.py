#!/usr/bin/env python3
"""
Simple but Effective Hair Removal
Uses proven morphological operations and advanced inpainting
No complex dependencies - just OpenCV and NumPy
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

def remove_hair_simple(image_path, output_path=None, show_steps=False):
    """
    Simple but effective hair removal using morphological operations
    """
    print(f"Processing: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return None
    
    print(f"Image size: {img.shape}")
    
    # Step 1: Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Step 2: Multiple morphological operations to detect hairs
    print("Detecting hairs...")
    
    # Different kernel sizes and shapes for comprehensive hair detection
    kernels = [
        # Horizontal and vertical lines
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21)),
        # Elliptical kernels for curved hairs
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
    ]
    
    # Apply black-hat transform with all kernels
    blackhat_results = []
    for kernel in kernels:
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        blackhat_results.append(blackhat)
    
    # Combine all results
    combined = np.maximum.reduce(blackhat_results)
    
    # Step 3: Create mask with aggressive thresholding
    print("Creating mask...")
    
    # Use OTSU thresholding for automatic threshold selection
    _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # If mask is too sparse, use a lower threshold
    if np.count_nonzero(mask) < 0.001 * mask.size:
        # Use percentile-based threshold
        non_zero = combined[combined > 0]
        if len(non_zero) > 0:
            threshold = np.percentile(non_zero, 50)  # More aggressive
            _, mask = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Close gaps in hair strands
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate to ensure complete coverage
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask = cv2.dilate(mask, kernel_dilate, iterations=2)
    
    # Step 4: Advanced inpainting
    print("Inpainting...")
    
    # Use multiple inpainting methods and blend them
    inpaint1 = cv2.inpaint(img, mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)
    inpaint2 = cv2.inpaint(img, mask, inpaintRadius=12, flags=cv2.INPAINT_NS)
    
    # Blend the two results for better quality
    # TELEA is better for textured areas, NS for smooth areas
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_orig, cv2.CV_64F)
    texture_score = np.abs(laplacian)
    texture_threshold = np.percentile(texture_score, 70)
    
    # Create blending weights
    smooth_regions = (texture_score < texture_threshold).astype(np.float32)
    smooth_regions = cv2.GaussianBlur(smooth_regions, (5, 5), 0)
    
    # Blend the results
    result = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        result[:,:,c] = (
            smooth_regions * inpaint2[:,:,c] +  # NS for smooth areas
            (1 - smooth_regions) * inpaint1[:,:,c]  # TELEA for textured areas
        )
    
    result = result.astype(np.uint8)
    
    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"Result saved to: {output_path}")
        
        # Save mask for inspection
        mask_path = str(Path(output_path).with_name(Path(output_path).stem + "_mask.png"))
        cv2.imwrite(mask_path, mask)
        print(f"Mask saved to: {mask_path}")
        
        # Save intermediate steps
        gray_path = str(Path(output_path).with_name(Path(output_path).stem + "_gray.png"))
        cv2.imwrite(gray_path, enhanced)
        
        blackhat_path = str(Path(output_path).with_name(Path(output_path).stem + "_blackhat.png"))
        cv2.imwrite(blackhat_path, combined)
    
    # Display results if requested
    if show_steps:
        # Resize for display if too large
        display_height = 400
        h, w = img.shape[:2]
        if h > display_height:
            scale = display_height / h
            display_width = int(w * scale)
            
            img_display = cv2.resize(img, (display_width, display_height))
            enhanced_display = cv2.resize(enhanced, (display_width, display_height))
            combined_display = cv2.resize(combined, (display_width, display_height))
            mask_display = cv2.resize(mask, (display_width, display_height))
            result_display = cv2.resize(result, (display_width, display_height))
        else:
            img_display = img
            enhanced_display = enhanced
            combined_display = combined
            mask_display = mask
            result_display = result
        
        cv2.imshow("Original", img_display)
        cv2.imshow("Enhanced Grayscale", enhanced_display)
        cv2.imshow("Hair Detection", combined_display)
        cv2.imshow("Mask", mask_display)
        cv2.imshow("Final Result", result_display)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("Hair removal completed successfully!")
    return result

def main():
    parser = argparse.ArgumentParser(description="Simple Hair Removal")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("--display", action="store_true", help="Display results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_hair_removed{input_path.suffix}")
    
    remove_hair_simple(args.input, args.output, args.display)

if __name__ == "__main__":
    main()
