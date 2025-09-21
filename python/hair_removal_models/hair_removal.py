#!/usr/bin/env python3
"""
Enhanced Hair Removal with Maximum Accuracy
Following the 4-step process:
1. Convert RGB images into grayscale images
2. Apply Morphological Black-Hat transformation on the grayscale images
3. Create a mask for the inpainting task
4. Apply inpainting algorithm on the original image using this mask

@author: Enhanced by AI Assistant
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

def convert_to_grayscale(image):
    """
    Step 1: Convert RGB images into grayscale images
    """
    print("Step 1: Converting RGB to grayscale...")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale

def apply_blackhat_transformation(grayscale_image):
    """
    Step 2: Apply Morphological Black-Hat transformation on the grayscale images
    Enhanced with multiple kernel sizes for maximum accuracy
    """
    print("Step 2: Applying Morphological Black-Hat transformation...")
    
    # Multiple kernel sizes for comprehensive hair detection
    hair_masks = []
    
    # Different kernel shapes and sizes for better hair detection
    kernel_sizes = [5, 7, 9, 11, 13, 15, 17, 19]
    
    for size in kernel_sizes:
        # Rectangular kernel for straight hairs
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        blackhat_rect = cv2.morphologyEx(grayscale_image, cv2.MORPH_BLACKHAT, kernel_rect)
        hair_masks.append(blackhat_rect)
        
        # Elliptical kernel for curved hairs
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        blackhat_ellipse = cv2.morphologyEx(grayscale_image, cv2.MORPH_BLACKHAT, kernel_ellipse)
        hair_masks.append(blackhat_ellipse)
    
    # Directional kernels for hair strands
    for width in [11, 15, 19]:
        # Horizontal hair detection
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
        blackhat_h = cv2.morphologyEx(grayscale_image, cv2.MORPH_BLACKHAT, kernel_h)
        hair_masks.append(blackhat_h)
        
        # Vertical hair detection
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, width))
        blackhat_v = cv2.morphologyEx(grayscale_image, cv2.MORPH_BLACKHAT, kernel_v)
        hair_masks.append(blackhat_v)
    
    # Combine all black-hat results for maximum coverage
    combined_blackhat = np.maximum.reduce(hair_masks)
    
    # Apply Gaussian blur for smoother results
    blurred_blackhat = cv2.GaussianBlur(combined_blackhat, (3, 3), 0)
    
    return blurred_blackhat

def create_inpainting_mask(blackhat_image):
    """
    Step 3: Create a mask for the inpainting task
    Enhanced mask creation with adaptive thresholding
    """
    print("Step 3: Creating mask for inpainting...")
    
    # Adaptive thresholding for better mask creation
    non_zero_pixels = blackhat_image[blackhat_image > 0]
    if len(non_zero_pixels) > 0:
        # Use 75th percentile for aggressive hair detection
        threshold_value = np.percentile(non_zero_pixels, 75)
    else:
        threshold_value = 15  # Default threshold
    
    # Create binary mask
    _, binary_mask = cv2.threshold(blackhat_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to refine the mask
    # Close small gaps in hair strands
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Remove very small components (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask)
    min_area = 10  # Minimum area for a hair strand
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            refined_mask[labels == i] = 0
    
    # Slight dilation to ensure complete hair coverage
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=1)
    
    return final_mask

def apply_inpainting_algorithm(original_image, mask):
    """
    Step 4: Apply inpainting algorithm on the original image using this mask
    Enhanced with multiple inpainting methods for best results
    """
    print("Step 4: Applying inpainting algorithm...")
    
    # Method 1: Fast Marching Method (TELEA) - good for textures
    inpainted_telea = cv2.inpaint(original_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Method 2: Navier-Stokes based method - good for smooth areas
    inpainted_ns = cv2.inpaint(original_image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    
    # Method 3: Enhanced TELEA with larger radius
    inpainted_telea_large = cv2.inpaint(original_image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    
    # Analyze image texture to choose best method for each region
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = np.abs(laplacian)
    texture_threshold = np.percentile(texture_score, 70)
    
    # Create blending mask based on texture
    smooth_regions = (texture_score < texture_threshold).astype(np.float32)
    smooth_regions = cv2.GaussianBlur(smooth_regions, (5, 5), 0)
    
    # Blend the inpainting results based on texture
    result = np.zeros_like(original_image, dtype=np.float32)
    for c in range(3):
        result[:,:,c] = (
            smooth_regions * inpainted_ns[:,:,c] +
            (1 - smooth_regions) * (0.6 * inpainted_telea[:,:,c] + 0.4 * inpainted_telea_large[:,:,c])
        )
    
    return result.astype(np.uint8)

def process_hair_removal(image_path, output_path=None, show_steps=True):
    """
    Complete 4-step hair removal process with maximum accuracy
    """
    print(f"Processing: {image_path}")
    
    # Load the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    print(f"Original image size: {original_image.shape}")
    
    # Step 1: Convert RGB images into grayscale images
    grayscale_image = convert_to_grayscale(original_image)
    
    # Step 2: Apply Morphological Black-Hat transformation
    blackhat_result = apply_blackhat_transformation(grayscale_image)
    
    # Step 3: Create a mask for the inpainting task
    inpainting_mask = create_inpainting_mask(blackhat_result)
    
    # Step 4: Apply inpainting algorithm
    final_result = apply_inpainting_algorithm(original_image, inpainting_mask)
    
    # Save the result if output path is provided
    if output_path:
        cv2.imwrite(output_path, final_result)
        print(f"Result saved to: {output_path}")
        
        # Also save the mask for reference
        mask_path = output_path.replace('.jpg', '_mask.jpg').replace('.png', '_mask.png')
        cv2.imwrite(mask_path, inpainting_mask)
        print(f"Mask saved to: {mask_path}")
    
    # Display results if requested
    if show_steps:
        # Resize images for display (if too large)
        display_height = 400
        h, w = original_image.shape[:2]
        if h > display_height:
            scale = display_height / h
            display_width = int(w * scale)
            
            original_display = cv2.resize(original_image, (display_width, display_height))
            grayscale_display = cv2.resize(grayscale_image, (display_width, display_height))
            blackhat_display = cv2.resize(blackhat_result, (display_width, display_height))
            mask_display = cv2.resize(inpainting_mask, (display_width, display_height))
            result_display = cv2.resize(final_result, (display_width, display_height))
        else:
            original_display = original_image
            grayscale_display = grayscale_image
            blackhat_display = blackhat_result
            mask_display = inpainting_mask
            result_display = final_result
        
        # Display all steps
        cv2.imshow("Step 0: Original Image", original_display)
        cv2.imshow("Step 1: Grayscale", grayscale_display)
        cv2.imshow("Step 2: Black-Hat Result", blackhat_display)
        cv2.imshow("Step 3: Inpainting Mask", mask_display)
        cv2.imshow("Step 4: Final Result", result_display)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return final_result

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description="Enhanced Hair Removal - 4 Step Process")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("--no-display", action="store_true", help="Don't display step-by-step results")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Set default output path if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_hair_removed{input_path.suffix}")
    
    # Process the image
    result = process_hair_removal(
        image_path=args.input,
        output_path=args.output,
        show_steps=not args.no_display
    )
    
    if result is not None:
        print("Hair removal completed successfully!")
    else:
        print("Hair removal failed!")

if __name__ == "__main__":
    # If run directly, process the default image
    default_path = '/Users/hunter/Desktop/Project/AI/Dataset/Dataset_hair_training_testing/A skin lesion hair mask dataset with fine-grained annotations/dermoscopic_image/ISIC_0000032.png'
    if os.path.exists(default_path):
        print("Processing default image...")
        process_hair_removal(default_path, 
                           output_path='/Users/hunter/Desktop/Project/AI/image/hair_removed_image/IMG_0341.jpg')
    else:
        main()