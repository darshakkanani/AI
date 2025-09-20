#!/usr/bin/env python3
"""
Most Accurate Hair Removal Model
- Optimized Black-Hat morphological operations for precise hair detection
- Advanced multi-method inpainting for natural hair removal
- Maximum accuracy output with professional quality
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
from PIL import Image
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccurateHairRemover:
    """Most accurate hair removal using optimized black-hat and inpainting"""
    
    def __init__(self):
        logger.info("Most Accurate Hair Removal Model initialized")
        
    def detect_hair_blackhat(self, image: np.ndarray) -> np.ndarray:
        """Optimized black-hat morphological operations for maximum hair detection accuracy"""
        logger.info("Applying optimized black-hat hair detection...")
        
        # Convert to optimal color spaces for hair detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Collection of hair masks from different methods
        hair_masks = []
        
        # Method 1: Multi-scale rectangular kernels (best for straight hairs)
        logger.info("Detecting straight hair patterns...")
        for size in [7, 11, 15, 19, 23]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            hair_masks.append(blackhat)
        
        # Method 2: Multi-scale elliptical kernels (best for curved hairs)
        logger.info("Detecting curved hair patterns...")
        for size in [5, 9, 13, 17]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            hair_masks.append(blackhat)
        
        # Method 3: LAB L-channel black-hat (most sensitive to hair)
        logger.info("Analyzing L-channel for fine hair detection...")
        l_channel = lab[:,:,0]
        for size in [5, 9, 13, 17, 21]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            l_blackhat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel)
            hair_masks.append(l_blackhat)
        
        # Method 4: Directional kernels for hair strand orientation
        logger.info("Detecting directional hair strands...")
        
        # Horizontal hair strands
        for width in [11, 15, 19]:
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
            hair_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
            hair_masks.append(hair_h)
        
        # Vertical hair strands  
        for height in [11, 15, 19]:
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
            hair_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v)
            hair_masks.append(hair_v)
        
        # Diagonal hair strands (45-degree angles)
        for size in [7, 11, 15]:
            # Create diagonal kernels
            kernel_d1 = np.eye(size, dtype=np.uint8)
            kernel_d2 = np.fliplr(np.eye(size, dtype=np.uint8))
            
            hair_d1 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_d1)
            hair_d2 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_d2)
            hair_masks.extend([hair_d1, hair_d2])
        
        # Method 5: Cross-shaped kernels for intersection detection
        logger.info("Detecting hair intersections...")
        for size in [7, 11]:
            kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
            hair_cross = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_cross)
            hair_masks.append(hair_cross)
        
        # Combine all masks using maximum operation for best coverage
        logger.info("Combining all detection results...")
        combined_mask = np.maximum.reduce(hair_masks).astype(np.float32)
        
        # Apply Gaussian blur to smooth the combined mask
        combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 0)
        
        return combined_mask
    
    def refine_hair_mask(self, mask: np.ndarray) -> np.ndarray:
        """Precise mask refinement for maximum accuracy"""
        logger.info("Refining hair mask for maximum accuracy...")
        
        # Adaptive thresholding for optimal hair selection
        non_zero_pixels = mask[mask > 0]
        if len(non_zero_pixels) > 0:
            # Use 80th percentile for aggressive but accurate detection
            threshold = np.percentile(non_zero_pixels, 80)
        else:
            threshold = np.mean(mask) + 1.5 * np.std(mask)
        
        # Apply threshold
        _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        
        # Morphological operations for clean mask
        # Close small gaps in hair strands
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Remove tiny components (noise)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask)
        min_area = 12  # Minimum area for a hair strand
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                refined_mask[labels == i] = 0
        
        # Slight dilation to ensure complete hair coverage
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        final_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=1)
        
        return final_mask
    
    def advanced_inpainting(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Most accurate inpainting using multiple methods for natural results"""
        logger.info("Applying advanced multi-method inpainting...")
        
        # Method 1: Fast Marching Method (TELEA) - excellent for textures
        inpainted_telea = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Method 2: Navier-Stokes (NS) - excellent for smooth regions
        inpainted_ns = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        
        # Method 3: Enhanced TELEA with larger radius for complex areas
        inpainted_telea_large = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        
        # Analyze image texture to blend methods optimally
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local texture using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_map = np.abs(laplacian)
        
        # Smooth the texture map
        texture_map = cv2.GaussianBlur(texture_map, (5, 5), 0)
        
        # Create blending weights based on texture
        texture_threshold_low = np.percentile(texture_map, 30)
        texture_threshold_high = np.percentile(texture_map, 70)
        
        # Normalize texture map to [0, 1]
        texture_norm = np.clip((texture_map - texture_threshold_low) / 
                              (texture_threshold_high - texture_threshold_low), 0, 1)
        
        # Blend the three inpainting results
        result = np.zeros_like(image, dtype=np.float32)
        
        for c in range(3):
            # Smooth areas: more NS
            # Medium texture: blend of NS and TELEA
            # High texture: more TELEA with large radius
            result[:,:,c] = (
                (1 - texture_norm) * (0.7 * inpainted_ns[:,:,c] + 0.3 * inpainted_telea[:,:,c]) +
                texture_norm * (0.6 * inpainted_telea_large[:,:,c] + 0.4 * inpainted_telea[:,:,c])
            )
        
        # Final smoothing only in inpainted regions
        final_result = result.astype(np.uint8)
        
        # Apply bilateral filter only to inpainted areas for natural blending
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
        smoothed = cv2.bilateralFilter(final_result, 9, 75, 75)
        
        # Blend smoothed result only in hair regions
        final_result = (mask_3channel * smoothed + (1 - mask_3channel) * image).astype(np.uint8)
        
        return final_result
    
    def remove_hair(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Complete accurate hair removal pipeline"""
        logger.info("Starting most accurate hair removal process...")
        
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Step 1: Optimized black-hat hair detection
        hair_mask_raw = self.detect_hair_blackhat(img_array)
        
        # Step 2: Precise mask refinement
        hair_mask = self.refine_hair_mask(hair_mask_raw)
        
        # Step 3: Advanced multi-method inpainting
        inpainted_result = self.advanced_inpainting(img_array, hair_mask)
        
        # Convert back to PIL Images
        result_image = Image.fromarray(inpainted_result)
        mask_image = Image.fromarray(hair_mask)
        
        logger.info("Most accurate hair removal completed successfully")
        return result_image, mask_image
    
    def process_image(self, input_path: str, output_path: str, save_mask: bool = True) -> bool:
        """Process a single image with maximum accuracy"""
        try:
            logger.info(f"Loading image: {input_path}")
            image = Image.open(input_path)
            original_size = image.size
            logger.info(f"Processing image size: {original_size}")
            
            # Apply most accurate hair removal
            clean_image, hair_mask = self.remove_hair(image)
            
            # Save results with maximum quality
            clean_image.save(output_path, quality=98, optimize=True)
            logger.info(f"Most accurate clean skin image saved: {output_path}")
            
            if save_mask:
                base, ext = os.path.splitext(output_path)
                mask_path = f"{base}_hair_mask{ext}"
                hair_mask.save(mask_path, quality=98)
                logger.info(f"Hair detection mask saved: {mask_path}")
            
            # Memory cleanup
            del image, clean_image, hair_mask
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False
    
    def process_batch(self, input_dir: str, output_dir: str, save_masks: bool = True) -> int:
        """Process multiple images with maximum accuracy"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        
        if not image_files:
            logger.error(f"No images found in {input_dir}")
            return 0
        
        logger.info(f"Processing {len(image_files)} images with maximum accuracy...")
        
        success_count = 0
        for i, img_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {img_file.name}")
            output_file = output_path / img_file.name
            if self.process_image(str(img_file), str(output_file), save_masks):
                success_count += 1
        
        logger.info(f"Successfully processed {success_count}/{len(image_files)} images with maximum accuracy")
        return success_count

def main():
    parser = argparse.ArgumentParser(description="Most Accurate Hair Removal Model")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("output", help="Output image file or directory")
    parser.add_argument("--no-mask", action="store_true", help="Don't save hair detection mask")
    
    args = parser.parse_args()
    
    hair_remover = AccurateHairRemover()
    
    input_path = Path(args.input)
    save_masks = not args.no_mask
    
    if input_path.is_file():
        success = hair_remover.process_image(args.input, args.output, save_masks)
        return 0 if success else 1
    elif input_path.is_dir():
        success_count = hair_remover.process_batch(args.input, args.output, save_masks)
        return 0 if success_count > 0 else 1
    else:
        logger.error(f"Input path not found: {args.input}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
