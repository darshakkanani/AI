#!/usr/bin/env python3
"""
Advanced Hair Removal Model
- Multi-scale hair detection using morphological operations
- Advanced inpainting for natural skin restoration
- Edge-preserving smoothing
- Skin tone analysis and enhancement
- Memory-efficient processing
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedHairRemover:
    """Advanced hair removal model for clear skin visibility"""
    
    def __init__(self):
        logger.info("Advanced Hair Removal Model initialized")
        
    def detect_hair_multiscale(self, image: np.ndarray) -> np.ndarray:
        """Multi-scale hair detection using advanced morphological operations"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Initialize mask collection
        hair_masks = []
        
        # Method 1: Multi-scale black-hat operations on grayscale
        logger.info("Applying multi-scale morphological detection...")
        for kernel_size in [7, 11, 15, 19]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            hair_masks.append(blackhat)
        
        # Method 2: L-channel analysis (best for hair detection)
        l_channel = lab[:,:,0]
        for kernel_size in [5, 9, 13]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            l_blackhat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel)
            hair_masks.append(l_blackhat)
        
        # Method 3: V-channel analysis (value channel in HSV)
        v_channel = hsv[:,:,2]
        inverted_v = 255 - v_channel
        for kernel_size in [3, 7, 11]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            v_blackhat = cv2.morphologyEx(inverted_v, cv2.MORPH_BLACKHAT, kernel)
            hair_masks.append(v_blackhat)
        
        # Method 4: Directional filters for hair strands
        logger.info("Applying directional hair detection...")
        # Horizontal hair detection
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        hair_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
        hair_masks.append(hair_h)
        
        # Vertical hair detection
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        hair_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v)
        hair_masks.append(hair_v)
        
        # Diagonal hair detection
        kernel_d1 = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]], dtype=np.uint8)
        kernel_d2 = np.array([[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0]], dtype=np.uint8)
        hair_d1 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_d1)
        hair_d2 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_d2)
        hair_masks.extend([hair_d1, hair_d2])
        
        # Combine all masks using maximum operation
        logger.info("Combining detection results...")
        combined_mask = np.maximum.reduce(hair_masks).astype(np.float32)
        
        return combined_mask
    
    def refine_hair_mask(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Advanced mask refinement using adaptive thresholding and morphology"""
        logger.info("Refining hair mask...")
        
        # Adaptive thresholding based on mask statistics
        non_zero_mask = mask[mask > 0]
        if len(non_zero_mask) > 0:
            # Use 85th percentile for aggressive hair detection
            threshold_val = np.percentile(non_zero_mask, 85)
        else:
            threshold_val = np.mean(mask) + 2 * np.std(mask)
        
        # Apply threshold
        _, binary_mask = cv2.threshold(mask, threshold_val, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        
        # Morphological refinement
        # Close small gaps in hair strands
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Remove noise (small isolated pixels)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Remove very small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask)
        min_area = 15  # Minimum hair strand area
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                refined_mask[labels == i] = 0
        
        # Dilate slightly to ensure complete hair coverage
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        refined_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=1)
        
        return refined_mask
    
    def advanced_inpainting(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Advanced inpainting with skin-aware restoration"""
        logger.info("Applying advanced skin-aware inpainting...")
        
        # Method 1: Fast Marching Method (good for texture)
        inpainted_fm = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        # Method 2: Navier-Stokes based (good for smooth areas)
        inpainted_ns = cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)
        
        # Combine both methods for best results
        # Use NS for smooth skin areas, FM for textured areas
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect smooth vs textured regions
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_score = np.abs(laplacian)
        texture_threshold = np.percentile(texture_score, 70)
        
        # Create blending mask
        smooth_regions = (texture_score < texture_threshold).astype(np.float32)
        smooth_regions = cv2.GaussianBlur(smooth_regions, (5, 5), 0)
        
        # Blend the two inpainting results
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            result[:,:,c] = (smooth_regions * inpainted_ns[:,:,c] + 
                           (1 - smooth_regions) * inpainted_fm[:,:,c])
        
        return result.astype(np.uint8)
    
    def enhance_skin_clarity(self, image: np.ndarray) -> np.ndarray:
        """Enhance skin clarity and smoothness"""
        logger.info("Enhancing skin clarity...")
        
        # Convert to different color spaces for analysis
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Skin tone enhancement in LAB space
        l, a, b = cv2.split(lab)
        
        # Enhance L channel (lightness) for better skin appearance
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Slight smoothing of A and B channels for more natural skin tones
        a_smooth = cv2.bilateralFilter(a, 9, 75, 75)
        b_smooth = cv2.bilateralFilter(b, 9, 75, 75)
        
        # Reconstruct LAB image
        enhanced_lab = cv2.merge([l_enhanced, a_smooth, b_smooth])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Apply subtle skin smoothing
        # Use bilateral filter to smooth while preserving edges
        smooth_skin = cv2.bilateralFilter(enhanced_rgb, 15, 80, 80)
        
        # Blend original and smoothed for natural look
        alpha = 0.3  # 30% smoothing
        final_result = cv2.addWeighted(enhanced_rgb, 1-alpha, smooth_skin, alpha, 0)
        
        return final_result
    
    def remove_hair(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Complete hair removal pipeline"""
        logger.info("Starting advanced hair removal process...")
        
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Step 1: Multi-scale hair detection
        hair_mask_raw = self.detect_hair_multiscale(img_array)
        
        # Step 2: Refine hair mask
        hair_mask = self.refine_hair_mask(hair_mask_raw, img_array)
        
        # Step 3: Advanced inpainting
        inpainted = self.advanced_inpainting(img_array, hair_mask)
        
        # Step 4: Enhance skin clarity
        final_result = self.enhance_skin_clarity(inpainted)
        
        # Convert back to PIL Images
        result_image = Image.fromarray(final_result)
        mask_image = Image.fromarray(hair_mask)
        
        logger.info("Hair removal completed successfully")
        return result_image, mask_image
    
    def process_image(self, input_path: str, output_path: str, save_mask: bool = True) -> bool:
        """Process a single image for hair removal"""
        try:
            logger.info(f"Loading image: {input_path}")
            image = Image.open(input_path)
            original_size = image.size
            logger.info(f"Processing image size: {original_size}")
            
            # Apply hair removal
            clean_image, hair_mask = self.remove_hair(image)
            
            # Save results
            clean_image.save(output_path, quality=95, optimize=True)
            logger.info(f"Clean skin image saved: {output_path}")
            
            if save_mask:
                base, ext = os.path.splitext(output_path)
                mask_path = f"{base}_hair_mask{ext}"
                hair_mask.save(mask_path)
                logger.info(f"Hair mask saved: {mask_path}")
            
            # Memory cleanup
            del image, clean_image, hair_mask
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False
    
    def process_batch(self, input_dir: str, output_dir: str, save_masks: bool = True) -> int:
        """Process multiple images"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        
        if not image_files:
            logger.error(f"No images found in {input_dir}")
            return 0
        
        logger.info(f"Processing {len(image_files)} images for hair removal...")
        
        success_count = 0
        for i, img_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {img_file.name}")
            output_file = output_path / img_file.name
            if self.process_image(str(img_file), str(output_file), save_masks):
                success_count += 1
        
        logger.info(f"Successfully processed {success_count}/{len(image_files)} images")
        return success_count

def main():
    parser = argparse.ArgumentParser(description="Advanced Hair Removal for Clear Skin")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("output", help="Output image file or directory")
    parser.add_argument("--no-mask", action="store_true", help="Don't save hair mask")
    
    args = parser.parse_args()
    
    hair_remover = AdvancedHairRemover()
    
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
