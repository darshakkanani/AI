#!/usr/bin/env python3
"""
Simple but Effective AI Image Resizer
- Smart multi-step resizing for best quality
- Edge-preserving smoothing
- Adaptive sharpening
- Lightweight and fast
- Memory-efficient for large images
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartImageResizer:
    """Simple but effective image resizer with smart enhancement"""
    
    def __init__(self, target_size: Tuple[int, int] = (380, 380)):
        self.target_size = target_size
        logger.info(f"Smart resizer initialized for {target_size}")

    def smart_resize(self, image: Image.Image) -> Image.Image:
        """Memory-efficient smart multi-step resizing for best quality"""
        original_size = image.size
        target_w, target_h = self.target_size
        
        logger.info(f"Processing {original_size} -> {self.target_size}")
        
        # Memory optimization: work with smaller chunks for very large images
        if original_size[0] * original_size[1] > 10_000_000:  # > 10MP
            logger.info("Large image detected - using memory-efficient processing")
            # For very large images, resize in multiple steps to save memory
            return self._resize_large_image(image)
        
        # Convert to high-quality array
        img_array = np.array(image.convert('RGB')).astype(np.float32)
        
        # Step 1: Smart interpolation method selection
        scale_factor = min(target_w / original_size[0], target_h / original_size[1])
        
        if scale_factor > 1.5:
            # Upscaling - use CUBIC for smoothness
            interpolation = cv2.INTER_CUBIC
        elif scale_factor < 0.5:
            # Heavy downscaling - use AREA for best quality
            interpolation = cv2.INTER_AREA
        else:
            # Normal scaling - use LANCZOS for sharpness
            interpolation = cv2.INTER_LANCZOS4
        
        # Step 2: Multi-step resizing for large scale changes
        current_size = original_size
        current_img = img_array
        
        while True:
            # Calculate intermediate size
            scale_x = target_w / current_size[0]
            scale_y = target_h / current_size[1]
            
            # If we're close to target, resize directly
            if 0.5 <= min(scale_x, scale_y) <= 2.0:
                final_img = cv2.resize(current_img, self.target_size, interpolation=interpolation)
                break
            
            # Otherwise, resize by factor of 2
            if scale_x > 2 or scale_y > 2:
                # Upscale by 2x
                new_size = (current_size[0] * 2, current_size[1] * 2)
                current_img = cv2.resize(current_img, new_size, interpolation=cv2.INTER_CUBIC)
            else:
                # Downscale by 2x
                new_size = (current_size[0] // 2, current_size[1] // 2)
                current_img = cv2.resize(current_img, new_size, interpolation=cv2.INTER_AREA)
            
            current_size = new_size
            
            # Memory cleanup
            gc.collect()
        
        return Image.fromarray(np.clip(final_img, 0, 255).astype(np.uint8))

    def _resize_large_image(self, image: Image.Image) -> Image.Image:
        """Memory-efficient resizing for very large images"""
        original_size = image.size
        
        # First, do a quick downsample to manageable size if needed
        max_intermediate_size = 2000
        if max(original_size) > max_intermediate_size:
            # Calculate intermediate size maintaining aspect ratio
            scale = max_intermediate_size / max(original_size)
            intermediate_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            
            logger.info(f"Step 1: Downsampling to {intermediate_size} for memory efficiency")
            # Use PIL's high-quality resize for the first step
            image = image.resize(intermediate_size, Image.LANCZOS)
            
            # Force garbage collection
            gc.collect()
        
        # Now proceed with normal smart resize
        img_array = np.array(image.convert('RGB')).astype(np.float32)
        final_img = cv2.resize(img_array, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return Image.fromarray(np.clip(final_img, 0, 255).astype(np.uint8))

    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Smart enhancement based on image characteristics"""
        # Convert to array for analysis
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Analyze image characteristics
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        logger.info(f"Image analysis - Blur: {blur_score:.1f}, Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
        
        enhanced = image
        
        # Smart sharpening based on blur score
        if blur_score < 100:  # Image is blurry
            # Apply unsharp masking
            img_array = np.array(enhanced)
            gaussian = cv2.GaussianBlur(img_array, (0, 0), 1.0)
            sharpened = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
            enhanced = Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))
            logger.info("Applied sharpening for blurry image")
        
        # Smart contrast adjustment
        if contrast < 30:  # Low contrast
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)
            logger.info("Enhanced contrast for flat image")
        
        # Smart brightness adjustment
        if brightness < 80:  # Too dark
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.1)
            logger.info("Brightened dark image")
        elif brightness > 200:  # Too bright
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(0.95)
            logger.info("Reduced brightness for overexposed image")
        
        return enhanced

    def process_image(self, input_path: str, output_path: str) -> bool:
        """Process a single image with smart resizing"""
        try:
            logger.info(f"Loading image: {input_path}")
            image = Image.open(input_path)
            original_size = image.size
            logger.info(f"Original size: {original_size}")
            
            # Memory usage estimation
            memory_mb = (original_size[0] * original_size[1] * 3) / (1024 * 1024)
            logger.info(f"Estimated memory usage: {memory_mb:.1f} MB (CPU only)")
            
            # Step 1: Smart resize
            logger.info("Applying smart resize...")
            resized_image = self.smart_resize(image)
            
            # Clean up original image from memory
            del image
            gc.collect()
            
            # Step 2: Smart enhancement
            logger.info("Applying smart enhancement...")
            final_image = self.enhance_image(resized_image)
            
            # Clean up intermediate image
            del resized_image
            gc.collect()
            
            # Save with high quality
            final_image.save(output_path, quality=95, optimize=True)
            logger.info(f"Smart result saved: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False

    def process_batch(self, input_dir: str, output_dir: str) -> int:
        """Process multiple images"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        
        if not image_files:
            logger.error(f"No images found in {input_dir}")
            return 0
        
        logger.info(f"Processing {len(image_files)} images with smart resizer...")
        
        success_count = 0
        for i, img_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {img_file.name}")
            output_file = output_path / img_file.name
            if self.process_image(str(img_file), str(output_file)):
                success_count += 1
            
            # Clean up between images
            gc.collect()
        
        logger.info(f"Successfully processed {success_count}/{len(image_files)} images")
        return success_count

def main():
    parser = argparse.ArgumentParser(description="Simple but Effective Image Resizer")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("output", help="Output image file or directory")
    parser.add_argument("--size", default="380x380", help="Target size (WxH, default: 380x380)")
    
    args = parser.parse_args()
    
    try:
        width, height = map(int, args.size.split('x'))
        target_size = (width, height)
    except ValueError:
        logger.error("Invalid size format. Use WIDTHxHEIGHT (e.g., 380x380)")
        return 1
    
    resizer = SmartImageResizer(target_size=target_size)
    
    input_path = Path(args.input)
    if input_path.is_file():
        success = resizer.process_image(args.input, args.output)
        return 0 if success else 1
    elif input_path.is_dir():
        success_count = resizer.process_batch(args.input, args.output)
        return 0 if success_count > 0 else 1
    else:
        logger.error(f"Input path not found: {args.input}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
