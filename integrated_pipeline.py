#!/usr/bin/env python3
"""
Integrated Image Processing Pipeline
- First resizes images using advanced deep learning techniques
- Then automatically applies enterprise-grade hair removal
- Optimized for batch processing and enterprise deployment
"""

import argparse
import os
import logging
import time
import math
from typing import Tuple, List, Optional, Union
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

# Import components from existing modules
from image_resizing import AdvancedImageResizer, load_image, pil_to_tensor, tensor_to_pil
from hair_removal import HairRemovalProcessor, EnterpriseHairRemoval

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ------------------------------
# Integrated Pipeline Class
# ------------------------------

class IntegratedImageProcessor:
    """
    Integrated processor that combines image resizing and hair removal
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (380, 380),
                 resize_mode: str = 'balanced',
                 device: str = 'auto',
                 batch_size: int = 1):
        """
        Initialize the integrated processor
        
        Args:
            target_size: Target dimensions (width, height)
            resize_mode: Resizing quality mode ('ultra', 'balanced', 'fast', 'classical')
            device: Processing device ('auto', 'cpu', 'cuda')
            batch_size: Batch size for processing
        """
        self.target_size = target_size
        self.resize_mode = resize_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.batch_size = batch_size
        
        # Initialize components
        logging.info("Initializing image resizer...")
        self.resizer = AdvancedImageResizer(target_size, mode=resize_mode)
        
        logging.info("Initializing hair removal processor...")
        self.hair_remover = HairRemovalProcessor(device=device, batch_size=batch_size)
        
        logging.info(f"Integrated pipeline initialized on {self.device}")
        logging.info(f"Target size: {target_size[0]}x{target_size[1]}")
        logging.info(f"Resize mode: {resize_mode}")
    
    def process_single_image(self, 
                           img: Image.Image, 
                           return_intermediate: bool = False,
                           return_mask: bool = False) -> Union[Image.Image, Tuple]:
        """
        Process a single image through the complete pipeline
        
        Args:
            img: Input PIL Image
            return_intermediate: Whether to return intermediate results
            return_mask: Whether to return hair mask
            
        Returns:
            Final processed image, or tuple with intermediate results
        """
        original_size = img.size
        logging.info(f"Processing image: {original_size[0]}x{original_size[1]} -> {self.target_size[0]}x{self.target_size[1]}")
        
        # Step 1: Resize the image
        start_time = time.time()
        resized_img = self.resizer.resize(img)
        resize_time = time.time() - start_time
        logging.info(f"Resizing completed in {resize_time:.3f}s")
        
        # Step 2: Remove hair from resized image
        start_time = time.time()
        if return_mask:
            final_img, hair_mask = self.hair_remover.process_single_image(resized_img, return_mask=True)
        else:
            final_img = self.hair_remover.process_single_image(resized_img, return_mask=False)
        hair_removal_time = time.time() - start_time
        logging.info(f"Hair removal completed in {hair_removal_time:.3f}s")
        
        # Return results based on flags
        if return_intermediate and return_mask:
            return final_img, resized_img, hair_mask
        elif return_intermediate:
            return final_img, resized_img
        elif return_mask:
            return final_img, hair_mask
        else:
            return final_img
    
    def process_batch(self, 
                     images: List[Image.Image], 
                     return_intermediate: bool = False,
                     return_masks: bool = False) -> List[Union[Image.Image, Tuple]]:
        """
        Process multiple images in batch
        
        Args:
            images: List of PIL Images
            return_intermediate: Whether to return intermediate results
            return_masks: Whether to return hair masks
            
        Returns:
            List of processed results
        """
        results = []
        total_images = len(images)
        
        logging.info(f"Processing batch of {total_images} images...")
        
        for i, img in enumerate(images):
            logging.info(f"Processing image {i+1}/{total_images}")
            
            result = self.process_single_image(
                img, 
                return_intermediate=return_intermediate,
                return_mask=return_masks
            )
            results.append(result)
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def benchmark_performance(self, test_image: Image.Image, num_runs: int = 5) -> dict:
        """
        Benchmark the complete pipeline performance
        
        Args:
            test_image: Test image for benchmarking
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics dictionary
        """
        logging.info(f"Running performance benchmark ({num_runs} runs)...")
        
        times = []
        resize_times = []
        hair_removal_times = []
        
        for run in range(num_runs):
            # Resize timing
            start_time = time.time()
            resized_img = self.resizer.resize(test_image)
            resize_time = time.time() - start_time
            resize_times.append(resize_time)
            
            # Hair removal timing
            start_time = time.time()
            final_img = self.hair_remover.process_single_image(resized_img)
            hair_removal_time = time.time() - start_time
            hair_removal_times.append(hair_removal_time)
            
            total_time = resize_time + hair_removal_time
            times.append(total_time)
            
            logging.info(f"Run {run+1}: {total_time:.3f}s (resize: {resize_time:.3f}s, hair removal: {hair_removal_time:.3f}s)")
        
        return {
            'total_avg_time': np.mean(times),
            'total_min_time': np.min(times),
            'total_max_time': np.max(times),
            'total_std_time': np.std(times),
            'total_fps': 1.0 / np.mean(times),
            'resize_avg_time': np.mean(resize_times),
            'hair_removal_avg_time': np.mean(hair_removal_times),
            'resize_fps': 1.0 / np.mean(resize_times),
            'hair_removal_fps': 1.0 / np.mean(hair_removal_times)
        }

# ------------------------------
# CLI Interface
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Integrated Image Processing Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input image path or directory")
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument("--size", "-s", default="380x380", help="Target size (WxH)")
    parser.add_argument("--resize-mode", "-r", default="balanced", 
                       choices=["ultra", "balanced", "fast", "classical"],
                       help="Resize quality mode")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--device", "-d", choices=['auto', 'cpu', 'cuda'], default='auto', help="Processing device")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate resized images")
    parser.add_argument("--save-mask", "-m", action="store_true", help="Save hair masks")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Parse target size
    try:
        width, height = map(int, args.size.split("x"))
        target_size = (width, height)
    except:
        raise ValueError("Invalid size format. Use WIDTHxHEIGHT (e.g., 380x380)")
    
    # Setup device
    device = args.device
    if torch.cuda.is_available() and device == 'auto':
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = 'cpu'
    
    logging.info(f"Device: {device}")
    logging.info(f"Target size: {width}x{height}")
    logging.info(f"Resize mode: {args.resize_mode}")
    
    # Initialize processor
    processor = IntegratedImageProcessor(
        target_size=target_size,
        resize_mode=args.resize_mode,
        device=device,
        batch_size=args.batch_size
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single image processing
        img = load_image(args.input)
        logging.info(f"Input: {args.input} ({img.size[0]}x{img.size[1]})")
        
        start_time = time.time()
        
        # Process with appropriate return flags
        if args.save_intermediate and args.save_mask:
            result, intermediate, mask = processor.process_single_image(
                img, return_intermediate=True, return_mask=True
            )
        elif args.save_intermediate:
            result, intermediate = processor.process_single_image(
                img, return_intermediate=True, return_mask=False
            )
        elif args.save_mask:
            result, mask = processor.process_single_image(
                img, return_intermediate=False, return_mask=True
            )
        else:
            result = processor.process_single_image(
                img, return_intermediate=False, return_mask=False
            )
        
        total_time = time.time() - start_time
        
        # Save results
        result.save(args.output, quality=95, optimize=True)
        logging.info(f"Final result saved: {args.output}")
        
        if args.save_intermediate:
            intermediate_path = args.output.replace('.', '_resized.')
            intermediate.save(intermediate_path, quality=95, optimize=True)
            logging.info(f"Intermediate result saved: {intermediate_path}")
        
        if args.save_mask:
            mask_path = args.output.replace('.', '_mask.')
            mask.save(mask_path, quality=95, optimize=True)
            logging.info(f"Hair mask saved: {mask_path}")
        
        logging.info(f"Total processing time: {total_time:.3f}s")
        
        # Benchmark if requested
        if args.benchmark:
            logging.info("\n--- PERFORMANCE BENCHMARK ---")
            bench_results = processor.benchmark_performance(img)
            logging.info(f"Total pipeline: {bench_results['total_fps']:.1f} FPS (avg: {bench_results['total_avg_time']:.3f}s)")
            logging.info(f"Resize component: {bench_results['resize_fps']:.1f} FPS (avg: {bench_results['resize_avg_time']:.3f}s)")
            logging.info(f"Hair removal component: {bench_results['hair_removal_fps']:.1f} FPS (avg: {bench_results['hair_removal_avg_time']:.3f}s)")
    
    elif os.path.isdir(args.input):
        # Batch processing
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in os.listdir(args.input) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        if not image_files:
            logging.error(f"No images found in {args.input}")
            return
        
        logging.info(f"Found {len(image_files)} images for batch processing")
        
        # Load images
        images = []
        for filename in image_files:
            img_path = os.path.join(args.input, filename)
            img = load_image(img_path)
            images.append((img, filename))
        
        # Process batch
        start_time = time.time()
        results = processor.process_batch(
            [img for img, _ in images], 
            return_intermediate=args.save_intermediate,
            return_masks=args.save_mask
        )
        total_time = time.time() - start_time
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        
        for i, (result_data, (_, filename)) in enumerate(zip(results, images)):
            # Handle different return formats
            if args.save_intermediate and args.save_mask:
                result, intermediate, mask = result_data
            elif args.save_intermediate:
                result, intermediate = result_data
            elif args.save_mask:
                result, mask = result_data
            else:
                result = result_data
            
            # Save final result
            output_path = os.path.join(args.output, filename)
            result.save(output_path, quality=95, optimize=True)
            
            # Save intermediate if requested
            if args.save_intermediate:
                intermediate_path = os.path.join(args.output, f"resized_{filename}")
                intermediate.save(intermediate_path, quality=95, optimize=True)
            
            # Save mask if requested
            if args.save_mask:
                mask_path = os.path.join(args.output, f"mask_{filename}")
                mask.save(mask_path, quality=95, optimize=True)
        
        logging.info(f"Batch processing completed: {len(image_files)} images in {total_time:.2f}s")
        logging.info(f"Average time per image: {total_time/len(image_files):.3f}s")
        logging.info(f"Throughput: {len(image_files)/total_time:.1f} images/second")
    
    else:
        logging.error(f"Input path not found: {args.input}")

if __name__ == "__main__":
    main()
