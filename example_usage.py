#!/usr/bin/env python3
"""
Example usage of the Integrated Image Processing Pipeline
This script demonstrates how to use the pipeline programmatically
"""

from PIL import Image
from integrated_pipeline import IntegratedImageProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def example_single_image():
    """Example: Process a single image"""
    print("=== Single Image Processing Example ===")
    
    # Initialize the processor
    processor = IntegratedImageProcessor(
        target_size=(380, 380),  # Resize to 380x380
        resize_mode='balanced',   # Use balanced quality mode
        device='auto'            # Auto-detect GPU/CPU
    )
    
    # Load an image (replace with your image path)
    try:
        img = Image.open("input_image.jpg")
        print(f"Loaded image: {img.size}")
        
        # Process the image (resize + hair removal)
        result = processor.process_single_image(img)
        
        # Save the result
        result.save("output_processed.jpg", quality=95)
        print("Processed image saved as 'output_processed.jpg'")
        
    except FileNotFoundError:
        print("Please place an image named 'input_image.jpg' in the current directory")

def example_with_intermediate_results():
    """Example: Process image and save intermediate results"""
    print("\n=== Processing with Intermediate Results ===")
    
    processor = IntegratedImageProcessor(
        target_size=(380, 380),
        resize_mode='balanced'
    )
    
    try:
        img = Image.open("input_image.jpg")
        
        # Process with intermediate results and hair mask
        final_result, resized_img, hair_mask = processor.process_single_image(
            img, 
            return_intermediate=True, 
            return_mask=True
        )
        
        # Save all results
        final_result.save("final_result.jpg", quality=95)
        resized_img.save("resized_only.jpg", quality=95)
        hair_mask.save("hair_mask.jpg", quality=95)
        
        print("Saved: final_result.jpg, resized_only.jpg, hair_mask.jpg")
        
    except FileNotFoundError:
        print("Please place an image named 'input_image.jpg' in the current directory")

def example_batch_processing():
    """Example: Process multiple images"""
    print("\n=== Batch Processing Example ===")
    
    processor = IntegratedImageProcessor(
        target_size=(380, 380),
        resize_mode='fast',  # Use fast mode for batch processing
        batch_size=2
    )
    
    # Create sample images list (replace with your images)
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    
    try:
        # Load images
        images = []
        for path in image_paths:
            img = Image.open(path)
            images.append(img)
        
        # Process batch
        results = processor.process_batch(images)
        
        # Save results
        for i, result in enumerate(results):
            result.save(f"batch_output_{i+1}.jpg", quality=95)
        
        print(f"Processed {len(results)} images")
        
    except FileNotFoundError:
        print("Please ensure image1.jpg, image2.jpg, image3.jpg exist in the current directory")

def example_benchmark():
    """Example: Benchmark performance"""
    print("\n=== Performance Benchmark Example ===")
    
    processor = IntegratedImageProcessor(
        target_size=(380, 380),
        resize_mode='balanced'
    )
    
    try:
        img = Image.open("input_image.jpg")
        
        # Run benchmark
        results = processor.benchmark_performance(img, num_runs=3)
        
        print(f"Total Pipeline Performance:")
        print(f"  Average time: {results['total_avg_time']:.3f}s")
        print(f"  FPS: {results['total_fps']:.1f}")
        print(f"Resize Component:")
        print(f"  Average time: {results['resize_avg_time']:.3f}s")
        print(f"  FPS: {results['resize_fps']:.1f}")
        print(f"Hair Removal Component:")
        print(f"  Average time: {results['hair_removal_avg_time']:.3f}s")
        print(f"  FPS: {results['hair_removal_fps']:.1f}")
        
    except FileNotFoundError:
        print("Please place an image named 'input_image.jpg' in the current directory")

def example_different_modes():
    """Example: Compare different resize modes"""
    print("\n=== Resize Mode Comparison ===")
    
    modes = ['fast', 'balanced', 'ultra', 'classical']
    
    try:
        img = Image.open("input_image.jpg")
        
        for mode in modes:
            print(f"\nProcessing with {mode} mode...")
            
            processor = IntegratedImageProcessor(
                target_size=(380, 380),
                resize_mode=mode
            )
            
            result = processor.process_single_image(img)
            result.save(f"output_{mode}_mode.jpg", quality=95)
            print(f"Saved: output_{mode}_mode.jpg")
            
    except FileNotFoundError:
        print("Please place an image named 'input_image.jpg' in the current directory")

if __name__ == "__main__":
    print("Integrated Image Processing Pipeline - Examples")
    print("=" * 50)
    
    # Run examples
    example_single_image()
    example_with_intermediate_results()
    example_batch_processing()
    example_benchmark()
    example_different_modes()
    
    print("\n" + "=" * 50)
    print("Examples completed! Check the output files.")
