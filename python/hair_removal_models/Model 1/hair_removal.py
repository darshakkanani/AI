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
import logging
from typing import Optional, Tuple, Dict

def convert_to_grayscale(image, use_clahe: bool = True):
    """
    Step 1: Convert RGB images into grayscale images
    """
    logging.info("Step 1: Converting RGB to grayscale...")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        # Improve contrast to make thin hairs more separable
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grayscale = clahe.apply(grayscale)
    return grayscale

def apply_blackhat_transformation(grayscale_image):
    """
    Step 2: Apply Morphological Black-Hat transformation on the grayscale images
    Enhanced with multiple kernel sizes for maximum accuracy
    """
    logging.info("Step 2: Applying Morphological Black-Hat transformation...")
    
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

def compute_gabor_response(grayscale_image: np.ndarray) -> np.ndarray:
    """
    Compute max Gabor filter bank response to enhance hair-like linear structures.
    Returns a float32 array normalized to [0,1].
    """
    logging.info("Enhancing hair response with Gabor filter bank...")
    thetas = [0, 20, 40, 60, 80, 100, 120, 140]
    # Wavelengths roughly matching hair thickness in pixels
    lambdas = [6, 8, 10, 12]
    sigma = 3.0
    gamma = 0.5  # aspect ratio
    responses = []
    img = grayscale_image.astype(np.float32) / 255.0
    for th in thetas:
        theta = np.deg2rad(th)
        for lam in lambdas:
            ksize = int(4 * sigma) | 1
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(img, cv2.CV_32F, kernel)
            responses.append(resp)
    gabor_max = np.maximum.reduce(responses)
    # Normalize to [0,1]
    g_min, g_max = float(gabor_max.min()), float(gabor_max.max())
    if g_max - g_min > 1e-6:
        gabor_norm = (gabor_max - g_min) / (g_max - g_min)
    else:
        gabor_norm = np.zeros_like(gabor_max, dtype=np.float32)
    return gabor_norm

def _aggr_params(level: int) -> Dict:
    """Map aggressiveness level (1-5) to mask + inpainting parameters."""
    lvl = int(max(1, min(5, level)))
    # Percentile gets lower (more aggressive) as level increases
    pct_map = {1: 85, 2: 80, 3: 75, 4: 65, 5: 55}
    # Morphology iterations and dilation size increase with level
    close_iter_map = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
    open_iter_map  = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2}
    dilate_iter_map= {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
    dilate_size_map= {1: 2, 2: 2, 3: 3, 4: 3, 5: 4}
    # Minimum area shrinks with aggressiveness
    min_area_map   = {1: 20, 2: 15, 3: 10, 4: 8, 5: 5}
    # Inpainting radii grow with aggressiveness
    telea_r_small  = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7}[lvl]
    ns_r           = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}[lvl]
    telea_r_large  = {1: 6, 2: 7, 3: 8, 4: 9, 5: 10}[lvl]
    # Blending: bias more toward TELEA on higher levels
    telea_w = {1: 0.5, 2: 0.55, 3: 0.6, 4: 0.65, 5: 0.7}[lvl]
    return {
        "percentile": pct_map[lvl],
        "close_iter": close_iter_map[lvl],
        "open_iter":  open_iter_map[lvl],
        "dilate_iter":dilate_iter_map[lvl],
        "dilate_size":dilate_size_map[lvl],
        "min_area":   min_area_map[lvl],
        "telea_r_small": telea_r_small,
        "ns_r": ns_r,
        "telea_r_large": telea_r_large,
        "telea_weight": telea_w,
    }

def create_inpainting_mask(blackhat_image: np.ndarray, return_stages: bool = False, params: Optional[Dict] = None):
    """
    Step 3: Create a mask for the inpainting task
    Enhanced mask creation with adaptive thresholding
    """
    logging.info("Step 3: Creating mask for inpainting...")
    params = params or {}
    percentile = params.get("percentile", 75)
    
    # Adaptive thresholding for better mask creation
    non_zero_pixels = blackhat_image[blackhat_image > 0]
    if len(non_zero_pixels) > 0:
        # Use configured percentile for aggressive hair detection
        threshold_value = np.percentile(non_zero_pixels, percentile)
    else:
        threshold_value = 15  # Default threshold
    
    # Create binary mask
    _, binary_mask = cv2.threshold(blackhat_image, threshold_value, 255, cv2.THRESH_BINARY)
    # Fallback to OTSU if mask is too sparse
    if np.count_nonzero(binary_mask) < 0.001 * binary_mask.size:
        _, binary_mask = cv2.threshold(blackhat_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to refine the mask
    # Close small gaps in hair strands
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=int(params.get("close_iter", 2)))
    
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open, iterations=int(params.get("open_iter", 1)))
    
    # Remove very small components (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask)
    min_area = int(params.get("min_area", 10))  # Minimum area for a hair strand
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            refined_mask[labels == i] = 0
    
    # Slight dilation to ensure complete hair coverage
    ksz = int(params.get("dilate_size", 2))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    final_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=int(params.get("dilate_iter", 1)))
    
    if return_stages:
        return final_mask, {
            "binary": binary_mask,
            "refined": refined_mask,
            "threshold_value": threshold_value,
        }
    return final_mask

def apply_inpainting_algorithm(original_image, mask, telea_r_small: int = 3, ns_r: int = 5, telea_r_large: int = 7, telea_weight: float = 0.6):
    """
    Step 4: Apply inpainting algorithm on the original image using this mask
    Enhanced with multiple inpainting methods for best results
    """
    logging.info("Step 4: Applying inpainting algorithm...")
    
    # Method 1: Fast Marching Method (TELEA) - good for textures
    inpainted_telea = cv2.inpaint(original_image, mask, inpaintRadius=int(telea_r_small), flags=cv2.INPAINT_TELEA)
    
    # Method 2: Navier-Stokes based method - good for smooth areas
    inpainted_ns = cv2.inpaint(original_image, mask, inpaintRadius=int(ns_r), flags=cv2.INPAINT_NS)
    
    # Method 3: Enhanced TELEA with larger radius
    inpainted_telea_large = cv2.inpaint(original_image, mask, inpaintRadius=int(telea_r_large), flags=cv2.INPAINT_TELEA)
    
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
            (1 - smooth_regions) * (float(telea_weight) * inpainted_telea[:,:,c] + (1.0 - float(telea_weight)) * inpainted_telea_large[:,:,c])
        )
    
    return result.astype(np.uint8)

def _ensure_parent_dir(path_str: str):
    out_path = Path(path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

def _resize_if_needed(img: np.ndarray, max_size: int | None):
    if not max_size:
        return img
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def _advanced_sam_lama_pipeline(original_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Advanced SAM+LaMa pipeline for maximum accuracy hair removal.
    Uses SAM for precise hair segmentation and LaMa for high-quality inpainting.
    Returns (result, mask).
    """
    logging.info("Using SAM+LaMa pipeline for maximum accuracy...")
    
    try:
        import torch
        import torchvision.transforms as transforms
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import requests
        from PIL import Image
        import io
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        # Download SAM model if not exists
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        if not os.path.exists(sam_checkpoint):
            logging.info("Downloading SAM model...")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            response = requests.get(url, stream=True)
            with open(sam_checkpoint, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info("SAM model downloaded successfully")
        
        # Initialize SAM
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        # Create mask generator with hair-optimized settings
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Remove very small segments
        )
        
        # Generate masks
        logging.info("Generating hair masks with SAM...")
        masks = mask_generator.generate(original_image)
        
        # Filter masks to identify hair-like structures
        hair_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            
            # Calculate mask properties to identify hair
            area = mask_data['area']
            bbox = mask_data['bbox']
            w, h = bbox[2], bbox[3]
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
            
            # Hair characteristics: elongated, medium area, high contrast
            if (aspect_ratio > 2.5 and  # Elongated
                100 < area < 5000 and   # Medium area
                mask_data['stability_score'] > 0.85):  # High quality
                
                # Additional check: hair should be darker than surrounding skin
                mask_region = original_image[mask]
                if len(mask_region) > 0:
                    avg_intensity = np.mean(cv2.cvtColor(mask_region.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY))
                    if avg_intensity < 80:  # Dark structures (hair)
                        hair_mask = np.logical_or(hair_mask, mask).astype(np.uint8)
        
        # Refine hair mask
        hair_mask = hair_mask * 255
        
        # Morphological operations to connect nearby hair segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Dilate slightly to ensure complete coverage
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        hair_mask = cv2.dilate(hair_mask, kernel_dilate, iterations=1)
        
        # Use LaMa for high-quality inpainting
        logging.info("Applying LaMa inpainting...")
        result = _lama_inpaint(original_image, hair_mask)
        
        return result, hair_mask
        
    except Exception as e:
        logging.error(f"SAM+LaMa pipeline failed: {e}")
        logging.info("Falling back to enhanced classical method...")
        # Fallback to enhanced classical method
        grayscale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(grayscale)
        
        # Enhanced black-hat with multiple kernels
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        ]
        
        responses = []
        for kernel in kernels:
            resp = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
            responses.append(resp)
        
        combined = np.maximum.reduce(responses)
        
        # Aggressive thresholding
        _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological refinement
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # High-quality inpainting
        result = cv2.inpaint(original_image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        
        return result, mask

def _lama_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    High-quality inpainting using LaMa model.
    Falls back to advanced OpenCV inpainting if LaMa is not available.
    """
    try:
        # Try to use LaMa if available
        from lama_cleaner.model_manager import ModelManager
        from lama_cleaner.schema import Config
        
        # Initialize LaMa model
        model = ModelManager(name="lama", device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert to PIL format
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask)
        
        # Configure LaMa
        config = Config(
            ldm_steps=20,
            ldm_sampler="plms",
            hd_strategy="Original",
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=512,
            hd_strategy_resize_limit=2048,
        )
        
        # Apply LaMa inpainting
        result_pil = model(pil_image, pil_mask, config)
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        
        logging.info("LaMa inpainting completed successfully")
        return result
        
    except Exception as e:
        logging.warning(f"LaMa inpainting failed, using advanced OpenCV: {e}")
        
        # Advanced OpenCV inpainting fallback
        # Use multiple methods and blend
        telea_result = cv2.inpaint(image, mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)
        ns_result = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_NS)
        
        # Blend based on local texture
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_score = np.abs(laplacian)
        texture_threshold = np.percentile(texture_score, 70)
        
        smooth_regions = (texture_score < texture_threshold).astype(np.float32)
        smooth_regions = cv2.GaussianBlur(smooth_regions, (5, 5), 0)
        
        # Blend results
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            result[:,:,c] = (
                smooth_regions * ns_result[:,:,c] +
                (1 - smooth_regions) * telea_result[:,:,c]
            )
        
        return result.astype(np.uint8)

def process_hair_removal(image_path, output_path=None, show_steps=False, method: str = "classical", save_mask: bool = True, max_size: int | None = None, save_intermediates: bool = True, aggr: int = 3):
    """
    Complete hair removal process with optional method selection.
    method: 'classical' (default) or 'sam-lama' (placeholder).
    """
    logging.info(f"Processing: {image_path}")
    
    # Load the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        logging.error(f"Could not load image from {image_path}")
        return None
    
    logging.debug(f"Original image size: {original_image.shape}")
    
    # Optional downscale for memory efficiency
    original_proc = _resize_if_needed(original_image, max_size)
    params = _aggr_params(aggr)
    
    if method.lower() == "classical":
        # Step 1: Convert RGB images into grayscale images
        grayscale_image = convert_to_grayscale(original_proc, use_clahe=True)
        
        # Step 2: Apply Morphological Black-Hat transformation
        blackhat_result = apply_blackhat_transformation(grayscale_image)

        # Gabor response to enhance linear hair structures
        gabor_resp = compute_gabor_response(grayscale_image)
        gabor_uint8 = (np.clip(gabor_resp, 0, 1) * 255).astype(np.uint8)
        
        # Fuse cues for a stronger mask source
        fused_for_mask = np.maximum(blackhat_result, (0.8 * gabor_uint8).astype(np.uint8))
        
        # Step 3: Create a mask for the inpainting task
        inpainting_mask, stages = create_inpainting_mask(fused_for_mask, return_stages=True, params=params)
        
        # Step 4: Apply inpainting algorithm
        final_result = apply_inpainting_algorithm(
            original_proc,
            inpainting_mask,
            telea_r_small=params["telea_r_small"],
            ns_r=params["ns_r"],
            telea_r_large=params["telea_r_large"],
            telea_weight=params["telea_weight"],
        )
    elif method.lower() == "sam-lama":
        try:
            final_result, inpainting_mask = _advanced_sam_lama_pipeline(original_proc)
            # For consistency when saving intermediates
            grayscale_image = cv2.cvtColor(original_proc, cv2.COLOR_BGR2GRAY)
            blackhat_result = np.zeros_like(grayscale_image)
            gabor_uint8 = np.zeros_like(grayscale_image)
            fused_for_mask = inpainting_mask
            stages = {"binary": inpainting_mask, "refined": inpainting_mask, "threshold_value": 0}
        except RuntimeError as e:
            logging.error(str(e))
            return None
    else:
        logging.error(f"Unknown method: {method}")
        return None

    # If we downscaled for processing, optionally upscale back to original size for saving
    if final_result.shape[:2] != original_image.shape[:2]:
        final_result = cv2.resize(final_result, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        inpainting_mask = cv2.resize(inpainting_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        grayscale_image = cv2.resize(grayscale_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_AREA)
        blackhat_result = cv2.resize(blackhat_result, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_AREA)
        gabor_uint8 = cv2.resize(gabor_uint8, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_AREA)
        fused_for_mask = cv2.resize(fused_for_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_AREA)
        stages["binary"] = cv2.resize(stages["binary"], (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        stages["refined"] = cv2.resize(stages["refined"], (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Save the result if output path is provided
    if output_path:
        _ensure_parent_dir(output_path)
        cv2.imwrite(output_path, final_result)
        logging.info(f"Result saved to: {output_path}")
        
        if save_mask:
            out_p = Path(output_path)
            mask_path = str(out_p.with_name(f"{out_p.stem}_mask{out_p.suffix}"))
            cv2.imwrite(mask_path, inpainting_mask)
            logging.info(f"Mask saved to: {mask_path}")
        
        if save_intermediates:
            out_p = Path(output_path)
            cv2.imwrite(str(out_p.with_name(f"{out_p.stem}_gray{out_p.suffix}")), grayscale_image)
            cv2.imwrite(str(out_p.with_name(f"{out_p.stem}_blackhat{out_p.suffix}")), blackhat_result)
            cv2.imwrite(str(out_p.with_name(f"{out_p.stem}_gabor{out_p.suffix}")), gabor_uint8)
            cv2.imwrite(str(out_p.with_name(f"{out_p.stem}_fused{out_p.suffix}")), fused_for_mask)
            cv2.imwrite(str(out_p.with_name(f"{out_p.stem}_mask_binary{out_p.suffix}")), stages["binary"])
            cv2.imwrite(str(out_p.with_name(f"{out_p.stem}_mask_refined{out_p.suffix}")), stages["refined"])
            # Save an overlay of mask on the original for quick QA
            overlay = original_image.copy()
            red = np.zeros_like(original_image)
            red[:, :, 2] = 255
            mask_3 = cv2.cvtColor(inpainting_mask, cv2.COLOR_GRAY2BGR)
            overlay = np.where(mask_3 > 0, (0.5 * red + 0.5 * original_image).astype(np.uint8), original_image)
            cv2.imwrite(str(out_p.with_name(f"{out_p.stem}_overlay{out_p.suffix}")), overlay)
            logging.info("Saved intermediate images (gray, blackhat, gabor, fused, mask_binary, mask_refined)")
    
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
        
        cv2.imshow("Step 0: Original Image", original_display)
        cv2.imshow("Step 1: Grayscale", grayscale_display)
        cv2.imshow("Step 2: Black-Hat Result", blackhat_display)
        cv2.imshow("Step 3: Inpainting Mask", mask_display)
        cv2.imshow("Step 4: Final Result", result_display)
        
        logging.info("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return final_result

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description="Enhanced Hair Removal - Multi-Method")
    parser.add_argument("input", nargs="?", help="Input image path (use this or --input-dir)")
    parser.add_argument("-o", "--output", help="Output image path (or use --output-dir for batch)")
    parser.add_argument("--input-dir", help="Directory of input images (batch mode)")
    parser.add_argument("--output-dir", help="Directory to save outputs (batch mode)")
    parser.add_argument("--method", choices=["classical", "sam-lama"], default="classical", help="Hair removal method")
    parser.add_argument("--display", action="store_true", help="Display step-by-step results (off by default)")
    parser.add_argument("--no-mask", action="store_true", help="Do not save mask image")
    parser.add_argument("--max-size", type=int, default=None, help="Max dimension for processing (e.g., 1024) for memory efficiency")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--no-intermediates", action="store_true", help="Do not save intermediate images alongside output")
    parser.add_argument("--aggr", type=int, default=3, help="Aggressiveness level 1-5 (higher removes more aggressively)")
    
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='[%(levelname)s] %(message)s')

    # Batch mode
    if args.input_dir:
        if not args.output_dir:
            logging.error("--output-dir is required when using --input-dir")
            return
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        if not in_dir.exists():
            logging.error(f"Input directory does not exist: {in_dir}")
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = [p for p in in_dir.rglob("*") if p.suffix.lower() in exts]
        if not images:
            logging.warning(f"No images found in {in_dir}")
            return
        logging.info(f"Found {len(images)} images. Starting batch processing...")
        for p in images:
            rel = p.relative_to(in_dir)
            out_p = out_dir / rel
            out_p = out_p.with_name(f"{out_p.stem}_hair_removed{out_p.suffix}")
            try:
                process_hair_removal(
                    image_path=str(p),
                    output_path=str(out_p),
                    show_steps=args.display,
                    method=args.method,
                    save_mask=not args.no_mask,
                    max_size=args.max_size,
                    save_intermediates=not args.no_intermediates,
                    aggr=args.aggr,
                )
            except Exception as e:
                logging.exception(f"Failed to process {p}: {e}")
        logging.info("Batch processing completed.")
        return

    # Single image mode
    if not args.input:
        logging.error("Please provide an input image path or use --input-dir for batch mode.")
        return
    
    # Set default output path if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_hair_removed{input_path.suffix}")
    
    # Process the image
    result = process_hair_removal(
        image_path=args.input,
        output_path=args.output,
        show_steps=args.display,
        method=args.method,
        save_mask=not args.no_mask,
        max_size=args.max_size,
        save_intermediates=not args.no_intermediates,
        aggr=args.aggr,
    )
    
    if result is not None:
        logging.info("Hair removal completed successfully!")
    else:
        logging.error("Hair removal failed!")

if __name__ == "__main__":
    main()