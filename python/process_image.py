#!/usr/bin/env python3
"""
World's Best AI Image Resizing Pipeline
- Uses Real-ESRGAN for state-of-the-art image resizing to 380x380
- Produces photorealistic results with enhanced details
- Automatic model download and GPU acceleration
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Real-ESRGAN Architecture ---

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for feature extraction"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RealESRGANGenerator(nn.Module):
    """Real-ESRGAN Generator for world-class image resizing"""
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RealESRGANGenerator, self).__init__()
        self.scale = scale
        
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        # Upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        
        return out

# --- Model Management ---

def download_file(url, output_path):
    """Downloads a file from a URL with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def download_realesrgan_model():
    """Download Real-ESRGAN model weights"""
    model_path = "RealESRGAN_x4plus.pth"
    if not os.path.exists(model_path):
        logger.info("Downloading Real-ESRGAN model (world's best image resizer)...")
        # Official Real-ESRGAN model from GitHub releases
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        try:
            download_file(url, model_path)
            logger.info("Real-ESRGAN model downloaded successfully!")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None
    return model_path

# --- World-Class Image Processor ---

class WorldBestImageResizer:
    """World's best AI image resizer using Real-ESRGAN"""
    
    def __init__(self, target_size: Tuple[int, int] = (380, 380)):
        self.target_size = target_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load Real-ESRGAN model
        model_path = download_realesrgan_model()
        if model_path:
            self.model = RealESRGANGenerator(scale=4)
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['params_ema'] if 'params_ema' in checkpoint else checkpoint)
                self.model.to(self.device)
                self.model.eval()
                self.use_ai = True
                logger.info("Real-ESRGAN model loaded - using world's best AI resizing!")
            except Exception as e:
                logger.warning(f"Failed to load Real-ESRGAN: {e}. Using high-quality fallback.")
                self.use_ai = False
        else:
            self.use_ai = False
            logger.info("Using high-quality traditional resizing as fallback.")

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """Preprocess image for Real-ESRGAN"""
        # Convert to RGB and normalize
        img_array = np.array(img.convert('RGB')).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL Image"""
        tensor = torch.clamp(tensor, 0, 1)
        img_array = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def resize_with_ai(self, image: Image.Image) -> Image.Image:
        """Resize using Real-ESRGAN (world's best method)"""
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # AI upscaling
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Convert back to PIL
        upscaled_image = self.postprocess_image(output_tensor)
        
        # Resize to exact target size with high-quality interpolation
        final_image = upscaled_image.resize(self.target_size, Image.LANCZOS)
        
        return final_image

    def resize_traditional(self, image: Image.Image) -> Image.Image:
        """High-quality traditional resize as fallback"""
        # Use the best traditional method available
        img_array = np.array(image.convert('RGB'))
        resized = cv2.resize(img_array, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Apply unsharp masking for enhanced details
        gaussian = cv2.GaussianBlur(resized, (0, 0), 1.0)
        sharpened = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
        
        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image using the best available method"""
        if self.use_ai:
            return self.resize_with_ai(image)
        else:
            return self.resize_traditional(image)

    def process_image(self, input_path: str, output_path: str) -> bool:
        """Process a single image with world-class resizing"""
        try:
            logger.info(f"Loading image: {input_path}")
            image = Image.open(input_path)
            original_size = image.size
            logger.info(f"Original size: {original_size}")
            
            logger.info(f"Resizing to {self.target_size} using world's best AI...")
            resized_image = self.resize_image(image)
            
            # Save with maximum quality
            resized_image.save(output_path, quality=98, optimize=True)
            logger.info(f"World-class result saved: {output_path}")
            
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
        
        logger.info(f"Processing {len(image_files)} images with world's best AI...")
        
        success_count = 0
        for i, img_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {img_file.name}")
            output_file = output_path / img_file.name
            if self.process_image(str(img_file), str(output_file)):
                success_count += 1
        
        logger.info(f"Successfully processed {success_count}/{len(image_files)} images")
        return success_count

def main():
    parser = argparse.ArgumentParser(description="World's Best AI Image Resizer")
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
    
    resizer = WorldBestImageResizer(target_size=target_size)
    
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
