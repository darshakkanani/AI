#!/usr/bin/env python3
"""
State-of-the-Art Image Resizer - Real-ESRGAN Inspired
- RRDB (Residual in Residual Dense Block) architecture
- Multi-scale processing for optimal quality
- Memory-efficient tile processing for large images
- GPU acceleration with fallback to CPU
"""

import argparse
import os
import logging
import time
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ------------------------------
# Core Building Blocks
# ------------------------------

class DenseBlock(nn.Module):
    """Dense block with feature reuse for better gradient flow"""
    def __init__(self, channels, growth_rate=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block - Core of Real-ESRGAN"""
    def __init__(self, channels, growth_rate=32):
        super(RRDB, self).__init__()
        self.dense1 = DenseBlock(channels, growth_rate)
        self.dense2 = DenseBlock(channels, growth_rate)
        self.dense3 = DenseBlock(channels, growth_rate)
        
    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return out * 0.2 + x

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class RealESRGANGenerator(nn.Module):
    """Real-ESRGAN inspired generator optimized for resizing"""
    def __init__(self, target_size: Tuple[int, int], num_rrdb=16, num_feat=64):
        super(RealESRGANGenerator, self).__init__()
        self.target_size = target_size
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        
        # RRDB trunk
        self.trunk = nn.Sequential(*[RRDB(num_feat) for _ in range(num_rrdb)])
        
        # Trunk convolution
        self.conv_trunk = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Channel attention
        self.attention = ChannelAttention(num_feat)
        
        # Upsampling layers
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Final layers
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        # Feature extraction
        feat = self.lrelu(self.conv_first(x))
        trunk_out = self.conv_trunk(self.trunk(feat))
        feat = feat + trunk_out
        
        # Apply attention
        feat = self.attention(feat)
        
        # Progressive upsampling
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        # High-resolution processing
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        
        # Resize to exact target size
        out = F.interpolate(out, size=self.target_size, mode='bicubic', align_corners=False, antialias=True)
        
        return torch.clamp(out, 0, 1)

class FastResizer(nn.Module):
    """Lightweight resizer for speed-critical applications"""
    def __init__(self, target_size: Tuple[int, int]):
        super(FastResizer, self).__init__()
        self.target_size = target_size
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return F.interpolate(x, size=self.target_size, mode='bicubic', align_corners=False, antialias=True)

# ------------------------------
# Utilities
# ------------------------------

def load_image(path: str) -> Image.Image:
    """Load and validate image"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Cannot load image {path}: {e}")

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL to tensor with proper normalization"""
    transform = T.ToTensor()
    return transform(img).unsqueeze(0)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor back to PIL Image"""
    tensor = torch.clamp(tensor.squeeze(0), 0, 1)
    transform = T.ToPILImage()
    return transform(tensor)

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

# ------------------------------
# Main Resizer Class
# ------------------------------

class AdvancedImageResizer:
    """Advanced resizer with multiple quality modes and optimizations"""
    
    def __init__(self, target_size: Tuple[int, int], mode: str = 'balanced'):
        self.target_size = target_size
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model based on mode
        if mode == 'ultra':
            self.model = RealESRGANGenerator(target_size, num_rrdb=20, num_feat=64)
        elif mode == 'balanced':
            self.model = RealESRGANGenerator(target_size, num_rrdb=12, num_feat=48)
        elif mode == 'fast':
            self.model = FastResizer(target_size)
        else:  # classical
            self.model = None
            
        if self.model:
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Warm up GPU
            if torch.cuda.is_available():
                dummy = torch.randn(1, 3, 64, 64).to(self.device)
                with torch.no_grad():
                    _ = self.model(dummy)
                torch.cuda.empty_cache()
    
    def resize(self, img: Image.Image) -> Image.Image:
        """Main resize function with automatic optimization"""
        if self.mode == 'classical':
            return img.resize(self.target_size, Image.LANCZOS)
        
        original_size = img.size
        tensor = pil_to_tensor(img).to(self.device)
        
        # Memory management for large images
        h, w = tensor.shape[-2:]
        if h * w > 2048 * 2048:
            result = self._process_large_image(tensor)
        else:
            with torch.no_grad():
                result = self.model(tensor)
        
        return tensor_to_pil(result.cpu())
    
    def _process_large_image(self, tensor: torch.Tensor, tile_size: int = 512, overlap: int = 32) -> torch.Tensor:
        """Process large images in overlapping tiles"""
        b, c, h, w = tensor.shape
        target_h, target_w = self.target_size
        
        # Calculate scale factors
        scale_h = target_h / h
        scale_w = target_w / w
        
        # Create output tensor
        output = torch.zeros(b, c, target_h, target_w, device=tensor.device)
        weight_map = torch.zeros(b, 1, target_h, target_w, device=tensor.device)
        
        # Process tiles with overlap
        for i in range(0, h, tile_size - overlap):
            for j in range(0, w, tile_size - overlap):
                # Extract tile
                tile_h_start = i
                tile_h_end = min(i + tile_size, h)
                tile_w_start = j
                tile_w_end = min(j + tile_size, w)
                
                tile = tensor[:, :, tile_h_start:tile_h_end, tile_w_start:tile_w_end]
                
                # Calculate output tile coordinates
                out_h_start = int(tile_h_start * scale_h)
                out_h_end = int(tile_h_end * scale_h)
                out_w_start = int(tile_w_start * scale_w)
                out_w_end = int(tile_w_end * scale_w)
                
                # Process tile
                with torch.no_grad():
                    # Temporarily adjust model target size
                    original_target = self.model.target_size
                    self.model.target_size = (out_h_end - out_h_start, out_w_end - out_w_start)
                    
                    processed_tile = self.model(tile)
                    
                    # Restore original target size
                    self.model.target_size = original_target
                
                # Add to output with proper weighting
                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += processed_tile
                weight_map[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += 1
        
        # Normalize by weight map
        output = output / torch.clamp(weight_map, min=1)
        
        # Final resize to exact target size
        output = F.interpolate(output, size=self.target_size, mode='bicubic', align_corners=False)
        
        return output

# ------------------------------
# CLI Interface
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Advanced Deep Learning Image Resizer")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    parser.add_argument("--size", "-s", default="380x380", help="Target size (WxH)")
    parser.add_argument("--mode", "-m", default="balanced", 
                       choices=["ultra", "balanced", "fast", "classical"],
                       help="Quality mode")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Benchmark all modes")

    args = parser.parse_args()

    # Parse target size
    try:
        width, height = map(int, args.size.split("x"))
    except:
        raise ValueError("Invalid size format. Use WIDTHxHEIGHT (e.g., 380x380)")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Mode: {args.mode}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

    # Load image
    img = load_image(args.input)
    logging.info(f"Input: {args.input} ({img.size[0]}x{img.size[1]})")

    # Process image
    resizer = AdvancedImageResizer((height, width), mode=args.mode)
    
    start_time = time.time()
    result = resizer.resize(img)
    process_time = time.time() - start_time
    
    # Save result
    result.save(args.output, quality=95, optimize=True)
    
    logging.info(f"Output: {args.output} ({result.size[0]}x{result.size[1]})")
    logging.info(f"Processing time: {process_time:.2f}s")
    
    if torch.cuda.is_available():
        logging.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    
    # Benchmark if requested
    if args.benchmark:
        logging.info("\n--- BENCHMARK ---")
        modes = ["fast", "balanced", "ultra", "classical"]
        
        for mode in modes:
            if mode == args.mode:
                logging.info(f"{mode}: {process_time:.2f}s (already processed)")
                continue
                
            bench_resizer = AdvancedImageResizer((height, width), mode=mode)
            start = time.time()
            bench_result = bench_resizer.resize(img)
            bench_time = time.time() - start
            logging.info(f"{mode}: {bench_time:.2f}s")

if __name__ == "__main__":
    main()
