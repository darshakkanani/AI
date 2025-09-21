import argparse
from pathlib import Path
from typing import Optional
from PIL import Image
import matplotlib.pyplot as plt

# Local imports
from u_sq_seg import segment_hair
from lama import inpaint_with_lama


def run_pipeline(image_path: str, mask_path: Optional[str] = None, output_path: Optional[str] = None, show: bool = False):
    image_path = Path(image_path)
    if mask_path is None:
        mask_path = str(image_path.with_suffix('').as_posix() + "_hair_mask.png")
    if output_path is None:
        out_dir = image_path.parent / "hair_removed_image"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{image_path.stem}_hair_removed{image_path.suffix}")

    # 1) Segment hair
    mask_path = segment_hair(str(image_path), mask_path)

    # 2) Inpaint with LaMa
    final_output = inpaint_with_lama(str(image_path), mask_path, output_path)

    if show:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(Image.open(str(image_path))); plt.title("Original"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(Image.open(mask_path), cmap="gray"); plt.title("Hair Mask"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(Image.open(final_output)); plt.title("Hair Removed"); plt.axis('off')
        plt.tight_layout(); plt.show()

    return mask_path, final_output


def main():
    parser = argparse.ArgumentParser(description="Integrated Hair Removal Pipeline: U²-Net (segmentation) + LaMa (inpainting)")
    parser.add_argument("input", help="Path to input image (e.g., .../image/resized_image/IMG_0341.jpg)")
    parser.add_argument("-m", "--mask", help="Optional path to save hair mask (PNG)")
    parser.add_argument("-o", "--output", help="Optional path to save final inpainted image")
    parser.add_argument("--show", action="store_true", help="Show a comparison figure")
    args = parser.parse_args()

    mask_path, final_output = run_pipeline(args.input, args.mask, args.output, args.show)
    print(f"[INFO] Mask: {mask_path}")
    print(f"[INFO] Output: {final_output}")


if __name__ == "__main__":
    main()
