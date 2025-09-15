#!/usr/bin/env python3
"""
Crop large images into 512x512 sub-images with sequential naming.

This script processes all images in the 'screenshots' folder and saves
512x512 tiles to the 'cropped' folder with sequential names (001.jpg, 002.jpg, etc.).

Usage
-----
python crop_images.py [--background-color R G B]

Examples:
python crop_images.py                        # Black background (0, 0, 0) - best for ML
python crop_images.py --background-color 255 255 255  # White background
python crop_images.py --background-color 128 128 128  # Gray background

Dependencies
------------
pip install pillow
"""
import argparse
import os
from pathlib import Path
from typing import Iterable

from PIL import Image

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def iter_images(root: Path) -> Iterable[Path]:
    """Iterate through all valid image files in the root directory."""
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p

def ensure_rgb(img: Image.Image, background_color: tuple = (255, 255, 255)) -> Image.Image:
    """Convert image to RGB if needed, handling transparency properly."""
    if img.mode == 'RGB':
        return img
    elif img.mode == 'RGBA':
        # Create a background and paste the RGBA image on it
        rgb_img = Image.new('RGB', img.size, background_color)
        rgb_img.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
        return rgb_img
    elif img.mode == 'L':
        return img.convert('RGB')
    else:
        return img.convert('RGB')

def crop_image_to_tiles(
    image_path: Path,
    tile_size: int = 512,
    counter: int = 1,
    background_color: tuple = (0, 0, 0)  # Default to black for better ML compatibility
) -> int:
    """
    Crop a single image into 512x512 tiles.
    
    Args:
        image_path: Path to the input image
        tile_size: Size of each tile (default 512)
        counter: Starting counter for sequential naming
    
    Returns:
        Updated counter after processing this image
    """
    with Image.open(image_path) as img:
        img = ensure_rgb(img, background_color)
        width, height = img.size
        
        print(f"Processing {image_path.name} ({width}x{height})")
        
        tiles_created = 0
        
        # Calculate how many complete tiles we can get
        tiles_x = width // tile_size
        tiles_y = height // tile_size
        
        if tiles_x == 0 or tiles_y == 0:
            print(f"  Warning: Image {image_path.name} is smaller than {tile_size}x{tile_size}, skipping.")
            return counter
        
        # Create tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate crop coordinates
                left = x * tile_size
                top = y * tile_size
                right = left + tile_size
                bottom = top + tile_size
                
                # Crop the tile
                tile = img.crop((left, top, right, bottom))
                
                # Save with sequential naming
                output_path = cropped_dir / f"{counter:03d}.jpg"
                tile.save(output_path, "JPEG", quality=95)
                
                counter += 1
                tiles_created += 1
        
        print(f"  Created {tiles_created} tiles from {image_path.name}")
        return counter

def main():
    """Main function to process all images."""
    global cropped_dir
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Crop large images into 512x512 tiles")
    parser.add_argument('--background-color', type=int, nargs=3, metavar=('R', 'G', 'B'),
                       default=[0, 0, 0], help='Background color for transparent areas (RGB values 0-255). Default: 0 0 0 (black)')
    args = parser.parse_args()
    
    background_color = tuple(args.background_color)
    
    # Set up directories
    script_dir = Path(__file__).parent
    screenshots_dir = script_dir / "screenshots"
    cropped_dir = script_dir / "cropped"
    
    # Check if screenshots directory exists
    if not screenshots_dir.exists():
        print(f"Error: Screenshots directory not found at {screenshots_dir}")
        print("Please create the 'screenshots' folder and add your images there.")
        return
    
    # Create cropped directory if it doesn't exist
    cropped_dir.mkdir(exist_ok=True)
    
    # Get all images
    images = list(iter_images(screenshots_dir))
    if not images:
        print(f"No images found in {screenshots_dir}")
        return
    
    print(f"Found {len(images)} image(s) to process")
    print(f"Output directory: {cropped_dir}")
    print(f"Background color for transparency: RGB{background_color}")
    print("-" * 50)
    
    # Process all images with sequential counter
    counter = 1
    total_tiles = 0
    
    for img_path in sorted(images):  # Sort for consistent processing order
        old_counter = counter
        counter = crop_image_to_tiles(img_path, counter=counter, background_color=background_color)
        tiles_from_this_image = counter - old_counter
        total_tiles += tiles_from_this_image
    
    print("-" * 50)
    print(f"Complete! Processed {len(images)} image(s)")
    print(f"Created {total_tiles} tiles in {cropped_dir}")
    print(f"Tiles are named: 001.jpg to {total_tiles:03d}.jpg")

if __name__ == '__main__':
    main()
