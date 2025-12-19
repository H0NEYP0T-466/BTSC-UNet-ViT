#!/usr/bin/env python3
"""
Build NFBS histogram reference for histogram matching.

This script processes a subset of NFBS training images to create a reference
histogram that can be used to harmonize external data to NFBS characteristics.

The reference includes:
- Mean intensity histogram across the dataset
- Global mean and standard deviation
- Histogram bins and frequencies

Usage:
    python tools/build_nfbs_hist_ref.py --nfbs_path /path/to/nfbs/images --output artifacts/nfbs_hist_ref.npz

Requirements:
    - NFBS dataset images in a directory (e.g., PNG, JPG, or NIfTI format)
    - Output directory must exist

Author: BTSC Team
Date: 2025-12-19
"""
import argparse
import os
from pathlib import Path
from typing import List
import numpy as np
import cv2
import nibabel as nib
from tqdm import tqdm


def load_nfbs_images(nfbs_path: str, max_images: int = 50, extensions: List[str] = None) -> List[np.ndarray]:
    """
    Load NFBS images from directory.
    
    Args:
        nfbs_path: Path to NFBS dataset directory
        max_images: Maximum number of images to load (default: 50)
        extensions: List of file extensions to consider (default: ['.png', '.jpg', '.jpeg'])
        
    Returns:
        List of image arrays
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.nii', '.nii.gz']
    
    nfbs_dir = Path(nfbs_path)
    
    if not nfbs_dir.exists():
        raise ValueError(f"NFBS path does not exist: {nfbs_path}")
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(list(nfbs_dir.rglob(f"*{ext}")))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {nfbs_path} with extensions {extensions}")
    
    print(f"Found {len(image_files)} images in {nfbs_path}")
    
    # Limit to max_images
    if len(image_files) > max_images:
        print(f"Limiting to first {max_images} images")
        image_files = image_files[:max_images]
    
    # Load images
    images = []
    for img_path in tqdm(image_files, desc="Loading NFBS images"):
        try:
            if img_path.suffix in ['.nii', '.gz']:
                # Load NIfTI
                nii = nib.load(str(img_path))
                img_data = nii.get_fdata()
                
                # For 3D volumes, take middle slice
                if img_data.ndim == 3:
                    mid_slice = img_data.shape[2] // 2
                    img = img_data[:, :, mid_slice]
                else:
                    img = img_data
            else:
                # Load regular image
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is not None and img.size > 0:
                # Normalize to float32
                img = img.astype(np.float32)
                images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue
    
    print(f"Successfully loaded {len(images)} images")
    return images


def build_reference_histogram(images: List[np.ndarray], num_bins: int = 256) -> dict:
    """
    Build reference histogram from list of images.
    
    Args:
        images: List of image arrays
        num_bins: Number of histogram bins (default: 256)
        
    Returns:
        Dictionary with reference statistics
    """
    print("Computing reference histogram...")
    
    # Collect all pixel values
    all_values = []
    for img in tqdm(images, desc="Collecting pixel values"):
        all_values.append(img.flatten())
    
    all_values = np.concatenate(all_values)
    
    # Compute statistics
    mean = np.mean(all_values)
    std = np.std(all_values)
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    
    print(f"Dataset statistics:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Std: {std:.2f}")
    print(f"  Min: {min_val:.2f}")
    print(f"  Max: {max_val:.2f}")
    
    # Compute histogram
    hist, bin_edges = np.histogram(all_values, bins=num_bins, density=True)
    
    # Create a representative reference image by averaging
    # Resize all images to a common size first
    target_size = (256, 256)
    resized_images = []
    for img in tqdm(images, desc="Resizing images"):
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        resized_images.append(resized)
    
    # Average all images
    reference_img = np.mean(resized_images, axis=0).astype(np.float32)
    
    # Package reference data
    reference_data = {
        'reference_hist': reference_img,
        'histogram': hist,
        'bin_edges': bin_edges,
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'num_images': len(images)
    }
    
    return reference_data


def save_reference(reference_data: dict, output_path: str):
    """
    Save reference data to NPZ file.
    
    Args:
        reference_data: Dictionary with reference statistics
        output_path: Path to output .npz file
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(output_path, **reference_data)
    
    print(f"\nReference saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build NFBS histogram reference for preprocessing pipeline"
    )
    parser.add_argument(
        '--nfbs_path',
        type=str,
        required=True,
        help='Path to NFBS dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='artifacts/nfbs_hist_ref.npz',
        help='Output path for reference file (default: artifacts/nfbs_hist_ref.npz)'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=50,
        help='Maximum number of images to process (default: 50)'
    )
    parser.add_argument(
        '--num_bins',
        type=int,
        default=256,
        help='Number of histogram bins (default: 256)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NFBS Histogram Reference Builder")
    print("=" * 80)
    print(f"NFBS path: {args.nfbs_path}")
    print(f"Output: {args.output}")
    print(f"Max images: {args.max_images}")
    print(f"Histogram bins: {args.num_bins}")
    print()
    
    # Load images
    images = load_nfbs_images(args.nfbs_path, max_images=args.max_images)
    
    if len(images) == 0:
        print("Error: No images loaded. Please check the NFBS path.")
        return 1
    
    # Build reference
    reference_data = build_reference_histogram(images, num_bins=args.num_bins)
    
    # Save reference
    save_reference(reference_data, args.output)
    
    print("\n" + "=" * 80)
    print("Reference build complete!")
    print("=" * 80)
    print("\nTo use this reference in preprocessing:")
    print(f"  1. Ensure the file exists at: {args.output}")
    print(f"  2. Set histogram_match_to_nfbs.enabled: true in btsc/configs/brain_preproc.yaml")
    print(f"  3. Set histogram_match_to_nfbs.reference_stats_path: {args.output}")
    print("\nTo regenerate this reference:")
    print(f"  python tools/build_nfbs_hist_ref.py --nfbs_path <path> --output {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
