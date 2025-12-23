"""
Contrast & Sharpening Module
=============================
MRI-safe contrast enhancement and noise-aware sharpening.

Provides:
1. CLAHE: Conservative Contrast Limited Adaptive Histogram Equalization
2. Noise-aware USM: Unsharp masking with detail mask thresholding
"""
import time
import numpy as np
import cv2
from typing import Optional, Tuple


def clahe_enhance(
    img: np.ndarray,
    clipLimit: float = 2.0,
    tileGrid: Tuple[int, int] = (8, 8),
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Enhance contrast using CLAHE with conservative parameters.
    
    MRI-safe: preserves grayscale, avoids over-enhancement that creates
    white speckles or halo artifacts.
    
    Performance target: < 20ms for 512×512 images
    
    Args:
        img: Input grayscale image (uint8)
        clipLimit: Threshold for contrast limiting (default: 2.0)
                   Lower = less aggressive, fewer artifacts
        tileGrid: Grid size for histogram equalization (default: (8, 8))
        image_id: Optional image identifier for logging
        
    Returns:
        Contrast-enhanced image (uint8)
        
    References:
        Zuiderveld (1994): CLAHE algorithm
    """
    start_time = time.time()
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGrid)
    
    # Apply CLAHE
    enhanced = clahe.apply(img)
    
    duration = time.time() - start_time
    if image_id:
        print(f"[CLAHE] Enhanced in {duration*1000:.1f}ms, clipLimit={clipLimit}, grid={tileGrid}")
    
    return enhanced


def sharpen_noise_aware(
    img: np.ndarray,
    radius: float = 1.5,
    amount: float = 1.2,
    threshold: float = 0.01,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Apply noise-aware unsharp masking with detail mask thresholding.
    
    Algorithm:
    1. Bilateral/guided filter to create edge-preserving blur
    2. Compute detail mask from gradient magnitude
    3. Threshold detail mask to avoid sharpening noise/uniform regions
    4. Apply unsharp mask only where detail > threshold
    5. Clamp to avoid overshoot
    
    Performance target: < 20ms for 512×512 images
    
    Args:
        img: Input grayscale image (uint8)
        radius: Gaussian blur radius for USM (default: 1.5)
        amount: Sharpening strength (default: 1.2)
        threshold: Detail threshold [0, 1] (default: 0.01)
                   Higher = sharpen only strong edges
        image_id: Optional image identifier for logging
        
    Returns:
        Sharpened image (uint8)
        
    References:
        Polesel et al. (2000): Adaptive unsharp masking
    """
    start_time = time.time()
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Convert to float for processing
    img_float = img.astype(np.float32) / 255.0
    
    # Create edge-preserving blur using bilateral filter
    # This preserves edges while smoothing uniform regions
    blurred = cv2.bilateralFilter(img, 5, 30, 30)
    blurred_float = blurred.astype(np.float32) / 255.0
    
    # Compute detail (high-frequency component)
    detail = img_float - blurred_float
    
    # Create detail mask from gradient magnitude
    # This identifies where edges/details are present
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude to [0, 1]
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    
    # Threshold: only sharpen where gradient > threshold
    detail_mask = (grad_mag > threshold).astype(np.float32)
    
    # Apply adaptive sharpening
    # Sharpen more where detail_mask is high (edges)
    sharpened = img_float + amount * detail * detail_mask
    
    # Clamp to avoid overshoot
    sharpened = np.clip(sharpened, 0, 1)
    
    # Convert back to uint8
    result = (sharpened * 255).astype(np.uint8)
    
    duration = time.time() - start_time
    if image_id:
        detail_pct = np.mean(detail_mask) * 100
        print(f"[Sharpen] Processed in {duration*1000:.1f}ms, radius={radius}, amount={amount}, detail={detail_pct:.1f}%")
    
    return result
