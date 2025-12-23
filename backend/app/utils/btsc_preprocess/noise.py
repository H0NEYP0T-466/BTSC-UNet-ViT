"""
Noise Removal Module
====================
Specialized noise reduction for medical imaging.

Provides three primary methods:
1. Salt & Pepper: Adaptive median filtering with impulse detection
2. Gaussian Noise: Fast Non-Local Means (NLM) tuned for MRI
3. Speckle Noise: Wavelet-based BayesShrink denoising
"""
import time
import numpy as np
import cv2
from typing import Optional

try:
    from skimage.restoration import denoise_wavelet, estimate_sigma
    from skimage import img_as_float, img_as_ubyte
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def remove_salt_and_pepper(
    img: np.ndarray,
    max_kernel: int = 7,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Remove salt-and-pepper (impulse) noise using adaptive median filtering.
    
    Algorithm grows kernel size (3→5→7) adaptively based on impulse detection.
    Falls back to standard median blur for uniform processing.
    
    Performance target: < 30ms for 512×512 images
    
    Args:
        img: Input grayscale image (uint8)
        max_kernel: Maximum kernel size for adaptive median (default: 7)
        image_id: Optional image identifier for logging
        
    Returns:
        Denoised image (uint8)
        
    References:
        Hwang & Haddad (1995): Adaptive median filters
    """
    start_time = time.time()
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Detect impulse noise (0 or 255 spikes)
    impulse_mask = (img == 0) | (img == 255)
    impulse_fraction = np.mean(impulse_mask)
    
    # If significant impulse noise detected, use adaptive approach
    if impulse_fraction > 0.01:  # > 1% impulse pixels
        # Simple adaptive median: grow kernel for impulse pixels
        result = img.copy()
        
        # First pass: 3x3 median on all impulse pixels
        kernel_3 = cv2.medianBlur(img, 3)
        result[impulse_mask] = kernel_3[impulse_mask]
        
        # Second pass: 5x5 for still-impulse pixels
        still_impulse = (result == 0) | (result == 255)
        if np.any(still_impulse):
            kernel_5 = cv2.medianBlur(img, 5)
            result[still_impulse] = kernel_5[still_impulse]
        
        # Third pass: 7x7 for remaining impulse pixels (if max_kernel allows)
        if max_kernel >= 7:
            still_impulse = (result == 0) | (result == 255)
            if np.any(still_impulse):
                kernel_7 = cv2.medianBlur(img, 7)
                result[still_impulse] = kernel_7[still_impulse]
    else:
        # Low impulse noise: use simple median blur
        result = cv2.medianBlur(img, 3)
    
    duration = time.time() - start_time
    if image_id:
        print(f"[S&P] Denoised in {duration*1000:.1f}ms, impulse={impulse_fraction*100:.2f}%")
    
    return result


def denoise_gaussian_nlmeans(
    img: np.ndarray,
    h_scale: float = 0.8,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Denoise Gaussian noise using fast Non-Local Means (NLM).
    
    Estimates noise sigma and applies cv2.fastNlMeansDenoising with
    tuned parameters for MRI. Falls back to skimage if OpenCV unavailable.
    
    Performance target: < 120ms for 512×512 images
    
    Args:
        img: Input grayscale image (uint8)
        h_scale: Filter strength multiplier (default: 0.8)
                 h = h_scale * estimated_sigma
        image_id: Optional image identifier for logging
        
    Returns:
        Denoised image (uint8)
        
    References:
        Buades et al. (2005): Non-Local Means denoising
    """
    start_time = time.time()
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Estimate noise sigma using robust MAD estimator
    # Median Absolute Deviation in high-frequency component
    img_float = img.astype(np.float32)
    # Simple Laplacian-based estimate
    laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
    sigma_estimate = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745
    
    # Clamp sigma to reasonable range
    sigma_estimate = np.clip(sigma_estimate, 5, 30)
    
    # Compute h parameter
    h = int(h_scale * sigma_estimate)
    h = max(5, min(h, 20))  # Clamp to [5, 20]
    
    # Use OpenCV's fast NLM (optimized)
    try:
        result = cv2.fastNlMeansDenoising(
            img, 
            None, 
            h=h, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
    except Exception as e:
        # Fallback: try skimage if OpenCV fails
        if SKIMAGE_AVAILABLE:
            img_norm = img_as_float(img)
            result_norm = denoise_nl_means_skimage(img_norm, h=h/255.0)
            result = img_as_ubyte(np.clip(result_norm, 0, 1))
        else:
            # Last resort: bilateral filter
            result = cv2.bilateralFilter(img, 5, 50, 50)
    
    duration = time.time() - start_time
    if image_id:
        print(f"[Gaussian NLM] Denoised in {duration*1000:.1f}ms, h={h}, sigma_est={sigma_estimate:.1f}")
    
    return result


def denoise_nl_means_skimage(img_float, h=0.1):
    """Fallback NLM using skimage (slower but more compatible)."""
    if not SKIMAGE_AVAILABLE:
        return img_float
    
    from skimage.restoration import denoise_nl_means
    sigma_est = estimate_sigma(img_float, average_sigmas=True)
    return denoise_nl_means(
        img_float,
        h=h,
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        preserve_range=True
    )


def denoise_speckle_wavelet(
    img: np.ndarray,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Denoise speckle (multiplicative) noise using wavelet BayesShrink.
    
    Converts to log-domain, applies wavelet soft-thresholding (BayesShrink),
    then exponentiates back. Preserves intensity range.
    
    Performance target: < 120ms for 512×512 images
    
    Args:
        img: Input grayscale image (uint8)
        image_id: Optional image identifier for logging
        
    Returns:
        Denoised image (uint8)
        
    References:
        Chang et al. (2000): BayesShrink wavelet denoising
    """
    start_time = time.time()
    
    if not SKIMAGE_AVAILABLE:
        # Fallback to median filter if skimage not available
        print("[Speckle] Warning: skimage not available, using median filter fallback")
        return cv2.medianBlur(img, 5)
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Convert to float [0, 1]
    img_float = img_as_float(img)
    
    # Add small offset to avoid log(0)
    img_float = np.clip(img_float, 0.01, 1.0)
    
    # Transform to log-domain (speckle becomes additive)
    log_img = np.log(img_float + 1e-6)
    
    # Wavelet denoising with BayesShrink
    try:
        sigma_est = estimate_sigma(log_img, average_sigmas=True)
        log_denoised = denoise_wavelet(
            log_img,
            method='BayesShrink',
            mode='soft',
            wavelet='db4',
            rescale_sigma=True,
            wavelet_levels=3
        )
    except Exception as e:
        print(f"[Speckle] Wavelet failed: {e}, using bilateral fallback")
        # Fallback to bilateral filter in log-domain
        log_img_scaled = ((log_img - log_img.min()) / (log_img.max() - log_img.min() + 1e-8) * 255).astype(np.uint8)
        log_denoised_scaled = cv2.bilateralFilter(log_img_scaled, 5, 30, 30)
        log_denoised = (log_denoised_scaled / 255.0) * (log_img.max() - log_img.min()) + log_img.min()
    
    # Transform back from log-domain
    img_denoised = np.exp(log_denoised)
    
    # Restore original range [0, 1] then to uint8
    img_denoised = np.clip(img_denoised, 0, 1)
    result = img_as_ubyte(img_denoised)
    
    duration = time.time() - start_time
    if image_id:
        print(f"[Speckle Wavelet] Denoised in {duration*1000:.1f}ms")
    
    return result
