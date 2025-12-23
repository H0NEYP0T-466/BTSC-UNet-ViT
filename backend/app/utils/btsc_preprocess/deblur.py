"""
Deblurring Module
=================
Deblurring and motion artifact correction for medical imaging.

Provides three primary methods:
1. Gaussian Blur: Wiener/Richardson-Lucy with Gaussian PSF
2. Bilateral/Median Blur: Edge-aware Unsharp Masking
3. Motion Artifacts: RL deconvolution with estimated motion PSF
"""
import time
import numpy as np
import cv2
from typing import Optional, Tuple

try:
    from skimage.restoration import wiener, richardson_lucy, unsupervised_wiener
    from skimage.filters import gaussian
    from scipy.ndimage import convolve
    from scipy.signal import convolve2d
    RESTORATION_AVAILABLE = True
except ImportError:
    RESTORATION_AVAILABLE = False


def estimate_gaussian_psf_sigma(img: np.ndarray) -> float:
    """
    Estimate Gaussian blur sigma from edge width analysis.
    
    Uses variance of Laplacian (Tenengrad) to estimate blur level,
    then maps to approximate PSF sigma.
    
    Args:
        img: Input grayscale image (uint8)
        
    Returns:
        Estimated sigma for Gaussian PSF (typical range: 0.5-3.0)
    """
    # Compute Laplacian variance (blur metric)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    lap_var = laplacian.var()
    
    # Map variance to sigma (empirical mapping for typical MRI)
    # Low variance = more blur = higher sigma
    # High variance = sharp = lower sigma
    if lap_var > 100:
        sigma = 0.5  # Very sharp, minimal blur
    elif lap_var > 50:
        sigma = 1.0
    elif lap_var > 20:
        sigma = 1.5
    elif lap_var > 10:
        sigma = 2.0
    else:
        sigma = 2.5  # Very blurry
    
    return sigma


def create_gaussian_psf(sigma: float, size: int = 15) -> np.ndarray:
    """
    Create 2D Gaussian PSF kernel.
    
    Args:
        sigma: Standard deviation of Gaussian
        size: Kernel size (must be odd)
        
    Returns:
        Normalized Gaussian PSF
    """
    if size % 2 == 0:
        size += 1
    
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def deblur_gaussian_wiener(
    img: np.ndarray,
    sigma: Optional[float] = None,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Deblur Gaussian-blurred images using Wiener or Richardson-Lucy.
    
    Estimates Gaussian PSF sigma from edge width if not provided,
    then applies unsupervised Wiener or Richardson-Lucy deconvolution.
    
    Performance target: < 250ms for 512×512 images
    
    Args:
        img: Input grayscale image (uint8)
        sigma: Gaussian PSF sigma (auto-estimated if None)
        image_id: Optional image identifier for logging
        
    Returns:
        Deblurred image (uint8)
        
    References:
        Richardson (1972), Lucy (1974): RL deconvolution
        Wiener (1949): Wiener filtering
    """
    start_time = time.time()
    
    if not RESTORATION_AVAILABLE:
        print("[Deblur Gaussian] Warning: restoration libs not available, using sharpening fallback")
        return deblur_edge_aware_usm(img, image_id=image_id)
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Estimate sigma if not provided
    if sigma is None:
        sigma = estimate_gaussian_psf_sigma(img)
    
    sigma = np.clip(sigma, 0.5, 3.0)
    
    # Convert to float [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    # Create Gaussian PSF
    psf = create_gaussian_psf(sigma, size=min(15, int(6*sigma)+1))
    
    try:
        # Try unsupervised Wiener (auto-estimates noise)
        deblurred = unsupervised_wiener(img_float, psf, max_num_iter=15)[0]
    except Exception as e:
        try:
            # Fallback to Richardson-Lucy
            deblurred = richardson_lucy(img_float, psf, num_iter=10, clip=False)
        except Exception as e2:
            print(f"[Deblur Gaussian] Both methods failed, using USM fallback")
            return deblur_edge_aware_usm(img, image_id=image_id)
    
    # Convert back to uint8
    deblurred = np.clip(deblurred, 0, 1)
    result = (deblurred * 255).astype(np.uint8)
    
    duration = time.time() - start_time
    if image_id:
        print(f"[Deblur Gaussian] Processed in {duration*1000:.1f}ms, sigma={sigma:.2f}")
    
    return result


def deblur_edge_aware_usm(
    img: np.ndarray,
    amount: Optional[float] = None,
    radius: Optional[float] = None,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Deblur using edge-aware Unsharp Masking (USM).
    
    Pre-smooths with bilateral filter to suppress noise amplification,
    then applies unsharp mask with auto-tuned parameters based on blur level.
    
    Suitable for bilateral/median blur or mild Gaussian blur.
    
    Performance target: < 100ms for 512×512 images
    
    Args:
        img: Input grayscale image (uint8)
        amount: Sharpening strength (auto-tuned if None)
        radius: Blur radius for USM (auto-tuned if None)
        image_id: Optional image identifier for logging
        
    Returns:
        Deblurred image (uint8)
    """
    start_time = time.time()
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Estimate blur level
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    lap_var = laplacian.var()
    
    # Auto-tune parameters based on blur level
    if amount is None:
        if lap_var < 10:
            amount = 1.8  # Very blurry, more aggressive
        elif lap_var < 50:
            amount = 1.4
        else:
            amount = 1.0  # Sharp, gentle sharpening
    
    if radius is None:
        radius = 2.0 if lap_var < 20 else 1.5
    
    # Pre-smooth with bilateral to suppress noise
    smoothed = cv2.bilateralFilter(img, 5, 30, 30)
    
    # Apply Gaussian blur for USM
    blurred = cv2.GaussianBlur(smoothed, (0, 0), radius)
    
    # Unsharp mask: original + amount * (original - blurred)
    result = cv2.addWeighted(smoothed, 1 + amount, blurred, -amount, 0)
    
    # Clamp to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    duration = time.time() - start_time
    if image_id:
        print(f"[Deblur USM] Processed in {duration*1000:.1f}ms, amount={amount:.2f}, radius={radius:.2f}")
    
    return result


def estimate_motion_kernel(
    img: np.ndarray
) -> Tuple[float, float]:
    """
    Estimate motion blur kernel parameters from log-spectrum.
    
    Uses Radon transform and log-spectrum analysis to detect directional
    motion streaks and estimate angle and length.
    
    Args:
        img: Input grayscale image (uint8)
        
    Returns:
        Tuple of (angle_degrees, length_pixels)
        Falls back to (0, 0) if detection fails
    """
    try:
        # FFT and log-spectrum
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        log_spectrum = np.log(magnitude_spectrum + 1)
        
        # Normalize for peak detection
        log_spectrum = (log_spectrum - log_spectrum.min()) / (log_spectrum.max() - log_spectrum.min() + 1e-8)
        
        # Simple directional energy detection (simplified Radon)
        # Check horizontal and vertical streak energy
        h, w = log_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Sample a strip through center
        margin = h // 8
        horizontal_strip = log_spectrum[center_h-margin:center_h+margin, :]
        vertical_strip = log_spectrum[:, center_w-margin:center_w+margin]
        
        h_energy = np.var(horizontal_strip)
        v_energy = np.var(vertical_strip)
        
        # Estimate angle and length based on dominant direction
        if h_energy > v_energy * 1.2:
            # Horizontal motion (angle near 0 or 180)
            angle = 0.0
            # Estimate length from spectrum width
            length = min(15, max(5, int(h_energy * 20)))
        elif v_energy > h_energy * 1.2:
            # Vertical motion (angle near 90)
            angle = 90.0
            length = min(15, max(5, int(v_energy * 20)))
        else:
            # Minimal or diagonal motion, use default
            angle = 0.0
            length = 5
        
        return angle, float(length)
        
    except Exception as e:
        # Detection failed, return default
        return 0.0, 0.0


def create_motion_psf(angle: float, length: float, size: int = 15) -> np.ndarray:
    """
    Create motion blur PSF kernel.
    
    Args:
        angle: Motion angle in degrees
        length: Motion length in pixels
        size: Kernel size (must be odd)
        
    Returns:
        Normalized motion PSF
    """
    if length < 1:
        # No motion, return delta function
        psf = np.zeros((size, size))
        psf[size//2, size//2] = 1.0
        return psf
    
    psf = np.zeros((size, size))
    center = size // 2
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Draw line representing motion
    half_len = int(length / 2)
    for i in range(-half_len, half_len + 1):
        x = int(center + i * cos_a)
        y = int(center + i * sin_a)
        if 0 <= x < size and 0 <= y < size:
            psf[y, x] = 1.0
    
    # Normalize
    psf = psf / (np.sum(psf) + 1e-8)
    
    return psf


def correct_motion_artifacts(
    img: np.ndarray,
    angle: Optional[float] = None,
    length: Optional[float] = None,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Correct patient motion artifacts (PMA).
    
    Estimates linear motion kernel via log-spectrum + Radon peak detection,
    or uses provided parameters. Applies Richardson-Lucy or unsupervised
    Wiener with TV regularization, then light NLM post-smoothing.
    
    Performance target: < 300ms for 512×512 images
    
    Args:
        img: Input grayscale image (uint8)
        angle: Motion angle in degrees (auto-estimated if None)
        length: Motion length in pixels (auto-estimated if None)
        image_id: Optional image identifier for logging
        
    Returns:
        Motion-corrected image (uint8)
        
    References:
        Fish et al. (1992): MRI motion artifact reduction
    """
    start_time = time.time()
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Estimate motion parameters if not provided
    if angle is None or length is None:
        est_angle, est_length = estimate_motion_kernel(img)
        if angle is None:
            angle = est_angle
        if length is None:
            length = est_length
    
    # If no significant motion detected, return light bilateral filter
    if length < 3:
        result = cv2.bilateralFilter(img, 3, 20, 20)
        duration = time.time() - start_time
        if image_id:
            print(f"[PMA] No significant motion detected, light filter in {duration*1000:.1f}ms")
        return result
    
    if not RESTORATION_AVAILABLE:
        print("[PMA] Warning: restoration libs not available, using bilateral fallback")
        result = cv2.bilateralFilter(img, 5, 30, 30)
        duration = time.time() - start_time
        if image_id:
            print(f"[PMA] Fallback filter in {duration*1000:.1f}ms")
        return result
    
    # Convert to float [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    # Create motion PSF
    psf = create_motion_psf(angle, length, size=15)
    
    try:
        # Try Richardson-Lucy deconvolution (faster)
        corrected = richardson_lucy(img_float, psf, num_iter=10, clip=False)
    except Exception as e:
        try:
            # Fallback to unsupervised Wiener
            corrected = unsupervised_wiener(img_float, psf, max_num_iter=10)[0]
        except Exception as e2:
            print(f"[PMA] Deconvolution failed, using bilateral fallback")
            result = cv2.bilateralFilter(img, 5, 30, 30)
            duration = time.time() - start_time
            if image_id:
                print(f"[PMA] Fallback in {duration*1000:.1f}ms")
            return result
    
    # Convert back to uint8
    corrected = np.clip(corrected, 0, 1)
    result = (corrected * 255).astype(np.uint8)
    
    # Light NLM post-smoothing to reduce ringing artifacts
    result = cv2.fastNlMeansDenoising(result, None, h=5, templateWindowSize=5, searchWindowSize=15)
    
    duration = time.time() - start_time
    if image_id:
        print(f"[PMA] Corrected in {duration*1000:.1f}ms, angle={angle:.1f}°, length={length:.1f}px")
    
    return result
