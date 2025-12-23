"""
Quality Detection Module
========================
Auto-detect image quality issues for intelligent preprocessing.

Provides detectors for:
1. Noise type: Salt & pepper vs Gaussian vs Speckle
2. Blur level: Variance of Laplacian and Tenengrad
3. Motion artifacts: Spectral streak score
"""
import time
import numpy as np
import cv2
from typing import Dict, Optional, Tuple


def detect_noise_type(
    img: np.ndarray,
    image_id: Optional[str] = None
) -> Dict[str, any]:
    """
    Detect predominant noise type in image.
    
    Uses multiple heuristics:
    - Impulse fraction: count of 0/255 pixels (salt & pepper)
    - Sigma estimate: Laplacian-based noise estimation (Gaussian)
    - Log-domain variance: multiplicative noise test (Speckle)
    
    Args:
        img: Input grayscale image (uint8)
        image_id: Optional image identifier for logging
        
    Returns:
        Dictionary with:
            - type: "salt_pepper" | "gaussian" | "speckle" | "none"
            - scores: dict with confidence scores for each type
            - details: dict with diagnostic values
    """
    start_time = time.time()
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    results = {
        'type': 'none',
        'scores': {},
        'details': {}
    }
    
    # Test 1: Impulse noise (salt & pepper)
    # Count pixels at extremes (0 or 255)
    # Exclude background by looking at central region or using simple threshold
    # to avoid counting background 0s as impulse noise
    impulse_mask = (img == 0) | (img == 255)
    
    # Create a simple foreground mask (pixels > 10)
    foreground = img > 10
    foreground_pixels = np.sum(foreground)
    
    if foreground_pixels > 0:
        # Only count impulses in foreground
        impulse_in_fg = np.sum(impulse_mask & foreground)
        impulse_fraction = impulse_in_fg / foreground_pixels
    else:
        # No foreground, fall back to global
        impulse_fraction = np.mean(impulse_mask)
    
    results['scores']['salt_pepper'] = min(impulse_fraction * 100, 1.0)  # Scale to [0, 1]
    results['details']['impulse_fraction'] = impulse_fraction
    
    # Test 2: Gaussian noise
    # Estimate noise sigma using MAD on Laplacian
    img_float = img.astype(np.float32)
    laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
    sigma_estimate = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745
    # Normalize to [0, 1] score (sigma > 10 is noisy)
    results['scores']['gaussian'] = min(sigma_estimate / 30.0, 1.0)
    results['details']['sigma_estimate'] = sigma_estimate
    
    # Test 3: Speckle (multiplicative) noise
    # Check variance in log-domain
    img_safe = np.clip(img_float, 1, 255)
    log_img = np.log(img_safe)
    log_var = np.var(log_img)
    # High log variance indicates multiplicative noise
    # Normalize to [0, 1] score
    results['scores']['speckle'] = min(log_var / 2.0, 1.0)
    results['details']['log_variance'] = log_var
    
    # Determine predominant type
    scores = results['scores']
    max_score_type = max(scores, key=scores.get)
    max_score = scores[max_score_type]
    
    # Thresholds for detection
    if max_score_type == 'salt_pepper' and impulse_fraction > 0.01:
        results['type'] = 'salt_pepper'
    elif max_score_type == 'gaussian' and sigma_estimate > 8:
        results['type'] = 'gaussian'
    elif max_score_type == 'speckle' and log_var > 0.5:
        results['type'] = 'speckle'
    else:
        results['type'] = 'none'
    
    duration = time.time() - start_time
    if image_id:
        print(f"[Noise Detect] Type={results['type']} in {duration*1000:.1f}ms, "
              f"impulse={impulse_fraction*100:.2f}%, sigma={sigma_estimate:.1f}, log_var={log_var:.3f}")
    
    return results


def detect_blur(
    img: np.ndarray,
    image_id: Optional[str] = None
) -> Dict[str, any]:
    """
    Detect blur level using variance of Laplacian and Tenengrad.
    
    Lower variance = more blur.
    
    Args:
        img: Input grayscale image (uint8)
        image_id: Optional image identifier for logging
        
    Returns:
        Dictionary with:
            - is_blurred: bool
            - laplacian_var: float (variance of Laplacian)
            - tenengrad: float (gradient magnitude metric)
            - blur_score: float in [0, 1] (0=sharp, 1=very blurred)
    """
    start_time = time.time()
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Variance of Laplacian (Pech-Pacheco et al.)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    lap_var = laplacian.var()
    
    # Tenengrad (Sobel-based gradient magnitude)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(grad_x**2 + grad_y**2)
    
    # Blur score: map lap_var to [0, 1]
    # Empirical thresholds for MRI:
    # lap_var > 100: sharp (score=0)
    # lap_var < 10: very blurred (score=1)
    if lap_var > 100:
        blur_score = 0.0
    elif lap_var < 10:
        blur_score = 1.0
    else:
        # Linear interpolation
        blur_score = 1.0 - (lap_var - 10) / 90.0
    
    is_blurred = lap_var < 50  # Threshold for "needs deblurring"
    
    duration = time.time() - start_time
    if image_id:
        print(f"[Blur Detect] Blurred={is_blurred} in {duration*1000:.1f}ms, "
              f"lap_var={lap_var:.1f}, tenengrad={tenengrad:.1f}, blur_score={blur_score:.2f}")
    
    return {
        'is_blurred': is_blurred,
        'laplacian_var': lap_var,
        'tenengrad': tenengrad,
        'blur_score': blur_score
    }


def detect_motion(
    img: np.ndarray,
    image_id: Optional[str] = None
) -> Dict[str, any]:
    """
    Detect motion artifacts via directional energy in log-spectrum.
    
    Motion blur creates characteristic streaks in the frequency domain.
    
    Args:
        img: Input grayscale image (uint8)
        image_id: Optional image identifier for logging
        
    Returns:
        Dictionary with:
            - has_motion: bool
            - streak_score: float in [0, 1]
            - angle_estimate: float (degrees)
            - details: dict with diagnostic values
    """
    start_time = time.time()
    
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    results = {
        'has_motion': False,
        'streak_score': 0.0,
        'angle_estimate': 0.0,
        'details': {}
    }
    
    try:
        # FFT and log-spectrum
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        log_spectrum = np.log(magnitude_spectrum + 1)
        
        # Normalize
        log_spectrum = (log_spectrum - log_spectrum.min()) / (log_spectrum.max() - log_spectrum.min() + 1e-8)
        
        # Detect directional energy (simplified)
        h, w = log_spectrum.shape
        center_h, center_w = h // 2, w // 2
        margin = h // 8
        
        # Sample strips through center
        horizontal_strip = log_spectrum[center_h-margin:center_h+margin, :]
        vertical_strip = log_spectrum[:, center_w-margin:center_w+margin]
        
        h_energy = np.var(horizontal_strip)
        v_energy = np.var(vertical_strip)
        
        # Overall energy
        mean_energy = (h_energy + v_energy) / 2
        
        # Streak score: higher variance in one direction suggests motion
        energy_ratio = max(h_energy, v_energy) / (min(h_energy, v_energy) + 1e-8)
        streak_score = min((energy_ratio - 1.0) / 10.0, 1.0)  # Normalize
        
        results['streak_score'] = streak_score
        results['details']['h_energy'] = h_energy
        results['details']['v_energy'] = v_energy
        results['details']['energy_ratio'] = energy_ratio
        
        # Estimate angle
        if h_energy > v_energy * 1.2:
            results['angle_estimate'] = 0.0  # Horizontal motion
        elif v_energy > h_energy * 1.2:
            results['angle_estimate'] = 90.0  # Vertical motion
        else:
            results['angle_estimate'] = 45.0  # Diagonal or unclear
        
        # Threshold for motion detection
        results['has_motion'] = streak_score > 0.15 or energy_ratio > 2.0
        
    except Exception as e:
        print(f"[Motion Detect] Error: {e}")
        results['has_motion'] = False
    
    duration = time.time() - start_time
    if image_id:
        print(f"[Motion Detect] Motion={results['has_motion']} in {duration*1000:.1f}ms, "
              f"streak_score={results['streak_score']:.2f}, angle={results['angle_estimate']:.1f}°")
    
    return results
