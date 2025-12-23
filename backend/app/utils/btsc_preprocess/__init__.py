"""
BTSC Image Quality Preprocessing Module
========================================
Robust, clinically-aware image quality preprocessing for MRI scans.

This module provides specialized noise reduction, deblurring, motion artifact
correction, and quality-aware enhancement operations for medical imaging.
"""

from .noise import (
    remove_salt_and_pepper,
    denoise_gaussian_nlmeans,
    denoise_speckle_wavelet,
)

from .deblur import (
    deblur_gaussian_wiener,
    deblur_edge_aware_usm,
    correct_motion_artifacts,
)

from .contrast import (
    clahe_enhance,
    sharpen_noise_aware,
)

from .detect import (
    detect_noise_type,
    detect_blur,
    detect_motion,
)

__all__ = [
    # Noise removal
    'remove_salt_and_pepper',
    'denoise_gaussian_nlmeans',
    'denoise_speckle_wavelet',
    # Deblurring
    'deblur_gaussian_wiener',
    'deblur_edge_aware_usm',
    'correct_motion_artifacts',
    # Contrast & sharpening
    'clahe_enhance',
    'sharpen_noise_aware',
    # Quality detection
    'detect_noise_type',
    'detect_blur',
    'detect_motion',
]
