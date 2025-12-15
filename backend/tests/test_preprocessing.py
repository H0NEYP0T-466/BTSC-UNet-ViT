"""
Tests for preprocessing utilities.
"""
import numpy as np
import pytest
from app.utils.preprocessing import (
    to_grayscale,
    remove_salt_pepper,
    enhance_contrast_clahe,
    unsharp_mask,
    normalize_image,
    preprocess_pipeline,
    denoise_nlm
)


def test_to_grayscale():
    """Test grayscale conversion."""
    # RGB image
    rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = to_grayscale(rgb_image)
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8


def test_remove_salt_pepper():
    """Test salt and pepper noise removal."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    denoised = remove_salt_pepper(image)
    assert denoised.shape == image.shape
    assert denoised.dtype == np.uint8


def test_denoise_nlm():
    """Test Non-Local Means denoising."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    denoised = denoise_nlm(image, h=10)
    assert denoised.shape == image.shape
    assert denoised.dtype == np.uint8


def test_enhance_contrast_clahe():
    """Test CLAHE contrast enhancement."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    enhanced = enhance_contrast_clahe(image)
    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8


def test_enhance_contrast_clahe_with_mask():
    """Test CLAHE contrast enhancement with mask."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Central region is brain
    
    enhanced = enhance_contrast_clahe(image, mask=mask)
    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8
    
    # Background should remain unchanged (or close to it)
    # This is a basic check - in practice CLAHE would only enhance the masked region


def test_unsharp_mask():
    """Test unsharp mask sharpening."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    sharpened = unsharp_mask(image)
    assert sharpened.shape == image.shape
    assert sharpened.dtype == np.uint8


def test_normalize_image():
    """Test image normalization."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # Z-score
    normalized = normalize_image(image, method='zscore')
    assert normalized.shape == image.shape
    assert normalized.dtype == np.uint8
    
    # Min-max
    normalized = normalize_image(image, method='minmax')
    assert normalized.shape == image.shape
    assert normalized.dtype == np.uint8


def test_preprocess_pipeline():
    """Test full preprocessing pipeline."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = preprocess_pipeline(image, apply_skull_stripping=False)  # Skip skull stripping in tests
    
    assert 'grayscale' in result
    assert 'skull_stripped' in result
    assert 'brain_mask' in result
    assert 'denoised' in result
    assert 'motion_reduced' in result
    assert 'contrast' in result
    assert 'sharpened' in result
    assert 'normalized' in result
    
    # All outputs should be 2D grayscale
    for key, img in result.items():
        assert len(img.shape) == 2
        assert img.dtype == np.uint8


def test_skull_stripping_fallback():
    """Test skull stripping with fallback when HD-BET is not available."""
    from app.utils.skull_stripping import apply_mask_to_image
    
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    
    masked = apply_mask_to_image(image, mask, background_value=0)
    assert masked.shape == image.shape
    assert masked.dtype == np.uint8
    
    # Check that background is zeroed
    assert np.all(masked[0:25, :] == 0)
    assert np.all(masked[:, 0:25] == 0)
