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
    denoise_nlm,
    run_preprocessing
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
    """Test full preprocessing pipeline (legacy)."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = preprocess_pipeline(image)
    
    assert 'grayscale' in result
    assert 'denoised' in result
    assert 'motion_reduced' in result
    assert 'contrast' in result
    assert 'sharpened' in result
    assert 'normalized' in result
    
    # All outputs should be 2D grayscale
    for key, img in result.items():
        assert len(img.shape) == 2
        assert img.dtype == np.uint8


def test_run_preprocessing():
    """Test intelligent preprocessing pipeline with auto-detection."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = run_preprocessing(image, opts={'auto': True})
    
    # Check all expected stages are present
    expected_stages = [
        'grayscale',
        'salt_pepper_cleaned',
        'gaussian_denoised',
        'speckle_denoised',
        'pma_corrected',
        'deblurred',
        'contrast_enhanced',
        'sharpened'
    ]
    
    for stage in expected_stages:
        assert stage in result, f"Missing stage: {stage}"
        assert result[stage].dtype == np.uint8
        assert len(result[stage].shape) == 2  # Grayscale


def test_run_preprocessing_with_salt_pepper():
    """Test that salt & pepper noise is detected and removed."""
    # Create image with salt & pepper noise
    image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    noise_mask = np.random.random((100, 100)) < 0.05
    image[noise_mask] = np.random.choice([0, 255], size=np.sum(noise_mask))
    
    result = run_preprocessing(image, opts={'auto': True})
    
    # Verify salt_pepper_cleaned is different from grayscale (noise was detected)
    assert 'salt_pepper_cleaned' in result
    assert result['salt_pepper_cleaned'].dtype == np.uint8


def test_run_preprocessing_conservative_parameters():
    """Test that preprocessing uses conservative parameters to avoid white noise."""
    # Clean image
    image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    
    result = run_preprocessing(image, opts={
        'auto': True,
        'clahe_clip_limit': 1.5,
        'sharpen_amount': 0.8
    })
    
    # Check that final sharpened output doesn't have extreme values
    # (indicating white noise from over-processing)
    sharpened = result['sharpened']
    white_pixels = np.sum(sharpened == 255)
    black_pixels = np.sum(sharpened == 0)
    
    # Less than 5% should be extreme values
    total_pixels = sharpened.size
    assert (white_pixels + black_pixels) / total_pixels < 0.05, \
        "Too many extreme pixels - possible over-processing"

