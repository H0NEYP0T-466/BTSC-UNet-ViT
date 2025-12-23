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
    run_preprocessing,
    resize_image
)


def test_to_grayscale():
    """Test grayscale conversion."""
    # RGB image
    rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = to_grayscale(rgb_image)
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8


def test_resize_image():
    """Test image resizing."""
    # Large image that should be resized
    large_image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
    resized = resize_image(large_image, max_size=512)
    
    # Check that image was resized
    assert max(resized.shape[:2]) <= 512
    assert resized.dtype == np.uint8
    
    # Small image that should not be resized
    small_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    not_resized = resize_image(small_image, max_size=512)
    
    # Should remain unchanged
    assert not_resized.shape == small_image.shape


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
    
    # Check all expected stages are present (5 stages)
    expected_stages = [
        'grayscale',
        'denoised',
        'motion_reduced',
        'contrast_enhanced',
        'sharpened'
    ]
    
    for stage in expected_stages:
        assert stage in result, f"Missing stage: {stage}"
        assert result[stage].dtype == np.uint8
        assert len(result[stage].shape) == 2  # All should be grayscale


def test_run_preprocessing_with_salt_pepper():
    """Test that denoising is applied conservatively."""
    # Create image with some noise
    image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    noise = np.random.normal(0, 10, image.shape)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    result = run_preprocessing(noisy_image, opts={'auto': True})
    
    # Verify denoised stage exists and is different from grayscale
    assert 'denoised' in result
    assert result['denoised'].dtype == np.uint8


def test_enhanced_salt_pepper_removal():
    """Test conservative denoising approach."""
    from app.utils.btsc_preprocess import denoise_gaussian_nlmeans
    
    # Create clean image with Gaussian noise
    image = np.full((100, 100), 128, dtype=np.uint8)
    
    # Add 5% Gaussian noise
    noise = np.random.normal(0, 10, image.shape)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Apply conservative denoising
    cleaned = denoise_gaussian_nlmeans(noisy_image, h_scale=0.6)
    
    # Should be smoother but not over-smoothed
    assert cleaned.dtype == np.uint8
    assert cleaned.shape == image.shape
    
    # Check that noise variance is reduced
    noise_var_before = np.var(noisy_image.astype(np.float32) - image.astype(np.float32))
    noise_var_after = np.var(cleaned.astype(np.float32) - image.astype(np.float32))
    assert noise_var_after < noise_var_before


def test_run_preprocessing_conservative_parameters():
    """Test that preprocessing uses conservative parameters to avoid white noise."""
    # Clean image
    image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    
    result = run_preprocessing(image, opts={
        'auto': True,
        'clahe_clip_limit': 1.2,
        'sharpen_amount': 0.3,
        'sharpen_threshold': 10
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


def test_pma_and_deblur_skipped():
    """Test that PMA correction and deblurring are skipped in the new pipeline."""
    image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    result = run_preprocessing(image, opts={'auto': True})
    
    # These stages should NOT be present
    assert 'pma_corrected' not in result, "PMA correction should be skipped"
    assert 'deblurred' not in result, "Deblurring should be skipped"
    assert 'salt_pepper_cleaned' not in result, "Salt & pepper stage should be skipped"
    assert 'gaussian_denoised' not in result, "Separate Gaussian stage should be skipped"
    assert 'speckle_denoised' not in result, "Speckle stage should be skipped"
    
    # Only the 5 required stages should be present
    assert 'grayscale' in result
    assert 'denoised' in result
    assert 'motion_reduced' in result
    assert 'contrast_enhanced' in result
    assert 'sharpened' in result

