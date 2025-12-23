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
    create_brain_mask
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


def test_normalize_image_with_mask():
    """Test image normalization with mask."""
    image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Central region is brain
    
    # Z-score with mask
    normalized = normalize_image(image, method='zscore', mask=mask)
    assert normalized.shape == image.shape
    assert normalized.dtype == np.uint8
    # Background should be zero
    assert np.all(normalized[:25, :] == 0)  # Top background
    assert np.all(normalized[75:, :] == 0)  # Bottom background
    
    # Min-max with mask
    normalized = normalize_image(image, method='minmax', mask=mask)
    assert normalized.shape == image.shape
    assert normalized.dtype == np.uint8


def test_create_brain_mask():
    """Test automatic brain mask creation."""
    # Create a synthetic brain-like image (dark background, bright center)
    image = np.zeros((100, 100), dtype=np.uint8)
    # Add a circular "brain" region
    center = (50, 50)
    radius = 30
    Y, X = np.ogrid[:100, :100]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    image[dist_from_center <= radius] = 150  # Brain region
    
    mask = create_brain_mask(image, threshold_value=10)
    assert mask.shape == image.shape
    assert mask.dtype == np.uint8
    # Mask should be mostly 0 in background, 255 in brain region
    assert np.sum(mask > 0) > 0  # Some foreground detected


def test_preprocess_pipeline():
    """Test full preprocessing pipeline."""
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


def test_preprocess_pipeline_with_brain_mask():
    """Test preprocessing pipeline with brain mask enabled."""
    # Create a synthetic brain MRI-like image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a circular "brain" region
    center = (50, 50)
    radius = 35
    Y, X = np.ogrid[:100, :100]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    brain_region = dist_from_center <= radius
    image[brain_region] = [150, 150, 150]  # Gray brain
    
    config = {'use_brain_mask': True, 'brain_mask_threshold': 10}
    result = preprocess_pipeline(image, config=config)
    
    assert 'normalized' in result
    # Background should remain dark (not have artifacts)
    normalized = result['normalized']
    # Check corners (background) are dark
    corner_mean = np.mean([
        normalized[0:5, 0:5].mean(),
        normalized[0:5, 95:100].mean(),
        normalized[95:100, 0:5].mean(),
        normalized[95:100, 95:100].mean()
    ])
    assert corner_mean < 50, "Background should remain dark after preprocessing"