"""
Unit tests for brain_extraction module.
Tests core preprocessing functions for brain segmentation.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from btsc.preprocess.brain_extraction import (
    n4_bias_correction,
    intensity_clip,
    zscore_norm,
    otsu_brain_mask,
    adaptive_threshold_mask,
    postprocess_mask,
    create_overlay,
    extract_brain_region,
    apply_pipeline
)


def test_intensity_clip():
    """Test intensity clipping removes outliers."""
    # Create test image with outliers
    img = np.random.rand(100, 100) * 100
    img[0, 0] = 255  # Outlier
    img[99, 99] = 0  # Outlier
    
    clipped = intensity_clip(img, pmin=1.0, pmax=99.0)
    
    # Check that outliers are clipped
    assert clipped[0, 0] < 255
    assert clipped[99, 99] > 0
    assert clipped.shape == img.shape


def test_zscore_norm():
    """Test z-score normalization."""
    img = np.random.rand(100, 100) * 255
    
    normalized = zscore_norm(img)
    
    # Check that mean is close to 0 and std is close to 1
    assert abs(normalized.mean()) < 1.0
    assert abs(normalized.std() - 1.0) < 0.1
    assert normalized.shape == img.shape


def test_zscore_norm_with_mask():
    """Test z-score normalization with mask."""
    img = np.random.rand(100, 100) * 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Brain region
    
    normalized = zscore_norm(img, within_mask=mask)
    
    # Normalization should be based on masked region
    assert normalized.shape == img.shape


def test_otsu_brain_mask():
    """Test Otsu thresholding for brain mask."""
    # Create phantom image with brain (high intensity) and background (low intensity)
    img = np.zeros((256, 256), dtype=np.float32)
    img[50:200, 50:200] = 200  # Brain region
    img += np.random.randn(256, 256) * 10  # Add noise
    
    mask = otsu_brain_mask(img)
    
    # Check mask properties
    assert mask.dtype == np.uint8
    assert mask.shape == img.shape
    assert mask.max() == 255
    assert mask.min() == 0
    
    # Check that brain region is mostly masked
    brain_region_mask = mask[50:200, 50:200]
    assert np.sum(brain_region_mask > 0) > 0.8 * brain_region_mask.size


def test_adaptive_threshold_mask_single():
    """Test adaptive thresholding with single method."""
    img = np.random.rand(100, 100) * 255
    
    # Test each method
    for method in ['otsu', 'yen', 'li', 'triangle']:
        mask = adaptive_threshold_mask(img, method=method, return_all=False)
        
        assert mask.dtype == np.uint8
        assert mask.shape == img.shape
        assert mask.max() <= 255
        assert mask.min() >= 0


def test_adaptive_threshold_mask_all():
    """Test adaptive thresholding with all methods."""
    img = np.random.rand(100, 100) * 255
    
    masks = adaptive_threshold_mask(img, return_all=True)
    
    # Check that all methods are returned
    assert isinstance(masks, dict)
    assert 'otsu' in masks
    assert 'yen' in masks
    assert 'li' in masks
    assert 'triangle' in masks
    
    # Check properties of each mask
    for name, mask in masks.items():
        assert mask.dtype == np.uint8
        assert mask.shape == img.shape


def test_postprocess_mask_basic():
    """Test basic mask postprocessing."""
    # Create noisy mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:200, 50:200] = 255  # Main brain region
    mask[10:15, 10:15] = 255  # Small noise
    mask[100:105, 100:105] = 0  # Small hole
    
    cleaned = postprocess_mask(mask, min_size=100, closing=3, opening=1, fill_holes=True)
    
    assert cleaned.dtype == np.uint8
    assert cleaned.shape == mask.shape
    
    # Small noise should be removed
    assert np.sum(cleaned[10:15, 10:15]) < np.sum(mask[10:15, 10:15])
    
    # Hole should be filled
    assert np.sum(cleaned[100:105, 100:105]) > 0


def test_postprocess_mask_keep_largest():
    """Test keeping largest component."""
    # Create mask with two components
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 255  # Large component
    mask[200:220, 200:220] = 255  # Small component
    
    cleaned = postprocess_mask(mask, keep_largest=True, min_size=0)
    
    # Only largest component should remain
    assert np.sum(cleaned[50:150, 50:150]) > 0
    assert np.sum(cleaned[200:220, 200:220]) == 0


def test_postprocess_mask_stability():
    """Test that postprocessing is stable under small noise."""
    # Create clean mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:200, 50:200] = 255
    
    # Add small noise
    noisy_mask = mask.copy()
    noise_indices = np.random.choice(256*256, size=100, replace=False)
    noisy_mask.flat[noise_indices] = 255 - noisy_mask.flat[noise_indices]
    
    # Process both
    cleaned_original = postprocess_mask(mask, min_size=1000)
    cleaned_noisy = postprocess_mask(noisy_mask, min_size=1000)
    
    # Results should be very similar
    diff = np.abs(cleaned_original.astype(float) - cleaned_noisy.astype(float)).sum()
    total_pixels = mask.size
    similarity = 1.0 - (diff / (255 * total_pixels))
    
    assert similarity > 0.95  # At least 95% similar


def test_create_overlay():
    """Test overlay creation."""
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    
    overlay = create_overlay(img, mask)
    
    # Check overlay properties
    assert overlay.shape == (100, 100, 3)  # RGB
    assert overlay.dtype == np.uint8


def test_extract_brain_region():
    """Test brain region extraction."""
    img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 60:140] = 255
    
    cropped = extract_brain_region(img, mask, pad=5)
    
    # Check that cropped region is smaller than original
    assert cropped.shape[0] < img.shape[0]
    assert cropped.shape[1] < img.shape[1]
    
    # Check that cropped region includes the mask bounds
    assert cropped.shape[0] >= 100  # Height of mask region
    assert cropped.shape[1] >= 80   # Width of mask region


def test_extract_brain_region_empty_mask():
    """Test brain extraction with empty mask."""
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)  # Empty mask
    
    cropped = extract_brain_region(img, mask)
    
    # Should return original image
    assert np.array_equal(cropped, img)


def test_apply_pipeline_disabled():
    """Test apply_pipeline with disabled config."""
    img = np.random.rand(100, 100).astype(np.float32) * 255
    cfg = {'enable': False}
    
    result = apply_pipeline(img, cfg)
    
    # Should return original image only
    assert 'stages' in result
    assert 'original' in result['stages']
    assert 'candidates' in result
    assert len(result['candidates']) == 0


def test_apply_pipeline_basic():
    """Test apply_pipeline with basic config."""
    img = np.random.rand(128, 128).astype(np.float32) * 255
    
    cfg = {
        'enable': True,
        'steps': {
            'n4_bias_correction': False,  # Skip N4 (requires ITK)
            'intensity_clip': {'pmin': 1.0, 'pmax': 99.0},
            'zscore_norm': True,
            'histogram_match_to_nfbs': {'enabled': False},  # Skip (no reference)
            'smoothing': {'gaussian_sigma': 1.0},
            'thresholding': {
                'primary': 'otsu',
                'candidates': ['otsu', 'yen'],
                'postprocess': {
                    'min_size': 100,
                    'closing': 3,
                    'opening': 1,
                    'fill_holes': True,
                    'keep_largest': True
                }
            }
        }
    }
    
    result = apply_pipeline(img, cfg)
    
    # Check result structure
    assert 'stages' in result
    assert 'candidates' in result
    assert 'final' in result
    assert 'config_used' in result
    
    # Check stages
    assert 'original' in result['stages']
    assert 'clipped' in result['stages']
    assert 'normalized' in result['stages']
    
    # Check candidates
    assert 'otsu' in result['candidates']
    assert 'yen' in result['candidates']
    
    # Check final
    assert 'final_brain_mask' in result['final']
    assert 'overlay' in result['final']
    assert 'cropped' in result['final']
    
    # Check shapes
    for stage_img in result['stages'].values():
        assert stage_img.shape == img.shape
    
    for mask in result['candidates'].values():
        assert mask.shape == img.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
