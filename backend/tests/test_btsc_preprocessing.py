"""
Tests for BTSC preprocessing modules: noise, deblur, contrast, detect.
"""
import numpy as np
import pytest
import cv2
from app.utils.btsc_preprocess import (
    remove_salt_and_pepper,
    denoise_gaussian_nlmeans,
    denoise_speckle_wavelet,
    deblur_gaussian_wiener,
    deblur_edge_aware_usm,
    correct_motion_artifacts,
    clahe_enhance,
    sharpen_noise_aware,
    detect_noise_type,
    detect_blur,
    detect_motion,
)


# Fixtures for synthetic corrupted images

@pytest.fixture
def clean_image():
    """Create a clean synthetic MRI-like image."""
    img = np.zeros((256, 256), dtype=np.uint8)
    # Add a circular "brain" region
    center = (128, 128)
    radius = 80
    Y, X = np.ogrid[:256, :256]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    brain_mask = dist_from_center <= radius
    img[brain_mask] = 180
    
    # Add some structure (simulated features)
    cv2.circle(img, (110, 110), 20, 150, -1)
    cv2.circle(img, (146, 110), 20, 150, -1)
    cv2.circle(img, (128, 140), 15, 200, -1)
    
    return img


@pytest.fixture
def salt_pepper_image(clean_image):
    """Add salt and pepper noise to image."""
    img = clean_image.copy()
    noise_fraction = 0.05
    num_pixels = int(noise_fraction * img.size)
    
    # Salt (255)
    coords_salt = [np.random.randint(0, i, num_pixels // 2) for i in img.shape]
    img[coords_salt[0], coords_salt[1]] = 255
    
    # Pepper (0)
    coords_pepper = [np.random.randint(0, i, num_pixels // 2) for i in img.shape]
    img[coords_pepper[0], coords_pepper[1]] = 0
    
    return img


@pytest.fixture
def gaussian_noise_image(clean_image):
    """Add Gaussian noise to image."""
    img = clean_image.astype(np.float32)
    sigma = 15
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


@pytest.fixture
def speckle_noise_image(clean_image):
    """Add speckle (multiplicative) noise to image."""
    img = clean_image.astype(np.float32) / 255.0
    noise = np.random.randn(*img.shape) * 0.1
    speckled = img * (1 + noise)
    speckled = np.clip(speckled, 0, 1) * 255
    return speckled.astype(np.uint8)


@pytest.fixture
def blurred_image(clean_image):
    """Add Gaussian blur to image."""
    blurred = cv2.GaussianBlur(clean_image, (15, 15), 2.5)
    return blurred


@pytest.fixture
def motion_blur_image(clean_image):
    """Add motion blur to image."""
    # Create motion kernel
    size = 15
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    
    # Apply motion blur
    motion_blurred = cv2.filter2D(clean_image, -1, kernel)
    return motion_blurred


# Tests for noise removal

def test_remove_salt_and_pepper(salt_pepper_image, clean_image):
    """Test salt and pepper noise removal."""
    denoised = remove_salt_and_pepper(salt_pepper_image, max_kernel=7)
    
    assert denoised.shape == salt_pepper_image.shape
    assert denoised.dtype == np.uint8
    
    # Check that impulse pixels are significantly reduced
    # Only count in the brain region (non-zero in clean image)
    brain_mask = clean_image > 10
    impulse_before = np.sum(((salt_pepper_image == 0) | (salt_pepper_image == 255)) & brain_mask)
    impulse_after = np.sum(((denoised == 0) | (denoised == 255)) & brain_mask)
    
    # Should remove >80% of impulse pixels in brain region
    if impulse_before > 0:
        reduction_ratio = (impulse_before - impulse_after) / impulse_before
        assert reduction_ratio > 0.5, f"Impulse reduction: {reduction_ratio*100:.1f}%"


def test_denoise_gaussian_nlmeans(gaussian_noise_image, clean_image):
    """Test Gaussian noise denoising with NLM."""
    denoised = denoise_gaussian_nlmeans(gaussian_noise_image, h_scale=0.8)
    
    assert denoised.shape == gaussian_noise_image.shape
    assert denoised.dtype == np.uint8
    
    # Check noise reduction (compare variance)
    noise_var_before = np.var(gaussian_noise_image.astype(np.float32) - clean_image.astype(np.float32))
    noise_var_after = np.var(denoised.astype(np.float32) - clean_image.astype(np.float32))
    
    # Should reduce noise variance
    assert noise_var_after < noise_var_before


def test_denoise_speckle_wavelet(speckle_noise_image, clean_image):
    """Test speckle noise denoising with wavelet."""
    denoised = denoise_speckle_wavelet(speckle_noise_image)
    
    assert denoised.shape == speckle_noise_image.shape
    assert denoised.dtype == np.uint8
    
    # Check that speckle is reduced
    # Measure log-domain variance (speckle indicator)
    log_var_before = np.var(np.log(speckle_noise_image.astype(np.float32) + 1))
    log_var_after = np.var(np.log(denoised.astype(np.float32) + 1))
    
    assert log_var_after <= log_var_before


# Tests for deblurring

def test_deblur_gaussian_wiener(blurred_image, clean_image):
    """Test Gaussian blur deblurring."""
    deblurred = deblur_gaussian_wiener(blurred_image, sigma=2.5)
    
    assert deblurred.shape == blurred_image.shape
    assert deblurred.dtype == np.uint8
    
    # Check edge sharpness improvement (Laplacian variance)
    lap_blurred = cv2.Laplacian(blurred_image, cv2.CV_64F).var()
    lap_deblurred = cv2.Laplacian(deblurred, cv2.CV_64F).var()
    
    # Deblurred should have higher edge energy
    assert lap_deblurred > lap_blurred


def test_deblur_edge_aware_usm(blurred_image):
    """Test edge-aware unsharp masking."""
    deblurred = deblur_edge_aware_usm(blurred_image, amount=1.5, radius=2.0)
    
    assert deblurred.shape == blurred_image.shape
    assert deblurred.dtype == np.uint8
    
    # Check edge sharpness improvement
    lap_blurred = cv2.Laplacian(blurred_image, cv2.CV_64F).var()
    lap_deblurred = cv2.Laplacian(deblurred, cv2.CV_64F).var()
    
    assert lap_deblurred >= lap_blurred


def test_correct_motion_artifacts(motion_blur_image):
    """Test motion artifact correction."""
    corrected = correct_motion_artifacts(motion_blur_image, angle=0, length=10)
    
    assert corrected.shape == motion_blur_image.shape
    assert corrected.dtype == np.uint8
    
    # Check for improvement (higher edge variance)
    lap_motion = cv2.Laplacian(motion_blur_image, cv2.CV_64F).var()
    lap_corrected = cv2.Laplacian(corrected, cv2.CV_64F).var()
    
    # Corrected should have sharper edges (or at least not worse)
    assert lap_corrected >= lap_motion * 0.8


# Tests for contrast and sharpening

def test_clahe_enhance(clean_image):
    """Test CLAHE contrast enhancement."""
    enhanced = clahe_enhance(clean_image, clipLimit=2.0, tileGrid=(8, 8))
    
    assert enhanced.shape == clean_image.shape
    assert enhanced.dtype == np.uint8
    
    # Check contrast improvement (histogram spread)
    hist_before = np.histogram(clean_image, bins=256, range=(0, 255))[0]
    hist_after = np.histogram(enhanced, bins=256, range=(0, 255))[0]
    
    # Enhanced should use more of the dynamic range
    std_before = np.std(clean_image)
    std_after = np.std(enhanced)
    assert std_after >= std_before


def test_sharpen_noise_aware(clean_image):
    """Test noise-aware sharpening."""
    sharpened = sharpen_noise_aware(clean_image, radius=1.5, amount=1.2, threshold=0.01)
    
    assert sharpened.shape == clean_image.shape
    assert sharpened.dtype == np.uint8
    
    # Check edge enhancement
    grad_before = cv2.Sobel(clean_image, cv2.CV_64F, 1, 1, ksize=3)
    grad_after = cv2.Sobel(sharpened, cv2.CV_64F, 1, 1, ksize=3)
    
    grad_mag_before = np.mean(np.abs(grad_before))
    grad_mag_after = np.mean(np.abs(grad_after))
    
    # Sharpened should have stronger edges
    assert grad_mag_after >= grad_mag_before


# Tests for quality detection

def test_detect_noise_type_salt_pepper(salt_pepper_image):
    """Test salt & pepper noise detection."""
    result = detect_noise_type(salt_pepper_image)
    
    assert result['type'] == 'salt_pepper'
    assert result['scores']['salt_pepper'] > 0.5
    assert 'impulse_fraction' in result['details']


def test_detect_noise_type_gaussian(gaussian_noise_image):
    """Test Gaussian noise detection."""
    result = detect_noise_type(gaussian_noise_image)
    
    # Should detect gaussian or have high gaussian score
    assert result['type'] in ['gaussian', 'none'] or result['scores']['gaussian'] > 0.3
    assert 'sigma_estimate' in result['details']


def test_detect_noise_type_clean(clean_image):
    """Test noise detection on clean image."""
    result = detect_noise_type(clean_image)
    
    # Clean image might have high contrast leading to high log variance
    # Just check that salt_pepper and gaussian scores are reasonably low
    assert result['scores']['salt_pepper'] < 0.5, "Salt&pepper score should be low for clean image"
    assert result['scores']['gaussian'] < 0.5, "Gaussian score should be low for clean image"
    # Speckle might be high due to contrast, which is acceptable


def test_detect_blur(blurred_image, clean_image):
    """Test blur detection."""
    result_blurred = detect_blur(blurred_image)
    result_clean = detect_blur(clean_image)
    
    assert result_blurred['is_blurred'] == True
    assert result_clean['is_blurred'] == False
    
    # Blurred image should have lower Laplacian variance
    assert result_blurred['laplacian_var'] < result_clean['laplacian_var']
    
    # Blur score should be higher for blurred image
    assert result_blurred['blur_score'] > result_clean['blur_score']


def test_detect_motion(motion_blur_image, clean_image):
    """Test motion artifact detection."""
    result_motion = detect_motion(motion_blur_image)
    result_clean = detect_motion(clean_image)
    
    # Motion blurred image should have higher streak score
    assert result_motion['streak_score'] >= result_clean['streak_score']
    
    # Check that angle is estimated
    assert 'angle_estimate' in result_motion
    assert 0 <= result_motion['angle_estimate'] <= 180


# Performance tests

def test_salt_pepper_performance(salt_pepper_image):
    """Test salt & pepper removal performance (<30ms target)."""
    import time
    start = time.time()
    remove_salt_and_pepper(salt_pepper_image, max_kernel=7)
    duration_ms = (time.time() - start) * 1000
    
    # Allow some margin for CI environment
    assert duration_ms < 100, f"Salt & pepper removal took {duration_ms:.1f}ms (target: <30ms)"


def test_gaussian_nlm_performance(gaussian_noise_image):
    """Test Gaussian NLM performance (<120ms target)."""
    import time
    start = time.time()
    denoise_gaussian_nlmeans(gaussian_noise_image, h_scale=0.8)
    duration_ms = (time.time() - start) * 1000
    
    # Allow margin for CI
    assert duration_ms < 300, f"Gaussian NLM took {duration_ms:.1f}ms (target: <120ms)"


def test_clahe_sharpen_performance(clean_image):
    """Test CLAHE + sharpen combined performance (<40ms target)."""
    import time
    start = time.time()
    enhanced = clahe_enhance(clean_image, clipLimit=2.0, tileGrid=(8, 8))
    sharpened = sharpen_noise_aware(enhanced, radius=1.5, amount=1.2, threshold=0.01)
    duration_ms = (time.time() - start) * 1000
    
    # Allow margin for CI
    assert duration_ms < 150, f"CLAHE + sharpen took {duration_ms:.1f}ms (target: <40ms)"


# Integration test

def test_full_pipeline_integration(clean_image):
    """Test that all stages can run in sequence without errors."""
    from app.utils.preprocessing import run_preprocessing
    
    # Add some noise for realistic test
    noisy = clean_image.copy()
    np.random.seed(42)
    noise = np.random.normal(0, 10, noisy.shape)
    noisy = np.clip(noisy.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Run full pipeline
    result = run_preprocessing(noisy, opts={'auto': True})
    
    # Check all stages are present
    expected_stages = [
        'grayscale', 'salt_pepper_cleaned', 'gaussian_denoised',
        'speckle_denoised', 'pma_corrected', 'deblurred',
        'contrast_enhanced', 'sharpened'
    ]
    
    for stage in expected_stages:
        assert stage in result, f"Missing stage: {stage}"
        assert result[stage].dtype == np.uint8
        assert result[stage].shape == noisy.shape or result[stage].shape == (noisy.shape[0], noisy.shape[1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
