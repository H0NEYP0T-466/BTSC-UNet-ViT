"""
Image preprocessing utilities for BTSC-UNet-ViT.
Implements intelligent noise/blur detection and targeted correction.
Supports: Salt & Pepper, Gaussian noise, Speckle noise, Gaussian blur,
Bilateral blur, Median blur, and Patient Motion Artifacts (PMA).
"""
import time
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from skimage import filters, restoration, exposure
from skimage.filters import unsharp_mask as sk_unsharp_mask
from scipy.ndimage import median_filter
from app.utils.logger import get_logger

# Import BTSC preprocessing modules for intelligent processing
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

logger = get_logger(__name__)


def resize_image(
    image: np.ndarray,
    max_size: int = 512,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Resize image to reduce memory usage while preserving aspect ratio.
    
    Images larger than max_size in either dimension will be scaled down
    proportionally. Smaller images are left unchanged.
    
    Args:
        image: Input image (RGB or grayscale)
        max_size: Maximum dimension size (default: 512)
        image_id: Image identifier for logging
        
    Returns:
        Resized image
    """
    start_time = time.time()
    h, w = image.shape[:2]
    
    # Only resize if image is larger than max_size
    if h <= max_size and w <= max_size:
        logger.info(f"Image already optimal size ({h}x{w}), skipping resize", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'resize'
        })
        return image
    
    # Calculate new dimensions preserving aspect ratio
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    
    logger.info(f"Resizing image from {h}x{w} to {new_h}x{new_w} (max_size={max_size})", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'resize'
    })
    
    # Use INTER_AREA for downscaling (best quality)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    duration = time.time() - start_time
    logger.info(f"Image resized in {duration:.3f}s", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'resize'
    })
    
    return resized


def to_grayscale(image: np.ndarray, image_id: Optional[str] = None) -> np.ndarray:
    """
    Convert image to grayscale if needed.
    
    Args:
        image: Input image (RGB or grayscale)
        image_id: Image identifier for logging
        
    Returns:
        Grayscale image
    """
    start_time = time.time()
    logger.info("Converting to grayscale", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'grayscale_conversion'
    })
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        # RGB to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        logger.info(f"Converted RGB to grayscale, shape: {gray.shape}", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'grayscale_conversion'
        })
    elif len(image.shape) == 2:
        # Already grayscale
        gray = image.copy()
        logger.info(f"Image already grayscale, shape: {gray.shape}", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'grayscale_conversion'
        })
    else:
        gray = image[:, :, 0] if len(image.shape) == 3 else image
        logger.info(f"Extracted single channel, shape: {gray.shape}", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'grayscale_conversion'
        })
    
    # Ensure consistent dtype and range
    if gray.dtype != np.uint8:
        gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
    
    duration = time.time() - start_time
    logger.info(f"Grayscale conversion completed in {duration:.3f}s", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'grayscale_conversion'
    })
    
    return gray


def remove_salt_pepper(
    image: np.ndarray,
    kernel_size: int = 3,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Remove salt and pepper noise using median filter.
    
    Args:
        image: Input grayscale image
        kernel_size: Median filter kernel size
        image_id: Image identifier for logging
        
    Returns:
        Denoised image
    """
    start_time = time.time()
    logger.info(f"Removing salt & pepper noise with median filter (kernel={kernel_size})", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'denoise_salt_pepper'
    })
    
    denoised = cv2.medianBlur(image, kernel_size)
    
    duration = time.time() - start_time
    logger.info(f"Image denoised successfully in {duration:.3f}s, method=median, kernel={kernel_size}", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'denoise_salt_pepper'
    })
    
    return denoised


def denoise_nlm(
    image: np.ndarray,
    h: int = 10,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Denoise using Non-Local Means for heavy noise.
    
    Args:
        image: Input grayscale image
        h: Filter strength
        image_id: Image identifier for logging
        
    Returns:
        Denoised image
    """
    start_time = time.time()
    logger.info(f"Denoising with Non-Local Means (h={h})", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'denoise_nlm'
    })
    
    denoised = cv2.fastNlMeansDenoising(image, None, h=h, templateWindowSize=7, searchWindowSize=21)
    
    duration = time.time() - start_time
    logger.info(f"NLM denoising completed in {duration:.3f}s", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'denoise_nlm'
    })
    
    return denoised


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    mask: Optional[np.ndarray] = None,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        mask: Optional binary mask to apply CLAHE only inside mask (0 = background, 255 = foreground)
        image_id: Image identifier for logging
        
    Returns:
        Contrast-enhanced image
    """
    start_time = time.time()
    logger.info(f"Enhancing contrast with CLAHE (clip_limit={clip_limit}, grid={tile_grid_size})", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'contrast_enhancement'
    })
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if mask is not None:
        # Apply CLAHE only inside the mask to avoid background noise
        logger.info("Applying CLAHE inside brain mask only", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'contrast_enhancement'
        })
        
        # Create a copy to preserve original
        enhanced = image.copy()
        
        # Apply CLAHE only to masked region
        masked_region = image.copy()
        masked_region[mask == 0] = 0  # Zero out background
        
        # Apply CLAHE
        enhanced_masked = clahe.apply(masked_region)
        
        # Copy enhanced values only where mask is present
        enhanced[mask > 0] = enhanced_masked[mask > 0]
    else:
        # Apply CLAHE to entire image
        enhanced = clahe.apply(image)
    
    duration = time.time() - start_time
    logger.info(f"Contrast enhancement completed successfully in {duration:.3f}s", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'contrast_enhancement'
    })
    
    return enhanced


def unsharp_mask(
    image: np.ndarray,
    radius: float = 1.0,
    amount: float = 1.0,
    mask: Optional[np.ndarray] = None,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Apply unsharp mask for edge sharpening.
    
    Args:
        image: Input grayscale image
        radius: Radius of Gaussian blur
        amount: Strength of sharpening
        mask: Optional binary mask to apply sharpening only inside mask (0 = background, 255 = foreground)
        image_id: Image identifier for logging
        
    Returns:
        Sharpened image
    """
    start_time = time.time()
    logger.info(f"Applying unsharp mask (radius={radius}, amount={amount})", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'sharpening'
    })
    
    # Normalize to [0, 1] for scikit-image
    img_normalized = image.astype(np.float32) / 255.0
    sharpened = sk_unsharp_mask(img_normalized, radius=radius, amount=amount)
    sharpened = (np.clip(sharpened, 0, 1) * 255).astype(np.uint8)
    
    if mask is not None:
        # Apply sharpening only inside the mask
        logger.info("Applying sharpening inside brain mask only", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'sharpening'
        })
        result = image.copy()
        result[mask > 0] = sharpened[mask > 0]
        sharpened = result
    
    duration = time.time() - start_time
    logger.info(f"Sharpening completed successfully in {duration:.3f}s", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'sharpening'
    })
    
    return sharpened


def reduce_motion_artifact(
    image: np.ndarray,
    image_id: Optional[str] = None,
    preserve_detail: bool = True
) -> np.ndarray:
    """
    Reduce motion artifacts while preserving image detail and quality.
    Uses very light edge-preserving bilateral filtering to maintain sharpness.
    
    Args:
        image: Input grayscale image
        image_id: Image identifier for logging
        preserve_detail: If True, uses minimal edge-preserving bilateral filter (recommended)
        
    Returns:
        Motion-corrected image with preserved detail
    """
    start_time = time.time()
    logger.info("Reducing motion artifacts while preserving detail", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'motion_reduction'
    })
    
    if preserve_detail:
        # Use very light bilateral filter to preserve detail
        # Parameters optimized to MINIMIZE blur while reducing motion artifacts:
        # d=3 (very small neighborhood), sigmaColor=30, sigmaSpace=30
        # This keeps edges sharp and preserves fine details
        deblurred = cv2.bilateralFilter(image, 3, 30, 30)
        logger.info("Applied minimal edge-preserving bilateral filter", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'motion_reduction'
        })
    else:
        # Skip motion reduction entirely to preserve maximum detail
        deblurred = image.copy()
        logger.info("Skipped motion reduction to preserve detail", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'motion_reduction'
        })
    
    duration = time.time() - start_time
    logger.info(f"Motion artifact reduction completed in {duration:.3f}s", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'motion_reduction'
    })
    
    return deblurred


def normalize_image(
    image: np.ndarray,
    method: str = "zscore",
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Normalize image intensity.
    
    Args:
        image: Input grayscale image
        method: Normalization method ('zscore' or 'minmax')
        image_id: Image identifier for logging
        
    Returns:
        Normalized image
    """
    start_time = time.time()
    logger.info(f"Normalizing image with method={method}", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'normalization'
    })
    
    img_float = image.astype(np.float32)
    
    if method == "zscore":
        mean = np.mean(img_float)
        std = np.std(img_float)
        normalized = (img_float - mean) / (std + 1e-8)
        # Scale back to [0, 255]
        normalized = ((normalized - normalized.min()) / (normalized.max() - normalized.min()) * 255)
        logger.info(f"Z-score normalization: mean={mean:.2f}, std={std:.2f}", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'normalization'
        })
    else:  # minmax
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        normalized = (img_float - min_val) / (max_val - min_val + 1e-8) * 255
        logger.info(f"Min-max normalization: min={min_val:.2f}, max={max_val:.2f}", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'normalization'
        })
    
    normalized = normalized.astype(np.uint8)
    
    duration = time.time() - start_time
    logger.info(f"Normalization completed in {duration:.3f}s", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'normalization'
    })
    
    return normalized


def preprocess_pipeline(
    image: np.ndarray,
    config: Optional[Dict] = None,
    image_id: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Run complete preprocessing pipeline.
    
    Pipeline order (optimized for quality and accuracy):
    1. Convert to grayscale
    2. Denoising (removes noise while preserving edges)
    3. Light motion artifact reduction (minimal blur)
    4. Contrast enhancement (CLAHE)
    5. Sharpening (recovers fine details)
    6. Normalization (standardizes intensity range)
    
    Args:
        image: Input image (RGB or grayscale)
        config: Optional configuration dict with preprocessing parameters
        image_id: Image identifier for logging
        
    Returns:
        Dictionary with intermediate preprocessing outputs
    """
    start_time = time.time()
    logger.info("Preprocessing pipeline started", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'preprocess'
    })
    
    config = config or {}
    
    # Step 1: Convert to grayscale
    grayscale = to_grayscale(image, image_id=image_id)
    
    # Step 2: Denoise (preserves edges while removing noise)
    # Use Non-Local Means with lighter settings to preserve detail
    if config.get('use_nlm_denoising', True):
        denoised = denoise_nlm(
            grayscale,
            h=config.get('nlm_h', 8),  # Reduced from 10 to 8 for less blur
            image_id=image_id
        )
    else:
        # Fallback to median filter (very light)
        denoised = remove_salt_pepper(
            grayscale,
            kernel_size=config.get('median_kernel_size', 3),
            image_id=image_id
        )
    
    # Step 3: Reduce motion artifacts (minimal filtering)
    motion_reduced = reduce_motion_artifact(
        denoised, 
        image_id=image_id,
        preserve_detail=config.get('preserve_detail', True)
    )
    
    # Step 4: Enhance contrast (CLAHE)
    contrast = enhance_contrast_clahe(
        motion_reduced,
        clip_limit=config.get('clahe_clip_limit', 2.0),
        tile_grid_size=config.get('clahe_tile_grid_size', (8, 8)),
        mask=None,
        image_id=image_id
    )
    
    # Step 5: Sharpen (recover fine details)
    sharpened = unsharp_mask(
        contrast,
        radius=config.get('unsharp_radius', 1.5),
        amount=config.get('unsharp_amount', 1.5),
        mask=None,
        image_id=image_id
    )
    
    # Step 6: Normalize (standardize intensity range)
    normalized = normalize_image(
        sharpened,
        method=config.get('normalize_method', 'zscore'),
        image_id=image_id
    )
    
    duration = time.time() - start_time
    logger.info(f"Preprocessing pipeline completed in {duration:.3f}s", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'preprocess'
    })
    
    return {
        'grayscale': grayscale,
        'denoised': denoised,
        'motion_reduced': motion_reduced,
        'contrast': contrast,
        'sharpened': sharpened,
        'normalized': normalized
    }


def run_preprocessing(
    image: np.ndarray,
    opts: Optional[Dict] = None,
    image_id: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Run intelligent preprocessing pipeline with auto-detection.
    
    This pipeline detects and corrects specific image quality issues:
    - Image size: Resizes large images to reduce memory usage
    - Noise: Gaussian (via conservative NLM denoising)
    - Motion: Minimal edge-preserving bilateral filter (no PMA correction)
    
    Pipeline stages:
    1. Grayscale conversion
    2. Conservative denoising (NLM with reduced parameters)
    3. Motion reduction (minimal bilateral filter, preserves detail)
    4. Contrast enhancement (CLAHE with conservative clip limit)
    5. Sharpening (conservative unsharp mask)
    
    Note: PMA correction and deblurring are SKIPPED in inference to avoid
    over-smoothing and preserve lesion edges. Speckle noise removal is also
    SKIPPED (speckle noise is for data augmentation only, not inference).
    
    Args:
        image: Input image (RGB or grayscale)
        opts: Options dict:
            - auto: bool - Auto-detect and apply corrections (default: True)
            - max_size: int - Maximum image dimension in pixels (default: 512)
            - clahe_clip_limit: float - CLAHE clip limit (default: 1.2)
            - clahe_tile_grid: tuple - CLAHE tile grid (default: (8, 8))
            - sharpen_amount: float - Sharpening strength (default: 0.3)
            - sharpen_radius: float - Sharpening radius (default: 0.5)
            - sharpen_threshold: float - Sharpening threshold (default: 10)
            - skip_pma: bool - Skip PMA correction (default: True)
            - skip_deblur: bool - Skip deblurring (default: True)
        image_id: Image identifier for logging
        
    Returns:
        Dictionary with preprocessing stages:
        - grayscale: Grayscale conversion
        - denoised: After conservative denoising
        - motion_reduced: After minimal motion reduction (bilateral filter)
        - contrast_enhanced: After CLAHE enhancement
        - sharpened: After edge sharpening (final output)
    """
    start_time = time.time()
    logger.info("Intelligent preprocessing pipeline started", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'preprocess_intelligent'
    })
    
    opts = opts or {}
    auto_detect = opts.get('auto', True)
    max_size = opts.get('max_size', 512)  # Maximum image dimension
    
    # Conservative parameters to avoid white noise/over-processing
    clahe_clip = opts.get('clahe_clip_limit', 1.2)  # Reduced from 1.5
    clahe_grid = opts.get('clahe_tile_grid', (8, 8))
    sharpen_amount = opts.get('sharpen_amount', 0.3)  # Reduced from 0.8
    sharpen_radius = opts.get('sharpen_radius', 0.5)  # Reduced from 1.0
    sharpen_threshold = opts.get('sharpen_threshold', 10)  # Increased from 0.02
    
    # Inference-only: skip PMA and deblur to avoid over-smoothing
    skip_pma = opts.get('skip_pma', True)
    skip_deblur = opts.get('skip_deblur', True)
    
    results = {}
    
    # Step 1: Convert to grayscale
    grayscale = to_grayscale(image, image_id=image_id)
    results['grayscale'] = grayscale
    
    # Working image starts as grayscale
    current = grayscale.copy()
    
    # Step 2: Conservative denoising (NLM with reduced h parameter)
    # Speckle noise removal is SKIPPED (speckle noise is for augmentation only, not inference)
    if auto_detect:
        logger.info("Applying conservative Gaussian noise removal (NLM with h=6)", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'denoise'
        })
        current = denoise_gaussian_nlmeans(current, h_scale=0.6, image_id=image_id)
        results['denoised'] = current
    else:
        results['denoised'] = current.copy()
    
    # Step 3: Motion reduction (minimal bilateral filter, NO PMA correction)
    # Bilateral filter preserves edges while reducing noise/motion artifacts
    logger.info("Applying minimal motion reduction (bilateral filter, preserves edges)", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'motion_reduction'
    })
    # Use very light bilateral filter to preserve detail
    current = cv2.bilateralFilter(current, 3, 30, 30)
    results['motion_reduced'] = current
    
    # Step 4: Contrast enhancement (CLAHE with conservative parameters)
    logger.info(f"Applying CLAHE contrast enhancement (clip={clahe_clip})", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'contrast_enhancement'
    })
    current = clahe_enhance(
        current,
        clipLimit=clahe_clip,
        tileGrid=clahe_grid,
        image_id=image_id
    )
    results['contrast_enhanced'] = current
    
    # Step 5: Conservative sharpening (reduced parameters to avoid noise)
    logger.info(f"Applying conservative sharpening (amount={sharpen_amount}, threshold={sharpen_threshold})", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'sharpening'
    })
    current = sharpen_noise_aware(
        current,
        radius=sharpen_radius,
        amount=sharpen_amount,
        threshold=sharpen_threshold,
        image_id=image_id
    )
    results['sharpened'] = current
    
    duration = time.time() - start_time
    logger.info(f"Intelligent preprocessing completed in {duration:.3f}s (PMA and deblur SKIPPED)", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'preprocess_complete'
    })
    
    return results
