"""
Image preprocessing utilities for BTSC-UNet-ViT.
Implements denoising, normalization, contrast enhancement, and brain extraction.
"""
import time
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from skimage import filters, restoration, exposure
from skimage.filters import unsharp_mask as sk_unsharp_mask
from scipy.ndimage import median_filter
from app.utils.logger import get_logger
from app.utils.brain_extraction import extract_brain_hdbet

logger = get_logger(__name__)


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
    image_id: Optional[str] = None,
    apply_skull_stripping: bool = True
) -> Dict[str, np.ndarray]:
    """
    Run complete preprocessing pipeline with HD-BET brain extraction.
    
    Pipeline order (optimized for quality and accuracy):
    1. Convert to grayscale
    2. Brain extraction using HD-BET (skull-stripping)
    3. Denoising (removes noise while preserving edges)
    4. Light motion artifact reduction (minimal blur)
    5. Contrast enhancement (CLAHE applied only to brain tissue)
    6. Sharpening (recovers fine details, applied only to brain tissue)
    7. Normalization (standardizes intensity range)
    
    Benefits of HD-BET integration:
    - Removes skull, neck, eyes, and nose from processing
    - Prevents false positives from bright non-brain structures
    - CLAHE and sharpening applied only to brain tissue
    - Improves tumor segmentation accuracy
    
    Args:
        image: Input image (RGB or grayscale)
        config: Optional configuration dict with preprocessing parameters
        image_id: Image identifier for logging
        apply_skull_stripping: Whether to apply HD-BET brain extraction (default: True)
        
    Returns:
        Dictionary with intermediate preprocessing outputs including brain mask
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
    
    # Step 2: Brain extraction using HD-BET
    if apply_skull_stripping:
        brain_extracted, brain_mask = extract_brain_hdbet(grayscale, image_id=image_id)
        logger.info("HD-BET brain extraction applied", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'preprocess'
        })
    else:
        brain_extracted = grayscale.copy()
        brain_mask = np.ones_like(grayscale, dtype=np.uint8) * 255
        logger.info("Skipping brain extraction (disabled)", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'preprocess'
        })
    
    # Step 3: Denoise (preserves edges while removing noise)
    # Use Non-Local Means with lighter settings to preserve detail
    if config.get('use_nlm_denoising', True):
        denoised = denoise_nlm(
            brain_extracted,
            h=config.get('nlm_h', 8),  # Reduced from 10 to 8 for less blur
            image_id=image_id
        )
    else:
        # Fallback to median filter (very light)
        denoised = remove_salt_pepper(
            brain_extracted,
            kernel_size=config.get('median_kernel_size', 3),
            image_id=image_id
        )
    
    # Step 4: Reduce motion artifacts (minimal filtering)
    motion_reduced = reduce_motion_artifact(
        denoised, 
        image_id=image_id,
        preserve_detail=config.get('preserve_detail', True)
    )
    
    # Step 5: Enhance contrast (CLAHE applied only within brain mask)
    # This prevents neck, eyes, and nose from becoming overly bright
    contrast = enhance_contrast_clahe(
        motion_reduced,
        clip_limit=config.get('clahe_clip_limit', 2.0),
        tile_grid_size=config.get('clahe_tile_grid_size', (8, 8)),
        mask=brain_mask if apply_skull_stripping else None,  # Apply only to brain
        image_id=image_id
    )
    
    # Step 6: Sharpen (recover fine details, only within brain mask)
    sharpened = unsharp_mask(
        contrast,
        radius=config.get('unsharp_radius', 1.5),  # Increased from 1.0 to 1.5 for better sharpness
        amount=config.get('unsharp_amount', 1.5),  # Increased from 1.0 to 1.5 for better detail
        mask=brain_mask if apply_skull_stripping else None,  # Apply only to brain
        image_id=image_id
    )
    
    # Step 7: Normalize (standardize intensity range)
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
        'brain_extracted': brain_extracted,
        'brain_mask': brain_mask,
        'denoised': denoised,
        'motion_reduced': motion_reduced,
        'contrast': contrast,
        'sharpened': sharpened,
        'normalized': normalized
    }
