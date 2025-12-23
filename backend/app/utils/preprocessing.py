"""
Image preprocessing utilities for BTSC-UNet-ViT.
Implements denoising, normalization, contrast enhancement.
"""
import time
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from skimage import filters, restoration, exposure
from skimage.filters import unsharp_mask as sk_unsharp_mask
from scipy.ndimage import median_filter
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_brain_mask(
    image: np.ndarray,
    threshold_value: int = 10,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Create an automatic brain mask using thresholding and morphological operations.
    This mask is used to apply preprocessing only to the brain region,
    avoiding artifacts on the black background.
    
    Args:
        image: Input grayscale image (uint8)
        threshold_value: Minimum intensity to consider as brain tissue
        image_id: Image identifier for logging
        
    Returns:
        Binary mask (0 = background, 255 = brain region)
    """
    start_time = time.time()
    logger.info(f"Creating automatic brain mask (threshold={threshold_value})", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'brain_mask_creation'
    })
    
    # Apply fixed threshold to separate brain from background
    # Using a low threshold (default 10) to include most brain tissue
    _, mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Close small holes in the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Open to remove small noise regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill internal holes using contour-based approach
    # Find contours and fill them to handle any internal holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Create a filled mask from the largest contour (brain region)
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, contours, -1, 255, cv2.FILLED)
        mask = mask_filled
    
    duration = time.time() - start_time
    logger.info(f"Brain mask created in {duration:.3f}s, coverage: {np.sum(mask > 0) / mask.size * 100:.1f}%", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'brain_mask_creation'
    })
    
    return mask


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
    mask: Optional[np.ndarray] = None,
    image_id: Optional[str] = None
) -> np.ndarray:
    """
    Normalize image intensity, optionally only within a mask region.
    
    Args:
        image: Input grayscale image
        method: Normalization method ('zscore' or 'minmax')
        mask: Optional binary mask to normalize only inside mask (0 = background, 255 = foreground)
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
    
    if mask is not None:
        # Normalize only within the masked region to avoid background artifacts
        logger.info("Applying normalization inside brain mask only", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'normalization'
        })
        
        # Get values only inside the mask for computing statistics
        mask_bool = mask > 0
        masked_values = img_float[mask_bool]
        
        if method == "zscore":
            mean = np.mean(masked_values)
            std = np.std(masked_values)
            # Normalize the entire image but only change values inside the mask
            normalized = img_float.copy()
            normalized[mask_bool] = (masked_values - mean) / (std + 1e-8)
            # Scale masked region back to [0, 255]
            masked_norm = normalized[mask_bool]
            if masked_norm.max() != masked_norm.min():
                normalized[mask_bool] = ((masked_norm - masked_norm.min()) / 
                                          (masked_norm.max() - masked_norm.min()) * 255)
            # Keep background as black (0)
            normalized[~mask_bool] = 0
            logger.info(f"Z-score normalization (masked): mean={mean:.2f}, std={std:.2f}", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'normalization'
            })
        else:  # minmax
            min_val = np.min(masked_values)
            max_val = np.max(masked_values)
            normalized = img_float.copy()
            normalized[mask_bool] = (masked_values - min_val) / (max_val - min_val + 1e-8) * 255
            # Keep background as black (0)
            normalized[~mask_bool] = 0
            logger.info(f"Min-max normalization (masked): min={min_val:.2f}, max={max_val:.2f}", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'normalization'
            })
    else:
        # Original behavior: normalize entire image
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
    
    # Create automatic brain mask to prevent artifacts on background
    # This mask is used for CLAHE, sharpening, and normalization
    # to avoid whitish lines and tile artifacts on the black background
    use_brain_mask = config.get('use_brain_mask', True)
    brain_mask = None
    if use_brain_mask:
        brain_mask = create_brain_mask(
            motion_reduced,
            threshold_value=config.get('brain_mask_threshold', 10),
            image_id=image_id
        )
    
    # Step 4: Enhance contrast (CLAHE) - apply within brain mask to avoid tile artifacts
    contrast = enhance_contrast_clahe(
        motion_reduced,
        clip_limit=config.get('clahe_clip_limit', 2.0),
        tile_grid_size=config.get('clahe_tile_grid_size', (8, 8)),
        mask=brain_mask,
        image_id=image_id
    )
    
    # Step 5: Sharpen (recover fine details) - apply within brain mask to avoid edge artifacts
    sharpened = unsharp_mask(
        contrast,
        radius=config.get('unsharp_radius', 1.5),
        amount=config.get('unsharp_amount', 1.5),
        mask=brain_mask,
        image_id=image_id
    )
    
    # Step 6: Normalize (standardize intensity range) - apply within brain mask
    normalized = normalize_image(
        sharpened,
        method=config.get('normalize_method', 'zscore'),
        mask=brain_mask,
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
