"""
Skull stripping utilities for 2D MRI images.
Removes skull, eyes, skin, and background to extract brain-only tissue.

For 2D images, we use:
1. HD-BET when possible (by converting 2D -> 3D NIfTI, processing, then extracting back)
2. Fallback to Otsu thresholding + morphological operations for simple cases
"""
import time
import os
import tempfile
from typing import Optional, Tuple
import numpy as np
import torch
import cv2
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global variable to cache HD-BET model
_hdbet_predictor = None


def get_hdbet_device() -> torch.device:
    """
    Determine the best device for HD-BET inference.
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        logger.info("HD-BET will use GPU (CUDA)", extra={
            'image_id': None,
            'path': None,
            'stage': 'hdbet_init'
        })
        return torch.device('cuda')
    else:
        logger.info("HD-BET will use CPU", extra={
            'image_id': None,
            'path': None,
            'stage': 'hdbet_init'
        })
        return torch.device('cpu')


def load_hdbet_predictor():
    """
    Load HD-BET predictor (lazy loading).
    
    Returns:
        HD-BET predictor instance
    """
    global _hdbet_predictor
    
    if _hdbet_predictor is None:
        try:
            from HD_BET.hd_bet_prediction import get_hdbet_predictor
            
            logger.info("Loading HD-BET model...", extra={
                'image_id': None,
                'path': None,
                'stage': 'hdbet_load'
            })
            
            device = get_hdbet_device()
            _hdbet_predictor = get_hdbet_predictor(
                use_tta=False,
                device=device,
                verbose=False
            )
            
            logger.info("HD-BET model loaded successfully", extra={
                'image_id': None,
                'path': None,
                'stage': 'hdbet_load'
            })
            
        except ImportError as e:
            logger.error(f"HD-BET not installed: {e}", extra={
                'image_id': None,
                'path': None,
                'stage': 'hdbet_load_error'
            })
            raise ImportError(
                "HD-BET is not installed. "
                "Please install it with: pip install HD-BET"
            )
        except Exception as e:
            logger.error(f"Failed to load HD-BET model: {e}", extra={
                'image_id': None,
                'path': None,
                'stage': 'hdbet_load_error'
            })
            raise
    
    return _hdbet_predictor


def simple_brain_extraction(
    image: np.ndarray,
    image_id: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple brain extraction using Otsu thresholding and morphological operations.
    This is a fallback method when HD-BET is not available or fails.
    
    Args:
        image: Input 2D grayscale MRI image (H, W)
        image_id: Image identifier for logging
        
    Returns:
        Tuple of:
            - brain_only: Skull-stripped brain tissue (same shape as input)
            - brain_mask: Binary brain mask (0 = background, 255 = brain)
    """
    logger.info("Using simple brain extraction (Otsu + morphology)", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'skull_stripping'
    })
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # Close small holes
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Remove small objects
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find the largest connected component (likely the brain)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels > 1:
        # Find largest component (excluding background at label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        brain_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        brain_mask = binary
    
    # Apply mask to original image
    brain_only = image.copy()
    brain_only[brain_mask == 0] = 0
    
    return brain_only, brain_mask


def skull_strip_hdbet_2d(
    image: np.ndarray,
    image_id: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform skull stripping using HD-BET on a 2D MRI image.
    Converts 2D image to 3D NIfTI, processes with HD-BET, then extracts result.
    
    Args:
        image: Input 2D grayscale MRI image (H, W)
        image_id: Image identifier for logging
        
    Returns:
        Tuple of:
            - brain_only: Skull-stripped brain tissue (same shape as input)
            - brain_mask: Binary brain mask (0 = background, 255 = brain)
    """
    start_time = time.time()
    
    logger.info("Starting skull stripping with HD-BET", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'skull_stripping'
    })
    
    try:
        import SimpleITK as sitk
        from HD_BET.hd_bet_prediction import hdbet_predict
        
        # Load predictor
        predictor = load_hdbet_predictor()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Convert 2D image to 3D volume (add depth dimension)
            # Replicate the slice 3 times to create a minimal 3D volume
            image_3d = np.stack([image, image, image], axis=0)  # (3, H, W)
            
            # Normalize to proper intensity range for MRI (0-1000)
            image_3d = (image_3d.astype(np.float32) / 255.0 * 1000).astype(np.float32)
            
            # Convert to SimpleITK image
            sitk_image = sitk.GetImageFromArray(image_3d)
            sitk_image.SetSpacing([1.0, 1.0, 1.0])  # Set spacing
            
            # Save as NIfTI
            input_file = os.path.join(tmpdir, 'input.nii.gz')
            output_file = os.path.join(tmpdir, 'output.nii.gz')
            sitk.WriteImage(sitk_image, input_file)
            
            # Run HD-BET prediction
            hdbet_predict(
                input_file_or_folder=input_file,
                output_file_or_folder=output_file,
                predictor=predictor,
                keep_brain_mask=True,
                compute_brain_extracted_image=True
            )
            
            # Read brain mask
            mask_file = os.path.join(tmpdir, 'output_bet.nii.gz')
            if os.path.exists(mask_file):
                mask_sitk = sitk.ReadImage(mask_file)
                mask_3d = sitk.GetArrayFromImage(mask_sitk)
                
                # Extract middle slice (index 1)
                brain_mask_2d = mask_3d[1].astype(np.uint8)
                
                # Convert to 0-255 range
                brain_mask_2d = (brain_mask_2d > 0).astype(np.uint8) * 255
                
                # Apply mask to original image
                brain_only = image.copy()
                brain_only[brain_mask_2d == 0] = 0
            else:
                raise FileNotFoundError("HD-BET did not generate brain mask")
        
        duration = time.time() - start_time
        
        # Calculate statistics
        brain_area_pct = (np.sum(brain_mask_2d > 0) / brain_mask_2d.size) * 100
        
        logger.info(
            f"HD-BET skull stripping completed in {duration:.3f}s, brain_area={brain_area_pct:.2f}%",
            extra={
                'image_id': image_id,
                'path': None,
                'stage': 'skull_stripping'
            }
        )
        
        return brain_only, brain_mask_2d
        
    except Exception as e:
        logger.warning(f"HD-BET failed: {e}, falling back to simple extraction", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'skull_stripping_fallback'
        })
        
        # Fallback to simple method
        return simple_brain_extraction(image, image_id=image_id)


def skull_strip_hdbet(
    image: np.ndarray,
    image_id: Optional[str] = None,
    use_hdbet: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform skull stripping on a 2D MRI image.
    
    Args:
        image: Input 2D grayscale MRI image (H, W)
        image_id: Image identifier for logging
        use_hdbet: Whether to attempt HD-BET (True) or use simple method (False)
        
    Returns:
        Tuple of:
            - brain_only: Skull-stripped brain tissue (same shape as input)
            - brain_mask: Binary brain mask (0 = background, 255 = brain)
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if use_hdbet:
        try:
            return skull_strip_hdbet_2d(image, image_id=image_id)
        except Exception as e:
            logger.warning(f"HD-BET not available: {e}, using simple method", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'skull_stripping'
            })
            return simple_brain_extraction(image, image_id=image_id)
    else:
        return simple_brain_extraction(image, image_id=image_id)


def apply_mask_to_image(
    image: np.ndarray,
    mask: np.ndarray,
    background_value: int = 0
) -> np.ndarray:
    """
    Apply a binary mask to an image.
    
    Args:
        image: Input image (H, W)
        mask: Binary mask (H, W) with values 0 or 255
        background_value: Value to set for non-masked pixels
        
    Returns:
        Masked image
    """
    masked = image.copy()
    masked[mask == 0] = background_value
    return masked
