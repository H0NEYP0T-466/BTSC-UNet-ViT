"""
Skull stripping utilities using HD-BET.
Removes skull, eyes, skin, and background to extract brain-only tissue.
"""
import time
from typing import Optional, Tuple
import numpy as np
import torch
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global variable to cache HD-BET model
_hdbet_model = None


def get_hdbet_device() -> str:
    """
    Determine the best device for HD-BET inference.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        logger.info("HD-BET will use GPU (CUDA)", extra={
            'image_id': None,
            'path': None,
            'stage': 'hdbet_init'
        })
        return 'cuda'
    else:
        logger.info("HD-BET will use CPU", extra={
            'image_id': None,
            'path': None,
            'stage': 'hdbet_init'
        })
        return 'cpu'


def load_hdbet_model():
    """
    Load HD-BET model (lazy loading).
    
    Returns:
        HD-BET model instance
    """
    global _hdbet_model
    
    if _hdbet_model is None:
        try:
            from HD_BET.run import load_model as hdbet_load_model
            
            logger.info("Loading HD-BET model...", extra={
                'image_id': None,
                'path': None,
                'stage': 'hdbet_load'
            })
            
            device = get_hdbet_device()
            _hdbet_model = hdbet_load_model(device=device)
            
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
    
    return _hdbet_model


def skull_strip_hdbet(
    image: np.ndarray,
    image_id: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform skull stripping using HD-BET on a 2D MRI image.
    
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
    
    # Ensure grayscale
    if len(image.shape) == 3:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # HD-BET expects 3D volumes (slices, height, width)
    # For 2D images, we add a slice dimension
    image_3d = np.expand_dims(image, axis=0)  # (1, H, W)
    
    # Normalize to [0, 1] for HD-BET
    image_normalized = image_3d.astype(np.float32) / 255.0
    
    try:
        # Load model
        model = load_hdbet_model()
        device = get_hdbet_device()
        
        # Convert to tensor and add batch dimension
        # HD-BET expects (batch, channels, depth, height, width)
        input_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        
        logger.info(f"HD-BET input tensor shape: {input_tensor.shape}", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'skull_stripping'
        })
        
        # Run inference
        with torch.no_grad():
            # HD-BET returns probabilities for brain tissue
            brain_prob = model(input_tensor)
            
            # Threshold to get binary mask
            brain_mask_3d = (brain_prob > 0.5).cpu().numpy().astype(np.uint8)
            
            # Remove batch and channel dimensions
            brain_mask_3d = brain_mask_3d[0, 0]  # (1, H, W)
            
            # Extract 2D mask (first slice)
            brain_mask_2d = brain_mask_3d[0]  # (H, W)
            
            # Apply mask to original image
            brain_only = image.copy()
            brain_only[brain_mask_2d == 0] = 0  # Set non-brain pixels to black
            
            # Convert mask to 0-255 range for visualization
            brain_mask_2d = (brain_mask_2d * 255).astype(np.uint8)
        
        duration = time.time() - start_time
        
        # Calculate statistics
        brain_area_pct = (np.sum(brain_mask_2d > 0) / brain_mask_2d.size) * 100
        
        logger.info(
            f"Skull stripping completed in {duration:.3f}s, brain_area={brain_area_pct:.2f}%",
            extra={
                'image_id': image_id,
                'path': None,
                'stage': 'skull_stripping'
            }
        )
        
        return brain_only, brain_mask_2d
        
    except Exception as e:
        logger.error(f"HD-BET skull stripping failed: {e}", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'skull_stripping_error'
        })
        
        # Fallback: Return original image and a full mask
        logger.warning("Falling back to no skull stripping", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'skull_stripping_fallback'
        })
        
        brain_mask = np.ones_like(image, dtype=np.uint8) * 255
        return image.copy(), brain_mask


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
