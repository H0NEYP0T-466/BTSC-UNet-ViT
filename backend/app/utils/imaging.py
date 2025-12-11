"""
Imaging utilities for file I/O, overlays, and conversions.
"""
import uuid
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from app.utils.logger import get_logger

logger = get_logger(__name__)


def read_image(path: str, as_rgb: bool = True) -> np.ndarray:
    """
    Read image from file.
    
    Args:
        path: Path to image file
        as_rgb: If True, return RGB; otherwise BGR
        
    Returns:
        Image as numpy array
    """
    logger.info(f"Reading image from {path}", extra={
        'image_id': None,
        'path': path,
        'stage': 'io'
    })
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image from {path}")
    
    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    logger.info(f"Image loaded, shape: {img.shape}, dtype: {img.dtype}", extra={
        'image_id': None,
        'path': path,
        'stage': 'io'
    })
    
    return img


def save_image(image: np.ndarray, path: str) -> str:
    """
    Save image to file.
    
    Args:
        image: Image array to save
        path: Output path
        
    Returns:
        Path where image was saved
    """
    logger.info(f"Saving image to {path}", extra={
        'image_id': None,
        'path': path,
        'stage': 'io'
    })
    
    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert RGB to BGR if needed for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(path, image)
    
    logger.info(f"Image saved successfully", extra={
        'image_id': None,
        'path': path,
        'stage': 'io'
    })
    
    return path


def generate_unique_filename(prefix: str = "img", extension: str = "png") -> str:
    """
    Generate a unique filename using UUID.
    
    Args:
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        Unique filename
    """
    unique_id = uuid.uuid4().hex[:12]
    return f"{prefix}_{unique_id}.{extension}"


def create_overlay(
    base_image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Create an overlay of mask on base image.
    
    Args:
        base_image: Base image (grayscale or RGB)
        mask: Binary mask
        alpha: Transparency of overlay (0-1)
        color: Color of mask overlay (RGB)
        
    Returns:
        Overlay image
    """
    logger.info("Creating mask overlay", extra={
        'image_id': None,
        'path': None,
        'stage': 'visualization'
    })
    
    # Ensure base image is RGB
    if len(base_image.shape) == 2:
        base_rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = base_image.copy()
    
    # Ensure mask is binary
    mask_binary = (mask > 0).astype(np.uint8) * 255
    
    # Create colored overlay
    overlay = base_rgb.copy()
    overlay[mask_binary > 0] = color
    
    # Blend
    result = cv2.addWeighted(base_rgb, 1 - alpha, overlay, alpha, 0)
    
    logger.info("Mask overlay created successfully", extra={
        'image_id': None,
        'path': None,
        'stage': 'visualization'
    })
    
    return result


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply binary mask to image.
    
    Args:
        image: Input image
        mask: Binary mask
        
    Returns:
        Masked image
    """
    logger.info("Applying mask to image", extra={
        'image_id': None,
        'path': None,
        'stage': 'masking'
    })
    
    mask_binary = (mask > 0).astype(np.uint8)
    
    if len(image.shape) == 3:
        masked = image * mask_binary[:, :, np.newaxis]
    else:
        masked = image * mask_binary
    
    logger.info("Mask applied successfully", extra={
        'image_id': None,
        'path': None,
        'stage': 'masking'
    })
    
    return masked


def crop_to_bounding_box(image: np.ndarray, mask: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Crop image to bounding box of mask with padding.
    
    Args:
        image: Input image
        mask: Binary mask
        padding: Padding around bounding box
        
    Returns:
        Cropped image
    """
    logger.info("Cropping to bounding box", extra={
        'image_id': None,
        'path': None,
        'stage': 'cropping'
    })
    
    # Find contours
    mask_binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning("No contours found in mask, returning original image", extra={
            'image_id': None,
            'path': None,
            'stage': 'cropping'
        })
        return image
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    cropped = image[y1:y2, x1:x2]
    
    logger.info(f"Cropped to bbox: ({x1},{y1}) to ({x2},{y2}), size: {cropped.shape}", extra={
        'image_id': None,
        'path': None,
        'stage': 'cropping'
    })
    
    return cropped


def bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to numpy array.
    
    Args:
        image_bytes: Image bytes
        
    Returns:
        Image as numpy array
    """
    logger.info("Converting bytes to numpy array", extra={
        'image_id': None,
        'path': None,
        'stage': 'conversion'
    })
    
    # Convert bytes to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to numpy
    img_array = np.array(pil_image)
    
    logger.info(f"Converted to numpy, shape: {img_array.shape}", extra={
        'image_id': None,
        'path': None,
        'stage': 'conversion'
    })
    
    return img_array


import io
