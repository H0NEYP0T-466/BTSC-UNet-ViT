"""
Brain UNet inference for brain extraction.
"""
import time
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import torch
import cv2
from app.models.brain_unet.model import get_brain_unet_model
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BrainUNetInference:
    """Brain UNet inference class for brain segmentation."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Brain UNet inference.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or settings.CHECKPOINTS_BRAIN_UNET / "brain_unet_best.pth"
        
        # Load model
        self.model = self._load_model()
        
        logger.info(
            f"BrainUNet inference initialized: device={self.device}, "
            f"model_path={self.model_path}",
            extra={'image_id': None, 'path': str(self.model_path), 'stage': 'infer_init'}
        )
    
    def _load_model(self) -> torch.nn.Module:
        """Load the trained model."""
        # Create model
        model = get_brain_unet_model(
            in_channels=1,
            out_channels=1,
            features=(32, 64, 128, 256, 512)
        )
        
        # Load checkpoint if exists
        if self.model_path.exists():
            logger.info(f"Loading checkpoint from {self.model_path}", extra={
                'image_id': None, 'path': str(self.model_path), 'stage': 'model_load'
            })
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(
                f"Loaded checkpoint: dice={checkpoint.get('dice_score', 'N/A')}",
                extra={'image_id': None, 'path': str(self.model_path), 'stage': 'model_load'}
            )
        else:
            logger.warning(
                f"No checkpoint found at {self.model_path}, using untrained model",
                extra={'image_id': None, 'path': str(self.model_path), 'stage': 'model_load'}
            )
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def segment_brain(
        self,
        image: np.ndarray,
        image_id: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment brain from input image.
        
        Args:
            image: Input grayscale image (H, W) in range [0, 255]
            image_id: Optional image identifier for logging
            
        Returns:
            Dictionary with:
                - mask: Binary brain mask (H, W) in range [0, 255]
                - brain_extracted: Brain-only image (H, W) in range [0, 255]
                - overlay: Brain mask overlay on original image
        """
        start_time = time.time()
        
        logger.info("Starting brain segmentation", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'brain_segment'
        })
        
        # Store original shape and image
        original_shape = image.shape
        original_image = image.copy()
        
        # Preprocess image
        # Normalize to [0, 1]
        image_normalized = image.astype(np.float32) / 255.0
        
        # Resize to model input size (256x256)
        image_resized = cv2.resize(image_normalized, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor [1, 1, H, W]
        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.sigmoid(output)
            mask_prob = prediction.squeeze().cpu().numpy()
        
        # Binarize mask (threshold at 0.5)
        mask_binary = (mask_prob > 0.5).astype(np.float32)
        
        # Resize back to original shape
        mask_resized = cv2.resize(mask_binary, (original_shape[1], original_shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Convert to uint8 [0, 255]
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        
        # Apply mask to get brain-extracted image
        brain_extracted = original_image.copy()
        brain_extracted[mask_resized < 0.5] = 0
        
        # Create overlay visualization
        overlay = self._create_overlay(original_image, mask_uint8)
        
        # Calculate statistics
        brain_percentage = (np.sum(mask_resized > 0.5) / mask_resized.size) * 100
        
        duration = time.time() - start_time
        
        logger.info(
            f"Brain segmentation completed in {duration:.3f}s, "
            f"brain_area={brain_percentage:.2f}%",
            extra={'image_id': image_id, 'path': None, 'stage': 'brain_segment'}
        )
        
        return {
            'mask': mask_uint8,
            'brain_extracted': brain_extracted,
            'overlay': overlay
        }
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create an overlay visualization of mask on image.
        
        Args:
            image: Original grayscale image [0, 255]
            mask: Binary mask [0, 255]
            
        Returns:
            RGB overlay image
        """
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
        
        # Create colored mask (green for brain)
        mask_colored = np.zeros_like(image_rgb)
        mask_colored[:, :, 1] = mask  # Green channel
        
        # Blend
        alpha = 0.3
        overlay = cv2.addWeighted(image_rgb, 1.0, mask_colored, alpha, 0)
        
        return overlay


# Singleton instance
_brain_unet_inference = None


def get_brain_unet_inference(
    model_path: Optional[Path] = None,
    device: Optional[str] = None
) -> BrainUNetInference:
    """
    Get singleton Brain UNet inference instance.
    
    Args:
        model_path: Optional path to model checkpoint
        device: Optional device to use
        
    Returns:
        BrainUNetInference instance
    """
    global _brain_unet_inference
    
    if _brain_unet_inference is None:
        _brain_unet_inference = BrainUNetInference(
            model_path=model_path,
            device=device
        )
    
    return _brain_unet_inference
