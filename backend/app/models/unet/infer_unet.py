"""
UNet inference module for segmentation.
"""
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import cv2
from app.models.unet.model import get_unet_model
from app.config import settings
from app.utils.logger import get_logger
from app.utils.imaging import create_overlay, apply_mask, crop_to_bounding_box

logger = get_logger(__name__)


class UNetInference:
    """UNet inference class for brain tumor segmentation."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize UNet inference.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path or str(
            settings.CHECKPOINTS_UNET / settings.UNET_CHECKPOINT_NAME
        )
        
        logger.info(f"Initializing UNet inference on device: {self.device}", extra={
            'image_id': None,
            'path': self.checkpoint_path,
            'stage': 'unet_init'
        })
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        logger.info("UNet model loaded and ready for inference", extra={
            'image_id': None,
            'path': self.checkpoint_path,
            'stage': 'unet_init'
        })
    
    def _load_model(self) -> torch.nn.Module:
        """Load UNet model from checkpoint."""
        model = get_unet_model(
            in_channels=settings.UNET_IN_CHANNELS,
            out_channels=settings.UNET_OUT_CHANNELS,
            features=settings.UNET_CHANNELS
        )
        model = model.to(self.device)
        
        # Load checkpoint if exists
        checkpoint_file = Path(self.checkpoint_path)
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_path}", extra={
                'image_id': None,
                'path': self.checkpoint_path,
                'stage': 'unet_load'
            })
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("Checkpoint loaded successfully", extra={
                'image_id': None,
                'path': self.checkpoint_path,
                'stage': 'unet_load'
            })
        else:
            logger.warning(f"Checkpoint not found at {self.checkpoint_path}, using untrained model", extra={
                'image_id': None,
                'path': self.checkpoint_path,
                'stage': 'unet_load'
            })
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for UNet input.
        
        Args:
            image: Input image (grayscale, H x W)
            
        Returns:
            Preprocessed tensor (1, 1, H, W)
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1]
        image_normalized = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch and channel dimensions
        tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def postprocess_mask(self, output: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """
        Postprocess model output to binary mask.
        
        Args:
            output: Model output tensor
            threshold: Threshold for binarization
            
        Returns:
            Binary mask (H, W)
        """
        # Apply sigmoid and threshold
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        return binary_mask
    
    def segment_image(
        self,
        image: np.ndarray,
        image_id: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment brain tumor in image.
        
        Args:
            image: Input image (preprocessed, grayscale)
            image_id: Image identifier for logging
            
        Returns:
            Dictionary with 'mask', 'overlay', and 'segmented' images
        """
        start_time = time.time()
        logger.info("UNet segmentation started", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'unet_inference'
        })
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        mask = self.postprocess_mask(output)
        
        # Calculate mask statistics
        mask_area_pct = (np.sum(mask > 0) / mask.size) * 100
        
        logger.info(f"UNet inference completed, mask_area_pct={mask_area_pct:.2f}%", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'unet_inference'
        })
        
        # Create overlay
        overlay = create_overlay(image, mask, alpha=0.5, color=(255, 0, 0))
        
        # Create segmented image (cropped)
        segmented = crop_to_bounding_box(image, mask, padding=10)
        
        duration = time.time() - start_time
        logger.info(f"Segmentation processing completed in {duration:.3f}s", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'unet_inference'
        })
        
        logger.info("Passing to next layer: ViT classification", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'unet_inference'
        })
        
        return {
            'mask': mask,
            'overlay': overlay,
            'segmented': segmented
        }


# Singleton instance
_unet_inference = None


def get_unet_inference() -> UNetInference:
    """Get singleton UNet inference instance."""
    global _unet_inference
    if _unet_inference is None:
        _unet_inference = UNetInference()
    return _unet_inference
