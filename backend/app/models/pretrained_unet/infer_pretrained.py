"""
Pretrained UNet inference module for brain tumor segmentation.
This module provides inference using a pretrained MONAI UNet model.
"""
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import cv2
from app.models.pretrained_unet.model import PretrainedUNet
from app.config import settings
from app.utils.logger import get_logger
from app.utils.imaging import create_overlay, crop_to_bounding_box

logger = get_logger(__name__)


class PretrainedUNetInference:
    """Pretrained UNet inference class for brain tumor segmentation."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize Pretrained UNet inference.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set checkpoint path
        if checkpoint_path is None:
            checkpoint_dir = settings.CHECKPOINTS_DIR / "pretrained_unet"
            checkpoint_path = str(checkpoint_dir / "unet_pretrained.pth")
        
        self.checkpoint_path = checkpoint_path
        
        logger.info(f"Initializing Pretrained UNet inference on device: {self.device}", extra={
            'image_id': None,
            'path': self.checkpoint_path,
            'stage': 'pretrained_unet_init'
        })
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        logger.info("Pretrained UNet model loaded and ready for inference", extra={
            'image_id': None,
            'path': self.checkpoint_path,
            'stage': 'pretrained_unet_init'
        })
    
    def _load_model(self) -> torch.nn.Module:
        """Load Pretrained UNet model from checkpoint."""
        # Create model with optimized architecture
        model = PretrainedUNet(
            in_channels=1,
            out_channels=1,
            spatial_dims=2,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
        model = model.to(self.device)
        
        # Load checkpoint if exists
        checkpoint_file = Path(self.checkpoint_path)
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_path}", extra={
                'image_id': None,
                'path': self.checkpoint_path,
                'stage': 'pretrained_unet_load'
            })
            
            # Load with weights_only for security (PyTorch >= 1.13)
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:
                # Fallback for older PyTorch versions (< 1.13)
                logger.warning("Using legacy torch.load (PyTorch < 1.13). Consider upgrading for security.", extra={
                    'image_id': None,
                    'path': self.checkpoint_path,
                    'stage': 'pretrained_unet_load'
                })
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded pretrained weights successfully", extra={
                    'image_id': None,
                    'path': self.checkpoint_path,
                    'stage': 'pretrained_unet_load'
                })
            else:
                logger.warning("Checkpoint format unexpected, using model as initialized", extra={
                    'image_id': None,
                    'path': self.checkpoint_path,
                    'stage': 'pretrained_unet_load'
                })
        else:
            logger.warning(
                f"Checkpoint not found at {self.checkpoint_path}. "
                "Run 'python -m app.models.pretrained_unet.download_model' to download the model.",
                extra={
                    'image_id': None,
                    'path': self.checkpoint_path,
                    'stage': 'pretrained_unet_load'
                }
            )
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for Pretrained UNet input.
        
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
        
        # Convert to tensor
        tensor = torch.from_numpy(image_normalized)
        
        # Add channel and batch dimensions: (H, W) -> (1, 1, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def postprocess_mask(
        self,
        output: torch.Tensor,
        threshold: Optional[float] = None,
        min_area: Optional[int] = None
    ) -> np.ndarray:
        """
        Postprocess model output to binary mask with improved tumor detection.
        
        Args:
            output: Model output tensor
            threshold: Threshold for binarization (default from settings)
            min_area: Minimum area for connected components (default from settings)
            
        Returns:
            Binary mask (H, W) with only tumor regions
        """
        from app.config import settings
        
        # Use config defaults if not provided
        if threshold is None:
            threshold = settings.SEGMENTATION_THRESHOLD
        if min_area is None:
            min_area = settings.SEGMENTATION_MIN_AREA
        
        # Apply sigmoid and threshold
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove small noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill small holes
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small connected components (further noise reduction)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Keep only components larger than min_area
        cleaned_mask = np.zeros_like(binary_mask)
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == i] = 255
        
        return cleaned_mask
    
    def segment_image(
        self,
        image: np.ndarray,
        image_id: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment brain tumor in image using pretrained UNet.
        
        Args:
            image: Input image (preprocessed, grayscale)
            image_id: Image identifier for logging
            threshold: Threshold for binary classification (default from settings)
            
        Returns:
            Dictionary with 'mask', 'overlay', and 'segmented' images
        """
        start_time = time.time()
        logger.info("Pretrained UNet segmentation started", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pretrained_unet_inference'
        })
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess with improved tumor detection (uses config defaults)
        mask = self.postprocess_mask(output, threshold=threshold)
        
        # Calculate mask statistics
        mask_area_pct = (np.sum(mask > 0) / mask.size) * 100
        
        logger.info(
            f"Pretrained UNet inference completed, mask_area_pct={mask_area_pct:.2f}%",
            extra={
                'image_id': image_id,
                'path': None,
                'stage': 'pretrained_unet_inference'
            }
        )
        
        # Create overlay
        overlay = create_overlay(image, mask, alpha=0.5, color=(255, 0, 0))
        
        # Create segmented image (cropped to bounding box)
        segmented = crop_to_bounding_box(image, mask, padding=10)
        
        duration = time.time() - start_time
        logger.info(f"Segmentation processing completed in {duration:.3f}s", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pretrained_unet_inference'
        })
        
        logger.info("Passing to next layer: ViT classification", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pretrained_unet_inference'
        })
        
        return {
            'mask': mask,
            'overlay': overlay,
            'segmented': segmented
        }


# Singleton instance
_pretrained_unet_inference = None


def get_pretrained_unet_inference() -> PretrainedUNetInference:
    """Get singleton Pretrained UNet inference instance."""
    global _pretrained_unet_inference
    if _pretrained_unet_inference is None:
        _pretrained_unet_inference = PretrainedUNetInference()
    return _pretrained_unet_inference
