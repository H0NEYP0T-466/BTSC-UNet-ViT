"""
UNet Tumor inference module for segmentation.
"""
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import cv2
import matplotlib.cm as cm
from app.models.unet_tumor.model import get_unet_tumor_model
from app.config import settings
from app.utils.logger import get_logger
from app.utils.imaging import create_overlay, crop_to_bounding_box

logger = get_logger(__name__)

# Threshold for minimum tumor pixels - if below this, use probability map for visibility
MIN_TUMOR_PIXELS_THRESHOLD = 100


class UNetTumorInference:
    """UNet Tumor inference class for tumor segmentation on PNG images."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize UNet Tumor inference.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path or str(
            settings.CHECKPOINTS_UNET_TUMOR / settings.UNET_TUMOR_CHECKPOINT_NAME
        )
        
        logger.info(f"Initializing UNet Tumor inference on device: {self.device}", extra={
            'image_id': None,
            'path': self.checkpoint_path,
            'stage': 'unet_tumor_init'
        })
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        logger.info("UNet Tumor model loaded and ready for inference", extra={
            'image_id': None,
            'path': self.checkpoint_path,
            'stage': 'unet_tumor_init'
        })
    
    def _load_model(self) -> torch.nn.Module:
        """Load UNet Tumor model from checkpoint."""
        model = get_unet_tumor_model(
            in_channels=settings.UNET_TUMOR_IN_CHANNELS,
            out_channels=settings.UNET_TUMOR_OUT_CHANNELS,
            features=settings.UNET_TUMOR_CHANNELS,
            dropout_rate=settings.UNET_TUMOR_DROPOUT
        )
        model = model.to(self.device)
        
        # Load checkpoint if exists
        checkpoint_file = Path(self.checkpoint_path)
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_path}", extra={
                'image_id': None,
                'path': self.checkpoint_path,
                'stage': 'unet_tumor_load'
            })
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("Checkpoint loaded successfully", extra={
                'image_id': None,
                'path': self.checkpoint_path,
                'stage': 'unet_tumor_load'
            })
        else:
            logger.warning(f"Checkpoint not found at {self.checkpoint_path}, using untrained model", extra={
                'image_id': None,
                'path': self.checkpoint_path,
                'stage': 'unet_tumor_load'
            })
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for UNet Tumor input.
        
        Args:
            image: Input image (grayscale or RGB, H x W or H x W x C)
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Ensure RGB format
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize to expected input size (256x256)
        target_size = (256, 256)
        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1))
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # (1, 3, H, W)
        
        return tensor.to(self.device)
    
    def postprocess_mask(self, output: torch.Tensor, original_size: tuple, threshold: float = 0.5) -> np.ndarray:
        """
        Postprocess model output to binary mask.
        
        Args:
            output: Model output tensor
            original_size: Original image size (H, W)
            threshold: Threshold for binarization
            
        Returns:
            Binary mask (H, W) with values 0-255 for visualization
        """
        # Apply sigmoid and threshold
        mask_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Resize to original size
        mask_prob_resized = cv2.resize(mask_prob, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Create binary mask
        binary_mask = (mask_prob_resized > threshold).astype(np.uint8) * 255
        
        # If very few tumor pixels, use probability map for visibility
        tumor_pixels = np.sum(binary_mask > 0)
        if tumor_pixels < MIN_TUMOR_PIXELS_THRESHOLD:
            return (mask_prob_resized * 255).astype(np.uint8)
        
        return binary_mask
    
    def segment_image(
        self,
        image: np.ndarray,
        image_id: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment tumor in image.
        
        Args:
            image: Input image (preprocessed, grayscale or RGB)
            image_id: Image identifier for logging
            
        Returns:
            Dictionary with 'mask', 'overlay', 'segmented', and 'heatmap' images
        """
        start_time = time.time()
        logger.info("UNet Tumor segmentation started", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'unet_tumor_inference'
        })
        
        # Store original size
        if len(image.shape) == 2:
            original_size = image.shape
        else:
            original_size = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Get probability map for heatmap visualization
        prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
        prob_map_resized = cv2.resize(prob_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Postprocess to binary mask
        mask = self.postprocess_mask(output, original_size)
        
        # Calculate mask statistics
        mask_area_pct = (np.sum(mask > 0) / mask.size) * 100
        
        logger.info(
            f"UNet Tumor inference completed, mask_area={mask_area_pct:.3f}%",
            extra={'image_id': image_id, 'path': None, 'stage': 'unet_tumor_inference'}
        )
        
        # Create heatmap with 'hot' colormap
        colormap = cm.get_cmap('hot')
        heatmap_colored = colormap(prob_map_resized)
        heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Create overlay with enhanced visibility
        overlay = create_overlay(image, mask, alpha=0.5, color=(0, 255, 0))  # Green for tumor model
        
        # Create heatmap overlay
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        heatmap_overlay = cv2.addWeighted(image_rgb, 0.6, heatmap_rgb, 0.4, 0)
        
        # Create segmented image (cropped)
        segmented = crop_to_bounding_box(image, mask, padding=10)
        
        duration = time.time() - start_time
        logger.info(f"UNet Tumor segmentation processing completed in {duration:.3f}s", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'unet_tumor_inference'
        })
        
        return {
            'mask': mask,
            'overlay': overlay,
            'segmented': segmented,
            'heatmap': heatmap_overlay,
            'probability_map': (prob_map_resized * 255).astype(np.uint8)
        }


# Singleton instance
_unet_tumor_inference = None


def get_unet_tumor_inference() -> UNetTumorInference:
    """Get singleton UNet Tumor inference instance."""
    global _unet_tumor_inference
    if _unet_tumor_inference is None:
        _unet_tumor_inference = UNetTumorInference()
    return _unet_tumor_inference
