"""
ViT inference module for brain tumor classification.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from app.models.vit.model import get_vit_model
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ViTInference:
    """ViT inference class for brain tumor classification."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize ViT inference.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path or str(
            settings.CHECKPOINTS_VIT / settings.VIT_CHECKPOINT_NAME
        )
        self.class_names = settings.VIT_CLASS_NAMES
        
        logger.info(f"Initializing ViT inference on device: {self.device}", extra={
            'image_id': None,
            'path': self.checkpoint_path,
            'stage': 'vit_init'
        })
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((settings.VIT_IMAGE_SIZE, settings.VIT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("ViT model loaded and ready for inference", extra={
            'image_id': None,
            'path': self.checkpoint_path,
            'stage': 'vit_init'
        })
    
    def _load_model(self) -> torch.nn.Module:
        """Load ViT model from checkpoint."""
        model = get_vit_model()
        model = model.to(self.device)
        
        # Load checkpoint if exists
        checkpoint_file = Path(self.checkpoint_path)
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_path}", extra={
                'image_id': None,
                'path': self.checkpoint_path,
                'stage': 'vit_load'
            })
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("Checkpoint loaded successfully", extra={
                'image_id': None,
                'path': self.checkpoint_path,
                'stage': 'vit_load'
            })
        else:
            logger.warning(
                f"Checkpoint not found at {self.checkpoint_path}, using untrained/pretrained model",
                extra={'image_id': None, 'path': self.checkpoint_path, 'stage': 'vit_load'}
            )
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for ViT input.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def classify(
        self,
        image: np.ndarray,
        image_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Classify brain tumor in segmented image.
        
        Args:
            image: Input image (segmented tumor region)
            image_id: Image identifier for logging
            
        Returns:
            Dictionary with 'class', 'confidence', and 'logits'
        """
        start_time = time.time()
        logger.info("ViT classification started", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'vit_inference'
        })
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Get prediction
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = self.class_names[predicted_idx.item()]
        confidence_value = confidence.item()
        logits_list = logits[0].cpu().numpy().tolist()
        
        duration = time.time() - start_time
        
        logger.info(
            f"ViT classification completed: class={predicted_class}, "
            f"confidence={confidence_value:.4f}, duration={duration:.3f}s",
            extra={
                'image_id': image_id,
                'path': None,
                'stage': 'vit_inference'
            }
        )
        
        return {
            'class': predicted_class,
            'confidence': float(confidence_value),
            'logits': logits_list,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }


# Singleton instance
_vit_inference = None


def get_vit_inference() -> ViTInference:
    """Get singleton ViT inference instance."""
    global _vit_inference
    if _vit_inference is None:
        _vit_inference = ViTInference()
    return _vit_inference
