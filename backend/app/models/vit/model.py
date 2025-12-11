"""
Vision Transformer (ViT) model for brain tumor classification.
"""
import torch
import torch.nn as nn
from typing import Optional
import timm
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ViTClassifier(nn.Module):
    """Vision Transformer for brain tumor classification."""
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 4,
        pretrained: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        logger.info(f"Loading ViT model: {model_name}, pretrained={pretrained}", extra={
            'image_id': None,
            'path': None,
            'stage': 'vit_init'
        })
        
        # Load pretrained ViT
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Replace classification head
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown head structure for model {model_name}")
        
        logger.info(f"ViT classifier initialized with {num_classes} classes", extra={
            'image_id': None,
            'path': None,
            'stage': 'vit_init'
        })
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # Always keep head trainable
        if hasattr(self.backbone, 'head'):
            for param in self.backbone.head.parameters():
                param.requires_grad = True
        elif hasattr(self.backbone, 'fc'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        
        status = "frozen" if freeze else "unfrozen"
        logger.info(f"Backbone {status}, head trainable", extra={
            'image_id': None,
            'path': None,
            'stage': 'vit_config'
        })


def get_vit_model(
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    pretrained: bool = True
) -> ViTClassifier:
    """
    Get ViT classifier model.
    
    Args:
        model_name: Name of ViT model from timm
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        ViT classifier model
    """
    model_name = model_name or settings.VIT_MODEL_NAME
    num_classes = num_classes or settings.VIT_NUM_CLASSES
    
    model = ViTClassifier(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(
        f"Created ViT model: total_params={num_params}, trainable_params={num_trainable}",
        extra={'image_id': None, 'path': None, 'stage': 'vit_creation'}
    )
    
    return model
