"""
Utility functions for Brain UNet training.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross Entropy Loss.
    Effective for segmentation tasks with class imbalance.
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined Dice + BCE loss.
        
        Args:
            predictions: Model outputs (logits) [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W]
            
        Returns:
            Combined loss value
        """
        # BCE loss (uses logits)
        bce_loss = self.bce(predictions, targets)
        
        # Dice loss (uses probabilities)
        probs = torch.sigmoid(predictions)
        
        # Flatten for dice calculation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return total_loss


def visualize_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    save_path: Path,
    num_samples: int = 4
):
    """
    Visualize a batch of images, masks, and predictions.
    
    Args:
        images: Input images [B, 1, H, W]
        masks: Ground truth masks [B, 1, H, W]
        predictions: Predicted masks (after sigmoid) [B, 1, H, W]
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    batch_size = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Get numpy arrays
        img = images[i, 0].cpu().numpy()
        mask = masks[i, 0].cpu().numpy()
        pred = predictions[i, 0].cpu().numpy()
        
        # Plot image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {save_path}", extra={
        'image_id': None, 'path': str(save_path), 'stage': 'visualization'
    })


def calculate_iou(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) metric.
    
    Args:
        predictions: Predicted masks [B, H, W] or [B, 1, H, W]
        targets: Ground truth masks [B, H, W] or [B, 1, H, W]
        threshold: Threshold for binarizing predictions
        
    Returns:
        IoU score
    """
    # Binarize predictions
    pred_binary = (predictions > threshold).astype(np.float32)
    target_binary = (targets > threshold).astype(np.float32)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return float(iou)
