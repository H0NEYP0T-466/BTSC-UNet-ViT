"""
Utility functions for UNet Tumor model training.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks
            
        Returns:
            Dice loss value
        """
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for better training stability."""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined Dice + BCE loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks
            
        Returns:
            Combined loss value
        """
        # BCE loss
        bce_loss = self.bce(predictions, targets)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(predictions)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = targets.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks
            
        Returns:
            Focal loss value
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def visualize_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    num_samples: int = 4,
    save_path: Optional[str] = None
):
    """
    Visualize a batch of images, masks, and predictions.
    
    Args:
        images: Input images (B, C, H, W)
        masks: Ground truth masks (B, 1, H, W)
        predictions: Predicted masks (B, 1, H, W)
        num_samples: Number of samples to visualize
        save_path: Path to save the figure
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Image
        img = images[i].cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        else:
            img = img[0]
        
        # Mask
        mask = masks[i, 0].cpu().numpy()
        
        # Prediction
        pred = predictions[i, 0].cpu().numpy()
        
        # Plot
        if num_samples == 1:
            ax_row = axes
        else:
            ax_row = axes[i]
        
        ax_row[0].imshow(img if img.ndim == 3 else img, cmap='gray' if img.ndim == 2 else None)
        ax_row[0].set_title('Input Image')
        ax_row[0].axis('off')
        
        ax_row[1].imshow(mask, cmap='gray')
        ax_row[1].set_title('Ground Truth')
        ax_row[1].axis('off')
        
        ax_row[2].imshow(pred, cmap='gray')
        ax_row[2].set_title('Prediction')
        ax_row[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def dice_coefficient(predictions: np.ndarray, targets: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient.
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient value
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    intersection = np.sum(predictions * targets)
    dice = (2. * intersection + smooth) / (np.sum(predictions) + np.sum(targets) + smooth)
    
    return dice


def iou_score(predictions: np.ndarray, targets: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (IoU) score.
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    intersection = np.sum(predictions * targets)
    union = np.sum(predictions) + np.sum(targets) - intersection
    
    return (intersection + smooth) / (union + smooth)
