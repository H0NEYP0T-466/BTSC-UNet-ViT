"""
UNet utilities including loss functions and visualization helpers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss for handling extreme class imbalance in binary segmentation.
    
    This is crucial for datasets where tumor pixels are < 1% (e.g., 0.17% in our case).
    
    Why this combination works for extreme imbalance:
    - Dice loss: Focuses on overlap between prediction and ground truth, naturally handles
      class imbalance by considering intersection/union rather than pixel-by-pixel comparison.
      However, Dice can have unstable gradients when predictions are far from ground truth.
    - BCE loss: Provides stable gradients across all training stages and helps with
      pixel-level accuracy. However, BCE alone suffers from class imbalance issues.
    - Combined: Dice handles imbalance and focuses on overlap, while BCE provides stable
      gradients. This combination outperforms weighted BCE or focal loss for extreme
      imbalance in medical imaging (see Sudre et al. 2017, "Generalised Dice overlap as
      a deep learning loss function for highly unbalanced segmentations").
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        """
        Args:
            dice_weight: Weight for Dice loss component
            bce_weight: Weight for BCE loss component
            smooth: Smoothing factor for Dice calculation
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined Dice + BCE loss.
        
        Args:
            logits: Raw model outputs (before sigmoid), shape (B, 1, H, W)
            targets: Ground truth masks, shape (B, 1, H, W)
        
        Returns:
            Combined loss value
        """
        # BCE loss on logits
        bce_loss = self.bce(logits, targets)
        
        # Dice loss (apply sigmoid first)
        probs = torch.sigmoid(logits)
        
        # Flatten for dice calculation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        # Dice loss (1 - dice_coeff)
        dice_loss = 1 - dice_coeff
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return total_loss


def visualize_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    num_samples: int = 4,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a batch of images, masks, and predictions for sanity checking.
    
    Args:
        images: Image tensor (B, C, H, W)
        masks: Ground truth masks (B, 1, H, W)
        predictions: Predicted masks (B, 1, H, W), optional
        num_samples: Number of samples to visualize
        save_path: Path to save visualization, if provided
    """
    num_samples = min(num_samples, images.shape[0])
    
    # Determine number of columns
    num_cols = 3 if predictions is not None else 2
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 4, num_samples * 4))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Get image (use first channel or RGB composite)
        img = images[i].cpu().numpy()
        if img.shape[0] == 1:
            img_display = img[0]
        elif img.shape[0] == 3:
            img_display = np.transpose(img, (1, 2, 0))
        else:
            # For multi-channel (e.g., 4 channels), show first channel
            img_display = img[0]
        
        # Normalize for display
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
        
        # Display image
        axes[i, 0].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
        axes[i, 0].set_title(f'Sample {i+1}: Image')
        axes[i, 0].axis('off')
        
        # Display ground truth mask with 'hot' colormap for visibility
        mask = masks[i, 0].cpu().numpy()
        im1 = axes[i, 1].imshow(mask, cmap='hot', interpolation='nearest')
        axes[i, 1].set_title(f'Sample {i+1}: Ground Truth\n(Tumor ratio: {mask.mean()*100:.3f}%)')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Display prediction if available
        if predictions is not None:
            pred = predictions[i, 0].cpu().numpy()
            im2 = axes[i, 2].imshow(pred, cmap='hot', interpolation='nearest')
            axes[i, 2].set_title(f'Sample {i+1}: Prediction\n(Tumor ratio: {pred.mean()*100:.3f}%)')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}", extra={
            'image_id': None, 'path': save_path, 'stage': 'visualization'
        })
    
    plt.close()


def visualize_tumor_mask(
    mask: np.ndarray,
    title: str = "Tumor Mask",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize tumor mask with enhanced visibility for tiny tumor regions.
    
    Args:
        mask: Mask array (H, W) with values in [0, 1]
        title: Plot title
        save_path: Path to save visualization, if provided
    """
    plt.figure(figsize=(8, 8))
    
    # Use 'hot' colormap which makes small values more visible
    plt.imshow(mask, cmap='hot', interpolation='nearest')
    plt.title(f"{title}\nTumor ratio: {mask.mean()*100:.3f}%")
    plt.colorbar(label='Tumor Probability')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Tumor mask visualization saved to {save_path}", extra={
            'image_id': None, 'path': save_path, 'stage': 'visualization'
        })
    
    plt.close()
