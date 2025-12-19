"""
Brain UNet training script for NFBS dataset.
"""
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from app.models.brain_unet.model import get_brain_unet_model
from app.models.brain_unet.utils import DiceBCELoss, visualize_batch, calculate_iou
from app.config import settings
from app.utils.logger import get_logger
from app.utils.metrics import dice_coefficient

logger = get_logger(__name__)


def pixel_accuracy(preds, masks):
    """Calculate pixel-wise accuracy."""
    correct = (preds == masks).sum().item()
    total = masks.numel()
    return correct / total if total > 0 else 0.0


class BrainUNetTrainer:
    """Brain UNet trainer for brain segmentation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        checkpoint_dir: Optional[Path] = None,
        visualize_every: int = 5,
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints/brain_unet")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_every = visualize_every
        self.use_amp = use_amp and device == 'cuda'

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Use combined Dice + BCE loss
        self.criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # AMP scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.best_dice = 0.0

        logger.info(
            f"BrainUNet trainer initialized: lr={learning_rate}, device={device}, "
            f"loss=DiceBCE, scheduler=ReduceLROnPlateau, use_amp={self.use_amp}",
            extra={'image_id': None, 'path': str(self.checkpoint_dir), 'stage': 'train_init'}
        )

    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch. Returns (loss, dice, iou, acc)."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # Forward with AMP
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

            # Calculate metrics (keep on GPU, no numpy conversion)
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                # Fast Dice calculation on GPU
                intersection = (preds * masks).sum()
                dice = (2.0 * intersection) / (preds.sum() + masks.sum() + 1e-8)
                dice = dice.item()

            total_loss += loss.item()
            total_dice += dice

            # Update progress bar (reduced info to speed up display)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_dice = total_dice / len(self.train_loader)

        # Return 0.0 for iou and acc since we don't calculate them in training for speed
        # Full metrics are calculated in validation
        return avg_loss, avg_dice, 0.0, 0.0

    def validate(self, epoch: int) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_acc = 0.0

        # For visualization
        sample_images = None
        sample_masks = None
        sample_preds = None

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                # Forward with AMP
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Calculate metrics (keep on GPU when possible)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                # Fast GPU-based metrics - minimize CPU transfers
                intersection = (preds * masks).sum()
                dice = (2.0 * intersection) / (preds.sum() + masks.sum() + 1e-8)
                
                # IoU calculation on GPU
                intersection_iou = (preds * masks).sum()
                union = (preds + masks).clamp(0, 1).sum()
                iou = intersection_iou / (union + 1e-8)
                
                # Accuracy on GPU
                acc = (preds == masks).float().mean()
                
                # Single transfer to CPU for all metrics
                dice = dice.item()
                iou = iou.item()
                acc = acc.item()

                total_loss += loss.item()
                total_dice += dice
                total_iou += iou
                total_acc += acc

                # Save first batch for visualization
                if batch_idx == 0:
                    sample_images = images
                    sample_masks = masks
                    sample_preds = preds

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice:.4f}',
                    'iou': f'{iou:.4f}',
                    'acc': f'{acc:.4f}'
                })

        avg_loss = total_loss / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)
        avg_acc = total_acc / len(self.val_loader)

        # Visualize samples
        if epoch % self.visualize_every == 0 and sample_images is not None:
            vis_path = self.checkpoint_dir / f"vis_epoch_{epoch}.png"
            visualize_batch(sample_images, sample_masks, sample_preds, vis_path)

        return avg_loss, avg_dice, avg_iou, avg_acc

    def train(self, num_epochs: int):
        """Train the model for multiple epochs."""
        logger.info(f"Starting training for {num_epochs} epochs", extra={
            'image_id': None, 'path': str(self.checkpoint_dir), 'stage': 'train_start'
        })

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*80}")

            # Train
            train_loss, train_dice, train_iou, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_dice, val_iou, val_acc = self.validate(epoch)

            # Update learning rate
            self.scheduler.step(val_dice)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train: loss={train_loss:.4f}, dice={train_dice:.4f}, "
                  f"iou={train_iou:.4f}, acc={train_acc:.4f}")
            print(f"  Val:   loss={val_loss:.4f}, dice={val_dice:.4f}, "
                  f"iou={val_iou:.4f}, acc={val_acc:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Save best model
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                best_path = self.checkpoint_dir / "brain_unet_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'dice_score': val_dice,
                    'iou_score': val_iou
                }, best_path)
                print(f"  ✅ Saved best model (dice={val_dice:.4f}) to {best_path}")

            # Save last model
            last_path = self.checkpoint_dir / "brain_unet_last.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'dice_score': val_dice,
                'iou_score': val_iou
            }, last_path)

            logger.info(
                f"Epoch {epoch} completed: train_dice={train_dice:.4f}, "
                f"val_dice={val_dice:.4f}, best_dice={self.best_dice:.4f}",
                extra={'image_id': None, 'path': str(self.checkpoint_dir), 'stage': 'train_epoch'}
            )

        print(f"\n{'='*80}")
        print(f"Training completed! Best Dice: {self.best_dice:.4f}")
        print(f"{'='*80}")
