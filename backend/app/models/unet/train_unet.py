"""
UNet training script for BraTS dataset.
Optimized for custom BraTS .h5 dataset with extreme class imbalance.
"""
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from app.models.unet.model import get_unet_model
from app.models.unet.utils import DiceBCELoss, visualize_batch
from app.config import settings
from app.utils.logger import get_logger
from app.utils.metrics import dice_coefficient

logger = get_logger(__name__)


def pixel_accuracy(preds, masks):
    """Calculate pixel-wise accuracy."""
    correct = (preds == masks).sum().item()
    total = masks.numel()
    return correct / total if total > 0 else 0.0


class UNetTrainer:
    """UNet trainer for brain tumor segmentation with extreme class imbalance handling."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        checkpoint_dir: Optional[Path] = None,
        visualize_every: int = 5  # Visualize every N epochs
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir or settings.CHECKPOINTS_UNET
        self.visualize_every = visualize_every

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # ✅ FIX: Use combined Dice + BCE loss for extreme class imbalance
        self.criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        
        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        self.best_dice = 0.0

        logger.info(
            f"UNet trainer initialized: lr={learning_rate}, device={device}, "
            f"loss=DiceBCE, scheduler=ReduceLROnPlateau",
            extra={'image_id': None, 'path': str(self.checkpoint_dir), 'stage': 'train_init'}
        )

    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_acc = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                acc = pixel_accuracy(preds, masks)
                dice = dice_coefficient(preds.cpu().numpy(), masks.cpu().numpy())

            total_loss += loss.item()
            total_dice += dice
            total_acc += acc
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{dice:.4f}",
                'acc': f"{acc:.4f}"
            })
            
            # Visualize first batch every N epochs
            if batch_idx == 0 and epoch % self.visualize_every == 0:
                save_path = self.checkpoint_dir / f'train_vis_epoch_{epoch}.png'
                visualize_batch(
                    images[:4], masks[:4], preds[:4],
                    num_samples=4, save_path=str(save_path)
                )

        avg_loss = total_loss / len(self.train_loader)
        avg_dice = total_dice / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader)
        return avg_loss, avg_dice, avg_acc

    def validate_epoch(self, epoch: int) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_acc = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Calculate Dice and Accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()
                dice = dice_coefficient(preds.cpu().numpy(), masks.cpu().numpy())
                acc = pixel_accuracy(preds, masks)

                total_loss += loss.item()
                total_dice += dice
                total_acc += acc
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{dice:.4f}",
                    'acc': f"{acc:.4f}"
                })
                
                # Visualize first batch every N epochs
                if batch_idx == 0 and epoch % self.visualize_every == 0:
                    save_path = self.checkpoint_dir / f'val_vis_epoch_{epoch}.png'
                    visualize_batch(
                        images[:4], masks[:4], preds[:4],
                        num_samples=4, save_path=str(save_path)
                    )

        avg_loss = total_loss / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        avg_acc = total_acc / len(self.val_loader)

        return avg_loss, avg_dice, avg_acc

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice
        }

        # Save last checkpoint
        last_path = self.checkpoint_dir / 'unet_last.pth'
        torch.save(checkpoint, last_path)
        logger.info(f"Checkpoint saved: {last_path}", extra={
            'image_id': None,
            'path': str(last_path),
            'stage': 'checkpoint_save'
        })

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / settings.UNET_CHECKPOINT_NAME
            torch.save(checkpoint, best_path)
            logger.info(f"Best checkpoint saved: {best_path} (Dice: {self.best_dice:.4f})", extra={
                'image_id': None,
                'path': str(best_path),
                'stage': 'checkpoint_save'
            })
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load checkpoint with graceful handling of missing scheduler state.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available (for backward compatibility)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load best dice if available
        if 'best_dice' in checkpoint:
            self.best_dice = checkpoint['best_dice']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}", extra={
            'image_id': None,
            'path': str(checkpoint_path),
            'stage': 'checkpoint_load'
        })

    def train(self, num_epochs: int):
        """Train for multiple epochs."""
        logger.info(
            f"Starting UNet training for {num_epochs} epochs with DiceBCE loss",
            extra={'image_id': None, 'path': None, 'stage': 'train_start'}
        )

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_dice, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_dice, val_acc = self.validate_epoch(epoch)

            epoch_duration = time.time() - epoch_start

            # Update learning rate based on validation Dice
            self.scheduler.step(val_dice)

            # Log epoch metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} completed: "
                f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}, val_acc={val_acc:.4f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}, duration={epoch_duration:.2f}s",
                extra={'image_id': None, 'path': None, 'stage': 'train_epoch'}
            )

            # Save checkpoint
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                logger.info(f"🎉 New best Dice score: {self.best_dice:.4f}", extra={
                    'image_id': None,
                    'path': None,
                    'stage': 'train_epoch'
                })

            self.save_checkpoint(epoch, is_best=is_best)

        logger.info(f"Training completed. Best Dice: {self.best_dice:.4f}", extra={
            'image_id': None,
            'path': None,
            'stage': 'train_complete'
        })


def main():
    """Main training function."""
    # Set seed for reproducibility
    torch.manual_seed(settings.SEED)

    logger.info(f"Training seed set to {settings.SEED}", extra={
        'image_id': None,
        'path': None,
        'stage': 'train_init'
    })

    # Check if dataset exists
    if not settings.BRATS_ROOT.exists():
        logger.error(
            f"Dataset not found at {settings.BRATS_ROOT}. "
            f"Please ensure the dataset is available.",
            extra={'image_id': None, 'path': str(settings.BRATS_ROOT), 'stage': 'train_init'}
        )
        raise FileNotFoundError(f"Dataset not found: {settings.BRATS_ROOT}")

    logger.info(f"Loading dataset from {settings.BRATS_ROOT}", extra={
        'image_id': None,
        'path': str(settings.BRATS_ROOT),
        'stage': 'train_init'
    })

    # Import datamodule
    from app.models.unet.datamodule import create_unet_dataloaders

    # Create dataloaders
    train_loader, val_loader = create_unet_dataloaders(
        root_dir=settings.BRATS_ROOT,
        batch_size=settings.BATCH_SIZE,
        num_workers=settings.NUM_WORKERS,
        train_split=0.8,
        image_size=(256, 256),
        transform=None  # Can add albumentations transforms here
    )

    logger.info(f"Dataloaders created: train batches={len(train_loader)}, val batches={len(val_loader)}", extra={
        'image_id': None,
        'path': None,
        'stage': 'train_init'
    })

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}", extra={
        'image_id': None,
        'path': None,
        'stage': 'train_init'
    })

    model = get_unet_model(
        in_channels=settings.UNET_IN_CHANNELS,
        out_channels=settings.UNET_OUT_CHANNELS,
        features=settings.UNET_CHANNELS
    )

    logger.info("Model created successfully", extra={
        'image_id': None,
        'path': None,
        'stage': 'train_init'
    })

    # Create trainer and train
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=settings.LEARNING_RATE,
        checkpoint_dir=settings.CHECKPOINTS_UNET,
        visualize_every=5
    )

    trainer.train(settings.NUM_EPOCHS)


if __name__ == "__main__":
    from app.logging_config import setup_logging
    setup_logging(settings.LOG_LEVEL)
    main()
