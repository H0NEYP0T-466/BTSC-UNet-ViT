"""
UNet Tumor training script for PNG-based tumor dataset.
Features:
- Data augmentation to prevent overfitting
- ReduceLROnPlateau scheduler
- EarlyStopping callback
- Combined Dice + BCE loss
"""
import time
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from app.models.unet_tumor.model import get_unet_tumor_model
from app.models.unet_tumor.utils import DiceBCELoss, visualize_batch, dice_coefficient
from app.utils.logger import get_logger

logger = get_logger(__name__)


def pixel_accuracy(preds: torch.Tensor, masks: torch.Tensor) -> float:
    """Calculate pixel-wise accuracy."""
    correct = (preds == masks).sum().item()
    total = masks.numel()
    return correct / total if total > 0 else 0.0


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_score: Validation metric (higher is better, e.g., Dice score)
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}", extra={
                    'image_id': None, 'path': None, 'stage': 'early_stopping'
                })
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


class UNetTumorTrainer:
    """UNet Tumor trainer with augmentation, ReduceLROnPlateau, and EarlyStopping."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        checkpoint_dir: Optional[Path] = None,
        visualize_every: int = 5,
        patience: int = 15,
        use_mixed_precision: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: UNet model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device to train on
            learning_rate: Initial learning rate
            checkpoint_dir: Directory to save checkpoints
            visualize_every: Visualize every N epochs
            patience: Early stopping patience
            use_mixed_precision: Use mixed precision training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('/content/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_every = visualize_every

        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Combined Dice + BCE loss
        self.criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        
        # ReduceLROnPlateau scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize Dice score
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        # Mixed precision training
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None

        # Metrics tracking
        self.best_dice = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []

        logger.info(
            f"UNetTumor trainer initialized: lr={learning_rate}, device={device}, "
            f"loss=DiceBCE, scheduler=ReduceLROnPlateau, early_stopping_patience={patience}, "
            f"mixed_precision={self.use_mixed_precision}",
            extra={'image_id': None, 'path': str(self.checkpoint_dir), 'stage': 'train_init'}
        )

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_acc = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
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

    def validate_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_acc = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:02d} [Val]  ")
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

    def save_checkpoint(self, epoch: int, val_dice: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'val_dice': val_dice
        }

        # Save last checkpoint
        last_path = self.checkpoint_dir / 'unet_tumor_last.pth'
        torch.save(checkpoint, last_path)
        logger.info(f"Checkpoint saved: {last_path}", extra={
            'image_id': None,
            'path': str(last_path),
            'stage': 'checkpoint_save'
        })

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'unet_tumor_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Best checkpoint saved: {best_path} (Dice: {self.best_dice:.4f})", extra={
                'image_id': None,
                'path': str(best_path),
                'stage': 'checkpoint_save'
            })

    def plot_training_curves(self):
        """Plot training curves."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Dice curves
        ax2.plot(self.train_dices, label='Train Dice', marker='o')
        ax2.plot(self.val_dices, label='Val Dice', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Training and Validation Dice Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training curves saved to: {plot_path}", extra={
            'image_id': None, 'path': str(plot_path), 'stage': 'visualization'
        })
        plt.close()

    def train(self, num_epochs: int):
        """Train for multiple epochs."""
        logger.info(
            f"Starting UNet Tumor training for {num_epochs} epochs",
            extra={'image_id': None, 'path': None, 'stage': 'train_start'}
        )

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_dice, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_dice, val_acc = self.validate_epoch(epoch)

            epoch_duration = time.time() - epoch_start

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dices.append(train_dice)
            self.val_dices.append(val_dice)

            # Update learning rate based on validation Dice
            self.scheduler.step(val_dice)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log epoch metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f}, "
                f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}, "
                f"lr={current_lr:.6f}, time={epoch_duration:.2f}s",
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

            self.save_checkpoint(epoch, val_dice, is_best=is_best)
            
            # Plot curves every 10 epochs
            if epoch % 10 == 0:
                self.plot_training_curves()

            # Check early stopping
            if self.early_stopping(val_dice):
                logger.info(f"Early stopping triggered at epoch {epoch}", extra={
                    'image_id': None,
                    'path': None,
                    'stage': 'early_stopping'
                })
                break

        # Final plots
        self.plot_training_curves()

        logger.info(f"Training completed. Best Dice: {self.best_dice:.4f}", extra={
            'image_id': None,
            'path': None,
            'stage': 'train_complete'
        })
