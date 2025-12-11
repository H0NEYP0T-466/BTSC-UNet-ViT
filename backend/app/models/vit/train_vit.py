"""
ViT training script for fine-tuning on segmented brain tumor images.
"""
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from app.models.vit.model import get_vit_model
from app.config import settings
from app.utils.logger import get_logger
from app.utils.metrics import calculate_classification_metrics

logger = get_logger(__name__)


class ViTTrainer:
    """ViT trainer for brain tumor classification."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        checkpoint_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir or settings.CHECKPOINTS_VIT
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_accuracy = 0.0
        
        logger.info(f"ViT trainer initialized: lr={learning_rate}, device={device}", extra={
            'image_id': None,
            'path': str(self.checkpoint_dir),
            'stage': 'train_init'
        })
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log batch progress periodically
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}",
                    extra={'image_id': None, 'path': None, 'stage': 'train_batch'}
                )
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate_epoch(self, epoch: int) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        import numpy as np
        metrics = calculate_classification_metrics(
            np.array(all_preds),
            np.array(all_labels),
            settings.VIT_NUM_CLASSES
        )
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'metrics': metrics
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'vit_last.pth'
        torch.save(checkpoint, last_path)
        logger.info(f"Checkpoint saved: {last_path}", extra={
            'image_id': None,
            'path': str(last_path),
            'stage': 'checkpoint_save'
        })
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / settings.VIT_CHECKPOINT_NAME
            torch.save(checkpoint, best_path)
            logger.info(f"Best checkpoint saved: {best_path}", extra={
                'image_id': None,
                'path': str(best_path),
                'stage': 'checkpoint_save'
            })
    
    def train(self, num_epochs: int):
        """Train for multiple epochs with manual logging."""
        logger.info(f"Starting ViT training for {num_epochs} epochs", extra={
            'image_id': None,
            'path': None,
            'stage': 'train_start'
        })
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            epoch_duration = time.time() - epoch_start
            
            # Log epoch metrics (manual verbose logging)
            logger.info(
                f"Epoch {epoch}/{num_epochs} completed: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"accuracy={val_metrics['accuracy']:.4f}, "
                f"f1_macro={val_metrics['f1_macro']:.4f}, "
                f"precision={val_metrics['precision']:.4f}, "
                f"recall={val_metrics['recall']:.4f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}, "
                f"duration={epoch_duration:.2f}s",
                extra={
                    'image_id': None,
                    'path': None,
                    'stage': 'train_epoch'
                }
            )
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['accuracy']
                logger.info(f"New best accuracy: {self.best_accuracy:.4f}", extra={
                    'image_id': None,
                    'path': None,
                    'stage': 'train_epoch'
                })
            
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
        
        logger.info(f"Training completed. Best accuracy: {self.best_accuracy:.4f}", extra={
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
    if not settings.SEGMENTED_DATASET_ROOT.exists():
        logger.error(
            f"Dataset not found at {settings.SEGMENTED_DATASET_ROOT}. "
            f"Please ensure the dataset is available.",
            extra={'image_id': None, 'path': str(settings.SEGMENTED_DATASET_ROOT), 'stage': 'train_init'}
        )
        raise FileNotFoundError(f"Dataset not found: {settings.SEGMENTED_DATASET_ROOT}")
    
    logger.info(f"Loading dataset from {settings.SEGMENTED_DATASET_ROOT}", extra={
        'image_id': None,
        'path': str(settings.SEGMENTED_DATASET_ROOT),
        'stage': 'train_init'
    })
    
    # Import datamodule
    from app.models.vit.datamodule import create_vit_dataloaders
    
    # Create dataloaders
    train_loader, val_loader = create_vit_dataloaders(
        root_dir=settings.SEGMENTED_DATASET_ROOT,
        batch_size=settings.BATCH_SIZE,
        num_workers=settings.NUM_WORKERS,
        train_split=0.8,
        image_size=settings.VIT_IMAGE_SIZE,
        augment=True
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
    
    model = get_vit_model()
    
    # Optionally freeze backbone initially
    # model.freeze_backbone(freeze=True)
    
    logger.info("Model created successfully", extra={
        'image_id': None,
        'path': None,
        'stage': 'train_init'
    })
    
    # Create trainer and train
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=settings.LEARNING_RATE,
        checkpoint_dir=settings.CHECKPOINTS_VIT
    )
    
    trainer.train(settings.NUM_EPOCHS)


if __name__ == "__main__":
    from app.logging_config import setup_logging
    setup_logging(settings.LOG_LEVEL)
    main()
