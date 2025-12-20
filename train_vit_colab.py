"""
Google Colab Training Script for ViT Brain Tumor Classification
================================================================

This script is optimized for training the ViT model on Google Colab with:
- 15GB GPU RAM (T4)
- 12GB System RAM
- Custom brain tumor classification dataset with 4 classes:
  - notumor
  - glioma
  - meningioma
  - pituitary

Dataset Structure:
==================
/content/dataset/Vit_Dataset/
├── notumor/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── glioma/
│   ├── image1.jpg
│   └── ...
├── meningioma/
│   └── ...
└── pituitary/
    └── ...

Usage in Colab:
===============

1. Mount Google Drive and upload your Vit_Dataset folder:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Clone the repository or upload code:
   ```bash
   !git clone https://github.com/H0NEYP0T-466/BTSC-UNet-ViT.git
   %cd BTSC-UNet-ViT
   ```

3. Install dependencies:
   ```bash
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   !pip install timm pillow opencv-python matplotlib tqdm pydantic pydantic-settings scikit-learn
   ```

4. Ensure your dataset is accessible:
   - Either copy to /content/dataset/Vit_Dataset
   - Or create symlink: !ln -s /content/drive/MyDrive/Vit_Dataset /content/dataset/Vit_Dataset

5. Run training:
   ```python
   !python train_vit_colab.py
   ```
   
   Or with custom parameters:
   ```python
   !python train_vit_colab.py --epochs 50 --batch_size 32 --lr 1e-4
   ```

Features:
=========
- Data augmentation with albumentations/torchvision
- ReduceLROnPlateau for adaptive learning rate
- EarlyStopping to prevent overfitting
- Mixed precision training for faster convergence
- Gradient clipping for stability
- Class-balanced sampling for imbalanced datasets
- Progress visualization and metrics logging
- Auto-saves best model checkpoint
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Tuple, Optional

# Add backend to path - handle both root and subdirectory execution
script_dir = Path(__file__).parent
backend_path = script_dir / "backend"
if not backend_path.exists():
    # If backend not found relative to script, try absolute path
    backend_path = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_path))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import from our modules
from app.models.vit.model import get_vit_model
from app.models.vit.datamodule import ViTDataset, get_vit_transforms


def setup_colab_environment():
    """Setup Google Colab environment."""
    print("=" * 80)
    print("GOOGLE COLAB VIT TRAINING SETUP")
    print("=" * 80)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("⚠️  WARNING: No GPU detected! Training will be very slow.")
        print("   Enable GPU: Runtime → Change runtime type → GPU")
    
    # Check dataset
    dataset_path = Path("/content/dataset/Vit_Dataset")
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found at {dataset_path}")
        print("\nPlease ensure your dataset is available at /content/dataset/Vit_Dataset")
        print("You can either:")
        print("  1. Copy dataset to /content/dataset/Vit_Dataset")
        print("  2. Create symlink: !ln -s /content/drive/MyDrive/Vit_Dataset /content/dataset/Vit_Dataset")
        return False
    
    # Count images per class
    class_names = ["notumor", "glioma", "meningioma", "pituitary"]
    total_images = 0
    print(f"\n📊 Dataset structure:")
    for class_name in class_names:
        class_dir = dataset_path / class_name
        if class_dir.exists():
            count = len(list(class_dir.rglob("*.jpg"))) + \
                   len(list(class_dir.rglob("*.jpeg"))) + \
                   len(list(class_dir.rglob("*.png")))
            total_images += count
            print(f"   {class_name}: {count} images")
        else:
            print(f"   ⚠️  {class_name}: folder not found")
    
    print(f"\n✅ Total images found: {total_images}")
    
    print("\n" + "=" * 80)
    return True


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0.0, verbose=True):
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
        
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"   EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class ViTTrainerColab:
    """ViT trainer optimized for Google Colab."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        learning_rate: float,
        checkpoint_dir: Path,
        num_classes: int = 4,
        patience: int = 10,
        use_mixed_precision: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.num_classes = num_classes
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,  # L2 regularization to prevent overfitting
            betas=(0.9, 0.999)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler - ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        # Mixed precision training
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Metrics tracking
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        print(f"\n🏋️  Trainer Configuration:")
        print(f"   Device: {device}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Optimizer: AdamW (weight_decay=0.01)")
        print(f"   Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
        print(f"   Early stopping: patience={patience}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        print(f"   Gradient clipping: enabled (max_norm=1.0)")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                # Gradient clipping to prevent exploding gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:02d} [Val]  ")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'vit_last.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'vit_best.pth'
            torch.save(checkpoint, best_path)
            print(f"   💾 Best model saved: val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Acc', marker='o')
        ax2.plot(self.val_accs, label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n   📊 Training curves saved to: {plot_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = self.checkpoint_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"   📊 Confusion matrix saved to: {cm_path}")
        plt.close()
    
    def train(self, num_epochs: int):
        """Train for multiple epochs."""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, y_pred, y_true = self.validate_epoch(epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            epoch_duration = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\n{'─' * 80}")
            print(f"Epoch {epoch}/{num_epochs} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"   Time: {epoch_duration:.2f}s")
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_acc > self.best_accuracy
            if is_best:
                self.best_accuracy = val_acc
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, val_acc, is_best=is_best)
            
            # Early stopping check
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break
            
            # Plot curves every 10 epochs
            if epoch % 10 == 0:
                self.plot_training_curves()
        
        total_duration = time.time() - start_time
        
        # Final plots
        self.plot_training_curves()
        
        # Get final predictions for confusion matrix
        _, _, y_pred, y_true = self.validate_epoch(num_epochs)
        class_names = ["notumor", "glioma", "meningioma", "pituitary"]
        self.plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Print classification report
        print("\n" + "=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80)
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED!")
        print("=" * 80)
        print(f"   Total training time: {total_duration / 60:.2f} minutes")
        print(f"   Best validation accuracy: {self.best_accuracy:.2f}%")
        print(f"   Best validation loss: {self.best_loss:.4f}")
        print(f"   Best model saved to: {self.checkpoint_dir}/vit_best.pth")


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    """Main training function for Google Colab."""
    parser = argparse.ArgumentParser(description="Train ViT on Google Colab")
    parser.add_argument("--dataset_path", type=str, default="/content/dataset/Vit_Dataset",
                       help="Path to dataset folder containing class folders")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (32 works well for 15GB GPU)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size (ViT default is 224x224)")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of data loading workers")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Training set split ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--no_augment", action="store_true",
                       help="Disable data augmentation")
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_colab_environment():
        return
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"\n🚀 Starting training with parameters:")
    print(f"   Dataset: {args.dataset_path}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Image size: {args.image_size}x{args.image_size}")
    print(f"   Train split: {args.train_split}")
    print(f"   Data augmentation: {not args.no_augment}")
    print(f"   Early stopping patience: {args.patience}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("\n📁 Loading dataset...")
    class_names = ["notumor", "glioma", "meningioma", "pituitary"]
    
    # Get transforms
    train_transform, val_transform = get_vit_transforms(
        image_size=args.image_size,
        augment=not args.no_augment
    )
    
    # Create full dataset to count samples
    full_dataset = ViTDataset(
        root_dir=Path(args.dataset_path),
        class_names=class_names,
        transform=None,  # We'll apply transforms after split
        image_size=args.image_size
    )
    
    if len(full_dataset) == 0:
        print("❌ ERROR: No data loaded! Check dataset path.")
        return
    
    print(f"✅ Total dataset loaded: {len(full_dataset)} samples")
    
    # Split dataset
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use indices to create subsets
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate datasets with appropriate transforms
    train_dataset_with_aug = ViTDataset(
        root_dir=Path(args.dataset_path),
        class_names=class_names,
        transform=train_transform,
        image_size=args.image_size
    )
    
    val_dataset_no_aug = ViTDataset(
        root_dir=Path(args.dataset_path),
        class_names=class_names,
        transform=val_transform,
        image_size=args.image_size
    )
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(train_dataset_with_aug, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
    
    # Calculate class weights for balanced sampling (optional, helps with imbalanced datasets)
    class_counts = np.bincount([full_dataset.labels[i] for i in train_indices])
    
    # Check for empty classes and handle division by zero
    if np.any(class_counts == 0):
        print(f"   ⚠️  Warning: Some classes have zero samples in training split")
        print(f"   Class counts: {class_counts}")
        # Use all samples without weighted sampling if any class is empty
        sample_weights = [1.0 for _ in train_indices]
    else:
        # Calculate weights normally
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[full_dataset.labels[i]] for i in train_indices]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {val_size}")
    print(f"   Class distribution (train): {class_counts}")
    print(f"\n📊 BASE DATASET SIZE: {len(full_dataset)} images")
    print(f"   (Note: Augmentation is applied on-the-fly during training, not pre-computed)")
    print(f"   Each training epoch will see augmented variations of these images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use weighted sampler for balanced batches
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✅ Dataloaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🧠 Creating ViT model...")
    print(f"   Device: {device}")
    print(f"   Model: vit_base_patch16_224")
    print(f"   Number of classes: {len(class_names)}")
    
    model = get_vit_model(num_classes=len(class_names), pretrained=True)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = ViTTrainerColab(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir=checkpoint_dir,
        num_classes=len(class_names),
        patience=args.patience,
        use_mixed_precision=True
    )
    
    # Train
    print(f"\n💡 Tips:")
    print(f"   - Training visualizations saved to: {checkpoint_dir}")
    print(f"   - Best model will be saved as: {checkpoint_dir}/vit_best.pth")
    print(f"   - Monitor GPU usage with: !nvidia-smi")
    print(f"   - Training may take 2-4 hours for {args.epochs} epochs")
    print()
    
    try:
        trainer.train(args.epochs)
        
        print(f"\n💾 To download the model:")
        print(f"   from google.colab import files")
        print(f"   files.download('{checkpoint_dir}/vit_best.pth')")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print(f"   Last checkpoint saved to: {checkpoint_dir}/vit_last.pth")
    except Exception as e:
        print(f"\n\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
