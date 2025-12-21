"""
Google Colab Training Script for UNet Tumor Segmentation (PNG Dataset)
======================================================================

This script is optimized for training the UNet Tumor model on Google Colab with:
- 15GB GPU RAM (T4)
- 12GB System RAM
- PNG-based dataset with images/ and masks/ folders

Dataset Structure:
==================
/content/UNet_Tumor_Dataset/
├── images/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── masks/
    ├── image001.png
    ├── image002.png
    └── ...

Features:
=========
- Data augmentation to prevent overfitting (~3x effective dataset size)
- ReduceLROnPlateau for adaptive learning rate
- EarlyStopping to prevent overfitting
- Mixed precision training for faster convergence
- Gradient clipping for stability
- Combined Dice + BCE loss
- Progress visualization and metrics logging
- Auto-saves best model checkpoint

Usage in Colab:
===============

1. Mount Google Drive and upload your UNet_Tumor_Dataset folder:
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
   !pip install opencv-python matplotlib tqdm pydantic pydantic-settings pillow
   ```

4. Ensure your dataset is accessible:
   - Either copy to /content/UNet_Tumor_Dataset
   - Or create symlink: !ln -s /content/drive/MyDrive/UNet_Tumor_Dataset /content/UNet_Tumor_Dataset

5. Run training:
   ```python
   !python train_unet_tumor_colab.py
   ```
   
   Or with custom parameters:
   ```python
   !python train_unet_tumor_colab.py --epochs 100 --batch_size 16 --lr 1e-4 --patience 15
   ```

Expected Dataset Size After Augmentation:
=========================================
With 3000 base images and augmentation including:
- Random horizontal flip (50%)
- Random vertical flip (50%)
- Random 90-degree rotations (4 variants)
- Random brightness/contrast adjustments
- Random Gaussian noise

Effective dataset size per epoch: ~3000 * 3 = ~9000 augmented samples
(On-the-fly augmentation, each epoch sees different variations)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Tuple

# Add backend to path - handle both root and subdirectory execution
script_dir = Path(__file__).parent
backend_path = script_dir / "backend"
if not backend_path.exists():
    backend_path = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_path))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import from our modules
from app.models.unet_tumor.model import get_unet_tumor_model
from app.models.unet_tumor.datamodule import UNetTumorDataset, create_unet_tumor_dataloaders
from app.models.unet_tumor.train_unet_tumor import UNetTumorTrainer


def setup_colab_environment(dataset_path: str):
    """Setup Google Colab environment."""
    print("=" * 80)
    print("GOOGLE COLAB UNET TUMOR TRAINING SETUP")
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
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found at {dataset_path}")
        print("\nPlease ensure your dataset is available at the specified path")
        print("You can either:")
        print("  1. Copy dataset to /content/UNet_Tumor_Dataset")
        print("  2. Create symlink: !ln -s /content/drive/MyDrive/UNet_Tumor_Dataset /content/UNet_Tumor_Dataset")
        return False
    
    # Check images and masks directories
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'
    
    if not images_dir.exists() or not masks_dir.exists():
        print(f"\n❌ Dataset structure invalid!")
        print(f"   Expected: {dataset_path}/images/ and {dataset_path}/masks/")
        return False
    
    # Count PNG files
    image_files = list(images_dir.glob('*.png'))
    mask_files = list(masks_dir.glob('*.png'))
    
    print(f"\n📊 Dataset structure:")
    print(f"   Images folder: {len(image_files)} PNG files")
    print(f"   Masks folder: {len(mask_files)} PNG files")
    
    # Check for matching files
    image_names = set(f.name for f in image_files)
    mask_names = set(f.name for f in mask_files)
    matching = len(image_names & mask_names)
    
    print(f"   Matching pairs: {matching}")
    
    if matching == 0:
        print("\n❌ No matching image-mask pairs found!")
        return False
    
    # Show augmentation info
    print(f"\n📈 With data augmentation:")
    print(f"   Base dataset size: {matching} images")
    print(f"   Augmentation factor: ~3x (on-the-fly)")
    print(f"   Effective samples per epoch: ~{matching * 3}")
    
    print("\n" + "=" * 80)
    return True


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    """Main training function for Google Colab."""
    parser = argparse.ArgumentParser(description="Train UNet Tumor on Google Colab")
    parser.add_argument("--dataset_path", type=str, default="/content/UNet_Tumor_Dataset",
                       help="Path to dataset folder containing images/ and masks/")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (16 works well for 15GB GPU)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Image size (will resize to image_size x image_size)")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of data loading workers")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Training set split ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate for regularization")
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_colab_environment(args.dataset_path):
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
    print(f"   Early stopping patience: {args.patience}")
    print(f"   Dropout rate: {args.dropout}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("\n📁 Loading dataset...")
    
    train_loader, val_loader = create_unet_tumor_dataloaders(
        root_dir=Path(args.dataset_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        image_size=(args.image_size, args.image_size),
        seed=args.seed
    )
    
    print(f"✅ Dataloaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🧠 Creating UNet Tumor model...")
    print(f"   Device: {device}")
    print(f"   Input channels: 3 (RGB)")
    print(f"   Output channels: 1 (binary segmentation)")
    print(f"   Dropout rate: {args.dropout}")
    
    model = get_unet_tumor_model(
        in_channels=3,
        out_channels=1,
        features=(32, 64, 128, 256, 512),
        dropout_rate=args.dropout
    )
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = UNetTumorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir=checkpoint_dir,
        visualize_every=5,
        patience=args.patience,
        use_mixed_precision=True
    )
    
    # Train
    print(f"\n💡 Tips:")
    print(f"   - Training visualizations saved to: {checkpoint_dir}")
    print(f"   - Best model will be saved as: {checkpoint_dir}/unet_tumor_best.pth")
    print(f"   - Monitor GPU usage with: !nvidia-smi")
    print(f"   - Training may take 2-4 hours for {args.epochs} epochs")
    print()
    
    try:
        trainer.train(args.epochs)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED!")
        print("=" * 80)
        print(f"\n📊 Final Results:")
        print(f"   Best Dice Score: {trainer.best_dice:.4f}")
        print(f"   Best model saved to: {checkpoint_dir}/unet_tumor_best.pth")
        print(f"   Last model saved to: {checkpoint_dir}/unet_tumor_last.pth")
        print(f"\n💾 To download the model:")
        print(f"   from google.colab import files")
        print(f"   files.download('{checkpoint_dir}/unet_tumor_best.pth')")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print(f"   Last checkpoint saved to: {checkpoint_dir}/unet_tumor_last.pth")
    except Exception as e:
        print(f"\n\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
