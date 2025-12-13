"""
Google Colab Training Script for UNet Brain Tumor Segmentation
================================================================

This script is optimized for training the UNet model on Google Colab with:
- 15GB GPU RAM
- 12GB System RAM
- Custom BraTS .h5 dataset with 4-channel images and binary masks

Usage in Colab:
===============

1. Mount Google Drive and upload your UNet_Dataset folder:
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
   !pip install h5py opencv-python matplotlib tqdm pydantic pydantic-settings
   ```

4. Ensure your dataset is accessible:
   - Either copy to /content/UNet_Dataset
   - Or create symlink: !ln -s /content/drive/MyDrive/UNet_Dataset /content/UNet_Dataset

5. Run training:
   ```python
   !python train_unet_colab.py
   ```
   
   Or with custom parameters:
   ```python
   !python train_unet_colab.py --epochs 100 --batch_size 16 --lr 1e-4
   ```

Features:
=========
- Handles extreme class imbalance (0.17% tumor) with Dice+BCE loss
- Proper 4-channel image loading and normalization
- Binary mask extraction from 3-channel masks
- Enhanced visualization for tiny tumors
- Auto-saves best model checkpoint
- Progress visualization every 5 epochs
"""

import os
import sys
import argparse
from pathlib import Path

# Add backend to path - handle both root and subdirectory execution
script_dir = Path(__file__).parent
backend_path = script_dir / "backend"
if not backend_path.exists():
    # If backend not found relative to script, try absolute path
    backend_path = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_path))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

# Import from our modules
from app.models.unet.model import get_unet_model
from app.models.unet.datamodule import UNetDataset
from app.models.unet.train_unet import UNetTrainer


def setup_colab_environment():
    """Setup Google Colab environment."""
    print("=" * 80)
    print("GOOGLE COLAB UNET TRAINING SETUP")
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
    dataset_path = Path("/content/UNet_Dataset")
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found at {dataset_path}")
        print("\nPlease ensure your dataset is available at /content/UNet_Dataset")
        print("You can either:")
        print("  1. Copy dataset to /content/UNet_Dataset")
        print("  2. Create symlink: !ln -s /content/drive/MyDrive/UNet_Dataset /content/UNet_Dataset")
        return False
    
    # Count .h5 files
    h5_files = list(dataset_path.glob("*.h5"))
    print(f"\n✅ Dataset found: {len(h5_files)} .h5 files")
    
    # Check a sample file
    if h5_files:
        import h5py
        sample_file = h5_files[0]
        print(f"\n📊 Sample file analysis: {sample_file.name}")
        with h5py.File(sample_file, 'r') as f:
            print(f"   Keys: {list(f.keys())}")
            if 'image' in f:
                print(f"   Image shape: {f['image'].shape}")
                print(f"   Image dtype: {f['image'].dtype}")
            if 'mask' in f:
                print(f"   Mask shape: {f['mask'].shape}")
                print(f"   Mask dtype: {f['mask'].dtype}")
    
    print("\n" + "=" * 80)
    return True


def main():
    """Main training function for Google Colab."""
    parser = argparse.ArgumentParser(description="Train UNet on Google Colab")
    parser.add_argument("--dataset_path", type=str, default="/content/UNet_Dataset",
                       help="Path to dataset folder containing .h5 files")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (16 works well for 15GB GPU)")
    parser.add_argument("--epochs", type=int, default=50,
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
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_colab_environment():
        return
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"\n🚀 Starting training with parameters:")
    print(f"   Dataset: {args.dataset_path}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Image size: {args.image_size}x{args.image_size}")
    print(f"   Train split: {args.train_split}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print("\n📁 Loading dataset...")
    full_dataset = UNetDataset(
        root_dir=Path(args.dataset_path),
        transform=None,
        image_size=(args.image_size, args.image_size)
    )
    
    if len(full_dataset) == 0:
        print("❌ ERROR: No data loaded! Check dataset path.")
        return
    
    print(f"✅ Dataset loaded: {len(full_dataset)} samples")
    
    # Split dataset
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    print(f"\n🧠 Creating UNet model...")
    print(f"   Device: {device}")
    print(f"   Input channels: 4 (BraTS modalities)")
    print(f"   Output channels: 1 (binary segmentation)")
    
    model = get_unet_model(
        in_channels=4,
        out_channels=1,
        features=(16, 32, 64, 128, 256)
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {num_trainable:,}")
    
    # Create trainer
    print(f"\n🏋️  Initializing trainer...")
    print(f"   Loss function: Dice + BCE (for extreme class imbalance)")
    print(f"   Optimizer: Adam with weight decay")
    print(f"   Scheduler: ReduceLROnPlateau")
    
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir=checkpoint_dir,
        visualize_every=5
    )
    
    # Train
    print(f"\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"\n💡 Tips:")
    print(f"   - Training visualizations saved to: {checkpoint_dir}")
    print(f"   - Best model will be saved as: {checkpoint_dir}/unet_best.pth")
    print(f"   - Monitor GPU usage with: !nvidia-smi")
    print(f"   - Training may take 2-4 hours for {args.epochs} epochs")
    print()
    
    try:
        trainer.train(args.epochs)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📊 Final Results:")
        print(f"   Best Dice Score: {trainer.best_dice:.4f}")
        print(f"   Best model saved to: {checkpoint_dir}/unet_best.pth")
        print(f"   Last model saved to: {checkpoint_dir}/unet_last.pth")
        print(f"\n💾 To download the model:")
        print(f"   from google.colab import files")
        print(f"   files.download('{checkpoint_dir}/unet_best.pth')")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print(f"   Last checkpoint saved to: {checkpoint_dir}/unet_last.pth")
    except Exception as e:
        print(f"\n\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
