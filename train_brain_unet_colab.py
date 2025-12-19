"""
Google Colab Training Script for Brain UNet (NFBS Dataset)
===========================================================

This script is optimized for training the Brain UNet model on Google Colab with:
- 15GB GPU RAM
- NFBS dataset with T1w MRI scans and brain masks
- Single-channel input (T1w) and binary mask output

Usage in Colab:
===============

1. Mount Google Drive and upload your NFBS_Dataset folder:
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
   !pip install nibabel opencv-python matplotlib tqdm pydantic pydantic-settings numpy scikit-image
   ```

4. Ensure your dataset is accessible:
   - Either copy to /content/NFBS_Dataset
   - Or create symlink: !ln -s /content/drive/MyDrive/NFBS_Dataset /content/NFBS_Dataset

5. Run training:
   ```python
   !python train_brain_unet_colab.py
   ```
   
   Or with custom parameters:
   ```python
   !python train_brain_unet_colab.py --epochs 100 --batch_size 32 --lr 1e-4
   ```

Features:
=========
- Loads NFBS dataset with T1w images and brain masks
- Handles 3D volumes by extracting 2D slices
- Binary brain segmentation (brain vs background)
- Dice + BCE loss for balanced training
- Auto-saves best model checkpoint
- Progress visualization every 5 epochs
"""

import os
import sys
import argparse
from pathlib import Path

# Add backend to path
script_dir = Path(__file__).parent
backend_path = script_dir / "backend"
if not backend_path.exists():
    backend_path = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_path))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

# Import from our modules
from app.models.brain_unet.model import get_brain_unet_model
from app.models.brain_unet.datamodule import create_brain_unet_dataloaders
from app.models.brain_unet.train_unet import BrainUNetTrainer


def setup_colab_environment(dataset_path: Path):
    """Setup Google Colab environment."""
    print("=" * 80)
    print("GOOGLE COLAB BRAIN UNET TRAINING SETUP")
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
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found at {dataset_path}")
        print("\nPlease ensure your dataset is available at the specified path")
        print("You can either:")
        print("  1. Copy dataset to /content/NFBS_Dataset")
        print("  2. Create symlink: !ln -s /content/drive/MyDrive/NFBS_Dataset /content/NFBS_Dataset")
        return False
    
    # Count subject folders
    subjects = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"\n✅ Dataset found: {len(subjects)} subjects")
    
    # Check a sample subject
    if subjects:
        sample_subject = subjects[0]
        print(f"\n📊 Sample subject analysis: {sample_subject.name}")
        t1_files = list(sample_subject.glob("*_T1w.nii.gz"))
        mask_files = list(sample_subject.glob("*_brainmask.nii.gz"))
        print(f"   T1w files: {len(t1_files)}")
        print(f"   Mask files: {len(mask_files)}")
        
        if t1_files and mask_files:
            try:
                import nibabel as nib
                t1_img = nib.load(t1_files[0])
                mask_img = nib.load(mask_files[0])
                print(f"   T1w shape: {t1_img.shape}")
                print(f"   Mask shape: {mask_img.shape}")
                print(f"   Voxel size: {t1_img.header.get_zooms()}")
            except Exception as e:
                print(f"   Error loading sample: {e}")
    
    print("\n" + "=" * 80)
    return True


def main():
    """Main training function for Google Colab."""
    parser = argparse.ArgumentParser(description="Train Brain UNet on Google Colab")
    parser.add_argument("--dataset_path", type=str, default="/content/NFBS_Dataset",
                       help="Path to NFBS dataset folder containing subject directories")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/checkpoints/brain_unet",
                       help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (32 works well for 15GB GPU with 256x256 images)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--image_size", type=int, default=256,
                       help="Image size (will resize to image_size x image_size)")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of data loading workers (0 recommended with caching)")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Training set split ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--slice_start", type=int, default=None,
                       help="Start slice index (optional)")
    parser.add_argument("--slice_end", type=int, default=None,
                       help="End slice index (optional)")
    parser.add_argument("--cache_in_memory", type=bool, default=True,
                       help="Pre-load dataset into memory (faster training, requires more RAM)")
    parser.add_argument("--use_amp", type=bool, default=True,
                       help="Use Automatic Mixed Precision for faster training")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    # Setup environment
    if not setup_colab_environment(dataset_path):
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
    print(f"   Cache in memory: {args.cache_in_memory}")
    print(f"   Use AMP: {args.use_amp}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("\n📁 Loading dataset...")
    
    slice_range = None
    if args.slice_start is not None and args.slice_end is not None:
        slice_range = (args.slice_start, args.slice_end)
        print(f"   Using slice range: [{args.slice_start}, {args.slice_end})")
    
    try:
        train_loader, val_loader = create_brain_unet_dataloaders(
            root_dir=dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_split=args.train_split,
            image_size=(args.image_size, args.image_size),
            transform=None,
            slice_range=slice_range,
            cache_in_memory=args.cache_in_memory
        )
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"✅ Dataloaders created:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🧠 Creating Brain UNet model...")
    print(f"   Device: {device}")
    print(f"   Input channels: 1 (T1w MRI)")
    print(f"   Output channels: 1 (binary brain mask)")
    
    model = get_brain_unet_model(
        in_channels=1,
        out_channels=1,
        features=(32, 64, 128, 256, 512)
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {num_trainable:,}")
    
    # Create trainer
    print(f"\n🏋️  Initializing trainer...")
    print(f"   Loss function: Dice + BCE (for brain segmentation)")
    print(f"   Optimizer: Adam with weight decay")
    print(f"   Scheduler: ReduceLROnPlateau")
    print(f"   Mixed Precision: {args.use_amp}")
    
    trainer = BrainUNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir=checkpoint_dir,
        visualize_every=5,
        use_amp=args.use_amp
    )
    
    # Train
    print(f"\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"\n💡 Tips:")
    print(f"   - Training visualizations saved to: {checkpoint_dir}")
    print(f"   - Best model will be saved as: {checkpoint_dir}/brain_unet_best.pth")
    print(f"   - Monitor GPU usage with: !nvidia-smi")
    print(f"   - With optimizations, training should be much faster (~5-10 min/epoch)")
    print()
    
    try:
        trainer.train(args.epochs)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📊 Final Results:")
        print(f"   Best Dice Score: {trainer.best_dice:.4f}")
        print(f"   Best model saved to: {checkpoint_dir}/brain_unet_best.pth")
        print(f"   Last model saved to: {checkpoint_dir}/brain_unet_last.pth")
        print(f"\n💾 To download the model:")
        print(f"   from google.colab import files")
        print(f"   files.download('{checkpoint_dir}/brain_unet_best.pth')")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print(f"   Last checkpoint saved to: {checkpoint_dir}/brain_unet_last.pth")
    except Exception as e:
        print(f"\n\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
