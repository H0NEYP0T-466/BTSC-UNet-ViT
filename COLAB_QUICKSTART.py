"""
Quick Start Guide for ViT Training on Google Colab
===================================================

Copy and paste this into Google Colab cells to start training.
"""

# ============================================================================
# CELL 1: Mount Google Drive
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# CELL 2: Clone Repository and Install Dependencies
# ============================================================================
# Clone repository
!git clone https://github.com/H0NEYP0T-466/BTSC-UNet-ViT.git
%cd BTSC-UNet-ViT

# Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q timm pillow opencv-python matplotlib tqdm pydantic pydantic-settings scikit-learn seaborn

print("✅ Setup complete!")

# ============================================================================
# CELL 3: Link Dataset (Choose Option A or B)
# ============================================================================
# Option A: Create symlink (recommended - faster, saves space)
!mkdir -p /content/dataset
!ln -s /content/drive/MyDrive/Vit_Dataset /content/dataset/Vit_Dataset

# Option B: Copy dataset (slower but more stable)
# !mkdir -p /content/dataset
# !cp -r /content/drive/MyDrive/Vit_Dataset /content/dataset/

# Verify dataset
!ls -l /content/dataset/Vit_Dataset/

# ============================================================================
# CELL 4: Check Dataset Structure
# ============================================================================
import os
from pathlib import Path

dataset_path = Path("/content/dataset/Vit_Dataset")
class_names = ["notumor", "glioma", "meningioma", "pituitary"]

print("Dataset Structure:")
print("=" * 50)

total_images = 0
for class_name in class_names:
    class_dir = dataset_path / class_name
    if class_dir.exists():
        count = len(list(class_dir.rglob("*.jpg"))) + \
               len(list(class_dir.rglob("*.jpeg"))) + \
               len(list(class_dir.rglob("*.png")))
        total_images += count
        print(f"{class_name:12s}: {count:6d} images")
    else:
        print(f"{class_name:12s}: ⚠️  NOT FOUND")

print("=" * 50)
print(f"{'TOTAL':12s}: {total_images:6d} images")

# ============================================================================
# CELL 5: Start Training (Basic)
# ============================================================================
!python train_vit_colab.py

# ============================================================================
# CELL 6: Start Training (Custom Parameters)
# ============================================================================
# Adjust parameters as needed
!python train_vit_colab.py \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --patience 10 \
  --train_split 0.8

# ============================================================================
# CELL 7: Monitor GPU Usage During Training (Optional)
# ============================================================================
# Run this in a separate cell while training is running
!nvidia-smi

# Or for continuous monitoring:
# !watch -n 1 nvidia-smi

# ============================================================================
# CELL 8: View Training Curves (After Training)
# ============================================================================
from IPython.display import Image, display

print("Training Curves:")
display(Image(filename='/content/checkpoints/training_curves.png'))

print("\nConfusion Matrix:")
display(Image(filename='/content/checkpoints/confusion_matrix.png'))

# ============================================================================
# CELL 9: Check Best Model Performance
# ============================================================================
import torch

checkpoint_path = '/content/checkpoints/vit_best.pth'
checkpoint = torch.load(checkpoint_path, weights_only=False)

print("Best Model Statistics:")
print("=" * 50)
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation Accuracy: {checkpoint['val_accuracy']:.2f}%")
print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
print(f"Best Accuracy: {checkpoint['best_accuracy']:.2f}%")
print("=" * 50)

# ============================================================================
# CELL 10: Download Trained Model
# ============================================================================
from google.colab import files

# Download best model
files.download('/content/checkpoints/vit_best.pth')

# Optional: Download last checkpoint
# files.download('/content/checkpoints/vit_last.pth')

# Optional: Download visualizations
# files.download('/content/checkpoints/training_curves.png')
# files.download('/content/checkpoints/confusion_matrix.png')

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Common Issues and Solutions:

1. OUT OF MEMORY ERROR:
   - Reduce batch size: --batch_size 16 or --batch_size 8
   - Reduce num_workers: --num_workers 1

2. DATASET NOT FOUND:
   - Verify path: !ls /content/dataset/Vit_Dataset
   - Check folder names match: notumor, glioma, meningioma, pituitary
   - Ensure symlink/copy was successful

3. SLOW TRAINING:
   - Check GPU is enabled: Runtime → Change runtime type → GPU
   - Verify GPU usage: !nvidia-smi
   - Check if using T4 GPU (should show in nvidia-smi output)

4. TRAINING STOPPED EARLY:
   - This is normal! EarlyStopping prevents overfitting
   - Check logs for "Early stopping triggered" message
   - Model already achieved best performance

5. LOW ACCURACY:
   - Increase epochs: --epochs 100
   - Try different learning rate: --lr 5e-5 or --lr 2e-4
   - Check dataset quality and labels
   - Increase patience: --patience 15

6. DISCONNECTED FROM COLAB:
   - Training state is lost if disconnected
   - Consider using last checkpoint: vit_last.pth
   - Use session timeout prevention tricks
"""

# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

"""
Optimization Tips:

1. FASTER TRAINING:
   - Use symlink instead of copying dataset (saves time)
   - Keep batch_size at 32 for T4 GPU
   - Mixed precision is enabled by default

2. BETTER ACCURACY:
   - Train for full 50 epochs
   - Use data augmentation (enabled by default)
   - Ensure balanced dataset (check class distribution)
   - Use appropriate learning rate (1e-4 is good default)

3. PREVENT OVERFITTING:
   - Use early stopping (patience=10)
   - Enable augmentation (--no_augment flag disabled)
   - Weight decay is applied (0.01)
   - Monitor train vs val loss curves

4. RESOURCE MANAGEMENT:
   - Close other Colab notebooks
   - Clear outputs regularly
   - Don't run other GPU-intensive tasks
   - Use "RAM disk" option if available

5. CHECKPOINTING:
   - Models saved every epoch (vit_last.pth)
   - Best model saved when val accuracy improves
   - Download frequently to avoid data loss
"""

# ============================================================================
# INTEGRATION WITH BACKEND
# ============================================================================

"""
After Training:

1. Download the trained model (vit_best.pth)

2. Copy to backend:
   - Local: backend/app/resources/checkpoints/vit/vit_best.pth
   - Colab: /content/resources/checkpoints/vit/vit_best.pth

3. The model will be automatically loaded by the pipeline

4. Test the pipeline:
   - Pipeline: preprocessing → ViT classification → conditional segmentation
   - If "notumor": segmentation is skipped (faster)
   - If tumor: segmentation runs (full analysis)

5. Monitor performance:
   - Check classification accuracy
   - Verify segmentation is skipped for healthy scans
   - Monitor inference time improvements
"""
