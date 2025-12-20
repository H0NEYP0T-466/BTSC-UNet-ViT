# ViT Training on Google Colab

This document provides instructions for training the Vision Transformer (ViT) model for brain tumor classification on Google Colab.

## Overview

The ViT model classifies brain tumor images into 4 categories:
- **notumor**: No tumor detected
- **glioma**: Glioma tumor
- **meningioma**: Meningioma tumor
- **pituitary**: Pituitary tumor

## Dataset Structure

Your dataset should be organized as follows:

```
Vit_Dataset/
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
```

**Note**: The dataset mentioned in the problem statement contains ~90k images across these 4 classes.

## Setup Instructions for Google Colab

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clone Repository

```bash
!git clone https://github.com/H0NEYP0T-466/BTSC-UNet-ViT.git
%cd BTSC-UNet-ViT
```

### 3. Install Dependencies

```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install timm pillow opencv-python matplotlib tqdm pydantic pydantic-settings scikit-learn seaborn
```

### 4. Prepare Dataset

Option A: Copy dataset to Colab storage
```bash
!mkdir -p /content/dataset
!cp -r /content/drive/MyDrive/Vit_Dataset /content/dataset/
```

Option B: Create symlink (faster, doesn't use Colab disk space)
```bash
!mkdir -p /content/dataset
!ln -s /content/drive/MyDrive/Vit_Dataset /content/dataset/Vit_Dataset
```

### 5. Run Training

**Basic training (with defaults):**
```bash
!python train_vit_colab.py
```

**Custom parameters:**
```bash
!python train_vit_colab.py \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --patience 10
```

## Training Script Features

### Anti-Overfitting Mechanisms

The training script includes several features to prevent overfitting:

1. **EarlyStopping**: Stops training if validation loss doesn't improve for 10 epochs (configurable)
2. **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus
3. **Weight Decay**: L2 regularization (0.01) on optimizer
4. **Data Augmentation**: Applied on-the-fly during training
5. **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
6. **Mixed Precision Training**: Faster training with same accuracy

### Data Augmentation

The script applies the following augmentations to training data:
- Random horizontal flip (50% probability)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformations (±10% translation)

### Optimization for T4 GPU

The script is optimized for Google Colab's T4 GPU (15.6GB VRAM, 12GB RAM):

- **Batch size**: 32 (default, adjust based on memory)
- **Mixed precision training**: Enabled by default for faster training
- **Efficient data loading**: 2 workers, pin_memory enabled
- **Weighted sampling**: Handles class imbalance efficiently

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--lr` | 1e-4 | Learning rate |
| `--patience` | 10 | Early stopping patience |
| `--image_size` | 224 | Input image size (ViT default) |
| `--train_split` | 0.8 | Training set ratio |
| `--num_workers` | 2 | Data loader workers |
| `--no_augment` | False | Disable data augmentation |

## Output Files

The training script generates:

1. **Checkpoints**:
   - `vit_best.pth`: Best model (highest validation accuracy)
   - `vit_last.pth`: Last epoch checkpoint

2. **Visualizations**:
   - `training_curves.png`: Loss and accuracy curves
   - `confusion_matrix.png`: Final confusion matrix

3. **Logs**: Console output with:
   - Dataset statistics
   - Training/validation metrics per epoch
   - Learning rate changes
   - Early stopping notifications

## Monitoring Training

### Check GPU Usage
```bash
!nvidia-smi
```

### View Training Progress
The script displays:
- Real-time progress bars for each epoch
- Epoch summaries with loss and accuracy
- Learning rate updates
- Best model notifications

## Expected Training Time

- **50 epochs**: ~2-4 hours on T4 GPU
- **Time per epoch**: ~3-5 minutes (depends on dataset size)

## Download Trained Model

After training completes:

```python
from google.colab import files
files.download('/content/checkpoints/vit_best.pth')
```

## Troubleshooting

### Out of Memory Error
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--num_workers` to 1

### Dataset Not Found
- Verify dataset path: `!ls /content/dataset/Vit_Dataset`
- Check folder names match: notumor, glioma, meningioma, pituitary

### Slow Training
- Ensure GPU is enabled: Runtime → Change runtime type → GPU
- Check GPU usage: `!nvidia-smi`

## Integration with Pipeline

The trained ViT model is used in the inference pipeline:

**New Pipeline Flow**:
1. **Preprocessing**: Image enhancement and normalization
2. **ViT Classification**: Classify tumor type
3. **Conditional Segmentation**: 
   - If **notumor**: Skip segmentation (end pipeline)
   - If **tumor detected**: Perform UNet segmentation

This approach is more efficient as it skips expensive segmentation for healthy scans.

## Example Usage

Complete workflow in one cell:

```python
# Mount drive
from google.colab import drive
drive.mount('/content/drive')

# Clone and setup
!git clone https://github.com/H0NEYP0T-466/BTSC-UNet-ViT.git
%cd BTSC-UNet-ViT

# Install dependencies
!pip install -q torch torchvision timm pillow opencv-python matplotlib tqdm pydantic pydantic-settings scikit-learn seaborn

# Link dataset
!mkdir -p /content/dataset
!ln -s /content/drive/MyDrive/Vit_Dataset /content/dataset/Vit_Dataset

# Train
!python train_vit_colab.py --epochs 50 --batch_size 32

# Download model
from google.colab import files
files.download('/content/checkpoints/vit_best.pth')
```

## Notes

- The script prints the final dataset size after augmentation setup
- Augmentation is applied **on-the-fly** during training, not pre-computed
- The actual dataset size remains the same; augmentation creates variations during training
- Weighted sampling helps with class imbalance
- Mixed precision training is automatically disabled if GPU is not available

## Support

For issues or questions, please refer to the main repository README or create an issue on GitHub.
