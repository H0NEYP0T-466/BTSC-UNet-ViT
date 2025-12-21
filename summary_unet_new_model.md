# UNet Tumor Model Summary

## Overview

This document describes the new **UNet Tumor model** for tumor segmentation on PNG-based datasets. This model is separate from the existing UNet model which handles BraTS .h5 data with 4-channel modalities.

## Dataset Structure

The UNet Tumor model expects a dataset in the following format:

```
UNet_Tumor_Dataset/
├── images/          # PNG images (RGB or grayscale)
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── masks/           # PNG masks (same filenames as images)
    ├── image001.png
    ├── image002.png
    └── ...
```

### Dataset Requirements
- **Image format**: PNG (RGB or grayscale)
- **Mask format**: PNG (binary - white for tumor, black for background)
- **Matching pairs**: Each image must have a corresponding mask with the same filename

## Expected Dataset Size After Augmentation

### Base Dataset
- **Original images**: ~3,000 images
- **Train/Val split**: 80/20

### Augmentation Applied (On-the-fly)
The following augmentations are applied during training to prevent overfitting:

| Augmentation | Probability | Variants |
|-------------|-------------|----------|
| Random Horizontal Flip | 50% | 2x |
| Random Vertical Flip | 50% | 2x |
| Random 90° Rotation | 25% each | 4x |
| Random Brightness | 50% | Variable |
| Random Contrast | 50% | Variable |
| Random Gaussian Noise | 30% | Variable |

### Effective Dataset Size

| Split | Base Images | Effective Per Epoch |
|-------|-------------|---------------------|
| Training | 2,400 | ~7,200 (3x augmentation) |
| Validation | 600 | 600 (no augmentation) |
| **Total** | **3,000** | **~7,800** |

**Note**: Since augmentation is applied on-the-fly, each training epoch sees different variations of the images. Over 100 epochs, the model effectively sees ~720,000 unique augmented training samples.

## Model Architecture

### UNet Tumor Model
- **Input Channels**: 3 (RGB images)
- **Output Channels**: 1 (binary segmentation)
- **Feature Channels**: (32, 64, 128, 256, 512)
- **Dropout Rate**: 0.2 (for regularization)
- **Total Parameters**: ~31M

### Comparison with Original UNet (BraTS)
| Feature | UNet (BraTS) | UNet Tumor (PNG) |
|---------|--------------|------------------|
| Input Channels | 4 (T1, T1ce, T2, FLAIR) | 3 (RGB) |
| Feature Channels | (16, 32, 64, 128, 256) | (32, 64, 128, 256, 512) |
| Dropout | No | Yes (0.2) |
| Dataset Format | .h5 files | PNG images |

## Overfitting Prevention Strategies

The model implements multiple strategies to prevent overfitting:

### 1. Data Augmentation
- Random flips, rotations, and color adjustments
- Effectively 3x dataset size per epoch
- Different augmentations each epoch

### 2. ReduceLROnPlateau Scheduler
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',           # Maximize Dice score
    factor=0.5,           # Reduce LR by half
    patience=5,           # Wait 5 epochs before reducing
    min_lr=1e-7           # Minimum learning rate
)
```

### 3. Early Stopping
```python
early_stopping = EarlyStopping(
    patience=15,          # Stop after 15 epochs without improvement
    min_delta=0.0         # Minimum change to qualify as improvement
)
```

### 4. Regularization
- **Dropout**: 0.2 dropout rate in convolution blocks
- **Weight Decay**: 0.01 L2 regularization in AdamW optimizer
- **Gradient Clipping**: Max norm of 1.0

### 5. Loss Function
Combined Dice + BCE loss for better handling of class imbalance:
```python
loss = 0.5 * dice_loss + 0.5 * bce_loss
```

## Training Configuration

### Recommended Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 16 | Optimal for 15GB GPU |
| Learning Rate | 1e-4 | Initial learning rate |
| Epochs | 100 | Maximum epochs |
| Image Size | 256x256 | Input resolution |
| Early Stopping Patience | 15 | Epochs before stopping |
| LR Reduction Patience | 5 | Epochs before reducing LR |

### Training Command
```bash
python train_unet_tumor_colab.py \
    --dataset_path /content/UNet_Tumor_Dataset \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --dropout 0.2
```

## Pipeline Integration

The UNet Tumor model is integrated into the inference pipeline as **UNet2**:

```
Input Image
    ↓
Preprocessing (grayscale, denoise, contrast, sharpen, normalize)
    ↓
ViT Classification
    ↓
[If tumor detected]
    ↓
UNet1 Segmentation (BraTS model) + UNet2 Segmentation (PNG model)
    ↓
Results Display (both segmentations shown)
```

## Model Checkpoints

The trained model is saved to:
- **Best model**: `checkpoints/unet_tumor/unet_tumor_best.pth`
- **Last model**: `checkpoints/unet_tumor/unet_tumor_last.pth`

## API Endpoints

The model results are available through the `/api/inference` endpoint:

```json
{
  "tumor_segmentation": {
    "mask": "/artifacts/...",
    "overlay": "/artifacts/...",
    "segmented": "/artifacts/..."
  },
  "tumor_segmentation2": {
    "mask": "/artifacts/...",
    "overlay": "/artifacts/...",
    "segmented": "/artifacts/..."
  }
}
```

## Files Created

| File | Description |
|------|-------------|
| `backend/app/models/unet_tumor/__init__.py` | Module exports |
| `backend/app/models/unet_tumor/model.py` | UNet architecture |
| `backend/app/models/unet_tumor/datamodule.py` | Dataset with augmentation |
| `backend/app/models/unet_tumor/train_unet_tumor.py` | Training script |
| `backend/app/models/unet_tumor/infer_unet_tumor.py` | Inference module |
| `backend/app/models/unet_tumor/utils.py` | Utility functions |
| `train_unet_tumor_colab.py` | Colab training script |

## Frontend Changes

The frontend now displays both segmentation results:
- **UNet1 (BraTS)**: Original model for multi-modal MRI
- **UNet2 (PNG)**: New model for standard images
