# Brain UNet Training Guide

## Overview

This guide explains how to train the Brain UNet model for brain extraction/segmentation using the NFBS (Neurofeedback Skull-stripped) dataset.

## Dataset Structure

The NFBS dataset should be organized as follows:

```
NFBS_Dataset/
├── A00028185/
│   ├── sub-A00028185_ses-NFB3_T1w.nii.gz
│   ├── sub-A00028185_ses-NFB3_T1w_brain.nii.gz
│   └── sub-A00028185_ses-NFB3_T1w_brainmask.nii.gz
├── A00028352/
│   ├── sub-A00028352_ses-NFB3_T1w.nii.gz
│   ├── sub-A00028352_ses-NFB3_T1w_brain.nii.gz
│   └── sub-A00028352_ses-NFB3_T1w_brainmask.nii.gz
└── ...
```

Each subject folder contains:
- `*_T1w.nii.gz`: Raw T1-weighted MRI scan (3D volume)
- `*_T1w_brainmask.nii.gz`: Binary brain mask (3D volume)

## Training on Google Colab

### 1. Setup Environment

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/H0NEYP0T-466/BTSC-UNet-ViT.git
%cd BTSC-UNet-ViT

# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install nibabel opencv-python matplotlib tqdm pydantic pydantic-settings numpy scikit-image
```

### 2. Prepare Dataset

```bash
# Option 1: Copy dataset to Colab
!cp -r /content/drive/MyDrive/NFBS_Dataset /content/NFBS_Dataset

# Option 2: Create symlink
!ln -s /content/drive/MyDrive/NFBS_Dataset /content/NFBS_Dataset
```

### 3. Run Training

#### Basic Training
```python
!python train_brain_unet_colab.py
```

#### Advanced Training with Custom Parameters
```python
!python train_brain_unet_colab.py \
    --dataset_path /content/NFBS_Dataset \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --image_size 256 \
    --checkpoint_dir /content/checkpoints/brain_unet
```

#### Parameters Explained

- `--dataset_path`: Path to NFBS dataset (default: `/content/NFBS_Dataset`)
- `--checkpoint_dir`: Where to save model checkpoints (default: `/content/checkpoints/brain_unet`)
- `--batch_size`: Batch size for training (default: 32, good for 15GB GPU)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--image_size`: Image size for training (default: 256)
- `--num_workers`: Number of data loading workers (default: 2)
- `--train_split`: Train/val split ratio (default: 0.8)
- `--slice_start`: Optional start slice index for limiting dataset
- `--slice_end`: Optional end slice index for limiting dataset

### 4. Monitor Training

The training script will:
- Display progress bars for each epoch
- Show metrics: loss, dice score, IoU, accuracy
- Save visualizations every 5 epochs to checkpoint directory
- Auto-save best model based on dice score
- Save last checkpoint after each epoch

Expected output:
```
================================================================================
GOOGLE COLAB BRAIN UNET TRAINING SETUP
================================================================================
✅ GPU Available: Tesla T4
✅ GPU Memory: 15.0 GB

✅ Dataset found: 125 subjects

📊 Sample subject analysis: A00028185
   T1w files: 1
   Mask files: 1
   T1w shape: (256, 256, 192)
   Mask shape: (256, 256, 192)
...

Epoch 1/50 [Train]: 100%|██████████| 250/250 [02:15<00:00, loss=0.2345, dice=0.8123]
Epoch 1/50 [Val]:   100%|██████████| 63/63 [00:30<00:00, loss=0.2156, dice=0.8342]

Epoch 1 Summary:
  Train: loss=0.2345, dice=0.8123, iou=0.7856, acc=0.9534
  Val:   loss=0.2156, dice=0.8342, iou=0.8012, acc=0.9601
  LR: 0.000100
  ✅ Saved best model (dice=0.8342) to /content/checkpoints/brain_unet/brain_unet_best.pth
```

### 5. Download Trained Model

```python
from google.colab import files

# Download best model
files.download('/content/checkpoints/brain_unet/brain_unet_best.pth')

# Download last checkpoint
files.download('/content/checkpoints/brain_unet/brain_unet_last.pth')
```

## Model Architecture

The Brain UNet uses a U-Net architecture with:
- **Input**: Single-channel T1w MRI (256x256)
- **Output**: Binary brain mask (256x256)
- **Features**: (32, 64, 128, 256, 512) channels at each level
- **Total Parameters**: ~31 million
- **Loss Function**: Combined Dice Loss + Binary Cross-Entropy (50/50)

## Performance Expectations

With NFBS dataset (125 subjects, ~15,000 slices):
- **Training Time**: ~5-10 minutes per epoch with optimizations on 15GB GPU (previously 1h 11min)
- **Expected Dice Score**: 0.90-0.95 after training
- **Expected IoU**: 0.85-0.92 after training
- **Memory Usage**: ~8-10GB GPU RAM with batch size 32

### Performance Optimizations (Applied)
The training has been optimized with:
1. **In-memory data caching**: Pre-loads all slices at initialization (80% faster)
2. **GPU-native metrics**: Removes CPU-GPU transfers during training (15% faster)
3. **Automatic Mixed Precision (AMP)**: Uses FP16 for faster computation (20% faster)
4. **Optimized data loading**: Non-blocking transfers and persistent workers

## Pipeline Integration

Once trained, the Brain UNet is integrated into the full pipeline:

```
User Upload Image
    ↓
Preprocessing (grayscale, denoise, contrast, sharpen, normalize)
    ↓
Brain UNet (extract brain tissue)
    ↓
Tumor UNet (segment tumor from brain)
    ↓
ViT Classifier (classify tumor type)
    ↓
Results
```

## Testing

To test the model and dataset:

```bash
cd backend
python -m tests.test_brain_unet
```

This will test:
1. ✅ Brain UNet model creation
2. ✅ NFBS dataset loading
3. ✅ Utility functions (loss, IoU)
4. ✅ Inference setup

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size
```python
!python train_brain_unet_colab.py --batch_size 16
```

### Issue: Dataset not found
**Solution**: Verify dataset path
```python
import os
print(os.listdir('/content'))
# Should show NFBS_Dataset folder
```

### Issue: Training too slow
**Solution 1**: Ensure cache_in_memory is enabled (default)
```python
!python train_brain_unet_colab.py --cache_in_memory True
```

**Solution 2**: Ensure AMP is enabled (default)
```python
!python train_brain_unet_colab.py --use_amp True
```

**Solution 3**: Reduce number of slices or use smaller image size
```python
!python train_brain_unet_colab.py --slice_start 60 --slice_end 140 --image_size 128
```

### Issue: Out of Memory during data loading
**Solution**: Disable in-memory caching
```python
!python train_brain_unet_colab.py --cache_in_memory False --num_workers 2
```

## Model Deployment

After training, copy the best checkpoint to the backend:

```bash
# Local deployment
cp /content/checkpoints/brain_unet/brain_unet_best.pth \
   backend/resources/checkpoints/brain_unet/brain_unet_best.pth

# Or upload to Google Drive and download to local
```

Then restart the backend server:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The Brain UNet will be automatically loaded on first inference request.

## API Endpoints

### Brain Segmentation Only
```bash
curl -X POST "http://localhost:8080/api/segment-brain" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.nii.gz"
```

### Full Pipeline (Preprocessing → Brain → Tumor → Classification)
```bash
curl -X POST "http://localhost:8080/api/inference" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.nii.gz"
```

## References

- NFBS Dataset: [OpenNeuro](https://openneuro.org)
- U-Net Paper: [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Brain Extraction: Removes skull, eyes, neck from MRI scans
