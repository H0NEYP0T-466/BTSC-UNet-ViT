# UNet Training Guide for Custom BraTS Dataset

## Overview

This guide explains how to train the UNet model on a custom BraTS-like dataset with **extreme class imbalance** (tumor pixels < 1%).

## Dataset Format

Your dataset should be in `.h5` format with the following structure:

```
UNet_Dataset/
├── volume_194_slice_92.h5
├── volume_195_slice_93.h5
└── ...
```

Each `.h5` file contains:
- **`image`**: Shape `(240, 240, 4)` - 4 MRI modalities (T1, T1ce, T2, FLAIR)
- **`mask`**: Shape `(240, 240, 3)` - Binary tumor mask (will be collapsed to single channel)

## Key Features

### 1. Proper Data Handling
- ✅ **4-channel input support** for multi-modal MRI
- ✅ **Per-channel normalization** (handles preprocessed data correctly)
- ✅ **Binary mask extraction** using `np.max()` (not `np.sum()`)
- ✅ **Extreme class imbalance handling** (works with 0.17% tumor fraction)

### 2. Advanced Loss Function
- **Dice Loss**: Focuses on overlap, naturally handles imbalance
- **BCE Loss**: Provides stable gradients
- **Combined Loss**: `Loss = 0.5 * Dice + 0.5 * BCE`

### 3. Enhanced Visualization
- **'hot' colormap** for tumor masks (makes tiny tumors visible)
- **Probability maps** for continuous predictions
- **Heatmap overlays** for web display
- **Training progress visualization** (saved every 5 epochs)

### 4. Optimization for Google Colab
- **Batch size: 16** (optimized for 15GB GPU)
- **Epochs: 50** (better convergence for imbalanced data)
- **Learning rate scheduler** (ReduceLROnPlateau)
- **Gradient clipping** (stability)
- **Weight decay** (regularization)

## Training on Google Colab

### Step 1: Setup Environment

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/H0NEYP0T-466/BTSC-UNet-ViT.git
%cd BTSC-UNet-ViT

# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install h5py opencv-python matplotlib tqdm pydantic pydantic-settings
```

### Step 2: Prepare Dataset

```bash
# Option 1: Copy from Google Drive
!cp -r /content/drive/MyDrive/UNet_Dataset /content/UNet_Dataset

# Option 2: Create symbolic link
!ln -s /content/drive/MyDrive/UNet_Dataset /content/UNet_Dataset
```

### Step 3: Validate Dataset

```bash
# Test dataset loading and visualization
!python backend/tests/test_dataset_validation.py \
    --dataset_path /content/UNet_Dataset \
    --output_dir /content/test_output
```

This will generate:
- `raw_data_visualization.png` - Shows raw .h5 file content
- `dataset_loader_visualization.png` - Shows processed samples
- Console output with statistics

### Step 4: Train Model

```bash
# Basic training (default parameters)
!python train_unet_colab.py

# Custom training parameters
!python train_unet_colab.py \
    --dataset_path /content/UNet_Dataset \
    --checkpoint_dir /content/checkpoints \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --image_size 256
```

**Parameters:**
- `--dataset_path`: Path to folder with .h5 files
- `--checkpoint_dir`: Where to save model checkpoints
- `--batch_size`: Batch size (16 recommended for 15GB GPU)
- `--epochs`: Number of epochs (50-100 recommended)
- `--lr`: Learning rate (1e-4 default)
- `--image_size`: Input image size (256x256 default)

### Step 5: Monitor Training

Training will output:
```
Epoch 1/50:
  [Train] loss=0.8234, dice=0.1234, acc=0.9982
  [Val]   loss=0.7845, dice=0.1567, acc=0.9983
  ✅ New best Dice score: 0.1567
```

Visualizations are saved to checkpoint directory:
- `train_vis_epoch_X.png` - Training samples
- `val_vis_epoch_X.png` - Validation samples
- `unet_best.pth` - Best model checkpoint
- `unet_last.pth` - Latest model checkpoint

### Step 6: Download Model

```python
from google.colab import files
files.download('/content/checkpoints/unet_best.pth')
```

## Training Locally

### Prerequisites
```bash
pip install torch torchvision h5py opencv-python matplotlib tqdm pydantic pydantic-settings
```

### Run Training
```bash
cd backend
python -m app.models.unet.train_unet
```

Configuration is in `backend/app/config.py`:
```python
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 50
LEARNING_RATE: float = 1e-4
UNET_IN_CHANNELS: int = 4
UNET_OUT_CHANNELS: int = 1
```

## Testing Inference

### Test on Sample File

```bash
python backend/tests/test_unet_inference_validation.py \
    --checkpoint /content/checkpoints/unet_best.pth \
    --sample /content/UNet_Dataset/volume_194_slice_92.h5 \
    --output_dir /content/inference_test
```

This generates:
- `inference_test_results.png` - Side-by-side comparison
- Console metrics (Dice score, tumor ratio, etc.)

### Use in Web Application

The model is automatically loaded by the backend API:

```python
from app.models.unet.infer_unet import get_unet_inference

# Get inference instance
unet = get_unet_inference()

# Segment image
results = unet.segment_image(image, image_id="test_001")

# Results contain:
# - results['mask']: Binary mask (0-255)
# - results['overlay']: Image with red overlay
# - results['heatmap']: Enhanced visualization for tiny tumors
# - results['probability_map']: Continuous prediction (0-255)
```

## Understanding Results

### Metrics

1. **Dice Score**: Overlap between prediction and ground truth
   - 0.0 = No overlap
   - 1.0 = Perfect overlap
   - For extreme imbalance, 0.3-0.5 is good

2. **Pixel Accuracy**: % of correctly classified pixels
   - Usually > 99% due to class imbalance
   - Not a good metric for tumor detection

3. **Tumor Ratio**: % of pixels predicted as tumor
   - Ground truth: ~0.17%
   - Good prediction: 0.1-0.5%
   - Bad prediction: 50%+ (whole brain marked as tumor)

### Visualization Tips

1. **Use 'hot' colormap** for masks:
   ```python
   plt.imshow(mask, cmap='hot', interpolation='nearest')
   ```

2. **Show probability maps** (not just binary):
   ```python
   prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
   plt.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
   ```

3. **Adjust alpha** for overlays:
   ```python
   overlay = 0.7 * image + 0.3 * colored_mask
   ```

## Troubleshooting

### Issue: Whole brain marked as tumor
**Cause**: Incorrect mask handling or loss function
**Solution**: Use `np.max()` to collapse mask channels + Dice+BCE loss

### Issue: No tumor detected (all black)
**Cause**: Extreme class imbalance, threshold too high
**Solution**: 
- Check probability map (not just binary)
- Lower threshold to 0.3-0.4
- Ensure using Dice+BCE loss

### Issue: Training diverges (NaN loss)
**Cause**: Exploding gradients, bad normalization
**Solution**:
- Enable gradient clipping (already added)
- Check data normalization
- Reduce learning rate

### Issue: Out of memory
**Cause**: Batch size too large
**Solution**:
- Reduce batch size to 8
- Reduce image size to 224x224
- Reduce number of workers

### Issue: Model not learning (Dice stays at 0)
**Cause**: Learning rate too low, not enough epochs
**Solution**:
- Increase epochs to 100
- Try learning rate 2e-4
- Check dataset has tumor samples

## Expected Training Time

On Google Colab (T4 GPU, 15GB):
- **Dataset size**: 10,000 samples
- **Batch size**: 16
- **Epochs**: 50
- **Time per epoch**: ~3-5 minutes
- **Total time**: ~2.5-4 hours

## Model Architecture

```
UNet (4 → 1 channels)
├── Encoder
│   ├── Level 1: 4 → 16
│   ├── Level 2: 16 → 32
│   ├── Level 3: 32 → 64
│   ├── Level 4: 64 → 128
│   └── Level 5: 128 → 256
├── Bottleneck: 256 → 512
└── Decoder (with skip connections)
    ├── Level 5: 512 → 256
    ├── Level 4: 256 → 128
    ├── Level 3: 128 → 64
    ├── Level 2: 64 → 32
    └── Level 1: 32 → 16
└── Output: 16 → 1 (sigmoid)

Total parameters: ~7.8M
```

## References

- **Dice Loss**: Milletari et al. (2016) "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
- **UNet**: Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **Class Imbalance**: Sudre et al. (2017) "Generalised Dice overlap as a deep learning loss function"

## Support

For issues or questions:
1. Check `backend/tests/dataset_issue_diagnosis.txt`
2. Run validation tests: `test_dataset_validation.py`
3. Check training visualizations in checkpoint directory
4. Review logs in console output

## License

This project is part of BTSC-UNet-ViT. See main repository for license.
