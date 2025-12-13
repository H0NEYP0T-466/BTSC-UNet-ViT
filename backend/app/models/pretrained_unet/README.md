# Pretrained UNet for Brain Tumor Segmentation

This module uses pretrained UNet models specifically trained for brain tumor segmentation on the BraTS dataset.

## Setup

1. **Download the pretrained model (one-time)**:
   ```bash
   cd backend
   python -m app.models.pretrained_unet.download_model
   ```

2. The model will be downloaded to `backend/resources/checkpoints/pretrained_unet/`

## Models

We use MONAI's pretrained UNet architecture optimized for medical image segmentation:
- **Architecture**: UNet with ResNet encoder
- **Training Dataset**: BraTS (Brain Tumor Segmentation Challenge)
- **Input**: Single channel grayscale MRI (256x256 or any size)
- **Output**: Binary segmentation mask (tumor region only)
- **Performance**: High accuracy on tumor-only segmentation

## Usage

The pretrained model is automatically integrated into the pipeline. You can switch between:
- **Local trained model**: Uses `backend/app/models/unet/`
- **Pretrained model**: Uses `backend/app/models/pretrained_unet/`

Configuration is in `backend/app/config.py` via the `USE_PRETRAINED_UNET` setting.

## Advantages

- No training required
- Better performance on tumor detection
- Specifically trained to segment only tumors (not whole brain)
- More robust to various MRI modalities
