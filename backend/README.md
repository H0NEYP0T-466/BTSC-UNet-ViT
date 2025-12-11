# BTSC-UNet-ViT Backend

Brain Tumor Segmentation and Classification backend using FastAPI, UNet, and Vision Transformer (ViT).

## Features

- **Preprocessing Pipeline**: Denoising, contrast enhancement, normalization, sharpening
- **UNet Segmentation**: Brain tumor segmentation with trained UNet model
- **ViT Classification**: Fine-tuned Vision Transformer for tumor classification
- **Comprehensive Logging**: Verbose logging at every stage with structured context
- **RESTful API**: FastAPI endpoints for all operations

## Architecture

```
Preprocessing → UNet Segmentation → ViT Classification
```

### Classes
- `no_tumor`: No tumor detected
- `giloma`: Glioma tumor
- `meningioma`: Meningioma tumor
- `pituitary`: Pituitary tumor

## Setup

### Prerequisites
- Python 3.10+
- CUDA (optional, for GPU acceleration)

### ⚠️ Security Update
**Critical:** This project requires PyTorch 2.6.0+ and MONAI 1.5.1+ to address security vulnerabilities. See [../SECURITY.md](../SECURITY.md) for details.

### Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Configure environment:
```bash
# Create .env file (optional, defaults in config.py)
cp .env.example .env
# Edit .env with your paths
```

### Configuration

Edit `app/config.py` or set environment variables:

- `DATASET_ROOT`: Path to dataset folder (defaults to `backend/dataset`)
- `BRATS_ROOT`: Path to UNet dataset with .h5 files (defaults to `backend/dataset/UNet_Dataset`)
- `SEGMENTED_DATASET_ROOT`: Path to ViT classification dataset (defaults to `backend/dataset/Vit_Dataset`)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Dataset Structure

Place your datasets in the following structure:

```
backend/
├── dataset/
│   ├── UNet_Dataset/          # For UNet segmentation training
│   │   ├── *.h5               # H5 files containing images and masks
│   │   └── ...
│   └── Vit_Dataset/           # For ViT classification training
│       ├── giloma/            # Class folders
│       │   ├── *.jpg
│       │   └── ...
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
```

## Training

### UNet Training (on .h5 dataset)

1. Place your .h5 files in `backend/dataset/UNet_Dataset/`
   - Each .h5 file should contain image and mask data
   - Supported keys: 'image', 'images', 'data', 'X' for images
   - Supported keys: 'mask', 'masks', 'label', 'segmentation' for masks

2. Run training:
```bash
cd backend
python -m app.models.unet.train_unet
```

Checkpoints saved to: `app/resources/checkpoints/unet/`
- Best model: `unet_best.pth`
- Last checkpoint: `unet_last.pth`

### ViT Training (on classified images)

1. Organize your images into class folders in `backend/dataset/Vit_Dataset/`:
   - `giloma/` - Glioma tumor images
   - `meningioma/` - Meningioma tumor images
   - `notumor/` - No tumor images
   - `pituitary/` - Pituitary tumor images

2. Run training:
```bash
cd backend
python -m app.models.vit.train_vit
```

Checkpoints saved to: `app/resources/checkpoints/vit/`
- Best model: `vit_best.pth`
- Last checkpoint: `vit_last.pth`

**Note:** The training scripts will automatically:
- Create train/validation splits (80/20)
- Apply data augmentation for training
- Save checkpoints after each epoch
- Log detailed metrics and progress

## Running the Server

### Development
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server runs at: http://localhost:8000

API Documentation: http://localhost:8000/docs

## API Endpoints

### Health Check
```
GET /api/health
```

### Preprocessing
```
POST /api/preprocess
Body: multipart/form-data with 'file'
Returns: URLs for all preprocessing stages
```

### Segmentation
```
POST /api/segment
Body: multipart/form-data with 'file'
Returns: Mask, overlay, and segmented tumor URLs
```

### Classification
```
POST /api/classify
Body: multipart/form-data with 'file'
Returns: Class, confidence, and logits
```

### Full Inference Pipeline
```
POST /api/inference
Body: multipart/form-data with 'file'
Returns: Complete results with all intermediate artifacts
```

## Logging

Logs are written to:
- Console (stdout)
- File: `app/resources/app.log`

Format includes:
- Timestamp
- Log level
- Module and function
- Message
- Context fields: image_id, path, stage

## Project Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app
│   ├── config.py                # Configuration
│   ├── logging_config.py        # Logging setup
│   ├── routers/                 # API endpoints
│   │   ├── health.py
│   │   ├── preprocessing.py
│   │   ├── segmentation.py
│   │   └── classification.py
│   ├── models/                  # ML models
│   │   ├── unet/
│   │   │   ├── model.py
│   │   │   ├── train_unet.py
│   │   │   └── infer_unet.py
│   │   └── vit/
│   │       ├── model.py
│   │       ├── train_vit.py
│   │       └── infer_vit.py
│   ├── services/                # Business logic
│   │   ├── pipeline_service.py
│   │   ├── storage_service.py
│   │   └── dataset_service.py
│   ├── utils/                   # Utilities
│   │   ├── preprocessing.py
│   │   ├── imaging.py
│   │   ├── metrics.py
│   │   └── logger.py
│   ├── schemas/                 # Pydantic models
│   │   ├── requests.py
│   │   └── responses.py
│   └── resources/               # Artifacts
│       ├── checkpoints/
│       ├── uploads/
│       └── artifacts/
├── tests/                       # Tests
└── requirements.txt
```

## Testing

Run tests:
```bash
pytest tests/
```

## Preprocessing Operations

1. **Grayscale Conversion**: Convert to single channel
2. **Denoising**: Median filter for salt & pepper noise
3. **Motion Artifact Reduction**: Deblurring and normalization
4. **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. **Edge Sharpening**: Unsharp mask
6. **Normalization**: Z-score or min-max normalization

## Model Details

### UNet
- Input: 1-channel grayscale image
- Output: 1-channel binary mask
- Architecture: 5-level encoder-decoder with skip connections
- Loss: Binary Cross-Entropy with Logits

### ViT
- Base model: `vit_base_patch16_224` from timm
- Input: 224×224 RGB image
- Output: 4 classes
- Fine-tuned from ImageNet pretrained weights

## Troubleshooting

### Model not found
- Ensure checkpoints are at:
  - `backend/app/resources/checkpoints/unet/unet_best.pth`
  - `backend/app/resources/checkpoints/vit/vit_best.pth`
- Train models or download pretrained weights

### Out of memory (GPU)
- Reduce batch size in training
- Use CPU inference: Set device='cpu'

### CORS errors
- Add frontend URL to `CORS_ORIGINS` in config.py

## License

MIT

## Authors

BTSC-UNet-ViT Team
