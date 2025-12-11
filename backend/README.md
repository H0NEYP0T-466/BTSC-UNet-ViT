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

- `DATASET_ROOT`: Path to 90k image dataset (X:/file/FAST_API/BTSC-UNet-ViT/dataset)
- `BRATS_ROOT`: Path to BraTS dataset for UNet training
- `SEGMENTED_DATASET_ROOT`: Output path for segmented dataset
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Training

### UNet Training (on BraTS dataset)

1. Prepare BraTS dataset at `BRATS_ROOT`
2. Implement dataloader in `app/models/unet/datamodule.py`
3. Run training:
```bash
python -m app.models.unet.train_unet
```

Checkpoints saved to: `app/resources/checkpoints/unet/`

### ViT Training (on segmented images)

1. Preprocess and segment the 90k dataset:
```python
from app.services.dataset_service import get_dataset_service
service = get_dataset_service()
service.preprocess_and_segment_dataset()
```

2. Implement dataloader in `app/models/vit/datamodule.py`
3. Run training:
```bash
python -m app.models.vit.train_vit
```

Checkpoints saved to: `app/resources/checkpoints/vit/`

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
