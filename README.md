# BTSC-UNet-ViT

Brain Tumor Segmentation and Classification using UNet and Vision Transformer (ViT).

## Overview

Full-stack application for automated brain tumor analysis in MRI images:
- **Preprocessing**: HD-BET brain extraction, edge-preserving denoising, contrast enhancement, normalization
- **Segmentation**: Pretrained UNet-based tumor-only detection
- **Classification**: ViT-based tumor type classification

### Tumor Classes
- No Tumor
- Glioma
- Meningioma
- Pituitary Tumor

## ✨ New Features

### Pretrained UNet Model (Recommended)
- **No training required** - Ready to use immediately
- **Tumor-only segmentation** - Precisely segments tumor regions, not the whole brain
- **MONAI-based architecture** - Medical imaging optimized
- **Improved preprocessing** - Edge-preserving bilateral filtering

See [PRETRAINED_UNET_SETUP.md](PRETRAINED_UNET_SETUP.md) for detailed setup instructions.

## Architecture

```
Frontend (React + TypeScript) → Backend (FastAPI + Python) → Models (Pretrained UNet + ViT)
```

### Pipeline
1. User uploads brain MRI image
2. Image preprocessing (HD-BET brain extraction, edge-preserving denoising, contrast enhancement, normalization)
3. Pretrained UNet segments tumor region only
4. ViT classifies tumor type
5. Results displayed with confidence scores

## Quick Start

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup HD-BET brain extraction (one-time setup)
python setup_hdbet.py

# Download pretrained model (one-time setup)
python -m app.models.pretrained_unet.download_model

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API: http://localhost:8000/docs

### Frontend Setup

```bash
# Install dependencies
npm install

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# Run development server
npm run dev
```

App: http://localhost:5173

## Features

### Backend
- **Verbose Logging**: Structured logging at every stage
- **RESTful API**: FastAPI with automatic OpenAPI docs
- **HD-BET Brain Extraction**: Skull-stripping to isolate brain tissue
- **Improved Preprocessing Pipeline**: Edge-preserving 7-stage image enhancement
- **Pretrained UNet Segmentation**: MONAI-based tumor-only detection (default)
- **Local UNet Training**: Optional local training on BraTS dataset
- **ViT Classification**: Pretrained transformer fine-tuned on medical images
- **Batch Processing**: Dataset preprocessing service
- **Model Selection**: Easy toggle between pretrained and local models

### Frontend
- **Dark Theme**: Modern UI with #111 background and #00C2FF accent
- **Drag & Drop**: Easy file upload
- **Real-time Visualization**: View all processing stages
- **Final Image Indicator**: Highlights the preprocessed image passed to models
- **Responsive Design**: Works on all devices
- **No Tailwind**: Clean, component-based CSS

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Configuration
│   │   ├── logging_config.py    # Logging setup
│   │   ├── routers/             # API endpoints
│   │   ├── models/              # UNet & ViT models
│   │   │   ├── unet/            # Local trained UNet
│   │   │   ├── pretrained_unet/ # Pretrained MONAI UNet (NEW)
│   │   │   └── vit/             # Vision Transformer
│   │   ├── services/            # Business logic
│   │   ├── utils/               # Preprocessing, imaging
│   │   └── schemas/             # Pydantic models
│   ├── tests/                   # Tests
│   ├── requirements.txt
│   └── README.md
├── src/                         # Frontend
│   ├── components/              # React components
│   ├── pages/                   # Page layouts
│   ├── services/                # API client
│   └── theme/                   # CSS variables & styles
├── PRETRAINED_UNET_SETUP.md     # Pretrained model guide (NEW)
├── package.json
└── README.md
```

## Documentation

- [HD-BET Brain Extraction Setup](hdbet.md) - **Required first-time setup** for brain extraction
- [Pretrained UNet Setup](PRETRAINED_UNET_SETUP.md) - Quick setup for pretrained models
- [Backend README](backend/README.md) - API, training, deployment
- [Frontend README](frontend_README.md) - Components, styling, development

## API Endpoints

### Health Check
```
GET /api/health
```

### Full Inference Pipeline
```
POST /api/inference
Body: multipart/form-data with 'file'
```

### Individual Stages
```
POST /api/preprocess    # Preprocessing only
POST /api/segment       # Segmentation only
POST /api/classify      # Classification only
```

## Model Setup & Training

### Option 1: Use Pretrained UNet (Recommended - No Training Required)

```bash
cd backend
python -m app.models.pretrained_unet.download_model
```

This creates a ready-to-use MONAI UNet model optimized for brain tumor segmentation.

### Option 2: Train Local UNet (on BraTS dataset)

```bash
cd backend
# Set USE_PRETRAINED_UNET=False in config.py first
python -m app.models.unet.train_unet
```

### ViT Training (on segmented dataset)
```bash
# First, segment the dataset
python -c "from app.services.dataset_service import get_dataset_service; get_dataset_service().preprocess_and_segment_dataset()"

# Then train ViT
python -m app.models.vit.train_vit
```

## Configuration

### Backend (.env)
```bash
DATASET_ROOT=X:/file/FAST_API/BTSC-UNet-ViT/dataset
BRATS_ROOT=X:/data/BraTS
LOG_LEVEL=INFO
```

### Model Selection (backend/app/config.py)
```python
# Use pretrained UNet (default, recommended)
USE_PRETRAINED_UNET: bool = True

# Or use local trained model
USE_PRETRAINED_UNET: bool = False
```

### Frontend (.env)
```bash
VITE_API_URL=http://localhost:8000
```

## Development

### Run Tests
```bash
cd backend
pytest tests/
```

### Linting
```bash
# Frontend
npm run lint

# Backend
pip install flake8
flake8 app/
```

## Deployment

### Backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
npm run build
# Serve dist/ with nginx or any static server
```

## Technologies

### Backend
- FastAPI
- PyTorch 2.6.0+ (security patched)
- timm (Vision Transformers)
- MONAI 1.5.1+ (security patched)
- HD-BET (brain extraction)
- scikit-image
- OpenCV

**⚠️ Security Note:** PyTorch and MONAI have been updated to address critical vulnerabilities. See [SECURITY.md](SECURITY.md).

### Frontend
- React 19
- TypeScript
- Axios
- Vite

## License

MIT

## Citation

If you use this project, please cite:
```
@software{btsc_unet_vit,
  title = {BTSC-UNet-ViT: Brain Tumor Segmentation and Classification},
  author = {BTSC-UNet-ViT Team},
  year = {2024}
}
```

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Contact

For questions or issues, please open a GitHub issue.

## Acknowledgments

- BraTS dataset for segmentation training
- timm library for pretrained ViT models
- MONAI for medical imaging utilities
