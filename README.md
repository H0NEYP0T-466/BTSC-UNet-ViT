# BTSC-UNet-ViT

Brain Tumor Segmentation and Classification using UNet and Vision Transformer (ViT).

## Overview

Full-stack application for automated brain tumor analysis in MRI images:
- **Preprocessing**: Edge-preserving denoising, contrast enhancement, normalization
- **Segmentation**: UNet-based tumor detection
- **Classification**: ViT-based tumor type classification

### Tumor Classes
- No Tumor
- Glioma
- Meningioma
- Pituitary Tumor

## Architecture

```
Frontend (React + TypeScript) → Backend (FastAPI + Python) → Models (UNet + ViT)
```

### Pipeline
1. User uploads brain MRI image
2. Image preprocessing (edge-preserving denoising, contrast enhancement, normalization)
3. UNet segments tumor region
4. ViT classifies tumor type
5. Results displayed with confidence scores

## Quick Start

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

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
- **Preprocessing Pipeline**: Edge-preserving image enhancement
- **UNet Segmentation**: Tumor detection
- **ViT Classification**: Pretrained transformer fine-tuned on medical images
- **Batch Processing**: Dataset preprocessing service

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
│   │   │   ├── unet/            # UNet model
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
├── package.json
└── README.md
```

## Documentation

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

### UNet Training (on BraTS dataset)

```bash
cd backend
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
