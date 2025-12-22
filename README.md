<h1 align="center">BTSC-UNet-ViT</h1>

<p align="center">Brain Tumor Segmentation and Classification using UNet and Vision Transformer (ViT)</p>

<p align="center">

  <!-- Core -->
  ![GitHub License](https://img.shields.io/github/license/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=brightgreen)
  ![GitHub Stars](https://img.shields.io/github/stars/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=yellow)
  ![GitHub Forks](https://img.shields.io/github/forks/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=blue)
  ![GitHub Issues](https://img.shields.io/github/issues/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=red)
  ![GitHub Pull Requests](https://img.shields.io/github/issues-pr/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=orange)
  ![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)

  <!-- Activity -->
  ![Last Commit](https://img.shields.io/github/last-commit/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=purple)
  ![Commit Activity](https://img.shields.io/github/commit-activity/m/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=teal)
  ![Repo Size](https://img.shields.io/github/repo-size/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=blueviolet)
  ![Code Size](https://img.shields.io/github/languages/code-size/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=indigo)

  <!-- Languages -->
  ![Top Language](https://img.shields.io/github/languages/top/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=critical)
  ![Languages Count](https://img.shields.io/github/languages/count/H0NEYP0T-466/BTSC-UNet-ViT?style=for-the-badge&color=success)

  <!-- Community -->
  ![Documentation](https://img.shields.io/badge/Docs-Available-green?style=for-the-badge&logo=readthedocs&logoColor=white)
  ![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=for-the-badge)

</p>

---

## 🔗 Quick Links

- [🚀 Quick Start](#-quick-start)
- [✨ Features](#-features)
- [📂 Project Structure](#-project-structure)
- [🛠 Tech Stack](#-tech-stack)
- [📦 Dependencies](#-dependencies)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🛡 Security](#-security)

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#-quick-start)
4. [Features](#-features)
5. [Project Structure](#-project-structure)
6. [Documentation](#documentation)
7. [API Endpoints](#api-endpoints)
8. [Model Setup & Training](#model-setup--training)
9. [Configuration](#configuration)
10. [Development](#development)
11. [Deployment](#deployment)
12. [Tech Stack](#-tech-stack)
13. [Dependencies](#-dependencies)
14. [Contributing](#-contributing)
15. [License](#-license)
16. [Security](#-security)
17. [Code of Conduct](#-code-of-conduct)
18. [Citation](#citation)
19. [Contact](#contact)
20. [Acknowledgments](#acknowledgments)

---

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

## 🚀 Quick Start

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

## ✨ Features

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

## 📂 Project Structure

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

---

## 🛠 Tech Stack

### Languages

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

### Frameworks & Libraries

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![scikit--image](https://img.shields.io/badge/scikit--image-013220?style=for-the-badge)

### DevOps / CI / Tools

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)
![npm](https://img.shields.io/badge/npm-CB3837?style=for-the-badge&logo=npm&logoColor=white)
![ESLint](https://img.shields.io/badge/ESLint-4B32C3?style=for-the-badge&logo=eslint&logoColor=white)

### AI/ML

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-00758F?style=for-the-badge)
![timm](https://img.shields.io/badge/timm-FF6F00?style=for-the-badge)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

---

## 📦 Dependencies

This project uses the following dependencies across frontend and backend:

<details>
<summary><b>Frontend Runtime Dependencies</b></summary>

![axios](https://img.shields.io/npm/v/axios?style=for-the-badge&label=axios&color=5A29E4)
![react](https://img.shields.io/npm/v/react?style=for-the-badge&label=react&color=61DAFB)
![react-dom](https://img.shields.io/npm/v/react-dom?style=for-the-badge&label=react-dom&color=61DAFB)

- **axios** ^1.13.2 - Promise-based HTTP client for API requests
- **react** ^19.2.0 - Core React library for building UI components
- **react-dom** ^19.2.0 - React DOM bindings

</details>

<details>
<summary><b>Frontend Dev Dependencies</b></summary>

![typescript](https://img.shields.io/npm/v/typescript?style=for-the-badge&label=typescript&color=007ACC)
![vite](https://img.shields.io/npm/v/vite?style=for-the-badge&label=vite&color=646CFF)
![eslint](https://img.shields.io/npm/v/eslint?style=for-the-badge&label=eslint&color=4B32C3)

- **@eslint/js** ^9.39.1 - ESLint JavaScript plugin
- **@types/node** ^24.10.4 - TypeScript definitions for Node.js
- **@types/react** ^19.2.5 - TypeScript definitions for React
- **@types/react-dom** ^19.2.3 - TypeScript definitions for React DOM
- **@vitejs/plugin-react** ^5.1.1 - Vite plugin for React
- **eslint** ^9.39.1 - JavaScript/TypeScript linter
- **eslint-plugin-react-hooks** ^7.0.1 - ESLint rules for React Hooks
- **eslint-plugin-react-refresh** ^0.4.24 - ESLint rules for React Fast Refresh
- **globals** ^16.5.0 - Global variable definitions
- **typescript** ~5.9.3 - TypeScript compiler
- **typescript-eslint** ^8.46.4 - ESLint parser and plugin for TypeScript
- **vite** ^7.2.4 - Build tool and dev server

</details>

<details>
<summary><b>Backend Runtime Dependencies</b></summary>

![fastapi](https://img.shields.io/pypi/v/fastapi?style=for-the-badge&label=fastapi&color=009688)
![torch](https://img.shields.io/pypi/v/torch?style=for-the-badge&label=torch&color=EE4C2C)
![monai](https://img.shields.io/pypi/v/monai?style=for-the-badge&label=monai&color=00758F)
![numpy](https://img.shields.io/pypi/v/numpy?style=for-the-badge&label=numpy&color=013243)
![scikit-image](https://img.shields.io/pypi/v/scikit-image?style=for-the-badge&label=scikit-image&color=013220)
![opencv-python](https://img.shields.io/pypi/v/opencv-python?style=for-the-badge&label=opencv-python&color=5C3EE8)
![pillow](https://img.shields.io/pypi/v/pillow?style=for-the-badge&label=pillow&color=FFD43B)

- **fastapi** 0.115.5 - Modern web framework for building APIs
- **uvicorn[standard]** 0.32.1 - ASGI server for FastAPI
- **pydantic** 2.10.3 - Data validation using Python type hints
- **pydantic-settings** 2.6.1 - Settings management with Pydantic
- **python-multipart** 0.0.20 - Multipart form data parser
- **numpy** 1.26.4 - Fundamental package for scientific computing
- **scipy** 1.14.1 - Scientific computing and technical computing
- **scikit-image** 0.24.0 - Image processing algorithms
- **scikit-learn** 1.5.2 - Machine learning library
- **opencv-python** 4.10.0.84 - Computer vision library
- **pillow** 11.0.0 - Python Imaging Library
- **torch** 2.6.0 - PyTorch deep learning framework (security patched)
- **torchvision** 0.21.0 - PyTorch vision library
- **timm** 1.0.12 - PyTorch Image Models (pretrained models)
- **monai** 1.5.1 - Medical imaging AI framework (security patched)
- **matplotlib** 3.9.3 - Plotting library
- **albumentations** 1.4.23 - Image augmentation library
- **tqdm** 4.67.1 - Progress bar library
- **python-dotenv** 1.0.1 - Environment variable management
- **h5py** 3.11.0 - HDF5 file format interface
- **nibabel** 5.3.2 - Neuroimaging file I/O
- **SimpleITK** 2.3.1 - Medical image processing toolkit
- **PyYAML** 6.0.2 - YAML parser and emitter

</details>

---

## 🤝 Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guidelines before submitting pull requests.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🛡 Security

Security is a top priority. If you discover a security vulnerability, please follow our responsible disclosure guidelines in [SECURITY.md](SECURITY.md).

---

## 📏 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) to understand our community standards.

---

## Citation

If you use this project, please cite:
```
@software{btsc_unet_vit,
  title = {BTSC-UNet-ViT: Brain Tumor Segmentation and Classification},
  author = {BTSC-UNet-ViT Team},
  year = {2024}
}
```

---

## Contact

For questions or issues, please open a GitHub issue.

---

## Acknowledgments

- BraTS dataset for segmentation training
- timm library for pretrained ViT models
- MONAI for medical imaging utilities

---

<p align="center">Made with ❤️ by <a href="https://github.com/H0NEYP0T-466">H0NEYP0T-466</a></p>
