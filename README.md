<h1 align="center">BTSC-UNet-ViT</h1>

<p align="center">
  <strong>Brain Tumor Segmentation and Classification using UNet and Vision Transformer (ViT)</strong>
</p>

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

## 📝 Description

Full-stack application for automated brain tumor analysis in MRI images. The pipeline combines deep learning techniques for preprocessing, segmentation, and classification of brain tumors from medical imaging data.

**Key Capabilities:**
- **Preprocessing**: Edge-preserving denoising, contrast enhancement, normalization
- **Segmentation**: UNet-based tumor detection
- **Classification**: ViT-based tumor type classification

**Tumor Classes:**
- No Tumor
- Glioma
- Meningioma
- Pituitary Tumor

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 📖 Backend Docs | [backend/README.md](backend/README.md) |
| 🐛 Issues | [GitHub Issues](https://github.com/H0NEYP0T-466/BTSC-UNet-ViT/issues) |
| 🤝 Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 🛡️ Security | [SECURITY.md](SECURITY.md) |

---

## 📑 Table of Contents

- [Description](#-description)
- [Links](#-links)
- [Installation](#-installation)
- [Usage](#-usage)
- [Features](#-features)
- [Folder Structure](#-folder-structure)
- [API Endpoints](#-api-endpoints)
- [Model Training](#-model-training)
- [Configuration](#-configuration)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)
- [Security](#-security)
- [Code of Conduct](#-code-of-conduct)
- [Tech Stack](#-tech-stack)
- [Dependencies & Packages](#-dependencies--packages)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 🚀 Installation

### Prerequisites

- **Node.js** >= 18.x
- **Python** >= 3.9
- **pip** (Python package manager)
- **npm** (Node package manager)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**API Documentation:** http://localhost:8000/docs

### Frontend Setup

```bash
# Install dependencies
npm install

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# Run development server
npm run dev
```

**Application:** http://localhost:5173

---

## ⚡ Usage

### Architecture

```
Frontend (React + TypeScript) → Backend (FastAPI + Python) → Models (UNet + ViT)
```

### Pipeline Flow

1. User uploads brain MRI image
2. Image preprocessing (edge-preserving denoising, contrast enhancement, normalization)
3. UNet segments tumor region
4. ViT classifies tumor type
5. Results displayed with confidence scores

### Quick Start

```bash
# Terminal 1: Start Backend
cd backend && uvicorn app.main:app --reload --port 8000

# Terminal 2: Start Frontend
npm run dev
```

Navigate to http://localhost:5173 and upload a brain MRI image to analyze.

---

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

---

## 📂 Folder Structure

```
BTSC-UNet-ViT/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Configuration
│   │   ├── logging_config.py    # Logging setup
│   │   ├── routers/             # API endpoints
│   │   │   ├── health.py
│   │   │   ├── segmentation.py
│   │   │   ├── classification.py
│   │   │   └── preprocessing.py
│   │   ├── models/              # UNet & ViT models
│   │   │   ├── unet/            # UNet model
│   │   │   ├── unet_tumor/      # UNet tumor model
│   │   │   └── vit/             # Vision Transformer
│   │   ├── services/            # Business logic
│   │   │   ├── storage_service.py
│   │   │   ├── dataset_service.py
│   │   │   └── pipeline_service.py
│   │   ├── utils/               # Preprocessing, imaging
│   │   │   ├── imaging.py
│   │   │   ├── preprocessing.py
│   │   │   ├── metrics.py
│   │   │   └── logger.py
│   │   ├── schemas/             # Pydantic models
│   │   └── resources/           # Model checkpoints
│   ├── tests/                   # Tests
│   ├── requirements.txt
│   └── README.md
├── src/                         # Frontend
│   ├── components/              # React components
│   │   ├── Header/
│   │   ├── Footer/
│   │   ├── UploadCard/
│   │   ├── ImagePreview/
│   │   ├── PredictionCard/
│   │   ├── PreprocessedGallery/
│   │   ├── SegmentationOverlay/
│   │   └── ErrorBoundary.tsx
│   ├── pages/                   # Page layouts
│   │   └── HomePage.tsx
│   ├── services/                # API client
│   │   ├── api.ts
│   │   └── types.ts
│   └── theme/                   # CSS variables & styles
│       ├── variables.css
│       └── global.css
├── public/                      # Static assets
├── .github/                     # GitHub templates
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── package.json
├── vite.config.ts
├── tsconfig.json
├── eslint.config.js
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── LICENSE
└── README.md
```

---

## 🌐 API Endpoints

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

---

## 🧠 Model Training

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

---

## ⚙️ Configuration

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

---

## 🔧 Development

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

### Build for Production

**Backend:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Frontend:**
```bash
npm run build
# Serve dist/ with nginx or any static server
```

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report bugs, and suggest features.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🛡️ Security

For security concerns, please review our [Security Policy](SECURITY.md). Report vulnerabilities responsibly through GitHub Security Advisories.

**⚠️ Security Note:** PyTorch and MONAI have been updated to address critical vulnerabilities. See [SECURITY.md](SECURITY.md) for details.

---

## 📏 Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand the standards of behavior expected when participating in this project.

---

## 🛠️ Tech Stack

### Languages

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

### Frameworks & Libraries

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)

### Machine Learning & AI

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-00A7B5?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAEBSURBVCiRhZIxTsMwFIa/l1SkDkgssNERFhZuwMgBGLkBI+IATJyAMbC0M3dgYOQAjIwoSBUdkJCY8kP+OHaSuuKT/uw8//9zzrMDAAJQ0Z9a4FYJvFWBN0rgpQy8VgIvZeCNEngpAy+VwEsZ+FgGXsrAKyXwUgZeKIFnMvBcCTyVgadK4IkMPJaBx0rgkQw8lIEHMvBACdyXgXtK4K4MXJeB6zJwTQauyMAFGTgnA2dl4IwMnJaB0zJwSgZOysBxGTgmA0dlYEAGBmRgQAb2y8AeGdgtAztloFMGOmSgXQbaZKBVBlpkoFkGmmSgUQYaZKBeBupkoFYGamSgWgb+s8EfEq51mxH8qRIAAAAASUVORK5CYII=&logoColor=white)
![timm](https://img.shields.io/badge/timm-FF6F00?style=for-the-badge&logo=python&logoColor=white)
![scikit--image](https://img.shields.io/badge/scikit--image-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

### DevOps / CI / Tools

![ESLint](https://img.shields.io/badge/ESLint-4B32C3?style=for-the-badge&logo=eslint&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

## 📦 Dependencies & Packages

<details>
<summary><strong>🔧 Runtime Dependencies (Frontend - npm)</strong></summary>

[![axios](https://img.shields.io/npm/v/axios?style=for-the-badge&label=axios&color=5A29E4)](https://www.npmjs.com/package/axios)
[![react](https://img.shields.io/npm/v/react?style=for-the-badge&label=react&color=61DAFB)](https://www.npmjs.com/package/react)
[![react-dom](https://img.shields.io/npm/v/react-dom?style=for-the-badge&label=react-dom&color=61DAFB)](https://www.npmjs.com/package/react-dom)

</details>

<details>
<summary><strong>🛠️ Dev Dependencies (Frontend - npm)</strong></summary>

[![@eslint/js](https://img.shields.io/npm/v/@eslint/js?style=for-the-badge&label=@eslint/js&color=4B32C3)](https://www.npmjs.com/package/@eslint/js)
[![@types/node](https://img.shields.io/npm/v/@types/node?style=for-the-badge&label=@types/node&color=3178C6)](https://www.npmjs.com/package/@types/node)
[![@types/react](https://img.shields.io/npm/v/@types/react?style=for-the-badge&label=@types/react&color=3178C6)](https://www.npmjs.com/package/@types/react)
[![@types/react-dom](https://img.shields.io/npm/v/@types/react-dom?style=for-the-badge&label=@types/react-dom&color=3178C6)](https://www.npmjs.com/package/@types/react-dom)
[![@vitejs/plugin-react](https://img.shields.io/npm/v/@vitejs/plugin-react?style=for-the-badge&label=@vitejs/plugin-react&color=646CFF)](https://www.npmjs.com/package/@vitejs/plugin-react)
[![eslint](https://img.shields.io/npm/v/eslint?style=for-the-badge&label=eslint&color=4B32C3)](https://www.npmjs.com/package/eslint)
[![eslint-plugin-react-hooks](https://img.shields.io/npm/v/eslint-plugin-react-hooks?style=for-the-badge&label=eslint-plugin-react-hooks&color=4B32C3)](https://www.npmjs.com/package/eslint-plugin-react-hooks)
[![eslint-plugin-react-refresh](https://img.shields.io/npm/v/eslint-plugin-react-refresh?style=for-the-badge&label=eslint-plugin-react-refresh&color=4B32C3)](https://www.npmjs.com/package/eslint-plugin-react-refresh)
[![globals](https://img.shields.io/npm/v/globals?style=for-the-badge&label=globals&color=F7DF1E)](https://www.npmjs.com/package/globals)
[![typescript](https://img.shields.io/npm/v/typescript?style=for-the-badge&label=typescript&color=3178C6)](https://www.npmjs.com/package/typescript)
[![typescript-eslint](https://img.shields.io/npm/v/typescript-eslint?style=for-the-badge&label=typescript-eslint&color=3178C6)](https://www.npmjs.com/package/typescript-eslint)
[![vite](https://img.shields.io/npm/v/vite?style=for-the-badge&label=vite&color=646CFF)](https://www.npmjs.com/package/vite)

</details>

<details>
<summary><strong>🐍 Runtime Dependencies (Backend - pip)</strong></summary>

[![fastapi](https://img.shields.io/pypi/v/fastapi?style=for-the-badge&label=fastapi&color=009688)](https://pypi.org/project/fastapi/)
[![uvicorn](https://img.shields.io/pypi/v/uvicorn?style=for-the-badge&label=uvicorn&color=499848)](https://pypi.org/project/uvicorn/)
[![pydantic](https://img.shields.io/pypi/v/pydantic?style=for-the-badge&label=pydantic&color=E92063)](https://pypi.org/project/pydantic/)
[![pydantic-settings](https://img.shields.io/pypi/v/pydantic-settings?style=for-the-badge&label=pydantic-settings&color=E92063)](https://pypi.org/project/pydantic-settings/)
[![python-multipart](https://img.shields.io/pypi/v/python-multipart?style=for-the-badge&label=python-multipart&color=3776AB)](https://pypi.org/project/python-multipart/)
[![numpy](https://img.shields.io/pypi/v/numpy?style=for-the-badge&label=numpy&color=013243)](https://pypi.org/project/numpy/)
[![scipy](https://img.shields.io/pypi/v/scipy?style=for-the-badge&label=scipy&color=8CAAE6)](https://pypi.org/project/scipy/)
[![scikit-image](https://img.shields.io/pypi/v/scikit-image?style=for-the-badge&label=scikit-image&color=F7931E)](https://pypi.org/project/scikit-image/)
[![scikit-learn](https://img.shields.io/pypi/v/scikit-learn?style=for-the-badge&label=scikit-learn&color=F7931E)](https://pypi.org/project/scikit-learn/)
[![opencv-python](https://img.shields.io/pypi/v/opencv-python?style=for-the-badge&label=opencv-python&color=5C3EE8)](https://pypi.org/project/opencv-python/)
[![pillow](https://img.shields.io/pypi/v/pillow?style=for-the-badge&label=pillow&color=3776AB)](https://pypi.org/project/pillow/)
[![torch](https://img.shields.io/pypi/v/torch?style=for-the-badge&label=torch&color=EE4C2C)](https://pypi.org/project/torch/)
[![torchvision](https://img.shields.io/pypi/v/torchvision?style=for-the-badge&label=torchvision&color=EE4C2C)](https://pypi.org/project/torchvision/)
[![timm](https://img.shields.io/pypi/v/timm?style=for-the-badge&label=timm&color=FF6F00)](https://pypi.org/project/timm/)
[![monai](https://img.shields.io/pypi/v/monai?style=for-the-badge&label=monai&color=00A7B5)](https://pypi.org/project/monai/)
[![matplotlib](https://img.shields.io/pypi/v/matplotlib?style=for-the-badge&label=matplotlib&color=11557C)](https://pypi.org/project/matplotlib/)
[![albumentations](https://img.shields.io/pypi/v/albumentations?style=for-the-badge&label=albumentations&color=00C4CC)](https://pypi.org/project/albumentations/)
[![tqdm](https://img.shields.io/pypi/v/tqdm?style=for-the-badge&label=tqdm&color=FFC107)](https://pypi.org/project/tqdm/)
[![python-dotenv](https://img.shields.io/pypi/v/python-dotenv?style=for-the-badge&label=python-dotenv&color=ECD53F)](https://pypi.org/project/python-dotenv/)
[![h5py](https://img.shields.io/pypi/v/h5py?style=for-the-badge&label=h5py&color=3776AB)](https://pypi.org/project/h5py/)
[![nibabel](https://img.shields.io/pypi/v/nibabel?style=for-the-badge&label=nibabel&color=3776AB)](https://pypi.org/project/nibabel/)
[![SimpleITK](https://img.shields.io/pypi/v/SimpleITK?style=for-the-badge&label=SimpleITK&color=3776AB)](https://pypi.org/project/SimpleITK/)
[![PyYAML](https://img.shields.io/pypi/v/PyYAML?style=for-the-badge&label=PyYAML&color=3776AB)](https://pypi.org/project/PyYAML/)

</details>

---

## 📚 Citation

If you use this project, please cite:

```bibtex
@software{btsc_unet_vit,
  title = {BTSC-UNet-ViT: Brain Tumor Segmentation and Classification},
  author = {BTSC-UNet-ViT Team},
  year = {2024},
  url = {https://github.com/H0NEYP0T-466/BTSC-UNet-ViT}
}
```

---

## 🙏 Acknowledgments

- [BraTS Dataset](https://www.med.upenn.edu/cbica/brats/) for segmentation training
- [timm](https://github.com/huggingface/pytorch-image-models) library for pretrained ViT models
- [MONAI](https://monai.io/) for medical imaging utilities

---

<p align="center">Made with ❤️ by H0NEYP0T-466</p>
