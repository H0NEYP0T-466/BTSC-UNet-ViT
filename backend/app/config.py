"""
Configuration settings for BTSC-UNet-ViT backend. 
Automatically detects Colab environment and adjusts dataset paths. 
"""
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


def detect_environment_base() -> Path:
    """
    Detect if running inside Google Colab. 
    If /content exists, assume Colab and use that as root.
    Else, use current project directory.
    """
    if Path("/content").exists():
        return Path("/content")  # Colab root
    return Path(__file__).resolve().parent.parent  # backend root


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "BTSC-UNet-ViT"
    APP_VERSION:  str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # API
    API_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = ["*"]

    # Environment-aware base directory
    BASE_DIR: Path = detect_environment_base()

    # Resource directories
    RESOURCES_DIR: Path = BASE_DIR / "resources"
    CHECKPOINTS_DIR: Path = RESOURCES_DIR / "checkpoints"
    UPLOADS_DIR:  Path = RESOURCES_DIR / "uploads"
    ARTIFACTS_DIR: Path = RESOURCES_DIR / "artifacts"

    # Dataset roots (adjusted for Colab local folder)
    DATASET_ROOT: Path = BASE_DIR / "UNet_Dataset"
    SEGMENTED_DATASET_ROOT: Path = DATASET_ROOT / "Vit_Dataset"
    BRATS_ROOT: Path = DATASET_ROOT  # your UNet dataset folder

    # Model checkpoint directories
    CHECKPOINTS_UNET: Path = CHECKPOINTS_DIR / "unet"
    CHECKPOINTS_BRAIN_UNET: Path = CHECKPOINTS_DIR / "brain_unet"
    CHECKPOINTS_VIT: Path = CHECKPOINTS_DIR / "vit"
    UNET_CHECKPOINT_NAME: str = "unet_best.pth"
    BRAIN_UNET_CHECKPOINT_NAME: str = "brain_unet_best.pth"
    VIT_CHECKPOINT_NAME: str = "vit_best.pth"

    # UNet settings
    UNET_IN_CHANNELS: int = 4  # ✅ FIXED - BraTS has 4 modalities (T1, T1ce, T2, FLAIR)
    UNET_OUT_CHANNELS: int = 1
    UNET_CHANNELS:  tuple = (16, 32, 64, 128, 256)
    UNET_STRIDES: tuple = (2, 2, 2, 2)

    # Vision Transformers
    VIT_MODEL_NAME: str = "vit_base_patch16_224"
    VIT_NUM_CLASSES:  int = 4
    VIT_IMAGE_SIZE: int = 224
    VIT_CLASS_NAMES: List[str] = ["no_tumor", "glioma", "meningioma", "pituitary"]

    # Training settings
    BATCH_SIZE: int = 16  # ✅ Increased from 8 to 16 for better GPU utilization (15GB GPU RAM)
    NUM_EPOCHS: int = 50  # ✅ Increased from 20 to 50 for better convergence with extreme class imbalance
    LEARNING_RATE: float = 1e-4
    NUM_WORKERS: int = 2  # Reduced from 4 to 2 (per system warning)
    SEED: int = 42

    # Preprocessing parameters
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple = (8, 8)
    MEDIAN_KERNEL_SIZE: int = 3
    NLM_H: int = 8  # Reduced from 10 to 8 for less blur
    UNSHARP_RADIUS: float = 1.5  # Increased from 1.0 to 1.5 for better sharpness
    UNSHARP_AMOUNT: float = 1.5  # Increased from 1.0 to 1.5 for better detail
    MOTION_PRESERVE_DETAIL: bool = True  # Use minimal edge-preserving bilateral filter
    
    # Segmentation post-processing parameters
    SEGMENTATION_MIN_AREA: int = 100  # Minimum area for connected components (pixels)
    SEGMENTATION_THRESHOLD: float = 0.5  # Threshold for binary segmentation

    class Config:
        env_file = ".env"
        case_sensitive = True


# Initialize settings
settings = Settings()

# Ensure required directories exist
settings.RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
settings.CHECKPOINTS_UNET.mkdir(parents=True, exist_ok=True)
settings.CHECKPOINTS_BRAIN_UNET.mkdir(parents=True, exist_ok=True)
settings.CHECKPOINTS_VIT.mkdir(parents=True, exist_ok=True)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)