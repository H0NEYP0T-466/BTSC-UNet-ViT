"""
Configuration settings for BTSC-UNet-ViT backend.
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "BTSC-UNet-ViT"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API
    API_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    RESOURCES_DIR: Path = BASE_DIR / "resources"
    CHECKPOINTS_DIR: Path = RESOURCES_DIR / "checkpoints"
    UPLOADS_DIR: Path = RESOURCES_DIR / "uploads"
    ARTIFACTS_DIR: Path = RESOURCES_DIR / "artifacts"
    
    # Dataset paths (configurable via environment variables)
    DATASET_ROOT: Path = BASE_DIR.parent / "dataset"  # backend/dataset
    SEGMENTED_DATASET_ROOT: Path = DATASET_ROOT / "Vit_Dataset"  # backend/dataset/Vit_Dataset
    BRATS_ROOT: Path = DATASET_ROOT / "UNet_Dataset"  # backend/dataset/UNet_Dataset
    
    # Model checkpoints
    CHECKPOINTS_UNET: Path = CHECKPOINTS_DIR / "unet"
    CHECKPOINTS_VIT: Path = CHECKPOINTS_DIR / "vit"
    UNET_CHECKPOINT_NAME: str = "unet_best.pth"
    VIT_CHECKPOINT_NAME: str = "vit_best.pth"
    
    # UNet settings
    UNET_IN_CHANNELS: int = 1
    UNET_OUT_CHANNELS: int = 1
    UNET_CHANNELS: tuple = (16, 32, 64, 128, 256)
    UNET_STRIDES: tuple = (2, 2, 2, 2)
    
    # ViT settings
    VIT_MODEL_NAME: str = "vit_base_patch16_224"
    VIT_NUM_CLASSES: int = 4
    VIT_IMAGE_SIZE: int = 224
    VIT_CLASS_NAMES: List[str] = ["no_tumor", "glioma", "meningioma", "pituitary"]
    
    # Training settings
    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4
    NUM_WORKERS: int = 4
    SEED: int = 42
    
    # Preprocessing settings
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple = (8, 8)
    MEDIAN_KERNEL_SIZE: int = 3
    NLM_H: int = 10
    UNSHARP_RADIUS: float = 1.0
    UNSHARP_AMOUNT: float = 1.0
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure required directories exist
settings.RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
settings.CHECKPOINTS_UNET.mkdir(parents=True, exist_ok=True)
settings.CHECKPOINTS_VIT.mkdir(parents=True, exist_ok=True)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
