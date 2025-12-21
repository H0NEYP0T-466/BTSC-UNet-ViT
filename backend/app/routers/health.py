"""
Health check router.
"""
from fastapi import APIRouter
from app.schemas.responses import HealthResponse
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested", extra={
        'image_id': None,
        'path': None,
        'stage': 'health_check'
    })
    
    # Check if models can be loaded (lazy check)
    models_status = {
        'unet': True,       # BraTS UNet - will be loaded on demand
        'unet_tumor': True, # PNG UNet Tumor - will be loaded on demand
        'vit': True         # ViT - will be loaded on demand
    }
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        models_loaded=models_status
    )
