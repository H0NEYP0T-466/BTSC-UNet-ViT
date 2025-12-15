"""
Preprocessing router.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.responses import PreprocessResponse, LogContext
from app.utils.preprocessing import preprocess_pipeline
from app.utils.imaging import bytes_to_numpy
from app.services.storage_service import get_storage_service
from app.config import settings
from app.utils.logger import get_logger
import time

logger = get_logger(__name__)

router = APIRouter(prefix="/preprocess", tags=["preprocessing"])


@router.post("", response_model=PreprocessResponse)
async def preprocess_image(file: UploadFile = File(...)):
    """
    Preprocess uploaded image with skull stripping and noise-free contrast enhancement.
    
    Pipeline:
    - Converts to grayscale
    - Skull stripping (HD-BET) - removes non-brain tissue
    - Denoises (before contrast enhancement)
    - Reduces motion artifacts
    - Enhances contrast (CLAHE applied only inside brain mask)
    - Sharpens edges
    - Normalizes intensity (after all enhancements)
    """
    start_time = time.time()
    
    logger.info(f"Received preprocessing request: filename={file.filename}", extra={
        'image_id': None,
        'path': file.filename,
        'stage': 'preprocess_request'
    })
    
    try:
        # Read image bytes
        contents = await file.read()
        image = bytes_to_numpy(contents)
        
        # Get storage service
        storage = get_storage_service()
        image_id = storage.generate_image_id()
        
        logger.info("Starting preprocessing", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'preprocess'
        })
        
        # Save original
        original_url = storage.save_upload(image, image_id)
        
        # Preprocess with skull stripping
        config = {
            'median_kernel_size': settings.MEDIAN_KERNEL_SIZE,
            'clahe_clip_limit': settings.CLAHE_CLIP_LIMIT,
            'clahe_tile_grid_size': settings.CLAHE_TILE_GRID_SIZE,
            'unsharp_radius': settings.UNSHARP_RADIUS,
            'unsharp_amount': settings.UNSHARP_AMOUNT,
            'normalize_method': 'zscore',
            'use_nlm_denoising': True,  # Use Non-Local Means for better denoising
            'nlm_h': settings.NLM_H
        }
        
        preprocessed = preprocess_pipeline(
            image, 
            config=config, 
            image_id=image_id,
            apply_skull_stripping=True
        )
        
        # Save all stages
        urls = {}
        for stage_name, stage_image in preprocessed.items():
            rel_path = storage.save_artifact(stage_image, image_id, stage_name)
            urls[stage_name] = storage.get_artifact_url(rel_path)
        
        duration = time.time() - start_time
        
        logger.info(f"Preprocessing completed in {duration:.3f}s", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'preprocess_complete'
        })
        
        return PreprocessResponse(
            image_id=image_id,
            original_url=storage.get_artifact_url(original_url),
            grayscale_url=urls['grayscale'],
            skull_stripped_url=urls['skull_stripped'],
            brain_mask_url=urls['brain_mask'],
            denoised_url=urls['denoised'],
            motion_reduced_url=urls['motion_reduced'],
            contrast_url=urls['contrast'],
            sharpened_url=urls['sharpened'],
            normalized_url=urls['normalized'],
            log_context=LogContext(
                image_id=image_id,
                duration=duration,
                stage='preprocess_complete'
            )
        )
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", extra={
            'image_id': None,
            'path': file.filename,
            'stage': 'preprocess_error'
        })
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")
