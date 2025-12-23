"""
Preprocessing router.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.responses import PreprocessResponse, LogContext
from app.utils.preprocessing import run_preprocessing, to_grayscale
from app.utils.btsc_preprocess import detect_noise_type, detect_blur, detect_motion
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
    Preprocess uploaded image with intelligent quality-focused enhancement.
    
    Pipeline (5 stages):
    - Converts to grayscale
    - Conservative denoising (NLM with reduced parameters, no speckle)
    - Minimal motion reduction (bilateral filter, NO PMA correction)
    - Contrast enhancement (CLAHE with conservative clip limit of 1.2)
    - Conservative sharpening (reduced parameters to avoid white noise)
    
    Note: PMA correction and deblurring are SKIPPED to avoid over-smoothing.
    Speckle noise is NOT applied in inference (augmentation-only).
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
        
        logger.info("Starting intelligent preprocessing", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'preprocess'
        })
        
        # Save original
        original_url = storage.save_upload(image, image_id)
        
        # Preprocess with auto-detection
        config = {
            'auto': True,
            'clahe_clip_limit': settings.CLAHE_CLIP_LIMIT,
            'clahe_tile_grid': settings.CLAHE_TILE_GRID_SIZE,
            'sharpen_amount': settings.UNSHARP_AMOUNT,
            'sharpen_radius': settings.UNSHARP_RADIUS,
            'sharpen_threshold': settings.SHARPEN_THRESHOLD,
            'skip_pma': settings.SKIP_PMA_CORRECTION,
            'skip_deblur': settings.SKIP_DEBLUR,
        }
        
        preprocessed = run_preprocessing(
            image, 
            opts=config, 
            image_id=image_id
        )
        
        # Get detection results for response (informational only)
        grayscale = to_grayscale(image, image_id=image_id)
        noise_info = detect_noise_type(grayscale, image_id=image_id)
        
        # Save the 5 required stages only
        urls = {}
        required_stages = ['grayscale', 'denoised', 'motion_reduced', 'contrast_enhanced', 'sharpened']
        for stage_name in required_stages:
            if stage_name in preprocessed:
                rel_path = storage.save_artifact(preprocessed[stage_name], image_id, stage_name)
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
            denoised_url=urls['denoised'],
            motion_reduced_url=urls['motion_reduced'],
            contrast_enhanced_url=urls['contrast_enhanced'],
            sharpened_url=urls['sharpened'],
            noise_detected=noise_info['type'],
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
