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
    
    Pipeline with auto-detection:
    - Converts to grayscale
    - Detects and removes Salt & Pepper noise (median filter)
    - Detects and removes Gaussian noise (NLM denoising)
    - Detects and removes Speckle noise (wavelet denoising)
    - Detects and corrects Patient Motion Artifacts (PMA)
    - Detects and corrects blur (Wiener/USM deconvolution)
    - Enhances contrast (CLAHE with conservative parameters)
    - Sharpens edges (noise-aware unsharp masking)
    
    Only applies corrections when issues are detected to avoid over-processing.
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
            'sharpen_amount': 0.8,  # Conservative sharpening
            'sharpen_threshold': 0.02,  # Higher threshold to avoid noise
        }
        
        preprocessed = run_preprocessing(
            image, 
            opts=config, 
            image_id=image_id
        )
        
        # Get detection results for response
        grayscale = to_grayscale(image, image_id=image_id)
        noise_info = detect_noise_type(grayscale, image_id=image_id)
        blur_info = detect_blur(grayscale, image_id=image_id)
        motion_info = detect_motion(grayscale, image_id=image_id)
        
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
            salt_pepper_cleaned_url=urls['salt_pepper_cleaned'],
            gaussian_denoised_url=urls['gaussian_denoised'],
            speckle_denoised_url=urls['speckle_denoised'],
            pma_corrected_url=urls['pma_corrected'],
            deblurred_url=urls['deblurred'],
            contrast_enhanced_url=urls['contrast_enhanced'],
            sharpened_url=urls['sharpened'],
            noise_detected=noise_info['type'],
            blur_detected=blur_info['is_blurred'],
            motion_detected=motion_info['has_motion'],
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
