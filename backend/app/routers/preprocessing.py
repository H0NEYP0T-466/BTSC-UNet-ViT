"""
Preprocessing router.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from app.schemas.responses import PreprocessResponse, LogContext
from app.utils.preprocessing import run_preprocessing, to_grayscale
from app.utils.btsc_preprocess import detect_noise_type, detect_blur, detect_motion
from app.utils.imaging import bytes_to_numpy, resize_image
from app.services.storage_service import get_storage_service
from app.config import settings
from app.utils.logger import get_logger
import time

logger = get_logger(__name__)

router = APIRouter(prefix="/preprocess", tags=["preprocessing"])


@router.post("", response_model=PreprocessResponse)
async def preprocess_image(
    file: UploadFile = File(...),
    skip_preprocessing: Optional[bool] = Form(False)
):
    """
    Preprocess uploaded image with intelligent quality-focused enhancement.
    
    Pipeline with auto-detection:
    - Resizes image if larger than MAX_IMAGE_SIZE (maintains aspect ratio)
    - Converts to grayscale
    - Detects and removes Salt & Pepper noise (median filter)
    - Detects and removes Gaussian noise (NLM denoising)
    - Detects and removes Speckle noise (wavelet denoising)
    - Detects and corrects Patient Motion Artifacts (PMA)
    - Detects and corrects blur (Wiener/USM deconvolution)
    - Enhances contrast (CLAHE with conservative parameters)
    - Sharpens edges (noise-aware unsharp masking)
    
    Only applies corrections when issues are detected to avoid over-processing.
    
    Args:
        file: Uploaded image file
        skip_preprocessing: If True, only converts to grayscale and skips all enhancement stages
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
        
        # Resize if needed
        image = resize_image(image, max_size=settings.MAX_IMAGE_SIZE)
        
        # Get storage service
        storage = get_storage_service()
        image_id = storage.generate_image_id()
        
        logger.info(f"Starting preprocessing (skip={skip_preprocessing})", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'preprocess'
        })
        
        # Save original (after resize)
        original_url = storage.save_upload(image, image_id)
        
        if skip_preprocessing:
            # Skip all preprocessing, only convert to grayscale
            logger.info("Skipping preprocessing stages (skip_preprocessing=True)", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'preprocess_skip'
            })
            
            grayscale = to_grayscale(image, image_id=image_id)
            
            # Save only grayscale, use it for all stages
            urls = {}
            for stage_name in ['grayscale', 'salt_pepper_cleaned', 'gaussian_denoised', 
                              'speckle_denoised', 'pma_corrected', 'deblurred', 
                              'contrast_enhanced', 'sharpened']:
                rel_path = storage.save_artifact(grayscale, image_id, stage_name)
                urls[stage_name] = storage.get_artifact_url(rel_path)
            
            duration = time.time() - start_time
            
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
                noise_detected='none',
                blur_detected=False,
                motion_detected=False,
                log_context=LogContext(
                    image_id=image_id,
                    duration=duration,
                    stage='preprocess_skipped'
                )
            )
        
        # Preprocess with auto-detection
        config = {
            'auto': True,
            'clahe_clip_limit': settings.CLAHE_CLIP_LIMIT,
            'clahe_tile_grid': settings.CLAHE_TILE_GRID_SIZE,
            'sharpen_amount': settings.UNSHARP_AMOUNT,
            'sharpen_threshold': settings.SHARPEN_THRESHOLD,
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
