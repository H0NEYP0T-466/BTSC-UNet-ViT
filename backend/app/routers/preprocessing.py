"""
Preprocessing router.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.schemas.responses import PreprocessResponse, LogContext
from app.utils.preprocessing import preprocess_pipeline, run_preprocessing
from app.utils.imaging import bytes_to_numpy
from app.services.storage_service import get_storage_service
from app.config import settings
from app.utils.logger import get_logger
from typing import Optional
import time

logger = get_logger(__name__)

router = APIRouter(prefix="/preprocess", tags=["preprocessing"])


@router.post("", response_model=PreprocessResponse)
async def preprocess_image(
    file: UploadFile = File(...),
    auto: bool = Query(True, description="Auto-detect quality issues"),
    noise_type: Optional[str] = Query(None, description="Force noise type: salt_pepper, gaussian, speckle, none"),
    blur_type: Optional[str] = Query(None, description="Force blur type: gaussian, bilateral, median, none"),
    motion: Optional[bool] = Query(None, description="Force motion correction on/off"),
    use_comprehensive: bool = Query(True, description="Use comprehensive 8-stage pipeline (default True)")
):
    """
    Preprocess uploaded image with quality-focused enhancement.
    
    Two pipeline modes available:
    
    **Comprehensive Pipeline (use_comprehensive=True, default):**
    1. Grayscale conversion
    2. Salt & Pepper noise removal (adaptive median)
    3. Gaussian denoising (fast NLM)
    4. Speckle denoising (wavelet BayesShrink)
    5. Motion artifact correction (RL/Wiener)
    6. Deblurring (Wiener/USM based on blur type)
    7. Contrast enhancement (CLAHE)
    8. Noise-aware sharpening (USM with detail mask)
    
    **Legacy Pipeline (use_comprehensive=False):**
    1. Grayscale
    2. Denoised (NLM)
    3. Motion reduced (bilateral)
    4. Contrast enhanced (CLAHE)
    5. Sharpened (USM)
    6. Normalized (z-score)
    
    **Quality Detection (when auto=True):**
    - Noise type: impulse fraction, sigma estimate, log-domain variance
    - Blur: variance of Laplacian (Tenengrad)
    - Motion: spectral streak score
    
    **Manual Overrides (when auto=False):**
    - Set noise_type, blur_type, motion to force specific processing
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
        
        if use_comprehensive:
            # Use comprehensive 8-stage pipeline
            opts = {
                'auto': auto,
                'noise_type': noise_type,
                'blur_type': blur_type,
                'motion': motion,
            }
            preprocessed = run_preprocessing(image, opts=opts, image_id=image_id)
            
            # Save all stages
            urls = {}
            for stage_name, stage_image in preprocessed.items():
                rel_path = storage.save_artifact(stage_image, image_id, stage_name)
                urls[stage_name] = storage.get_artifact_url(rel_path)
            
            duration = time.time() - start_time
            
            logger.info(f"Comprehensive preprocessing completed in {duration:.3f}s", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'preprocess_complete'
            })
            
            # Return comprehensive response with all 8 stages
            return PreprocessResponse(
                image_id=image_id,
                original_url=storage.get_artifact_url(original_url),
                grayscale_url=urls['grayscale'],
                # Map new stages to response fields for backward compatibility
                denoised_url=urls['salt_pepper_cleaned'],
                motion_reduced_url=urls['pma_corrected'],
                contrast_url=urls['contrast_enhanced'],
                sharpened_url=urls['sharpened'],
                normalized_url=urls['sharpened'],  # Use sharpened as final stage
                # Additional stages (if frontend supports extended response)
                salt_pepper_cleaned_url=urls.get('salt_pepper_cleaned'),
                gaussian_denoised_url=urls.get('gaussian_denoised'),
                speckle_denoised_url=urls.get('speckle_denoised'),
                pma_corrected_url=urls.get('pma_corrected'),
                deblurred_url=urls.get('deblurred'),
                contrast_enhanced_url=urls.get('contrast_enhanced'),
                sharpened_url2=urls.get('sharpened'),
                log_context=LogContext(
                    image_id=image_id,
                    duration=duration,
                    stage='preprocess_complete'
                )
            )
        else:
            # Use legacy pipeline
            config = {
                'median_kernel_size': settings.MEDIAN_KERNEL_SIZE,
                'clahe_clip_limit': settings.CLAHE_CLIP_LIMIT,
                'clahe_tile_grid_size': settings.CLAHE_TILE_GRID_SIZE,
                'unsharp_radius': settings.UNSHARP_RADIUS,
                'unsharp_amount': settings.UNSHARP_AMOUNT,
                'normalize_method': 'zscore',
                'use_nlm_denoising': True,
                'nlm_h': settings.NLM_H,
                'preserve_detail': True
            }
            
            preprocessed = preprocess_pipeline(
                image, 
                config=config, 
                image_id=image_id
            )
            
            # Save all stages
            urls = {}
            for stage_name, stage_image in preprocessed.items():
                rel_path = storage.save_artifact(stage_image, image_id, stage_name)
                urls[stage_name] = storage.get_artifact_url(rel_path)
            
            duration = time.time() - start_time
            
            logger.info(f"Legacy preprocessing completed in {duration:.3f}s", extra={
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
