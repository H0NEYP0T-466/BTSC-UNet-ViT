"""
Brain segmentation router.
"""
import time
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.responses import BrainSegmentResponse, LogContext
from app.utils.preprocessing import preprocess_pipeline
from app.utils.imaging import bytes_to_numpy
from app.models.brain_unet.infer_unet import get_brain_unet_inference
from app.services.storage_service import get_storage_service
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/segment-brain", tags=["brain-segmentation"])


@router.post("", response_model=BrainSegmentResponse)
async def segment_brain(file: UploadFile = File(...)):
    """
    Segment brain tissue from uploaded MRI image using Brain UNet.
    
    Returns brain mask, brain-extracted image, overlay visualization,
    and optionally preprocessing stages and candidate masks.
    """
    start_time = time.time()
    
    logger.info(f"Received brain segmentation request: filename={file.filename}", extra={
        'image_id': None,
        'path': file.filename,
        'stage': 'brain_segment_request'
    })
    
    try:
        # Read image bytes
        contents = await file.read()
        image = bytes_to_numpy(contents)
        
        # Get services
        storage = get_storage_service()
        image_id = storage.generate_image_id()
        
        logger.info("Starting brain segmentation pipeline", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'brain_segment'
        })
        
        # Preprocess first (original preprocessing pipeline)
        preprocessed = preprocess_pipeline(image, image_id=image_id)
        normalized = preprocessed['normalized']
        
        logger.info("Passing preprocessed image to Brain UNet with advanced preprocessing", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'brain_segment'
        })
        
        # Segment brain using Brain UNet model with advanced preprocessing
        brain_unet = get_brain_unet_inference()
        
        brain_segmentation = brain_unet.segment_brain(
            normalized, 
            image_id=image_id,
            save_intermediates=True
        )
        
        # Calculate mask statistics
        brain_area_pct = (np.sum(brain_segmentation['mask'] > 0) / brain_segmentation['mask'].size) * 100
        
        # Save main artifacts
        mask_url = storage.save_artifact(brain_segmentation['mask'], image_id, 'brain_mask')
        overlay_url = storage.save_artifact(brain_segmentation['overlay'], image_id, 'brain_overlay')
        brain_extracted_url = storage.save_artifact(brain_segmentation['brain_extracted'], image_id, 'brain_extracted')
        
        # Save preprocessing stages if available
        preprocessing_stages_urls = None
        if 'preprocessing' in brain_segmentation:
            preprocessing_stages_urls = {}
            for stage_name, stage_img in brain_segmentation['preprocessing'].items():
                stage_url = storage.save_artifact(stage_img, image_id, f'preproc_{stage_name}')
                preprocessing_stages_urls[stage_name] = storage.get_artifact_url(stage_url)
        
        # Save candidate masks if available
        candidate_masks_urls = None
        if 'candidates' in brain_segmentation and brain_segmentation['candidates']:
            candidate_masks_urls = {}
            for mask_name, mask_img in brain_segmentation['candidates'].items():
                mask_url_candidate = storage.save_artifact(mask_img, image_id, f'candidate_{mask_name}')
                candidate_masks_urls[mask_name] = storage.get_artifact_url(mask_url_candidate)
        
        # Save candidate mask overlays if available (shows each algorithm's mask applied on original image)
        candidate_overlays_urls = None
        if 'candidate_overlays' in brain_segmentation and brain_segmentation['candidate_overlays']:
            candidate_overlays_urls = {}
            for overlay_name, overlay_img in brain_segmentation['candidate_overlays'].items():
                overlay_url_candidate = storage.save_artifact(overlay_img, image_id, f'candidate_overlay_{overlay_name}')
                candidate_overlays_urls[overlay_name] = storage.get_artifact_url(overlay_url_candidate)
        
        duration = time.time() - start_time
        
        logger.info(
            f"Brain segmentation completed in {duration:.3f}s, brain_area_pct={brain_area_pct:.2f}%",
            extra={
                'image_id': image_id,
                'path': None,
                'stage': 'brain_segment_complete'
            }
        )
        
        return BrainSegmentResponse(
            image_id=image_id,
            mask_url=storage.get_artifact_url(mask_url),
            overlay_url=storage.get_artifact_url(overlay_url),
            brain_extracted_url=storage.get_artifact_url(brain_extracted_url),
            brain_area_pct=float(brain_area_pct),
            preprocessing_stages=preprocessing_stages_urls,
            candidate_masks=candidate_masks_urls,
            candidate_overlays=candidate_overlays_urls,
            used_fallback=brain_segmentation.get('used_fallback', False),
            fallback_method=brain_segmentation.get('fallback_method'),
            log_context=LogContext(
                image_id=image_id,
                duration=duration,
                stage='brain_segment_complete'
            )
        )
    
    except Exception as e:
        logger.error(f"Brain segmentation failed: {e}", extra={
            'image_id': None,
            'path': file.filename,
            'stage': 'brain_segment_error'
        })
        raise HTTPException(status_code=500, detail=f"Brain segmentation failed: {str(e)}")
