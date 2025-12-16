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
    
    Returns brain mask, brain-extracted image, and overlay visualization.
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
        
        # Preprocess first
        preprocessed = preprocess_pipeline(image, image_id=image_id)
        normalized = preprocessed['normalized']
        
        logger.info("Passing preprocessed image to Brain UNet", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'brain_segment'
        })
        
        # Segment brain using Brain UNet model
        brain_unet = get_brain_unet_inference()
        
        brain_segmentation = brain_unet.segment_brain(normalized, image_id=image_id)
        
        # Calculate mask statistics
        brain_area_pct = (np.sum(brain_segmentation['mask'] > 0) / brain_segmentation['mask'].size) * 100
        
        # Save artifacts
        mask_url = storage.save_artifact(brain_segmentation['mask'], image_id, 'brain_mask')
        overlay_url = storage.save_artifact(brain_segmentation['overlay'], image_id, 'brain_overlay')
        brain_extracted_url = storage.save_artifact(brain_segmentation['brain_extracted'], image_id, 'brain_extracted')
        
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
