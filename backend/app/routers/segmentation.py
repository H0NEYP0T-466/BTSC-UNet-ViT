"""
Segmentation router.
"""
import time
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.responses import SegmentResponse, LogContext
from app.utils.preprocessing import preprocess_pipeline
from app.utils.imaging import bytes_to_numpy
from app.models.unet.infer_unet import get_unet_inference
from app.services.storage_service import get_storage_service
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/segment", tags=["segmentation"])


@router.post("", response_model=SegmentResponse)
async def segment_image(file: UploadFile = File(...)):
    """
    Segment brain tumor in uploaded image using UNet.
    
    Returns segmentation mask, overlay, and cropped tumor region.
    """
    start_time = time.time()
    
    logger.info(f"Received segmentation request: filename={file.filename}", extra={
        'image_id': None,
        'path': file.filename,
        'stage': 'segment_request'
    })
    
    try:
        # Read image bytes
        contents = await file.read()
        image = bytes_to_numpy(contents)
        
        # Get services
        storage = get_storage_service()
        image_id = storage.generate_image_id()
        
        logger.info("Starting segmentation pipeline", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'segment'
        })
        
        # Preprocess first
        preprocessed = preprocess_pipeline(image, image_id=image_id)
        normalized = preprocessed['normalized']
        
        logger.info("Passing preprocessed image to UNet", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'segment'
        })
        
        # Segment
        unet = get_unet_inference()
        segmentation = unet.segment_image(normalized, image_id=image_id)
        
        # Calculate mask statistics
        mask_area_pct = (np.sum(segmentation['mask'] > 0) / segmentation['mask'].size) * 100
        
        # Save artifacts
        mask_url = storage.save_artifact(segmentation['mask'], image_id, 'seg_mask')
        overlay_url = storage.save_artifact(segmentation['overlay'], image_id, 'seg_overlay')
        segmented_url = storage.save_artifact(segmentation['segmented'], image_id, 'seg_segmented')
        
        duration = time.time() - start_time
        
        logger.info(
            f"Segmentation completed in {duration:.3f}s, mask_area_pct={mask_area_pct:.2f}%",
            extra={
                'image_id': image_id,
                'path': None,
                'stage': 'segment_complete'
            }
        )
        
        return SegmentResponse(
            image_id=image_id,
            mask_url=storage.get_artifact_url(mask_url),
            overlay_url=storage.get_artifact_url(overlay_url),
            segmented_url=storage.get_artifact_url(segmented_url),
            mask_area_pct=float(mask_area_pct),
            log_context=LogContext(
                image_id=image_id,
                duration=duration,
                stage='segment_complete'
            )
        )
    
    except Exception as e:
        logger.error(f"Segmentation failed: {e}", extra={
            'image_id': None,
            'path': file.filename,
            'stage': 'segment_error'
        })
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
