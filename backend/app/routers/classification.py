"""
Classification router.
"""
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.responses import ClassifyResponse, LogContext
from app.utils.imaging import bytes_to_numpy
from app.models.vit.infer_vit import get_vit_inference
from app.services.storage_service import get_storage_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/classify", tags=["classification"])


@router.post("", response_model=ClassifyResponse)
async def classify_image(file: UploadFile = File(...)):
    """
    Classify brain tumor type in uploaded (segmented) image using ViT.
    
    Returns predicted class, confidence, and logits.
    Classes: no_tumor, glioma, meningioma, pituitary
    """
    start_time = time.time()
    
    logger.info(f"Received classification request: filename={file.filename}", extra={
        'image_id': None,
        'path': file.filename,
        'stage': 'classify_request'
    })
    
    try:
        # Read image bytes
        contents = await file.read()
        image = bytes_to_numpy(contents)
        
        # Get services
        storage = get_storage_service()
        image_id = storage.generate_image_id()
        
        logger.info("Starting classification", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'classify'
        })
        
        # Classify
        vit = get_vit_inference()
        classification = vit.classify(image, image_id=image_id)
        
        duration = time.time() - start_time
        
        logger.info(
            f"Classification completed in {duration:.3f}s: "
            f"class={classification['class']}, confidence={classification['confidence']:.4f}",
            extra={
                'image_id': image_id,
                'path': None,
                'stage': 'classify_complete'
            }
        )
        
        return ClassifyResponse(
            image_id=image_id,
            class_name=classification['class'],
            confidence=classification['confidence'],
            logits=classification['logits'],
            probabilities=classification['probabilities'],
            log_context=LogContext(
                image_id=image_id,
                duration=duration,
                stage='classify_complete'
            )
        )
    
    except Exception as e:
        logger.error(f"Classification failed: {e}", extra={
            'image_id': None,
            'path': file.filename,
            'stage': 'classify_error'
        })
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
