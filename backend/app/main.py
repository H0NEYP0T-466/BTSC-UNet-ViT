"""
Main FastAPI application for BTSC-UNet-ViT.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.logging_config import setup_logging
from app.routers import health, preprocessing, segmentation, classification
from app.schemas.responses import InferenceResponse
from app.services.pipeline_service import get_pipeline_service
from app.utils.imaging import bytes_to_numpy
from app.utils.logger import get_logger
import time

# Setup logging
setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Brain Tumor Segmentation and Classification using UNet and ViT",
    debug=settings.DEBUG
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving artifacts
app.mount("/files", StaticFiles(directory=str(settings.RESOURCES_DIR)), name="files")

# Include routers
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(preprocessing.router, prefix=settings.API_PREFIX)
app.include_router(segmentation.router, prefix=settings.API_PREFIX)
app.include_router(classification.router, prefix=settings.API_PREFIX)

logger.info(f"FastAPI application initialized: {settings.APP_NAME} v{settings.APP_VERSION}", extra={
    'image_id': None,
    'path': None,
    'stage': 'app_init'
})


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BTSC-UNet-ViT API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.post(f"{settings.API_PREFIX}/inference", response_model=InferenceResponse)
async def run_inference(file: UploadFile = File(...)):
    """
    Run full inference pipeline: preprocessing -> segmentation -> classification.
    
    This is the main endpoint that orchestrates all stages.
    """
    start_time = time.time()
    
    logger.info(f"Received full inference request: filename={file.filename}", extra={
        'image_id': None,
        'path': file.filename,
        'stage': 'inference_request'
    })
    
    try:
        # Read image bytes
        contents = await file.read()
        image = bytes_to_numpy(contents)
        
        # Run pipeline
        pipeline = get_pipeline_service()
        result = pipeline.run_inference(image)
        
        duration = time.time() - start_time
        
        logger.info(
            f"Full inference pipeline completed in {duration:.3f}s",
            extra={
                'image_id': result['image_id'],
                'path': None,
                'stage': 'inference_complete'
            }
        )
        
        return InferenceResponse(**result)
    
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}", extra={
            'image_id': None,
            'path': file.filename,
            'stage': 'inference_error'
        })
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Application startup", extra={
        'image_id': None,
        'path': None,
        'stage': 'startup'
    })
    logger.info(f"CORS origins: {settings.CORS_ORIGINS}", extra={
        'image_id': None,
        'path': None,
        'stage': 'startup'
    })


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Application shutdown", extra={
        'image_id': None,
        'path': None,
        'stage': 'shutdown'
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
