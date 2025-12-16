"""
Pipeline service for orchestrating preprocessing, segmentation, and classification.
"""
import time
from typing import Dict, Optional
import numpy as np
from app.utils.preprocessing import preprocess_pipeline
from app.models.unet.infer_unet import get_unet_inference
from app.models.vit.infer_vit import get_vit_inference
from app.services.storage_service import get_storage_service
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineService:
    """Service for orchestrating the full inference pipeline."""
    
    def __init__(self):
        self.storage = get_storage_service()
        self.unet = None  # Lazy load
        self.vit = None  # Lazy load
        
        logger.info("Pipeline service initialized", extra={
            'image_id': None,
            'path': None,
            'stage': 'pipeline_init'
        })
    
    def _ensure_models_loaded(self):
        """Ensure models are loaded (lazy initialization)."""
        # Load local trained UNet model
        if self.unet is None:
            logger.info("Loading local trained UNet model", extra={
                'image_id': None,
                'path': None,
                'stage': 'model_load'
            })
            self.unet = get_unet_inference()
        
        if self.vit is None:
            logger.info("Loading ViT model", extra={
                'image_id': None,
                'path': None,
                'stage': 'model_load'
            })
            self.vit = get_vit_inference()
    
    def run_inference(self, image: np.ndarray) -> Dict:
        """
        Run full inference pipeline.
        
        Args:
            image: Input image (RGB or grayscale)
            
        Returns:
            Dictionary with all results and artifact URLs
        """
        start_time = time.time()
        
        # Generate image ID
        image_id = self.storage.generate_image_id()
        
        logger.info(f"Starting inference pipeline for image {image_id}", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_start'
        })
        
        # Save original image
        original_url = self.storage.save_upload(image, image_id)
        
        # Step 1: Preprocessing
        logger.info("Step 1: Preprocessing", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_preprocess'
        })
        
        preprocess_config = {
            'median_kernel_size': settings.MEDIAN_KERNEL_SIZE,
            'clahe_clip_limit': settings.CLAHE_CLIP_LIMIT,
            'clahe_tile_grid_size': settings.CLAHE_TILE_GRID_SIZE,
            'unsharp_radius': settings.UNSHARP_RADIUS,
            'unsharp_amount': settings.UNSHARP_AMOUNT,
            'preserve_detail': settings.MOTION_PRESERVE_DETAIL,
            'normalize_method': 'zscore',
            'use_nlm_denoising': True,
            'nlm_h': settings.NLM_H
        }
        
        preprocessed = preprocess_pipeline(
            image, 
            config=preprocess_config, 
            image_id=image_id
        )
        
        # Save preprocessing artifacts
        preprocess_urls = {}
        for stage_name, stage_image in preprocessed.items():
            url = self.storage.save_artifact(stage_image, image_id, stage_name)
            preprocess_urls[stage_name] = self.storage.get_artifact_url(url)
        
        logger.info("Preprocessing completed, passing to next layer: UNet segmentation", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_preprocess'
        })
        
        # Step 2: Segmentation
        logger.info("Step 2: Local UNet segmentation", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_segment'
        })
        
        self._ensure_models_loaded()
        
        # Use local trained UNet model
        segmentation_results = self.unet.segment_image(
            preprocessed['normalized'],
            image_id=image_id
        )
        
        # Save segmentation artifacts
        segment_urls = {}
        for seg_type, seg_image in segmentation_results.items():
            url = self.storage.save_artifact(seg_image, image_id, f"seg_{seg_type}")
            segment_urls[seg_type] = self.storage.get_artifact_url(url)
        
        logger.info("Segmentation completed, passing to next layer: ViT classification", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_segment'
        })
        
        # Step 3: Classification
        logger.info("Step 3: ViT classification", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_classify'
        })
        
        classification_results = self.vit.classify(
            segmentation_results['segmented'],
            image_id=image_id
        )
        
        # Compile results
        total_duration = time.time() - start_time
        
        logger.info(
            f"Inference pipeline completed in {total_duration:.3f}s: "
            f"class={classification_results['class']}, "
            f"confidence={classification_results['confidence']:.4f}",
            extra={
                'image_id': image_id,
                'path': None,
                'stage': 'pipeline_complete'
            }
        )
        
        return {
            'image_id': image_id,
            'original_url': self.storage.get_artifact_url(original_url),
            'preprocessing': preprocess_urls,
            'segmentation': segment_urls,
            'classification': classification_results,
            'duration_seconds': total_duration,
            'log_context': {
                'image_id': image_id,
                'total_duration': total_duration
            }
        }


# Singleton instance
_pipeline_service = None


def get_pipeline_service() -> PipelineService:
    """Get singleton pipeline service instance."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service
