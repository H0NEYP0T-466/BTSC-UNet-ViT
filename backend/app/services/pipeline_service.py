"""
Pipeline service for orchestrating preprocessing, brain segmentation, tumor segmentation, and classification.
"""
import time
from typing import Dict, Optional
import numpy as np
from app.utils.preprocessing import preprocess_pipeline
from app.models.brain_unet.infer_unet import get_brain_unet_inference
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
        self.brain_unet = None  # Lazy load
        self.tumor_unet = None  # Lazy load
        self.vit = None  # Lazy load
        
        logger.info("Pipeline service initialized", extra={
            'image_id': None,
            'path': None,
            'stage': 'pipeline_init'
        })
    
    def _ensure_models_loaded(self):
        """Ensure models are loaded (lazy initialization)."""
        # Load Brain UNet model
        if self.brain_unet is None:
            logger.info("Loading Brain UNet model", extra={
                'image_id': None,
                'path': None,
                'stage': 'model_load'
            })
            self.brain_unet = get_brain_unet_inference()
        
        # Load Tumor UNet model
        if self.tumor_unet is None:
            logger.info("Loading Tumor UNet model", extra={
                'image_id': None,
                'path': None,
                'stage': 'model_load'
            })
            self.tumor_unet = get_unet_inference()
        
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
        
        Pipeline flow:
        1. Preprocessing (grayscale, denoise, contrast, sharpen, normalize)
        2. Brain segmentation (extract brain tissue using Brain UNet)
        3. Tumor segmentation (segment tumor from brain using Tumor UNet)
        4. Classification (classify tumor type using ViT)
        
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
        
        logger.info("Preprocessing completed, passing to next layer: Brain UNet", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_preprocess'
        })
        
        # Step 2: Brain Segmentation
        logger.info("Step 2: Brain segmentation using Brain UNet", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_brain_segment'
        })
        
        self._ensure_models_loaded()
        
        # Use Brain UNet to extract brain tissue
        brain_segmentation_results = self.brain_unet.segment_brain(
            preprocessed['normalized'],
            image_id=image_id
        )
        
        # Save brain segmentation artifacts
        brain_segment_urls = {}
        for seg_type, seg_image in brain_segmentation_results.items():
            url = self.storage.save_artifact(seg_image, image_id, f"brain_{seg_type}")
            brain_segment_urls[seg_type] = self.storage.get_artifact_url(url)
        
        logger.info("Brain segmentation completed, passing to next layer: Tumor UNet", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_brain_segment'
        })
        
        # Step 3: Tumor Segmentation
        logger.info("Step 3: Tumor segmentation using Tumor UNet", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_tumor_segment'
        })
        
        # Use brain-extracted image for tumor segmentation
        tumor_segmentation_results = self.tumor_unet.segment_image(
            brain_segmentation_results['brain_extracted'],
            image_id=image_id
        )
        
        # Save tumor segmentation artifacts
        tumor_segment_urls = {}
        for seg_type, seg_image in tumor_segmentation_results.items():
            url = self.storage.save_artifact(seg_image, image_id, f"tumor_{seg_type}")
            tumor_segment_urls[seg_type] = self.storage.get_artifact_url(url)
        
        logger.info("Tumor segmentation completed, passing to next layer: ViT classification", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_tumor_segment'
        })
        
        # Step 4: Classification
        logger.info("Step 4: ViT classification", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_classify'
        })
        
        classification_results = self.vit.classify(
            tumor_segmentation_results['segmented'],
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
            'brain_segmentation': brain_segment_urls,
            'tumor_segmentation': tumor_segment_urls,
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
