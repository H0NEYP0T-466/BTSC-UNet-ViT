"""
Pipeline service for orchestrating preprocessing, tumor segmentation, and classification.
"""
import time
from typing import Dict, Optional
import numpy as np
from app.utils.preprocessing import preprocess_pipeline, to_grayscale
from app.models.unet.infer_unet import get_unet_inference
from app.models.unet_tumor.infer_unet_tumor import get_unet_tumor_inference
from app.models.vit.infer_vit import get_vit_inference
from app.services.storage_service import get_storage_service
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Constants for no tumor classification (supports legacy and current naming)
NO_TUMOR_CLASSES = {'notumor', 'no_tumor'}


class PipelineService:
    """Service for orchestrating the full inference pipeline."""
    
    def __init__(self):
        self.storage = get_storage_service()
        self.tumor_unet = None  # Lazy load - BraTS UNet
        self.tumor_unet2 = None  # Lazy load - PNG UNet Tumor
        self.vit = None  # Lazy load
        
        logger.info("Pipeline service initialized", extra={
            'image_id': None,
            'path': None,
            'stage': 'pipeline_init'
        })
    
    def _ensure_models_loaded(self):
        """Ensure models are loaded (lazy initialization)."""
        # Load Tumor UNet model (BraTS H5)
        if self.tumor_unet is None:
            logger.info("Loading Tumor UNet model (BraTS)", extra={
                'image_id': None,
                'path': None,
                'stage': 'model_load'
            })
            self.tumor_unet = get_unet_inference()
        
        # Load Tumor UNet2 model (PNG)
        if self.tumor_unet2 is None:
            logger.info("Loading Tumor UNet2 model (PNG)", extra={
                'image_id': None,
                'path': None,
                'stage': 'model_load'
            })
            self.tumor_unet2 = get_unet_tumor_inference()
        
        if self.vit is None:
            logger.info("Loading ViT model", extra={
                'image_id': None,
                'path': None,
                'stage': 'model_load'
            })
            self.vit = get_vit_inference()
    
    def run_inference(self, image: np.ndarray, skip_preprocessing: bool = False) -> Dict:
        """
        Run full inference pipeline.
        
        Pipeline flow (NEW):
        1. Preprocessing (grayscale, denoise, contrast, sharpen, normalize) - OPTIONAL
        2. ViT Classification (classify tumor type)
        3. Conditional Tumor Segmentation (only if tumor detected)
        
        Args:
            image: Input image (RGB or grayscale)
            skip_preprocessing: If True, skip preprocessing and pass image directly to ViT
            
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
        
        # Step 1: Preprocessing (CONDITIONAL)
        if skip_preprocessing:
            logger.info("Skipping preprocessing pipeline (skip_preprocessing=True) - passing raw image to ViT", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'pipeline_preprocess'
            })
            
            # No preprocessing: pass raw image directly to ViT
            # Return empty preprocessing dict - no display needed
            preprocess_urls = {}
            
            # Use original image as the final preprocessed image
            preprocessed_image = image
            
        else:
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
            
            # Save preprocessing artifacts (only numpy arrays, skip nested dicts)
            preprocess_urls = {}
            for stage_name, stage_image in preprocessed.items():
                # Skip nested dictionaries for defensive programming
                if isinstance(stage_image, dict):
                    continue
                url = self.storage.save_artifact(stage_image, image_id, stage_name)
                preprocess_urls[stage_name] = self.storage.get_artifact_url(url)
            
            # Use normalized image as the final preprocessed image
            preprocessed_image = preprocessed['normalized']
            
            logger.info("Preprocessing completed, passing to next layer: ViT classification", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'pipeline_preprocess'
            })
        
        # Step 2: ViT Classification (NEW ORDER)
        logger.info("Step 2: ViT classification", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'pipeline_classify'
        })
        
        self._ensure_models_loaded()
        
        # Use preprocessed image for classification
        classification_results = self.vit.classify(
            preprocessed_image,
            image_id=image_id
        )
        
        predicted_class = classification_results['class']
        
        logger.info(
            f"ViT classification completed: class={predicted_class}, "
            f"confidence={classification_results['confidence']:.4f}",
            extra={
                'image_id': image_id,
                'path': None,
                'stage': 'pipeline_classify'
            }
        )
        
        # Step 3: Conditional Tumor Segmentation (with both UNet models)
        tumor_segment_urls = {}
        tumor_segment2_urls = {}
        
        # Check for notumor using centralized constant (backward compatible)
        if predicted_class in NO_TUMOR_CLASSES:
            # Skip segmentation for no tumor cases
            logger.info(
                "No tumor detected by ViT, skipping tumor segmentation",
                extra={
                    'image_id': image_id,
                    'path': None,
                    'stage': 'pipeline_skip_segment'
                }
            )
        else:
            # Perform tumor segmentation with UNet1 (BraTS model)
            logger.info(
                f"Tumor detected ({predicted_class}), performing tumor segmentation with UNet1",
                extra={
                    'image_id': image_id,
                    'path': None,
                    'stage': 'pipeline_tumor_segment'
                }
            )
            
            tumor_segmentation_results = self.tumor_unet.segment_image(
                preprocessed_image,
                image_id=image_id
            )
            
            # Save UNet1 segmentation artifacts
            for seg_type, seg_image in tumor_segmentation_results.items():
                if isinstance(seg_image, dict):
                    continue
                url = self.storage.save_artifact(seg_image, image_id, f"tumor_{seg_type}")
                tumor_segment_urls[seg_type] = self.storage.get_artifact_url(url)
            
            logger.info("UNet1 tumor segmentation completed", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'pipeline_tumor_segment'
            })
            
            # Perform tumor segmentation with UNet2 (PNG model)
            logger.info(
                f"Performing tumor segmentation with UNet2 (PNG model)",
                extra={
                    'image_id': image_id,
                    'path': None,
                    'stage': 'pipeline_tumor_segment2'
                }
            )
            
            tumor_segmentation2_results = self.tumor_unet2.segment_image(
                preprocessed_image,
                image_id=image_id
            )
            
            # Save UNet2 segmentation artifacts
            for seg_type, seg_image in tumor_segmentation2_results.items():
                if isinstance(seg_image, dict):
                    continue
                url = self.storage.save_artifact(seg_image, image_id, f"tumor2_{seg_type}")
                tumor_segment2_urls[seg_type] = self.storage.get_artifact_url(url)
            
            logger.info("UNet2 tumor segmentation completed", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'pipeline_tumor_segment2'
            })
        
        # Compile results
        total_duration = time.time() - start_time
        
        logger.info(
            f"Inference pipeline completed in {total_duration:.3f}s: "
            f"class={predicted_class}, "
            f"confidence={classification_results['confidence']:.4f}",
            extra={
                'image_id': image_id,
                'path': None,
                'stage': 'pipeline_complete'
            }
        )
        
        result = {
            'image_id': image_id,
            'original_url': self.storage.get_artifact_url(original_url),
            'preprocessing': preprocess_urls,
            'classification': classification_results,
            'duration_seconds': total_duration,
            'log_context': {
                'image_id': image_id,
                'total_duration': total_duration
            }
        }
        
        # Only add tumor_segmentation if it was performed
        if tumor_segment_urls:
            result['tumor_segmentation'] = tumor_segment_urls
        
        # Add tumor_segmentation2 (UNet Tumor model)
        if tumor_segment2_urls:
            result['tumor_segmentation2'] = tumor_segment2_urls
        
        return result


# Singleton instance
_pipeline_service = None


def get_pipeline_service() -> PipelineService:
    """Get singleton pipeline service instance."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service
