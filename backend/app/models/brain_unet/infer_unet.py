"""
Brain UNet inference for brain extraction.
"""
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
import cv2

# Add btsc module to path
project_root = Path(__file__).resolve().parents[4]  # Go up to project root
sys.path.insert(0, str(project_root))

from btsc.preprocess.brain_extraction import apply_pipeline
from app.models.brain_unet.model import get_brain_unet_model
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BrainUNetInference:
    """Brain UNet inference class for brain segmentation."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        enable_advanced_preproc: bool = True
    ):
        """
        Initialize Brain UNet inference.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            enable_advanced_preproc: Enable advanced brain preprocessing pipeline (default: True)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or (settings.CHECKPOINTS_BRAIN_UNET / settings.BRAIN_UNET_CHECKPOINT_NAME)
        self.enable_advanced_preproc = enable_advanced_preproc
        
        # Load preprocessing config
        self.preproc_config = self._load_preproc_config()
        
        # Load model
        self.model = self._load_model()
        
        logger.info(
            f"BrainUNet inference initialized: device={self.device}, "
            f"model_path={self.model_path}, advanced_preproc={self.enable_advanced_preproc}",
            extra={'image_id': None, 'path': str(self.model_path), 'stage': 'infer_init'}
        )
    
    def _load_preproc_config(self) -> Dict[str, Any]:
        """Load brain preprocessing configuration."""
        config_path = project_root / "btsc" / "configs" / "brain_preproc.yaml"
        
        if not config_path.exists():
            logger.warning(
                f"Brain preprocessing config not found at {config_path}, using defaults",
                extra={'image_id': None, 'path': str(config_path), 'stage': 'config_load'}
            )
            return {'enable': False}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(
                f"Loaded brain preprocessing config from {config_path}",
                extra={'image_id': None, 'path': str(config_path), 'stage': 'config_load'}
            )
            
            return config
        except Exception as e:
            logger.error(
                f"Failed to load preprocessing config: {e}",
                extra={'image_id': None, 'path': str(config_path), 'stage': 'config_load'}
            )
            return {'enable': False}
    
    def _load_model(self) -> torch.nn.Module:
        """Load the trained model."""
        # Create model
        model = get_brain_unet_model(
            in_channels=1,
            out_channels=1,
            features=(32, 64, 128, 256, 512)
        )
        
        # Load checkpoint if exists
        if self.model_path.exists():
            logger.info(f"Loading checkpoint from {self.model_path}", extra={
                'image_id': None, 'path': str(self.model_path), 'stage': 'model_load'
            })
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(
                f"Loaded checkpoint: dice={checkpoint.get('dice_score', 'N/A')}",
                extra={'image_id': None, 'path': str(self.model_path), 'stage': 'model_load'}
            )
        else:
            logger.warning(
                f"No checkpoint found at {self.model_path}, using untrained model",
                extra={'image_id': None, 'path': str(self.model_path), 'stage': 'model_load'}
            )
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def segment_brain(
        self,
        image: np.ndarray,
        image_id: Optional[str] = None,
        save_intermediates: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Segment brain from input image with optional advanced preprocessing.
        
        Args:
            image: Input grayscale image (H, W) in range [0, 255]
            image_id: Optional image identifier for logging
            save_intermediates: Save intermediate preprocessing stages (default: False)
            output_dir: Directory to save intermediates (default: outputs/preproc/{image_id})
            
        Returns:
            Dictionary with:
                - mask: Binary brain mask (H, W) in range [0, 255]
                - brain_extracted: Brain-only image (H, W) in range [0, 255]
                - overlay: Brain mask overlay on original image
                - preprocessing: Dict with preprocessing stages (if advanced preproc enabled)
                - candidates: Dict with candidate masks (if advanced preproc enabled)
        """
        start_time = time.time()
        
        logger.info("Starting brain segmentation", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'brain_segment'
        })
        
        # Store original shape and image
        original_shape = image.shape
        original_image = image.copy()
        
        result = {
            'mask': None,
            'brain_extracted': None,
            'overlay': None,
            'preprocessing': None,
            'candidates': None,
            'candidate_overlays': None,
            'used_fallback': False,
            'fallback_method': None
        }
        
        # Apply advanced preprocessing if enabled
        # Always try to enable it to generate candidate masks for fallback
        should_preprocess = self.enable_advanced_preproc and self.preproc_config.get('enable', False)
        
        if should_preprocess:
            logger.info("Applying advanced brain preprocessing pipeline", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'brain_preproc'
            })
            
            # Run preprocessing pipeline
            preproc_result = apply_pipeline(image.astype(np.float32), self.preproc_config)
            
            # Add preprocessing results to output
            result['preprocessing'] = preproc_result['stages']
            result['candidates'] = preproc_result['candidates']
            
            # Create overlays for each candidate mask to show them applied on the original image
            result['candidate_overlays'] = {}
            for method_name, candidate_mask in result['candidates'].items():
                overlay = self._create_overlay(original_image, candidate_mask)
                result['candidate_overlays'][method_name] = overlay
            
            # Save intermediates if requested
            if save_intermediates:
                self._save_preprocessing_stages(
                    preproc_result,
                    image_id,
                    output_dir or f"outputs/preproc/{image_id}"
                )
            
            logger.info("Advanced preprocessing completed", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'brain_preproc'
            })
        
        # FIX: Always use original image for model input to match training data format
        # Training data uses min-max normalization to [0, 1] on raw NIfTI data
        # We must do the same here regardless of preprocessing
        model_input = original_image.astype(np.float32)
        
        # Normalize to [0, 1] using min-max normalization (SAME AS TRAINING)
        # This is critical for the model to work correctly
        image_normalized = (model_input - model_input.min()) / (model_input.max() - model_input.min() + 1e-8)
        
        # Resize to model input size (256x256)
        image_resized = cv2.resize(image_normalized.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor [1, 1, H, W]
        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.sigmoid(output)
            mask_prob = prediction.squeeze().cpu().numpy()
        
        # Binarize mask (threshold at 0.5)
        mask_binary = (mask_prob > 0.5).astype(np.float32)
        
        # Resize back to original shape
        mask_resized = cv2.resize(mask_binary, (original_shape[1], original_shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Convert to uint8 [0, 255]
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        
        # Calculate brain percentage from UNet prediction
        brain_percentage = (np.sum(mask_resized > 0.5) / mask_resized.size) * 100
        
        # Check if UNet mask is empty or near-empty (fallback threshold: 0.1%)
        used_fallback = False
        fallback_method = None
        
        if brain_percentage < 0.1:
            logger.warning(
                f"UNet produced near-empty mask ({brain_percentage:.4f}%). Attempting fallback to candidate masks.",
                extra={'image_id': image_id, 'path': None, 'stage': 'brain_segment_fallback'}
            )
            
            # Try to use candidate masks if available
            if 'candidates' in result and result['candidates']:
                # Select best candidate mask (one with largest non-zero area)
                best_method = None
                best_area = 0
                
                for method_name, candidate_mask in result['candidates'].items():
                    candidate_area = np.sum(candidate_mask > 0)
                    if candidate_area > best_area:
                        best_area = candidate_area
                        best_method = method_name
                
                if best_method and best_area > 0:
                    # Use best candidate mask as fallback
                    mask_uint8 = result['candidates'][best_method]
                    mask_binary = (mask_uint8 > 0).astype(np.float32)
                    brain_percentage = (np.sum(mask_binary) / mask_binary.size) * 100
                    used_fallback = True
                    fallback_method = best_method
                    
                    logger.info(
                        f"Using fallback mask from '{best_method}' method, brain_area={brain_percentage:.2f}%",
                        extra={'image_id': image_id, 'path': None, 'stage': 'brain_segment_fallback'}
                    )
                else:
                    logger.warning(
                        "No valid candidate masks available for fallback. Using empty UNet mask.",
                        extra={'image_id': image_id, 'path': None, 'stage': 'brain_segment_fallback'}
                    )
            else:
                logger.warning(
                    "Advanced preprocessing not enabled. No candidate masks available for fallback.",
                    extra={'image_id': image_id, 'path': None, 'stage': 'brain_segment_fallback'}
                )
        
        # Apply mask to get brain-extracted image
        brain_extracted = original_image.copy()
        mask_binary = (mask_uint8 > 0).astype(np.float32)
        brain_extracted[mask_binary < 0.5] = 0
        
        # Create overlay visualization
        overlay = self._create_overlay(original_image, mask_uint8)
        
        # Update result with mask, overlay, and fallback info
        result['mask'] = mask_uint8
        result['brain_extracted'] = brain_extracted
        result['overlay'] = overlay
        result['used_fallback'] = used_fallback
        result['fallback_method'] = fallback_method
        
        duration = time.time() - start_time
        
        logger.info(
            f"Brain segmentation completed in {duration:.3f}s, "
            f"brain_area={brain_percentage:.2f}%, used_fallback={used_fallback}",
            extra={'image_id': image_id, 'path': None, 'stage': 'brain_segment'}
        )
        
        return result
    
    def _save_preprocessing_stages(
        self,
        preproc_result: Dict[str, Any],
        image_id: str,
        output_dir: str
    ):
        """
        Save preprocessing stages to disk.
        
        Args:
            preproc_result: Result from apply_pipeline()
            image_id: Image identifier
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save stages
        for stage_name, stage_img in preproc_result['stages'].items():
            # Normalize to uint8
            img_uint8 = ((stage_img - stage_img.min()) / (stage_img.max() - stage_img.min() + 1e-8) * 255).astype(np.uint8)
            save_path = output_path / f"{stage_name}.png"
            cv2.imwrite(str(save_path), img_uint8)
        
        # Save candidate masks
        for mask_name, mask_img in preproc_result['candidates'].items():
            save_path = output_path / f"mask_{mask_name}.png"
            cv2.imwrite(str(save_path), mask_img)
        
        # Save final results
        final = preproc_result['final']
        for key, img in final.items():
            if img is not None and img.size > 0:
                if img.max() > 1.0:
                    img_uint8 = img.astype(np.uint8)
                else:
                    img_uint8 = (img * 255).astype(np.uint8)
                save_path = output_path / f"{key}.png"
                cv2.imwrite(str(save_path), img_uint8)
        
        logger.info(
            f"Saved preprocessing stages to {output_dir}",
            extra={'image_id': image_id, 'path': output_dir, 'stage': 'save_preproc'}
        )
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create an overlay visualization of mask on image.
        
        Args:
            image: Original grayscale image [0, 255]
            mask: Binary mask [0, 255]
            
        Returns:
            RGB overlay image
        """
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
        
        # Create colored mask (green for brain)
        mask_colored = np.zeros_like(image_rgb)
        mask_colored[:, :, 1] = mask  # Green channel
        
        # Blend
        alpha = 0.3
        overlay = cv2.addWeighted(image_rgb, 1.0, mask_colored, alpha, 0)
        
        return overlay


# Singleton instance
_brain_unet_inference = None


def get_brain_unet_inference(
    model_path: Optional[Path] = None,
    device: Optional[str] = None
) -> BrainUNetInference:
    """
    Get singleton Brain UNet inference instance.
    
    Args:
        model_path: Optional path to model checkpoint
        device: Optional device to use
        
    Returns:
        BrainUNetInference instance
    """
    global _brain_unet_inference
    
    if _brain_unet_inference is None:
        _brain_unet_inference = BrainUNetInference(
            model_path=model_path,
            device=device
        )
    
    return _brain_unet_inference
