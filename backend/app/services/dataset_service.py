"""
Dataset service for batch preprocessing and segmentation of the 90k image dataset.
"""
import time
from pathlib import Path
from typing import Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from app.utils.preprocessing import preprocess_pipeline
from app.utils.imaging import read_image, save_image
from app.models.unet_tumor.infer_unet_tumor import get_unet_tumor_inference
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetService:
    """Service for batch processing the dataset."""
    
    def __init__(self):
        self.unet = None
        logger.info("Dataset service initialized", extra={
            'image_id': None,
            'path': None,
            'stage': 'dataset_init'
        })
    
    def _ensure_unet_loaded(self):
        """Ensure UNet Tumor model is loaded."""
        if self.unet is None:
            logger.info("Loading UNet Tumor for dataset processing", extra={
                'image_id': None,
                'path': None,
                'stage': 'dataset_model_load'
            })
            self.unet = get_unet_tumor_inference()
    
    def process_single_image(
        self,
        image_path: Path,
        output_dir: Path,
        class_name: str
    ) -> bool:
        """
        Process a single image: preprocess and segment.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory
            class_name: Class name for organization
            
        Returns:
            Success status
        """
        try:
            image_id = image_path.stem
            
            # Read image
            image = read_image(str(image_path))
            
            # Preprocess
            preprocessed = preprocess_pipeline(image, image_id=image_id)
            
            # Segment
            segmentation = self.unet.segment_image(preprocessed['normalized'], image_id=image_id)
            
            # Save preprocessed
            preprocess_path = output_dir / class_name / 'images_preprocessed' / f"{image_id}.png"
            preprocess_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(preprocessed['normalized'], str(preprocess_path))
            
            # Save segmented
            segment_path = output_dir / class_name / 'images_segmented' / f"{image_id}.png"
            segment_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(segmentation['segmented'], str(segment_path))
            
            return True
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}", extra={
                'image_id': image_path.stem,
                'path': str(image_path),
                'stage': 'dataset_process'
            })
            return False
    
    def preprocess_and_segment_dataset(
        self,
        input_root: Optional[str] = None,
        output_root: Optional[str] = None,
        max_workers: int = 4
    ):
        """
        Batch process entire dataset: preprocess and segment all images.
        
        Args:
            input_root: Root directory of input dataset
            output_root: Root directory for output
            max_workers: Number of parallel workers
        """
        start_time = time.time()
        
        input_root = Path(input_root or settings.DATASET_ROOT)
        output_root = Path(output_root or settings.SEGMENTED_DATASET_ROOT)
        
        logger.info(f"Starting dataset preprocessing and segmentation", extra={
            'image_id': None,
            'path': str(input_root),
            'stage': 'dataset_batch_start'
        })
        
        logger.info(f"Concurrency settings: max_workers={max_workers}", extra={
            'image_id': None,
            'path': str(input_root),
            'stage': 'dataset_batch_start'
        })
        
        # Ensure UNet is loaded
        self._ensure_unet_loaded()
        
        # Get class folders
        class_folders = ['giloma', 'meningioma', 'notumor', 'pituitary']
        
        total_processed = 0
        total_failed = 0
        
        for class_name in class_folders:
            class_dir = input_root / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}", extra={
                    'image_id': None,
                    'path': str(class_dir),
                    'stage': 'dataset_batch'
                })
                continue
            
            # Get all images
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            logger.info(f"Processing class '{class_name}': {len(image_files)} images", extra={
                'image_id': None,
                'path': str(class_dir),
                'stage': 'dataset_batch'
            })
            
            # Process images
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_single_image,
                        img_path,
                        output_root,
                        class_name
                    ): img_path
                    for img_path in image_files
                }
                
                pbar = tqdm(as_completed(futures), total=len(image_files), desc=f"Processing {class_name}")
                for future in pbar:
                    success = future.result()
                    if success:
                        total_processed += 1
                    else:
                        total_failed += 1
            
            logger.info(f"Class '{class_name}' completed: {len(image_files)} processed", extra={
                'image_id': None,
                'path': str(class_dir),
                'stage': 'dataset_batch'
            })
        
        duration = time.time() - start_time
        
        logger.info(
            f"Dataset preprocessing completed: "
            f"total={total_processed}, failed={total_failed}, duration={duration:.2f}s",
            extra={
                'image_id': None,
                'path': str(output_root),
                'stage': 'dataset_batch_complete'
            }
        )


# Singleton instance
_dataset_service = None


def get_dataset_service() -> DatasetService:
    """Get singleton dataset service instance."""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service
