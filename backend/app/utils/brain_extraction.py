"""
Brain extraction utilities using HD-BET.
HD-BET (Hierarchical Deep Brain Extraction Tool) for skull-stripping in brain MRI images.
"""
import time
from typing import Optional, Tuple
import numpy as np
import tempfile
import os
from pathlib import Path
import nibabel as nib
from app.utils.logger import get_logger

logger = get_logger(__name__)


def extract_brain_hdbet(
    image: np.ndarray,
    image_id: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract brain tissue from MRI image using HD-BET.
    
    HD-BET performs skull-stripping to isolate brain tissue, which helps:
    1. Remove non-brain structures (skull, neck, eyes, nose)
    2. Prevent false positives when these structures become overly bright after CLAHE
    3. Focus preprocessing and segmentation on brain tissue only
    
    Args:
        image: Input grayscale MRI image (H x W)
        image_id: Image identifier for logging
        
    Returns:
        Tuple of (brain_extracted_image, brain_mask)
        - brain_extracted_image: Image with only brain tissue (background zeroed)
        - brain_mask: Binary mask (0=background, 255=brain tissue)
    """
    start_time = time.time()
    logger.info("Starting HD-BET brain extraction", extra={
        'image_id': image_id,
        'path': None,
        'stage': 'brain_extraction'
    })
    
    try:
        # Import HD-BET (lazy import to avoid startup overhead)
        from HD_BET.run import run_hd_bet
        
        # Create temporary directory for HD-BET processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # HD-BET expects NIfTI format, so we need to convert
            # Create a simple NIfTI file from 2D image
            # We'll create a 3D volume with single slice for compatibility
            image_3d = image[:, :, np.newaxis]  # Add z-dimension (H, W, 1)
            
            # Create NIfTI image with identity affine
            nifti_img = nib.Nifti1Image(image_3d.astype(np.float32), affine=np.eye(4))
            
            # Save input as NIfTI
            input_path = temp_dir_path / "input.nii.gz"
            nib.save(nifti_img, str(input_path))
            
            logger.info(f"Saved temporary NIfTI file: {input_path}", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'brain_extraction'
            })
            
            # Run HD-BET
            # Parameters:
            # - mode='fast' for faster processing (uses lower resolution)
            # - device='cpu' or 'cuda' depending on availability
            output_path = temp_dir_path / "output.nii.gz"
            
            logger.info("Running HD-BET inference...", extra={
                'image_id': image_id,
                'path': None,
                'stage': 'brain_extraction'
            })
            
            # Run HD-BET skull stripping
            # Note: HD-BET will create output.nii.gz and output_mask.nii.gz
            run_hd_bet(
                mri_fnames=[str(input_path)],
                output_fnames=[str(output_path)],
                mode='fast',  # Fast mode for speed
                device='cpu',  # Use CPU (can be changed to 'cuda' if GPU available)
                postprocess=True,  # Apply postprocessing for cleaner masks
                do_tta=False,  # Disable test-time augmentation for speed
                keep_mask=True,  # Keep the mask file
                overwrite=True
            )
            
            # Load the brain-extracted image and mask
            mask_path = temp_dir_path / "output_mask.nii.gz"
            
            if not output_path.exists():
                logger.error("HD-BET failed to generate output", extra={
                    'image_id': image_id,
                    'path': None,
                    'stage': 'brain_extraction'
                })
                raise RuntimeError("HD-BET failed to generate output file")
            
            if not mask_path.exists():
                logger.error("HD-BET failed to generate mask", extra={
                    'image_id': image_id,
                    'path': None,
                    'stage': 'brain_extraction'
                })
                raise RuntimeError("HD-BET failed to generate mask file")
            
            # Load results
            brain_nifti = nib.load(str(output_path))
            mask_nifti = nib.load(str(mask_path))
            
            # Extract 2D slices from 3D volumes
            brain_3d = brain_nifti.get_fdata()
            mask_3d = mask_nifti.get_fdata()
            
            # Get the single slice we processed
            brain_extracted = brain_3d[:, :, 0].astype(np.float32)
            brain_mask_float = mask_3d[:, :, 0].astype(np.float32)
            
            # Convert mask to binary (0 or 255)
            brain_mask = (brain_mask_float > 0.5).astype(np.uint8) * 255
            
            # Ensure brain_extracted is in uint8 range for consistency
            if brain_extracted.max() > 0:
                brain_extracted = ((brain_extracted - brain_extracted.min()) / 
                                 (brain_extracted.max() - brain_extracted.min()) * 255).astype(np.uint8)
            else:
                brain_extracted = brain_extracted.astype(np.uint8)
            
            duration = time.time() - start_time
            
            # Calculate mask statistics
            brain_pixels = np.sum(brain_mask > 0)
            total_pixels = brain_mask.size
            brain_percentage = (brain_pixels / total_pixels) * 100
            
            logger.info(
                f"HD-BET brain extraction completed in {duration:.3f}s, "
                f"brain_area={brain_percentage:.2f}%",
                extra={
                    'image_id': image_id,
                    'path': None,
                    'stage': 'brain_extraction'
                }
            )
            
            return brain_extracted, brain_mask
    
    except ImportError as e:
        logger.error(f"HD-BET not installed: {e}", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'brain_extraction'
        })
        logger.warning("Skipping brain extraction, returning original image", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'brain_extraction'
        })
        # Return original image and full mask if HD-BET not available
        full_mask = np.ones_like(image, dtype=np.uint8) * 255
        return image.copy(), full_mask
    
    except Exception as e:
        logger.error(f"HD-BET brain extraction failed: {e}", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'brain_extraction'
        })
        logger.warning("Falling back to original image without brain extraction", extra={
            'image_id': image_id,
            'path': None,
            'stage': 'brain_extraction'
        })
        # Return original image and full mask on error
        full_mask = np.ones_like(image, dtype=np.uint8) * 255
        return image.copy(), full_mask
