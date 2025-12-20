"""
Brain extraction and preprocessing pipeline for NFBS-compatible processing.

This module provides robust brain-only preprocessing and skull-stripping to
harmonize external data (e.g., Kaggle datasets) with NFBS training data characteristics.

Key Features:
- N4 bias field correction (SimpleITK)
- RAS reorientation and isotropic resampling
- Intensity normalization and histogram matching
- Multiple brain extraction methods (Otsu, Yen, Li, Triangle)
- Morphological mask postprocessing
- Configurable pipeline with intermediate outputs

Author: BTSC Team
Date: 2025-12-19
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, morphology, exposure, measure
import SimpleITK as sitk
import nibabel as nib


def n4_bias_correction(
    img: Union[np.ndarray, sitk.Image],
    shrink_factor: int = 4,
    num_iterations: list = None
) -> np.ndarray:
    """
    Apply N4 bias field correction to MRI image.
    
    N4ITK is an improved version of the N3 algorithm for correcting intensity
    non-uniformity (bias field) in MRI images. This is crucial for robust segmentation.
    
    Args:
        img: Input image as numpy array (H, W) or SimpleITK Image
        shrink_factor: Downsampling factor for faster computation (default: 4)
        num_iterations: List of iterations per level (default: [50, 50, 50, 50])
        
    Returns:
        Bias-corrected image as numpy array (same shape as input)
        
    Example:
        >>> img = np.random.rand(256, 256).astype(np.float32)
        >>> corrected = n4_bias_correction(img)
        >>> corrected.shape
        (256, 256)
    """
    if num_iterations is None:
        num_iterations = [50, 50, 50, 50]
    
    # Convert numpy to SimpleITK if needed
    if isinstance(img, np.ndarray):
        # Normalize to float32 if not already
        if img.dtype != np.float32:
            img_float = img.astype(np.float32)
        else:
            img_float = img.copy()
        
        # Create SimpleITK image
        sitk_img = sitk.GetImageFromArray(img_float)
    else:
        sitk_img = img
    
    # Cast to float32 for N4
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    
    # Create mask (non-zero regions)
    mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    
    # Setup N4 corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(num_iterations)
    
    # Run correction
    corrected = corrector.Execute(sitk_img, mask)
    
    # Convert back to numpy
    corrected_np = sitk.GetArrayFromImage(corrected)
    
    return corrected_np


def reorient_to_ras(
    img: np.ndarray,
    affine: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Reorient image to RAS (Right-Anterior-Superior) coordinate system.
    
    RAS is a standard neuroimaging orientation:
    - Right: positive X (left to right)
    - Anterior: positive Y (posterior to anterior)
    - Superior: positive Z (inferior to superior)
    
    Args:
        img: Input image array (H, W) or (H, W, D)
        affine: Optional 4x4 affine transformation matrix
        
    Returns:
        Tuple of (reoriented_image, reoriented_affine)
        If no affine provided, returns (img, None)
        
    Note:
        For 2D images without affine, returns input unchanged.
        For full reorientation, provide affine from nibabel.
    """
    if affine is None:
        # No affine provided, return as-is
        return img, None
    
    # Create nibabel image
    nii = nib.Nifti1Image(img, affine)
    
    # Reorient to RAS
    nii_ras = nib.as_closest_canonical(nii)
    
    # Extract data and affine
    img_ras = nii_ras.get_fdata()
    affine_ras = nii_ras.affine
    
    return img_ras, affine_ras


def resample_isotropic(
    img: np.ndarray,
    spacing: float = 1.0,
    current_spacing: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Resample image to isotropic spacing (e.g., 1mm x 1mm).
    
    Isotropic spacing ensures uniform resolution in all dimensions,
    which is important for consistent model inference.
    
    Args:
        img: Input image array (H, W)
        spacing: Target isotropic spacing in mm (default: 1.0)
        current_spacing: Current spacing as (spacing_y, spacing_x) in mm.
                         If None, assumes already isotropic.
        
    Returns:
        Resampled image with isotropic spacing
        
    Example:
        >>> img = np.random.rand(512, 512)
        >>> resampled = resample_isotropic(img, spacing=1.0, current_spacing=(0.5, 0.5))
        >>> resampled.shape
        (256, 256)
    """
    if current_spacing is None:
        # Already isotropic or spacing unknown, return as-is
        return img
    
    # Calculate zoom factors
    zoom_y = current_spacing[0] / spacing
    zoom_x = current_spacing[1] / spacing
    
    # Resample using zoom
    resampled = ndimage.zoom(img, (zoom_y, zoom_x), order=1)
    
    return resampled


def intensity_clip(
    img: np.ndarray,
    pmin: float = 0.5,
    pmax: float = 99.5
) -> np.ndarray:
    """
    Clip image intensities to percentile range to remove outliers.
    
    Intensity clipping removes extreme outliers that can skew normalization
    and affect model performance.
    
    Args:
        img: Input image array
        pmin: Lower percentile threshold (default: 0.5)
        pmax: Upper percentile threshold (default: 99.5)
        
    Returns:
        Clipped image with outliers removed
        
    Example:
        >>> img = np.random.rand(256, 256)
        >>> clipped = intensity_clip(img, pmin=1.0, pmax=99.0)
    """
    lower = np.percentile(img, pmin)
    upper = np.percentile(img, pmax)
    
    clipped = np.clip(img, lower, upper)
    
    return clipped


def zscore_norm(
    img: np.ndarray,
    within_mask: Optional[np.ndarray] = None,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Z-score normalization (standardization) of image intensities.
    
    Z-score normalization: (x - mean) / std
    This centers the data around 0 with unit variance.
    
    Args:
        img: Input image array
        within_mask: Optional binary mask to compute statistics only within brain.
                     Shape must match img. Mask values: 0 (background), >0 (foreground)
        eps: Small epsilon to prevent division by zero (default: 1e-8)
        
    Returns:
        Z-score normalized image
        
    Example:
        >>> img = np.random.rand(256, 256) * 255
        >>> normalized = zscore_norm(img)
        >>> np.abs(normalized.mean()) < 0.1  # Mean close to 0
        True
    """
    if within_mask is not None:
        # Compute statistics only within mask
        masked_values = img[within_mask > 0]
        mean = np.mean(masked_values)
        std = np.std(masked_values)
    else:
        # Compute statistics over entire image
        mean = np.mean(img)
        std = np.std(img)
    
    # Normalize
    normalized = (img - mean) / (std + eps)
    
    return normalized


def histogram_match_to_nfbs(
    img: np.ndarray,
    reference_stats_path: str = "artifacts/nfbs_hist_ref.npz"
) -> np.ndarray:
    """
    Match image histogram to NFBS reference distribution.
    
    Histogram matching aligns the intensity distribution of the input image
    to match the training data (NFBS), reducing domain shift.
    
    Args:
        img: Input image array
        reference_stats_path: Path to NFBS reference statistics file (.npz)
                              Should contain 'reference_hist' array
        
    Returns:
        Histogram-matched image
        
    Note:
        If reference file doesn't exist, returns input unchanged with a warning.
        
    Example:
        >>> img = np.random.rand(256, 256)
        >>> matched = histogram_match_to_nfbs(img, "artifacts/nfbs_hist_ref.npz")
    """
    # Check if reference exists
    if not os.path.exists(reference_stats_path):
        print(f"Warning: NFBS reference not found at {reference_stats_path}. Skipping histogram matching.")
        return img
    
    # Load reference
    try:
        ref_data = np.load(reference_stats_path)
        reference_img = ref_data['reference_hist']
    except Exception as e:
        print(f"Warning: Failed to load NFBS reference: {e}. Skipping histogram matching.")
        return img
    
    # Match histograms using scikit-image
    matched = exposure.match_histograms(img, reference_img, channel_axis=None)
    
    return matched


def otsu_brain_mask(
    img: np.ndarray,
    gaussian_sigma: float = 1.0
) -> np.ndarray:
    """
    Generate brain mask using Otsu thresholding.
    
    Otsu's method automatically determines the optimal threshold for
    separating brain tissue from background.
    
    Args:
        img: Input image array (should be normalized)
        gaussian_sigma: Sigma for Gaussian smoothing before thresholding (default: 1.0)
        
    Returns:
        Binary brain mask (0 = background, 255 = brain)
        
    Example:
        >>> img = np.random.rand(256, 256)
        >>> mask = otsu_brain_mask(img)
        >>> mask.dtype
        dtype('uint8')
    """
    # Normalize to uint8 range
    img_uint8 = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Smooth image
    if gaussian_sigma > 0:
        smoothed = ndimage.gaussian_filter(img_uint8, sigma=gaussian_sigma)
    else:
        smoothed = img_uint8
    
    # Otsu threshold
    threshold = filters.threshold_otsu(smoothed)
    mask = (smoothed > threshold).astype(np.uint8) * 255
    
    return mask


def adaptive_threshold_mask(
    img: np.ndarray,
    method: str = "yen",
    return_all: bool = False,
    gaussian_sigma: float = 1.0
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate brain mask using adaptive thresholding methods.
    
    Provides multiple thresholding algorithms for comparison:
    - Yen: Good for bimodal distributions with unequal peaks
    - Li: Minimum cross-entropy thresholding
    - Triangle: Good for skewed histograms
    - Otsu: Minimizes intra-class variance (included for comparison)
    
    Args:
        img: Input image array (should be normalized)
        method: Thresholding method - 'yen', 'li', 'triangle', or 'otsu' (default: 'yen')
        return_all: If True, returns all methods as dict (default: False)
        gaussian_sigma: Sigma for Gaussian smoothing before thresholding (default: 1.0)
        
    Returns:
        If return_all=False: Single binary mask (0 = background, 255 = brain)
        If return_all=True: Dict with keys ['otsu', 'yen', 'li', 'triangle']
        
    Example:
        >>> img = np.random.rand(256, 256)
        >>> mask = adaptive_threshold_mask(img, method='yen')
        >>> all_masks = adaptive_threshold_mask(img, return_all=True)
        >>> list(all_masks.keys())
        ['otsu', 'yen', 'li', 'triangle']
    """
    # Normalize to uint8 range
    img_uint8 = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Smooth image
    if gaussian_sigma > 0:
        smoothed = ndimage.gaussian_filter(img_uint8, sigma=gaussian_sigma)
    else:
        smoothed = img_uint8
    
    if return_all:
        # Compute all methods
        masks = {}
        
        # Otsu
        thresh_otsu = filters.threshold_otsu(smoothed)
        masks['otsu'] = (smoothed > thresh_otsu).astype(np.uint8) * 255
        
        # Yen
        thresh_yen = filters.threshold_yen(smoothed)
        masks['yen'] = (smoothed > thresh_yen).astype(np.uint8) * 255
        
        # Li
        thresh_li = filters.threshold_li(smoothed)
        masks['li'] = (smoothed > thresh_li).astype(np.uint8) * 255
        
        # Triangle
        thresh_triangle = filters.threshold_triangle(smoothed)
        masks['triangle'] = (smoothed > thresh_triangle).astype(np.uint8) * 255
        
        return masks
    else:
        # Compute single method
        if method == 'otsu':
            threshold = filters.threshold_otsu(smoothed)
        elif method == 'yen':
            threshold = filters.threshold_yen(smoothed)
        elif method == 'li':
            threshold = filters.threshold_li(smoothed)
        elif method == 'triangle':
            threshold = filters.threshold_triangle(smoothed)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: otsu, yen, li, triangle")
        
        mask = (smoothed > threshold).astype(np.uint8) * 255
        return mask


def postprocess_mask(
    mask: np.ndarray,
    min_size: int = 5000,
    closing: int = 3,
    opening: int = 1,
    fill_holes: bool = True,
    keep_largest: bool = True
) -> np.ndarray:
    """
    Postprocess binary mask with morphological operations.
    
    Typical pipeline:
    1. Binary closing (fills small holes, connects nearby regions)
    2. Binary opening (removes small objects)
    3. Hole filling (removes internal holes)
    4. Size filtering (removes small components)
    5. Keep largest component (assumes brain is largest connected region)
    
    Args:
        mask: Binary mask (0 = background, 255 = foreground)
        min_size: Minimum component size in pixels (default: 5000)
        closing: Structuring element size for closing operation (default: 3)
                 Set to 0 to skip.
        opening: Structuring element size for opening operation (default: 1)
                 Set to 0 to skip.
        fill_holes: Whether to fill holes in mask (default: True)
        keep_largest: Whether to keep only the largest connected component (default: True)
        
    Returns:
        Postprocessed binary mask (0 = background, 255 = brain)
        
    Example:
        >>> mask = np.zeros((256, 256), dtype=np.uint8)
        >>> mask[50:200, 50:200] = 255  # Brain region
        >>> mask[60:70, 60:70] = 0  # Small hole
        >>> cleaned = postprocess_mask(mask, min_size=1000)
        >>> np.sum(cleaned[60:70, 60:70])  # Hole should be filled
        25500
    """
    # Convert to binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Morphological closing (connect nearby regions, fill small holes)
    if closing > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing, closing))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Morphological opening (remove small objects)
    if opening > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening, opening))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Fill holes
    if fill_holes:
        binary_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
    
    # Remove small components
    if min_size > 0:
        binary_mask = morphology.remove_small_objects(
            binary_mask.astype(bool),
            min_size=min_size
        ).astype(np.uint8)
    
    # Keep largest component only
    if keep_largest:
        labeled = measure.label(binary_mask)
        if labeled.max() > 0:
            # Find largest component
            largest_cc = np.argmax(np.bincount(labeled.flat)[1:]) + 1
            binary_mask = (labeled == largest_cc).astype(np.uint8)
    
    # Convert back to 0-255 range
    result = binary_mask * 255
    
    return result


def apply_pipeline(
    img: np.ndarray,
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply complete brain preprocessing pipeline with intermediate outputs.
    
    Pipeline stages (configured via cfg):
    1. N4 bias correction (optional)
    2. RAS reorientation (optional)
    3. Isotropic resampling (optional)
    4. Intensity clipping
    5. Z-score normalization
    6. Histogram matching to NFBS (optional)
    7. Smoothing (Gaussian)
    8. Multiple thresholding methods (Otsu, Yen, Li, Triangle)
    9. Mask postprocessing
    10. Generate overlays and cropped regions
    
    Args:
        img: Input image array (H, W)
        cfg: Configuration dictionary with keys:
            - enable: bool (default True)
            - steps: dict with preprocessing step configs
            - export_overlays: bool (default True)
            - outputs_dir: str (default "outputs/preproc")
        
    Returns:
        Dictionary with keys:
            - stages: List of stage outputs (n4, clipped, normalized, hist_matched, etc.)
            - candidates: Dict of candidate masks (otsu, yen, li, triangle)
            - final: Dict with final_brain_mask, overlay, cropped
            - config_used: Copy of config used
            
    Example:
        >>> img = np.random.rand(256, 256).astype(np.float32)
        >>> cfg = {'enable': True, 'steps': {}}
        >>> result = apply_pipeline(img, cfg)
        >>> 'final' in result
        True
        >>> 'candidates' in result
        True
    """
    # Check if pipeline is enabled
    if not cfg.get('enable', True):
        # Return original image only
        return {
            'stages': {'original': img},
            'candidates': {},
            'final': {},
            'config_used': cfg
        }
    
    # Initialize stages dict
    stages = {'original': img.copy()}
    
    # Get step configs
    steps = cfg.get('steps', {})
    
    # Current processing image
    current = img.copy()
    
    # 1. N4 bias correction
    if steps.get('n4_bias_correction', False):
        current = n4_bias_correction(current)
        stages['n4'] = current.copy()
    
    # 2. RAS reorientation (skip for 2D without affine)
    if steps.get('reorient_to_ras', False):
        # Note: Requires affine from cfg or nifti file
        affine = cfg.get('affine', None)
        if affine is not None:
            current, _ = reorient_to_ras(current, affine)
            stages['reoriented'] = current.copy()
    
    # 3. Isotropic resampling
    if steps.get('resample_isotropic', False):
        target_spacing = steps.get('resample_isotropic')
        if isinstance(target_spacing, (int, float)) and target_spacing > 0:
            current_spacing = cfg.get('current_spacing', None)
            if current_spacing is not None:
                current = resample_isotropic(current, target_spacing, current_spacing)
                stages['resampled'] = current.copy()
    
    # 4. Intensity clipping
    clip_cfg = steps.get('intensity_clip', {})
    if clip_cfg:
        pmin = clip_cfg.get('pmin', 0.5)
        pmax = clip_cfg.get('pmax', 99.5)
        current = intensity_clip(current, pmin, pmax)
        stages['clipped'] = current.copy()
    
    # 5. Z-score normalization
    if steps.get('zscore_norm', False):
        current = zscore_norm(current)
        stages['normalized'] = current.copy()
    
    # 6. Histogram matching
    hist_cfg = steps.get('histogram_match_to_nfbs', {})
    if hist_cfg.get('enabled', False):
        ref_path = hist_cfg.get('reference_stats_path', 'artifacts/nfbs_hist_ref.npz')
        current = histogram_match_to_nfbs(current, ref_path)
        stages['hist_matched'] = current.copy()
    
    # 7. Smoothing for thresholding
    smoothing_cfg = steps.get('smoothing', {})
    gaussian_sigma = smoothing_cfg.get('gaussian_sigma', 1.0)
    
    # 8. Thresholding - get all candidate masks
    thresh_cfg = steps.get('thresholding', {})
    candidates = thresh_cfg.get('candidates', ['otsu', 'yen', 'li', 'triangle'])
    
    # Get all candidate masks
    candidate_masks = adaptive_threshold_mask(current, return_all=True, gaussian_sigma=gaussian_sigma)
    
    # 9. Postprocess masks
    postproc_cfg = thresh_cfg.get('postprocess', {})
    postprocessed_masks = {}
    
    for name, raw_mask in candidate_masks.items():
        cleaned = postprocess_mask(
            raw_mask,
            min_size=postproc_cfg.get('min_size', 5000),
            closing=postproc_cfg.get('closing', 3),
            opening=postproc_cfg.get('opening', 1),
            fill_holes=postproc_cfg.get('fill_holes', True),
            keep_largest=postproc_cfg.get('keep_largest', True)
        )
        postprocessed_masks[name] = cleaned
    
    # Select primary method
    primary_method = thresh_cfg.get('primary', 'otsu')
    final_mask = postprocessed_masks.get(primary_method, postprocessed_masks['otsu'])
    
    # 10. Generate overlay and cropped region
    overlay = create_overlay(img, final_mask)
    cropped = extract_brain_region(img, final_mask)
    
    # Build result
    result = {
        'stages': stages,
        'candidates': postprocessed_masks,
        'final': {
            'final_brain_mask': final_mask,
            'overlay': overlay,
            'cropped': cropped
        },
        'config_used': cfg
    }
    
    return result


def create_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.3
) -> np.ndarray:
    """
    Create colored overlay of mask on image.
    
    Args:
        img: Original grayscale image
        mask: Binary mask (0-255)
        color: RGB color for mask (default: green)
        alpha: Transparency (default: 0.3)
        
    Returns:
        RGB overlay image
    """
    # Normalize image to 0-255 uint8
    img_uint8 = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Convert grayscale to RGB
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    
    # Create colored mask
    mask_colored = np.zeros_like(img_rgb)
    mask_binary = (mask > 0).astype(np.uint8)
    mask_colored[:, :, 0] = mask_binary * color[2]  # BGR format
    mask_colored[:, :, 1] = mask_binary * color[1]
    mask_colored[:, :, 2] = mask_binary * color[0]
    
    # Blend
    overlay = cv2.addWeighted(img_rgb, 1.0, mask_colored, alpha, 0)
    
    return overlay


def extract_brain_region(
    img: np.ndarray,
    mask: np.ndarray,
    pad: int = 5
) -> np.ndarray:
    """
    Extract brain region from image using mask with padding.
    
    Args:
        img: Original image
        mask: Binary brain mask (0-255)
        pad: Padding pixels around bounding box (default: 5)
        
    Returns:
        Cropped brain region
    """
    # Find bounding box
    mask_binary = (mask > 0).astype(np.uint8)
    coords = np.argwhere(mask_binary)
    
    if len(coords) == 0:
        # Empty mask, return original
        return img
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add padding
    h, w = img.shape[:2]
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(h, y_max + pad)
    x_max = min(w, x_max + pad)
    
    # Crop
    cropped = img[y_min:y_max, x_min:x_max]
    
    return cropped
