# Brain Segmentation Preprocessing Pipeline

## Overview

The brain segmentation preprocessing pipeline addresses domain shift between training data (NFBS) and external inference data (e.g., Kaggle datasets). This ensures robust brain extraction across different MRI sources.

## Problem Statement

- **Issue**: Brain segmentation quality degrades across training epochs
- **Root Cause**: Domain shift between NFBS training data and external inference data
- **Solution**: Harmonize external data to NFBS characteristics through advanced preprocessing

## Features

### 1. Bias Field Correction
- **N4ITK algorithm** corrects intensity non-uniformity in MRI scans
- Critical for consistent segmentation across different scanners

### 2. Intensity Normalization
- **Z-score normalization**: Centers data around 0 with unit variance
- **Intensity clipping**: Removes outliers that can skew normalization
- **Histogram matching**: Aligns intensity distribution to NFBS reference

### 3. Multiple Brain Extraction Methods
Four thresholding algorithms for robust brain extraction:
- **Otsu**: Minimizes intra-class variance (default)
- **Yen**: Optimal for bimodal distributions with unequal peaks
- **Li**: Minimum cross-entropy thresholding
- **Triangle**: Good for skewed histograms

### 4. Morphological Postprocessing
- **Closing**: Fills small holes and connects nearby regions
- **Opening**: Removes small objects and noise
- **Hole filling**: Removes internal holes in brain mask
- **Component filtering**: Keeps largest connected component (brain)

### 5. Visualization
- Preprocessing stages displayed in UI
- Candidate masks shown side-by-side for comparison
- Final mask, overlay, and cropped region displayed

## Configuration

### Config File: `btsc/configs/brain_preproc.yaml`

```yaml
enable: true  # Enable/disable preprocessing pipeline

steps:
  # N4 bias field correction (requires SimpleITK)
  n4_bias_correction: true
  
  # Reorient to RAS (Right-Anterior-Superior) standard
  reorient_to_ras: false  # Requires affine from NIfTI
  
  # Resample to isotropic spacing
  resample_isotropic: false  # Set to 1.0 for 1mm isotropic
  
  # Intensity clipping (remove outliers)
  intensity_clip:
    pmin: 0.5    # Lower percentile
    pmax: 99.5   # Upper percentile
  
  # Z-score normalization
  zscore_norm: true
  
  # Histogram matching to NFBS reference
  histogram_match_to_nfbs:
    enabled: true
    reference_stats_path: artifacts/nfbs_hist_ref.npz
  
  # Smoothing before thresholding
  smoothing:
    gaussian_sigma: 1.0
  
  # Brain extraction via thresholding
  thresholding:
    primary: otsu  # Primary method
    candidates:    # Methods to compute for comparison
      - otsu
      - yen
      - li
      - triangle
    
    postprocess:
      min_size: 5000      # Min component size (pixels)
      closing: 3          # Closing kernel size
      opening: 1          # Opening kernel size
      fill_holes: true    # Fill internal holes
      keep_largest: true  # Keep only largest component

# Export visualization overlays
export_overlays: true

# Output directory for intermediate results
outputs_dir: outputs/preproc
```

## Usage

### 1. Generate NFBS Histogram Reference

The histogram reference harmonizes external data to NFBS characteristics:

```bash
python tools/build_nfbs_hist_ref.py \
  --nfbs_path /path/to/nfbs/dataset \
  --output artifacts/nfbs_hist_ref.npz \
  --max_images 50
```

**Arguments:**
- `--nfbs_path`: Path to NFBS dataset directory
- `--output`: Output path for reference file (default: `artifacts/nfbs_hist_ref.npz`)
- `--max_images`: Max images to process (default: 50)
- `--num_bins`: Histogram bins (default: 256)

**Note**: A placeholder reference is included. Replace with real NFBS data for production.

### 2. Brain Segmentation Inference

The preprocessing pipeline runs automatically during brain segmentation:

```python
from app.models.brain_unet.infer_unet import get_brain_unet_inference

# Initialize with advanced preprocessing enabled (default)
brain_unet = get_brain_unet_inference(enable_advanced_preproc=True)

# Segment brain
result = brain_unet.segment_brain(
    image,
    image_id="sample_001",
    save_intermediates=True
)

# Access results
mask = result['mask']
brain_extracted = result['brain_extracted']
overlay = result['overlay']

# Access preprocessing stages (if enabled)
if 'preprocessing' in result:
    stages = result['preprocessing']  # Dict of stage images
    
# Access candidate masks (if enabled)
if 'candidates' in result:
    candidates = result['candidates']  # Dict of candidate masks
```

### 3. API Usage

The `/segment-brain` endpoint automatically includes preprocessing results:

```bash
curl -X POST "http://localhost:8000/api/segment-brain" \
  -F "file=@brain_mri.jpg"
```

**Response includes:**
```json
{
  "image_id": "abc123",
  "mask_url": "...",
  "overlay_url": "...",
  "brain_extracted_url": "...",
  "brain_area_pct": 45.2,
  "preprocessing_stages": {
    "original": "...",
    "n4": "...",
    "clipped": "...",
    "normalized": "...",
    "hist_matched": "..."
  },
  "candidate_masks": {
    "otsu": "...",
    "yen": "...",
    "li": "...",
    "triangle": "..."
  }
}
```

### 4. Training with Augmentation

Enhanced training with stronger augmentation and stabilization:

```python
from app.models.brain_unet.datamodule import get_train_augmentation, get_nfbs_dataloaders
from app.models.brain_unet.train_unet import BrainUNetTrainer
from app.models.brain_unet.model import get_brain_unet_model

# Get augmentation pipeline
train_transform = get_train_augmentation()

# Create dataloaders with augmentation
train_loader, val_loader = get_nfbs_dataloaders(
    root_dir="/path/to/nfbs",
    transform=train_transform,
    batch_size=16
)

# Create model with dropout
model = get_brain_unet_model(
    in_channels=1,
    out_channels=1,
    features=(32, 64, 128, 256, 512),
    bottleneck_dropout=0.2
)

# Train with early stopping and TensorBoard
trainer = BrainUNetTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-4,
    early_stopping_patience=10,
    tensorboard_dir="checkpoints/brain_unet/tensorboard"
)

trainer.train(num_epochs=100)
```

**Training features:**
- **Early stopping**: Stops if no improvement for 10 epochs
- **Learning rate schedule**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Gradient clipping**: Max norm = 1.0
- **Weight decay**: 1e-4
- **Dropout**: 0.2 in bottleneck
- **TensorBoard logging**: Loss, Dice, IoU, accuracy, learning rate

**View training progress:**
```bash
tensorboard --logdir checkpoints/brain_unet/tensorboard
```

### 5. Data Augmentation

Strong augmentation for brain segmentation training:

```python
from app.models.brain_unet.datamodule import get_train_augmentation

# Get augmentation pipeline
transform = get_train_augmentation()

# Augmentation includes:
# - Horizontal flip (50% probability)
# - Rotation ±10° (50% probability)
# - Brightness/contrast adjustment (50% probability)
# - Gamma adjustment (50% probability)
# - Gaussian noise (30% probability)
```

## UI Integration

The frontend automatically displays preprocessing results when available:

**Brain Segmentation Preprocessing Panel shows:**
1. **Preprocessing Stages**: All intermediate processing steps
2. **Candidate Masks**: Side-by-side comparison of thresholding methods
3. **Final Results**: Binary mask, overlay, and cropped region

The panel appears automatically when preprocessing data is present in the API response.

## Testing

Run unit tests for the preprocessing module:

```bash
cd backend
pytest tests/test_brain_extraction.py -v
```

**Tests cover:**
- Intensity clipping
- Z-score normalization
- Otsu thresholding
- Adaptive thresholding (all methods)
- Mask postprocessing
- Pipeline orchestration
- Edge cases (empty masks, noise stability)

## Performance Considerations

### Memory Usage
- N4 bias correction: ~2x input image size
- Histogram matching: ~1x input image size
- Multiple thresholding: ~4x mask size (small)

### Processing Time (256x256 image)
- N4 bias correction: ~1-2 seconds
- Intensity normalization: <0.1 seconds
- Histogram matching: <0.1 seconds
- Thresholding (all methods): <0.5 seconds
- Mask postprocessing: <0.2 seconds
- **Total**: ~2-3 seconds

### Optimization Tips
1. Disable N4 if speed is critical: `n4_bias_correction: false`
2. Reduce candidate methods: Keep only `otsu` for faster inference
3. Skip histogram matching if no reference available
4. Use GPU for model inference (already optimized)

## Troubleshooting

### Issue: NFBS reference not found
**Solution**: Generate reference using `tools/build_nfbs_hist_ref.py` or disable histogram matching:
```yaml
histogram_match_to_nfbs:
  enabled: false
```

### Issue: N4 correction fails
**Solution**: Ensure SimpleITK is installed:
```bash
pip install SimpleITK==2.3.1
```

### Issue: Empty brain mask
**Solution**:
1. Check input image quality
2. Try different thresholding method (yen, li, triangle)
3. Adjust postprocessing parameters (reduce min_size)

### Issue: Training doesn't improve
**Solution**:
1. Verify data augmentation is enabled
2. Check learning rate (try 1e-4 to 1e-3)
3. Monitor TensorBoard logs for gradient/loss patterns
4. Ensure weight decay and dropout are enabled
5. Verify training/validation split is balanced

## Backwards Compatibility

All new features are **opt-in** and backwards compatible:

- Default config keeps tumor segmentation unchanged
- Preprocessing can be disabled: `enable: false`
- Missing preprocessing fields in API response are handled gracefully
- Existing inference code works without modifications

## References

- **N4ITK**: Tustison et al., "N4ITK: Improved N3 Bias Correction", IEEE TMI 2010
- **Otsu**: Otsu, "A Threshold Selection Method from Gray-Level Histograms", IEEE SMC 1979
- **Yen**: Yen et al., "A New Criterion for Automatic Multilevel Thresholding", IEEE TIP 1995
- **Li**: Li & Lee, "Minimum Cross Entropy Thresholding", Pattern Recognition 1993

## Support

For issues or questions:
1. Check this documentation
2. Run unit tests: `pytest tests/test_brain_extraction.py -v`
3. Enable verbose logging in `btsc/configs/brain_preproc.yaml`
4. Check TensorBoard for training issues
5. Open an issue on GitHub with logs and configuration
