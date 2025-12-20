# Pipeline Architecture Changes

This document describes the changes made to the brain tumor detection pipeline.

## Overview

The pipeline has been refactored to improve efficiency by performing classification before segmentation.

## Old Pipeline Flow

```
Input Image
    ↓
Preprocessing
    ↓
Tumor Segmentation (UNet)
    ↓
Classification (ViT)
    ↓
Output
```

**Issues with old approach:**
- UNet segmentation always runs, even for healthy scans
- Computationally expensive segmentation on images without tumors
- Inefficient use of GPU resources

## New Pipeline Flow

```
Input Image
    ↓
Preprocessing
    ↓
Classification (ViT)
    ↓
Decision Point
    ├─ If "notumor": END (skip segmentation)
    └─ If tumor detected:
           ↓
       Tumor Segmentation (UNet)
           ↓
       Output
```

**Benefits of new approach:**
- Early exit for healthy scans (no tumor)
- Reduced computation time for negative cases
- More efficient GPU utilization
- Better user experience (faster results for healthy scans)

## Implementation Details

### 1. Pipeline Service Changes

File: `backend/app/services/pipeline_service.py`

**Key changes:**
- ViT classification moved before segmentation
- Added conditional logic to skip segmentation
- Updated logging to reflect new flow

```python
# Step 2: ViT Classification (runs first)
classification_results = self.vit.classify(
    preprocessed['normalized'],
    image_id=image_id
)

# Step 3: Conditional Segmentation
if predicted_class in ['notumor', 'no_tumor']:
    # Skip segmentation
    logger.info("No tumor detected, skipping segmentation")
else:
    # Perform segmentation for tumor cases
    tumor_segmentation_results = self.tumor_unet.segment_image(...)
```

### 2. ViT Dataset Configuration

File: `backend/app/config.py`

**Changes:**
- Added `VIT_DATASET_ROOT` pointing to classification dataset
- Updated `VIT_CLASS_NAMES` to match folder structure
- Maintained backward compatibility with `SEGMENTED_DATASET_ROOT`

```python
VIT_DATASET_ROOT: Path = BASE_DIR / "dataset" / "Vit_Dataset"
VIT_CLASS_NAMES: List[str] = ["notumor", "glioma", "meningioma", "pituitary"]
```

### 3. ViT Training Updates

Files: 
- `backend/app/models/vit/train_vit.py`
- `backend/app/models/vit/datamodule.py`

**Changes:**
- Updated to use `VIT_DATASET_ROOT` instead of `SEGMENTED_DATASET_ROOT`
- Now loads raw classification images, not segmented outputs
- Proper support for folder-based dataset structure

### 4. New Training Script

File: `train_vit_colab.py`

**Features:**
- Standalone Google Colab training script
- Optimized for T4 GPU (15.6GB VRAM, 12GB RAM)
- Anti-overfitting mechanisms:
  - EarlyStopping (patience=10)
  - ReduceLROnPlateau scheduler
  - Weight decay (L2 regularization)
  - Data augmentation
  - Gradient clipping
- Mixed precision training for speed
- Weighted sampling for class imbalance
- Comprehensive logging and visualization

## Dataset Structure

### ViT Classification Dataset

Location: `/content/dataset/Vit_Dataset/` (Colab) or `backend/dataset/Vit_Dataset/` (local)

```
Vit_Dataset/
├── notumor/      # ~22.5k images (example, varies)
├── glioma/       # ~22.5k images
├── meningioma/   # ~22.5k images
└── pituitary/    # ~22.5k images
Total: ~90k images
```

**Note:** This is the original classification dataset, NOT segmented outputs.

### UNet Segmentation Dataset

Location: `/content/UNet_Dataset/` (Colab)

```
UNet_Dataset/
├── image1.h5
├── image2.h5
└── ...
```

**Format:** HDF5 files with 4-channel images and masks.

## API Response Changes

### Old Response Structure

```json
{
  "image_id": "...",
  "preprocessing": {...},
  "tumor_segmentation": {...},
  "classification": {...}
}
```

### New Response Structure

**For tumor cases:**
```json
{
  "image_id": "...",
  "preprocessing": {...},
  "classification": {
    "class": "glioma",
    "confidence": 0.95
  },
  "tumor_segmentation": {...}
}
```

**For no tumor cases:**
```json
{
  "image_id": "...",
  "preprocessing": {...},
  "classification": {
    "class": "notumor",
    "confidence": 0.98
  }
  // No tumor_segmentation field
}
```

## Performance Impact

### Time Savings

Assuming:
- Preprocessing: 0.5s
- ViT classification: 0.2s
- UNet segmentation: 1.5s

**Old pipeline:**
- All cases: 0.5s + 1.5s + 0.2s = 2.2s

**New pipeline:**
- Tumor cases: 0.5s + 0.2s + 1.5s = 2.2s (same)
- No tumor cases: 0.5s + 0.2s = 0.7s (68% faster!)

**Expected improvement:**
- If 30% of scans are healthy: ~20% overall time reduction
- If 50% of scans are healthy: ~35% overall time reduction

### GPU Utilization

- Better GPU utilization by avoiding unnecessary UNet inference
- Allows for higher throughput on healthy scans
- More efficient batch processing

## Training Recommendations

### For ViT Model

1. **Dataset:** Use full 90k image dataset
2. **Epochs:** 50 (with early stopping)
3. **Batch size:** 32 (T4 GPU)
4. **Learning rate:** 1e-4 with ReduceLROnPlateau
5. **Augmentation:** Enabled (rotations, flips, color jitter)

### Anti-Overfitting Strategy

1. **Early stopping:** Stop if no improvement for 10 epochs
2. **Learning rate reduction:** Halve LR if plateau for 5 epochs
3. **Weight decay:** 0.01 for L2 regularization
4. **Gradient clipping:** Max norm 1.0
5. **Class balancing:** Weighted sampling for imbalanced data

## Migration Guide

### For Existing Deployments

1. **Update configuration:**
   - Ensure `VIT_DATASET_ROOT` points to classification dataset
   - Verify `VIT_CLASS_NAMES` matches folder structure

2. **Retrain ViT model:**
   - Use `train_vit_colab.py` with classification dataset
   - Dataset should NOT be segmented UNet outputs
   - Train for 50 epochs with early stopping

3. **Test pipeline:**
   - Test with tumor images (verify segmentation still works)
   - Test with healthy images (verify segmentation is skipped)
   - Monitor performance improvements

4. **Update API clients:**
   - Handle optional `tumor_segmentation` field
   - Check `classification.class` for tumor type

## Backward Compatibility

- `SEGMENTED_DATASET_ROOT` still available for compatibility
- API response structure is backward compatible (adds optional field)
- Old endpoints continue to work

## Future Improvements

1. **Caching:** Cache ViT results to avoid recomputation
2. **Batch processing:** Process multiple images in single batch
3. **Model optimization:** Quantization for faster inference
4. **Progressive enhancement:** Add uncertainty thresholds

## Conclusion

The new pipeline architecture provides:
- ✅ Faster inference for healthy scans
- ✅ Better resource utilization
- ✅ Improved user experience
- ✅ Maintainable and testable code
- ✅ Backward compatibility

The classification-first approach is more logical and efficient, aligning with how radiologists work: first classify, then segment if needed.
