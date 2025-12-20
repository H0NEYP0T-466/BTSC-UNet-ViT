# Brain Segmentation Dictionary Error - Fix Summary

## Issue Description
After retraining the model, the FastAPI backend was crashing during inference with the error:
```
ERROR | app.main:run_inference:102 | Inference pipeline failed: 'dict' object has no attribute 'shape'
```

This occurred after the Brain UNet preprocessing step, specifically when trying to save the `brain_preprocessing.png` artifact.

## Root Cause Analysis

### Error Location
The error occurred in `backend/app/services/pipeline_service.py` at line 146-147:
```python
for seg_type, seg_image in brain_segmentation_results.items():
    url = self.storage.save_artifact(seg_image, image_id, f"brain_{seg_type}")
```

### Why It Failed
The `brain_segmentation_results` dictionary returned from `brain_unet.segment_brain()` contains:
- `mask`: numpy array (H, W) - ✅ Can be saved as image
- `brain_extracted`: numpy array (H, W) - ✅ Can be saved as image  
- `overlay`: numpy array (H, W, 3) - ✅ Can be saved as image
- `preprocessing`: **dict** of numpy arrays - ❌ Cannot be saved as single image
- `candidates`: **dict** of numpy arrays - ❌ Cannot be saved as single image

When advanced brain preprocessing is enabled (via `brain_preproc.yaml`), the `preprocessing` and `candidates` fields are added as nested dictionaries. The `save_artifact()` method calls `save_image()`, which expects a numpy array with a `.shape` attribute. When passed a dict, it fails with `'dict' object has no attribute 'shape'`.

## Solution Implemented

### Code Changes
Added defensive filtering in `backend/app/services/pipeline_service.py` to skip nested dictionaries:

**Location 1: Preprocessing artifacts (lines 116-119)**
```python
# Save preprocessing artifacts (only numpy arrays, skip nested dicts)
preprocess_urls = {}
for stage_name, stage_image in preprocessed.items():
    # Skip nested dictionaries for defensive programming
    if isinstance(stage_image, dict):
        continue
    url = self.storage.save_artifact(stage_image, image_id, stage_name)
    preprocess_urls[stage_name] = self.storage.get_artifact_url(url)
```

**Location 2: Brain segmentation artifacts (lines 146-152)**
```python
# Save brain segmentation artifacts (only numpy arrays, skip nested dicts)
brain_segment_urls = {}
for seg_type, seg_image in brain_segmentation_results.items():
    # Skip nested dictionaries (e.g., 'preprocessing', 'candidates')
    if isinstance(seg_image, dict):
        continue
    url = self.storage.save_artifact(seg_image, image_id, f"brain_{seg_type}")
    brain_segment_urls[seg_type] = self.storage.get_artifact_url(url)
```

**Location 3: Tumor segmentation artifacts (lines 174-180)**
```python
# Save tumor segmentation artifacts (only numpy arrays, skip nested dicts)
tumor_segment_urls = {}
for seg_type, seg_image in tumor_segmentation_results.items():
    # Skip nested dictionaries for defensive programming
    if isinstance(seg_image, dict):
        continue
    url = self.storage.save_artifact(seg_image, image_id, f"tumor_{seg_type}")
    tumor_segment_urls[seg_type] = self.storage.get_artifact_url(url)
```

## Testing

### Unit Tests Created
1. **test_dict_filtering_logic.py** - Verifies the filtering logic works correctly:
   - Tests that dict items are skipped
   - Tests that array items are saved
   - Confirms the exact error would occur without filtering

2. **test_pipeline_brain_dict_fix.py** - Integration-style test with mocked dependencies

### Test Results
```
✓ Dict filtering logic test passed!
✓ All arrays case test passed!
✓ Confirmed error without filtering: 'dict' object has no attribute 'shape'
✓✓✓ All dict filtering tests passed! ✓✓✓
```

## Code Review & Security Scan
- ✅ Code review completed: 2 minor non-blocking suggestions about pytest
- ✅ Security scan: No security issues found

## Impact Assessment

### Benefits
- ✅ **Fixes the reported error**: The pipeline will no longer crash when advanced brain preprocessing is enabled
- ✅ **Backward compatible**: Existing functionality is preserved
- ✅ **Defensive programming**: Prevents similar issues in the future
- ✅ **Minimal changes**: Only 12 lines of production code added

### What Still Works
- All numpy array artifacts are saved correctly
- Preprocessing pipeline artifacts (grayscale, denoised, etc.)
- Brain segmentation artifacts (mask, brain_extracted, overlay)
- Tumor segmentation artifacts (mask, overlay, segmented, heatmap)

### What Gets Skipped (by design)
- Nested `preprocessing` dictionary from brain segmentation (internal stages)
- Nested `candidates` dictionary from brain segmentation (candidate masks)

These nested dictionaries were not being displayed in the UI anyway and are only used internally by the brain extraction pipeline.

## Files Changed

1. **backend/app/services/pipeline_service.py** - Main fix (12 lines added)
2. **backend/tests/test_dict_filtering_logic.py** - Unit tests (104 lines)
3. **backend/tests/test_pipeline_brain_dict_fix.py** - Integration tests (105 lines)

Total: 221 lines added, 3 lines removed

## How to Verify the Fix

### Before the Fix
Running inference would fail with:
```
ERROR | app.main:run_inference:102 | Inference pipeline failed: 'dict' object has no attribute 'shape'
```

### After the Fix
1. Start the backend: `uvicorn app.main:app --reload --port 8080`
2. Upload an image through the UI or API
3. The inference pipeline completes successfully
4. All artifacts are saved and accessible:
   - Original image
   - Preprocessing stages (grayscale, denoised, etc.)
   - Brain mask, brain extracted, brain overlay
   - Tumor mask, overlay, segmented, heatmap
   - Classification results

## Conclusion

This fix resolves the critical bug that prevented the inference pipeline from working after retraining the model with advanced brain preprocessing. The solution is minimal, surgical, and follows defensive programming principles to prevent similar issues in the future.
