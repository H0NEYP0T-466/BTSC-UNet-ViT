# Brain Segmentation Empty Mask Fix

## Problem Statement

The brain segmentation UNet model was producing empty masks during inference despite working correctly during training and validation. This was occurring when users uploaded PNG/JPG images for prediction.

## Root Cause Analysis

### Training Data Format
The brain UNet model was trained on the NFBS dataset which consists of:
- **Format**: NIfTI files (.nii.gz) containing 3D MRI volumes
- **Preprocessing**: Min-max normalization to [0, 1] range
  ```python
  # From datamodule.py line 232-233
  if image_slice.max() > 0:
      image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
  ```
- **Input distribution**: Values uniformly distributed in [0, 1]

### Inference Data Format (BROKEN)
During inference with uploaded PNG/JPG images:
- **Format**: RGB/Grayscale images (PNG/JPG) with values in [0, 255]
- **Preprocessing**: Advanced preprocessing pipeline with zscore normalization
  ```python
  # OLD APPROACH (BROKEN)
  # Applied zscore normalization: (x - mean) / std
  # This creates values with mean ≈ 0, std ≈ 1 (includes negative values!)
  model_input = preproc_result['stages']['normalized']  # zscore normalized
  
  # Then tried to rescale (but distribution is completely different)
  image_normalized = (model_input - model_input.min()) / (model_input.max() - model_input.min() + 1e-8)
  ```
- **Input distribution**: After rescaling, the distribution shape differs significantly from training data

### Why This Caused Empty Masks

The UNet model learned to recognize brain tissue based on the specific intensity distribution from training data (min-max normalized [0, 1]). When presented with differently distributed data (zscore → rescaled), the model couldn't recognize the patterns it learned, resulting in:
- Low confidence predictions across the entire image
- Thresholding at 0.5 produces nearly empty masks
- Brain percentage < 0.1% triggers fallback mechanism

## Solution

### Fix Applied
Always use the **original image with min-max normalization** for model input, matching the training data format exactly:

```python
# NEW APPROACH (FIXED) - from infer_unet.py lines 203-210
# FIX: Always use original image for model input to match training data format
# Training data uses min-max normalization to [0, 1] on raw NIfTI data
# We must do the same here regardless of preprocessing
model_input = original_image.astype(np.float32)

# Normalize to [0, 1] using min-max normalization (SAME AS TRAINING)
# This is critical for the model to work correctly
image_normalized = (model_input - model_input.min()) / (model_input.max() - model_input.min() + 1e-8)
```

### Key Changes
1. **Removed dependency on preprocessing pipeline output** for model input
2. **Always use original image** with direct min-max normalization
3. **Preprocessing pipeline still runs** to generate candidate masks for visualization/fallback
4. **Distribution now matches training** ensuring model works as expected

## Additional Enhancement: Candidate Overlay Visualizations

As requested, all 4 brain segmentation algorithms now display their masks applied on the original image:

### Algorithms Visualized
1. **Otsu** - Minimizes intra-class variance (primary method)
2. **Yen** - Good for bimodal distributions with unequal peaks
3. **Li** - Minimum cross-entropy thresholding
4. **Triangle** - Good for skewed histograms

### Backend Changes
- Added `candidate_overlays` field to store overlay images
- Each candidate mask is overlaid on the original image with green color (30% transparency)
- Overlays are saved and returned via API alongside candidate masks

### Frontend Changes
- Updated `BrainPreprocessingPanel` component with new section
- Displays two sections:
  1. **Binary Masks**: Raw masks from each algorithm
  2. **Applied on Original Image**: Overlays showing masks on the original image
- Allows visual comparison of all 4 methods

## Files Modified

### Backend
- `backend/app/models/brain_unet/infer_unet.py`: Fixed normalization logic, added overlay generation
- `backend/app/routers/brain_segmentation.py`: Added candidate_overlays handling
- `backend/app/schemas/responses.py`: Added candidate_overlays field to schemas
- `backend/app/services/pipeline_service.py`: Added candidate_overlays to pipeline

### Frontend
- `src/components/BrainPreprocessingPanel/BrainPreprocessingPanel.tsx`: Added overlay section
- `src/pages/HomePage.tsx`: Pass candidate_overlays to component
- `src/services/types.ts`: Added candidate_overlays to TypeScript types

## Testing Recommendations

1. **Test with PNG images** (typical user upload format)
2. **Test with JPG images** (compressed format)
3. **Verify brain masks are no longer empty** (brain percentage > 0.1%)
4. **Verify all 4 algorithms display** both masks and overlays
5. **Test fallback mechanism** still works if UNet fails

## Expected Behavior After Fix

1. **Empty mask issue resolved**: Brain UNet should produce meaningful masks (>0.1% brain area)
2. **Consistent with training**: Model receives same data distribution as during training
3. **Visual comparison enabled**: Users can see all 4 algorithm results with overlays
4. **Fallback still available**: If UNet still fails, best candidate mask is used

## Technical Notes

### Why Min-Max Normalization?
- **Training consistency**: Matches exactly what the model learned
- **Simple and effective**: No assumptions about data distribution
- **Preserves relative intensities**: Brain/non-brain contrast maintained
- **Works with any input range**: Handles [0, 255] or already normalized data

### Why Not Zscore?
- **Changes distribution shape**: Mean-centering and standardization alter the data characteristics
- **Sensitive to outliers**: Extreme values can skew normalization
- **Different from training**: Model hasn't learned this distribution
- **Requires rescaling**: Additional transformation introduces more differences

### Preprocessing Pipeline Still Useful
While we don't use it for model input, the advanced preprocessing pipeline:
- Generates candidate masks for fallback
- Provides visualization of different extraction methods
- Offers alternative approaches if UNet fails
- Helps users understand the segmentation process
