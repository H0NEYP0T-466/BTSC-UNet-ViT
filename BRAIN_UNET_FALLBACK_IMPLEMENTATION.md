# Brain UNet Fallback Implementation

## Overview
This document describes the brain UNet fallback mechanism implemented to handle cases where the deep learning model produces empty or near-empty masks. The system automatically falls back to classical segmentation methods (Otsu, Yen, Li, Triangle) to ensure robust brain extraction.

## Problem Statement
After retraining, the brain UNet model occasionally produces empty/black masks for certain input images. This could be due to:
- Domain shift (test images differ from training data)
- Model convergence issues
- Input preprocessing mismatches
- Edge cases not covered in training data

## Solution Architecture

### Backend Components

#### 1. Fallback Detection (`infer_unet.py`)
```python
# Threshold for detecting empty masks
FALLBACK_THRESHOLD = 0.1  # 0.1% brain area

# After UNet inference
brain_percentage = (np.sum(mask > 0) / mask.size) * 100
if brain_percentage < FALLBACK_THRESHOLD:
    # Trigger fallback to candidate masks
    used_fallback = True
```

#### 2. Candidate Mask Selection
When fallback is triggered, the system:
1. Evaluates all candidate masks (Otsu, Yen, Li, Triangle)
2. Selects the mask with the largest non-zero area
3. Uses this mask as the final brain segmentation
4. Logs which method was used

```python
best_method = None
best_area = 0

for method_name, candidate_mask in candidates.items():
    candidate_area = np.sum(candidate_mask > 0)
    if candidate_area > best_area:
        best_area = candidate_area
        best_method = method_name

if best_method and best_area > 0:
    mask = candidates[best_method]
    used_fallback = True
    fallback_method = best_method
```

#### 3. Response Schema Updates

**BrainSegmentResponse:**
```python
class BrainSegmentResponse(BaseModel):
    image_id: str
    mask_url: str
    overlay_url: str
    brain_extracted_url: str
    brain_area_pct: float
    
    # Advanced preprocessing outputs
    preprocessing_stages: Optional[Dict[str, str]] = None
    candidate_masks: Optional[Dict[str, str]] = None
    
    # NEW: Fallback fields
    used_fallback: bool = False
    fallback_method: Optional[str] = None
```

**InferenceResponse:**
```python
brain_segmentation: {
    mask: string,
    overlay: string,
    brain_extracted: string,
    preprocessing_stages?: { [key: string]: string },
    candidate_masks?: { [key: string]: string },
    used_fallback?: boolean,      # NEW
    fallback_method?: string       # NEW
}
```

### Frontend Components

#### 1. Fallback Indicator Banner
When `used_fallback` is true, a warning banner is displayed:
```tsx
{usedFallback && fallbackMethod && (
  <div className="fallback-indicator warning-banner">
    <span className="warning-icon">⚠️</span>
    <span className="warning-text">
      Brain UNet produced empty mask. Using fallback method: 
      <strong>{fallbackMethod.toUpperCase()}</strong>
    </span>
  </div>
)}
```

#### 2. Candidate Mask Highlighting
The selected fallback method is highlighted with a special badge:
```tsx
<span className={`candidate-badge ${
  usedFallback && name === fallbackMethod 
    ? 'fallback-badge' 
    : ''
}`}>
  {usedFallback && name === fallbackMethod 
    ? '✓ Used (Fallback)' 
    : name === 'otsu' 
      ? 'Primary' 
      : 'Alternative'}
</span>
```

## Usage Examples

### Backend API

#### Brain Segmentation Endpoint
```bash
POST /api/segment-brain
Content-Type: multipart/form-data

file: <brain_mri_image.png>
```

**Response (Normal UNet Success):**
```json
{
  "image_id": "abc123",
  "mask_url": "/files/uploads/abc123/brain_mask.png",
  "overlay_url": "/files/uploads/abc123/brain_overlay.png",
  "brain_extracted_url": "/files/uploads/abc123/brain_extracted.png",
  "brain_area_pct": 42.5,
  "used_fallback": false,
  "fallback_method": null,
  "candidate_masks": {
    "otsu": "/files/uploads/abc123/candidate_otsu.png",
    "yen": "/files/uploads/abc123/candidate_yen.png",
    "li": "/files/uploads/abc123/candidate_li.png",
    "triangle": "/files/uploads/abc123/candidate_triangle.png"
  }
}
```

**Response (Fallback Triggered):**
```json
{
  "image_id": "xyz789",
  "mask_url": "/files/uploads/xyz789/brain_mask.png",
  "overlay_url": "/files/uploads/xyz789/brain_overlay.png",
  "brain_extracted_url": "/files/uploads/xyz789/brain_extracted.png",
  "brain_area_pct": 38.2,
  "used_fallback": true,
  "fallback_method": "yen",
  "candidate_masks": {
    "otsu": "/files/uploads/xyz789/candidate_otsu.png",
    "yen": "/files/uploads/xyz789/candidate_yen.png",
    "li": "/files/uploads/xyz789/candidate_li.png",
    "triangle": "/files/uploads/xyz789/candidate_triangle.png"
  }
}
```

### Frontend Integration

```tsx
import { BrainPreprocessingPanel } from './components/BrainPreprocessingPanel';

function BrainSegmentationView({ result }) {
  return (
    <>
      {/* Main segmentation display */}
      <SegmentationOverlay
        title="Brain Segmentation"
        maskUrl={result.brain_segmentation.mask}
        overlayUrl={result.brain_segmentation.overlay}
        segmentedUrl={result.brain_segmentation.brain_extracted}
      />
      
      {/* Preprocessing and candidate masks panel */}
      <BrainPreprocessingPanel
        stages={result.brain_segmentation.preprocessing_stages}
        candidateMasks={result.brain_segmentation.candidate_masks}
        usedFallback={result.brain_segmentation.used_fallback}
        fallbackMethod={result.brain_segmentation.fallback_method}
      />
    </>
  );
}
```

## Configuration

### Adjusting Fallback Threshold
Edit `backend/app/models/brain_unet/infer_unet.py`:
```python
# Line ~232
if brain_percentage < 0.1:  # Change threshold here (default 0.1%)
    # Trigger fallback
```

### Enabling/Disabling Advanced Preprocessing
Edit `btsc/configs/brain_preproc.yaml`:
```yaml
enable: true  # Set to false to disable candidate mask generation
```

### Changing Primary Method
Edit `btsc/configs/brain_preproc.yaml`:
```yaml
thresholding:
  primary: otsu  # Options: otsu, yen, li, triangle
```

## Testing

### Unit Tests
Run fallback logic tests:
```bash
cd /home/runner/work/BTSC-UNet-ViT/BTSC-UNet-ViT
python -m pytest backend/tests/test_brain_unet_fallback.py -v
```

**Test Coverage:**
- ✅ Fallback triggered on empty mask
- ✅ No fallback on valid mask
- ✅ Fallback selects best candidate (largest area)

### Integration Testing
```bash
# Start backend
cd backend
python -m app.main

# Test with sample image
curl -X POST http://localhost:8080/api/segment-brain \
  -F "file=@/path/to/brain_mri.png" \
  | jq '.used_fallback'
```

## Logging

The system logs detailed information about fallback events:

```
WARNING | Brain UNet produced near-empty mask (0.0234%). 
          Attempting fallback to candidate masks.
INFO    | Using fallback mask from 'yen' method, brain_area=38.2%
INFO    | Brain segmentation completed in 2.345s, 
          brain_area=38.2%, used_fallback=True
```

## Performance Impact

- **Minimal overhead:** Candidate masks are computed during preprocessing (already part of the pipeline)
- **Fallback detection:** < 1ms (simple threshold check)
- **Fallback selection:** < 5ms (area comparison across 4 masks)
- **Total impact:** Negligible (< 0.5% of total inference time)

## Future Enhancements

1. **Ensemble Methods:** Combine multiple candidate masks using voting or averaging
2. **Confidence Scores:** Add confidence metrics for each candidate mask
3. **Smart Selection:** Use additional metrics (circularity, compactness) for selection
4. **User Feedback:** Allow users to manually select alternative methods in UI
5. **Auto-Retraining:** Log fallback cases for model improvement

## Troubleshooting

### Issue: Fallback always triggers
**Cause:** UNet model not trained properly or wrong weights loaded
**Solution:**
1. Check model checkpoint exists: `backend/app/resources/checkpoints/brain_unet/`
2. Verify training completed successfully
3. Check input preprocessing matches training pipeline

### Issue: Candidate masks are empty
**Cause:** Advanced preprocessing disabled or configuration error
**Solution:**
1. Verify `btsc/configs/brain_preproc.yaml` has `enable: true`
2. Check preprocessing pipeline logs for errors
3. Ensure input image is valid brain MRI

### Issue: Wrong fallback method selected
**Cause:** Candidate mask quality varies by input characteristics
**Solution:**
1. Review candidate masks visually in UI
2. Adjust thresholding parameters in config
3. Consider using ensemble methods for better robustness

## References

- **Brain Extraction Methods:**
  - Otsu: Otsu, N. (1979). "A threshold selection method from gray-level histograms"
  - Yen: Yen et al. (1995). "A new criterion for automatic multilevel thresholding"
  - Li: Li & Lee (1993). "Minimum cross entropy thresholding"
  - Triangle: Zack et al. (1977). "Automatic measurement of sister chromatid exchange frequency"

- **Related Files:**
  - `backend/app/models/brain_unet/infer_unet.py` - Main inference logic
  - `btsc/preprocess/brain_extraction.py` - Preprocessing and candidate masks
  - `src/components/BrainPreprocessingPanel/` - UI components
  - `backend/tests/test_brain_unet_fallback.py` - Unit tests

## Support

For issues or questions:
1. Check logs in `backend/` directory
2. Review test cases in `backend/tests/`
3. Open GitHub issue with:
   - Input image characteristics
   - Fallback method used
   - Expected vs actual behavior
   - Relevant log excerpts
