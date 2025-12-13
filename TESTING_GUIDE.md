# Testing Guide for Pretrained UNet Implementation

This document describes how to test the pretrained UNet implementation.

## Prerequisites

Before testing, ensure:

1. Dependencies are installed:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Pretrained model is downloaded:
   ```bash
   cd backend
   python -m app.models.pretrained_unet.download_model
   ```

3. Backend server is running:
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Manual Testing Steps

### 1. Test Model Download

```bash
cd backend
python -m app.models.pretrained_unet.download_model
```

**Expected Output:**
- Model file created at `backend/resources/checkpoints/pretrained_unet/unet_pretrained.pth`
- Success message displayed
- File size ~150-200MB

### 2. Test API Health Check

```bash
curl http://localhost:8000/api/health
```

**Expected Output:**
```json
{
  "status": "healthy",
  "message": "BTSC-UNet-ViT API is running"
}
```

### 3. Test Preprocessing Pipeline

Upload a test image for preprocessing:

```bash
curl -X POST "http://localhost:8000/api/preprocess" \
  -F "file=@path/to/brain_mri.jpg" \
  > preprocessing_result.json
```

**Expected Output:**
- JSON response with URLs for all preprocessing stages
- Image files saved in `backend/resources/artifacts/`
- Stages: grayscale, denoised, motion_reduced, contrast, sharpened, normalized

### 4. Test Pretrained UNet Segmentation

Upload a test image for segmentation:

```bash
curl -X POST "http://localhost:8000/api/segment" \
  -F "file=@path/to/brain_mri.jpg" \
  > segmentation_result.json
```

**Expected Output:**
- JSON response with mask, overlay, and segmented URLs
- Mask shows only tumor regions (not whole brain)
- Mask area percentage between 0-30% (typical for tumor)

### 5. Test Full Inference Pipeline

```bash
curl -X POST "http://localhost:8000/api/inference" \
  -F "file=@path/to/brain_mri.jpg" \
  > inference_result.json
```

**Expected Output:**
- Complete pipeline results including:
  - Preprocessing stages
  - Segmentation mask and overlay
  - Classification results with confidence
- Duration: 3-5 seconds (CPU), 1-2 seconds (GPU)

## Verification Checklist

### Preprocessing Quality

- [ ] Grayscale conversion works correctly
- [ ] Denoising reduces noise without excessive blur
- [ ] Motion reduction preserves edges (no excessive blur)
- [ ] Contrast enhancement improves visibility
- [ ] Sharpening enhances edges
- [ ] Normalized image maintains detail

### Segmentation Quality

- [ ] Mask highlights only tumor regions
- [ ] No segmentation of whole brain
- [ ] Small noise artifacts are filtered out
- [ ] Mask boundaries are clean
- [ ] Overlay visualization is clear
- [ ] Cropped image shows only tumor area

### Frontend Display

- [ ] All preprocessing stages display correctly
- [ ] Final normalized image has "→ To Models" badge
- [ ] Final image has cyan glow/highlight
- [ ] Segmentation mask displays correctly
- [ ] Overlay shows red tumor regions
- [ ] Classification results display with confidence

## Automated Tests

### Run Backend Unit Tests

```bash
cd backend
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Test preprocessing
pytest tests/test_preprocessing.py -v

# Test UNet inference
pytest tests/test_unet_inference.py -v

# Test API endpoints
pytest tests/test_api.py -v
```

### Run Frontend Linter

```bash
npm run lint
```

### Build Frontend

```bash
npm run build
```

## Integration Testing

### Test Model Switching

1. Set `USE_PRETRAINED_UNET = True` in `backend/app/config.py`
2. Restart backend server
3. Upload test image and verify it uses pretrained model (check logs)
4. Set `USE_PRETRAINED_UNET = False`
5. Restart backend server
6. Upload test image and verify it uses local model (check logs)

### Test Edge Cases

1. **Empty image**: Should return error
2. **Very small image**: Should be resized and processed
3. **Very large image**: Should be processed efficiently
4. **No tumor image**: Should return minimal mask
5. **Multiple tumors**: Should segment all regions

## Performance Testing

### Measure Inference Time

```python
import time
import requests

# Test multiple images
for i in range(10):
    start = time.time()
    with open(f'test_image_{i}.jpg', 'rb') as f:
        response = requests.post(
            'http://localhost:8000/api/inference',
            files={'file': f}
        )
    duration = time.time() - start
    print(f"Image {i}: {duration:.2f}s")
```

**Expected:**
- CPU: 3-5 seconds per image
- GPU: 1-2 seconds per image

### Memory Usage

Monitor memory during processing:

```bash
# Linux/Mac
watch -n 1 'ps aux | grep uvicorn'

# Or use htop
htop
```

**Expected:**
- Idle: ~200-300MB
- During inference: ~500-700MB
- Peak: <1GB

## Log Verification

Check logs for proper execution:

```bash
tail -f backend/resources/logs/app.log
```

**Expected Log Sequence:**
1. "Initializing Pretrained UNet inference"
2. "Loading checkpoint from..."
3. "Pretrained UNet segmentation started"
4. "Pretrained UNet inference completed"
5. "Passing to next layer: ViT classification"

## Troubleshooting Tests

### Model Not Found

**Symptom:** Error about missing checkpoint

**Solution:**
```bash
cd backend
python -m app.models.pretrained_unet.download_model
```

### Import Errors

**Symptom:** ModuleNotFoundError for monai, torch, etc.

**Solution:**
```bash
cd backend
pip install -r requirements.txt
```

### Poor Segmentation

**Symptom:** Mask covers whole image or has too much noise

**Solution:**
- Verify preprocessing is working correctly
- Check motion reduction isn't blurring too much
- Adjust threshold in pretrained UNet inference (default: 0.5)

### Frontend Not Showing Images

**Symptom:** Images not displayed in frontend

**Solution:**
- Check CORS settings in backend
- Verify artifact URLs are correct
- Check browser console for errors
- Verify backend server is accessible

## Success Criteria

All tests pass if:

✅ Model downloads successfully  
✅ API endpoints respond correctly  
✅ Preprocessing preserves image quality  
✅ Segmentation highlights only tumors  
✅ No excessive blur in motion reduction  
✅ Frontend displays final image indicator  
✅ Full pipeline completes in <5 seconds  
✅ Logs show proper execution flow  
✅ Memory usage stays under 1GB  
✅ Model switching works correctly  

## Continuous Integration

For CI/CD pipelines, add these checks:

```yaml
# .github/workflows/test.yml
- name: Test Backend
  run: |
    cd backend
    pip install -r requirements.txt
    pytest tests/ -v

- name: Test Frontend
  run: |
    npm install
    npm run lint
    npm run build

- name: Verify Model Structure
  run: |
    cd backend
    python -m app.models.pretrained_unet.download_model
    test -f resources/checkpoints/pretrained_unet/unet_pretrained.pth
```

## Reporting Issues

When reporting issues, include:

1. Steps to reproduce
2. Error messages from logs
3. System information (OS, Python version, CUDA version)
4. Sample images (if possible)
5. Expected vs actual behavior

## Next Steps After Testing

1. ✅ Verify all tests pass
2. Deploy to staging environment
3. Run load tests with multiple concurrent users
4. Fine-tune model on production dataset
5. Monitor performance metrics
6. Set up automated testing in CI/CD

---

**Note:** These tests verify the pretrained UNet implementation works correctly. For production deployment, additional tests for security, performance, and edge cases should be added.
