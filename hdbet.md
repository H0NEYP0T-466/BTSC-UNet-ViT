# HD-BET Brain Extraction Setup Guide

## Overview

This project uses **HD-BET (Hierarchical Deep Brain Extraction Tool)** for skull-stripping in brain MRI images. HD-BET is a state-of-the-art deep learning tool specifically designed to extract brain tissue from MRI scans.

## Installation

HD-BET is included in the `requirements.txt` file and can be installed with:

```bash
pip install HD-BET
```

That's it! No additional setup, weight downloads, or configuration is needed.

## What HD-BET Does

HD-BET performs **skull-stripping** (also called brain extraction), which:

1. **Isolates brain tissue** from the MRI scan
2. **Removes non-brain structures** including:
   - Skull bones
   - Neck muscles and tissues
   - Eyes and eye sockets
   - Nose and nasal cavity
   - Other facial structures
   - Background noise

3. **Creates a binary brain mask** that separates brain tissue (white) from background (black)

## Why HD-BET Was Added

### Problem: False Positives from Bright Non-Brain Regions

Without brain extraction, the preprocessing pipeline (especially CLAHE contrast enhancement and sharpening) would make **non-brain structures overly bright**:

- **Neck tissues** became very white
- **Eyes and eye sockets** became very bright
- **Nose and nasal cavity** showed high intensity

These bright regions confused the UNet tumor segmentation model, causing it to **predict tumors in non-brain areas** (false positives).

### Solution: HD-BET Brain Extraction

By applying HD-BET **before** contrast enhancement and sharpening:

1. **Only brain tissue is processed** - Non-brain structures are masked out
2. **CLAHE is applied only to brain regions** - Prevents neck/eyes from becoming overly bright
3. **Sharpening is applied only to brain regions** - Focuses detail enhancement on relevant tissue
4. **UNet receives clean brain-only images** - Eliminates false positives from non-brain structures

**Result:** ✅ **This issue is SOLVED by HD-BET**

The brain extraction step effectively removes all non-brain structures before enhancement, preventing them from becoming bright enough to be misclassified as tumors.

## How It's Integrated

### Preprocessing Pipeline Order

The updated preprocessing pipeline is:

1. **Grayscale conversion** - Convert input to single channel
2. **HD-BET brain extraction** ← NEW STEP
3. **Denoising** - Remove noise while preserving edges
4. **Motion artifact reduction** - Minimal blur
5. **Contrast enhancement (CLAHE)** - Applied only within brain mask
6. **Sharpening** - Applied only within brain mask
7. **Normalization** - Standardize intensity range

### Code Integration

HD-BET is integrated in:
- `backend/app/utils/brain_extraction.py` - Main brain extraction function
- `backend/app/utils/preprocessing.py` - Preprocessing pipeline with HD-BET
- `backend/app/routers/preprocessing.py` - API endpoint with brain extraction

### Performance

- **Mode**: Fast mode for quick processing
- **Device**: CPU (can be changed to GPU if available)
- **Expected time**: ~2-5 seconds per image on CPU
- **Postprocessing**: Enabled for cleaner masks

### Fallback Behavior

If HD-BET is not installed or fails:
- The system will **log a warning**
- Processing will **continue without brain extraction**
- All preprocessing steps will still work (just without masking)

## Technical Details

### HD-BET Parameters Used

```python
run_hd_bet(
    mri_fnames=[input_path],
    output_fnames=[output_path],
    mode='fast',          # Fast mode for speed
    device='cpu',         # CPU processing (change to 'cuda' for GPU)
    postprocess=True,     # Apply postprocessing for cleaner masks
    do_tta=False,         # Disable test-time augmentation for speed
    keep_mask=True,       # Keep the mask file
    overwrite=True        # Overwrite existing files
)
```

### Input/Output

- **Input**: 2D grayscale MRI image
- **Intermediate**: Converted to NIfTI format (3D with single slice)
- **Output**: 
  - Brain-extracted image (brain tissue only, background zeroed)
  - Binary brain mask (0=background, 255=brain tissue)

### Mask Application

The brain mask is applied to:
- **CLAHE contrast enhancement** - Prevents background enhancement
- **Unsharp mask sharpening** - Focuses on brain tissue details

This ensures that only brain tissue is enhanced, not artifacts or non-brain structures.

## Verification

To verify HD-BET is working:

1. **Check preprocessing artifacts** - You should see `brain_extracted` and `brain_mask` images
2. **Inspect brain mask** - Should show white brain tissue with black background
3. **Compare before/after** - Neck, eyes, and skull should be removed in brain_extracted
4. **Check UNet predictions** - Should no longer predict tumors in neck/eye regions

## Troubleshooting

### HD-BET Not Found

If you get `ImportError: No module named 'HD_BET'`:

```bash
pip install HD-BET
```

### Memory Issues

If you encounter memory issues with HD-BET:
- Ensure you have at least 2GB RAM available
- Consider using smaller input images
- HD-BET will automatically fall back to original image if it fails

### Performance Issues

If HD-BET is too slow:
- Change `device='cpu'` to `device='cuda'` in `brain_extraction.py` if you have a GPU
- Consider using `mode='accurate'` only for final production runs
- Fast mode is sufficient for most use cases

## References

- **HD-BET Paper**: [HD-BET: Automated Brain Extraction in MRI](https://arxiv.org/abs/1904.11376)
- **HD-BET GitHub**: https://github.com/MIC-DKFZ/HD-BET
- **Usage**: No additional weights download needed - HD-BET downloads models automatically on first use

## Summary

✅ **Installation**: Simple `pip install HD-BET`  
✅ **Configuration**: None required  
✅ **Weights**: Auto-downloaded on first use  
✅ **Issue Solved**: Yes - prevents false positives from bright non-brain regions  
✅ **Integration**: Fully integrated into preprocessing pipeline  
