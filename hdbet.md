# HD-BET Brain Extraction Setup Guide

## Quick Start

**New to HD-BET?** Follow these 3 simple steps:

1. **Install HD-BET:**
   ```bash
   pip install HD-BET
   ```

2. **Download model parameters:**
   ```bash
   cd backend
   python setup_hdbet.py
   ```

3. **Start the backend server:**
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8080
   ```

That's it! HD-BET is now ready to use.

---

## Overview

This project uses **HD-BET (Hierarchical Deep Brain Extraction Tool)** for skull-stripping in brain MRI images. HD-BET is a state-of-the-art deep learning tool specifically designed to extract brain tissue from MRI scans.

## Installation and Setup

### Step 1: Install HD-BET Package

HD-BET is included in the `requirements.txt` file. Install it with:

```bash
pip install HD-BET
```

### Step 2: Download HD-BET Model Parameters

HD-BET requires model parameters to be downloaded on first use. The parameters are automatically downloaded to your home directory when you first run HD-BET.

**IMPORTANT**: You must trigger the parameter download by running HD-BET at least once.

#### Quick Setup (Recommended)

We've provided a setup script that automatically downloads the HD-BET parameters:

**Windows:**
```cmd
cd backend
python setup_hdbet.py
```

**Linux/Mac:**
```bash
cd backend
python setup_hdbet.py
```

This script will:
1. Check if HD-BET is installed
2. Download the model parameters (first run only)
3. Run a test prediction to verify everything works
4. Show you where the parameters were saved

The script will display progress and confirm when the setup is complete.

#### What the Script Does

The setup script creates a test image and runs HD-BET once, which triggers the automatic download of model parameters to:
- **Windows**: `C:\Users\<YourUsername>\hd-bet_params\release_2.0.0\`
- **Linux/Mac**: `~/hd-bet_params/release_2.0.0/`

### Step 3: Verify Installation

After running the setup script, verify that the parameters were downloaded:

**Windows:**
```cmd
dir "%USERPROFILE%\hd-bet_params\release_2.0.0"
```

**Linux/Mac:**
```bash
ls -la ~/hd-bet_params/release_2.0.0/
```

You should see files including `dataset.json` and model checkpoint files.

### Alternative: Manual Download

If the automatic download fails, you can manually download the HD-BET parameters:

1. **Download from the official HD-BET repository**: The parameters are hosted by the HD-BET team and should be automatically downloaded. If this fails, check your internet connection and firewall settings.

2. **Create the directory structure**:
   - Windows: `C:\Users\<YourUsername>\hd-bet_params\release_2.0.0\`
   - Linux/Mac: `~/hd-bet_params/release_2.0.0/`

3. **Contact the repository owner** or check the HD-BET GitHub issues if automatic download continues to fail.

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

### Missing Model Parameters Error

If you see an error like:
```
[Errno 2] No such file or directory: 'C:\\Users\\<Username>\\hd-bet_params\\release_2.0.0\\dataset.json'
```

This means the HD-BET model parameters have not been downloaded yet. **Solution:**

1. Run the setup script to download the parameters:
   ```bash
   cd backend
   python setup_hdbet.py
   ```

2. If the setup script fails, try these manual steps:

   **Windows:**
   ```cmd
   mkdir "%USERPROFILE%\hd-bet_params"
   ```
   
   **Linux/Mac:**
   ```bash
   mkdir -p ~/hd-bet_params
   ```
   
   Then run the setup script again.

3. If you're behind a corporate firewall or proxy, the automatic download may fail. Check your network settings and ensure you can access external URLs.

### Setup Script Fails

If `setup_hdbet.py` fails with an error:

1. **Check your internet connection** - HD-BET needs to download ~100MB of model parameters from the internet.

2. **Check Python dependencies** - Ensure all packages in `requirements.txt` are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. **Check disk space** - Ensure you have at least 500MB of free space in your home directory.

4. **Try running with verbose output**:
   ```bash
   python setup_hdbet.py
   ```
   This will show detailed progress and error messages.

### Permission Issues

If you get permission errors when creating directories:

**Windows:**
- Run Command Prompt as Administrator
- Or create the directory manually: `mkdir "%USERPROFILE%\hd-bet_params"`

**Linux/Mac:**
- Check your home directory permissions: `ls -la ~`
- Create the directory manually: `mkdir -p ~/hd-bet_params`

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
