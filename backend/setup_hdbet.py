"""
HD-BET Setup Script

This script downloads the HD-BET model parameters required for brain extraction.
Run this script once after installing HD-BET to ensure the model parameters are available.

Usage:
    python setup_hdbet.py
"""
import sys
import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path


def setup_hdbet():
    """
    Download HD-BET parameters by running a test prediction.
    This triggers the automatic download of model parameters.
    """
    print("=" * 60)
    print("HD-BET Setup Script")
    print("=" * 60)
    print()
    
    try:
        print("Step 1: Importing HD-BET...")
        from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
        import torch
        print("✅ HD-BET imported successfully")
        print()
        
        print("Step 2: Creating test image...")
        # Create a test 3D image (256x256x1)
        test_image = np.random.rand(256, 256, 1).astype(np.float32)
        nifti_img = nib.Nifti1Image(test_image, affine=np.eye(4))
        print("✅ Test image created")
        print()
        
        print("Step 3: Downloading HD-BET model parameters...")
        print("This may take a few minutes on first run...")
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "test_input.nii.gz"
            output_path = Path(temp_dir) / "test_output.nii.gz"
            
            # Save test input
            nib.save(nifti_img, str(input_path))
            
            # Initialize HD-BET predictor (this downloads parameters)
            device = torch.device('cpu')
            print(f"Using device: {device}")
            
            predictor = get_hdbet_predictor(
                use_tta=False, 
                device=device, 
                verbose=True
            )
            
            if predictor is None:
                print("❌ Failed to initialize HD-BET predictor")
                print("Model parameters may not have been downloaded correctly.")
                return False
            
            print("✅ HD-BET predictor initialized successfully")
            print()
            
            print("Step 4: Running test prediction...")
            # Run a test prediction to verify everything works
            try:
                hdbet_predict(
                    input_file_or_folder=str(input_path),
                    output_file_or_folder=str(output_path),
                    predictor=predictor,
                    keep_brain_mask=True,
                    compute_brain_extracted_image=True
                )
                
                # Check if output was created
                if output_path.exists():
                    print("✅ Test prediction completed successfully")
                else:
                    print("⚠️  Test prediction ran but output file not found")
                    return False
                    
            except Exception as e:
                print(f"❌ Test prediction failed: {e}")
                return False
        
        print()
        print("=" * 60)
        print("✅ HD-BET SETUP COMPLETE!")
        print("=" * 60)
        print()
        print("HD-BET is now ready to use.")
        print("Model parameters have been downloaded to your home directory:")
        
        import os
        home_dir = Path.home()
        params_dir = home_dir / "hd-bet_params" / "release_2.0.0"
        print(f"  {params_dir}")
        
        if params_dir.exists():
            print()
            print("Parameters found:")
            for file in params_dir.glob("*"):
                print(f"  - {file.name}")
        
        print()
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import HD-BET: {e}")
        print()
        print("Please install HD-BET first:")
        print("  pip install HD-BET")
        print()
        return False
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        return False


if __name__ == "__main__":
    success = setup_hdbet()
    sys.exit(0 if success else 1)
