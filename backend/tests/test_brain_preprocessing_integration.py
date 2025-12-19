"""
Integration smoke test for brain preprocessing pipeline.
Verifies that the pipeline integrates correctly without breaking existing functionality.
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def test_brain_extraction_module_imports():
    """Test that all brain extraction functions can be imported."""
    try:
        from btsc.preprocess.brain_extraction import (
            intensity_clip,
            zscore_norm,
            otsu_brain_mask,
            adaptive_threshold_mask,
            postprocess_mask,
            create_overlay,
            extract_brain_region,
            apply_pipeline
        )
        print("✓ All brain extraction functions imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import brain extraction functions: {e}")
        return False


def test_config_exists():
    """Test that brain preprocessing config exists."""
    config_path = project_root / "btsc" / "configs" / "brain_preproc.yaml"
    if config_path.exists():
        print(f"✓ Brain preprocessing config found at {config_path}")
        return True
    else:
        print(f"✗ Brain preprocessing config not found at {config_path}")
        return False


def test_nfbs_reference_exists():
    """Test that NFBS reference exists."""
    ref_path = project_root / "artifacts" / "nfbs_hist_ref.npz"
    if ref_path.exists():
        print(f"✓ NFBS reference found at {ref_path}")
        return True
    else:
        print(f"✗ NFBS reference not found at {ref_path}")
        return False


def test_pipeline_runs():
    """Test that the pipeline can run end-to-end."""
    try:
        from btsc.preprocess.brain_extraction import apply_pipeline
        
        # Create synthetic image
        img = np.random.rand(128, 128).astype(np.float32) * 255
        
        # Minimal config (skip N4 and histogram matching)
        cfg = {
            'enable': True,
            'steps': {
                'n4_bias_correction': False,
                'intensity_clip': {'pmin': 1.0, 'pmax': 99.0},
                'zscore_norm': True,
                'histogram_match_to_nfbs': {'enabled': False},
                'smoothing': {'gaussian_sigma': 1.0},
                'thresholding': {
                    'primary': 'otsu',
                    'candidates': ['otsu'],
                    'postprocess': {
                        'min_size': 100,
                        'closing': 3,
                        'opening': 1,
                        'fill_holes': True,
                        'keep_largest': True
                    }
                }
            }
        }
        
        result = apply_pipeline(img, cfg)
        
        # Verify result structure
        assert 'stages' in result, "Missing 'stages' in result"
        assert 'candidates' in result, "Missing 'candidates' in result"
        assert 'final' in result, "Missing 'final' in result"
        assert 'config_used' in result, "Missing 'config_used' in result"
        
        print("✓ Pipeline executed successfully")
        print(f"  - Stages: {list(result['stages'].keys())}")
        print(f"  - Candidates: {list(result['candidates'].keys())}")
        print(f"  - Final: {list(result['final'].keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backwards_compatibility():
    """Test that disabling preprocessing maintains backwards compatibility."""
    try:
        from btsc.preprocess.brain_extraction import apply_pipeline
        
        img = np.random.rand(100, 100).astype(np.float32) * 255
        cfg = {'enable': False}
        
        result = apply_pipeline(img, cfg)
        
        # Should return minimal result with original image
        assert 'stages' in result
        assert 'original' in result['stages']
        assert result['candidates'] == {}
        
        print("✓ Backwards compatibility maintained (disabled mode works)")
        return True
    except Exception as e:
        print(f"✗ Backwards compatibility test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 80)
    print("Brain Preprocessing Pipeline - Integration Smoke Tests")
    print("=" * 80)
    print()
    
    tests = [
        ("Module Imports", test_brain_extraction_module_imports),
        ("Config Exists", test_config_exists),
        ("NFBS Reference Exists", test_nfbs_reference_exists),
        ("Pipeline Runs", test_pipeline_runs),
        ("Backwards Compatibility", test_backwards_compatibility),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Running: {name}")
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
