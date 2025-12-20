"""
Simple test to verify the dict filtering logic works correctly.
This test doesn't require the full app setup.
"""
import numpy as np


def test_dict_filtering_logic():
    """Test that the dict filtering logic works as expected."""
    
    # Simulate brain segmentation results with both arrays and dicts
    brain_segmentation_results = {
        'mask': np.zeros((512, 512), dtype=np.uint8),
        'brain_extracted': np.zeros((512, 512), dtype=np.uint8),
        'overlay': np.zeros((512, 512, 3), dtype=np.uint8),
        'preprocessing': {  # This should be skipped
            'stage1': np.zeros((512, 512), dtype=np.uint8),
            'stage2': np.zeros((512, 512), dtype=np.uint8),
        },
        'candidates': {  # This should be skipped
            'mask1': np.zeros((512, 512), dtype=np.uint8),
            'mask2': np.zeros((512, 512), dtype=np.uint8),
        }
    }
    
    # Simulate the filtering logic from pipeline_service.py
    saved_items = []
    skipped_items = []
    
    for seg_type, seg_image in brain_segmentation_results.items():
        # Skip nested dictionaries (e.g., 'preprocessing', 'candidates')
        if isinstance(seg_image, dict):
            skipped_items.append(seg_type)
            continue
        # In the real code, this would call storage.save_artifact()
        saved_items.append(seg_type)
    
    # Verify results
    print(f"Saved items: {saved_items}")
    print(f"Skipped items: {skipped_items}")
    
    assert len(saved_items) == 3, f"Expected 3 saved items, got {len(saved_items)}"
    assert 'mask' in saved_items, "mask should be saved"
    assert 'brain_extracted' in saved_items, "brain_extracted should be saved"
    assert 'overlay' in saved_items, "overlay should be saved"
    
    assert len(skipped_items) == 2, f"Expected 2 skipped items, got {len(skipped_items)}"
    assert 'preprocessing' in skipped_items, "preprocessing dict should be skipped"
    assert 'candidates' in skipped_items, "candidates dict should be skipped"
    
    print("✓ Dict filtering logic test passed!")


def test_all_arrays_case():
    """Test that when all items are arrays, all are saved."""
    
    results = {
        'mask': np.zeros((512, 512), dtype=np.uint8),
        'overlay': np.zeros((512, 512, 3), dtype=np.uint8),
        'segmented': np.zeros((512, 512), dtype=np.uint8),
    }
    
    saved_items = []
    
    for item_type, item_data in results.items():
        if isinstance(item_data, dict):
            continue
        saved_items.append(item_type)
    
    print(f"All arrays case - Saved items: {saved_items}")
    
    assert len(saved_items) == 3, f"Expected 3 saved items, got {len(saved_items)}"
    assert set(saved_items) == {'mask', 'overlay', 'segmented'}
    
    print("✓ All arrays case test passed!")


def test_error_without_filtering():
    """Demonstrate the error that would occur without filtering."""
    
    results = {
        'mask': np.zeros((512, 512), dtype=np.uint8),
        'preprocessing': {  # This is a dict
            'stage1': np.zeros((512, 512), dtype=np.uint8),
        }
    }
    
    # Without filtering, accessing .shape on the dict would fail
    for item_type, item_data in results.items():
        if item_type == 'preprocessing':
            # This would fail: 'dict' object has no attribute 'shape'
            try:
                _ = item_data.shape
                assert False, "Should have raised AttributeError"
            except AttributeError as e:
                print(f"✓ Confirmed error without filtering: {e}")
                assert "'dict' object has no attribute 'shape'" in str(e)


if __name__ == "__main__":
    test_dict_filtering_logic()
    test_all_arrays_case()
    test_error_without_filtering()
    print("\n✓✓✓ All dict filtering tests passed! ✓✓✓")
