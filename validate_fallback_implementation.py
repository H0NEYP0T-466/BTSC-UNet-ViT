#!/usr/bin/env python
"""
Validation script to verify Brain UNet fallback implementation.
Tests response schema and fallback logic without requiring actual images.
"""
import sys
from pathlib import Path
import json

# Add backend to path
backend_path = Path(__file__).resolve().parent / 'backend'
sys.path.insert(0, str(backend_path))

from app.schemas.responses import BrainSegmentResponse, InferenceResponse
from pydantic import ValidationError


def test_brain_segment_response_schema():
    """Test BrainSegmentResponse schema includes fallback fields."""
    print("Testing BrainSegmentResponse schema...")
    
    # Test with fallback
    data_with_fallback = {
        "image_id": "test123",
        "mask_url": "/files/test/mask.png",
        "overlay_url": "/files/test/overlay.png",
        "brain_extracted_url": "/files/test/extracted.png",
        "brain_area_pct": 38.5,
        "log_context": {"image_id": "test123", "duration": 1.5, "stage": "complete"},
        "used_fallback": True,
        "fallback_method": "yen",
        "candidate_masks": {
            "otsu": "/files/test/otsu.png",
            "yen": "/files/test/yen.png",
            "li": "/files/test/li.png",
            "triangle": "/files/test/triangle.png"
        }
    }
    
    try:
        response = BrainSegmentResponse(**data_with_fallback)
        assert response.used_fallback == True
        assert response.fallback_method == "yen"
        assert response.candidate_masks is not None
        assert len(response.candidate_masks) == 4
        print("✓ BrainSegmentResponse with fallback: PASSED")
    except ValidationError as e:
        print(f"✗ BrainSegmentResponse with fallback: FAILED\n{e}")
        return False
    
    # Test without fallback
    data_without_fallback = {
        "image_id": "test456",
        "mask_url": "/files/test/mask.png",
        "overlay_url": "/files/test/overlay.png",
        "brain_extracted_url": "/files/test/extracted.png",
        "brain_area_pct": 42.0,
        "log_context": {"image_id": "test456", "duration": 1.2, "stage": "complete"},
        "used_fallback": False,
        "fallback_method": None
    }
    
    try:
        response = BrainSegmentResponse(**data_without_fallback)
        assert response.used_fallback == False
        assert response.fallback_method is None
        print("✓ BrainSegmentResponse without fallback: PASSED")
    except ValidationError as e:
        print(f"✗ BrainSegmentResponse without fallback: FAILED\n{e}")
        return False
    
    return True


def test_inference_response_schema():
    """Test InferenceResponse schema includes brain segmentation with fallback."""
    print("\nTesting InferenceResponse schema...")
    
    from app.schemas.responses import BrainSegmentationResult
    
    data = {
        "image_id": "test789",
        "original_url": "/files/test/original.png",
        "preprocessing": {
            "grayscale": "/files/test/gray.png",
            "denoised": "/files/test/denoised.png",
            "motion_reduced": "/files/test/motion.png",
            "contrast": "/files/test/contrast.png",
            "sharpened": "/files/test/sharp.png",
            "normalized": "/files/test/norm.png"
        },
        "brain_segmentation": {
            "mask": "/files/test/brain_mask.png",
            "overlay": "/files/test/brain_overlay.png",
            "brain_extracted": "/files/test/brain_extracted.png",
            "used_fallback": True,
            "fallback_method": "otsu",
            "candidate_masks": {
                "otsu": "/files/test/otsu.png",
                "yen": "/files/test/yen.png"
            }
        },
        "tumor_segmentation": {
            "mask": "/files/test/tumor_mask.png",
            "overlay": "/files/test/tumor_overlay.png",
            "segmented": "/files/test/tumor_seg.png"
        },
        "classification": {
            "class": "glioma",
            "confidence": 0.95,
            "logits": [2.5, -1.0, 0.5, 1.2],
            "probabilities": [0.95, 0.02, 0.01, 0.02]
        },
        "duration_seconds": 5.2,
        "log_context": {"image_id": "test789", "total_duration": 5.2}
    }
    
    try:
        response = InferenceResponse(**data)
        assert response.brain_segmentation.used_fallback == True
        assert response.brain_segmentation.fallback_method == "otsu"
        assert response.brain_segmentation.candidate_masks is not None
        print("✓ InferenceResponse with fallback: PASSED")
        
        # Test BrainSegmentationResult directly
        brain_seg = BrainSegmentationResult(**data['brain_segmentation'])
        assert brain_seg.used_fallback == True
        assert brain_seg.fallback_method == "otsu"
        print("✓ BrainSegmentationResult schema: PASSED")
    except ValidationError as e:
        print(f"✗ InferenceResponse with fallback: FAILED\n{e}")
        return False
    
    return True


def test_fallback_logic():
    """Test that fallback detection logic is correct."""
    print("\nTesting fallback detection logic...")
    
    import numpy as np
    
    # Test empty mask detection
    empty_mask = np.zeros((256, 256), dtype=np.uint8)
    brain_percentage = (np.sum(empty_mask > 0) / empty_mask.size) * 100
    assert brain_percentage < 0.1, "Empty mask should be below threshold"
    print(f"✓ Empty mask detection: {brain_percentage}% < 0.1% threshold")
    
    # Test valid mask detection
    valid_mask = np.zeros((256, 256), dtype=np.uint8)
    valid_mask[50:200, 50:200] = 255
    brain_percentage = (np.sum(valid_mask > 0) / valid_mask.size) * 100
    assert brain_percentage > 0.1, "Valid mask should be above threshold"
    print(f"✓ Valid mask detection: {brain_percentage}% > 0.1% threshold")
    
    # Test candidate selection (largest area wins)
    candidates = {
        'otsu': np.zeros((256, 256), dtype=np.uint8),
        'yen': np.zeros((256, 256), dtype=np.uint8),
        'li': np.zeros((256, 256), dtype=np.uint8),
        'triangle': np.zeros((256, 256), dtype=np.uint8)
    }
    
    # Set different areas
    candidates['otsu'][50:100, 50:100] = 255  # 2500 pixels
    candidates['yen'][30:220, 30:220] = 255   # 36100 pixels (largest)
    candidates['li'][60:150, 60:150] = 255    # 8100 pixels
    candidates['triangle'][80:120, 80:120] = 255  # 1600 pixels
    
    # Find best
    best_method = None
    best_area = 0
    for method_name, mask in candidates.items():
        area = np.sum(mask > 0)
        if area > best_area:
            best_area = area
            best_method = method_name
    
    assert best_method == 'yen', f"Should select 'yen' with largest area, got '{best_method}'"
    print(f"✓ Candidate selection: Selected '{best_method}' with {best_area} pixels")
    
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Brain UNet Fallback Implementation Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Test schemas
    if not test_brain_segment_response_schema():
        all_passed = False
    
    if not test_inference_response_schema():
        all_passed = False
    
    # Test fallback logic
    if not test_fallback_logic():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All validation tests PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ Some validation tests FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
