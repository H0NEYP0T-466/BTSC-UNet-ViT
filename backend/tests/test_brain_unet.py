"""
Test script for Brain UNet model and NFBS dataset.
This script tests:
1. NFBS dataset loading
2. Brain UNet model creation
3. Brain UNet training setup
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

import torch
import numpy as np

def test_brain_unet_model():
    """Test Brain UNet model creation."""
    print("\n" + "=" * 80)
    print("TEST 1: Brain UNet Model Creation")
    print("=" * 80)
    
    try:
        from app.models.brain_unet.model import get_brain_unet_model
        
        # Create model
        model = get_brain_unet_model(
            in_channels=1,
            out_channels=1,
            features=(32, 64, 128, 256, 512)
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully")
        print(f"   Total parameters: {num_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 256, 256)
        output = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nfbs_dataset():
    """Test NFBS dataset loading."""
    print("\n" + "=" * 80)
    print("TEST 2: NFBS Dataset Loading")
    print("=" * 80)
    
    # Note: This test requires the dataset to be present
    # For CI/CD, you can mock this or skip if dataset not present
    
    dataset_path = Path("/content/NFBS_Dataset")
    if not dataset_path.exists():
        # Try alternative path for local testing
        dataset_path = Path("X:/NFBS_Dataset/NFBS_Dataset")
        
    if not dataset_path.exists():
        print(f"⚠️  Dataset not found at {dataset_path}")
        print("   Skipping dataset test (dataset required for actual training)")
        return True
    
    try:
        from app.models.brain_unet.datamodule import NFBSDataset
        
        # Test without caching
        print("\n   Testing dataset WITHOUT caching...")
        dataset_no_cache = NFBSDataset(
            root_dir=dataset_path,
            image_size=(256, 256),
            slice_range=(50, 60),  # Use very limited range for testing
            cache_in_memory=False
        )
        
        print(f"   ✅ Dataset loaded (no cache): {len(dataset_no_cache)} samples")
        
        # Test with caching
        print("\n   Testing dataset WITH caching...")
        dataset_cached = NFBSDataset(
            root_dir=dataset_path,
            image_size=(256, 256),
            slice_range=(50, 60),  # Use very limited range for testing
            cache_in_memory=True
        )
        
        print(f"   ✅ Dataset loaded (cached): {len(dataset_cached)} samples")
        print(f"   ✅ Cache size: {len(dataset_cached.cache)} slices")
        
        if len(dataset_cached) > 0:
            # Test getting a sample
            image, mask = dataset_cached[0]
            
            print(f"\n✅ Sample loaded successfully")
            print(f"   Image shape: {image.shape}")
            print(f"   Image dtype: {image.dtype}")
            print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"   Mask shape: {mask.shape}")
            print(f"   Mask dtype: {mask.dtype}")
            print(f"   Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
            print(f"   Mask values: {torch.unique(mask).numpy()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_brain_unet_utils():
    """Test Brain UNet utility functions."""
    print("\n" + "=" * 80)
    print("TEST 3: Brain UNet Utilities")
    print("=" * 80)
    
    try:
        from app.models.brain_unet.utils import DiceBCELoss, calculate_iou
        
        # Test loss function
        loss_fn = DiceBCELoss()
        
        # Create dummy predictions and targets
        predictions = torch.randn(2, 1, 256, 256)
        targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
        
        loss = loss_fn(predictions, targets)
        
        print(f"✅ Loss function works")
        print(f"   Loss value: {loss.item():.4f}")
        
        # Test IoU calculation
        pred_binary = torch.sigmoid(predictions).numpy()
        target_np = targets.numpy()
        
        iou = calculate_iou(pred_binary, target_np)
        
        print(f"✅ IoU calculation works")
        print(f"   IoU value: {iou:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_brain_unet_inference():
    """Test Brain UNet inference (without trained model)."""
    print("\n" + "=" * 80)
    print("TEST 4: Brain UNet Inference Setup")
    print("=" * 80)
    
    try:
        from app.models.brain_unet.infer_unet import BrainUNetInference
        
        # Note: This will use untrained model since we don't have checkpoint yet
        print("⚠️  Note: Testing with untrained model (no checkpoint available yet)")
        
        # Create inference instance
        inference = BrainUNetInference(device='cpu')
        
        print(f"✅ Inference instance created")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        result = inference.segment_brain(dummy_image)
        
        print(f"✅ Inference works (untrained model)")
        print(f"   Result keys: {list(result.keys())}")
        print(f"   Mask shape: {result['mask'].shape}")
        print(f"   Brain extracted shape: {result['brain_extracted'].shape}")
        print(f"   Overlay shape: {result['overlay'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("BRAIN UNET TEST SUITE")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Model Creation", test_brain_unet_model()))
    results.append(("NFBS Dataset", test_nfbs_dataset()))
    results.append(("Utilities", test_brain_unet_utils()))
    results.append(("Inference Setup", test_brain_unet_inference()))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("\n" + "-" * 80)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print("=" * 80)
    
    # Return exit code
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
