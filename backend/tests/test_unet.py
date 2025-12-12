"""
Comprehensive UNet Model Diagnostic Test
Run this to diagnose black mask issues with pretrained models. 

Usage:
    cd backend
    python test_unet_model.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import cv2
import h5py
from typing import Dict, Any

# Add backend to path
sys. path.insert(0, str(Path(__file__).parent))

from app.models.unet. model import get_unet_model
from app.config import settings
from app.models.unet.infer_unet import UNetInference


class UNetDiagnostics: 
    """Comprehensive UNet model diagnostics."""
    
    def __init__(self):
        self.results = {}
        self.device = 'cuda' if torch.cuda. is_available() else 'cpu'
        print(f"🔧 UNet Diagnostics Tool")
        print(f"Device: {self.device}")
        print("=" * 60)
    
    def test_1_checkpoint_exists(self) -> bool:
        """Test 1: Check if checkpoint file exists."""
        print("\n[Test 1] Checkpoint File Check")
        print("-" * 40)
        
        checkpoint_path = settings. CHECKPOINTS_UNET / settings.UNET_CHECKPOINT_NAME
        exists = checkpoint_path.exists()
        
        print(f"Checkpoint path:  {checkpoint_path}")
        print(f"File exists: {'✓ YES' if exists else '✗ NO'}")
        
        if exists:
            file_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
            print(f"File size: {file_size:.2f} MB")
            print(f"File extension: {checkpoint_path.suffix}")
            
            if checkpoint_path.suffix == '.h5':
                self._inspect_h5_structure(checkpoint_path)
        
        self.results['checkpoint_exists'] = exists
        return exists
    
    def _inspect_h5_structure(self, h5_path: Path):
        """Inspect H5 file structure."""
        print("\n  H5 File Structure:")
        try:
            with h5py. File(h5_path, 'r') as f:
                print(f"  Root keys: {list(f.keys())}")
                
                def print_structure(group, indent=2):
                    for key in list(group.keys())[:10]:  # Limit to first 10
                        print("  " * indent + f"├─ {key}")
                        if isinstance(group[key], h5py.Group):
                            print_structure(group[key], indent + 1)
                        else:
                            shape = group[key].shape
                            dtype = group[key].dtype
                            print("  " * (indent + 1) + f"└─ shape: {shape}, dtype: {dtype}")
                
                print_structure(f)
        except Exception as e:
            print(f"  ✗ Error reading H5: {e}")
    
    def test_2_model_architecture(self) -> Dict[str, Any]:
        """Test 2: Check PyTorch model architecture."""
        print("\n[Test 2] Model Architecture")
        print("-" * 40)
        
        try:
            model = get_unet_model(
                in_channels=settings.UNET_IN_CHANNELS,
                out_channels=settings.UNET_OUT_CHANNELS,
                features=settings.UNET_CHANNELS
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p. numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Input channels: {settings.UNET_IN_CHANNELS}")
            print(f"Output channels: {settings.UNET_OUT_CHANNELS}")
            print(f"Features: {settings.UNET_CHANNELS}")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
            
            self.results['model_params'] = total_params
            return {'success': True, 'params': total_params}
            
        except Exception as e:
            print(f"✗ Error creating model: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_3_weight_loading(self) -> bool:
        """Test 3: Check if weights loaded correctly."""
        print("\n[Test 3] Weight Loading Verification")
        print("-" * 40)
        
        try: 
            inference = UNetInference()
            model = inference.model
            
            # Check first layer weights
            first_param = next(model.parameters())
            weight_stats = {
                'mean': first_param. mean().item(),
                'std': first_param.std().item(),
                'min': first_param.min().item(),
                'max': first_param.max().item(),
                'abs_mean': first_param.abs().mean().item()
            }
            
            print("First layer weight statistics:")
            for key, val in weight_stats.items():
                print(f"  {key}: {val:.6f}")
            
            # Check if weights are initialized (not random)
            is_random = abs(weight_stats['mean']) < 0.01 and 0.9 < weight_stats['std'] < 1.1
            is_zero = weight_stats['abs_mean'] < 1e-6
            
            if is_zero: 
                print("\n⚠️  WARNING: Weights appear to be zeros!")
                print("   Model weights did NOT load correctly.")
                loaded = False
            elif is_random:
                print("\n⚠️  WARNING: Weights appear random (untrained)!")
                print("   Model might be using random initialization.")
                loaded = False
            else:
                print("\n✓ Weights appear to be trained/loaded")
                loaded = True
            
            self.results['weights_loaded'] = loaded
            self.results['weight_stats'] = weight_stats
            return loaded
            
        except Exception as e:
            print(f"✗ Error checking weights: {e}")
            return False
    
    def test_4_model_inference_dummy(self) -> Dict[str, Any]:
        """Test 4: Run inference with dummy input."""
        print("\n[Test 4] Dummy Inference Test")
        print("-" * 40)
        
        try: 
            inference = UNetInference()
            
            # Create dummy input (random noise)
            dummy_input = torch.randn(1, settings. UNET_IN_CHANNELS, 256, 256).to(self.device)
            
            print(f"Dummy input shape:  {dummy_input.shape}")
            print(f"Dummy input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
            
            # Run inference
            with torch.no_grad():
                output = inference.model(dummy_input)
            
            # Apply sigmoid
            sigmoid_output = torch.sigmoid(output)
            
            stats = {
                'output_shape': output.shape,
                'raw_min': output.min().item(),
                'raw_max': output. max().item(),
                'raw_mean': output.mean().item(),
                'sigmoid_min': sigmoid_output.min().item(),
                'sigmoid_max': sigmoid_output.max().item(),
                'sigmoid_mean': sigmoid_output.mean().item(),
                'pixels_gt_0.5': (sigmoid_output > 0.5).sum().item(),
                'pixels_gt_0.3': (sigmoid_output > 0.3).sum().item(),
                'pixels_gt_0.1': (sigmoid_output > 0.1).sum().item(),
            }
            
            print("\nModel output statistics:")
            print(f"  Raw logits:  [{stats['raw_min']:.3f}, {stats['raw_max']:.3f}], mean: {stats['raw_mean']:.3f}")
            print(f"  After sigmoid: [{stats['sigmoid_min']:.3f}, {stats['sigmoid_max']:.3f}], mean: {stats['sigmoid_mean']:.3f}")
            print(f"  Pixels > 0.5: {stats['pixels_gt_0.5']: ,} ({stats['pixels_gt_0.5']/(256*256)*100:.1f}%)")
            print(f"  Pixels > 0.3: {stats['pixels_gt_0.3']:,} ({stats['pixels_gt_0.3']/(256*256)*100:.1f}%)")
            print(f"  Pixels > 0.1: {stats['pixels_gt_0.1']:,} ({stats['pixels_gt_0.1']/(256*256)*100:.1f}%)")
            
            # Diagnosis
            if stats['sigmoid_mean'] < 0.01:
                print("\n⚠️  WARNING: Model outputs very low values (near 0)")
                print("   This will produce BLACK masks!")
            elif stats['sigmoid_mean'] > 0.99:
                print("\n⚠️  WARNING: Model outputs very high values (near 1)")
                print("   This will produce WHITE masks (entire image)!")
            else:
                print("\n✓ Model produces reasonable output range")
            
            self.results['dummy_inference'] = stats
            return stats
            
        except Exception as e:
            print(f"✗ Error in dummy inference:  {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def test_5_real_image_inference(self, test_image_path: str = None) -> Dict[str, Any]:
        """Test 5: Run inference with real brain image."""
        print("\n[Test 5] Real Image Inference Test")
        print("-" * 40)
        
        # Create synthetic test image if no real image provided
        if test_image_path and Path(test_image_path).exists():
            print(f"Loading test image: {test_image_path}")
            image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        else:
            print("Creating synthetic brain-like test image...")
            image = self._create_synthetic_brain_image()
        
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Image range: [{image.min()}, {image.max()}]")
        
        try:
            inference = UNetInference()
            
            # Test different preprocessing methods
            preprocessing_methods = {
                'standard': lambda img: img.astype(np.float32) / 255.0,
                'scaled': lambda img: (img. astype(np.float32) / 127.5) - 1.0,
                'normalized': lambda img: (img.astype(np.float32) - img.mean()) / (img.std() + 1e-7)
            }
            
            results = {}
            
            for method_name, preprocess_fn in preprocessing_methods.items():
                print(f"\n  Testing with '{method_name}' preprocessing:")
                
                # Preprocess
                img_norm = preprocess_fn(image)
                tensor = torch.from_numpy(img_norm)
                
                # Replicate channels if needed
                if settings.UNET_IN_CHANNELS > 1:
                    tensor = tensor.unsqueeze(0).repeat(settings.UNET_IN_CHANNELS, 1, 1)
                else:
                    tensor = tensor. unsqueeze(0)
                
                tensor = tensor.unsqueeze(0).to(self.device)
                
                print(f"    Preprocessed range: [{tensor.min():.3f}, {tensor.max():.3f}]")
                
                # Inference
                with torch.no_grad():
                    output = inference.model(tensor)
                
                sigmoid_output = torch.sigmoid(output)
                mask = sigmoid_output. cpu().numpy()[0, 0]
                
                # Test multiple thresholds
                thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
                threshold_results = {}
                
                for thresh in thresholds:
                    binary = (mask > thresh).astype(np.uint8)
                    white_pixels = binary. sum()
                    percentage = (white_pixels / binary.size) * 100
                    threshold_results[thresh] = {
                        'white_pixels': white_pixels,
                        'percentage': percentage
                    }
                
                results[method_name] = {
                    'sigmoid_min': mask.min(),
                    'sigmoid_max':  mask.max(),
                    'sigmoid_mean': mask.mean(),
                    'threshold_results':  threshold_results
                }
                
                print(f"    Sigmoid range: [{mask.min():.3f}, {mask.max():.3f}], mean: {mask.mean():.3f}")
                print(f"    Threshold results:")
                for thresh, tres in threshold_results.items():
                    print(f"      >{thresh}: {tres['percentage']:. 1f}% white")
            
            # Find best preprocessing method
            best_method = None
            best_score = 0
            
            for method_name, res in results.items():
                # Good segmentation should have 5-30% white pixels at 0.5 threshold
                pct = res['threshold_results'][0.5]['percentage']
                if 5 <= pct <= 30:
                    score = 100 - abs(15 - pct)  # Ideal is around 15%
                    if score > best_score:
                        best_score = score
                        best_method = method_name
            
            print(f"\n  Recommended preprocessing: {best_method or 'NONE - All failed'}")
            
            self. results['real_inference'] = results
            return results
            
        except Exception as e: 
            print(f"✗ Error in real inference: {e}")
            import traceback
            traceback.print_exc()
            return {'success':  False, 'error': str(e)}
    
    def _create_synthetic_brain_image(self) -> np.ndarray:
        """Create synthetic brain MRI-like image for testing."""
        img = np.zeros((256, 256), dtype=np.uint8)
        
        # Brain oval
        cv2.ellipse(img, (128, 128), (100, 120), 0, 0, 360, 180, -1)
        
        # Add some texture
        noise = np.random.randint(0, 30, (256, 256), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Add "tumor" region
        cv2.circle(img, (140, 100), 20, 255, -1)
        
        # Smooth
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        return img
    
    def test_6_configuration_check(self):
        """Test 6: Verify configuration settings."""
        print("\n[Test 6] Configuration Check")
        print("-" * 40)
        
        config = {
            'UNET_IN_CHANNELS': settings.UNET_IN_CHANNELS,
            'UNET_OUT_CHANNELS': settings. UNET_OUT_CHANNELS,
            'UNET_CHANNELS': settings.UNET_CHANNELS,
            'UNET_CHECKPOINT_NAME':  settings.UNET_CHECKPOINT_NAME,
            'CHECKPOINTS_UNET': str(settings.CHECKPOINTS_UNET),
            'BATCH_SIZE': settings.BATCH_SIZE,
        }
        
        print("Current configuration:")
        for key, val in config.items():
            print(f"  {key}: {val}")
        
        # Check for common issues
        warnings = []
        
        if settings.UNET_IN_CHANNELS not in [1, 3, 4]:
            warnings.append(f"⚠️  Unusual in_channels: {settings. UNET_IN_CHANNELS} (expected 1, 3, or 4)")
        
        if settings.UNET_OUT_CHANNELS != 1:
            warnings.append(f"⚠️  out_channels should be 1 for binary segmentation, got {settings.UNET_OUT_CHANNELS}")
        
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  {w}")
        else:
            print("\n✓ Configuration looks good")
        
        self.results['config'] = config
    
    def run_all_tests(self, test_image_path: str = None):
        """Run all diagnostic tests."""
        print("\n" + "=" * 60)
        print("🔬 Running All UNet Diagnostics")
        print("=" * 60)
        
        self.test_1_checkpoint_exists()
        self.test_2_model_architecture()
        self.test_3_weight_loading()
        self.test_4_model_inference_dummy()
        self.test_5_real_image_inference(test_image_path)
        self.test_6_configuration_check()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        print("\n✓ Tests completed successfully")
        print(f"  - Checkpoint exists: {self.results. get('checkpoint_exists', 'N/A')}")
        print(f"  - Weights loaded: {self.results. get('weights_loaded', 'N/A')}")
        print(f"  - Model parameters: {self.results.get('model_params', 'N/A'):,}")
        
        # Diagnosis
        print("\n🔍 DIAGNOSIS:")
        
        if not self.results.get('checkpoint_exists'):
            print("  ❌ PROBLEM: Checkpoint file not found!")
            print("     → Place your . h5 or .pth file in:", settings.CHECKPOINTS_UNET)
        
        elif not self.results.get('weights_loaded'):
            print("  ❌ PROBLEM: Model weights not loaded correctly!")
            print("     → Check H5 structure and weight conversion")
            print("     → Weights appear to be random/zeros")
        
        elif 'dummy_inference' in self.results:
            stats = self.results['dummy_inference']
            if isinstance(stats, dict) and 'sigmoid_mean' in stats:
                if stats['sigmoid_mean'] < 0.01:
                    print("  ❌ PROBLEM: Model outputs near-zero values (BLACK MASK)!")
                    print("     → Model might be broken or incompatible")
                    print("     → Try different preprocessing or threshold")
                elif stats['sigmoid_mean'] > 0.99:
                    print("  ❌ PROBLEM: Model outputs near-one values (WHITE MASK)!")
                    print("     → Model is over-segmenting")
                else:
                    print("  ✓ Model produces reasonable outputs")
        
        print("\n" + "=" * 60)


def main():
    """Main diagnostic runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='UNet Model Diagnostics')
    parser.add_argument('--image', type=str, help='Path to test image (optional)')
    args = parser.parse_args()
    
    diagnostics = UNetDiagnostics()
    diagnostics.run_all_tests(test_image_path=args.image)


if __name__ == "__main__":
    main()