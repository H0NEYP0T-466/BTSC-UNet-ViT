"""
Unit tests for Brain UNet fallback logic.
Tests that fallback to candidate masks works when UNet produces empty masks.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add project root and backend to path
project_root = Path(__file__).resolve().parents[2]
backend_path = project_root / 'backend'
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(project_root))

from app.models.brain_unet.infer_unet import BrainUNetInference


class TestBrainUNetFallback:
    """Test Brain UNet fallback functionality."""
    
    def test_fallback_triggered_on_empty_mask(self):
        """Test that fallback is triggered when UNet produces near-empty mask."""
        import torch
        
        # Create a mock model that returns all zeros (empty mask)
        mock_model = MagicMock()
        # Return actual tensor instead of MagicMock
        zeros_tensor = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        mock_model.return_value = zeros_tensor
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Create inference instance with mocked model and enabled preprocessing
        with patch('app.models.brain_unet.infer_unet.get_brain_unet_model') as mock_get_model, \
             patch('app.models.brain_unet.infer_unet.apply_pipeline') as mock_apply_pipeline:
            
            mock_get_model.return_value = mock_model
            
            # Mock preprocessing result with candidate masks
            mock_apply_pipeline.return_value = {
                'stages': {
                    'normalized': test_image.astype(np.float32),
                },
                'candidates': {
                    'otsu': np.ones((256, 256), dtype=np.uint8) * 255,  # Non-empty Otsu mask
                    'yen': np.ones((256, 256), dtype=np.uint8) * 200,
                    'li': np.ones((256, 256), dtype=np.uint8) * 150,
                    'triangle': np.ones((256, 256), dtype=np.uint8) * 100,
                },
                'final': {},
                'config_used': {}
            }
            
            # Create inference instance
            inference = BrainUNetInference(model_path=Path('/tmp/fake_model.pth'))
            inference.enable_advanced_preproc = True
            inference.preproc_config = {'enable': True}
            inference.model = mock_model
            
            # Run segmentation
            result = inference.segment_brain(test_image, image_id='test_001')
            
            # Assert fallback was used
            assert result['used_fallback'] is True, "Fallback should be triggered for empty UNet mask"
            assert result['fallback_method'] is not None, "Fallback method should be set"
            assert result['fallback_method'] in ['otsu', 'yen', 'li', 'triangle'], \
                f"Fallback method should be one of the candidates, got {result['fallback_method']}"
            
            # Assert mask is not empty (fallback mask used)
            mask_area = np.sum(result['mask'] > 0) / result['mask'].size * 100
            assert mask_area > 0.1, f"Fallback mask should not be empty, got {mask_area}%"
    
    def test_no_fallback_on_valid_mask(self):
        """Test that fallback is NOT triggered when UNet produces valid mask."""
        import torch
        
        # Create a mock model that returns a valid mask (50% brain region)
        mock_model = MagicMock()
        valid_mask = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        valid_mask[:, :, 64:192, 64:192] = 5.0  # High logit value for 50% of image
        mock_model.return_value = valid_mask
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Create inference instance with mocked model
        with patch('app.models.brain_unet.infer_unet.get_brain_unet_model') as mock_get_model, \
             patch('app.models.brain_unet.infer_unet.apply_pipeline') as mock_apply_pipeline:
            
            # Setup mocks
            mock_get_model.return_value = mock_model
            
            # Mock preprocessing result
            mock_apply_pipeline.return_value = {
                'stages': {
                    'normalized': test_image.astype(np.float32),
                },
                'candidates': {
                    'otsu': np.ones((256, 256), dtype=np.uint8) * 255,
                },
                'final': {},
                'config_used': {}
            }
            
            # Create inference instance
            inference = BrainUNetInference(model_path=Path('/tmp/fake_model.pth'))
            inference.enable_advanced_preproc = True
            inference.preproc_config = {'enable': True}
            inference.model = mock_model
            
            # Run segmentation
            result = inference.segment_brain(test_image, image_id='test_002')
            
            # Assert fallback was NOT used
            assert result['used_fallback'] is False, "Fallback should not be triggered for valid UNet mask"
            assert result['fallback_method'] is None, "Fallback method should be None"
            
            # Assert mask is valid
            mask_area = np.sum(result['mask'] > 0) / result['mask'].size * 100
            assert mask_area > 20, f"UNet mask should have significant area, got {mask_area}%"
    
    def test_fallback_selects_best_candidate(self):
        """Test that fallback selects the candidate mask with largest area."""
        import torch
        
        # Create a mock model that returns all zeros
        mock_model = MagicMock()
        zeros_tensor = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        mock_model.return_value = zeros_tensor
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Create inference instance with mocked model
        with patch('app.models.brain_unet.infer_unet.get_brain_unet_model') as mock_get_model, \
             patch('app.models.brain_unet.infer_unet.apply_pipeline') as mock_apply_pipeline:
            
            # Setup mocks
            mock_get_model.return_value = mock_model
            
            # Create candidate masks with different areas (yen has largest)
            otsu_mask = np.zeros((256, 256), dtype=np.uint8)
            otsu_mask[50:100, 50:100] = 255  # Small area
            
            yen_mask = np.zeros((256, 256), dtype=np.uint8)
            yen_mask[30:220, 30:220] = 255  # Largest area
            
            li_mask = np.zeros((256, 256), dtype=np.uint8)
            li_mask[60:150, 60:150] = 255  # Medium area
            
            triangle_mask = np.zeros((256, 256), dtype=np.uint8)
            triangle_mask[80:120, 80:120] = 255  # Smallest area
            
            # Mock preprocessing result
            mock_apply_pipeline.return_value = {
                'stages': {
                    'normalized': test_image.astype(np.float32),
                },
                'candidates': {
                    'otsu': otsu_mask,
                    'yen': yen_mask,
                    'li': li_mask,
                    'triangle': triangle_mask,
                },
                'final': {},
                'config_used': {}
            }
            
            # Create inference instance
            inference = BrainUNetInference(model_path=Path('/tmp/fake_model.pth'))
            inference.enable_advanced_preproc = True
            inference.preproc_config = {'enable': True}
            inference.model = mock_model
            
            # Run segmentation
            result = inference.segment_brain(test_image, image_id='test_003')
            
            # Assert fallback selected 'yen' (largest area)
            assert result['used_fallback'] is True
            assert result['fallback_method'] == 'yen', \
                f"Should select 'yen' with largest area, got '{result['fallback_method']}'"
            
            # Assert mask matches yen mask
            assert np.array_equal(result['mask'], yen_mask), \
                "Result mask should match the selected fallback mask (yen)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
