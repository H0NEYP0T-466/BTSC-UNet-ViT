"""
Tests for skip_preprocessing functionality.
"""
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.pipeline_service import PipelineService


def test_skip_preprocessing_parameter():
    """Test that skip_preprocessing parameter is handled correctly."""
    # Create a mock image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Mock the storage service
    with patch('app.services.pipeline_service.get_storage_service') as mock_storage:
        mock_storage_instance = Mock()
        mock_storage_instance.generate_image_id.return_value = 'test-image-id'
        mock_storage_instance.save_upload.return_value = 'uploads/test-image-id.png'
        mock_storage_instance.save_artifact.return_value = 'artifacts/test.png'
        mock_storage_instance.get_artifact_url.return_value = '/files/artifacts/test.png'
        mock_storage.return_value = mock_storage_instance
        
        # Create pipeline service
        pipeline = PipelineService()
        
        # Mock the models
        pipeline.vit = Mock()
        pipeline.vit.classify.return_value = {
            'class': 'glioma',
            'confidence': 0.95,
            'logits': [1.0, 2.0, 3.0, 4.0],
            'probabilities': [0.1, 0.2, 0.3, 0.4]
        }
        
        pipeline.tumor_unet = Mock()
        pipeline.tumor_unet.segment_image.return_value = {
            'mask': np.zeros((256, 256), dtype=np.uint8),
            'overlay': np.zeros((256, 256, 3), dtype=np.uint8),
            'segmented': np.zeros((256, 256), dtype=np.uint8)
        }
        
        pipeline.tumor_unet2 = Mock()
        pipeline.tumor_unet2.segment_image.return_value = {
            'mask': np.zeros((256, 256), dtype=np.uint8),
            'overlay': np.zeros((256, 256, 3), dtype=np.uint8),
            'segmented': np.zeros((256, 256), dtype=np.uint8)
        }
        
        # Test with skip_preprocessing=False (default behavior)
        with patch('app.services.pipeline_service.preprocess_pipeline') as mock_preprocess:
            mock_preprocess.return_value = {
                'grayscale': np.zeros((256, 256), dtype=np.uint8),
                'denoised': np.zeros((256, 256), dtype=np.uint8),
                'motion_reduced': np.zeros((256, 256), dtype=np.uint8),
                'contrast': np.zeros((256, 256), dtype=np.uint8),
                'sharpened': np.zeros((256, 256), dtype=np.uint8),
                'normalized': np.zeros((256, 256), dtype=np.uint8)
            }
            
            result = pipeline.run_inference(test_image, skip_preprocessing=False)
            
            # Verify preprocess_pipeline was called
            assert mock_preprocess.called, "preprocess_pipeline should be called when skip_preprocessing=False"
            assert 'preprocessing' in result
            # With full preprocessing, should have all 6 stages
            assert len(result['preprocessing']) == 6
        
        # Test with skip_preprocessing=True (skip preprocessing)
        result = pipeline.run_inference(test_image, skip_preprocessing=True)
        
        # Verify no preprocessing was done - raw image passed directly
        assert 'preprocessing' in result
        # With skip preprocessing, should have no preprocessing stages (empty dict)
        assert len(result['preprocessing']) == 0


def test_config_values_reduced():
    """Test that preprocessing parameters were reduced to preserve detail."""
    from app.config import settings
    
    # Test CLAHE clip limit
    assert settings.CLAHE_CLIP_LIMIT == 1.5, \
        f"CLAHE_CLIP_LIMIT should be 1.5 but is {settings.CLAHE_CLIP_LIMIT}"
    
    # Test unsharp mask parameters
    assert settings.UNSHARP_RADIUS == 1.0, \
        f"UNSHARP_RADIUS should be 1.0 but is {settings.UNSHARP_RADIUS}"
    
    assert settings.UNSHARP_AMOUNT == 1.0, \
        f"UNSHARP_AMOUNT should be 1.0 but is {settings.UNSHARP_AMOUNT}"


def test_inference_request_schema():
    """Test that InferenceRequest schema accepts skip_preprocessing parameter."""
    from app.schemas.requests import InferenceRequest
    
    # Test default value
    req1 = InferenceRequest()
    assert req1.skip_preprocessing == False, "Default skip_preprocessing should be False"
    
    # Test explicit True value
    req2 = InferenceRequest(skip_preprocessing=True)
    assert req2.skip_preprocessing == True, "skip_preprocessing should be True when set"
    
    # Test explicit False value
    req3 = InferenceRequest(skip_preprocessing=False)
    assert req3.skip_preprocessing == False, "skip_preprocessing should be False when set"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
