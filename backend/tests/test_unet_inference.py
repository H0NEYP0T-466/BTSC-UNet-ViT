"""
Tests for UNet inference channel handling.
"""
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from app.models.unet.infer_unet import UNetInference


@pytest.fixture
def mock_unet_model():
    """Create a mock UNet model."""
    model = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    return model


def test_preprocess_single_channel_to_multi_channel():
    """Test that single grayscale channel is replicated to match model's expected channels."""
    # Create a simple grayscale image
    grayscale_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # Mock the model loading
    with patch('app.models.unet.infer_unet.get_unet_model') as mock_get_model:
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_get_model.return_value = mock_model
        
        # Mock checkpoint file not existing so we skip loading
        with patch('pathlib.Path.exists', return_value=False):
            with patch('app.models.unet.infer_unet.settings') as mock_settings:
                mock_settings.UNET_IN_CHANNELS = 4  # Simulate 4-channel model
                mock_settings.UNET_OUT_CHANNELS = 1
                mock_settings.UNET_CHANNELS = (16, 32, 64, 128, 256)
                mock_settings.CHECKPOINTS_UNET = MagicMock()
                mock_settings.UNET_CHECKPOINT_NAME = 'test.pth'
                
                # Create inference instance
                inference = UNetInference(device='cpu')
                
                # Mock settings for preprocessing
                with patch('app.models.unet.infer_unet.settings') as mock_settings_preprocess:
                    mock_settings_preprocess.UNET_IN_CHANNELS = 4
                    
                    # Preprocess the image
                    tensor = inference.preprocess_image(grayscale_image)
                    
                    # Check output shape - should be (1, 4, 256, 256)
                    assert tensor.shape[0] == 1, "Batch dimension should be 1"
                    assert tensor.shape[1] == 4, f"Channel dimension should be 4, got {tensor.shape[1]}"
                    assert tensor.shape[2] == 256, "Height should be 256"
                    assert tensor.shape[3] == 256, "Width should be 256"


def test_preprocess_rgb_to_multi_channel():
    """Test that RGB image is converted to grayscale then replicated."""
    # Create a simple RGB image
    rgb_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Mock the model loading
    with patch('app.models.unet.infer_unet.get_unet_model') as mock_get_model:
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_get_model.return_value = mock_model
        
        # Mock checkpoint file not existing
        with patch('pathlib.Path.exists', return_value=False):
            with patch('app.models.unet.infer_unet.settings') as mock_settings:
                mock_settings.UNET_IN_CHANNELS = 4
                mock_settings.UNET_OUT_CHANNELS = 1
                mock_settings.UNET_CHANNELS = (16, 32, 64, 128, 256)
                mock_settings.CHECKPOINTS_UNET = MagicMock()
                mock_settings.UNET_CHECKPOINT_NAME = 'test.pth'
                
                inference = UNetInference(device='cpu')
                
                with patch('app.models.unet.infer_unet.settings') as mock_settings_preprocess:
                    mock_settings_preprocess.UNET_IN_CHANNELS = 4
                    
                    tensor = inference.preprocess_image(rgb_image)
                    
                    # Check output shape
                    assert tensor.shape == (1, 4, 256, 256), f"Expected (1, 4, 256, 256), got {tensor.shape}"


def test_preprocess_single_channel_model():
    """Test preprocessing when model expects single channel."""
    grayscale_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    with patch('app.models.unet.infer_unet.get_unet_model') as mock_get_model:
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_get_model.return_value = mock_model
        
        with patch('pathlib.Path.exists', return_value=False):
            with patch('app.models.unet.infer_unet.settings') as mock_settings:
                mock_settings.UNET_IN_CHANNELS = 1  # Single channel model
                mock_settings.UNET_OUT_CHANNELS = 1
                mock_settings.UNET_CHANNELS = (16, 32, 64, 128, 256)
                mock_settings.CHECKPOINTS_UNET = MagicMock()
                mock_settings.UNET_CHECKPOINT_NAME = 'test.pth'
                
                inference = UNetInference(device='cpu')
                
                with patch('app.models.unet.infer_unet.settings') as mock_settings_preprocess:
                    mock_settings_preprocess.UNET_IN_CHANNELS = 1
                    
                    tensor = inference.preprocess_image(grayscale_image)
                    
                    # Check output shape - should be (1, 1, 256, 256)
                    assert tensor.shape == (1, 1, 256, 256), f"Expected (1, 1, 256, 256), got {tensor.shape}"


def test_normalized_values():
    """Test that normalized values are in correct range."""
    grayscale_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    with patch('app.models.unet.infer_unet.get_unet_model') as mock_get_model:
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_get_model.return_value = mock_model
        
        with patch('pathlib.Path.exists', return_value=False):
            with patch('app.models.unet.infer_unet.settings') as mock_settings:
                mock_settings.UNET_IN_CHANNELS = 4
                mock_settings.UNET_OUT_CHANNELS = 1
                mock_settings.UNET_CHANNELS = (16, 32, 64, 128, 256)
                mock_settings.CHECKPOINTS_UNET = MagicMock()
                mock_settings.UNET_CHECKPOINT_NAME = 'test.pth'
                
                inference = UNetInference(device='cpu')
                
                with patch('app.models.unet.infer_unet.settings') as mock_settings_preprocess:
                    mock_settings_preprocess.UNET_IN_CHANNELS = 4
                    
                    tensor = inference.preprocess_image(grayscale_image)
                    
                    # Check that values are normalized to [0, 1]
                    assert tensor.min() >= 0.0, "Minimum value should be >= 0"
                    assert tensor.max() <= 1.0, "Maximum value should be <= 1"
