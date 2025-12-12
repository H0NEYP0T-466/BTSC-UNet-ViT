"""
Integration test to verify the UNet channel fix works end-to-end.
"""
import numpy as np
import torch
from unittest.mock import MagicMock, patch


def test_integration_inference_with_4_channel_model():
    """
    Integration test simulating the actual error scenario from the problem statement.
    This test verifies that a grayscale image can be successfully processed by a 
    4-channel model after the fix.
    """
    # Simulate a preprocessed grayscale image (512x512) like in the error log
    preprocessed_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    # Mock the UNet model that expects 4 channels
    with patch('app.models.unet.infer_unet.get_unet_model') as mock_get_model:
        # Create a real-ish model that checks input shape
        class FakeUNetModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 16, kernel_size=3, padding=1)
                
            def forward(self, x):
                # This will raise an error if input doesn't have 4 channels
                return self.conv(x)
        
        fake_model = FakeUNetModel()
        fake_model.eval()
        mock_get_model.return_value = fake_model
        
        # Mock checkpoint not existing
        with patch('pathlib.Path.exists', return_value=False):
            with patch('app.models.unet.infer_unet.settings') as mock_settings:
                mock_settings.UNET_IN_CHANNELS = 4
                mock_settings.UNET_OUT_CHANNELS = 1
                mock_settings.UNET_CHANNELS = (16, 32, 64, 128, 256)
                mock_settings.CHECKPOINTS_UNET = MagicMock()
                mock_settings.UNET_CHECKPOINT_NAME = 'unet_best.pth'
                
                from app.models.unet.infer_unet import UNetInference
                
                inference = UNetInference(device='cpu')
                
                # This should work now without the channel mismatch error
                tensor = inference.preprocess_image(preprocessed_image)
                
                # Verify the shape is correct
                assert tensor.shape == (1, 4, 512, 512), \
                    f"Expected shape (1, 4, 512, 512), got {tensor.shape}"
                
                # Try to pass through the model to verify no errors
                try:
                    with torch.no_grad():
                        output = fake_model(tensor)
                    # If we get here, the channel count is correct
                    assert output.shape[1] == 16, "Model should output 16 channels"
                    print("✓ Integration test passed: 4-channel model accepts grayscale input")
                except RuntimeError as e:
                    if "expected input" in str(e) and "to have" in str(e) and "channels" in str(e):
                        raise AssertionError(
                            f"Channel mismatch still exists! Error: {e}"
                        )
                    raise


def test_integration_backwards_compatible_1_channel():
    """
    Verify that models trained with 1 channel still work correctly.
    """
    preprocessed_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    with patch('app.models.unet.infer_unet.get_unet_model') as mock_get_model:
        class FakeUNetModel1Channel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
                
            def forward(self, x):
                return self.conv(x)
        
        fake_model = FakeUNetModel1Channel()
        fake_model.eval()
        mock_get_model.return_value = fake_model
        
        with patch('pathlib.Path.exists', return_value=False):
            with patch('app.models.unet.infer_unet.settings') as mock_settings:
                mock_settings.UNET_IN_CHANNELS = 1  # Single channel model
                mock_settings.UNET_OUT_CHANNELS = 1
                mock_settings.UNET_CHANNELS = (16, 32, 64, 128, 256)
                mock_settings.CHECKPOINTS_UNET = MagicMock()
                mock_settings.UNET_CHECKPOINT_NAME = 'unet_best.pth'
                
                from app.models.unet.infer_unet import UNetInference
                
                inference = UNetInference(device='cpu')
                tensor = inference.preprocess_image(preprocessed_image)
                
                assert tensor.shape == (1, 1, 256, 256), \
                    f"Expected shape (1, 1, 256, 256), got {tensor.shape}"
                
                with torch.no_grad():
                    output = fake_model(tensor)
                
                assert output.shape[1] == 16
                print("✓ Backwards compatibility test passed: 1-channel model still works")


if __name__ == "__main__":
    test_integration_inference_with_4_channel_model()
    test_integration_backwards_compatible_1_channel()
    print("\n✓ All integration tests passed!")
