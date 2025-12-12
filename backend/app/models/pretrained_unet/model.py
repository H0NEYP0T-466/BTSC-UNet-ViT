"""
Pretrained UNet model architecture for brain tumor segmentation.
Uses MONAI's UNet implementation with optimized architecture.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from monai.networks.nets import UNet as MONAIUNet


def get_pretrained_unet_model(
    in_channels: int = 1,
    out_channels: int = 1,
    spatial_dims: int = 2,
    channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2
) -> nn.Module:
    """
    Get pretrained UNet model for brain tumor segmentation.
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        out_channels: Number of output channels (1 for binary segmentation)
        spatial_dims: Number of spatial dimensions (2 for 2D images)
        channels: Number of channels at each level
        strides: Stride for each downsampling layer
        num_res_units: Number of residual units per block
        
    Returns:
        UNet model
    """
    model = MONAIUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm="batch"
    )
    
    return model


class PretrainedUNet(nn.Module):
    """
    Wrapper for pretrained UNet model with additional processing capabilities.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_dims: int = 2,
        channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Core UNet model
        self.unet = get_pretrained_unet_model(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, C, H, W) with segmentation mask
        """
        return self.unet(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary mask with thresholding.
        
        Args:
            x: Input tensor (B, C, H, W)
            threshold: Threshold for binary classification
            
        Returns:
            Binary mask tensor (B, C, H, W)
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()
        return mask
