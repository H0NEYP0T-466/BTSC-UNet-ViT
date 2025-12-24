"""
UNet Tumor model architecture for tumor segmentation on PNG images.
"""
import torch
import torch.nn as nn
from typing import Tuple
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.double_conv(x)


class UNetTumor(nn.Module):
    """
    UNet architecture for tumor segmentation on PNG images.
    Includes dropout for regularization to prevent overfitting.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.dropout_rate = dropout_rate
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        current_channels = in_channels
        for feature in features:
            self.encoder.append(DoubleConv(current_channels, feature, dropout_rate=dropout_rate))
            current_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_rate=dropout_rate)
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature, dropout_rate=dropout_rate))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        logger.info(f"UNetTumor initialized: in_channels={self.in_channels}, out_channels={self.out_channels}, features={features}, dropout={dropout_rate}", extra={
            'image_id': None,
            'path': None,
            'stage': 'model_init'
        })
    
    def forward(self, x):
        # Encoder
        skip_connections = []
        
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for idx in range(len(self.decoder)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]
            
            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True
                )
            
            x = torch.cat([skip_connection, x], dim=1)
            x = self.decoder[idx](x)
        
        return self.final_conv(x)


def get_unet_tumor_model(
    in_channels: int = 3,
    out_channels: int = 1,
    features: Tuple[int, ...] = (32, 64, 128, 256, 512),
    dropout_rate: float = 0.2
) -> UNetTumor:
    """
    Get UNet Tumor model instance.
    
    Args:
        in_channels: Number of input channels (3 for RGB PNG images)
        out_channels: Number of output channels (1 for binary segmentation)
        features: Feature channels for each encoder/decoder level
        dropout_rate: Dropout rate for regularization
        
    Returns:
        UNetTumor model
    """
    model = UNetTumor(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        dropout_rate=dropout_rate
    )
    logger.info(f"Created UNetTumor model with {sum(p.numel() for p in model.parameters())} parameters", extra={
        'image_id': None,
        'path': None,
        'stage': 'model_creation'
    })
    return model
