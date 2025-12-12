"""
Pretrained UNet module for brain tumor segmentation.
Uses pretrained models for better performance without local training.
"""
from .infer_pretrained import get_pretrained_unet_inference

__all__ = ['get_pretrained_unet_inference']
