"""
UNet datamodule for BraTS dataset with .h5 files.
Loads images and masks from .h5 files in dataset/UNet_Dataset/
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import cv2
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class UNetDataset(Dataset):
    """
    Brain tumor segmentation dataset for UNet.
    Loads images and masks from .h5 files.
    """
    
    def __init__(
        self,
        root_dir: Path,
        transform: Optional[callable] = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize UNet dataset.
        
        Args:
            root_dir: Root directory containing .h5 files
            transform: Optional transforms to apply
            image_size: Target image size (H, W)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Find all .h5 files
        self.h5_files = list(self.root_dir.glob('*.h5'))
        
        if not self.h5_files:
            logger.warning(f"No .h5 files found in {root_dir}", extra={
                'image_id': None, 'path': str(root_dir), 'stage': 'dataset_init'
            })
            self.data = []
            self.masks = []
        else:
            # Load all data into memory (if files are not too large)
            # Alternatively, store file paths and load on-demand
            self.data, self.masks = self._load_h5_files()
        
        logger.info(
            f"UNet Dataset initialized: {len(self.data)} samples from {len(self.h5_files)} "
            f".h5 files in {root_dir}",
            extra={'image_id': None, 'path': str(root_dir), 'stage': 'dataset_init'}
        )
    
    def _load_h5_files(self) -> Tuple[list, list]:
        """
        Load all .h5 files into memory.
        
        Returns:
            Tuple of (images_list, masks_list)
        """
        all_images = []
        all_masks = []
        
        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Try common key names for images and masks
                    # Adjust these based on actual .h5 file structure
                    image_keys = ['image', 'images', 'data', 'X', 'input']
                    mask_keys = ['mask', 'masks', 'label', 'labels', 'y', 'target', 'segmentation']
                    
                    # Find image data
                    image_data = None
                    for key in image_keys:
                        if key in f.keys():
                            image_data = f[key][:]
                            break
                    
                    # Find mask data
                    mask_data = None
                    for key in mask_keys:
                        if key in f.keys():
                            mask_data = f[key][:]
                            break
                    
                    if image_data is None:
                        # If no standard keys found, use first dataset
                        available_keys = list(f.keys())
                        logger.warning(
                            f"Standard image keys not found in {h5_path.name}. "
                            f"Available keys: {available_keys}. Using first key.",
                            extra={'image_id': None, 'path': str(h5_path), 'stage': 'dataset_load'}
                        )
                        if available_keys:
                            image_data = f[available_keys[0]][:]
                            if len(available_keys) > 1:
                                mask_data = f[available_keys[1]][:]
                    
                    if image_data is not None:
                        # Handle different data shapes
                        # Assuming shapes like (N, H, W) or (N, H, W, C) or (N, C, H, W)
                        if len(image_data.shape) == 3:
                            # (N, H, W) - add channel dimension
                            for i in range(len(image_data)):
                                all_images.append(image_data[i])
                                if mask_data is not None and i < len(mask_data):
                                    all_masks.append(mask_data[i])
                                else:
                                    # Create empty mask if not available
                                    all_masks.append(np.zeros_like(image_data[i]))
                        elif len(image_data.shape) == 4:
                            # (N, H, W, C) or (N, C, H, W)
                            for i in range(len(image_data)):
                                all_images.append(image_data[i])
                                if mask_data is not None and i < len(mask_data):
                                    all_masks.append(mask_data[i])
                                else:
                                    # Create empty mask
                                    if image_data.shape[-1] in [1, 3, 4]:  # (N, H, W, C)
                                        all_masks.append(np.zeros(image_data[i].shape[:2]))
                                    else:  # (N, C, H, W)
                                        all_masks.append(np.zeros(image_data[i].shape[1:]))
                        else:
                            logger.warning(
                                f"Unexpected image shape in {h5_path.name}: {image_data.shape}",
                                extra={'image_id': None, 'path': str(h5_path), 'stage': 'dataset_load'}
                            )
                
                logger.info(f"Loaded {h5_path.name}: {len(all_images)} samples so far", extra={
                    'image_id': None, 'path': str(h5_path), 'stage': 'dataset_load'
                })
                
            except Exception as e:
                logger.error(f"Error loading {h5_path}: {e}", extra={
                    'image_id': None, 'path': str(h5_path), 'stage': 'dataset_load'
                })
        
        return all_images, all_masks
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        image = self.data[idx]
        mask = self.masks[idx]
        
        # Ensure image is 2D (grayscale) - take first channel if multi-channel
        if len(image.shape) == 3:
            if image.shape[0] in [1, 3, 4]:  # (C, H, W)
                image = image[0]
            else:  # (H, W, C)
                image = image[:, :, 0]
        
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            if mask.shape[0] in [1, 3, 4]:  # (C, H, W)
                mask = mask[0]
            else:  # (H, W, C)
                mask = mask[:, :, 0]
        
        # Resize to target size
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        # Normalize mask to [0, 1]
        if mask.max() > 1.0:
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = mask.astype(np.float32)
        
        # Apply transforms if provided
        if self.transform:
            # For augmentation libraries like albumentations
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensors and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)
        
        return image, mask


def create_unet_dataloaders(
    root_dir: Optional[Path] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    train_split: float = 0.8,
    image_size: Tuple[int, int] = (256, 256),
    transform: Optional[callable] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for UNet.
    
    Args:
        root_dir: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        train_split: Fraction of data for training
        image_size: Target image size (H, W)
        transform: Optional transforms to apply
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    root_dir = root_dir or settings.BRATS_ROOT
    
    logger.info(f"Creating UNet dataloaders from {root_dir}", extra={
        'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'
    })
    
    # Create full dataset
    full_dataset = UNetDataset(
        root_dir=root_dir,
        transform=transform,
        image_size=image_size
    )
    
    if len(full_dataset) == 0:
        logger.error("No data loaded! Check dataset path and .h5 files.", extra={
            'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'
        })
        raise ValueError(f"No data found in {root_dir}")
    
    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(settings.SEED)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(
        f"UNet dataloaders created: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"batch_size={batch_size}",
        extra={'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'}
    )
    
    return train_loader, val_loader
