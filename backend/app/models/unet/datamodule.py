"""
UNet datamodule for BraTS dataset with .h5 files.
Loads images and masks from .h5 files in dataset/UNet_Dataset/
Each .h5 file represents ONE 2D slice sample. 
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
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
    Each .h5 file is ONE sample (one 2D slice).
    Loads images and masks from .h5 files on-demand (lazy loading).
    """
    
    def __init__(
        self,
        root_dir: Path,
        transform: Optional[callable] = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize UNet dataset with lazy loading. 
        
        Args:
            root_dir: Root directory containing .h5 files
            transform: Optional transforms to apply
            image_size: Target image size (H, W)
        """
        self. root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Find all .h5 files - each file is ONE sample
        self.h5_files = sorted(list(self.root_dir.glob('*.h5')))
        
        if not self. h5_files:
            logger.warning(f"No .h5 files found in {root_dir}", extra={
                'image_id': None, 'path': str(root_dir), 'stage': 'dataset_init'
            })
            self.sample_indices = []
        else:
            # Build index:  one entry per file
            self.sample_indices = self._build_index()
        
        logger.info(
            f"UNet Dataset initialized: {len(self.sample_indices)} samples (slices) from {len(self.h5_files)} "
            f".h5 files in {root_dir}",
            extra={'image_id': None, 'path': str(root_dir), 'stage': 'dataset_init'}
        )
    
    def _build_index(self) -> List[Tuple[Path, str, Optional[str]]]:
        """
        Build index:  one entry per . h5 file (each file = one sample).
        
        Returns:
            List of tuples: [(file_path, image_key, mask_key), ...]
        """
        sample_indices = []
        
        image_keys = ['image', 'images', 'data', 'X', 'input']
        mask_keys = ['mask', 'masks', 'label', 'labels', 'y', 'target', 'segmentation']
        
        for i, h5_path in enumerate(self.h5_files, 1):
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Find image dataset
                    image_key = None
                    for key in image_keys:
                        if key in f.keys():
                            image_key = key
                            break
                    
                    # Find mask dataset
                    mask_key = None
                    for key in mask_keys: 
                        if key in f.keys():
                            mask_key = key
                            break
                    
                    if image_key is None:
                        # If no standard keys found, use first dataset
                        available_keys = list(f.keys())
                        if available_keys:
                            image_key = available_keys[0]
                            if len(available_keys) > 1:
                                mask_key = available_keys[1]
                        else:
                            logger.warning(
                                f"No datasets found in {h5_path. name}",
                                extra={'image_id': None, 'path': str(h5_path), 'stage': 'dataset_index'}
                            )
                            continue
                    
                    # Add ONE entry for this file
                    sample_indices. append((h5_path, image_key, mask_key))
                
                # Log progress every 10000 files
                if i % 10000 == 0:
                    logger.info(f"Indexed {i} files, usable samples={len(sample_indices)}", extra={
                        'image_id': None, 'path': str(self.root_dir), 'stage': 'dataset_index'
                    })
                
            except Exception as e: 
                logger.error(f"Error indexing {h5_path}:  {e}", extra={
                    'image_id': None, 'path': str(h5_path), 'stage': 'dataset_index'
                })
        
        logger.info(f"Finished indexing:  {len(sample_indices)} usable samples from {len(self.h5_files)} files", extra={
            'image_id': None, 'path': str(self.root_dir), 'stage': 'dataset_index'
        })
        
        return sample_indices
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.  Loads ONE slice from ONE . h5 file.
        
        Returns:
            Tuple of (image_tensor, mask_tensor)
                - image_tensor: shape [C, H, W] where C is number of modalities (e.g., 4 for BraTS)
                - mask_tensor: shape [1, H, W] binary mask
        """
        # Get file path and keys
        h5_path, image_key, mask_key = self.sample_indices[idx]
        
        # Load the sample from the h5 file
        with h5py.File(h5_path, 'r') as f:
            image = f[image_key][...]  # Load entire dataset (one slice)
            
            # Load mask if available
            if mask_key is not None and mask_key in f:
                mask = f[mask_key][...]
            else:
                # Create empty mask if not available
                if len(image.shape) == 2:  # (H, W)
                    mask = np.zeros(image.shape, dtype=np.float32)
                elif len(image.shape) == 3:
                    if image.shape[0] in [1, 2, 3, 4, 8]:  # (C, H, W)
                        mask = np.zeros(image.shape[1:], dtype=np.float32)
                    else:  # (H, W, C)
                        mask = np.zeros(image.shape[:2], dtype=np.float32)
        
        # Process image to [C, H, W] format
        if len(image.shape) == 2:
            # (H, W) -> (1, H, W)
            image = image[np.newaxis, : , :]
        elif len(image.shape) == 3:
            if image.shape[0] not in [1, 2, 3, 4, 8]: 
                # (H, W, C) -> (C, H, W)
                image = np.transpose(image, (2, 0, 1))
            # else already (C, H, W)
        
        # Process mask to [H, W] format
        if len(mask.shape) == 3:
            if mask.shape[0] in [1, 2, 3, 4, 8]:  # (C, H, W)
                # Sum across channels for multi-class or take first channel
                mask = np.sum(mask, axis=0) if mask.shape[0] > 1 else mask[0]
            else:  # (H, W, C)
                mask = np.sum(mask, axis=2) if mask.shape[2] > 1 else mask[: , : , 0]
        
        # Resize each channel to target size
        num_channels = image.shape[0]
        resized_image = np.zeros((num_channels, self.image_size[0], self.image_size[1]), dtype=np.float32)
        for c in range(num_channels):
            resized_image[c] = cv2.resize(image[c], self.image_size, interpolation=cv2.INTER_LINEAR)
        
        resized_mask = cv2.resize(mask, self. image_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to [0, 1]
        if resized_image.max() > 1.0:
            resized_image = resized_image.astype(np. float32) / 255.0
        else:
            resized_image = resized_image. astype(np.float32)
        
        # Normalize mask to binary [0, 1]
        if resized_mask.max() > 1.0:
            resized_mask = (resized_mask > 0).astype(np.float32)
        else:
            resized_mask = (resized_mask > 0).astype(np.float32)
        
        # Apply transforms if provided (note: transform must handle multi-channel input)
        if self.transform:
            # For albumentations, you'd need to handle multi-channel differently
            # For now, skip transform or implement custom logic
            pass
        
        # Convert to tensors
        image_tensor = torch.from_numpy(resized_image).float()  # (C, H, W)
        mask_tensor = torch. from_numpy(resized_mask).unsqueeze(0).float()  # (1, H, W)
        
        return image_tensor, mask_tensor


def create_unet_dataloaders(
    root_dir:  Optional[Path] = None,
    batch_size: int = 8,
    num_workers:  int = 4,
    train_split: float = 0.8,
    image_size:  Tuple[int, int] = (256, 256),
    transform: Optional[callable] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for UNet.
    
    Args:
        root_dir:  Root directory of dataset
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
        'image_id':  None, 'path': str(root_dir), 'stage': 'dataloader_init'
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
        pin_memory=torch.cuda. is_available()
    )
    
    logger.info(
        f"UNet dataloaders created: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"batch_size={batch_size}",
        extra={'image_id': None, 'path':  str(root_dir), 'stage': 'dataloader_init'}
    )
    
    return train_loader, val_loader