"""
UNet Tumor datamodule for PNG-based dataset.
Dataset structure:
- images/: PNG images
- masks/: PNG masks (same names as images)

Includes data augmentation for training to prevent overfitting.
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from PIL import Image
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Augmentation parameters
AUGMENT_NOISE_PROBABILITY = 0.3  # Probability of applying Gaussian noise
AUGMENT_NOISE_STD = 10  # Standard deviation for Gaussian noise
AUGMENT_BRIGHTNESS_RANGE = (0.8, 1.2)  # Min/max brightness multiplier
AUGMENT_CONTRAST_RANGE = (0.8, 1.2)  # Min/max contrast multiplier


class UNetTumorDataset(Dataset):
    """
    Tumor segmentation dataset for UNet from PNG images.
    Loads images and masks from separate folders with matching filenames.
    Supports data augmentation for training.
    """
    
    def __init__(
        self,
        root_dir: Path,
        transform: Optional[callable] = None,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False
    ):
        """
        Initialize UNet Tumor dataset.
        
        Args:
            root_dir: Root directory containing 'images' and 'masks' folders
            transform: Optional transforms to apply
            image_size: Target image size (H, W)
            augment: Whether to apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.augment = augment
        
        # Find all image files
        self.images_dir = self.root_dir / 'images'
        self.masks_dir = self.root_dir / 'masks'
        
        if not self.images_dir.exists() or not self.masks_dir.exists():
            logger.warning(f"Images or masks directory not found in {root_dir}", extra={
                'image_id': None, 'path': str(root_dir), 'stage': 'dataset_init'
            })
            self.image_files = []
        else:
            # Get all PNG files
            self.image_files = sorted([
                f for f in self.images_dir.glob('*.png')
            ])
            
            # Filter to only include files that have corresponding masks
            valid_files = []
            for img_path in self.image_files:
                mask_path = self.masks_dir / img_path.name
                if mask_path.exists():
                    valid_files.append(img_path)
                else:
                    logger.warning(f"Mask not found for image: {img_path.name}", extra={
                        'image_id': None, 'path': str(img_path), 'stage': 'dataset_init'
                    })
            self.image_files = valid_files
        
        logger.info(
            f"UNet Tumor Dataset initialized: {len(self.image_files)} images from {root_dir}",
            extra={'image_id': None, 'path': str(root_dir), 'stage': 'dataset_init'}
        )
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation transformations.
        
        Augmentations applied:
        - Random horizontal flip (50% chance)
        - Random vertical flip (50% chance)
        - Random rotation (0, 90, 180, 270 degrees)
        - Random brightness/contrast adjustment
        - Random Gaussian noise
        
        Args:
            image: Input image (H, W, C)
            mask: Input mask (H, W)
            
        Returns:
            Tuple of augmented (image, mask)
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        # Random vertical flip
        if np.random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k=k).copy()
            mask = np.rot90(mask, k=k).copy()
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(AUGMENT_BRIGHTNESS_RANGE[0], AUGMENT_BRIGHTNESS_RANGE[1])
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(AUGMENT_CONTRAST_RANGE[0], AUGMENT_CONTRAST_RANGE[1])
            mean = np.mean(image)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Random Gaussian noise
        if np.random.random() > (1 - AUGMENT_NOISE_PROBABILITY):
            noise = np.random.normal(0, AUGMENT_NOISE_STD, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image, mask
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image_tensor, mask_tensor)
                - image_tensor: shape [C, H, W] where C=3 for RGB
                - mask_tensor: shape [1, H, W] binary mask
        """
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            # Try with PIL as fallback
            pil_image = Image.open(img_path).convert('RGB')
            image = np.array(pil_image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Try with PIL as fallback
            pil_mask = Image.open(mask_path).convert('L')
            mask = np.array(pil_mask)
        
        # Resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize mask to binary [0, 1]
        mask = (mask > 127).astype(np.float32)
        
        # Apply custom transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Convert to tensors
        # Image: (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        # Mask: (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image_tensor, mask_tensor


def get_augmentation_transforms():
    """
    Get augmentation function for on-the-fly augmentation.
    Returns None since augmentation is built into the dataset.
    """
    return None


def create_unet_tumor_dataloaders(
    root_dir: Optional[Path] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    train_split: float = 0.8,
    image_size: Tuple[int, int] = (256, 256),
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for UNet Tumor.
    
    Training set uses augmentation, validation set does not.
    
    Args:
        root_dir: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        train_split: Fraction of data for training
        image_size: Target image size (H, W)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if root_dir is None:
        raise ValueError("root_dir must be provided")
    
    root_dir = Path(root_dir)
    
    logger.info(f"Creating UNet Tumor dataloaders from {root_dir}", extra={
        'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'
    })
    
    # Create dataset without augmentation to get full size
    full_dataset = UNetTumorDataset(
        root_dir=root_dir,
        transform=None,
        image_size=image_size,
        augment=False
    )
    
    if len(full_dataset) == 0:
        logger.error("No data loaded! Check dataset path.", extra={
            'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'
        })
        raise ValueError(f"No data found in {root_dir}")
    
    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Get indices for split
    torch.manual_seed(seed)
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate datasets with/without augmentation
    train_dataset = UNetTumorDataset(
        root_dir=root_dir,
        transform=None,
        image_size=image_size,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = UNetTumorDataset(
        root_dir=root_dir,
        transform=None,
        image_size=image_size,
        augment=False  # No augmentation for validation
    )
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(
        f"UNet Tumor dataloaders created: train={train_size}, val={val_size}, batch_size={batch_size}",
        extra={'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'}
    )
    
    return train_loader, val_loader
