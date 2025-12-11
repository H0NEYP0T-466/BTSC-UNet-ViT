"""
ViT datamodule for brain tumor classification dataset.
Loads images from folder structure: dataset/Vit_Dataset/{class_name}/images/
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ViTDataset(Dataset):
    """
    Brain tumor classification dataset for ViT.
    Loads images from folder structure: root_dir/{class_name}/images/
    """
    
    def __init__(
        self,
        root_dir: Path,
        class_names: List[str] = None,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224
    ):
        """
        Initialize ViT dataset.
        
        Args:
            root_dir: Root directory containing class folders
            class_names: List of class names (folders)
            transform: Optional transforms to apply
            image_size: Target image size for ViT
        """
        self.root_dir = Path(root_dir)
        self.class_names = class_names or settings.VIT_CLASS_NAMES
        self.transform = transform
        self.image_size = image_size
        
        # Build image list and labels
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}", extra={
                    'image_id': None, 'path': str(class_dir), 'stage': 'dataset_init'
                })
                continue
            
            # Find all images in class directory (handle various extensions)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
                for img_path in class_dir.rglob(ext):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
        
        logger.info(
            f"ViT Dataset initialized: {len(self.image_paths)} images, "
            f"{len(self.class_names)} classes from {root_dir}",
            extra={'image_id': None, 'path': str(root_dir), 'stage': 'dataset_init'}
        )
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            default_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = default_transform(image)
        
        return image, label


def get_vit_transforms(image_size: int = 224, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and validation transforms for ViT.
    
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation for training
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Base transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Training transforms with augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def create_vit_dataloaders(
    root_dir: Optional[Path] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    train_split: float = 0.8,
    image_size: int = 224,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for ViT.
    
    Args:
        root_dir: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        train_split: Fraction of data for training
        image_size: Target image size
        augment: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    root_dir = root_dir or settings.SEGMENTED_DATASET_ROOT
    
    logger.info(f"Creating ViT dataloaders from {root_dir}", extra={
        'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'
    })
    
    # Get transforms
    train_transform, val_transform = get_vit_transforms(image_size, augment)
    
    # Create datasets with respective transforms
    train_dataset_full = ViTDataset(
        root_dir=root_dir,
        transform=train_transform,
        image_size=image_size
    )
    
    val_dataset_full = ViTDataset(
        root_dir=root_dir,
        transform=val_transform,
        image_size=image_size
    )
    
    # Split indices
    total_size = len(train_dataset_full)
    indices = list(range(total_size))
    train_size = int(train_split * total_size)
    
    # Set random seed for reproducible split
    import random
    random.seed(settings.SEED)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
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
        f"ViT dataloaders created: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"batch_size={batch_size}",
        extra={'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'}
    )
    
    return train_loader, val_loader
