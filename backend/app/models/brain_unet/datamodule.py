"""
Brain UNet datamodule for NFBS dataset.
Loads T1w images and brain masks from NFBS_Dataset structure.
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import cv2
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class NFBSDataset(Dataset):
    """
    NFBS (Neurofeedback Skull-stripped) dataset for brain segmentation.
    Each subject folder contains:
    - sub-*_T1w.nii.gz: Raw T1w MRI scan
    - sub-*_T1w_brainmask.nii.gz: Binary brain mask
    
    Optimization: Pre-loads and caches all slices in memory during initialization
    to avoid repeated disk I/O during training.
    """
    
    def __init__(
        self,
        root_dir: Path,
        transform: Optional[callable] = None,
        image_size: Tuple[int, int] = (256, 256),
        slice_range: Optional[Tuple[int, int]] = None,
        cache_in_memory: bool = True
    ):
        """
        Initialize NFBS dataset.
        
        Args:
            root_dir: Root directory containing subject folders (e.g., A00028185, A00028352, etc.)
            transform: Optional transforms to apply
            image_size: Target image size (H, W)
            slice_range: Optional (start, end) slice indices to use from 3D volumes
            cache_in_memory: If True, pre-load and cache all slices in memory (faster training)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.slice_range = slice_range
        self.cache_in_memory = cache_in_memory
        
        # Find all subject folders
        self.subjects = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        if not self.subjects:
            logger.warning(f"No subject folders found in {root_dir}", extra={
                'image_id': None, 'path': str(root_dir), 'stage': 'dataset_init'
            })
            self.sample_indices = []
            self.cache = {}
        else:
            # Build index: (subject_path, slice_idx) for each valid slice
            self.sample_indices = self._build_index()
            
            # Pre-load all slices into memory if caching is enabled
            self.cache = {}
            if self.cache_in_memory:
                logger.info(f"Pre-loading {len(self.sample_indices)} slices into memory...", extra={
                    'image_id': None, 'path': str(root_dir), 'stage': 'dataset_init'
                })
                self._preload_data()
        
        logger.info(
            f"NFBS Dataset initialized: {len(self.sample_indices)} slices from "
            f"{len(self.subjects)} subjects in {root_dir}, cache_in_memory={cache_in_memory}",
            extra={'image_id': None, 'path': str(root_dir), 'stage': 'dataset_init'}
        )
    
    def _build_index(self) -> List[Tuple[Path, int]]:
        """
        Build index of all valid (subject, slice) pairs.
        
        Returns:
            List of tuples: [(subject_path, slice_idx), ...]
        """
        sample_indices = []
        
        for subject_path in self.subjects:
            try:
                # Find T1w and mask files
                t1_files = list(subject_path.glob("*_T1w.nii.gz"))
                mask_files = list(subject_path.glob("*_brainmask.nii.gz"))
                
                # Skip if files missing
                if len(t1_files) != 1 or len(mask_files) != 1:
                    logger.warning(
                        f"Missing files in {subject_path.name}: "
                        f"t1={len(t1_files)}, mask={len(mask_files)}",
                        extra={'image_id': None, 'path': str(subject_path), 'stage': 'dataset_index'}
                    )
                    continue
                
                # Load to get shape
                t1_img = nib.load(t1_files[0])
                t1_shape = t1_img.shape
                
                # Determine slice range
                if self.slice_range:
                    start_slice, end_slice = self.slice_range
                else:
                    # Use middle 80% of slices to avoid empty edge slices
                    num_slices = t1_shape[2]
                    start_slice = int(num_slices * 0.1)
                    end_slice = int(num_slices * 0.9)
                
                # Add all slices in range
                for slice_idx in range(start_slice, end_slice):
                    sample_indices.append((subject_path, slice_idx))
            
            except Exception as e:
                logger.error(
                    f"Error indexing {subject_path.name}: {e}",
                    extra={'image_id': None, 'path': str(subject_path), 'stage': 'dataset_index'}
                )
        
        logger.info(
            f"Finished indexing: {len(sample_indices)} slices from {len(self.subjects)} subjects",
            extra={'image_id': None, 'path': str(self.root_dir), 'stage': 'dataset_index'}
        )
        
        return sample_indices
    
    def _preload_data(self):
        """
        Pre-load all slices into memory cache.
        This significantly speeds up training by avoiding repeated disk I/O.
        
        Optimization: Load each subject's 3D volume once and extract all slices,
        instead of loading the same volume multiple times (once per slice).
        """
        from tqdm import tqdm
        from collections import defaultdict
        
        # Group indices by subject to minimize file I/O
        subject_slices = defaultdict(list)
        for idx, (subject_path, slice_idx) in enumerate(self.sample_indices):
            subject_slices[subject_path].append((idx, slice_idx))
        
        # Process each subject once
        with tqdm(total=len(self.sample_indices), desc="Loading slices into memory") as pbar:
            for subject_path, slice_list in subject_slices.items():
                try:
                    # Find files (once per subject)
                    t1_files = list(subject_path.glob("*_T1w.nii.gz"))
                    mask_files = list(subject_path.glob("*_brainmask.nii.gz"))
                    
                    # Load 3D volumes (once per subject)
                    t1_img = nib.load(t1_files[0])
                    mask_img = nib.load(mask_files[0])
                    
                    t1_data = t1_img.get_fdata()
                    mask_data = mask_img.get_fdata()
                    
                    # Extract all slices from this subject
                    for idx, slice_idx in slice_list:
                        # Extract 2D slice
                        image_slice = t1_data[:, :, slice_idx].astype(np.float32)
                        mask_slice = mask_data[:, :, slice_idx].astype(np.float32)
                        
                        # Normalize image to [0, 1]
                        if image_slice.max() > 0:
                            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                        
                        # Binarize mask (any non-zero value is brain)
                        mask_slice = (mask_slice > 0).astype(np.float32)
                        
                        # Resize to target size
                        image_slice = cv2.resize(image_slice, self.image_size, interpolation=cv2.INTER_LINEAR)
                        mask_slice = cv2.resize(mask_slice, self.image_size, interpolation=cv2.INTER_NEAREST)
                        
                        # Store in cache
                        self.cache[idx] = (image_slice, mask_slice)
                        pbar.update(1)
                        
                except Exception as e:
                    logger.error(
                        f"Error preloading {subject_path.name}: {e}",
                        extra={'image_id': None, 'path': str(subject_path), 'stage': 'dataset_preload'}
                    )
                    # Update progress bar for failed slices
                    pbar.update(len(slice_list))
        
        logger.info(f"Pre-loaded {len(self.cache)} slices into memory", extra={
            'image_id': None, 'path': str(self.root_dir), 'stage': 'dataset_preload'
        })
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image_tensor, mask_tensor)
                - image_tensor: shape [1, H, W] (single channel T1w)
                - mask_tensor: shape [1, H, W] (binary brain mask)
        """
        # Use cached data if available
        if self.cache_in_memory and idx in self.cache:
            image_slice, mask_slice = self.cache[idx]
            # Make copies to avoid modifying cached data
            # (necessary if transforms modify in-place)
            # Copy overhead is negligible compared to avoided disk I/O
            image_slice = image_slice.copy()
            mask_slice = mask_slice.copy()
        else:
            # Load from disk (fallback for non-cached mode)
            subject_path, slice_idx = self.sample_indices[idx]
            
            # Find files
            t1_files = list(subject_path.glob("*_T1w.nii.gz"))
            mask_files = list(subject_path.glob("*_brainmask.nii.gz"))
            
            # Load 3D volumes
            t1_img = nib.load(t1_files[0])
            mask_img = nib.load(mask_files[0])
            
            t1_data = t1_img.get_fdata()
            mask_data = mask_img.get_fdata()
            
            # Extract 2D slice
            image_slice = t1_data[:, :, slice_idx].astype(np.float32)
            mask_slice = mask_data[:, :, slice_idx].astype(np.float32)
            
            # Normalize image to [0, 1]
            if image_slice.max() > 0:
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
            
            # Binarize mask (any non-zero value is brain)
            mask_slice = (mask_slice > 0).astype(np.float32)
            
            # Resize to target size
            image_slice = cv2.resize(image_slice, self.image_size, interpolation=cv2.INTER_LINEAR)
            mask_slice = cv2.resize(mask_slice, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms if provided
        if self.transform:
            # For albumentations
            try:
                transformed = self.transform(image=image_slice, mask=mask_slice)
                image_slice = transformed['image']
                mask_slice = transformed['mask']
            except:
                pass
        
        # Convert to tensors [1, H, W]
        image_tensor = torch.from_numpy(image_slice).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()
        
        return image_tensor, mask_tensor


def create_brain_unet_dataloaders(
    root_dir: Optional[Path] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    train_split: float = 0.8,
    image_size: Tuple[int, int] = (256, 256),
    transform: Optional[callable] = None,
    slice_range: Optional[Tuple[int, int]] = None,
    cache_in_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for Brain UNet.
    
    Args:
        root_dir: Root directory of NFBS dataset
        batch_size: Batch size
        num_workers: Number of worker processes (set to 0 if cache_in_memory=True)
        train_split: Fraction of data for training
        image_size: Target image size (H, W)
        transform: Optional transforms to apply
        slice_range: Optional (start, end) slice indices to use
        cache_in_memory: If True, pre-load all data into memory (faster training)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    root_dir = root_dir or Path("/content/NFBS_Dataset")
    
    logger.info(f"Creating Brain UNet dataloaders from {root_dir}", extra={
        'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'
    })
    
    # When caching in memory, num_workers=0 is optimal to avoid multiprocessing overhead
    original_num_workers = num_workers
    if cache_in_memory and num_workers > 0:
        logger.warning(
            f"Setting num_workers=0 (was {num_workers}) because cache_in_memory=True. "
            "Multiprocessing overhead is unnecessary when data is cached in memory.",
            extra={'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'}
        )
        num_workers = 0
    
    # Create full dataset
    full_dataset = NFBSDataset(
        root_dir=root_dir,
        transform=transform,
        image_size=image_size,
        slice_range=slice_range,
        cache_in_memory=cache_in_memory
    )
    
    if len(full_dataset) == 0:
        logger.error("No data loaded! Check dataset path.", extra={
            'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'
        })
        raise ValueError(f"No data found in {root_dir}")
    
    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    logger.info(
        f"Brain UNet dataloaders created: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"batch_size={batch_size}",
        extra={'image_id': None, 'path': str(root_dir), 'stage': 'dataloader_init'}
    )
    
    return train_loader, val_loader
