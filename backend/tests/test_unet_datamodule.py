"""
Tests for UNet datamodule with lazy loading.
"""
import pytest
import numpy as np
import h5py
import tempfile
import shutil
from pathlib import Path
from app.models.unet.datamodule import UNetDataset, create_unet_dataloaders


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory with sample h5 files."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Create sample h5 files
    for i in range(3):
        h5_path = temp_path / f"test_file_{i}.h5"
        with h5py.File(h5_path, 'w') as f:
            # Create 10 sample images per file
            images = np.random.rand(10, 128, 128).astype(np.float32)
            masks = np.random.rand(10, 128, 128).astype(np.float32)
            f.create_dataset('image', data=images)
            f.create_dataset('mask', data=masks)
    
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_unet_dataset_lazy_loading(temp_dataset_dir):
    """Test that dataset uses lazy loading and doesn't load all data into memory."""
    dataset = UNetDataset(
        root_dir=temp_dataset_dir,
        image_size=(256, 256)
    )
    
    # Should have 30 samples total (3 files * 10 samples each)
    assert len(dataset) == 30
    
    # Verify that sample_indices is populated
    assert hasattr(dataset, 'sample_indices')
    assert len(dataset.sample_indices) == 30
    
    # Verify that data is NOT loaded into memory
    assert not hasattr(dataset, 'data')
    assert not hasattr(dataset, 'masks')


def test_unet_dataset_getitem(temp_dataset_dir):
    """Test that __getitem__ loads data correctly on-demand."""
    dataset = UNetDataset(
        root_dir=temp_dataset_dir,
        image_size=(256, 256)
    )
    
    # Get first sample
    image, mask = dataset[0]
    
    # Check that tensors are returned with correct shape
    assert image.shape == (1, 256, 256)  # (C, H, W)
    assert mask.shape == (1, 256, 256)
    
    # Check data types
    assert image.dtype.is_floating_point
    assert mask.dtype.is_floating_point


def test_unet_dataset_multiple_samples(temp_dataset_dir):
    """Test accessing multiple samples."""
    dataset = UNetDataset(
        root_dir=temp_dataset_dir,
        image_size=(256, 256)
    )
    
    # Access samples from different files
    samples = [dataset[i] for i in [0, 10, 20]]
    
    # All should have correct shapes
    for image, mask in samples:
        assert image.shape == (1, 256, 256)
        assert mask.shape == (1, 256, 256)


def test_unet_dataset_empty_dir():
    """Test dataset with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = UNetDataset(
            root_dir=Path(temp_dir),
            image_size=(256, 256)
        )
        
        # Should have 0 samples
        assert len(dataset) == 0
        assert len(dataset.sample_indices) == 0


def test_create_unet_dataloaders(temp_dataset_dir):
    """Test dataloader creation with lazy loading."""
    train_loader, val_loader = create_unet_dataloaders(
        root_dir=temp_dataset_dir,
        batch_size=4,
        num_workers=0,  # Use 0 workers for testing
        train_split=0.8,
        image_size=(256, 256)
    )
    
    # Check that dataloaders are created
    assert train_loader is not None
    assert val_loader is not None
    
    # Check that data can be loaded
    batch = next(iter(train_loader))
    images, masks = batch
    
    # Verify batch shape
    assert images.shape[0] <= 4  # Batch size
    assert images.shape[1:] == (1, 256, 256)  # (C, H, W)
    assert masks.shape[1:] == (1, 256, 256)
