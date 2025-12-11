"""
Placeholder for UNet datamodule (BraTS dataset).
Users should implement actual BraTS data loading here.
"""
from app.utils.logger import get_logger

logger = get_logger(__name__)


# TODO: Implement BraTS dataset loader
# Example structure:
# class BraTSDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         ...
#     
#     def __getitem__(self, idx):
#         # Load image and mask
#         ...
#     
#     def __len__(self):
#         ...
