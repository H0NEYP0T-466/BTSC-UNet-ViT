"""
Storage service for managing file artifacts.
"""
import uuid
from pathlib import Path
from typing import Optional
import numpy as np
from app.config import settings
from app.utils.imaging import save_image, generate_unique_filename
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StorageService:
    """Service for storing and managing image artifacts."""
    
    def __init__(self):
        self.uploads_dir = settings.UPLOADS_DIR
        self.artifacts_dir = settings.ARTIFACTS_DIR
        
        # Ensure directories exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Storage service initialized", extra={
            'image_id': None,
            'path': str(self.artifacts_dir),
            'stage': 'storage_init'
        })
    
    def generate_image_id(self) -> str:
        """Generate unique image ID."""
        return uuid.uuid4().hex[:12]
    
    def save_upload(self, image: np.ndarray, image_id: str) -> str:
        """
        Save uploaded image.
        
        Args:
            image: Image array
            image_id: Unique image identifier
            
        Returns:
            Relative path to saved image
        """
        filename = f"{image_id}_original.png"
        filepath = self.uploads_dir / filename
        save_image(image, str(filepath))
        
        logger.info(f"Uploaded image saved: {filename}", extra={
            'image_id': image_id,
            'path': str(filepath),
            'stage': 'storage_save'
        })
        
        return f"uploads/{filename}"
    
    def save_artifact(
        self,
        image: np.ndarray,
        image_id: str,
        artifact_type: str
    ) -> str:
        """
        Save processing artifact.
        
        Args:
            image: Image array
            image_id: Unique image identifier
            artifact_type: Type of artifact (e.g., 'grayscale', 'denoised', 'mask')
            
        Returns:
            Relative path to saved artifact
        """
        filename = f"{image_id}_{artifact_type}.png"
        filepath = self.artifacts_dir / filename
        save_image(image, str(filepath))
        
        logger.info(f"Artifact saved: {artifact_type}", extra={
            'image_id': image_id,
            'path': str(filepath),
            'stage': 'storage_save'
        })
        
        return f"artifacts/{filename}"
    
    def get_artifact_url(self, relative_path: str) -> str:
        """
        Get URL for artifact (for serving via FastAPI static files).
        
        Args:
            relative_path: Relative path to artifact
            
        Returns:
            URL path
        """
        return f"/files/{relative_path}"


# Singleton instance
_storage_service = None


def get_storage_service() -> StorageService:
    """Get singleton storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
