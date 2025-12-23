"""
Request schemas for API endpoints.
"""
from typing import Optional
from pydantic import BaseModel, Field


class PreprocessRequest(BaseModel):
    """Request for preprocessing endpoint."""
    # File will be handled via FastAPI's UploadFile
    pass


class SegmentRequest(BaseModel):
    """Request for segmentation endpoint."""
    # File will be handled via FastAPI's UploadFile
    pass


class ClassifyRequest(BaseModel):
    """Request for classification endpoint."""
    # File will be handled via FastAPI's UploadFile
    pass


class InferenceRequest(BaseModel):
    """Request for full inference pipeline."""
    skip_preprocessing: Optional[bool] = Field(
        default=False,
        description="Skip preprocessing pipeline and pass image directly to ViT"
    )
