"""
Response schemas for API endpoints.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class LogContext(BaseModel):
    """Context information for logging."""
    image_id: Optional[str] = None
    duration: Optional[float] = None
    stage: Optional[str] = None


class PreprocessResponse(BaseModel):
    """Response for preprocessing endpoint."""
    image_id: str
    original_url: str
    grayscale_url: str
    denoised_url: str
    motion_reduced_url: str
    contrast_url: str
    sharpened_url: str
    normalized_url: str
    # Extended comprehensive pipeline stages (optional)
    salt_pepper_cleaned_url: Optional[str] = None
    gaussian_denoised_url: Optional[str] = None
    speckle_denoised_url: Optional[str] = None
    pma_corrected_url: Optional[str] = None
    deblurred_url: Optional[str] = None
    contrast_enhanced_url: Optional[str] = None
    sharpened_url2: Optional[str] = None
    log_context: LogContext


class SegmentResponse(BaseModel):
    """Response for tumor segmentation endpoint."""
    image_id: str
    mask_url: str
    overlay_url: str
    segmented_url: str
    mask_area_pct: float
    log_context: LogContext


class ClassifyResponse(BaseModel):
    """Response for classification endpoint."""
    image_id: str
    class_name: str = Field(alias="class")
    confidence: float
    logits: List[float]
    probabilities: List[float]
    log_context: LogContext
    
    class Config:
        populate_by_name = True


class InferenceResponse(BaseModel):
    """Response for full inference pipeline."""
    image_id: str
    original_url: str
    preprocessing: Dict[str, str]
    tumor_segmentation: Optional[Dict[str, str]] = None
    tumor_segmentation2: Optional[Dict[str, str]] = None  # UNet Tumor model results
    classification: Dict[str, Any]
    duration_seconds: float
    log_context: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
