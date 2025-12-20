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
    log_context: LogContext


class BrainSegmentResponse(BaseModel):
    """Response for brain segmentation endpoint."""
    image_id: str
    mask_url: str
    overlay_url: str
    brain_extracted_url: str
    brain_area_pct: float
    log_context: LogContext
    # New fields for advanced preprocessing
    preprocessing_stages: Optional[Dict[str, str]] = None
    candidate_masks: Optional[Dict[str, str]] = None
    # Fallback fields
    used_fallback: bool = False
    fallback_method: Optional[str] = None


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
    brain_segmentation: Dict[str, str]
    tumor_segmentation: Dict[str, str]
    classification: Dict[str, Any]
    duration_seconds: float
    log_context: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
