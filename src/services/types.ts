/**
 * TypeScript types for API responses and data structures.
 */

export interface LogContext {
  image_id?: string;
  duration?: number;
  stage?: string;
}

export interface PreprocessResponse {
  image_id: string;
  original_url: string;
  grayscale_url: string;
  denoised_url: string;
  motion_reduced_url: string;
  contrast_url: string;
  sharpened_url: string;
  normalized_url: string;
  log_context: LogContext;
}

export interface SegmentResponse {
  image_id: string;
  mask_url: string;
  overlay_url: string;
  segmented_url: string;
  mask_area_pct: number;
  log_context: LogContext;
}

export interface ClassifyResponse {
  image_id: string;
  class: string;
  confidence: number;
  logits: number[];
  probabilities: number[];
  log_context: LogContext;
}

export interface InferenceResponse {
  image_id: string;
  original_url: string;
  preprocessing: {
    grayscale: string;
    denoised: string;
    motion_reduced: string;
    contrast: string;
    sharpened: string;
    normalized: string;
  };
  tumor_segmentation?: {
    mask: string;
    overlay: string;
    segmented: string;
    heatmap?: string;
    probability_map?: string;
  };
  tumor_segmentation2?: {
    mask: string;
    overlay: string;
    segmented: string;
    heatmap?: string;
    probability_map?: string;
  };
  classification: {
    class: string;
    confidence: number;
    logits: number[];
    probabilities: number[];
  };
  duration_seconds: number;
  log_context: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: {
    unet: boolean;
    unet_tumor: boolean;
    vit: boolean;
  };
}
