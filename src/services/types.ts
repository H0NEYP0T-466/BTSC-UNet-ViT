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
  // Noise removal stages
  salt_pepper_cleaned_url: string;
  gaussian_denoised_url: string;
  speckle_denoised_url: string;
  // Motion and blur correction stages
  pma_corrected_url: string;
  deblurred_url: string;
  // Enhancement stages
  contrast_enhanced_url: string;
  sharpened_url: string;  // Final output
  // Detection results
  noise_detected?: string;
  blur_detected?: boolean;
  motion_detected?: boolean;
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
    salt_pepper_cleaned: string;
    gaussian_denoised: string;
    speckle_denoised: string;
    pma_corrected: string;
    deblurred: string;
    contrast_enhanced: string;
    sharpened: string;  // Final output
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
