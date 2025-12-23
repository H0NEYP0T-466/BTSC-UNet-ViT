import './PreprocessedGallery.css';

interface PreprocessedGalleryProps {
  images: {
    [key: string]: string;
  };
}

interface Stage {
  key: string;
  label: string;
  tooltip: string;
  isFinal?: boolean;
}

export function PreprocessedGallery({ images }: PreprocessedGalleryProps) {
  // Determine which pipeline mode based on available images
  const hasComprehensiveStages = images.salt_pepper_cleaned_url || 
                                   images.gaussian_denoised_url || 
                                   images.speckle_denoised_url;
  
  // Define comprehensive 8-stage pipeline
  const comprehensiveStages: Stage[] = [
    { key: 'grayscale_url', label: 'Grayscale', tooltip: 'Original converted to grayscale' },
    { key: 'salt_pepper_cleaned_url', label: 'Salt & Pepper Cleaned', tooltip: 'Adaptive median filtering for impulse noise' },
    { key: 'gaussian_denoised_url', label: 'Gaussian Denoised', tooltip: 'Non-Local Means denoising' },
    { key: 'speckle_denoised_url', label: 'Speckle Denoised', tooltip: 'Wavelet BayesShrink for multiplicative noise' },
    { key: 'pma_corrected_url', label: 'Motion Artifact Corrected', tooltip: 'RL/Wiener with estimated motion PSF' },
    { key: 'deblurred_url', label: 'Deblurred', tooltip: 'Wiener/USM deblurring based on blur type' },
    { key: 'contrast_enhanced_url', label: 'Contrast Enhanced', tooltip: 'CLAHE with conservative parameters' },
    { key: 'sharpened_url2', label: 'Sharpened', tooltip: 'Noise-aware unsharp mask with detail thresholding', isFinal: true },
  ];
  
  // Define legacy 6-stage pipeline
  const legacyStages: Stage[] = [
    { key: 'grayscale_url', label: 'Grayscale', tooltip: 'Original converted to grayscale' },
    { key: 'denoised_url', label: 'Denoised', tooltip: 'Edge-preserving noise reduction' },
    { key: 'motion_reduced_url', label: 'Motion Reduced', tooltip: 'Minimal bilateral filtering' },
    { key: 'contrast_url', label: 'Contrast Enhanced', tooltip: 'CLAHE applied within brain mask' },
    { key: 'sharpened_url', label: 'Sharpened', tooltip: 'Unsharp mask for detail recovery' },
    { key: 'normalized_url', label: 'Normalized (Final)', tooltip: 'Z-score normalized', isFinal: true },
  ];
  
  const stages = hasComprehensiveStages ? comprehensiveStages : legacyStages;
  
  // Defensive programming: check if all required images are present
  const availableStages = stages.filter(stage => images[stage.key]);
  const missingStages = stages.filter(stage => !images[stage.key]);
  
  if (missingStages.length > 0) {
    console.info('[PreprocessedGallery] Missing stages:', missingStages.map(s => s.key));
  }

  return (
    <div className="preprocessed-gallery card">
      <h3 className="gallery-title">
        Preprocessing Stages
        <span className="gallery-subtitle">
          {hasComprehensiveStages ? '8-Stage Comprehensive Pipeline' : '6-Stage Legacy Pipeline'}
        </span>
      </h3>
      <div className="gallery-grid">
        {availableStages.map((stage, index) => (
          <div 
            key={stage.key} 
            className={`gallery-item ${stage.isFinal ? 'gallery-item-final' : ''}`}
            title={stage.tooltip}
          >
            <div className="gallery-image-container">
              {images[stage.key] ? (
                <img
                  src={images[stage.key]}
                  alt={stage.label}
                  className="gallery-image"
                  onError={(e) => {
                    console.error(`[PreprocessedGallery] Failed to load image for ${stage.key}`);
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              ) : (
                <div className="gallery-image-placeholder">
                  Image not available
                </div>
              )}
              <div className="stage-number">{index + 1}</div>
              {stage.isFinal && (
                <div className="final-badge">→ To Models</div>
              )}
            </div>
            <p className="gallery-label" title={stage.tooltip}>
              {stage.label}
            </p>
          </div>
        ))}
      </div>
      <div className="gallery-footer">
        <p className="gallery-info">
          {hasComprehensiveStages ? (
            <>
              ✨ <strong>Comprehensive Pipeline:</strong> Advanced noise removal (S&amp;P, Gaussian, Speckle), 
              motion correction, deblurring, and artifact-free enhancement
            </>
          ) : (
            <>
              <strong>Legacy Pipeline:</strong> Basic denoising and enhancement
            </>
          )}
        </p>
      </div>
    </div>
  );
}
