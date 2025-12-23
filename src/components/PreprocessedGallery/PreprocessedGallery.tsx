import './PreprocessedGallery.css';

interface PreprocessedGalleryProps {
  images: {
    [key: string]: string | undefined;
  };
}

interface Stage {
  key: string;
  label: string;
  isFinal?: boolean;
}

export function PreprocessedGallery({ images }: PreprocessedGalleryProps) {
  const FINAL_STAGE = 'normalized';
  
  const stages: Stage[] = [
    { key: 'grayscale', label: 'Grayscale' },
    { key: 'denoised', label: 'Denoised' },
    { key: 'motion_reduced', label: 'Motion Reduced' },
    { key: 'contrast', label: 'Contrast Enhanced' },
    { key: 'sharpened', label: 'Sharpened' },
    { key: FINAL_STAGE, label: 'Normalized (Final)', isFinal: true },
  ];

  // Filter to only show stages that have images
  const availableStages = stages.filter(stage => images[stage.key]);
  
  // Log missing stages for debugging
  const missingStages = stages.filter(stage => !images[stage.key]);
  if (missingStages.length > 0) {
    console.log('[PreprocessedGallery] Skipped stages (preprocessing disabled):', missingStages.map(s => s.key));
  }

  return (
    <div className="preprocessed-gallery card">
      <h3 className="gallery-title">
        Preprocessing Stages
        {availableStages.length < stages.length && (
          <span className="gallery-subtitle"> (Fast mode - minimal processing)</span>
        )}
      </h3>
      <div className="gallery-grid">
        {availableStages.map((stage) => (
          <div 
            key={stage.key} 
            className={`gallery-item ${stage.isFinal ? 'gallery-item-final' : ''}`}
          >
            <div className="gallery-image-container">
              <img
                src={images[stage.key]!}
                alt={stage.label}
                className="gallery-image"
                onError={(e) => {
                  console.error(`[PreprocessedGallery] Failed to load image for ${stage.key}`);
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
              {stage.isFinal && (
                <div className="final-badge">→ To Models</div>
              )}
            </div>
            <p className="gallery-label">{stage.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
