import './PreprocessedGallery.css';

interface PreprocessedGalleryProps {
  images: {
    [key: string]: string;
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

  // Defensive programming: check if all required images are present
  const missingStages = stages.filter(stage => !images[stage.key]);
  if (missingStages.length > 0) {
    console.error('[PreprocessedGallery] Missing image stages:', missingStages.map(s => s.key));
  }

  return (
    <div className="preprocessed-gallery card">
      <h3 className="gallery-title">Preprocessing Stages</h3>
      <div className="gallery-grid">
        {stages.map((stage) => (
          <div 
            key={stage.key} 
            className={`gallery-item ${stage.isFinal ? 'gallery-item-final' : ''}`}
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
                <div style={{
                  width: '100%',
                  height: '200px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-muted)'
                }}>
                  Image not available
                </div>
              )}
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
