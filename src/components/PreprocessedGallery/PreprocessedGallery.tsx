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
    { key: 'skull_stripped', label: 'Brain Only (HD-BET)' },
    { key: 'brain_mask', label: 'Brain Mask' },
    { key: 'denoised', label: 'Denoised' },
    { key: 'motion_reduced', label: 'Motion Reduced' },
    { key: 'contrast', label: 'Contrast Enhanced' },
    { key: 'sharpened', label: 'Sharpened' },
    { key: FINAL_STAGE, label: 'Normalized (Final)', isFinal: true },
  ];

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
              <img
                src={images[stage.key]}
                alt={stage.label}
                className="gallery-image"
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
