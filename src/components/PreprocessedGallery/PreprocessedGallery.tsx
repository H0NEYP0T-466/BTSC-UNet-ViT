import './PreprocessedGallery.css';

interface PreprocessedGalleryProps {
  images: {
    [key: string]: string;
  };
}

export function PreprocessedGallery({ images }: PreprocessedGalleryProps) {
  const stages = [
    { key: 'grayscale', label: 'Grayscale' },
    { key: 'denoised', label: 'Denoised' },
    { key: 'motion_reduced', label: 'Motion Reduced' },
    { key: 'contrast', label: 'Contrast Enhanced' },
    { key: 'sharpened', label: 'Sharpened' },
    { key: 'normalized', label: 'Normalized' },
  ];

  return (
    <div className="preprocessed-gallery card">
      <h3 className="gallery-title">Preprocessing Stages</h3>
      <div className="gallery-grid">
        {stages.map((stage) => (
          <div key={stage.key} className="gallery-item">
            <div className="gallery-image-container">
              <img
                src={images[stage.key]}
                alt={stage.label}
                className="gallery-image"
              />
            </div>
            <p className="gallery-label">{stage.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
