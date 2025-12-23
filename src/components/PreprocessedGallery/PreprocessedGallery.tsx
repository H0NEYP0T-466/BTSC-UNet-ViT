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
  category?: 'conversion' | 'noise' | 'correction' | 'enhancement';
}

export function PreprocessedGallery({ images }: PreprocessedGalleryProps) {
  const FINAL_STAGE = 'sharpened';
  
  const stages: Stage[] = [
    // Resize (NEW!)
    { key: 'resized', label: 'Resized', category: 'conversion' },
    // Conversion
    { key: 'grayscale', label: 'Grayscale', category: 'conversion' },
    // Noise removal
    { key: 'salt_pepper_cleaned', label: 'Salt & Pepper Cleaned', category: 'noise' },
    { key: 'gaussian_denoised', label: 'Gaussian Denoised', category: 'noise' },
    { key: 'speckle_denoised', label: 'Speckle Denoised', category: 'noise' },
    // Motion and blur correction
    { key: 'pma_corrected', label: 'PMA Corrected', category: 'correction' },
    { key: 'deblurred', label: 'Deblurred', category: 'correction' },
    // Enhancement
    { key: 'contrast_enhanced', label: 'Contrast Enhanced', category: 'enhancement' },
    { key: FINAL_STAGE, label: 'Sharpened (Final)', isFinal: true, category: 'enhancement' },
  ];

  // Defensive programming: check if all required images are present
  const missingStages = stages.filter(stage => !images[stage.key]);
  if (missingStages.length > 0) {
    console.error('[PreprocessedGallery] Missing image stages:', missingStages.map(s => s.key));
  }

  // Group stages by category for better organization
  const categoryLabels: Record<string, string> = {
    conversion: '📷 Conversion',
    noise: '🔇 Noise Removal',
    correction: '🔧 Blur & Motion Correction',
    enhancement: '✨ Enhancement',
  };

  const groupedStages = stages.reduce((acc, stage) => {
    const cat = stage.category || 'other';
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(stage);
    return acc;
  }, {} as Record<string, Stage[]>);

  return (
    <div className="preprocessed-gallery card">
      <h3 className="gallery-title">Preprocessing Stages</h3>
      
      {Object.entries(groupedStages).map(([category, categoryStages]) => (
        <div key={category} className="gallery-category">
          <h4 className="category-title">{categoryLabels[category] || category}</h4>
          <div className="gallery-grid">
            {categoryStages.map((stage) => (
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
                    <div className="gallery-image-placeholder">
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
      ))}
    </div>
  );
}
