import './BrainPreprocessingPanel.css';

export interface BrainPreprocessingPanelProps {
  stages?: {
    [key: string]: string;
  };
  candidateMasks?: {
    [key: string]: string;
  };
  finalMask?: string;
  overlay?: string;
  cropped?: string;
  usedFallback?: boolean;
  fallbackMethod?: string;
}

export function BrainPreprocessingPanel({
  stages,
  candidateMasks,
  finalMask,
  overlay,
  cropped,
  usedFallback,
  fallbackMethod
}: BrainPreprocessingPanelProps) {
  // If no preprocessing data, don't show the panel
  if (!stages && !candidateMasks) {
    return null;
  }

  return (
    <div className="brain-preprocessing-panel card">
      <h3 className="panel-title">Brain Segmentation Preprocessing</h3>
      <p className="panel-description">
        Advanced preprocessing pipeline harmonizes external data to NFBS characteristics
        for improved brain segmentation accuracy.
      </p>
      
      {/* Fallback indicator */}
      {usedFallback && fallbackMethod && (
        <div className="fallback-indicator warning-banner">
          <span className="warning-icon">⚠️</span>
          <span className="warning-text">
            Brain UNet produced empty mask. Using fallback method: <strong>{fallbackMethod.toUpperCase()}</strong>
          </span>
        </div>
      )}

      {/* Preprocessing Stages */}
      {stages && Object.keys(stages).length > 0 && (
        <div className="preprocessing-section">
          <h4 className="section-title">Preprocessing Stages</h4>
          <div className="stages-grid">
            {Object.entries(stages).map(([name, url]) => (
              <div key={name} className="stage-item">
                <img 
                  src={url} 
                  alt={`${name} stage`}
                  className="stage-image"
                />
                <p className="stage-name">{formatStageName(name)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Candidate Masks */}
      {candidateMasks && Object.keys(candidateMasks).length > 0 && (
        <div className="preprocessing-section">
          <h4 className="section-title">Brain Extraction Methods</h4>
          <p className="section-description">
            Multiple thresholding methods for comparison. {getCandidateDescription(usedFallback, fallbackMethod)}
          </p>
          <div className="candidates-grid">
            {Object.entries(candidateMasks).map(([name, url]) => (
              <div key={name} className="candidate-item">
                <img 
                  src={url} 
                  alt={`${name} mask`}
                  className="candidate-image"
                />
                <p className="candidate-name">{name.toUpperCase()}</p>
                <span className={`candidate-badge ${usedFallback && name === fallbackMethod ? 'fallback-badge' : ''}`}>
                  {usedFallback && name === fallbackMethod 
                    ? '✓ Used (Fallback)' 
                    : name === 'otsu' 
                      ? 'Primary' 
                      : 'Alternative'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Final Results */}
      {(finalMask || overlay || cropped) && (
        <div className="preprocessing-section">
          <h4 className="section-title">Final Results</h4>
          <div className="final-results-grid">
            {finalMask && (
              <div className="result-item-img">
                <img 
                  src={finalMask} 
                  alt="Final brain mask"
                  className="result-image"
                />
                <p className="result-name">Binary Mask</p>
              </div>
            )}
            {overlay && (
              <div className="result-item-img">
                <img 
                  src={overlay} 
                  alt="Overlay on original"
                  className="result-image"
                />
                <p className="result-name">Overlay</p>
              </div>
            )}
            {cropped && (
              <div className="result-item-img">
                <img 
                  src={cropped} 
                  alt="Cropped brain region"
                  className="result-image"
                />
                <p className="result-name">Cropped Region</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Format stage name for display (convert snake_case to Title Case).
 */
function formatStageName(name: string): string {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Get description text for candidate masks section.
 */
function getCandidateDescription(usedFallback?: boolean, fallbackMethod?: string): string {
  if (usedFallback && fallbackMethod) {
    return `Fallback method (${fallbackMethod.toUpperCase()}) was used.`;
  }
  return 'The primary method (Otsu) is used for final segmentation.';
}
