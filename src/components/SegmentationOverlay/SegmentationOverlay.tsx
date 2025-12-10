import { useState } from 'react';
import './SegmentationOverlay.css';

interface SegmentationOverlayProps {
  maskUrl: string;
  overlayUrl: string;
  segmentedUrl: string;
}

export function SegmentationOverlay({ maskUrl, overlayUrl, segmentedUrl }: SegmentationOverlayProps) {
  const [opacity, setOpacity] = useState(0.5);

  return (
    <div className="segmentation-overlay card">
      <h3 className="overlay-title">Segmentation Results</h3>
      
      <div className="overlay-grid">
        <div className="overlay-item">
          <div className="overlay-image-container">
            <img src={maskUrl} alt="Segmentation Mask" className="overlay-image" />
          </div>
          <p className="overlay-label">Binary Mask</p>
        </div>

        <div className="overlay-item">
          <div className="overlay-image-container">
            <img src={overlayUrl} alt="Overlay" className="overlay-image" />
          </div>
          <p className="overlay-label">Overlay on Original</p>
        </div>

        <div className="overlay-item">
          <div className="overlay-image-container">
            <img src={segmentedUrl} alt="Segmented Tumor" className="overlay-image" />
          </div>
          <p className="overlay-label">Cropped Tumor Region</p>
        </div>
      </div>

      <div className="opacity-control">
        <label htmlFor="opacity-slider" className="opacity-label">
          Overlay Opacity: {Math.round(opacity * 100)}%
        </label>
        <input
          id="opacity-slider"
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={opacity}
          onChange={(e) => setOpacity(parseFloat(e.target.value))}
          className="opacity-slider"
        />
      </div>
    </div>
  );
}
