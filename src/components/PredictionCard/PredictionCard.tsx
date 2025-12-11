import './PredictionCard.css';

interface PredictionCardProps {
  className: string;
  confidence: number;
  probabilities: number[];
  logits: number[];
}

const CLASS_DISPLAY_NAMES: { [key: string]: string } = {
  'no_tumor': 'No Tumor',
  'giloma': 'Glioma',
  'meningioma': 'Meningioma',
  'pituitary': 'Pituitary Tumor',
};

const CLASS_COLORS: { [key: string]: string } = {
  'no_tumor': '#10b981',
  'giloma': '#ef4444',
  'meningioma': '#f59e0b',
  'pituitary': '#3b82f6',
};

export function PredictionCard({ className, confidence, probabilities, logits }: PredictionCardProps) {
  const displayName = CLASS_DISPLAY_NAMES[className] || className;
  const color = CLASS_COLORS[className] || 'var(--accent)';
  const classNames = ['no_tumor', 'giloma', 'meningioma', 'pituitary'];

  return (
    <div className="prediction-card card">
      <h2 className="prediction-title">Classification Result</h2>
      
      <div className="prediction-result" style={{ borderColor: color }}>
        <div className="result-badge" style={{ backgroundColor: color }}>
          <span className="badge-text">{displayName}</span>
        </div>
        <div className="confidence-container">
          <span className="confidence-label">Confidence:</span>
          <span className="confidence-value">{(confidence * 100).toFixed(2)}%</span>
        </div>
        <div className="confidence-bar">
          <div 
            className="confidence-fill" 
            style={{ 
              width: `${confidence * 100}%`,
              backgroundColor: color 
            }}
          />
        </div>
      </div>

      <div className="probabilities-section">
        <h3 className="section-title">Class Probabilities</h3>
        <div className="probabilities-list">
          {classNames.map((cls, idx) => (
            <div key={cls} className="probability-item">
              <div className="probability-header">
                <span className="probability-label">
                  {CLASS_DISPLAY_NAMES[cls]}
                </span>
                <span className="probability-value">
                  {(probabilities[idx] * 100).toFixed(2)}%
                </span>
              </div>
              <div className="probability-bar">
                <div 
                  className="probability-fill" 
                  style={{ 
                    width: `${probabilities[idx] * 100}%`,
                    backgroundColor: CLASS_COLORS[cls]
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="logits-section">
        <h3 className="section-title">Raw Logits</h3>
        <div className="logits-grid">
          {classNames.map((cls, idx) => (
            <div key={cls} className="logit-item">
              <span className="logit-label">{CLASS_DISPLAY_NAMES[cls]}</span>
              <span className="logit-value">{logits[idx].toFixed(4)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
