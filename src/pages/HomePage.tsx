import { useState } from 'react';
import { Header } from '../components/Header/Header';
import { Footer } from '../components/Footer/Footer';
import { UploadCard } from '../components/UploadCard/UploadCard';
import { ImagePreview } from '../components/ImagePreview/ImagePreview';
import { PreprocessedGallery } from '../components/PreprocessedGallery/PreprocessedGallery';
import { BrainPreprocessingPanel } from '../components/BrainPreprocessingPanel/BrainPreprocessingPanel';
import { SegmentationOverlay } from '../components/SegmentationOverlay/SegmentationOverlay';
import { PredictionCard } from '../components/PredictionCard/PredictionCard';
import { apiClient } from '../services/api';
import type { InferenceResponse } from '../services/types';
import './HomePage.css';

export function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);

  const handleUpload = async (file: File) => {
    console.log('[HomePage] Starting inference for file:', file.name);
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiClient.runInference(file);
      console.log('[HomePage] Inference completed:', response);
      setResult(response);
    } catch (err: unknown) {
      console.error('[HomePage] Inference failed:', err);
      const error = err as { response?: { data?: { detail?: string } } };
      setError(error.response?.data?.detail || 'Failed to process image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="home-page">
      <Header />
      
      <main className="main-content">
        <div className="container">
          <div className="content-grid">
            {/* Upload Section */}
            <div className="upload-section">
              <UploadCard onUpload={handleUpload} isLoading={isLoading} />
              
              {error && (
                <div className="error-message">
                  <p className="error-text">⚠️ {error}</p>
                </div>
              )}
            </div>

            {/* Results Section */}
            {result && (
              <div className="results-section">
                {/* Original Image */}
                <div className="result-item">
                  <ImagePreview
                    imageUrl={apiClient.getResourceUrl(result.original_url)}
                    title="Original Image"
                  />
                </div>

                {/* Preprocessing Gallery */}
                <div className="result-item">
                  <PreprocessedGallery
                    images={{
                      grayscale: apiClient.getResourceUrl(result.preprocessing.grayscale),
                      denoised: apiClient.getResourceUrl(result.preprocessing.denoised),
                      motion_reduced: apiClient.getResourceUrl(result.preprocessing.motion_reduced),
                      contrast: apiClient.getResourceUrl(result.preprocessing.contrast),
                      sharpened: apiClient.getResourceUrl(result.preprocessing.sharpened),
                      normalized: apiClient.getResourceUrl(result.preprocessing.normalized),
                    }}
                  />
                </div>

                {/* Brain Segmentation Results */}
                <div className="result-item">
                  <SegmentationOverlay
                    title="Brain Segmentation"
                    maskUrl={apiClient.getResourceUrl(result.brain_segmentation.mask)}
                    overlayUrl={apiClient.getResourceUrl(result.brain_segmentation.overlay)}
                    segmentedUrl={apiClient.getResourceUrl(result.brain_segmentation.brain_extracted)}
                  />
                </div>

                {/* Brain Preprocessing Panel - NEW */}
                {(result.brain_segmentation.preprocessing_stages || 
                  result.brain_segmentation.candidate_masks) && (
                  <div className="result-item">
                    <BrainPreprocessingPanel
                      stages={result.brain_segmentation.preprocessing_stages ? 
                        Object.fromEntries(
                          Object.entries(result.brain_segmentation.preprocessing_stages).map(
                            ([key, value]) => [key, apiClient.getResourceUrl(value as string)]
                          )
                        ) : undefined
                      }
                      candidateMasks={result.brain_segmentation.candidate_masks ?
                        Object.fromEntries(
                          Object.entries(result.brain_segmentation.candidate_masks).map(
                            ([key, value]) => [key, apiClient.getResourceUrl(value as string)]
                          )
                        ) : undefined
                      }
                      usedFallback={result.brain_segmentation.used_fallback}
                      fallbackMethod={result.brain_segmentation.fallback_method}
                    />
                  </div>
                )}

                {/* Tumor Segmentation Results */}
                <div className="result-item">
                  <SegmentationOverlay
                    title="Tumor Segmentation"
                    maskUrl={apiClient.getResourceUrl(result.tumor_segmentation.mask)}
                    overlayUrl={apiClient.getResourceUrl(result.tumor_segmentation.overlay)}
                    segmentedUrl={apiClient.getResourceUrl(result.tumor_segmentation.segmented)}
                  />
                </div>

                {/* Classification Results */}
                <div className="result-item">
                  <PredictionCard
                    className={result.classification.class}
                    confidence={result.classification.confidence}
                    probabilities={result.classification.probabilities}
                    logits={result.classification.logits}
                  />
                </div>

                {/* Metadata */}
                <div className="result-item">
                  <div className="metadata-card card">
                    <h3 className="metadata-title">Processing Metadata</h3>
                    <div className="metadata-grid">
                      <div className="metadata-item">
                        <span className="metadata-label">Image ID:</span>
                        <span className="metadata-value">{result.image_id}</span>
                      </div>
                      <div className="metadata-item">
                        <span className="metadata-label">Duration:</span>
                        <span className="metadata-value">
                          {result.duration_seconds.toFixed(2)}s
                        </span>
                      </div>
                      <div className="metadata-item">
                        <span className="metadata-label">Class:</span>
                        <span className="metadata-value">{result.classification.class}</span>
                      </div>
                      <div className="metadata-item">
                        <span className="metadata-label">Confidence:</span>
                        <span className="metadata-value">
                          {(result.classification.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}
