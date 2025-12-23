import { useState } from 'react';
import { Header } from '../components/Header/Header';
import { Footer } from '../components/Footer/Footer';
import { UploadCard } from '../components/UploadCard/UploadCard';
import { ImagePreview } from '../components/ImagePreview/ImagePreview';
import { PreprocessedGallery } from '../components/PreprocessedGallery/PreprocessedGallery';
import { SegmentationOverlay } from '../components/SegmentationOverlay/SegmentationOverlay';
import { PredictionCard } from '../components/PredictionCard/PredictionCard';
import { apiClient } from '../services/api';
import type { InferenceResponse } from '../services/types';
import './HomePage.css';

export function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);

  const handleUpload = async (file: File, skipPreprocessing: boolean) => {
    console.log('[HomePage] Starting inference for file:', file.name, 'skipPreprocessing:', skipPreprocessing);
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiClient.runInference(file, skipPreprocessing);
      console.log('[HomePage] Inference completed:', response);
      
      // Validate response structure
      if (!response.image_id || !response.preprocessing || !response.classification) {
        throw new Error('Incomplete response from API');
      }
      
      // Log preprocessing keys
      console.log('[HomePage] Preprocessing keys:', Object.keys(response.preprocessing));
      
      // Log tumor segmentation keys if present
      if (response.tumor_segmentation) {
        console.log('[HomePage] Tumor segmentation keys (UNet1):', Object.keys(response.tumor_segmentation));
      } else {
        console.log('[HomePage] No tumor segmentation UNet1 (no tumor detected)');
      }
      
      // Log tumor segmentation2 keys if present
      if (response.tumor_segmentation2) {
        console.log('[HomePage] Tumor segmentation2 keys (UNet2):', Object.keys(response.tumor_segmentation2));
      } else {
        console.log('[HomePage] No tumor segmentation UNet2 (no tumor detected)');
      }
      
      // Log classification data
      console.log('[HomePage] Classification:', {
        class: response.classification.class,
        confidence: response.classification.confidence,
        probabilities_length: response.classification.probabilities?.length,
        logits_length: response.classification.logits?.length
      });
      
      setResult(response);
    } catch (err: unknown) {
      console.error('[HomePage] Inference failed:', err);
      const error = err as { response?: { data?: { detail?: string } }; message?: string };
      setError(error.response?.data?.detail || error.message || 'Failed to process image. Please try again.');
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
                      grayscale: result.preprocessing.grayscale ? apiClient.getResourceUrl(result.preprocessing.grayscale) : undefined,
                      denoised: result.preprocessing.denoised ? apiClient.getResourceUrl(result.preprocessing.denoised) : undefined,
                      motion_reduced: result.preprocessing.motion_reduced ? apiClient.getResourceUrl(result.preprocessing.motion_reduced) : undefined,
                      contrast: result.preprocessing.contrast ? apiClient.getResourceUrl(result.preprocessing.contrast) : undefined,
                      sharpened: result.preprocessing.sharpened ? apiClient.getResourceUrl(result.preprocessing.sharpened) : undefined,
                      normalized: apiClient.getResourceUrl(result.preprocessing.normalized),
                    }}
                  />
                </div>

                {/* Tumor Segmentation Results - UNet1 (BraTS model) */}
                {result.tumor_segmentation && (
                  <div className="result-item">
                    <SegmentationOverlay
                      title="Tumor Segmentation (UNet1 - BraTS)"
                      maskUrl={apiClient.getResourceUrl(result.tumor_segmentation.mask)}
                      overlayUrl={apiClient.getResourceUrl(result.tumor_segmentation.overlay)}
                      segmentedUrl={apiClient.getResourceUrl(result.tumor_segmentation.segmented)}
                    />
                  </div>
                )}

                {/* Tumor Segmentation Results - UNet2 (PNG model) */}
                {result.tumor_segmentation2 && (
                  <div className="result-item">
                    <SegmentationOverlay
                      title="Tumor Segmentation (UNet2 - PNG)"
                      maskUrl={apiClient.getResourceUrl(result.tumor_segmentation2.mask)}
                      overlayUrl={apiClient.getResourceUrl(result.tumor_segmentation2.overlay)}
                      segmentedUrl={apiClient.getResourceUrl(result.tumor_segmentation2.segmented)}
                    />
                  </div>
                )}

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
