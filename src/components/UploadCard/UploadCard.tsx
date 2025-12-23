import { useState, useRef } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import './UploadCard.css';

interface UploadCardProps {
  onUpload: (file: File, skipPreprocessing: boolean) => void;
  isLoading: boolean;
}

export function UploadCard({ onUpload, isLoading }: UploadCardProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [skipPreprocessing, setSkipPreprocessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        console.log('[UploadCard] File dropped:', file.name);
        onUpload(file, skipPreprocessing);
      } else {
        alert('Please upload an image file');
      }
    }
  };

  const handleFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      console.log('[UploadCard] File selected:', file.name);
      onUpload(file, skipPreprocessing);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="upload-card card">
      <h2 className="upload-title">Upload Brain MRI Image</h2>
      
      {/* Preprocessing toggle */}
      <div className="upload-options">
        <label className="toggle-container">
          <input
            type="checkbox"
            checked={skipPreprocessing}
            onChange={(e) => setSkipPreprocessing(e.target.checked)}
            disabled={isLoading}
          />
          <span className="toggle-label">
            Skip preprocessing (direct to ViT)
          </span>
        </label>
        <p className="toggle-hint">
          {skipPreprocessing 
            ? "⚡ Fast mode: Image goes directly to classification" 
            : "🔧 Full pipeline: Includes denoising, contrast enhancement, and sharpening"}
        </p>
      </div>
      
      <div
        className={`upload-zone ${isDragging ? 'dragging' : ''} ${isLoading ? 'loading' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {isLoading ? (
          <div className="upload-loading">
            <div className="spinner"></div>
            <p className="upload-text">Processing image...</p>
          </div>
        ) : (
          <>
            <div className="upload-icon">📤</div>
            <p className="upload-text">Drag & drop an image here</p>
            <p className="upload-text-secondary">or</p>
            <button
              className="upload-button"
              onClick={handleButtonClick}
              disabled={isLoading}
            >
              Browse Files
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              style={{ display: 'none' }}
            />
          </>
        )}
      </div>
      <p className="upload-hint">
        Supported formats: JPG, PNG, DICOM
      </p>
    </div>
  );
}
