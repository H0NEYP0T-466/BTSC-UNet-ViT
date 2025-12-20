import './ImagePreview.css';

interface ImagePreviewProps {
  imageUrl: string;
  title: string;
}

export function ImagePreview({ imageUrl, title }: ImagePreviewProps) {
  const handleImageError = () => {
    console.error(`[ImagePreview] Failed to load image: ${title}`);
  };

  return (
    <div className="image-preview card">
      <h3 className="preview-title">{title}</h3>
      <div className="preview-container">
        <img 
          src={imageUrl} 
          alt={title} 
          className="preview-image"
          onError={handleImageError}
        />
      </div>
    </div>
  );
}
