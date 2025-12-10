import './ImagePreview.css';

interface ImagePreviewProps {
  imageUrl: string;
  title: string;
}

export function ImagePreview({ imageUrl, title }: ImagePreviewProps) {
  return (
    <div className="image-preview card">
      <h3 className="preview-title">{title}</h3>
      <div className="preview-container">
        <img src={imageUrl} alt={title} className="preview-image" />
      </div>
    </div>
  );
}
