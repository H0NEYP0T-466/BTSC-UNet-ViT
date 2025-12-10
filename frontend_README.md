# BTSC-UNet-ViT Frontend

React + TypeScript frontend for Brain Tumor Segmentation and Classification.

## Features

- **Dark Theme (#111)**: Modern dark UI with contrasting accent color (#00C2FF)
- **Drag & Drop Upload**: Easy image upload interface
- **Real-time Processing**: View all intermediate stages
- **Visualization**: Preprocessing stages, segmentation overlays, classification results
- **Responsive Design**: Works on desktop and mobile devices

## Setup

### Prerequisites
- Node.js 18+ and npm

### Installation

1. Install dependencies:
```bash
npm install
```

2. Configure API endpoint:
```bash
# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env
```

### Development

Run development server:
```bash
npm run dev
```

Frontend runs at: http://localhost:5173

### Build

Build for production:
```bash
npm run build
```

Output in `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
src/
├── components/           # React components
│   ├── Header/          # App header
│   ├── Footer/          # App footer
│   ├── UploadCard/      # File upload with drag & drop
│   ├── ImagePreview/    # Image display
│   ├── PreprocessedGallery/  # Preprocessing stages grid
│   ├── SegmentationOverlay/  # Segmentation results
│   └── PredictionCard/  # Classification results
├── pages/
│   └── HomePage.tsx     # Main page layout
├── services/
│   ├── api.ts          # API client (Axios)
│   └── types.ts        # TypeScript types
├── theme/
│   ├── variables.css   # CSS variables (colors, spacing)
│   └── global.css      # Global styles
├── App.tsx             # Root component
└── main.tsx            # Entry point
```

## Component Design

### UploadCard
- Drag and drop zone
- File browser button
- Loading spinner during processing
- Accepts image files (JPG, PNG, DICOM)

### PreprocessedGallery
- Grid layout showing 6 preprocessing stages:
  - Grayscale
  - Denoised
  - Motion Reduced
  - Contrast Enhanced
  - Sharpened
  - Normalized

### SegmentationOverlay
- Three views: Binary mask, Overlay, Cropped tumor
- Opacity slider for overlay control

### PredictionCard
- Classification result badge with color coding
- Confidence bar
- Class probabilities (4 classes)
- Raw logits display

## Theme Customization

Edit `src/theme/variables.css`:

```css
:root {
  --bg-primary: #111;      /* Main background */
  --accent: #00C2FF;       /* Accent color */
  /* ... other variables */
}
```

## API Integration

The app uses Axios to communicate with the backend:

```typescript
// Example: Run full inference
const result = await apiClient.runInference(file);
```

Available endpoints:
- `/api/health` - Health check
- `/api/preprocess` - Preprocessing only
- `/api/segment` - Segmentation only
- `/api/classify` - Classification only
- `/api/inference` - Full pipeline (recommended)

## Styling

- **No Tailwind CSS** - Uses separate CSS files per component
- **Dark Theme** - Background #111, accent #00C2FF
- **CSS Variables** - Consistent theming
- **Responsive** - Mobile-friendly grid layouts

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Development Tips

### Hot Module Replacement (HMR)
Changes to `.tsx` and `.css` files reload automatically.

### Type Safety
Use TypeScript types from `services/types.ts` for API responses.

### Logging
Check browser console for API calls and debugging:
```
[API] POST /api/inference
[HomePage] Starting inference for file: image.jpg
```

## Troubleshooting

### CORS Errors
Ensure backend `CORS_ORIGINS` includes `http://localhost:5173`.

### Images not loading
Check that backend is serving static files at `/files` route.

### API connection failed
Verify `VITE_API_URL` in `.env` points to running backend.

## License

MIT

## Authors

BTSC-UNet-ViT Team
