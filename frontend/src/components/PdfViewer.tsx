import { useState } from 'react';
import './PdfViewer.css';

interface PdfViewerProps {
  url: string;
  variant: string;
}

export function PdfViewer({ url, variant }: PdfViewerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  return (
    <div className="pdf-viewer">
      {isLoading && !hasError && (
        <div className="pdf-loading">
          <div className="loading-spinner"></div>
          <p>Loading PDF preview...</p>
        </div>
      )}

      {hasError && (
        <div className="pdf-error">
          <div className="error-icon">📄</div>
          <h3>PDF Preview Unavailable</h3>
          <p>
            The PDF for <strong>{variant}</strong> variant could not be loaded.
          </p>
          <p className="error-hint">
            Make sure the PDF has been compiled from the LaTeX source.
          </p>
          <a href={url} download className="btn btn-primary">
            Try Download Instead
          </a>
        </div>
      )}

      <iframe
        src={url}
        title={`Resume - ${variant}`}
        onLoad={() => setIsLoading(false)}
        onError={() => {
          setIsLoading(false);
          setHasError(true);
        }}
        style={{ display: hasError ? 'none' : 'block' }}
      />
    </div>
  );
}

