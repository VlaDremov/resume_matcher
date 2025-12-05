import { useState } from 'react';
import { analyzeJob, saveVacancy, getPdfUrl, AnalyzeResponse } from './api';
import { JobInput } from './components/JobInput';
import { ScoreDisplay } from './components/ScoreDisplay';
import { PdfViewer } from './components/PdfViewer';
import './App.css';

function App() {
  const [jobDescription, setJobDescription] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saveFilename, setSaveFilename] = useState('');
  const [showSaveModal, setShowSaveModal] = useState(false);

  const handleAnalyze = async () => {
    if (!jobDescription.trim()) {
      setError('Please enter a job description');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const analysisResult = await analyzeJob(jobDescription, true, true);
      setResult(analysisResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSave = async () => {
    if (!jobDescription.trim() || !saveFilename.trim()) {
      return;
    }

    setIsSaving(true);

    try {
      await saveVacancy(jobDescription, saveFilename);
      setShowSaveModal(false);
      setSaveFilename('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">◈</span>
          <h1>Resume Matcher</h1>
        </div>
        <div className="header-subtitle">
          GPT-5 Powered Resume Optimization
        </div>
      </header>

      <main className="app-main">
        <div className="input-section">
          <JobInput
            value={jobDescription}
            onChange={setJobDescription}
            onAnalyze={handleAnalyze}
            onSave={() => setShowSaveModal(true)}
            isAnalyzing={isAnalyzing}
            disabled={isAnalyzing}
          />

          {error && (
            <div className="error-message">
              <span className="error-icon">⚠</span>
              {error}
            </div>
          )}
        </div>

        {result && (
          <div className="results-section">
            <ScoreDisplay result={result} />

            <div className="pdf-section">
              <div className="section-header">
                <h2>Best Match: {result.best_variant_display}</h2>
                <div className="download-buttons">
                  <a
                    href={getPdfUrl(result.best_variant)}
                    download
                    className="btn btn-secondary"
                  >
                    ↓ Download PDF
                  </a>
                </div>
              </div>

              <PdfViewer
                url={getPdfUrl(result.best_variant)}
                variant={result.best_variant}
              />
            </div>

            {result.key_matches.length > 0 && (
              <div className="keywords-section">
                <h3>Matched Keywords</h3>
                <div className="keywords-list">
                  {result.key_matches.map((keyword, idx) => (
                    <span key={idx} className="keyword-tag matched">
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {result.missing_keywords.length > 0 && (
              <div className="keywords-section">
                <h3>Missing Keywords</h3>
                <div className="keywords-list">
                  {result.missing_keywords.map((keyword, idx) => (
                    <span key={idx} className="keyword-tag missing">
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {result.reasoning && (
              <div className="reasoning-section">
                <h3>Analysis</h3>
                <p>{result.reasoning}</p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Save Modal */}
      {showSaveModal && (
        <div className="modal-overlay" onClick={() => setShowSaveModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Save to Database</h2>
            <p>Save this job description for future reference and keyword analysis.</p>
            
            <input
              type="text"
              placeholder="Filename (e.g., google_ml_engineer)"
              value={saveFilename}
              onChange={(e) => setSaveFilename(e.target.value)}
              className="modal-input"
            />

            <div className="modal-buttons">
              <button
                className="btn btn-secondary"
                onClick={() => setShowSaveModal(false)}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleSave}
                disabled={isSaving || !saveFilename.trim()}
              >
                {isSaving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

