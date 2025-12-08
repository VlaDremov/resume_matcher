import { useState } from 'react';
import { analyzeJob, saveVacancy, AnalyzeResponse } from './api';
import { JobInput } from './components/JobInput';
import { ScoreDisplay } from './components/ScoreDisplay';
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
      const analysisResult = await analyzeJob(jobDescription, true);
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
          Hybrid AI-Powered Resume Optimization
        </div>
      </header>

      <main className={`app-main ${result ? 'has-results' : ''}`}>
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

