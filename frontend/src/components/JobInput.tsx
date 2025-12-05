import './JobInput.css';

interface JobInputProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  onSave: () => void;
  isAnalyzing: boolean;
  disabled: boolean;
}

export function JobInput({
  value,
  onChange,
  onAnalyze,
  onSave,
  isAnalyzing,
  disabled,
}: JobInputProps) {
  return (
    <div className="job-input">
      <div className="input-header">
        <label htmlFor="job-description">Job Description</label>
        <span className="char-count">{value.length} characters</span>
      </div>

      <textarea
        id="job-description"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Paste the job description here...

Example:
We are looking for a Senior Machine Learning Engineer to join our team. 
You will work on designing and implementing ML pipelines, deploying models 
to production, and collaborating with cross-functional teams..."
        disabled={disabled}
        rows={12}
      />

      <div className="input-actions">
        <button
          className="btn btn-primary btn-analyze"
          onClick={onAnalyze}
          disabled={disabled || !value.trim()}
        >
          {isAnalyzing ? (
            <>
              <span className="spinner"></span>
              Analyzing with GPT-5...
            </>
          ) : (
            <>
              <span className="analyze-icon">◈</span>
              Analyze & Match
            </>
          )}
        </button>

        <button
          className="btn btn-secondary"
          onClick={onSave}
          disabled={disabled || !value.trim()}
        >
          <span className="save-icon">📁</span>
          Save to Database
        </button>
      </div>
    </div>
  );
}

