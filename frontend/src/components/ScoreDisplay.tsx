import { AnalyzeResponse } from '../api';
import './ScoreDisplay.css';

interface ScoreDisplayProps {
  result: AnalyzeResponse;
}

export function ScoreDisplay({ result }: ScoreDisplayProps) {
  const scoreColor = getScoreColor(result.relevancy_score);

  return (
    <div className="score-display">
      <div className="score-main">
        <div className="score-circle" style={{ '--score-color': scoreColor } as React.CSSProperties}>
          <svg viewBox="0 0 100 100" className="score-ring">
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="var(--bg-tertiary)"
              strokeWidth="8"
            />
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke={scoreColor}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${result.relevancy_score * 2.83} 283`}
              transform="rotate(-90 50 50)"
              className="score-progress"
            />
          </svg>
          <div className="score-value">
            <span className="score-number">{result.relevancy_score}</span>
            <span className="score-label">Relevancy</span>
          </div>
        </div>

        <div className="score-info">
          <h2>Match Analysis</h2>
          <p className="score-description">
            {getScoreDescription(result.relevancy_score)}
          </p>
        </div>
      </div>

      <div className="category-scores">
        <h3>Category Breakdown</h3>
        <div className="category-list">
          {result.category_scores.slice(0, 5).map((cat) => (
            <div key={cat.category} className="category-item">
              <div className="category-header">
                <span className="category-name">{cat.display_name}</span>
                <span className="category-value">{Math.round(cat.score * 100)}%</span>
              </div>
              <div className="category-bar">
                <div
                  className="category-fill"
                  style={{
                    width: `${cat.score * 100}%`,
                    background: cat.category === result.best_variant
                      ? 'var(--accent-primary)'
                      : 'var(--bg-tertiary)',
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function getScoreColor(score: number): string {
  if (score >= 80) return '#22c55e';
  if (score >= 60) return '#84cc16';
  if (score >= 40) return '#f59e0b';
  if (score >= 20) return '#f97316';
  return '#ef4444';
}

function getScoreDescription(score: number): string {
  if (score >= 80) {
    return 'Excellent match! Your resume strongly aligns with this job description.';
  }
  if (score >= 60) {
    return 'Good match. Your resume covers most key requirements.';
  }
  if (score >= 40) {
    return 'Moderate match. Consider emphasizing relevant skills.';
  }
  if (score >= 20) {
    return 'Partial match. Significant gaps between resume and job requirements.';
  }
  return 'Low match. This role may require different experience.';
}

