import { useState } from 'react';
import { AnalyzeResponse } from '../api';
import './ScoreDisplay.css';

interface ScoreDisplayProps {
  result: AnalyzeResponse;
}

export function ScoreDisplay({ result }: ScoreDisplayProps) {
  const [keywordsExpanded, setKeywordsExpanded] = useState(false);

  // * Sort category scores in descending order and take top 5
  const sortedScores = [...result.category_scores]
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  // * Find the best variant's score from category_scores
  const bestScore = result.category_scores.find(
    (cat) => cat.category === result.best_variant
  );
  const bestScorePercent = bestScore ? Math.round(bestScore.score * 100) : 0;

  const totalKeywords = result.key_matches.length + result.missing_keywords.length;
  const matchedCount = result.key_matches.length;

  return (
    <div className="score-display">
      {/* Best Match - Compact */}
      <div className="best-match">
        <span className="best-match-label">Best Match</span>
        <span className="best-match-value">
          {result.best_variant_display}
          <span className="best-match-score">{bestScorePercent}%</span>
        </span>
      </div>

      {/* Top 5 Resume Types */}
      <div className="scores-list">
        <h3 className="scores-header">Match Scores</h3>
        {sortedScores.map((cat) => {
          const percent = Math.round(cat.score * 100);
          const isBest = cat.category === result.best_variant;
          return (
            <div key={cat.category} className={`score-row ${isBest ? 'is-best' : ''}`}>
              <span className="score-name">{cat.display_name}</span>
              <div className="score-bar-container">
                <div
                  className="score-bar-fill"
                  style={{ width: `${percent}%` }}
                />
              </div>
              <span className="score-percent">{percent}%</span>
            </div>
          );
        })}
      </div>

      {/* Keyword Coverage - Expandable */}
      {totalKeywords > 0 && (
        <div className="keywords-section">
          <button
            className="keywords-toggle"
            onClick={() => setKeywordsExpanded(!keywordsExpanded)}
          >
            <span className="keywords-summary">
              Keywords: <strong>{matchedCount}/{totalKeywords}</strong> covered
            </span>
            <span className={`keywords-chevron ${keywordsExpanded ? 'expanded' : ''}`}>
              ▾
            </span>
          </button>

          {keywordsExpanded && (
            <div className="keywords-expanded">
              {result.key_matches.length > 0 && (
                <div className="keywords-group">
                  <span className="keywords-group-label matched">Matched</span>
                  <div className="keywords-tags">
                    {result.key_matches.map((keyword, idx) => (
                      <span key={idx} className="keyword-tag matched">
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {result.missing_keywords.length > 0 && (
                <div className="keywords-group">
                  <span className="keywords-group-label missing">Missing</span>
                  <div className="keywords-tags">
                    {result.missing_keywords.map((keyword, idx) => (
                      <span key={idx} className="keyword-tag missing">
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
