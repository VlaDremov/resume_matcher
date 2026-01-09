import { useState } from 'react';
import {
  AnalyzeResponse,
  CategorizedKeywords,
  KeywordCategory,
  KeywordWithMetadata,
  MarketTrendsInfo,
} from '../api';
import './ScoreDisplay.css';

interface ScoreDisplayProps {
  result: AnalyzeResponse;
}

// * Category display configuration
const CATEGORY_CONFIG: Record<KeywordCategory, { label: string; color: string }> = {
  research_ml: { label: 'Research ML', color: '#2563EB' },
  applied_production: { label: 'Applied Prod', color: '#16A34A' },
  genai_llm: { label: 'GenAI & LLM', color: '#F97316' },
  general: { label: 'General', color: '#6B7280' },
};

// * All category keys in display order
const CATEGORIES: KeywordCategory[] = [
  'research_ml',
  'applied_production',
  'genai_llm',
  'general',
];

// * Count total keywords across all categories
function countKeywords(categorized: CategorizedKeywords): number {
  return CATEGORIES.reduce((sum, cat) => sum + categorized[cat].length, 0);
}

// * Keyword tag component with importance indicator
function KeywordTag({ keyword }: { keyword: KeywordWithMetadata }) {
  const importanceClass = keyword.importance === 'critical'
    ? 'critical'
    : keyword.importance === 'important'
      ? 'important'
      : '';

  return (
    <span
      className={`keyword-tag-v2 ${keyword.is_matched ? 'matched' : 'missing'} ${importanceClass}`}
      title={`${keyword.importance.replace('_', ' ')}${keyword.is_trending ? ' (Trending)' : ''}`}
    >
      {keyword.keyword}
      {keyword.importance === 'critical' && (
        <span className="importance-indicator">!</span>
      )}
      {keyword.is_trending && (
        <span className="trending-indicator">↑</span>
      )}
    </span>
  );
}

// * Category section component
function CategorySection({
  category,
  keywords,
}: {
  category: KeywordCategory;
  keywords: KeywordWithMetadata[];
}) {
  if (keywords.length === 0) return null;

  const config = CATEGORY_CONFIG[category];

  return (
    <div className="category-section">
      <div className="category-header" style={{ '--category-color': config.color } as React.CSSProperties}>
        <span className="category-dot" />
        <span className="category-label">{config.label}</span>
        <span className="category-count">{keywords.length}</span>
      </div>
      <div className="keywords-tags-v2">
        {keywords.map((kw, idx) => (
          <KeywordTag key={`${kw.keyword}-${idx}`} keyword={kw} />
        ))}
      </div>
    </div>
  );
}

export function ScoreDisplay({ result }: ScoreDisplayProps) {
  const [keywordsExpanded, setKeywordsExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<'matched' | 'missing'>('matched');

  // * Sort category scores in descending order and take top 5
  const sortedScores = [...result.category_scores]
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  // * Find the best variant's score from category_scores
  const bestScore = result.category_scores.find(
    (cat) => cat.category === result.best_variant
  );
  const bestScorePercent = bestScore ? Math.round(bestScore.score * 100) : 0;

  // * Count keywords using new categorized data if available, else fallback to legacy
  const hasCategorizedData = result.categorized_matches && result.categorized_missing;

  const matchedCount = hasCategorizedData
    ? countKeywords(result.categorized_matches)
    : result.key_matches.length;

  const missingCount = hasCategorizedData
    ? countKeywords(result.categorized_missing)
    : result.missing_keywords.length;

  const totalKeywords = matchedCount + missingCount;

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

      {/* Keyword Coverage - Expandable with Categories */}
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
              {/* Use new categorized view if available */}
              {hasCategorizedData ? (
                <>
                  {/* Tab switcher */}
                  <div className="keywords-tabs">
                    <button
                      className={`tab ${activeTab === 'matched' ? 'active' : ''}`}
                      onClick={() => setActiveTab('matched')}
                    >
                      Matched ({matchedCount})
                    </button>
                    <button
                      className={`tab ${activeTab === 'missing' ? 'active' : ''}`}
                      onClick={() => setActiveTab('missing')}
                    >
                      Missing ({missingCount})
                    </button>
                  </div>

                  {/* Categorized keywords */}
                  <div className="categories-grid">
                    {CATEGORIES.map((category) => (
                      <CategorySection
                        key={category}
                        category={category}
                        keywords={
                          activeTab === 'matched'
                            ? result.categorized_matches[category]
                            : result.categorized_missing[category]
                        }
                      />
                    ))}
                  </div>

                  {/* Legend */}
                  <div className="keywords-legend">
                    <span className="legend-item">
                      <span className="legend-critical">!</span> Critical
                    </span>
                    <span className="legend-item">
                      <span className="legend-important"></span> Important
                    </span>
                    <span className="legend-item">
                      <span className="legend-nice"></span> Nice to have
                    </span>
                  </div>

                  {/* Market Trends Insight */}
                  {result.market_trends && result.market_trends.industry_insights && (
                    <div className="market-trends-section">
                      <div className="trends-header">
                        <span className="trends-icon">📊</span>
                        <span className="trends-label">Market Insights</span>
                      </div>
                      <p className="trends-insight">{result.market_trends.industry_insights}</p>
                      {result.market_trends.emerging_technologies.length > 0 && (
                        <div className="emerging-tech">
                          <span className="emerging-label">Emerging:</span>
                          <span className="emerging-list">
                            {result.market_trends.emerging_technologies.slice(0, 5).join(', ')}
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </>
              ) : (
                /* Fallback to legacy flat display */
                <>
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
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
