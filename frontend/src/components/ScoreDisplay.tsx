import { useState } from 'react';
import {
  AnalyzeResponse,
  CategorizedKeywords,
  KeywordWithMetadata,
} from '../api';
import './ScoreDisplay.css';

interface ScoreDisplayProps {
  result: AnalyzeResponse;
}

const CATEGORY_COLORS = [
  '#2563EB',
  '#16A34A',
  '#F97316',
  '#DC2626',
  '#0EA5E9',
  '#7C3AED',
  '#A16207',
];

// * Count total keywords across all categories
function countKeywords(categorized: CategorizedKeywords): number {
  return Object.values(categorized).reduce((sum, list) => sum + list.length, 0);
}

function formatCategoryLabel(category: string, displayName?: string): string {
  if (displayName) return displayName;
  if (category.toLowerCase() === 'general') return 'General';
  return category
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
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
  label,
  color,
  keywords,
}: {
  category: string;
  label: string;
  color: string;
  keywords: KeywordWithMetadata[];
}) {
  if (keywords.length === 0) return null;

  return (
    <div className="category-section">
      <div className="category-header" style={{ '--category-color': color } as React.CSSProperties}>
        <span className="category-dot" />
        <span className="category-label">{label}</span>
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

  const scoreOrder = result.category_scores.map((score) => score.category);
  const categoryLabelMap = new Map(
    result.category_scores.map((score) => [score.category, score.display_name])
  );
  const keywordCategories = new Set([
    ...Object.keys(result.categorized_matches || {}),
    ...Object.keys(result.categorized_missing || {}),
  ]);
  const orderedCategories = [
    ...scoreOrder,
    ...Array.from(keywordCategories).filter((category) => !scoreOrder.includes(category)),
  ];
  const categoryMeta = orderedCategories.map((category, index) => ({
    key: category,
    label: formatCategoryLabel(category, categoryLabelMap.get(category)),
    color: CATEGORY_COLORS[index % CATEGORY_COLORS.length],
  }));

  // * Find the best variant's score from category_scores
  const bestScore = result.category_scores.find(
    (cat) => cat.category === result.best_variant
  );
  const bestScorePercent = bestScore ? Math.round(bestScore.score * 100) : 0;

  // * Count keywords using new categorized data if available, else fallback to legacy
  const hasCategorizedData = !!result.categorized_matches && !!result.categorized_missing;

  const matchedCount = hasCategorizedData
    ? countKeywords(result.categorized_matches || {})
    : result.key_matches.length;

  const missingCount = hasCategorizedData
    ? countKeywords(result.categorized_missing || {})
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
                    {categoryMeta.map((category) => (
                      <CategorySection
                        key={category.key}
                        category={category.key}
                        label={category.label}
                        color={category.color}
                        keywords={
                          activeTab === 'matched'
                            ? result.categorized_matches[category.key] || []
                            : result.categorized_missing[category.key] || []
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
