"""
Market Trends Integration Module.

Fetches current job market trends and skill demand data
to enrich keyword analysis with real-time market context.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("resume_matcher.market_trends")


# * Cache file path
CACHE_PATH = Path("output/.market_trends_cache.json")


class TrendingSkill(BaseModel):
    """A skill with market demand metadata."""

    skill: str = Field(description="The skill name")
    category: str = Field(description="Tech category: mlops, nlp_llm, cloud_aws, data_engineering, classical_ml")
    demand_level: str = Field(description="Demand level: high, medium, low")
    trend: str = Field(description="Trend direction: rising, stable, declining")


class MarketTrendsResponse(BaseModel):
    """Response from market trends analysis."""

    trending_skills: list[TrendingSkill] = Field(
        default_factory=list,
        description="Skills currently in high demand",
    )
    emerging_technologies: list[str] = Field(
        default_factory=list,
        description="New technologies gaining traction",
    )
    industry_insights: str = Field(
        default="",
        description="Brief summary of current market conditions",
    )
    last_updated: str = Field(
        default="",
        description="ISO timestamp of when data was fetched",
    )


class MarketTrendsService:
    """
    Service for fetching and analyzing job market trends.

    Uses GPT to synthesize current skill demand data
    and caches results to minimize API calls.
    """

    def __init__(
        self,
        cache_duration_hours: int = 24,
        llm_client=None,
    ):
        """
        Initialize the market trends service.

        Args:
            cache_duration_hours: How long to cache trends data.
            llm_client: Optional pre-configured LLM client.
        """
        self._cache: Optional[MarketTrendsResponse] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(hours=cache_duration_hours)
        self._llm_client = llm_client
        self._load_cache()

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            try:
                from src.llm_client import get_client

                self._llm_client = get_client()
            except Exception as e:
                logger.warning("LLM client not available: %s", e)
        return self._llm_client

    def _load_cache(self):
        """Load cached trends from disk."""
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._cache = MarketTrendsResponse(**data["trends"])
                self._cache_timestamp = datetime.fromisoformat(data["timestamp"])

                logger.info(
                    "Loaded market trends cache from %s",
                    self._cache_timestamp.isoformat(),
                )
            except Exception as e:
                logger.warning("Could not load trends cache: %s", e)
                self._cache = None
                self._cache_timestamp = None

    def _save_cache(self):
        """Save trends cache to disk."""
        if self._cache is None:
            return

        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "trends": self._cache.model_dump(),
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            logger.debug("Saved market trends cache")
        except Exception as e:
            logger.warning("Could not save trends cache: %s", e)

    def _is_cache_valid(self) -> bool:
        """Check if cached trends are still valid."""
        if self._cache is None or self._cache_timestamp is None:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_duration

    async def fetch_trends_async(
        self,
        role_focus: str = "ML Engineer",
        force_refresh: bool = False,
    ) -> MarketTrendsResponse:
        """
        Fetch current market trends for ML/Data roles (async).

        Args:
            role_focus: Primary role to focus analysis on.
            force_refresh: Bypass cache and fetch fresh data.

        Returns:
            MarketTrendsResponse with trending skills and insights.
        """
        if not force_refresh and self._is_cache_valid():
            logger.info("Returning cached market trends")
            return self._cache

        if self.llm_client is None:
            logger.warning("LLM client not available, returning empty trends")
            return MarketTrendsResponse(
                industry_insights="Market data unavailable (no API key)",
                last_updated=datetime.now().isoformat(),
            )

        system_prompt = """You are a job market analyst specializing in ML/AI/Data roles.

Based on your knowledge of current job market trends, provide an analysis of:
1. Most in-demand skills for ML Engineers and related roles
2. Emerging technologies gaining traction
3. Brief market insights

Focus on these categories:
- mlops: MLOps, CI/CD, deployment, containerization
- nlp_llm: NLP, LLMs, RAG, transformers
- cloud_aws: Cloud platforms, AWS, infrastructure
- data_engineering: Data pipelines, ETL, streaming
- classical_ml: Traditional ML, statistics, A/B testing

Provide realistic demand levels and trends."""

        user_prompt = f"""Analyze current job market trends for {role_focus} positions.

Provide:
1. 15-20 trending skills with their category (mlops/nlp_llm/cloud_aws/data_engineering/classical_ml), demand level (high/medium/low), and trend (rising/stable/declining)
2. 5-10 emerging technologies that are gaining demand
3. A brief 2-3 sentence summary of current market conditions

Focus on skills that differentiate candidates in the current job market."""

        try:
            result = await self.llm_client.chat_structured_async(
                prompt=user_prompt,
                response_model=MarketTrendsResponse,
                system_prompt=system_prompt,
            )
            result.last_updated = datetime.now().isoformat()

            # * Update cache
            self._cache = result
            self._cache_timestamp = datetime.now()
            self._save_cache()

            logger.info(
                "Fetched market trends: %d trending skills, %d emerging tech",
                len(result.trending_skills),
                len(result.emerging_technologies),
            )
            return result

        except Exception as e:
            logger.error("Failed to fetch market trends: %s", e)
            # * Return cached data if available, otherwise empty
            if self._cache is not None:
                logger.info("Returning stale cached trends after fetch failure")
                return self._cache
            return MarketTrendsResponse(
                industry_insights="Market data temporarily unavailable",
                last_updated=datetime.now().isoformat(),
            )

    def fetch_trends(
        self,
        role_focus: str = "ML Engineer",
        force_refresh: bool = False,
    ) -> MarketTrendsResponse:
        """Sync wrapper for fetch_trends_async."""
        return asyncio.run(self.fetch_trends_async(role_focus, force_refresh))

    def enrich_keywords_with_trends(
        self,
        keywords: list[dict],
        trends: Optional[MarketTrendsResponse] = None,
    ) -> list[dict]:
        """
        Enrich keyword list with market trend metadata.

        Args:
            keywords: List of keyword dicts with 'keyword' and 'category' fields.
            trends: Optional pre-fetched trends data.

        Returns:
            Same list with added 'is_trending', 'demand_level', and 'trend' fields.
        """
        if trends is None:
            trends = self._cache

        if trends is None:
            return keywords

        # * Build lookup map from trends
        trending_map = {}
        for skill in trends.trending_skills:
            trending_map[skill.skill.lower()] = {
                "is_trending": True,
                "demand_level": skill.demand_level,
                "trend": skill.trend,
            }

        emerging_set = {t.lower() for t in trends.emerging_technologies}

        # * Enrich keywords
        enriched = []
        for kw in keywords:
            kw_text = kw.get("keyword", "").lower()
            entry = kw.copy()

            if kw_text in trending_map:
                entry.update(trending_map[kw_text])
            elif kw_text in emerging_set:
                entry["is_trending"] = True
                entry["demand_level"] = "high"
                entry["trend"] = "rising"
            else:
                entry["is_trending"] = False
                entry["demand_level"] = None
                entry["trend"] = None

            enriched.append(entry)

        return enriched

    def get_cached_trends(self) -> Optional[MarketTrendsResponse]:
        """Get cached trends without fetching."""
        return self._cache

    def clear_cache(self):
        """Clear the trends cache."""
        self._cache = None
        self._cache_timestamp = None
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
        logger.info("Cleared market trends cache")


# * Global service instance
_market_trends_service: Optional[MarketTrendsService] = None


def get_market_trends_service() -> MarketTrendsService:
    """Get shared MarketTrendsService instance."""
    global _market_trends_service
    if _market_trends_service is None:
        _market_trends_service = MarketTrendsService()
    return _market_trends_service
