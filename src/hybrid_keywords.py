"""
Hybrid Keyword Extraction Module.

Combines local TF-IDF/cluster-keyword extraction with GPT-powered
analysis for better keyword matching accuracy. Supports async
operations for parallel execution.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.cluster_artifacts import load_cluster_artifact
from src.keyword_engine import extract_keywords_from_text

logger = logging.getLogger("resume_matcher.hybrid_keywords")

DEFAULT_CLUSTER_ARTIFACT = Path("output/vacancy_clusters.json")


class KeywordAnalysis(BaseModel):
    """GPT-structured response for keyword analysis."""

    key_matches: list[str] = Field(
        default_factory=list,
        description="Keywords from job that match the resume content",
    )
    missing_keywords: list[str] = Field(
        default_factory=list,
        description="Important job keywords not found in resume",
    )


class KeywordWithMeta(BaseModel):
    """A keyword with category and importance metadata."""

    keyword: str
    category: str
    importance: str  # "critical", "important", "nice_to_have"
    is_matched: bool = False


CategorizedKeywordResult = dict[str, list[KeywordWithMeta]]


class HybridKeywordExtractor:
    """
    Hybrid keyword extractor combining local and GPT-based methods.

    Local extraction (fast, free):
    - TF-IDF keyword extraction
    - Cluster keyword matching (from artifact)

    GPT extraction (better understanding):
    - Contextual keyword analysis
    - Semantic matching beyond exact text

    Results are merged and deduplicated.
    Supports async operations for parallel execution.
    """

    def __init__(self, llm_client=None, artifact_path: str | Path | None = None):
        """
        Initialize the hybrid keyword extractor.

        Args:
            llm_client: Optional LLMClient for GPT-based extraction.
            artifact_path: Path to cluster artifact JSON (optional).
        """
        self._llm_client = llm_client
        self._use_gpt = os.getenv("USE_GPT_KEYWORDS", "true").lower() == "true"
        self._artifact_path = Path(artifact_path) if artifact_path else DEFAULT_CLUSTER_ARTIFACT
        self._artifact_mtime_ns: Optional[int] = None
        self._cluster_keywords: dict[str, set[str]] = {}
        self._load_cluster_keywords()

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            try:
                from src.llm_client import get_client
                self._llm_client = get_client()
            except Exception as e:
                logger.warning("LLM client not available: %s", e)
                self._llm_client = None
        return self._llm_client

    def _normalize_keyword(self, keyword: str) -> str:
        return " ".join(keyword.lower().split())

    def _load_cluster_keywords(self) -> None:
        if not self._artifact_path.exists():
            self._cluster_keywords = {}
            return

        try:
            artifact = load_cluster_artifact(self._artifact_path)
        except Exception as exc:
            logger.warning("Failed to load cluster artifact: %s", exc)
            self._cluster_keywords = {}
            return

        try:
            self._artifact_mtime_ns = self._artifact_path.stat().st_mtime_ns
        except OSError:
            self._artifact_mtime_ns = None

        cluster_keywords: dict[str, set[str]] = {}
        for cluster in artifact.clusters:
            keywords = (
                cluster.top_keywords
                + cluster.defining_technologies
                + cluster.defining_skills
            )
            normalized = {
                self._normalize_keyword(kw)
                for kw in keywords
                if kw and self._normalize_keyword(kw)
            }
            if normalized:
                cluster_keywords[cluster.slug] = normalized

        self._cluster_keywords = cluster_keywords

    def _maybe_reload_cluster_keywords(self) -> None:
        if not self._artifact_path.exists():
            return
        try:
            mtime_ns = self._artifact_path.stat().st_mtime_ns
        except OSError:
            return
        if self._artifact_mtime_ns is None or mtime_ns > self._artifact_mtime_ns:
            self._load_cluster_keywords()

    def _keyword_matches_cluster(self, keyword: str, cluster_keywords: set[str]) -> bool:
        kw_norm = self._normalize_keyword(keyword)
        if not kw_norm:
            return False
        kw_compact = kw_norm.replace(" ", "")
        for cluster_kw in cluster_keywords:
            if kw_norm == cluster_kw:
                return True
            cluster_compact = cluster_kw.replace(" ", "")
            if kw_compact == cluster_compact:
                return True
            if kw_norm in cluster_kw or cluster_kw in kw_norm:
                return True
            if kw_compact in cluster_compact or cluster_compact in kw_compact:
                return True
        return False

    def extract_keywords_local(
        self,
        job_text: str,
        resume_text: str,
    ) -> tuple[list[str], list[str]]:
        """
        Extract keywords using local TF-IDF and cluster keyword matching.

        Args:
            job_text: Job description text.
            resume_text: Resume variant text content.

        Returns:
            Tuple of (key_matches, missing_keywords).
        """
        # * Extract keywords from job description
        job_keywords = extract_keywords_from_text(job_text, top_n=50)
        job_kw_set = {kw.lower() for kw, _ in job_keywords}

        # * Add cluster keywords found in job
        self._maybe_reload_cluster_keywords()
        job_lower = job_text.lower()
        for keywords in self._cluster_keywords.values():
            for kw in keywords:
                if kw in job_lower:
                    job_kw_set.add(kw)

        # * Check which keywords appear in resume
        resume_lower = resume_text.lower()
        key_matches = []
        missing_keywords = []

        for kw in job_kw_set:
            if kw in resume_lower:
                key_matches.append(kw)
            else:
                missing_keywords.append(kw)

        # * Sort by relevance (longer/more specific keywords first)
        key_matches.sort(key=lambda x: (-len(x), x))
        missing_keywords.sort(key=lambda x: (-len(x), x))

        return key_matches[:20], missing_keywords[:15]

    def _build_gpt_prompt(
        self,
        job_text: str,
        resume_text: str,
    ) -> tuple[str, str]:
        """Build system prompt and user prompt for GPT extraction."""
        system_prompt = """You are an expert resume analyzer for ML Engineering/Data Science positions.
Given a job description and several versions of a resume, analyze each resume to:
1. Identify key_matches: Important technical skills and keywords from the job that ARE present in each resume.
2. Identify missing_keywords: Important technical skills and keywords from the job that are NOT in each resume.
Focus on:
- Technical skills (Python, TensorFlow, etc.)
- Tools and frameworks (Docker, Kubernetes, etc.)
- Methodologies (MLOps, CI/CD, etc.)
- Domain expertise (NLP, Computer Vision, etc.)
Be concise. Return only the most important 15-20 keywords per category."""

        prompt = f"""Job Description:
{job_text[:4000]}

Resume Content:
{resume_text}

Analyze which job keywords match the resume and which are missing."""

        return system_prompt, prompt

    def extract_keywords_gpt(
        self,
        job_text: str,
        resume_text: str,
    ) -> Optional[tuple[list[str], list[str]]]:
        """
        Extract keywords using GPT structured output (sync).

        Args:
            job_text: Job description text.
            resume_text: Resume variant text content.

        Returns:
            Tuple of (key_matches, missing_keywords), or None if failed.
        """
        if not self._use_gpt or self.llm_client is None:
            logger.debug("GPT extraction skipped: use_gpt=%s, client=%s", self._use_gpt, self.llm_client)
            return None

        system_prompt, prompt = self._build_gpt_prompt(job_text, resume_text)

        try:
            result = self.llm_client.chat_structured(
                prompt=prompt,
                response_model=KeywordAnalysis,
                system_prompt=system_prompt,
                # temperature=0.3,  # * Slightly higher for more diverse keyword extraction
            )
            logger.debug(
                "GPT extraction raw result: matches=%d, missing=%d",
                len(result.key_matches),
                len(result.missing_keywords),
            )
            return result.key_matches, result.missing_keywords
        except Exception as e:
            logger.warning("GPT keyword extraction failed: %s", e)
            return None

    async def extract_keywords_gpt_async(
        self,
        job_text: str,
        resume_text: str,
    ) -> Optional[tuple[list[str], list[str]]]:
        """
        Extract keywords using GPT structured output (async).

        Args:
            job_text: Job description text.
            resume_text: Resume variant text content.

        Returns:
            Tuple of (key_matches, missing_keywords), or None if failed.
        """
        if not self._use_gpt or self.llm_client is None:
            logger.debug("GPT extraction async skipped: use_gpt=%s, client=%s", self._use_gpt, self.llm_client)
            return None

        system_prompt, prompt = self._build_gpt_prompt(job_text, resume_text)

        try:
            result = await self.llm_client.chat_structured_async(
                prompt=prompt,
                response_model=KeywordAnalysis,
                system_prompt=system_prompt,
                # temperature=0.3,  # * Slightly higher for more diverse keyword extraction
            )
            logger.debug(
                "GPT extraction async raw result: matches=%d, missing=%d",
                len(result.key_matches),
                len(result.missing_keywords),
            )
            return result.key_matches, result.missing_keywords
        except Exception as e:
            logger.warning("GPT keyword extraction async failed: %s", e)
            return None

    def extract_keywords_hybrid(
        self,
        job_text: str,
        resume_text: str,
    ) -> tuple[list[str], list[str]]:
        """
        Extract keywords using hybrid local + GPT approach (sync).

        Combines results from both methods:
        - Local extraction for baseline coverage
        - GPT extraction for semantic understanding
        - Merged and deduplicated results

        Args:
            job_text: Job description text.
            resume_text: Resume variant text content.

        Returns:
            Tuple of (key_matches, missing_keywords).
        """
        # * Get local keywords
        local_matches, local_missing = self.extract_keywords_local(job_text, resume_text)
        logger.info(
            "Local extraction: %d matches, %d missing",
            len(local_matches),
            len(local_missing),
        )

        # * Try GPT keywords
        gpt_result = self.extract_keywords_gpt(job_text, resume_text)

        return self._merge_results(local_matches, local_missing, gpt_result)

    async def extract_keywords_hybrid_async(
        self,
        job_text: str,
        resume_text: str,
    ) -> tuple[list[str], list[str]]:
        """
        Extract keywords using hybrid local + GPT approach (async, parallel).

        Runs local and GPT extraction in parallel for faster execution.

        Args:
            job_text: Job description text.
            resume_text: Resume variant text content.

        Returns:
            Tuple of (key_matches, missing_keywords).
        """
        # * Run local extraction in thread pool (CPU-bound) and GPT async in parallel
        local_task = asyncio.to_thread(
            self.extract_keywords_local, job_text, resume_text
        )

        # * GPT task (async)
        gpt_task = self.extract_keywords_gpt_async(job_text, resume_text)

        # * Run both in parallel
        results = await asyncio.gather(
            local_task,
            gpt_task,
            return_exceptions=True,
        )

        local_result, gpt_result = results

        # * Handle local result
        if isinstance(local_result, Exception):
            logger.warning("Local keyword extraction failed: %s", local_result)
            local_matches, local_missing = [], []
        else:
            local_matches, local_missing = local_result

        logger.info(
            "Local extraction: %d matches, %d missing",
            len(local_matches),
            len(local_missing),
        )

        # * Handle GPT result
        if isinstance(gpt_result, Exception):
            logger.warning("GPT keyword extraction failed: %s", gpt_result)
            gpt_result = None

        return self._merge_results(local_matches, local_missing, gpt_result)

    def _merge_results(
        self,
        local_matches: list[str],
        local_missing: list[str],
        gpt_result: Optional[tuple[list[str], list[str]]],
    ) -> tuple[list[str], list[str]]:
        """Merge local and GPT extraction results."""
        if gpt_result is not None:
            gpt_matches, gpt_missing = gpt_result
            logger.info(
                "GPT extraction: %d matches, %d missing",
                len(gpt_matches),
                len(gpt_missing),
            )

            # * Merge results (union, GPT takes precedence for classification)
            key_matches = self._merge_keywords(local_matches, gpt_matches)
            missing_keywords = self._merge_keywords(local_missing, gpt_missing)

            # * Remove any "missing" that GPT identified as "matched"
            gpt_matches_lower = {k.lower() for k in gpt_matches}
            missing_keywords = [
                k for k in missing_keywords
                if k.lower() not in gpt_matches_lower
            ]
        else:
            key_matches = local_matches
            missing_keywords = local_missing
            logger.info("Using local-only extraction (GPT unavailable)")

        return key_matches[:20], missing_keywords[:15]

    def _merge_keywords(
        self,
        local: list[str],
        gpt: list[str],
    ) -> list[str]:
        """
        Merge keyword lists, preserving GPT order but adding local unique items.

        Args:
            local: Keywords from local extraction.
            gpt: Keywords from GPT extraction.

        Returns:
            Merged deduplicated list.
        """
        seen = set()
        result = []

        # * GPT keywords first (higher quality)
        for kw in gpt:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                result.append(kw)

        # * Add local keywords not in GPT
        for kw in local:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                result.append(kw)

        return result

    def _categorize_keywords_by_cluster(self, keywords: list[str]) -> dict[str, list[str]]:
        self._maybe_reload_cluster_keywords()
        if not self._cluster_keywords:
            return {"general": list(dict.fromkeys(keywords))}

        categorized: dict[str, list[str]] = {
            slug: [] for slug in self._cluster_keywords.keys()
        }
        categorized["general"] = []

        for kw in keywords:
            assigned = False
            for slug, cluster_keywords in self._cluster_keywords.items():
                if self._keyword_matches_cluster(kw, cluster_keywords):
                    categorized[slug].append(kw)
                    assigned = True
                    break
            if not assigned:
                categorized["general"].append(kw)

        return categorized

    def _score_importance(self, keyword: str, job_text: str) -> str:
        """
        Score keyword importance based on frequency and position in job description.

        Args:
            keyword: The keyword to score.
            job_text: Full job description text.

        Returns:
            Importance level: "critical", "important", or "nice_to_have".
        """
        text_lower = job_text.lower()
        kw_lower = keyword.lower()

        # * Count occurrences
        count = text_lower.count(kw_lower)

        # * Check position (earlier in text = more important)
        first_pos = text_lower.find(kw_lower)
        is_early = first_pos < len(job_text) // 3 if first_pos >= 0 else False

        # * Check if in requirements/qualifications section
        in_requirements = False
        for line in job_text.split("\n"):
            line_lower = line.lower()
            if ("require" in line_lower or "qualif" in line_lower or "must have" in line_lower):
                if kw_lower in line_lower:
                    in_requirements = True
                    break

        # * Determine importance
        if count >= 3 or in_requirements:
            return "critical"
        elif count >= 2 or is_early:
            return "important"
        else:
            return "nice_to_have"

    def categorize_and_rank_keywords(
        self,
        key_matches: list[str],
        missing_keywords: list[str],
        job_text: str,
    ) -> tuple[CategorizedKeywordResult, CategorizedKeywordResult]:
        """
        Categorize keywords and assign importance levels.

        Uses cluster artifact keyword sets and adds importance scoring.

        Args:
            key_matches: Keywords found in resume.
            missing_keywords: Keywords not found in resume.
            job_text: Original job description for importance scoring.

        Returns:
            Tuple of (categorized_matches, categorized_missing).
        """
        # * Categorize using cluster keywords
        matches_by_category = self._categorize_keywords_by_cluster(key_matches)
        missing_by_category = self._categorize_keywords_by_cluster(missing_keywords)

        def build_categorized_result(
            keywords_by_category: dict[str, list[str]],
            is_matched: bool,
        ) -> CategorizedKeywordResult:
            """Build categorized keyword mapping with importance scoring."""
            result_data: dict[str, list[KeywordWithMeta]] = {}
            importance_order = {"critical": 0, "important": 1, "nice_to_have": 2}

            for category, keywords in keywords_by_category.items():
                keyword_metas = []
                for kw in keywords:
                    importance = self._score_importance(kw, job_text)
                    keyword_metas.append(KeywordWithMeta(
                        keyword=kw,
                        category=category,
                        importance=importance,
                        is_matched=is_matched,
                    ))

                keyword_metas.sort(key=lambda x: importance_order.get(x.importance, 3))
                result_data[category] = keyword_metas

            return result_data

        categorized_matches = build_categorized_result(matches_by_category, is_matched=True)
        categorized_missing = build_categorized_result(missing_by_category, is_matched=False)

        logger.info(
            "Keywords categorized: matches=%d categories, missing=%d categories",
            sum(1 for values in categorized_matches.values() if values),
            sum(1 for values in categorized_missing.values() if values),
        )

        return categorized_matches, categorized_missing


# * Global instance
_hybrid_extractor: Optional[HybridKeywordExtractor] = None


def get_hybrid_keyword_extractor() -> HybridKeywordExtractor:
    """Get a shared HybridKeywordExtractor instance."""
    global _hybrid_extractor
    if _hybrid_extractor is None:
        _hybrid_extractor = HybridKeywordExtractor()
    return _hybrid_extractor


def extract_keywords_hybrid(
    job_text: str,
    resume_text: str,
) -> tuple[list[str], list[str]]:
    """
    Convenience function for hybrid keyword extraction (sync).

    Args:
        job_text: Job description text.
        resume_text: Resume variant text content.

    Returns:
        Tuple of (key_matches, missing_keywords).
    """
    extractor = get_hybrid_keyword_extractor()
    return extractor.extract_keywords_hybrid(job_text, resume_text)


async def extract_keywords_hybrid_async(
    job_text: str,
    resume_text: str,
) -> tuple[list[str], list[str]]:
    """
    Convenience function for hybrid keyword extraction (async).

    Args:
        job_text: Job description text.
        resume_text: Resume variant text content.

    Returns:
        Tuple of (key_matches, missing_keywords).
    """
    extractor = get_hybrid_keyword_extractor()
    return await extractor.extract_keywords_hybrid_async(job_text, resume_text)
