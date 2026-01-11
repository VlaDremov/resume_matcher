"""
Business Logic Services for the Resume Matcher API.

Supports async operations for parallel execution of LLM calls.
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Optional

from backend.schemas import (
    AnalyzeResponse,
    CategorizedKeywords,
    CategoryScore,
    KeywordWithMetadata,
    MarketTrendsInfo,
    ResumeVariantInfo,
    SaveVacancyResponse,
    TrendingSkillInfo,
    VacancyInfo,
)
from src.cluster_artifacts import load_cluster_artifact

# * Module logger
logger = logging.getLogger("resume_matcher.services")

# * Project paths
PROJECT_ROOT = Path(__file__).parent.parent
VACANCIES_DIR = PROJECT_ROOT / "vacancies"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESUME_PATH = PROJECT_ROOT / "resume.tex"
CLUSTER_ARTIFACT_PATH = OUTPUT_DIR / "vacancy_clusters.json"


def _load_cluster_lookup() -> dict[str, object]:
    try:
        artifact = load_cluster_artifact(CLUSTER_ARTIFACT_PATH)
    except FileNotFoundError:
        logger.warning("Cluster artifact not found at %s", CLUSTER_ARTIFACT_PATH)
        return {}
    except Exception as exc:
        logger.warning("Failed to load cluster artifact: %s", exc)
        return {}
    return {cluster.slug: cluster for cluster in artifact.clusters}


def _convert_to_schema_keywords(internal_result: dict[str, list[object]]) -> CategorizedKeywords:
    """Convert internal categorized keywords to schema-compatible mapping."""
    result_data: dict[str, list[KeywordWithMetadata]] = {}

    for category, internal_list in internal_result.items():
        schema_list = [
            KeywordWithMetadata(
                keyword=kw.keyword,
                category=category,
                importance=kw.importance,
                is_matched=kw.is_matched,
                is_trending=False,  # Will be populated by market trends later
                demand_level=None,
            )
            for kw in internal_list
        ]
        result_data[category] = schema_list

    return result_data


def _score_clusters_by_keywords(
    job_description: str,
    cluster_lookup: dict[str, object],
) -> dict[str, float]:
    if not cluster_lookup:
        return {}

    job_lower = job_description.lower()
    scores: dict[str, float] = {}

    for slug, cluster in cluster_lookup.items():
        keywords = (
            cluster.top_keywords
            + cluster.defining_technologies
            + cluster.defining_skills
        )
        matches = sum(1 for kw in keywords if kw.lower() in job_lower)
        scores[slug] = float(matches)

    total = sum(scores.values())
    if total > 0:
        scores = {slug: score / total for slug, score in scores.items()}
    return scores


class ResumeAnalysisService:
    """Service for analyzing job descriptions and matching resumes."""

    def __init__(self):
        """Initialize the analysis service."""
        self._cluster_matcher = None
        self._keyword_extractor = None
        self._market_trends = None

    @property
    def cluster_matcher(self):
        """Lazy-load cluster matcher with hybrid embeddings."""
        if self._cluster_matcher is None:
            from src.cluster_matcher import ClusterMatcher
            self._cluster_matcher = ClusterMatcher(CLUSTER_ARTIFACT_PATH)
        return self._cluster_matcher

    @property
    def keyword_extractor(self):
        """Lazy-load hybrid keyword extractor."""
        if self._keyword_extractor is None:
            from src.hybrid_keywords import HybridKeywordExtractor
            self._keyword_extractor = HybridKeywordExtractor(artifact_path=CLUSTER_ARTIFACT_PATH)
        return self._keyword_extractor

    @property
    def market_trends(self):
        """Lazy-load market trends service."""
        if self._market_trends is None:
            from src.market_trends import get_market_trends_service
            self._market_trends = get_market_trends_service()
        return self._market_trends

    def analyze(
        self,
        job_description: str,
        use_semantic: bool = True,
    ) -> AnalyzeResponse:
        """
        Analyze a job description and find the best resume match.

        Uses hybrid approach:
        - Semantic matching: Local + OpenAI embeddings (weighted)
        - Keyword extraction: TF-IDF + GPT (merged)

        Args:
            job_description: Job description text.
            use_semantic: Whether to use semantic matching.
        Returns:
            AnalyzeResponse with analysis results.
        """
        start = time.perf_counter()
        logger.info(
            "Analyze start use_semantic=%s job_chars=%s",
            use_semantic,
            len(job_description),
        )

        # * Get hybrid semantic match scores
        best_variant = None
        similarity_scores = {}
        cluster_lookup = _load_cluster_lookup()

        if use_semantic:
            try:
                semantic_start = time.perf_counter()
                semantic_result = self.cluster_matcher.match(job_description)
                best_variant = semantic_result["best_cluster"]
                similarity_scores = semantic_result["scores"]
                semantic_duration = time.perf_counter() - semantic_start
                logger.info(
                    "Cluster semantic match complete variant=%s duration=%.3fs categories=%s",
                    best_variant,
                    semantic_duration,
                    len(similarity_scores),
                )
            except Exception as e:
                logger.error("Cluster matching failed: %s", e, exc_info=True)
                similarity_scores = _score_clusters_by_keywords(job_description, cluster_lookup)
        else:
            similarity_scores = _score_clusters_by_keywords(job_description, cluster_lookup)
            logger.info(
                "Keyword-based cluster matching categories=%s",
                len(similarity_scores),
            )

        if not similarity_scores and cluster_lookup:
            similarity_scores = {slug: 0.0 for slug in cluster_lookup}

        if similarity_scores:
            best_variant = max(similarity_scores, key=similarity_scores.get)

        # * Build category scores
        category_scores = []
        for variant, score in similarity_scores.items():
            cluster = cluster_lookup.get(variant)
            display_name = cluster.name if cluster else variant.replace("_", " ").title()
            category_scores.append(CategoryScore(
                category=variant,
                score=round(score, 3),
                display_name=display_name,
            ))

        # * Sort by score descending
        category_scores.sort(key=lambda x: -x.score)

        # * Extract keywords using hybrid approach
        key_matches = []
        missing_keywords = []
        categorized_matches = {}
        categorized_missing = {}

        try:
            keywords_start = time.perf_counter()
            # * Get resume text for the best variant
            resume_text = ""
            if best_variant:
                variant_path = OUTPUT_DIR / f"resume_{best_variant}.tex"
                if variant_path.exists():
                    with open(variant_path, "r", encoding="utf-8") as f:
                        resume_text = f.read()

            key_matches, missing_keywords = self.keyword_extractor.extract_keywords_hybrid(
                job_text=job_description,
                resume_text=resume_text,
            )

            # * Categorize and rank keywords
            cat_matches, cat_missing = self.keyword_extractor.categorize_and_rank_keywords(
                key_matches=key_matches,
                missing_keywords=missing_keywords,
                job_text=job_description,
            )
            categorized_matches = _convert_to_schema_keywords(cat_matches)
            categorized_missing = _convert_to_schema_keywords(cat_missing)

            keywords_duration = time.perf_counter() - keywords_start
            logger.info(
                "Hybrid keyword extraction complete matches=%s missing=%s duration=%.3fs",
                len(key_matches),
                len(missing_keywords),
                keywords_duration,
            )
        except Exception as e:
            logger.error("Keyword extraction failed: %s", e, exc_info=True)

        total_duration = time.perf_counter() - start
        logger.info(
            "Analyze complete variant=%s duration=%.3fs",
            best_variant,
            total_duration,
        )

        best_variant_display = best_variant or ""
        if best_variant and best_variant in cluster_lookup:
            best_variant_display = cluster_lookup[best_variant].name
        elif best_variant:
            best_variant_display = best_variant.replace("_", " ").title()

        return AnalyzeResponse(
            best_variant=best_variant or "",
            best_variant_display=best_variant_display,
            category_scores=category_scores,
            categorized_matches=categorized_matches,
            categorized_missing=categorized_missing,
            key_matches=key_matches,
            missing_keywords=missing_keywords,
        )

    async def analyze_async(
        self,
        job_description: str,
        use_semantic: bool = True,
        include_market_trends: bool = False,
    ) -> AnalyzeResponse:
        """
        Analyze a job description and find the best resume match (async).

        Runs semantic matching and keyword extraction in parallel for faster execution.

        Args:
            job_description: Job description text.
            use_semantic: Whether to use semantic matching.

        Returns:
            AnalyzeResponse with analysis results.
        """
        start = time.perf_counter()
        logger.info(
            "Analyze async start use_semantic=%s job_chars=%s",
            use_semantic,
            len(job_description),
        )

        # * Get hybrid semantic match scores
        best_variant = None
        similarity_scores = {}
        cluster_lookup = _load_cluster_lookup()

        if use_semantic:
            try:
                semantic_start = time.perf_counter()
                semantic_result = await self.cluster_matcher.match_async(job_description)
                best_variant = semantic_result["best_cluster"]
                similarity_scores = semantic_result["scores"]
                semantic_duration = time.perf_counter() - semantic_start
                logger.info(
                    "Cluster semantic match async complete variant=%s duration=%.3fs categories=%s",
                    best_variant,
                    semantic_duration,
                    len(similarity_scores),
                )
            except Exception as e:
                logger.error("Cluster matching async failed: %s", e, exc_info=True)
                similarity_scores = await asyncio.to_thread(
                    _score_clusters_by_keywords, job_description, cluster_lookup
                )
        else:
            similarity_scores = await asyncio.to_thread(
                _score_clusters_by_keywords, job_description, cluster_lookup
            )
            logger.info(
                "Keyword-based cluster matching categories=%s",
                len(similarity_scores),
            )

        if not similarity_scores and cluster_lookup:
            similarity_scores = {slug: 0.0 for slug in cluster_lookup}

        if similarity_scores:
            best_variant = max(similarity_scores, key=similarity_scores.get)

        # * Build category scores
        category_scores = []
        for variant, score in similarity_scores.items():
            cluster = cluster_lookup.get(variant)
            display_name = cluster.name if cluster else variant.replace("_", " ").title()
            category_scores.append(CategoryScore(
                category=variant,
                score=round(score, 3),
                display_name=display_name,
            ))

        # * Sort by score descending
        category_scores.sort(key=lambda x: -x.score)

        # * Extract keywords using hybrid approach (async)
        key_matches = []
        missing_keywords = []
        categorized_matches = {}
        categorized_missing = {}

        try:
            keywords_start = time.perf_counter()
            # * Get resume text for the best variant
            resume_text = ""
            if best_variant:
                variant_path = OUTPUT_DIR / f"resume_{best_variant}.tex"
                if variant_path.exists():
                    with open(variant_path, "r", encoding="utf-8") as f:
                        resume_text = f.read()

            key_matches, missing_keywords = await self.keyword_extractor.extract_keywords_hybrid_async(
                job_text=job_description,
                resume_text=resume_text,
            )

            # * Categorize and rank keywords (run in thread as it's CPU-bound)
            cat_matches, cat_missing = await asyncio.to_thread(
                self.keyword_extractor.categorize_and_rank_keywords,
                key_matches,
                missing_keywords,
                job_description,
            )
            categorized_matches = _convert_to_schema_keywords(cat_matches)
            categorized_missing = _convert_to_schema_keywords(cat_missing)

            keywords_duration = time.perf_counter() - keywords_start
            logger.info(
                "Hybrid keyword extraction async complete matches=%s missing=%s duration=%.3fs",
                len(key_matches),
                len(missing_keywords),
                keywords_duration,
            )
        except Exception as e:
            logger.error("Keyword extraction async failed: %s", e, exc_info=True)

        # * Optionally fetch market trends
        market_trends_info = None
        if include_market_trends:
            try:
                trends_start = time.perf_counter()
                trends = await self.market_trends.fetch_trends_async()
                trends_duration = time.perf_counter() - trends_start

                market_trends_info = MarketTrendsInfo(
                    trending_skills=[
                        TrendingSkillInfo(
                            skill=s.skill,
                            category=s.category,
                            demand_level=s.demand_level,
                            trend=s.trend,
                        )
                        for s in trends.trending_skills
                    ],
                    emerging_technologies=trends.emerging_technologies,
                    industry_insights=trends.industry_insights,
                    last_updated=trends.last_updated,
                )

                logger.info(
                    "Market trends fetched duration=%.3fs skills=%d",
                    trends_duration,
                    len(trends.trending_skills),
                )
            except Exception as e:
                logger.warning("Market trends fetch failed: %s", e)

        total_duration = time.perf_counter() - start
        logger.info(
            "Analyze async complete variant=%s duration=%.3fs",
            best_variant,
            total_duration,
        )

        best_variant_display = best_variant or ""
        if best_variant and best_variant in cluster_lookup:
            best_variant_display = cluster_lookup[best_variant].name
        elif best_variant:
            best_variant_display = best_variant.replace("_", " ").title()

        return AnalyzeResponse(
            best_variant=best_variant or "",
            best_variant_display=best_variant_display,
            category_scores=category_scores,
            categorized_matches=categorized_matches,
            categorized_missing=categorized_missing,
            market_trends=market_trends_info,
            key_matches=key_matches,
            missing_keywords=missing_keywords,
        )


class VacancyService:
    """Service for managing vacancy files."""

    def save_vacancy(
        self,
        job_description: str,
        filename: str,
        company: Optional[str] = None,
        position: Optional[str] = None,
    ) -> SaveVacancyResponse:
        """
        Save a job description to the vacancies folder.

        Args:
            job_description: Job description text.
            filename: Filename without extension.
            company: Optional company name.
            position: Optional position title.

        Returns:
            SaveVacancyResponse with result.
        """
        # * Sanitize filename
        safe_filename = re.sub(r"[^\w\-]", "_", filename.lower())

        # * Ensure directory exists
        VACANCIES_DIR.mkdir(parents=True, exist_ok=True)

        # * If file exists, append _1, _2, etc. to make unique
        filepath = VACANCIES_DIR / f"{safe_filename}.txt"
        if filepath.exists():
            counter = 1
            while True:
                new_filename = f"{safe_filename}_{counter}"
                filepath = VACANCIES_DIR / f"{new_filename}.txt"
                if not filepath.exists():
                    safe_filename = new_filename
                    break
                counter += 1

        # * Build content with metadata header
        content_lines = []
        if company or position:
            content_lines.append(f"# Company: {company or 'Unknown'}")
            content_lines.append(f"# Position: {position or 'Unknown'}")
            content_lines.append("")

        content_lines.append(job_description)
        content = "\n".join(content_lines)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(
                "Vacancy saved filename=%s path=%s length=%s",
                safe_filename,
                filepath,
                len(content),
            )
            return SaveVacancyResponse(
                success=True,
                filepath=str(filepath),
                message=f"Vacancy saved to {filepath.name}",
            )
        except Exception as e:
            logger.error("Failed to save vacancy filename=%s error=%s", safe_filename, e, exc_info=True)
            return SaveVacancyResponse(
                success=False,
                filepath="",
                message=f"Failed to save: {e}",
            )

    def list_vacancies(self) -> list[VacancyInfo]:
        """
        List all saved vacancies.

        Returns:
            List of VacancyInfo objects.
        """
        vacancies = []

        if not VACANCIES_DIR.exists():
            return vacancies

        for filepath in sorted(VACANCIES_DIR.glob("*.txt")):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # * Extract metadata from header
                company = None
                position = None
                preview_start = 0

                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("# Company:"):
                        company = line.replace("# Company:", "").strip()
                        preview_start = i + 1
                    elif line.startswith("# Position:"):
                        position = line.replace("# Position:", "").strip()
                        preview_start = i + 1
                    elif line.strip() and not line.startswith("#"):
                        break

                # * Get preview (skip metadata)
                preview_content = "\n".join(lines[preview_start:])
                preview = preview_content[:200].strip()
                if len(preview_content) > 200:
                    preview += "..."

                vacancies.append(VacancyInfo(
                    filename=filepath.stem,
                    filepath=str(filepath),
                    company=company,
                    position=position,
                    preview=preview,
                ))

            except Exception as e:
                logger.warning("Could not read vacancy file=%s error=%s", filepath, e, exc_info=True)

        logger.info("Vacancies listed count=%s", len(vacancies))
        return vacancies


class ResumeVariantService:
    """Service for managing resume variants."""

    def list_variants(self) -> list[ResumeVariantInfo]:
        """
        List all available resume variants.

        Returns:
            List of ResumeVariantInfo objects.
        """
        variants = []
        cluster_lookup = _load_cluster_lookup()

        for variant_name, cluster in cluster_lookup.items():
            tex_path = OUTPUT_DIR / f"resume_{variant_name}.tex"
            pdf_path = OUTPUT_DIR / f"resume_{variant_name}.pdf"

            variants.append(ResumeVariantInfo(
                name=variant_name,
                display_name=cluster.name,
                description=cluster.summary or "",
                tex_exists=tex_path.exists(),
                pdf_exists=pdf_path.exists(),
                tex_path=str(tex_path) if tex_path.exists() else None,
                pdf_path=str(pdf_path) if pdf_path.exists() else None,
            ))

        logger.info("Variants listed count=%s", len(variants))
        return variants

    def get_variant_file(self, variant_name: str, file_type: str = "pdf") -> Optional[Path]:
        """
        Get the path to a variant file.

        Args:
            variant_name: Name of the variant.
            file_type: "pdf" or "tex".

        Returns:
            Path to the file, or None if not found.
        """
        ext = ".pdf" if file_type == "pdf" else ".tex"
        filepath = OUTPUT_DIR / f"resume_{variant_name}{ext}"

        if filepath.exists():
            return filepath
        return None


# * Global service instances
analysis_service = ResumeAnalysisService()
vacancy_service = VacancyService()
variant_service = ResumeVariantService()
