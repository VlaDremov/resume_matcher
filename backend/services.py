"""
Business Logic Services for the Resume Matcher API.
"""

import re
from pathlib import Path
from typing import Optional

from backend.schemas import (
    AnalyzeResponse,
    CategoryScore,
    ResumeVariantInfo,
    RewrittenBullet,
    SaveVacancyResponse,
    VacancyInfo,
)

# * Project paths
PROJECT_ROOT = Path(__file__).parent.parent
VACANCIES_DIR = PROJECT_ROOT / "vacancies"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESUME_PATH = PROJECT_ROOT / "resume.tex"


# * Display name mapping for variants
VARIANT_DISPLAY_NAMES = {
    "mlops": "MLOps & Platform Engineering",
    "nlp_llm": "NLP & LLM Engineering",
    "cloud_aws": "Cloud & AWS Infrastructure",
    "data_engineering": "Data Engineering & Pipelines",
    "classical_ml": "Classical ML & Analytics",
}

VARIANT_DESCRIPTIONS = {
    "mlops": "Optimized for MLOps, CI/CD, model deployment, and platform engineering roles",
    "nlp_llm": "Optimized for NLP, LLM, chatbots, and language model engineering roles",
    "cloud_aws": "Optimized for cloud infrastructure, AWS, and cloud-native ML roles",
    "data_engineering": "Optimized for data pipelines, ETL, and data engineering roles",
    "classical_ml": "Optimized for traditional ML, analytics, and data science roles",
}


class ResumeAnalysisService:
    """Service for analyzing job descriptions and matching resumes."""

    def __init__(self):
        """Initialize the analysis service."""
        self._llm_client = None
        self._semantic_matcher = None
        self._bullet_rewriter = None

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            from src.llm_client import get_client
            self._llm_client = get_client()
        return self._llm_client

    @property
    def semantic_matcher(self):
        """Lazy-load semantic matcher."""
        if self._semantic_matcher is None:
            from src.semantic_matcher import SemanticMatcher
            self._semantic_matcher = SemanticMatcher(OUTPUT_DIR, client=self.llm_client)
        return self._semantic_matcher

    @property
    def bullet_rewriter(self):
        """Lazy-load bullet rewriter."""
        if self._bullet_rewriter is None:
            from src.bullet_rewriter import BulletRewriter
            self._bullet_rewriter = BulletRewriter(client=self.llm_client)
        return self._bullet_rewriter

    def analyze(
        self,
        job_description: str,
        use_semantic: bool = True,
        rewrite_bullets: bool = True,
    ) -> AnalyzeResponse:
        """
        Analyze a job description and find the best resume match.

        Args:
            job_description: Job description text.
            use_semantic: Whether to use semantic matching.
            rewrite_bullets: Whether to rewrite bullets with GPT-5.

        Returns:
            AnalyzeResponse with analysis results.
        """
        # * Get semantic match scores
        if use_semantic:
            try:
                semantic_result = self.semantic_matcher.match(job_description)
                best_variant = semantic_result["best_variant"]
                similarity_scores = semantic_result["all_scores"]
            except Exception as e:
                print(f"! Semantic matching failed: {e}")
                best_variant = "classical_ml"
                similarity_scores = {}
        else:
            # * Fall back to keyword-based matching
            from src.keyword_engine import find_best_theme_for_job, match_job_to_categories
            best_variant, _ = find_best_theme_for_job(job_description)
            similarity_scores = match_job_to_categories(job_description)

        # * Build category scores
        category_scores = []
        for variant, score in similarity_scores.items():
            category_scores.append(CategoryScore(
                category=variant,
                score=round(score, 3),
                display_name=VARIANT_DISPLAY_NAMES.get(variant, variant.title()),
            ))

        # * Sort by score descending
        category_scores.sort(key=lambda x: -x.score)

        # * Analyze and rewrite bullets if requested
        rewritten_bullets = []
        key_matches = []
        missing_keywords = []
        reasoning = ""
        relevancy_score = 50  # * Default score

        if rewrite_bullets:
            try:
                # * Extract bullets from best variant
                variant_path = OUTPUT_DIR / f"resume_{best_variant}.tex"
                if variant_path.exists():
                    from src.bullet_rewriter import extract_bullets_from_latex

                    with open(variant_path, "r") as f:
                        latex_content = f.read()

                    bullets = extract_bullets_from_latex(latex_content)

                    # * Analyze with GPT-5
                    analysis = self.bullet_rewriter.analyze_and_rewrite(
                        job_description=job_description,
                        resume_bullets=bullets[:10],  # * Limit to first 10 bullets
                        resume_variant=best_variant,
                    )

                    relevancy_score = analysis.relevancy_score
                    key_matches = analysis.key_matches
                    missing_keywords = analysis.missing_keywords
                    reasoning = analysis.reasoning

                    # * Convert rewritten bullets
                    for rb in analysis.rewritten_bullets:
                        rewritten_bullets.append(RewrittenBullet(
                            original=rb.original,
                            rewritten=rb.rewritten,
                            keywords_added=rb.keywords_added,
                            confidence=rb.confidence,
                        ))

            except Exception as e:
                print(f"! Bullet rewriting failed: {e}")
                reasoning = f"Bullet rewriting failed: {e}"

        # * Calculate relevancy score from semantic similarity if not from GPT
        if not rewrite_bullets and similarity_scores:
            max_score = max(similarity_scores.values()) if similarity_scores else 0
            relevancy_score = int(max_score * 100)

        # * Get file paths
        tex_path = OUTPUT_DIR / f"resume_{best_variant}.tex"
        pdf_path = OUTPUT_DIR / f"resume_{best_variant}.pdf"

        return AnalyzeResponse(
            relevancy_score=relevancy_score,
            best_variant=best_variant,
            best_variant_display=VARIANT_DISPLAY_NAMES.get(best_variant, best_variant.title()),
            category_scores=category_scores,
            key_matches=key_matches,
            missing_keywords=missing_keywords,
            rewritten_bullets=rewritten_bullets,
            reasoning=reasoning,
            tex_path=str(tex_path) if tex_path.exists() else None,
            pdf_path=str(pdf_path) if pdf_path.exists() else None,
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
        filepath = VACANCIES_DIR / f"{safe_filename}.txt"

        # * Ensure directory exists
        VACANCIES_DIR.mkdir(parents=True, exist_ok=True)

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

            return SaveVacancyResponse(
                success=True,
                filepath=str(filepath),
                message=f"Vacancy saved to {filepath.name}",
            )
        except Exception as e:
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
                print(f"? Could not read {filepath}: {e}")

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

        for variant_name, display_name in VARIANT_DISPLAY_NAMES.items():
            tex_path = OUTPUT_DIR / f"resume_{variant_name}.tex"
            pdf_path = OUTPUT_DIR / f"resume_{variant_name}.pdf"

            variants.append(ResumeVariantInfo(
                name=variant_name,
                display_name=display_name,
                description=VARIANT_DESCRIPTIONS.get(variant_name, ""),
                tex_exists=tex_path.exists(),
                pdf_exists=pdf_path.exists(),
                tex_path=str(tex_path) if tex_path.exists() else None,
                pdf_path=str(pdf_path) if pdf_path.exists() else None,
            ))

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

