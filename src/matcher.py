"""
Job-to-Resume Matching Module.

Matches job descriptions to the most appropriate resume variant
based on keyword analysis and category scoring.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.keyword_engine import (
    TECH_TAXONOMY,
    extract_keywords_from_text,
    find_best_theme_for_job,
    get_resume_themes,
    match_job_to_categories,
)


class ResumeMatcher:
    """
    Matches job descriptions to resume variants.

    This class provides multiple matching strategies:
    1. Category-based matching (using predefined themes)
    2. TF-IDF cosine similarity
    3. Keyword overlap scoring
    """

    def __init__(self, variants_dir: str | Path):
        """
        Initialize the matcher with resume variants.

        Args:
            variants_dir: Directory containing resume variant .tex files.
        """
        self.variants_dir = Path(variants_dir)
        self.variants: dict[str, str] = {}
        self.variant_keywords: dict[str, list[str]] = {}
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self._load_variants()
        self._build_index()

    def _load_variants(self) -> None:
        """Load resume variants from disk."""
        if not self.variants_dir.exists():
            print(f"! Variants directory not found: {self.variants_dir}")
            return

        for tex_file in self.variants_dir.glob("resume_*.tex"):
            theme_name = tex_file.stem.replace("resume_", "")

            try:
                with open(tex_file, "r", encoding="utf-8") as f:
                    content = f.read()
                self.variants[theme_name] = content
            except Exception as e:
                print(f"? Could not load {tex_file}: {e}")

        print(f"Loaded {len(self.variants)} resume variants")

    def _build_index(self) -> None:
        """Build TF-IDF index for variants."""
        if not self.variants:
            return

        # * Extract keywords for each variant
        for theme_name, content in self.variants.items():
            keywords = extract_keywords_from_text(content, top_n=100)
            self.variant_keywords[theme_name] = [kw for kw, _ in keywords]

        # * Build TF-IDF index
        variant_texts = list(self.variants.values())

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )

        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(variant_texts)
        except ValueError:
            self.tfidf_matrix = None

    def match(self, job_description: str) -> dict:
        """
        Find the best resume variant for a job description.

        Args:
            job_description: Job description text.

        Returns:
            Dictionary containing:
            - best_variant: Name of the best matching variant
            - confidence: Confidence score (0-1)
            - all_scores: Scores for all variants
            - category_analysis: Job's category breakdown
        """
        if not self.variants:
            return {
                "best_variant": None,
                "confidence": 0.0,
                "all_scores": {},
                "category_analysis": {},
            }

        # * 1. Category-based matching
        theme_name, theme_score = find_best_theme_for_job(job_description)
        category_scores = match_job_to_categories(job_description)

        # * 2. TF-IDF similarity
        tfidf_scores = self._calculate_tfidf_similarity(job_description)

        # * 3. Keyword overlap
        keyword_scores = self._calculate_keyword_overlap(job_description)

        # * 4. Combine scores
        all_scores = {}
        themes = get_resume_themes()

        for variant_name in self.variants:
            # * Weight different scoring methods
            category_score = 0.0
            if variant_name in themes:
                primary_cat = themes[variant_name]["primary_category"]
                category_score = category_scores.get(primary_cat, 0.0)

            tfidf_score = tfidf_scores.get(variant_name, 0.0)
            keyword_score = keyword_scores.get(variant_name, 0.0)

            # * Weighted combination
            combined = (category_score * 0.4 + tfidf_score * 0.35 + keyword_score * 0.25)

            all_scores[variant_name] = {
                "combined": combined,
                "category": category_score,
                "tfidf": tfidf_score,
                "keyword": keyword_score,
            }

        # * Find best variant
        best_variant = max(all_scores, key=lambda x: all_scores[x]["combined"])
        confidence = all_scores[best_variant]["combined"]

        # * Normalize confidence to 0-1 range
        max_possible = 1.0  # * Theoretical maximum
        confidence = min(confidence / max_possible, 1.0)

        return {
            "best_variant": best_variant,
            "confidence": confidence,
            "all_scores": all_scores,
            "category_analysis": category_scores,
        }

    def _calculate_tfidf_similarity(self, job_description: str) -> dict[str, float]:
        """
        Calculate TF-IDF cosine similarity between job and variants.

        Args:
            job_description: Job description text.

        Returns:
            Dictionary mapping variant name to similarity score.
        """
        scores = {}

        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return scores

        try:
            # * Transform job description
            job_vector = self.tfidf_vectorizer.transform([job_description])

            # * Calculate similarities
            similarities = cosine_similarity(job_vector, self.tfidf_matrix).flatten()

            # * Map to variant names
            variant_names = list(self.variants.keys())
            for i, name in enumerate(variant_names):
                scores[name] = float(similarities[i])

        except Exception as e:
            print(f"? TF-IDF calculation error: {e}")

        return scores

    def _calculate_keyword_overlap(self, job_description: str) -> dict[str, float]:
        """
        Calculate keyword overlap between job and variants.

        Args:
            job_description: Job description text.

        Returns:
            Dictionary mapping variant name to overlap score.
        """
        scores = {}

        # * Extract job keywords
        job_keywords = extract_keywords_from_text(job_description, top_n=50)
        job_keyword_set = {kw.lower() for kw, _ in job_keywords}

        if not job_keyword_set:
            return scores

        for variant_name, variant_keywords in self.variant_keywords.items():
            variant_keyword_set = {kw.lower() for kw in variant_keywords}

            # * Calculate Jaccard similarity
            intersection = len(job_keyword_set & variant_keyword_set)
            union = len(job_keyword_set | variant_keyword_set)

            if union > 0:
                scores[variant_name] = intersection / union
            else:
                scores[variant_name] = 0.0

        return scores

    def get_variant_path(self, variant_name: str, file_type: str = "tex") -> Optional[Path]:
        """
        Get the file path for a variant.

        Args:
            variant_name: Name of the variant (e.g., "mlops").
            file_type: File type - "tex" or "pdf".

        Returns:
            Path to the file, or None if not found.
        """
        extension = ".tex" if file_type == "tex" else ".pdf"
        file_path = self.variants_dir / f"resume_{variant_name}{extension}"

        if file_path.exists():
            return file_path

        return None

    def explain_match(self, job_description: str, variant_name: str) -> str:
        """
        Generate a human-readable explanation of why a variant matches.

        Args:
            job_description: Job description text.
            variant_name: Name of the variant.

        Returns:
            Explanation string.
        """
        if variant_name not in self.variants:
            return f"Variant '{variant_name}' not found."

        themes = get_resume_themes()
        if variant_name not in themes:
            return f"No theme configuration for '{variant_name}'."

        theme = themes[variant_name]
        category_scores = match_job_to_categories(job_description)

        # * Find matching keywords
        job_lower = job_description.lower()
        matching_skills = []
        for skill in theme["skills_priority"]:
            if skill.lower() in job_lower:
                matching_skills.append(skill)

        matching_keywords = []
        for kw in theme["experience_keywords"]:
            if kw.lower() in job_lower:
                matching_keywords.append(kw)

        # * Build explanation
        lines = [
            f"Resume Variant: {theme['name']}",
            f"Primary Focus: {theme['primary_category'].replace('_', ' ').title()}",
            "",
        ]

        # * Category scores
        lines.append("Category Match Scores:")
        for cat, score in sorted(category_scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            lines.append(f"  {cat.replace('_', ' ').title():20} {bar:20} {score:.2f}")

        lines.append("")

        # * Matching skills
        if matching_skills:
            lines.append(f"Matching Skills ({len(matching_skills)}):")
            lines.append(f"  {', '.join(matching_skills[:10])}")
        else:
            lines.append("No direct skill matches found in job description.")

        lines.append("")

        # * Matching keywords
        if matching_keywords:
            lines.append(f"Matching Keywords ({len(matching_keywords)}):")
            lines.append(f"  {', '.join(matching_keywords[:10])}")

        return "\n".join(lines)


def match_job_to_resume(
    job_description: str,
    variants_dir: str | Path,
) -> dict:
    """
    Convenience function to match a job to the best resume variant.

    Args:
        job_description: Job description text (or path to file).
        variants_dir: Directory containing resume variants.

    Returns:
        Match result dictionary.
    """
    # * Check if job_description is a file path
    job_path = Path(job_description)
    if job_path.exists() and job_path.is_file():
        with open(job_path, "r", encoding="utf-8") as f:
            job_description = f.read()

    matcher = ResumeMatcher(variants_dir)
    return matcher.match(job_description)


def rank_all_variants(
    job_description: str,
    variants_dir: str | Path,
) -> list[dict]:
    """
    Rank all resume variants for a job description.

    Args:
        job_description: Job description text.
        variants_dir: Directory containing resume variants.

    Returns:
        List of variants sorted by match score (best first).
    """
    matcher = ResumeMatcher(variants_dir)
    result = matcher.match(job_description)

    ranked = []
    for variant_name, scores in result["all_scores"].items():
        ranked.append({
            "variant": variant_name,
            "score": scores["combined"],
            "details": scores,
            "tex_path": matcher.get_variant_path(variant_name, "tex"),
            "pdf_path": matcher.get_variant_path(variant_name, "pdf"),
        })

    # * Sort by score descending
    ranked.sort(key=lambda x: -x["score"])

    return ranked

