"""
Semantic Matching Module.

Uses hybrid embeddings (local + OpenAI) for semantic similarity matching
between job descriptions and resume variants.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.hybrid_embeddings import HybridEmbeddings, get_hybrid_embeddings

logger = logging.getLogger("resume_matcher.semantic_matcher")


class SemanticMatcher:
    """
    Hybrid embedding-based semantic matcher for resumes and job descriptions.

    Uses a combination of:
    - Local: Sentence Transformers (all-MiniLM-L6-v2)
    - OpenAI: text-embedding-3-small (if available)

    Scores are weighted and combined for better accuracy.
    """

    def __init__(
        self,
        variants_dir: str | Path,
        client=None,
        cache_embeddings: bool = True,
    ):
        """
        Initialize the semantic matcher.

        Args:
            variants_dir: Directory containing resume variant .tex files.
            client: Optional LLM client (for OpenAI embeddings).
            cache_embeddings: Whether to cache variant embeddings.
        """
        self.variants_dir = Path(variants_dir)
        self.cache_embeddings = cache_embeddings

        self.variants: dict[str, str] = {}
        self.variant_embeddings: dict[str, dict] = {}
        self.embeddings_cache_path = self.variants_dir / ".hybrid_embeddings_cache.json"

        # * Initialize hybrid embeddings
        self._hybrid = get_hybrid_embeddings()
        if client is not None:
            self._hybrid._openai_client = client

        self._load_variants()
        self._load_or_compute_embeddings()

    def _load_variants(self) -> None:
        """Load resume variants from disk."""
        if not self.variants_dir.exists():
            logger.warning("Variants directory not found: %s", self.variants_dir)
            return

        for tex_file in self.variants_dir.glob("resume_*.tex"):
            theme_name = tex_file.stem.replace("resume_", "")

            try:
                with open(tex_file, "r", encoding="utf-8") as f:
                    content = f.read()
                # * Extract text content (remove LaTeX commands for better embedding)
                self.variants[theme_name] = self._extract_text_content(content)
            except Exception as e:
                logger.warning("Could not load %s: %s", tex_file, e)

        logger.info("Loaded %d resume variants for semantic matching", len(self.variants))

    def _extract_text_content(self, latex_content: str) -> str:
        """
        Extract readable text from LaTeX content for embedding.

        Args:
            latex_content: Raw LaTeX content.

        Returns:
            Cleaned text suitable for embedding.
        """
        text = latex_content

        # * Remove comments
        text = re.sub(r"%.*$", "", text, flags=re.MULTILINE)

        # * Extract content from resumeItem
        items = re.findall(r"\\resumeItem\{([^}]+)\}", text)

        # * Extract content from textbf
        bolds = re.findall(r"\\textbf\{([^}]+)\}", text)

        # * Extract content from section headers
        sections = re.findall(r"\\section\{([^}]+)\}", text)

        # * Combine all extracted text
        all_text = []
        all_text.extend(sections)
        all_text.extend(bolds)
        all_text.extend(items)

        # * Clean up LaTeX artifacts
        result = " ".join(all_text)
        result = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", result)
        result = re.sub(r"[\\${}]", "", result)
        result = re.sub(r"\s+", " ", result)

        return result.strip()

    def _load_or_compute_embeddings(self) -> None:
        """Load cached embeddings or compute new ones."""
        # * Try to load from cache
        if self.cache_embeddings and self.embeddings_cache_path.exists():
            try:
                with open(self.embeddings_cache_path, "r") as f:
                    cached = json.load(f)

                # * Verify cache matches current variants
                if set(cached.keys()) == set(self.variants.keys()):
                    self.variant_embeddings = cached
                    logger.info("Loaded hybrid embeddings from cache")
                    return
            except Exception as e:
                logger.warning("Could not load cache: %s", e)

        # * Compute embeddings
        self._compute_embeddings()

    def _compute_embeddings(self) -> None:
        """Compute hybrid embeddings for all variants."""
        if not self.variants:
            return

        logger.info("Computing hybrid embeddings for resume variants...")

        for name, content in self.variants.items():
            try:
                embedding = self._hybrid.get_embedding(content)
                self.variant_embeddings[name] = embedding
                has_openai = "yes" if embedding.get("has_openai") else "no"
                logger.info("Computed embedding for %s (openai=%s)", name, has_openai)
            except Exception as e:
                logger.error("Failed to compute embedding for %s: %s", name, e)

        # * Save to cache
        if self.cache_embeddings and self.variant_embeddings:
            try:
                with open(self.embeddings_cache_path, "w") as f:
                    json.dump(self.variant_embeddings, f)
                logger.info("Saved hybrid embeddings cache")
            except Exception as e:
                logger.warning("Could not save cache: %s", e)

    def match(self, job_description: str) -> dict:
        """
        Match a job description to resume variants using hybrid semantic similarity (sync).

        Args:
            job_description: Job description text.

        Returns:
            Dictionary containing:
            - best_variant: Name of the best matching variant
            - similarity_score: Hybrid similarity score (0-1)
            - all_scores: Scores for all variants
        """
        if not self.variant_embeddings:
            return {
                "best_variant": None,
                "similarity_score": 0.0,
                "all_scores": {},
            }

        # * Get hybrid embedding for job description
        try:
            job_embedding = self._hybrid.get_embedding(job_description)
        except Exception as e:
            logger.error("Failed to get job embedding: %s", e)
            return {
                "best_variant": None,
                "similarity_score": 0.0,
                "all_scores": {},
            }

        return self._compute_scores(job_embedding)

    async def match_async(self, job_description: str) -> dict:
        """
        Match a job description to resume variants using hybrid semantic similarity (async).

        Uses async embedding for parallel local + OpenAI execution.

        Args:
            job_description: Job description text.

        Returns:
            Dictionary containing:
            - best_variant: Name of the best matching variant
            - similarity_score: Hybrid similarity score (0-1)
            - all_scores: Scores for all variants
        """
        if not self.variant_embeddings:
            return {
                "best_variant": None,
                "similarity_score": 0.0,
                "all_scores": {},
            }

        # * Get hybrid embedding for job description (async, parallel)
        try:
            job_embedding = await self._hybrid.get_embedding_async(job_description)
        except Exception as e:
            logger.error("Failed to get job embedding async: %s", e)
            return {
                "best_variant": None,
                "similarity_score": 0.0,
                "all_scores": {},
            }

        return self._compute_scores(job_embedding)

    def _compute_scores(self, job_embedding: dict) -> dict:
        """Compute similarity scores for all variants."""
        # * Calculate hybrid similarity with each variant
        all_scores = {}
        for name, variant_embedding in self.variant_embeddings.items():
            similarity = self._hybrid.compute_similarity(job_embedding, variant_embedding)
            all_scores[name] = float(similarity)

        # * Find best match
        best_variant = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_variant]

        return {
            "best_variant": best_variant,
            "similarity_score": best_score,
            "all_scores": all_scores,
        }

    def get_variant_text(self, variant_name: str) -> Optional[str]:
        """
        Get the text content of a variant.

        Args:
            variant_name: Name of the variant.

        Returns:
            Text content of the variant, or None if not found.
        """
        return self.variants.get(variant_name)

    def invalidate_cache(self) -> None:
        """Clear the embeddings cache and recompute."""
        if self.embeddings_cache_path.exists():
            self.embeddings_cache_path.unlink()
            logger.info("Hybrid embeddings cache cleared")

        self.variant_embeddings = {}
        self._compute_embeddings()
