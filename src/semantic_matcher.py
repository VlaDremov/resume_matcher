"""
Semantic Matching Module.

Uses OpenAI embeddings for semantic similarity matching between
job descriptions and resume variants.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.llm_client import LLMClient, get_client


class SemanticMatcher:
    """
    Embedding-based semantic matcher for resumes and job descriptions.

    Uses OpenAI text-embedding-3-small for computing semantic similarity
    between job descriptions and resume content.
    """

    def __init__(
        self,
        variants_dir: str | Path,
        client: Optional[LLMClient] = None,
        cache_embeddings: bool = True,
    ):
        """
        Initialize the semantic matcher.

        Args:
            variants_dir: Directory containing resume variant .tex files.
            client: Optional LLM client. Creates one if not provided.
            cache_embeddings: Whether to cache variant embeddings.
        """
        self.variants_dir = Path(variants_dir)
        self.client = client or get_client()
        self.cache_embeddings = cache_embeddings

        self.variants: dict[str, str] = {}
        self.variant_embeddings: dict[str, list[float]] = {}
        self.embeddings_cache_path = self.variants_dir / ".embeddings_cache.json"

        self._load_variants()
        self._load_or_compute_embeddings()

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
                # * Extract text content (remove LaTeX commands for better embedding)
                self.variants[theme_name] = self._extract_text_content(content)
            except Exception as e:
                print(f"? Could not load {tex_file}: {e}")

        print(f"Loaded {len(self.variants)} resume variants for semantic matching")

    def _extract_text_content(self, latex_content: str) -> str:
        """
        Extract readable text from LaTeX content for embedding.

        Args:
            latex_content: Raw LaTeX content.

        Returns:
            Cleaned text suitable for embedding.
        """
        import re

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
                    print("Loaded embeddings from cache")
                    return
            except Exception as e:
                print(f"? Could not load cache: {e}")

        # * Compute embeddings
        self._compute_embeddings()

    def _compute_embeddings(self) -> None:
        """Compute embeddings for all variants."""
        if not self.variants:
            return

        print("Computing embeddings for resume variants...")

        for name, content in self.variants.items():
            try:
                embedding = self.client.get_embedding(content)
                self.variant_embeddings[name] = embedding
                print(f"  Computed embedding for {name}")
            except Exception as e:
                print(f"! Failed to compute embedding for {name}: {e}")

        # * Save to cache
        if self.cache_embeddings and self.variant_embeddings:
            try:
                with open(self.embeddings_cache_path, "w") as f:
                    json.dump(self.variant_embeddings, f)
                print(f"Saved embeddings cache to {self.embeddings_cache_path}")
            except Exception as e:
                print(f"? Could not save cache: {e}")

    def match(self, job_description: str) -> dict:
        """
        Match a job description to resume variants using semantic similarity.

        Args:
            job_description: Job description text.

        Returns:
            Dictionary containing:
            - best_variant: Name of the best matching variant
            - similarity_score: Cosine similarity score (0-1)
            - all_scores: Scores for all variants
        """
        if not self.variant_embeddings:
            return {
                "best_variant": None,
                "similarity_score": 0.0,
                "all_scores": {},
            }

        # * Get embedding for job description
        try:
            job_embedding = self.client.get_embedding(job_description)
        except Exception as e:
            print(f"! Failed to get job embedding: {e}")
            return {
                "best_variant": None,
                "similarity_score": 0.0,
                "all_scores": {},
            }

        # * Calculate cosine similarity with each variant
        all_scores = {}
        job_vec = np.array(job_embedding)

        for name, variant_embedding in self.variant_embeddings.items():
            variant_vec = np.array(variant_embedding)
            similarity = self._cosine_similarity(job_vec, variant_vec)
            all_scores[name] = float(similarity)

        # * Find best match
        best_variant = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_variant]

        return {
            "best_variant": best_variant,
            "similarity_score": best_score,
            "all_scores": all_scores,
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score (0-1 for normalized vectors).
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def rank_variants(self, job_description: str) -> list[dict]:
        """
        Rank all variants by semantic similarity to job description.

        Args:
            job_description: Job description text.

        Returns:
            List of variants sorted by similarity (highest first).
        """
        result = self.match(job_description)

        ranked = []
        for name, score in result["all_scores"].items():
            tex_path = self.variants_dir / f"resume_{name}.tex"
            pdf_path = self.variants_dir / f"resume_{name}.pdf"

            ranked.append({
                "variant": name,
                "similarity_score": score,
                "tex_path": str(tex_path) if tex_path.exists() else None,
                "pdf_path": str(pdf_path) if pdf_path.exists() else None,
            })

        # * Sort by score descending
        ranked.sort(key=lambda x: -x["similarity_score"])

        return ranked

    def get_similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two arbitrary texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity score.
        """
        try:
            embeddings = self.client.get_embeddings_batch([text1, text2])
            if len(embeddings) < 2:
                return 0.0

            vec1 = np.array(embeddings[0])
            vec2 = np.array(embeddings[1])

            return float(self._cosine_similarity(vec1, vec2))
        except Exception as e:
            print(f"! Failed to compute similarity: {e}")
            return 0.0

    def invalidate_cache(self) -> None:
        """Clear the embeddings cache and recompute."""
        if self.embeddings_cache_path.exists():
            self.embeddings_cache_path.unlink()
            print("Embeddings cache cleared")

        self.variant_embeddings = {}
        self._compute_embeddings()


def semantic_match(
    job_description: str,
    variants_dir: str | Path,
) -> dict:
    """
    Convenience function for semantic matching.

    Args:
        job_description: Job description text.
        variants_dir: Directory containing resume variants.

    Returns:
        Match result dictionary.
    """
    matcher = SemanticMatcher(variants_dir)
    return matcher.match(job_description)

