"""
Hybrid Embeddings Module.

Combines local Sentence Transformers embeddings with OpenAI embeddings
for improved semantic matching accuracy. Supports async operations
for parallel execution.
"""

import asyncio
import logging
import os
from typing import Optional

import numpy as np

from src.local_embeddings import LocalEmbeddings, get_local_embeddings

logger = logging.getLogger("resume_matcher.hybrid_embeddings")

# * Configuration via environment variables
DEFAULT_LOCAL_WEIGHT = 0.4
DEFAULT_OPENAI_WEIGHT = 0.6


class HybridEmbeddings:
    """
    Hybrid embedding generator combining local and OpenAI embeddings.

    Uses weighted combination of:
    - Local: Sentence Transformers (fast, free, on-device)
    - OpenAI: text-embedding-3-small (higher quality, requires API)

    Falls back to local-only if OpenAI is unavailable.
    Supports async operations for parallel execution.
    """

    def __init__(
        self,
        local_weight: Optional[float] = None,
        openai_weight: Optional[float] = None,
        openai_client=None,
    ):
        """
        Initialize the hybrid embeddings generator.

        Args:
            local_weight: Weight for local embeddings (0-1).
            openai_weight: Weight for OpenAI embeddings (0-1).
            openai_client: Optional LLMClient instance for OpenAI.
        """
        self.local_weight = local_weight or float(
            os.getenv("LOCAL_EMBEDDING_WEIGHT", DEFAULT_LOCAL_WEIGHT)
        )
        self.openai_weight = openai_weight or float(
            os.getenv("OPENAI_EMBEDDING_WEIGHT", DEFAULT_OPENAI_WEIGHT)
        )

        # * Normalize weights
        total = self.local_weight + self.openai_weight
        if total > 0:
            self.local_weight /= total
            self.openai_weight /= total

        self._local = get_local_embeddings()
        self._openai_client = openai_client

    @property
    def openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            try:
                from src.llm_client import get_client
                self._openai_client = get_client()
            except Exception as e:
                logger.warning("OpenAI client not available: %s", e)
                self._openai_client = None
        return self._openai_client

    def get_embedding(self, text: str) -> dict:
        """
        Get hybrid embedding for text (sync).

        Args:
            text: Text to embed.

        Returns:
            Dictionary containing:
            - local: Local embedding vector
            - openai: OpenAI embedding vector (or None)
            - has_openai: Whether OpenAI embedding was obtained
        """
        result = {
            "local": [],
            "openai": None,
            "has_openai": False,
        }

        # * Get local embedding
        result["local"] = self._local.get_embedding(text)

        # * Try OpenAI embedding
        if self.openai_client is not None:
            try:
                result["openai"] = self.openai_client.get_embedding(text)
                result["has_openai"] = True
            except Exception as e:
                logger.warning("OpenAI embedding failed, using local only: %s", e)

        return result

    async def get_embedding_async(self, text: str) -> dict:
        """
        Get hybrid embedding for text (async, parallel execution).

        Runs local and OpenAI embeddings in parallel for faster execution.

        Args:
            text: Text to embed.

        Returns:
            Dictionary containing:
            - local: Local embedding vector
            - openai: OpenAI embedding vector (or None)
            - has_openai: Whether OpenAI embedding was obtained
        """
        result = {
            "local": [],
            "openai": None,
            "has_openai": False,
        }

        # * Run local embedding in thread pool (CPU-bound)
        local_task = asyncio.to_thread(self._local.get_embedding, text)

        # * Run OpenAI embedding async (if client available)
        openai_task = None
        if self.openai_client is not None:
            openai_task = self.openai_client.get_embedding_async(text)

        # * Wait for both to complete in parallel
        if openai_task is not None:
            results = await asyncio.gather(
                local_task,
                openai_task,
                return_exceptions=True,
            )
            local_result, openai_result = results

            # * Process local result
            if isinstance(local_result, Exception):
                logger.warning("Local embedding failed: %s", local_result)
            else:
                result["local"] = local_result

            # * Process OpenAI result
            if isinstance(openai_result, Exception):
                logger.warning("OpenAI embedding failed, using local only: %s", openai_result)
            else:
                result["openai"] = openai_result
                result["has_openai"] = True
        else:
            # * Only local embedding
            local_result = await local_task
            if isinstance(local_result, Exception):
                logger.warning("Local embedding failed: %s", local_result)
            else:
                result["local"] = local_result

        return result

    def compute_similarity(
        self,
        job_embedding: dict,
        variant_embedding: dict,
    ) -> float:
        """
        Compute hybrid similarity score between job and variant embeddings.

        Args:
            job_embedding: Hybrid embedding for job description.
            variant_embedding: Hybrid embedding for resume variant.

        Returns:
            Weighted similarity score (0-1).
        """
        scores = []
        weights = []

        # * Local similarity
        if job_embedding["local"] and variant_embedding["local"]:
            local_sim = self._cosine_similarity(
                job_embedding["local"],
                variant_embedding["local"],
            )
            scores.append(local_sim)
            weights.append(self.local_weight)

        # * OpenAI similarity
        if (
            job_embedding["has_openai"]
            and variant_embedding.get("openai")
            and job_embedding["openai"]
        ):
            openai_sim = self._cosine_similarity(
                job_embedding["openai"],
                variant_embedding["openai"],
            )
            scores.append(openai_sim)
            weights.append(self.openai_weight)

        if not scores:
            return 0.0

        # * Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return weighted_score

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


class HybridSemanticMatcher:
    """
    Semantic matcher using hybrid embeddings.

    Matches job descriptions to resume variants using a combination
    of local and OpenAI embeddings for better accuracy.
    """

    def __init__(
        self,
        variants: dict[str, str],
        local_weight: Optional[float] = None,
        openai_weight: Optional[float] = None,
    ):
        """
        Initialize the hybrid semantic matcher.

        Args:
            variants: Dictionary mapping variant name to text content.
            local_weight: Weight for local embeddings.
            openai_weight: Weight for OpenAI embeddings.
        """
        self.variants = variants
        self.embeddings = HybridEmbeddings(local_weight, openai_weight)
        self.variant_embeddings: dict[str, dict] = {}

        self._compute_variant_embeddings()

    def _compute_variant_embeddings(self) -> None:
        """Compute hybrid embeddings for all variants."""
        logger.info("Computing hybrid embeddings for %d variants", len(self.variants))

        for name, content in self.variants.items():
            self.variant_embeddings[name] = self.embeddings.get_embedding(content)
            logger.debug("Computed embedding for variant: %s", name)

        logger.info("Hybrid variant embeddings computed")

    def match(self, job_description: str) -> dict:
        """
        Match a job description to resume variants (sync).

        Args:
            job_description: Job description text.

        Returns:
            Dictionary containing:
            - best_variant: Name of the best matching variant
            - best_score: Similarity score for best variant
            - all_scores: Scores for all variants
        """
        if not self.variant_embeddings:
            return {
                "best_variant": None,
                "best_score": 0.0,
                "all_scores": {},
            }

        # * Get job embedding
        job_embedding = self.embeddings.get_embedding(job_description)

        # * Calculate similarity with each variant
        all_scores = {}
        for name, variant_emb in self.variant_embeddings.items():
            score = self.embeddings.compute_similarity(job_embedding, variant_emb)
            all_scores[name] = score

        # * Find best match
        best_variant = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_variant]

        return {
            "best_variant": best_variant,
            "best_score": best_score,
            "all_scores": all_scores,
        }

    async def match_async(self, job_description: str) -> dict:
        """
        Match a job description to resume variants (async).

        Args:
            job_description: Job description text.

        Returns:
            Dictionary containing:
            - best_variant: Name of the best matching variant
            - best_score: Similarity score for best variant
            - all_scores: Scores for all variants
        """
        if not self.variant_embeddings:
            return {
                "best_variant": None,
                "best_score": 0.0,
                "all_scores": {},
            }

        # * Get job embedding (async, parallel local+OpenAI)
        job_embedding = await self.embeddings.get_embedding_async(job_description)

        # * Calculate similarity with each variant
        all_scores = {}
        for name, variant_emb in self.variant_embeddings.items():
            score = self.embeddings.compute_similarity(job_embedding, variant_emb)
            all_scores[name] = score

        # * Find best match
        best_variant = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_variant]

        return {
            "best_variant": best_variant,
            "best_score": best_score,
            "all_scores": all_scores,
        }


# * Global instance
_hybrid_embeddings: Optional[HybridEmbeddings] = None


def get_hybrid_embeddings() -> HybridEmbeddings:
    """Get a shared HybridEmbeddings instance."""
    global _hybrid_embeddings
    if _hybrid_embeddings is None:
        _hybrid_embeddings = HybridEmbeddings()
    return _hybrid_embeddings
