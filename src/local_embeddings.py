"""
Local Embeddings Module.

Uses Sentence Transformers for on-device embedding generation.
Provides the same interface as OpenAI embeddings for easy swapping.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("resume_matcher.local_embeddings")

# * Model configuration
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 80MB, fast, good quality


class LocalEmbeddings:
    """
    Local embedding generator using Sentence Transformers.

    Uses all-MiniLM-L6-v2 by default - a lightweight model (~80MB)
    that provides good quality embeddings for semantic similarity.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the local embeddings generator.

        Args:
            model_name: Sentence Transformers model name.
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading local embedding model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                logger.info("Local embedding model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        if not text.strip():
            return []

        # * Truncate very long texts (model has 256 token limit for best results)
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error("Local embedding failed: %s", e, exc_info=True)
            return []

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Get embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # * Filter empty texts and truncate long ones
        max_chars = 8000
        processed = [
            t[:max_chars] if len(t) > max_chars else t
            for t in texts
            if t.strip()
        ]

        if not processed:
            return []

        try:
            embeddings = self.model.encode(processed, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error("Local batch embedding failed: %s", e, exc_info=True)
            return []

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First embedding vector.
            vec2: Second embedding vector.

        Returns:
            Cosine similarity score (0-1 for normalized vectors).
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


# * Global instance for reuse
_local_embeddings: Optional[LocalEmbeddings] = None


def get_local_embeddings() -> LocalEmbeddings:
    """
    Get a shared LocalEmbeddings instance.

    Returns:
        LocalEmbeddings instance with lazy-loaded model.
    """
    global _local_embeddings
    if _local_embeddings is None:
        _local_embeddings = LocalEmbeddings()
    return _local_embeddings
