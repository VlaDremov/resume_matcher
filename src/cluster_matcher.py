"""
Cluster Matcher Module.

Matches job descriptions to vacancy clusters using hybrid embeddings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.cluster_artifacts import ClusterArtifact, load_cluster_artifact
from src.hybrid_embeddings import HybridEmbeddings, get_hybrid_embeddings

logger = logging.getLogger("resume_matcher.cluster_matcher")

_CACHE_FILENAME = ".cluster_embeddings_cache.json"


class ClusterMatcher:
    """Hybrid embedding matcher for vacancy clusters."""

    def __init__(
        self,
        artifact_path: str | Path,
        embeddings: HybridEmbeddings | None = None,
        cache_embeddings: bool = True,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.cache_embeddings = cache_embeddings
        self.embeddings = embeddings or get_hybrid_embeddings()

        self._artifact: Optional[ClusterArtifact] = None
        self._artifact_mtime_ns: Optional[int] = None
        self.cluster_profiles: dict[str, str] = {}
        self.cluster_embeddings: dict[str, dict] = {}

        self.embeddings_cache_path = self.artifact_path.parent / _CACHE_FILENAME

        self._load_artifact()
        self._load_or_compute_embeddings()

    def _load_artifact(self) -> None:
        if not self.artifact_path.exists():
            logger.warning("Cluster artifact not found: %s", self.artifact_path)
            self._artifact = None
            self.cluster_profiles = {}
            self.cluster_embeddings = {}
            return

        self._artifact = load_cluster_artifact(self.artifact_path)
        try:
            self._artifact_mtime_ns = self.artifact_path.stat().st_mtime_ns
        except OSError:
            self._artifact_mtime_ns = None

        self.cluster_profiles = {
            cluster.slug: cluster.profile_text
            for cluster in self._artifact.clusters
            if cluster.profile_text
        }

    def _artifact_signature(self) -> list[list[object]]:
        if not self._artifact:
            return []
        return self._artifact.signature

    def _maybe_reload(self) -> None:
        if not self.artifact_path.exists():
            return
        try:
            mtime_ns = self.artifact_path.stat().st_mtime_ns
        except OSError:
            return
        if self._artifact_mtime_ns is None or mtime_ns > self._artifact_mtime_ns:
            self._load_artifact()
            self._load_or_compute_embeddings()

    def _load_or_compute_embeddings(self) -> None:
        if not self.cluster_profiles:
            return

        if self.cache_embeddings and self.embeddings_cache_path.exists():
            try:
                with open(self.embeddings_cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                signature = cached.get("signature", [])
                cached_embeddings = cached.get("clusters", {})
                if (
                    signature == self._artifact_signature()
                    and set(cached_embeddings.keys()) == set(self.cluster_profiles.keys())
                ):
                    self.cluster_embeddings = cached_embeddings
                    logger.info("Loaded cluster embeddings from cache")
                    return
            except Exception as exc:
                logger.warning("Could not load cluster embeddings cache: %s", exc)

        self._compute_embeddings()

    def _compute_embeddings(self) -> None:
        if not self.cluster_profiles:
            return

        logger.info("Computing hybrid embeddings for %d clusters...", len(self.cluster_profiles))
        self.cluster_embeddings = {}

        for slug, profile_text in self.cluster_profiles.items():
            try:
                embedding = self.embeddings.get_embedding(profile_text)
                self.cluster_embeddings[slug] = embedding
                has_openai = "yes" if embedding.get("has_openai") else "no"
                logger.info("Computed embedding for cluster %s (openai=%s)", slug, has_openai)
            except Exception as exc:
                logger.error("Failed to compute embedding for cluster %s: %s", slug, exc)

        if self.cache_embeddings and self.cluster_embeddings:
            payload = {
                "signature": self._artifact_signature(),
                "clusters": self.cluster_embeddings,
            }
            try:
                self.embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.embeddings_cache_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                logger.info("Saved cluster embeddings cache")
            except Exception as exc:
                logger.warning("Could not save cluster embeddings cache: %s", exc)

    def match(self, job_text: str) -> dict[str, object]:
        """Match a job description to the best cluster (sync)."""
        self._maybe_reload()
        if not self.cluster_embeddings:
            return {"best_cluster": None, "best_score": 0.0, "scores": {}}

        try:
            job_embedding = self.embeddings.get_embedding(job_text)
        except Exception as exc:
            logger.error("Failed to get job embedding: %s", exc)
            return {"best_cluster": None, "best_score": 0.0, "scores": {}}

        scores = {}
        for slug, cluster_embedding in self.cluster_embeddings.items():
            scores[slug] = float(self.embeddings.compute_similarity(job_embedding, cluster_embedding))

        best_cluster = max(scores, key=scores.get)
        return {
            "best_cluster": best_cluster,
            "best_score": scores[best_cluster],
            "scores": scores,
        }

    async def match_async(self, job_text: str) -> dict[str, object]:
        """Match a job description to the best cluster (async)."""
        self._maybe_reload()
        if not self.cluster_embeddings:
            return {"best_cluster": None, "best_score": 0.0, "scores": {}}

        try:
            job_embedding = await self.embeddings.get_embedding_async(job_text)
        except Exception as exc:
            logger.error("Failed to get job embedding async: %s", exc)
            return {"best_cluster": None, "best_score": 0.0, "scores": {}}

        scores = {}
        for slug, cluster_embedding in self.cluster_embeddings.items():
            scores[slug] = float(self.embeddings.compute_similarity(job_embedding, cluster_embedding))

        best_cluster = max(scores, key=scores.get)
        return {
            "best_cluster": best_cluster,
            "best_score": scores[best_cluster],
            "scores": scores,
        }
