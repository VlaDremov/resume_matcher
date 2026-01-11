"""
Cluster Artifact Models and IO.

Defines the persisted JSON artifact generated from vacancy clustering.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from src.vacancy_clustering import ClusterResult


class ClusterArtifact(BaseModel):
    """Versioned clustering artifact for reuse in downstream steps."""

    class Cluster(BaseModel):
        cluster_id: int
        slug: str
        name: str
        summary: str
        vacancies: list[str]
        top_keywords: list[str]
        keyword_counts: dict[str, int]
        defining_technologies: list[str]
        defining_skills: list[str]
        profile_text: str

    schema_version: int = 1
    generated_at: str
    vacancies_dir: str
    signature: list[list[object]]
    num_clusters: int
    pipeline_stats: dict[str, object] = Field(default_factory=dict)
    clusters: list[Cluster] = Field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _vacancy_signature(vacancies_dir: Path) -> list[list[object]]:
    if not vacancies_dir.exists():
        return []

    signature: list[list[object]] = []
    for file_path in sorted(vacancies_dir.glob("*.txt")):
        try:
            stat = file_path.stat()
        except OSError:
            continue
        signature.append([file_path.name, stat.st_mtime_ns, stat.st_size])
    return signature


def _build_profile_text(cluster: ClusterResult.Cluster) -> str:
    parts = [cluster.name.strip()]
    if cluster.summary:
        parts.append(cluster.summary.strip())
    if cluster.top_keywords:
        parts.append(f"Keywords: {', '.join(cluster.top_keywords)}.")
    if cluster.defining_technologies:
        parts.append(f"Technologies: {', '.join(cluster.defining_technologies)}.")
    if cluster.defining_skills:
        parts.append(f"Skills: {', '.join(cluster.defining_skills)}.")
    return " ".join(part.strip().rstrip(".") + "." for part in parts if part).strip()


def build_cluster_artifact(
    result: ClusterResult,
    vacancies_dir: Path,
    num_clusters: int,
) -> ClusterArtifact:
    """Convert ClusterResult into a reusable ClusterArtifact."""
    clusters: list[ClusterArtifact.Cluster] = []
    for cluster_id, (slug, cluster) in enumerate(sorted(result.clusters.items())):
        clusters.append(
            ClusterArtifact.Cluster(
                cluster_id=cluster_id,
                slug=slug,
                name=cluster.name,
                summary=cluster.summary,
                vacancies=cluster.vacancies,
                top_keywords=cluster.top_keywords,
                keyword_counts=cluster.keyword_counts,
                defining_technologies=cluster.defining_technologies,
                defining_skills=cluster.defining_skills,
                profile_text=_build_profile_text(cluster),
            )
        )

    resolved_clusters = num_clusters if num_clusters > 0 else len(clusters)
    return ClusterArtifact(
        schema_version=1,
        generated_at=_utc_now_iso(),
        vacancies_dir=str(vacancies_dir),
        signature=_vacancy_signature(vacancies_dir),
        num_clusters=resolved_clusters,
        pipeline_stats=result.pipeline_stats or {},
        clusters=clusters,
    )


def load_cluster_artifact(path: str | Path) -> ClusterArtifact:
    """Load cluster artifact from disk."""
    artifact_path = Path(path)
    with open(artifact_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return ClusterArtifact.model_validate(payload)


def save_cluster_artifact(path: str | Path, artifact: ClusterArtifact) -> None:
    """Save cluster artifact to disk."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact.model_dump(), f, indent=2)
