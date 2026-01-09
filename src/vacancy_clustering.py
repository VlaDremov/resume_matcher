"""
Vacancy Clustering Pipeline.

Clusters vacancy descriptions into market-aligned categories using
TF-IDF keyword extraction, optional GPT categorization, and local
embedding-based deduplication.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.data_extraction import load_vacancy_files
from src.keyword_engine import (
    TECH_TAXONOMY,
    extract_keywords_from_text,
    extract_keywords_tfidf,
    get_technology_patterns,
)
from src.local_embeddings import get_local_embeddings

logger = logging.getLogger("resume_matcher.vacancy_clustering")


CategoryLabel = Literal["research_ml", "applied_production", "genai_llm", "general"]
ImportanceLabel = Literal["critical", "important", "nice_to_have"]


class KeywordCategorization(BaseModel):
    """GPT output model for keyword categorization."""

    class CategorizedKeyword(BaseModel):
        keyword: str
        category: CategoryLabel
        importance: ImportanceLabel
        is_technology: bool

    keywords: list[CategorizedKeyword] = Field(default_factory=list)
    synonyms: dict[str, list[str]] = Field(default_factory=dict)


class ClusterResult(BaseModel):
    """Final clustering output."""

    class Cluster(BaseModel):
        name: str
        vacancies: list[str]
        top_keywords: list[str]
        keyword_counts: dict[str, int]
        defining_technologies: list[str]
        defining_skills: list[str]

    clusters: dict[str, Cluster]
    total_vacancies: int
    total_unique_keywords: int
    pipeline_stats: dict[str, int] = Field(default_factory=dict)


@dataclass
class KeywordMetadata:
    category_by_keyword: dict[str, CategoryLabel]
    importance_by_keyword: dict[str, ImportanceLabel]
    is_technology_by_keyword: dict[str, bool]
    display_by_keyword: dict[str, str]
    synonym_map: dict[str, str]


_CATEGORY_DISPLAY_NAMES = {
    "research_ml": "Research & Advanced ML",
    "applied_production": "Applied ML & Production Systems",
    "genai_llm": "Generative AI & LLM Engineering",
}

_IMPORTANCE_RANK = {"critical": 0, "important": 1, "nice_to_have": 2}


def _normalize_keyword(keyword: str) -> str:
    cleaned = re.sub(r"[\s/]+", " ", keyword.strip().lower())
    cleaned = re.sub(r"[^\w\s.+-]", "", cleaned)
    return cleaned.strip()


def _is_valid_keyword(keyword: str) -> bool:
    return bool(keyword) and len(keyword) > 2


def _format_keyword(keyword: str, display_map: dict[str, str]) -> str:
    return display_map.get(keyword, keyword)


def _average_embedding(current: list[float], new: list[float], count: int) -> list[float]:
    if count <= 0:
        return new
    return [
        (current[i] * count + new[i]) / (count + 1)
        for i in range(min(len(current), len(new)))
    ]


class VacancyClusteringPipeline:
    """Hybrid vacancy clustering using TF-IDF + GPT + embeddings."""

    def __init__(
        self,
        vacancies_dir: str | Path = "vacancies",
        llm_client=None,
        use_gpt: bool = True,
    ) -> None:
        self.vacancies_dir = Path(vacancies_dir)
        self.local_embeddings = get_local_embeddings()
        self._llm_client = llm_client
        self.use_gpt = use_gpt

    @property
    def llm_client(self):
        """Lazy-load the LLM client."""
        if self._llm_client is None:
            try:
                from src.llm_client import get_client
                self._llm_client = get_client()
            except Exception as e:
                logger.warning("LLM client not available: %s", e)
                self._llm_client = None
        return self._llm_client

    async def cluster_async(self, num_clusters: int = 3) -> ClusterResult:
        """Main entry point - runs full hybrid pipeline."""
        vacancies = load_vacancy_files(self.vacancies_dir)
        if not vacancies:
            return ClusterResult(
                clusters={},
                total_vacancies=0,
                total_unique_keywords=0,
            )

        vacancy_names = [f"{name}.txt" for name in vacancies.keys()]
        texts = list(vacancies.values())

        # Stage 1: Multi-method keyword extraction
        tfidf_keywords = self._extract_keywords_tfidf(texts)
        taxonomy_keywords = self._extract_keywords_taxonomy(texts)
        vacancy_keyword_scores, all_keywords, tfidf_unique = self._extract_keywords_stage1(
            vacancy_names,
            texts,
            tfidf_keywords,
            taxonomy_keywords,
        )

        # Stage 2: GPT categorization + synonym normalization
        categorized = await self._enhance_with_gpt_async(sorted(all_keywords))
        metadata = self._build_keyword_metadata(categorized, all_keywords)

        # Stage 3: Embedding-based deduplication
        canonical_keywords = sorted(metadata.category_by_keyword.keys())
        keyword_clusters = self._cluster_by_embeddings(canonical_keywords, threshold=0.75)
        global_counts = Counter()
        for scores in vacancy_keyword_scores.values():
            for keyword, score in scores.items():
                global_counts[keyword] += score
        cluster_map = self._build_cluster_map(keyword_clusters, metadata, global_counts)

        # Stage 4: Assign vacancies to market categories
        cluster_result = self._assign_vacancies_to_clusters(
            vacancy_keyword_scores=vacancy_keyword_scores,
            metadata=metadata,
            cluster_map=cluster_map,
            num_clusters=num_clusters,
        )

        gpt_available = self.use_gpt and self.llm_client is not None
        pipeline_stats = {
            "tfidf_keywords": len(tfidf_unique),
            "raw_keywords": len(all_keywords),
            "gpt_categorized": len(canonical_keywords),
            "embedding_merged": len(keyword_clusters),
            "gpt_used": int(gpt_available),
        }

        if gpt_available:
            try:
                from src.llm_client import get_usage_summary
                usage = get_usage_summary()
                pipeline_stats["gpt_tokens_used"] = int(usage.get("total_tokens", 0))
            except Exception:
                pipeline_stats["gpt_tokens_used"] = 0

        cluster_result.pipeline_stats = pipeline_stats
        cluster_result.total_unique_keywords = len(keyword_clusters)
        cluster_result.total_vacancies = len(vacancy_names)
        return cluster_result

    def cluster(self, num_clusters: int = 3) -> ClusterResult:
        """Sync wrapper for cluster_async."""
        return asyncio.run(self.cluster_async(num_clusters=num_clusters))

    def _extract_keywords_stage1(
        self,
        vacancy_names: list[str],
        texts: list[str],
        tfidf_keywords: dict[int, list[tuple[str, float]]],
        taxonomy_keywords: dict[int, list[str]],
    ) -> tuple[dict[str, Counter], set[str], set[str]]:
        """Stage 1 keyword extraction for each vacancy."""
        vacancy_keyword_scores: dict[str, Counter] = {}
        all_keywords: set[str] = set()
        tfidf_unique: set[str] = set()

        patterns = get_technology_patterns()

        for idx, (vacancy_name, text) in enumerate(zip(vacancy_names, texts)):
            keyword_scores: Counter[str] = Counter()
            text_lower = text.lower()

            # Base extraction (NER, taxonomy matches, basic patterns)
            for kw, score in extract_keywords_from_text(
                text, top_n=80, use_vacancy_base=False
            ):
                norm = _normalize_keyword(kw)
                if _is_valid_keyword(norm):
                    keyword_scores[norm] += float(score)

            # TF-IDF keywords
            for kw, score in tfidf_keywords.get(idx, []):
                norm = _normalize_keyword(kw)
                if _is_valid_keyword(norm):
                    keyword_scores[norm] += float(score) * 8
                    tfidf_unique.add(norm)

            # Taxonomy matches (boost)
            for kw in taxonomy_keywords.get(idx, []):
                norm = _normalize_keyword(kw)
                if _is_valid_keyword(norm) and norm in text_lower:
                    keyword_scores[norm] += 3

            # Explicit technology pattern matches (boost)
            for pattern in patterns:
                for match in pattern.findall(text):
                    term = match if isinstance(match, str) else match[0]
                    norm = _normalize_keyword(term)
                    if _is_valid_keyword(norm):
                        keyword_scores[norm] += 2

            # Keep top keywords per vacancy
            top_keywords = Counter(dict(keyword_scores.most_common(80)))
            vacancy_keyword_scores[vacancy_name] = top_keywords
            all_keywords.update(top_keywords.keys())

        return vacancy_keyword_scores, all_keywords, tfidf_unique

    def _extract_keywords_tfidf(self, texts: list[str]) -> dict[int, list[tuple[str, float]]]:
        """Stage 1a: TF-IDF extraction with bigrams."""
        return extract_keywords_tfidf(texts, top_n_per_doc=30)

    def _extract_keywords_taxonomy(self, texts: list[str]) -> dict[int, list[str]]:
        """Stage 1b: Match against TECH_TAXONOMY."""
        taxonomy_keywords: dict[int, list[str]] = {}

        for idx, text in enumerate(texts):
            matches = []
            text_lower = text.lower()
            for keyword_list in TECH_TAXONOMY.values():
                for keyword in keyword_list:
                    if keyword in text_lower:
                        matches.append(keyword)
            taxonomy_keywords[idx] = matches

        return taxonomy_keywords

    async def _enhance_with_gpt_async(self, keywords: list[str]) -> KeywordCategorization:
        """Stage 2: GPT categorization and importance scoring."""
        if not keywords:
            return KeywordCategorization()

        if not self.use_gpt or self.llm_client is None:
            return self._categorize_keywords_locally(keywords)

        system_prompt = (
            "You are an expert ML hiring analyst. Categorize each keyword into one of: "
            "research_ml, applied_production, genai_llm, general. "
            "Assign importance (critical, important, nice_to_have) based on typical ML roles. "
            "Set is_technology to true for specific tools/services/frameworks, false for general skills. "
            "Return only keywords from the input list. "
            "Provide a synonyms map where keys are canonical keywords and values are synonyms "
            "found in the input list."
        )

        batch_size = 80
        batches = [keywords[i:i + batch_size] for i in range(0, len(keywords), batch_size)]
        tasks = []
        for batch in batches:
            prompt = (
                "Categorize the following keywords:\n"
                + "\n".join(f"- {kw}" for kw in batch)
            )
            tasks.append(self.llm_client.chat_structured_async(
                prompt=prompt,
                response_model=KeywordCategorization,
                system_prompt=system_prompt,
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged = KeywordCategorization()
        by_keyword: dict[str, KeywordCategorization.CategorizedKeyword] = {}
        synonyms: dict[str, list[str]] = {}

        for result in results:
            if isinstance(result, Exception):
                logger.warning("GPT keyword categorization failed: %s", result)
                continue

            for item in result.keywords:
                norm = _normalize_keyword(item.keyword)
                if not _is_valid_keyword(norm):
                    continue

                existing = by_keyword.get(norm)
                if existing is None:
                    by_keyword[norm] = item
                else:
                    existing_rank = _IMPORTANCE_RANK.get(existing.importance, 3)
                    incoming_rank = _IMPORTANCE_RANK.get(item.importance, 3)
                    if incoming_rank < existing_rank:
                        by_keyword[norm] = item
                    elif existing.category == "general" and item.category != "general":
                        by_keyword[norm] = item

            for canonical, syns in result.synonyms.items():
                if not syns:
                    continue
                canonical_norm = _normalize_keyword(canonical)
                if not _is_valid_keyword(canonical_norm):
                    continue
                normalized_syns = []
                for syn in syns:
                    syn_norm = _normalize_keyword(syn)
                    if _is_valid_keyword(syn_norm):
                        normalized_syns.append(syn_norm)
                if not normalized_syns:
                    continue
                synonyms.setdefault(canonical_norm, [])
                synonyms[canonical_norm].extend(normalized_syns)

        merged.keywords = list(by_keyword.values())
        merged.synonyms = synonyms
        if not merged.keywords:
            return self._categorize_keywords_locally(keywords)
        return merged

    def _categorize_keywords_locally(self, keywords: list[str]) -> KeywordCategorization:
        """Fallback categorization using taxonomy and heuristics."""
        categorized = KeywordCategorization()
        category_by_keyword: dict[str, KeywordCategorization.CategorizedKeyword] = {}
        tech_patterns = get_technology_patterns()

        for keyword in keywords:
            norm = _normalize_keyword(keyword)
            if not _is_valid_keyword(norm):
                continue

            category = "general"
            for cat, cat_keywords in TECH_TAXONOMY.items():
                if any(cat_kw in norm for cat_kw in cat_keywords):
                    category = cat
                    break

            is_technology = any(p.search(keyword) for p in tech_patterns)
            importance: ImportanceLabel = "nice_to_have"
            if is_technology:
                importance = "important"

            category_by_keyword[norm] = KeywordCategorization.CategorizedKeyword(
                keyword=keyword,
                category=category,
                importance=importance,
                is_technology=is_technology,
            )

        categorized.keywords = list(category_by_keyword.values())
        categorized.synonyms = {}
        return categorized

    def _build_keyword_metadata(
        self,
        categorized: KeywordCategorization,
        all_keywords: set[str],
    ) -> KeywordMetadata:
        """Build normalized keyword metadata lookup tables."""
        category_by_keyword: dict[str, CategoryLabel] = {}
        importance_by_keyword: dict[str, ImportanceLabel] = {}
        is_technology_by_keyword: dict[str, bool] = {}
        display_by_keyword: dict[str, str] = {}
        synonym_map: dict[str, str] = {}

        for item in categorized.keywords:
            norm = _normalize_keyword(item.keyword)
            if not _is_valid_keyword(norm):
                continue
            category_by_keyword[norm] = item.category
            importance_by_keyword[norm] = item.importance
            is_technology_by_keyword[norm] = bool(item.is_technology)
            display_by_keyword[norm] = item.keyword.strip()

        for canonical, synonyms in categorized.synonyms.items():
            canonical_norm = _normalize_keyword(canonical)
            if not _is_valid_keyword(canonical_norm):
                continue
            for syn in synonyms:
                syn_norm = _normalize_keyword(syn)
                if _is_valid_keyword(syn_norm):
                    synonym_map[syn_norm] = canonical_norm

        # Ensure all keywords have entries
        for keyword in all_keywords:
            norm = _normalize_keyword(keyword)
            if not _is_valid_keyword(norm):
                continue
            if norm not in category_by_keyword:
                category_by_keyword[norm] = "general"
            if norm not in importance_by_keyword:
                importance_by_keyword[norm] = "nice_to_have"
            if norm not in is_technology_by_keyword:
                is_technology_by_keyword[norm] = False
            if norm not in display_by_keyword:
                display_by_keyword[norm] = keyword

        return KeywordMetadata(
            category_by_keyword=category_by_keyword,
            importance_by_keyword=importance_by_keyword,
            is_technology_by_keyword=is_technology_by_keyword,
            display_by_keyword=display_by_keyword,
            synonym_map=synonym_map,
        )

    def _cluster_by_embeddings(
        self,
        keywords: list[str],
        threshold: float = 0.75,
    ) -> list[list[str]]:
        """Stage 3: Group similar keywords using local embeddings."""
        if not keywords:
            return []

        embeddings = self.local_embeddings.get_embeddings_batch(keywords)
        clusters: list[list[str]] = []
        centroids: list[list[float]] = []
        counts: list[int] = []

        for keyword, embedding in zip(keywords, embeddings):
            if not embedding:
                clusters.append([keyword])
                centroids.append(embedding)
                counts.append(1)
                continue

            best_idx = None
            best_sim = threshold
            for idx, centroid in enumerate(centroids):
                if not centroid:
                    continue
                sim = self.local_embeddings.cosine_similarity(embedding, centroid)
                if sim >= best_sim:
                    best_sim = sim
                    best_idx = idx

            if best_idx is None:
                clusters.append([keyword])
                centroids.append(embedding)
                counts.append(1)
            else:
                clusters[best_idx].append(keyword)
                centroids[best_idx] = _average_embedding(centroids[best_idx], embedding, counts[best_idx])
                counts[best_idx] += 1

        return clusters

    def _build_cluster_map(
        self,
        clusters: list[list[str]],
        metadata: KeywordMetadata,
        global_counts: Counter,
    ) -> dict[str, str]:
        """Map each keyword to its representative cluster keyword."""
        if not clusters:
            return {}

        cluster_map: dict[str, str] = {}
        for cluster in clusters:
            # Choose representative: highest frequency, fallback to shortest
            representative = max(
                cluster,
                key=lambda kw: (global_counts.get(kw, 0), -len(kw)),
            )
            for keyword in cluster:
                cluster_map[keyword] = representative

        return cluster_map

    def _assign_vacancies_to_clusters(
        self,
        vacancy_keyword_scores: dict[str, Counter],
        metadata: KeywordMetadata,
        cluster_map: dict[str, str],
        num_clusters: int = 3,
    ) -> ClusterResult:
        """Stage 4: Assign each vacancy to best-fit cluster."""
        categories = ["research_ml", "applied_production", "genai_llm"]
        clusters: dict[str, ClusterResult.Cluster] = {}
        for category in categories:
            clusters[category] = ClusterResult.Cluster(
                name=_CATEGORY_DISPLAY_NAMES.get(category, category),
                vacancies=[],
                top_keywords=[],
                keyword_counts={},
                defining_technologies=[],
                defining_skills=[],
            )

        for vacancy_name, keyword_scores in vacancy_keyword_scores.items():
            category_scores = Counter()

            for keyword, score in keyword_scores.items():
                canonical = metadata.synonym_map.get(keyword, keyword)
                canonical = cluster_map.get(canonical, canonical)
                category = metadata.category_by_keyword.get(canonical, "general")
                if category != "general":
                    category_scores[category] += score

            best_category = category_scores.most_common(1)
            if best_category:
                assigned_category = best_category[0][0]
            else:
                assigned_category = "applied_production"

            clusters[assigned_category].vacancies.append(vacancy_name)

            # Aggregate keyword counts for this cluster (category + general keywords)
            for keyword, score in keyword_scores.items():
                canonical = metadata.synonym_map.get(keyword, keyword)
                canonical = cluster_map.get(canonical, canonical)
                category = metadata.category_by_keyword.get(canonical, "general")
                if category in (assigned_category, "general"):
                    clusters[assigned_category].keyword_counts[canonical] = (
                        clusters[assigned_category].keyword_counts.get(canonical, 0)
                        + int(math.ceil(score))
                    )

        # Build defining keywords for each cluster
        for category, cluster in clusters.items():
            sorted_keywords = sorted(
                cluster.keyword_counts.items(),
                key=lambda x: (-x[1], x[0]),
            )
            top_keywords = [kw for kw, _ in sorted_keywords[:10]]
            tech_keywords = [
                kw for kw, _ in sorted_keywords
                if metadata.is_technology_by_keyword.get(kw, False)
                and metadata.category_by_keyword.get(kw, "general") == category
            ]
            skill_keywords = [
                kw for kw, _ in sorted_keywords
                if not metadata.is_technology_by_keyword.get(kw, False)
                and metadata.category_by_keyword.get(kw, "general") == category
            ]

            cluster.top_keywords = [_format_keyword(kw, metadata.display_by_keyword) for kw in top_keywords]
            cluster.defining_technologies = [
                _format_keyword(kw, metadata.display_by_keyword) for kw in tech_keywords[:10]
            ]
            cluster.defining_skills = [
                _format_keyword(kw, metadata.display_by_keyword) for kw in skill_keywords[:10]
            ]
            cluster.keyword_counts = {
                _format_keyword(kw, metadata.display_by_keyword): count
                for kw, count in cluster.keyword_counts.items()
            }

        # Respect num_clusters by trimming if needed
        if num_clusters < len(categories):
            ordered = sorted(
                categories,
                key=lambda name: len(clusters[name].vacancies),
                reverse=True,
            )
            trimmed = {name: clusters[name] for name in ordered[:num_clusters]}
        else:
            trimmed = clusters

        return ClusterResult(
            clusters=trimmed,
            total_vacancies=len(vacancy_keyword_scores),
            total_unique_keywords=0,
        )
