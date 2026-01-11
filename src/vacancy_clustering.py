"""
Vacancy Clustering Pipeline.

Clusters vacancy descriptions into dynamic categories using vacancy
content, keyword extraction, optional GPT enrichment, and embedding-based
deduplication. Cluster labels are generated on the fly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.data_extraction import load_vacancy_files
from src.keyword_engine import (
    TECH_TAXONOMY,
    extract_keywords_from_text,
    extract_keywords_tfidf,
    get_technology_patterns,
)
from src.local_embeddings import get_local_embeddings

logger = logging.getLogger("resume_matcher.vacancy_clustering")


ImportanceLabel = Literal["critical", "important", "nice_to_have"]


class KeywordCategorization(BaseModel):
    """GPT output model for keyword enrichment."""
    model_config = ConfigDict(extra="forbid")

    class CategorizedKeyword(BaseModel):
        model_config = ConfigDict(extra="forbid")
        keyword: str
        importance: ImportanceLabel
        is_technology: bool
        is_noise: bool = False

    class SynonymGroup(BaseModel):
        model_config = ConfigDict(extra="forbid")
        canonical: str
        synonyms: list[str]

    keywords: list[CategorizedKeyword] = Field(default_factory=list)
    synonyms: list[SynonymGroup] = Field(default_factory=list)


class ClusterResult(BaseModel):
    """Final clustering output."""

    class Cluster(BaseModel):
        slug: str
        name: str
        summary: str = ""
        vacancies: list[str]
        top_keywords: list[str]
        keyword_counts: dict[str, int]
        defining_technologies: list[str]
        defining_skills: list[str]

    clusters: dict[str, Cluster]
    total_vacancies: int
    total_unique_keywords: int
    pipeline_stats: dict[str, object] = Field(default_factory=dict)


@dataclass
class KeywordMetadata:
    importance_by_keyword: dict[str, ImportanceLabel]
    is_technology_by_keyword: dict[str, bool]
    display_by_keyword: dict[str, str]
    synonym_map: dict[str, str]
    is_noise_by_keyword: dict[str, bool]


class ClusterLabel(BaseModel):
    """LLM-generated label for a cluster."""

    cluster_id: int
    name: str
    summary: str


class ClusterLabelsResponse(BaseModel):
    """LLM output for multiple cluster labels."""

    labels: list[ClusterLabel] = Field(default_factory=list)


_IMPORTANCE_RANK = {"critical": 0, "important": 1, "nice_to_have": 2}
_CACHE_PATH = Path("output/.vacancy_cluster_cache.json")
_NOISE_KEYWORDS = {
    "a", "an", "and", "the", "you", "your", "yours", "yourself", "we", "our",
    "ours", "ourselves", "they", "their", "theirs", "them", "themselves",
    "he", "him", "his", "she", "her", "hers", "it", "its", "this", "that",
    "these", "those", "who", "whom", "which", "what", "when", "where", "why",
    "how", "role", "roles", "position", "positions", "team", "teams", "company",
    "companies", "organization", "organizations", "candidate", "candidates",
    "requirements", "requirement", "qualification", "qualifications", "responsibility",
    "responsibilities", "preferred", "required", "must", "should", "ability",
    "abilities", "skills", "skill", "knowledge", "experience", "years", "year",
    "work", "working", "environment", "strong", "excellent", "good", "great",
}


def _normalize_keyword(keyword: str) -> str:
    cleaned = re.sub(r"[\s/]+", " ", keyword.strip().lower())
    cleaned = re.sub(r"[^\w\s.+-]", "", cleaned)
    return cleaned.strip()


def _is_valid_keyword(keyword: str) -> bool:
    return bool(keyword) and len(keyword) > 2


def _is_noise_keyword(keyword: str) -> bool:
    norm = _normalize_keyword(keyword)
    if not _is_valid_keyword(norm):
        return True
    if norm in _NOISE_KEYWORDS:
        return True
    tokens = [token for token in norm.split() if token]
    if tokens and all(token in _NOISE_KEYWORDS for token in tokens):
        return True
    if all(token.isdigit() for token in tokens):
        return True
    return False


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
        refresh_cache: bool = False,
        cache_path: str | Path | None = None,
    ) -> None:
        self.vacancies_dir = Path(vacancies_dir)
        self.local_embeddings = get_local_embeddings()
        self._llm_client = llm_client
        self.use_gpt = use_gpt
        self.refresh_cache = refresh_cache
        self.cache_path = Path(cache_path) if cache_path else _CACHE_PATH

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
        if not self.refresh_cache:
            cached = self._load_cache(num_clusters)
            if cached is not None:
                return cached

        vacancies = load_vacancy_files(self.vacancies_dir)
        if not vacancies:
            return ClusterResult(
                clusters={},
                total_vacancies=0,
                total_unique_keywords=0,
            )

        vacancy_items = sorted(vacancies.items(), key=lambda item: item[0])
        vacancy_names = [f"{name}.txt" for name, _ in vacancy_items]
        texts = [text for _, text in vacancy_items]

        # Stage 1: Multi-method keyword extraction
        tfidf_keywords = self._extract_keywords_tfidf(texts)
        taxonomy_keywords = self._extract_keywords_taxonomy(texts)
        vacancy_keyword_scores, all_keywords, tfidf_unique = self._extract_keywords_stage1(
            vacancy_names,
            texts,
            tfidf_keywords,
            taxonomy_keywords,
        )

        # Stage 2: GPT enrichment + synonym normalization
        categorized = await self._enhance_with_gpt_async(sorted(all_keywords))
        metadata = self._build_keyword_metadata(categorized, all_keywords)

        # Stage 3: Embedding-based deduplication
        canonical_keywords = sorted(metadata.display_by_keyword.keys())
        keyword_clusters = self._cluster_by_embeddings(canonical_keywords, threshold=0.75)
        global_counts = Counter()
        for scores in vacancy_keyword_scores.values():
            for keyword, score in scores.items():
                global_counts[keyword] += score
        cluster_map = self._build_cluster_map(keyword_clusters, metadata, global_counts)

        # Stage 4: Cluster vacancies dynamically
        vacancy_vectors, vector_source = self._compute_vacancy_vectors(texts)
        cluster_count, selection_method, silhouette = self._select_cluster_count(
            vacancy_vectors,
            num_clusters,
        )
        assignments = self._cluster_vacancies(vacancy_vectors, cluster_count)

        aggregated = self._aggregate_cluster_data(
            vacancy_names=vacancy_names,
            vacancy_keyword_scores=vacancy_keyword_scores,
            assignments=assignments,
            metadata=metadata,
            cluster_map=cluster_map,
        )

        gpt_available = self.use_gpt and self.llm_client is not None
        cluster_labels = {}
        if gpt_available:
            cluster_labels = await self._label_clusters_with_llm_async(aggregated, metadata)

        cluster_result = self._assemble_clusters(
            aggregated=aggregated,
            metadata=metadata,
            cluster_labels=cluster_labels,
        )

        pipeline_stats = {
            "tfidf_keywords": len(tfidf_unique),
            "raw_keywords": len(all_keywords),
            "canonical_keywords": len(canonical_keywords),
            "embedding_merged": len(keyword_clusters),
            "gpt_used": int(gpt_available),
            "vacancy_vector_source": vector_source,
            "cluster_count": cluster_count,
            "cluster_selection": selection_method,
            "silhouette_score": silhouette,
            "label_source": "llm" if cluster_labels else "fallback",
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

        self._save_cache(num_clusters, cluster_result)
        return cluster_result

    def cluster(self, num_clusters: int = 3) -> ClusterResult:
        """Sync wrapper for cluster_async."""
        return asyncio.run(self.cluster_async(num_clusters=num_clusters))

    def cluster_and_save(self, num_clusters: int, artifact_path: str | Path) -> "ClusterArtifact":
        """Cluster vacancies and write the cluster artifact to disk."""
        from src.cluster_artifacts import build_cluster_artifact, save_cluster_artifact

        result = self.cluster(num_clusters=num_clusters)
        artifact = build_cluster_artifact(result, self.vacancies_dir, num_clusters)
        save_cluster_artifact(artifact_path, artifact)
        return artifact

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
                if _is_valid_keyword(norm) and not _is_noise_keyword(norm):
                    keyword_scores[norm] += float(score)

            # TF-IDF keywords
            for kw, score in tfidf_keywords.get(idx, []):
                norm = _normalize_keyword(kw)
                if _is_valid_keyword(norm) and not _is_noise_keyword(norm):
                    keyword_scores[norm] += float(score) * 8
                    tfidf_unique.add(norm)

            # Taxonomy matches (boost)
            for kw in taxonomy_keywords.get(idx, []):
                norm = _normalize_keyword(kw)
                if _is_valid_keyword(norm) and not _is_noise_keyword(norm) and norm in text_lower:
                    keyword_scores[norm] += 3

            # Explicit technology pattern matches (boost)
            for pattern in patterns:
                for match in pattern.findall(text):
                    term = match if isinstance(match, str) else match[0]
                    norm = _normalize_keyword(term)
                    if _is_valid_keyword(norm) and not _is_noise_keyword(norm):
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
        """Stage 2: GPT enrichment (importance + technology detection)."""
        if not keywords:
            return KeywordCategorization()

        if not self.use_gpt or self.llm_client is None:
            return self._categorize_keywords_locally(keywords)

        system_prompt = (
            "You are an expert ML hiring analyst. For each keyword, assign importance "
            "(critical, important, nice_to_have) and set is_technology to true for specific "
            "tools/services/frameworks or false for general skills. Also set is_noise to true for "
            "irrelevant terms (pronouns, generic words like 'role', 'team', 'responsibilities') "
            "and false for genuine skills/technologies. Return only keywords from the input list. "
            "Provide synonyms as a list of objects with canonical and synonyms fields, where "
            "canonical is a keyword from the input list and synonyms are matching keywords found "
            "in the input list."
        )

        batch_size = 80
        batches = [keywords[i:i + batch_size] for i in range(0, len(keywords), batch_size)]
        tasks = []
        for batch in batches:
            prompt = (
                "Analyze the following keywords:\n"
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
                logger.warning("GPT keyword enrichment failed: %s", result)
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

            for group in result.synonyms:
                canonical = group.canonical
                syns = group.synonyms
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
        merged.synonyms = [
            KeywordCategorization.SynonymGroup(
                canonical=canonical,
                synonyms=sorted(set(syns)),
            )
            for canonical, syns in sorted(synonyms.items())
            if syns
        ]
        if not merged.keywords:
            return self._categorize_keywords_locally(keywords)
        return merged

    def _categorize_keywords_locally(self, keywords: list[str]) -> KeywordCategorization:
        """Fallback enrichment using taxonomy and heuristics."""
        categorized = KeywordCategorization()
        keyword_by_norm: dict[str, KeywordCategorization.CategorizedKeyword] = {}
        tech_patterns = get_technology_patterns()
        all_taxonomy = {kw for kws in TECH_TAXONOMY.values() for kw in kws}

        for keyword in keywords:
            norm = _normalize_keyword(keyword)
            if not _is_valid_keyword(norm):
                continue

            is_technology = any(p.search(keyword) for p in tech_patterns) or norm in all_taxonomy
            importance: ImportanceLabel = "nice_to_have"
            if is_technology:
                importance = "important"

            keyword_by_norm[norm] = KeywordCategorization.CategorizedKeyword(
                keyword=keyword,
                importance=importance,
                is_technology=is_technology,
                is_noise=_is_noise_keyword(norm),
            )

        categorized.keywords = list(keyword_by_norm.values())
        categorized.synonyms = []
        return categorized

    def _build_keyword_metadata(
        self,
        categorized: KeywordCategorization,
        all_keywords: set[str],
    ) -> KeywordMetadata:
        """Build normalized keyword metadata lookup tables."""
        importance_by_keyword: dict[str, ImportanceLabel] = {}
        is_technology_by_keyword: dict[str, bool] = {}
        display_by_keyword: dict[str, str] = {}
        synonym_map: dict[str, str] = {}
        is_noise_by_keyword: dict[str, bool] = {}

        for item in categorized.keywords:
            norm = _normalize_keyword(item.keyword)
            if not _is_valid_keyword(norm):
                continue
            importance_by_keyword[norm] = item.importance
            is_technology_by_keyword[norm] = bool(item.is_technology)
            display_by_keyword[norm] = item.keyword.strip()
            is_noise_by_keyword[norm] = bool(item.is_noise)

        for group in categorized.synonyms:
            canonical_norm = _normalize_keyword(group.canonical)
            if not _is_valid_keyword(canonical_norm):
                continue
            for syn in group.synonyms:
                syn_norm = _normalize_keyword(syn)
                if _is_valid_keyword(syn_norm):
                    synonym_map[syn_norm] = canonical_norm

        # Ensure all keywords have entries
        for keyword in all_keywords:
            norm = _normalize_keyword(keyword)
            if not _is_valid_keyword(norm):
                continue
            if norm not in importance_by_keyword:
                importance_by_keyword[norm] = "nice_to_have"
            if norm not in is_technology_by_keyword:
                is_technology_by_keyword[norm] = False
            if norm not in display_by_keyword:
                display_by_keyword[norm] = keyword
            if norm not in is_noise_by_keyword:
                is_noise_by_keyword[norm] = _is_noise_keyword(norm)

        return KeywordMetadata(
            importance_by_keyword=importance_by_keyword,
            is_technology_by_keyword=is_technology_by_keyword,
            display_by_keyword=display_by_keyword,
            synonym_map=synonym_map,
            is_noise_by_keyword=is_noise_by_keyword,
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
        if len(embeddings) != len(keywords):
            return [[keyword] for keyword in keywords]
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

    def _compute_vacancy_vectors(self, texts: list[str]) -> tuple[list[list[float]], str]:
        """Compute vectors for vacancy texts (local embeddings with TF-IDF fallback)."""
        embeddings: list[list[float]] = []
        for text in texts:
            try:
                embedding = self.local_embeddings.get_embedding(text)
            except Exception as exc:
                logger.warning("Local embedding failed: %s", exc)
                embedding = []
            embeddings.append(embedding)

        lengths = [len(vec) for vec in embeddings if vec]
        if len(embeddings) == len(texts) and lengths and len(set(lengths)) == 1 and all(embeddings):
            return embeddings, "local_embeddings"

        try:
            vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words="english",
                ngram_range=(1, 2),
            )
            matrix = vectorizer.fit_transform(texts)
            return matrix.toarray().tolist(), "tfidf"
        except Exception as exc:
            logger.warning("TF-IDF vectorization failed: %s", exc)
            return [[float(len(text))] for text in texts], "length_fallback"

    def _select_cluster_count(
        self,
        vectors: list[list[float]],
        requested: int,
    ) -> tuple[int, str, Optional[float]]:
        """Choose a cluster count using silhouette score when requested=0."""
        sample_count = len(vectors)
        if sample_count <= 1:
            return 1, "single", None

        if requested and requested > 0:
            return min(requested, sample_count), "requested", None

        max_k = min(8, sample_count)
        if max_k < 2:
            return 1, "single", None

        best_k = min(3, sample_count)
        best_score = None

        for k in range(2, max_k + 1):
            try:
                labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)
                if len(set(labels)) < 2:
                    continue
                score = float(silhouette_score(vectors, labels))
            except Exception as exc:
                logger.debug("Silhouette score failed k=%s: %s", k, exc)
                continue

            if best_score is None or score > best_score:
                best_score = score
                best_k = k

        if best_score is None:
            return min(3, sample_count), "auto_fallback", None

        return best_k, "auto", round(best_score, 4)

    def _cluster_vacancies(self, vectors: list[list[float]], num_clusters: int) -> list[int]:
        """Cluster vacancies using KMeans."""
        sample_count = len(vectors)
        if sample_count == 0:
            return []
        if num_clusters <= 1:
            return [0] * sample_count

        cluster_count = min(num_clusters, sample_count)
        try:
            model = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
            labels = model.fit_predict(vectors)
            return labels.tolist()
        except Exception as exc:
            logger.warning("Vacancy clustering failed: %s", exc)
            return [0] * sample_count

    def _aggregate_cluster_data(
        self,
        vacancy_names: list[str],
        vacancy_keyword_scores: dict[str, Counter],
        assignments: list[int],
        metadata: KeywordMetadata,
        cluster_map: dict[str, str],
    ) -> dict[int, dict[str, object]]:
        """Aggregate vacancy names and keyword counts per cluster."""
        aggregated: dict[int, dict[str, object]] = {}

        for idx, vacancy_name in enumerate(vacancy_names):
            cluster_id = assignments[idx] if idx < len(assignments) else 0
            entry = aggregated.setdefault(
                cluster_id,
                {"vacancies": [], "keyword_counts": Counter()},
            )
            entry["vacancies"].append(vacancy_name)

            keyword_scores = vacancy_keyword_scores.get(vacancy_name, Counter())
            for keyword, score in keyword_scores.items():
                canonical = metadata.synonym_map.get(keyword, keyword)
                canonical = cluster_map.get(canonical, canonical)
                if metadata.is_noise_by_keyword.get(canonical, False):
                    continue
                entry["keyword_counts"][canonical] += int(math.ceil(score))

        return aggregated

    def _extract_cluster_keywords(
        self,
        keyword_counts: Counter,
        metadata: KeywordMetadata,
    ) -> dict[str, list[str]]:
        """Extract top, technology, and skill keywords for a cluster."""
        sorted_keywords = sorted(
            keyword_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )
        top_keywords = [kw for kw, _ in sorted_keywords[:10]]
        tech_keywords = [
            kw for kw, _ in sorted_keywords
            if metadata.is_technology_by_keyword.get(kw, False)
        ]
        skill_keywords = [
            kw for kw, _ in sorted_keywords
            if not metadata.is_technology_by_keyword.get(kw, False)
        ]

        return {
            "top": top_keywords,
            "technologies": tech_keywords,
            "skills": skill_keywords,
        }

    async def _label_clusters_with_llm_async(
        self,
        aggregated: dict[int, dict[str, object]],
        metadata: KeywordMetadata,
    ) -> dict[int, ClusterLabel]:
        """Generate descriptive cluster labels using the LLM."""
        if not aggregated or self.llm_client is None:
            return {}

        cluster_inputs = []
        for cluster_id, entry in sorted(aggregated.items()):
            keyword_info = self._extract_cluster_keywords(entry["keyword_counts"], metadata)
            top_keywords = [
                _format_keyword(kw, metadata.display_by_keyword)
                for kw in keyword_info["top"][:8]
            ]
            technologies = [
                _format_keyword(kw, metadata.display_by_keyword)
                for kw in keyword_info["technologies"][:8]
            ]
            skills = [
                _format_keyword(kw, metadata.display_by_keyword)
                for kw in keyword_info["skills"][:8]
            ]

            cluster_inputs.append({
                "cluster_id": cluster_id,
                "vacancies": entry["vacancies"][:5],
                "top_keywords": top_keywords,
                "technologies": technologies,
                "skills": skills,
            })

        system_prompt = (
            "You label job clusters. For each cluster, provide a concise name "
            "(2-4 words) and a one-sentence summary. Use only the provided data."
        )

        prompt_lines = ["Label the following clusters:"]
        for cluster in cluster_inputs:
            prompt_lines.append(
                "Cluster {cluster_id}:\n"
                "Vacancies: {vacancies}\n"
                "Top keywords: {top_keywords}\n"
                "Technologies: {technologies}\n"
                "Skills: {skills}".format(**cluster)
            )

        try:
            result = await self.llm_client.chat_structured_async(
                prompt="\n\n".join(prompt_lines),
                response_model=ClusterLabelsResponse,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            logger.warning("Cluster labeling failed: %s", exc)
            return {}

        labels_by_id: dict[int, ClusterLabel] = {}
        for label in result.labels:
            if label.name.strip():
                labels_by_id[label.cluster_id] = label

        return labels_by_id

    def _fallback_cluster_name(
        self,
        cluster_id: int,
        keyword_info: dict[str, list[str]],
        metadata: KeywordMetadata,
    ) -> str:
        """Generate a deterministic fallback cluster name."""
        candidates = keyword_info["technologies"] or keyword_info["skills"] or keyword_info["top"]
        if candidates:
            display = [
                _format_keyword(kw, metadata.display_by_keyword)
                for kw in candidates[:2]
            ]
            return " and ".join(display)
        return f"Cluster {cluster_id + 1}"

    def _fallback_cluster_summary(
        self,
        keyword_info: dict[str, list[str]],
        metadata: KeywordMetadata,
    ) -> str:
        """Generate a fallback summary when LLM labeling is unavailable."""
        top_keywords = keyword_info["top"][:4]
        if not top_keywords:
            return "General ML roles with mixed requirements."
        formatted = [
            _format_keyword(kw, metadata.display_by_keyword)
            for kw in top_keywords
        ]
        return f"Focus on {', '.join(formatted)}."

    def _slugify(self, name: str, existing: set[str]) -> str:
        """Build a unique slug from a cluster name."""
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not base:
            base = "cluster"
        slug = base
        counter = 2
        while slug in existing:
            slug = f"{base}-{counter}"
            counter += 1
        existing.add(slug)
        return slug

    def _assemble_clusters(
        self,
        aggregated: dict[int, dict[str, object]],
        metadata: KeywordMetadata,
        cluster_labels: dict[int, ClusterLabel],
    ) -> ClusterResult:
        """Build ClusterResult from aggregated data and labels."""
        clusters: dict[str, ClusterResult.Cluster] = {}
        used_slugs: set[str] = set()

        for cluster_id, entry in sorted(aggregated.items()):
            keyword_counts: Counter = entry["keyword_counts"]
            keyword_info = self._extract_cluster_keywords(keyword_counts, metadata)

            label = cluster_labels.get(cluster_id)
            if label:
                name = label.name.strip()
                summary = label.summary.strip() or self._fallback_cluster_summary(keyword_info, metadata)
            else:
                name = self._fallback_cluster_name(cluster_id, keyword_info, metadata)
                summary = self._fallback_cluster_summary(keyword_info, metadata)

            slug = self._slugify(name, used_slugs)

            sorted_keywords = sorted(
                keyword_counts.items(),
                key=lambda x: (-x[1], x[0]),
            )
            top_keywords = [_format_keyword(kw, metadata.display_by_keyword) for kw, _ in sorted_keywords[:10]]
            technologies = [
                _format_keyword(kw, metadata.display_by_keyword)
                for kw, _ in sorted_keywords
                if metadata.is_technology_by_keyword.get(kw, False)
            ]
            skills = [
                _format_keyword(kw, metadata.display_by_keyword)
                for kw, _ in sorted_keywords
                if not metadata.is_technology_by_keyword.get(kw, False)
            ]

            clusters[slug] = ClusterResult.Cluster(
                slug=slug,
                name=name,
                summary=summary,
                vacancies=entry["vacancies"],
                top_keywords=top_keywords[:10],
                keyword_counts={
                    _format_keyword(kw, metadata.display_by_keyword): count
                    for kw, count in keyword_counts.items()
                },
                defining_technologies=technologies[:10],
                defining_skills=skills[:10],
            )

        return ClusterResult(
            clusters=clusters,
            total_vacancies=sum(len(entry["vacancies"]) for entry in aggregated.values()),
            total_unique_keywords=0,
        )

    def _vacancy_signature(self) -> tuple[tuple[str, int, int], ...]:
        """Build a signature of vacancy files for caching."""
        if not self.vacancies_dir.exists():
            return ()

        signature = []
        for file_path in sorted(self.vacancies_dir.glob("*.txt")):
            try:
                stat = file_path.stat()
            except OSError:
                continue
            signature.append((file_path.name, stat.st_mtime_ns, stat.st_size))

        return tuple(signature)

    def _load_cache(self, num_clusters: int) -> Optional[ClusterResult]:
        """Load cached clustering results if vacancy signature matches."""
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.debug("Failed to read cache: %s", exc)
            return None

        signature = tuple(tuple(item) for item in payload.get("signature", []))
        if signature != self._vacancy_signature():
            return None

        clusters_cache = payload.get("clusters", {})
        cached = clusters_cache.get(str(num_clusters))
        if not cached:
            return None

        try:
            result = ClusterResult.model_validate(cached)
        except Exception as exc:
            logger.debug("Failed to parse cache: %s", exc)
            return None

        result.pipeline_stats = result.pipeline_stats or {}
        result.pipeline_stats["cache_hit"] = 1
        return result

    def _save_cache(self, num_clusters: int, result: ClusterResult) -> None:
        """Persist clustering results to cache."""
        try:
            payload = {}
            if self.cache_path.exists():
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
        except Exception:
            payload = {}

        payload["signature"] = [list(item) for item in self._vacancy_signature()]
        clusters_cache = payload.get("clusters", {})
        clusters_cache[str(num_clusters)] = result.model_dump()
        payload["clusters"] = clusters_cache

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.debug("Failed to write cache: %s", exc)
