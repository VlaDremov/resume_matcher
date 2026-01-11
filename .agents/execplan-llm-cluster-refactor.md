# Cluster-Driven LLM and Taxonomy Refactor

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

The repository includes `PLANS.md` at the root. This ExecPlan must be maintained in accordance with `PLANS.md`.

## Purpose / Big Picture

After this change, the vacancy cluster artifact becomes the single source of truth for category keys across the pipeline. Resume skill ordering, keyword taxonomy, and market trend categories are all derived from the initial clusterization output instead of fixed three-theme constants. Resume variant generation sends LLM rewrite requests for every cluster in parallel, so only the clusterization step is the heavy, slow, token-expensive stage and everything that consumes the artifact is faster and deterministic. Users can see the behavior by clustering `vacancies/` into N clusters, generating N resume variants, and analyzing or matching a new job description while observing that categories match cluster slugs rather than static labels.

## Progress

- [x] (2026-01-11 21:58Z) Drafted the initial ExecPlan for the cluster-driven LLM and taxonomy refactor.
- [x] (2026-01-11 22:23Z) Extended the cluster artifact schema with derived category metadata and removed runtime import cycles for artifact access.
- [x] (2026-01-11 22:23Z) Implemented cluster-derived category loading and dynamic taxonomy in `src/keyword_engine.py`.
- [x] (2026-01-11 22:23Z) Updated resume generation category ordering to be cluster-driven with cached ordering per cluster slug.
- [x] (2026-01-11 22:23Z) Updated market trends prompts and response handling to use cluster-derived categories.
- [x] (2026-01-11 22:23Z) Parallelized GPT resume rewrites per cluster with a concurrency control.
- [ ] (2026-01-11 22:23Z) Run validation commands in this plan and record observed outputs.

## Surprises & Discoveries

- Observation: GPT resume rewriting is labeled as parallel but is awaited sequentially, so requests are not actually concurrent.
  Evidence: `src/resume_generator.py` `_generate_variants_from_clusters_with_gpt_async` awaits each theme task in a for-loop.
- Observation: `src/keyword_engine.py` hard-codes `TECH_TAXONOMY` and `ALL_TECH_KEYWORDS`, locking categories to three keys.
  Evidence: `src/keyword_engine.py` `TECH_TAXONOMY` is a static dict with `research_ml`, `applied_production`, `genai_llm`.
- Observation: Market trends prompts and model descriptions still reference fixed categories.
  Evidence: `src/market_trends.py` system prompt lists `research_ml`, `applied_production`, `genai_llm`.
- Observation: `src/cluster_artifacts.py` imports `src.vacancy_clustering`, which already imports `src.keyword_engine`, so adding artifact loading to `keyword_engine` risks a circular import.
  Evidence: `src/cluster_artifacts.py` `from src.vacancy_clustering import ClusterResult`.

## Decision Log

- Decision: Use the cluster artifact as the authoritative source for category keys and keyword pools, with a fallback to default static categories when no artifact exists.
  Rationale: The initial clusterization has to run without a prior artifact, but downstream steps must align to cluster slugs after the artifact is created.
  Date/Author: 2026-01-11, Codex.
- Decision: Replace static skill category ordering with a cluster-driven scoring model that ranks each skills category by overlap with the cluster keyword pool.
  Rationale: This removes the three-theme assumption and makes ordering reflect the actual cluster content.
  Date/Author: 2026-01-11, Codex.
- Decision: Run GPT rewrite requests per cluster concurrently and optionally cap concurrency with a semaphore controlled by an environment variable.
  Rationale: Parallelism reduces total generation time while allowing safe rate-limit tuning.
  Date/Author: 2026-01-11, Codex.
- Decision: Remove the runtime dependency from `src/cluster_artifacts.py` on `src.vacancy_clustering` using `TYPE_CHECKING` or forward references so keyword code can load artifacts without circular imports.
  Rationale: Dynamic taxonomy needs to read the artifact from `keyword_engine.py`, which is already imported by `vacancy_clustering.py`.
  Date/Author: 2026-01-11, Codex.
- Decision: Add a `categories` field to the cluster artifact schema and bump `schema_version` to 2 to record derived category metadata.
  Rationale: Persisting the derived category list avoids recomputation and makes downstream category usage deterministic.
  Date/Author: 2026-01-11, Codex.
- Decision: Default GPT rewrite concurrency to the number of clusters, with an environment override via `GPT_REWRITE_CONCURRENCY`.
  Rationale: Full fan-out makes resume rewriting time bound by the slowest cluster while allowing explicit throttling when rate limits require it.
  Date/Author: 2026-01-11, Codex.

## Outcomes & Retrospective

Implementation work is complete for schema updates, dynamic taxonomy, market trends categories, and async GPT rewrites. Validation runs are still pending, so the observable behavior has not yet been confirmed against the acceptance criteria.

## Context and Orientation

The CLI entry point is `main.py`. Vacancy clustering is implemented in `src/vacancy_clustering.py` and writes a JSON cluster artifact via `src/cluster_artifacts.py` to `output/vacancy_clusters.json`. Resume variants are generated in `src/resume_generator.py` based on cluster artifact data and optionally rewritten by GPT in `src/bullet_rewriter.py`. Keyword extraction and categorization live in `src/keyword_engine.py` and are used both by the clustering pipeline and by the `main.py analyze` command. Market trend enrichment is handled by `src/market_trends.py` and is wired into API responses via `backend/services.py`. Matching new job descriptions to the best cluster is implemented in `src/cluster_matcher.py` and uses the artifact’s `profile_text` embeddings.

Define the terms used here. A "cluster artifact" is the JSON file saved after clustering that contains cluster slugs, names, summaries, and keyword lists. A "cluster category" is a category key derived from a cluster slug and its keyword pool. A "taxonomy" is a mapping from category keys to keyword lists used to classify and boost keywords. "Order preferences" are the ordered list of skill section categories used to reorder the resume skills section.

## Plan of Work

First, extend the cluster artifact schema in `src/cluster_artifacts.py` to include a derived `categories` list and bump the schema version. Add a helper that dedupes each cluster’s keyword pool from `top_keywords`, `defining_technologies`, and `defining_skills` into category metadata (slug, name, summary, keywords), and expose it via `get_cluster_categories` so downstream code can read categories without recomputing. Remove the runtime import cycle with `TYPE_CHECKING` so artifact access is safe from `keyword_engine.py`. When the artifact does not exist or predates the new field, the helper must fall back to deriving categories from the cluster list.

Next, refactor `src/keyword_engine.py` to replace the fixed `TECH_TAXONOMY` and `ALL_TECH_KEYWORDS` with dynamic accessors that consult the cluster artifact. Rename the existing static map to `DEFAULT_TECH_TAXONOMY`, add `get_tech_taxonomy()` and `get_all_tech_keywords()` helpers that use the cluster-category loader with caching, and update `extract_keywords_from_text`, `cluster_keywords`, and `analyze_vacancies` to call those helpers instead of the old globals. Update `src/vacancy_clustering.py` calls that rely on `TECH_TAXONOMY` to use the new helper so they automatically pick up cluster-derived categories when the artifact exists.

Then update `src/resume_generator.py` so skill category ordering is driven by cluster keywords instead of a fixed `order_preferences` map. Use the parsed Technical Skills categories and score each category based on overlap with the cluster keyword pool or skills priority list, and order categories by that score with a stable tie-breaker (original order or alphabetical). Cache the computed order per cluster slug to avoid repeated scoring during generation, and remove the hard-coded three-key map entirely. The resulting order preferences are now keyed by cluster slugs derived from the artifact.

After that, update `src/market_trends.py` to build its system prompt categories from the cluster artifact. The prompt should enumerate each category slug with its name, summary, and a few keywords, and instruct the model to use only those category keys in `TrendingSkill.category`. Add a small post-processing step that maps unknown categories to the closest known slug (by name or keyword overlap) or a "general" fallback. When the artifact is missing, fall back to the legacy three categories so the feature keeps working without clustering.

Finally, make GPT resume rewriting truly asynchronous. In `_generate_variants_from_clusters_with_gpt_async` in `src/resume_generator.py`, create tasks for each cluster’s bullet and summary rewrites, run them concurrently with `asyncio.gather` or `asyncio.TaskGroup`, and optionally gate the calls with a semaphore controlled by an environment variable such as `GPT_REWRITE_CONCURRENCY`. Preserve the existing cache behavior and error fallback so a failed rewrite for one cluster does not block generating the rest.

## Concrete Steps

From the repository root, run clustering to refresh the artifact:

    python main.py cluster-vacancies --clusters 4 --output output/vacancy_clusters.json

Then run the keyword analysis to verify that categories are keyed by cluster slug:

    python main.py analyze --vacancies vacancies --output output/keyword_report.json

Generate resume variants with GPT rewriting enabled and observe that rewrite work is queued for all clusters before results are awaited:

    python main.py generate --clusters-artifact output/vacancy_clusters.json --use-gpt-rewrite --no-gpt-cache

Start the API and fetch an analysis with market trends enabled, verifying that trending skills are categorized by cluster slug:

    python main.py serve

In another terminal:

    curl -s -X POST http://localhost:8000/api/analyze \
      -H "Content-Type: application/json" \
      -d '{"job_description":"Example ML engineer role ...","use_semantic":true,"include_market_trends":true}'

## Validation and Acceptance

Acceptance is met when the keyword analysis output and API responses use cluster slugs (plus an optional "general") as category keys, with no reliance on the three fixed categories. Resume generation must show that category ordering changes based on cluster keywords rather than fixed mappings, and GPT rewriting must fire concurrent requests for all clusters so total runtime scales with the slowest cluster instead of the sum of all clusters. Market trends prompts must include the cluster-derived categories and `TrendingSkill.category` values must be limited to those keys or mapped deterministically when the model drifts.

## Idempotence and Recovery

Re-running `python main.py cluster-vacancies` safely overwrites the artifact and updates the cached category loader based on file mtime. If the artifact is missing or invalid, keyword extraction and market trends should fall back to the default static taxonomy to keep the CLI usable. If the GPT rewrite fan-out hits rate limits, lower `GPT_REWRITE_CONCURRENCY` or rerun with `--no-gpt-rewrite` to recover without changing artifacts.

## Artifacts and Notes

Example category definitions derived from the artifact (shape only):

    - slug: ml-platform
      name: ML Platform
      summary: Production ML systems and MLOps tooling.
      keywords: mlops, pipelines, monitoring, docker, kubernetes

Example log line to validate async rewrite fan-out:

    Queued GPT rewrite tasks for 4 clusters (concurrency=4)

## Interfaces and Dependencies

In `src/cluster_artifacts.py`, define derived category metadata and avoid circular imports:

    DEFAULT_CLUSTER_ARTIFACT = Path("output/vacancy_clusters.json")

    class ClusterCategory(BaseModel):
        slug: str
        name: str
        summary: str
        keywords: list[str]

    class ClusterArtifact(BaseModel):
        schema_version: int = 2
        categories: list[ClusterCategory]

    def get_cluster_categories(artifact: ClusterArtifact) -> list[ClusterCategory]

In `src/keyword_engine.py`, replace global constants with dynamic accessors:

    DEFAULT_TECH_TAXONOMY = { ...existing static map... }

    def get_tech_taxonomy(artifact_path: str | Path | None = None, refresh: bool = False) -> dict[str, list[str]]
    def get_all_tech_keywords(artifact_path: str | Path | None = None) -> set[str]

Ensure `extract_keywords_from_text`, `cluster_keywords`, and `analyze_vacancies` use these helpers.

In `src/resume_generator.py`, replace static ordering with cluster-driven scoring:

    def _get_category_order_for_theme(
        theme_config: dict,
        category_skills: dict[str, list[str]],
        original_order: list[str],
    ) -> list[str]

Store or cache the resulting order by `theme_config["slug"]` and remove the static `order_preferences` dict.

In `src/market_trends.py`, allow dynamic categories and prompt generation:

    class MarketTrendsService:
        def __init__(..., artifact_path: str | Path | None = None): ...
        def _load_cluster_categories(self) -> list[ClusterCategory]: ...
        def _build_category_prompt(self, categories: list[ClusterCategory]) -> str: ...

`TrendingSkill.category` must be one of the cluster slugs (or "general" when used as a fallback).

In `src/cluster_artifacts.py`, remove runtime imports that create cycles:

    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from src.vacancy_clustering import ClusterResult

Use forward references for type hints to keep runtime imports acyclic.

## Plan Change Note

Plan created on 2026-01-11 to satisfy the request for a cluster-driven LLM usage and taxonomy refactor, with async GPT resume generation and dynamic categories derived from the initial clustering artifact.

Plan updated on 2026-01-11 to record the implemented schema changes, dynamic taxonomy behavior, async GPT fan-out, and to align the Interfaces section with the concrete code paths now in place.
