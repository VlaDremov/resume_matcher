# Resume Variant Generation Plan

## Overview

Generate **3 keyword-optimized resume variants** by analyzing all job descriptions in `vacancies/` folder, identifying keyword clusters, and creating targeted resume versions that maximize keyword matching. Add a new CLI command for vacancy clustering.

---

## Analysis Summary

### Current State
- **36 job descriptions** in `vacancies/` folder
- **5 existing theme-based variants**: mlops, nlp_llm, cloud_aws, data_engineering, classical_ml
- **Existing infrastructure** in `resume_generator.py` handles: skills reordering, bullet enhancement, summary customization

### Identified Job Categories (from vacancy analysis)
Based on keyword clustering of all 36 vacancies, consolidated into **3 broad categories**:

| Category | Key Companies | Core Focus |
|----------|---------------|------------|
| **Research/Theory** | Google, Microsoft, Zalando, Delivery Hero | Algorithm innovation, statistical rigor, deep learning, model optimization |
| **Applied/Production** | Expedia, Strava, Monzo, HelloFresh, ASOS, eBay | Production ML systems, MLOps, A/B testing, infrastructure, CI/CD |
| **GenAI/LLM** | Docker, Miro, Cardo, Scalable, Moss, ML6 | LLM applications, RAG, agentic systems, prompt engineering |

### Unused Code to Clean Up
- `src/matcher.py` - Legacy TF-IDF matcher (368 lines, replaced by semantic_matcher.py)
- `src/linkedin_scraper.py` - Deprecated ToS-violating scraper (428 lines)
- `rewrite_bullets` parameter in services.py - Ignored, confusing
- `ErrorResponse` schema - Never used

---

## Proposed Resume Variants (3)

Based on vacancy keyword analysis, replace current 5 themes with 3 market-aligned variants:

### 1. **research_ml** - Research & Advanced ML
**Target:** Google, Microsoft, Zalando, Delivery Hero (algorithm-focused roles)
**Summary emphasis:** Statistical rigor, model optimization, deep learning research, experimentation
**Priority skills:** PyTorch, TensorFlow, Deep Learning, Statistical Analysis, XGBoost, CatBoost, Neural Networks
**Bullet keywords:** neural networks, optimization, experiments, statistical, algorithms, model performance, research
**Covers:** Research DS, classical ML, theory-heavy applied roles

### 2. **applied_production** - Applied ML & Production Systems
**Target:** Expedia, Strava, Monzo, HelloFresh, ASOS, eBay, Bolt (production-focused roles)
**Summary emphasis:** End-to-end ML pipelines, production systems, MLOps, A/B testing, infrastructure
**Priority skills:** Docker, Kubernetes, AWS Sagemaker, MLflow, Airflow, CI/CD, Feature Engineering, SQL
**Bullet keywords:** production, pipeline, deployment, A/B testing, monitoring, retraining, infrastructure, containerization, scaling
**Covers:** Applied ML Engineering, MLOps, Data Engineering, Platform roles

### 3. **genai_llm** - Generative AI & LLM Engineering
**Target:** Docker, Miro, Cardo, Scalable, Moss, ML6 (GenAI-focused roles)
**Summary emphasis:** LLM applications, agentic systems, RAG pipelines, prompt engineering
**Priority skills:** LangChain, LangGraph, RAG, FastAPI, Vector Databases, Embeddings, Prompt Engineering, REST APIs
**Bullet keywords:** LLM, GPT, agents, prompts, embeddings, retrieval, generation, Telegram bots, automated
**Covers:** GenAI Engineering, LLM Integration, AI Applications

---

## Implementation Plan

### Phase 1: Hybrid Keyword Clustering Pipeline + CLI Command
**Goal:** Create accurate vacancy clustering using TF-IDF, OpenAI API, and local embeddings

**Files to create/modify:**
- `src/vacancy_clustering.py` - NEW: Hybrid clustering pipeline
- `src/keyword_engine.py` - Add utility functions
- `main.py` - Add new `cluster-vacancies` CLI command

---

#### Pipeline Architecture (4 stages):

```
Input: vacancies/*.txt (36 files)
         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Multi-Method Keyword Extraction                     │
│ ├─ TF-IDF: Extract top 30 keywords per vacancy (bigrams)     │
│ ├─ Taxonomy Match: Match against TECH_TAXONOMY categories    │
│ ├─ spaCy NER: Extract noun phrases + named entities          │
│ └─ Pattern Match: Detect technologies (PyTorch, CatBoost)    │
│                                                              │
│ Output: Raw keyword set per vacancy (~50-100 keywords each)  │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Semantic Enhancement (OpenAI API)                   │
│ ├─ Batch process: Send all keywords to GPT for categorization│
│ ├─ Structured output: Use Pydantic model for reliable JSON   │
│ ├─ Importance scoring: Critical / Important / Nice-to-have   │
│ └─ Deduplication: Merge synonyms (e.g., "ML" = "machine      │
│                   learning")                                 │
│                                                              │
│ Output: Cleaned, categorized keywords with importance scores │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Embedding-Based Similarity Clustering               │
│ ├─ Local embeddings: all-MiniLM-L6-v2 for all unique keywords│
│ ├─ Similarity matrix: Pairwise cosine similarity             │
│ ├─ Threshold clustering: Group keywords with sim > 0.75      │
│ └─ Centroid selection: Pick representative keyword per group │
│                                                              │
│ Output: Deduplicated keyword clusters                        │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Vacancy Assignment to 3 Categories                  │
│ ├─ Score each vacancy: Count keywords per category           │
│ ├─ Primary assignment: Highest-scoring category wins         │
│ ├─ Secondary affinity: Track overlap with other categories   │
│ └─ Generate report: Cluster summary with keyword frequencies │
│                                                              │
│ Output: 3 clusters with vacancies + defining keywords        │
└─────────────────────────────────────────────────────────────┘
```

---

#### Tasks:

**1. Create `src/vacancy_clustering.py` (NEW FILE ~300 lines):**

```python
class VacancyClusteringPipeline:
    """Hybrid vacancy clustering using TF-IDF + OpenAI + embeddings"""

    def __init__(self, vacancies_dir: str = "vacancies"):
        self.vacancies_dir = Path(vacancies_dir)
        self.local_embeddings = get_local_embeddings()  # Lazy load
        self.llm_client = LLMClient()  # For GPT calls

    async def cluster_async(self, num_clusters: int = 3) -> ClusterResult:
        """Main entry point - runs full hybrid pipeline"""

    def _extract_keywords_tfidf(self, texts: list[str]) -> dict[str, list[str]]:
        """Stage 1a: TF-IDF extraction with bigrams"""

    def _extract_keywords_taxonomy(self, texts: list[str]) -> dict[str, list[str]]:
        """Stage 1b: Match against TECH_TAXONOMY"""

    async def _enhance_with_gpt_async(self, keywords: list[str]) -> CategorizedKeywords:
        """Stage 2: GPT categorization and importance scoring"""

    def _cluster_by_embeddings(self, keywords: list[str], threshold: float = 0.75) -> list[list[str]]:
        """Stage 3: Group similar keywords using local embeddings"""

    def _assign_vacancies_to_clusters(self, vacancy_keywords: dict, cluster_definitions: dict) -> ClusterResult:
        """Stage 4: Assign each vacancy to best-fit cluster"""
```

**Pydantic models for structured output:**

```python
class KeywordCategorization(BaseModel):
    """GPT output model for keyword categorization"""

    class CategorizedKeyword(BaseModel):
        keyword: str
        category: Literal["research_ml", "applied_production", "genai_llm", "general"]
        importance: Literal["critical", "important", "nice_to_have"]
        is_technology: bool  # True for PyTorch, False for "machine learning"

    keywords: list[CategorizedKeyword]
    synonyms: dict[str, list[str]]  # {"machine learning": ["ML", "ml"]}

class ClusterResult(BaseModel):
    """Final clustering output"""

    class Cluster(BaseModel):
        name: str
        vacancies: list[str]
        top_keywords: list[str]
        keyword_counts: dict[str, int]
        defining_technologies: list[str]  # PyTorch, CatBoost, etc.
        defining_skills: list[str]  # "deep learning", "production systems"

    clusters: dict[str, Cluster]
    total_vacancies: int
    total_unique_keywords: int
```

**2. Add utility functions to `src/keyword_engine.py`:**

```python
def extract_keywords_hybrid(text: str) -> list[tuple[str, float]]:
    """Combine TF-IDF + taxonomy + patterns with weighted scoring"""

def get_technology_patterns() -> list[re.Pattern]:
    """Regex patterns for detecting specific technologies:
    - Framework names: PyTorch, TensorFlow, CatBoost, XGBoost, LangChain
    - Cloud services: AWS Sagemaker, EC2, S3, GCP BigQuery
    - Tools: Docker, Kubernetes, MLflow, Airflow
    - Languages: Python, SQL, Go, Rust
    """
```

**3. Add CLI command in `main.py`:**

```python
@app.command()
def cluster_vacancies(
    output: str = typer.Option(None, "--output", "-o", help="Save JSON to file"),
    num_clusters: int = typer.Option(3, "--clusters", "-n", help="Number of clusters"),
    use_gpt: bool = typer.Option(True, "--gpt/--no-gpt", help="Use GPT for enhancement"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output")
):
    """Analyze and cluster all vacancies by keyword similarity"""
```

**CLI usage:**
```bash
# Basic clustering (uses GPT by default)
python main.py cluster-vacancies

# Fast local-only mode (no OpenAI API calls)
python main.py cluster-vacancies --no-gpt

# Save detailed JSON output
python main.py cluster-vacancies --output clusters.json --verbose

# Custom number of clusters
python main.py cluster-vacancies --clusters 5
```

---

#### Output Format:

**Console output:**
```
Clustering 36 vacancies into 3 categories...

Stage 1: Extracting keywords (TF-IDF + taxonomy)... ✓ 847 unique keywords
Stage 2: Enhancing with GPT... ✓ Categorized 847 → 312 canonical keywords
Stage 3: Clustering by embeddings... ✓ Merged to 186 keyword groups
Stage 4: Assigning vacancies... ✓

═══════════════════════════════════════════════════════════════
CLUSTER: research_ml (12 vacancies)
───────────────────────────────────────────────────────────────
Vacancies: google.txt, microsoft.txt, zalando_1.txt, deliveryhero.txt...
Technologies: PyTorch (10), TensorFlow (8), XGBoost (7), CatBoost (6)
Skills: deep learning (12), neural networks (9), statistical analysis (8)
Top Keywords: model optimization, research, experiments, algorithms

═══════════════════════════════════════════════════════════════
CLUSTER: applied_production (15 vacancies)
───────────────────────────────────────────────────────────────
Vacancies: expedia.txt, strava.txt, monzo.txt, asos.txt, ebay.txt...
Technologies: Docker (14), Kubernetes (11), MLflow (9), Airflow (8)
Skills: production systems (15), MLOps (12), A/B testing (11)
Top Keywords: deployment, pipeline, monitoring, infrastructure

═══════════════════════════════════════════════════════════════
CLUSTER: genai_llm (9 vacancies)
───────────────────────────────────────────────────────────────
Vacancies: docker.txt, miro.txt, cardo.txt, scalable.txt...
Technologies: LangChain (8), FastAPI (6), Vector DBs (5)
Skills: LLM applications (9), RAG (7), prompt engineering (6)
Top Keywords: agents, embeddings, retrieval, generation
```

**JSON output (`clusters.json`):**
```json
{
  "clusters": {
    "research_ml": {
      "name": "Research & Advanced ML",
      "vacancies": ["google.txt", "microsoft.txt", ...],
      "top_keywords": ["deep learning", "neural networks", "optimization"],
      "keyword_counts": {"deep learning": 12, "PyTorch": 10, ...},
      "defining_technologies": ["PyTorch", "TensorFlow", "XGBoost"],
      "defining_skills": ["deep learning", "statistical analysis", "research"]
    },
    ...
  },
  "total_vacancies": 36,
  "total_unique_keywords": 186,
  "pipeline_stats": {
    "tfidf_keywords": 847,
    "gpt_categorized": 312,
    "embedding_merged": 186,
    "gpt_tokens_used": 12450
  }
}
```

---

### Phase 2: Update Theme Definitions
**Goal:** Replace 5 hardcoded themes with 3 market-aligned variants

**Files to modify:**
- `src/keyword_engine.py` - Update `TECH_TAXONOMY` and `get_resume_themes()`

**Tasks:**
1. Update `TECH_TAXONOMY` dictionary with 3 new categories:
   ```python
   TECH_TAXONOMY = {
       "research_ml": [
           "deep learning", "neural network", "pytorch", "tensorflow",
           "statistical", "optimization", "research", "experiments",
           "xgboost", "catboost", "model performance", "algorithms",
           "classification", "regression", "forecasting", "prophet"
       ],
       "applied_production": [
           "production", "pipeline", "deployment", "mlops", "docker",
           "kubernetes", "ci/cd", "airflow", "mlflow", "sagemaker",
           "a/b testing", "monitoring", "feature engineering", "sql",
           "clickhouse", "infrastructure", "scaling", "automation"
       ],
       "genai_llm": [
           "llm", "langchain", "langgraph", "rag", "embeddings",
           "prompt engineering", "gpt", "vector database", "agents",
           "fastapi", "rest api", "retrieval", "generation",
           "telegram bot", "automated", "chatbot"
       ]
   }
   ```

2. Rewrite `get_resume_themes()` to return 3 theme configs with:
   - name, display_name
   - primary_category, secondary_categories
   - skills_priority (ordered list for skills section)
   - experience_keywords (for bullet reordering)
   - summary_template (theme-specific professional summary)

---

### Phase 3: Resume Generation
**Goal:** Generate 3 optimized LaTeX + PDF resume variants

**Files to modify:**
- `src/resume_generator.py` - Update theme handling and summaries

**Tasks:**
1. Update `generate_all_variants()` to use 3 new themes
2. Update `_enhance_summary()` with new summary templates:
   - **research_ml**: "ML Engineer with 6+ years experience in statistical modeling and deep learning. Proficient in PyTorch, XGBoost, and neural network optimization..."
   - **applied_production**: "ML Engineer with 6+ years experience building production ML systems. Expert in MLOps, Docker, Kubernetes, and end-to-end pipelines..."
   - **genai_llm**: "ML Engineer with 6+ years experience in LLM applications and generative AI. Skilled in LangChain, RAG pipelines, and agentic systems..."

3. Run generation: `python main.py generate`
4. Verify 3 new variants created:
   - `output/resume_research_ml.tex` + `.pdf`
   - `output/resume_applied_production.tex` + `.pdf`
   - `output/resume_genai_llm.tex` + `.pdf`

---

### Phase 4: Update Matching System
**Goal:** Ensure semantic matcher works with 3 new variants

**Files to modify:**
- `src/semantic_matcher.py` - Verify variant discovery

**Tasks:**
1. Delete old variants from `output/` folder (resume_mlops.tex, etc.)
2. Clear embedding cache: delete `.hybrid_embeddings_cache.json`
3. Verify `_load_variants()` discovers new naming convention
4. Regenerate embeddings for 3 new variants
5. Test matching:
   ```bash
   python main.py tailor vacancies/google.txt      # Should match research_ml
   python main.py tailor vacancies/expedia.txt     # Should match applied_production
   python main.py tailor vacancies/docker.txt      # Should match genai_llm
   ```

---

### Phase 5: Code Cleanup
**Goal:** Remove unused legacy code

**Files to delete:**
- `src/matcher.py` (368 lines) - Replaced by semantic_matcher.py

**Files to modify:**
- `main.py` - Remove `match` command that uses legacy matcher
- `backend/services.py` - Remove `rewrite_bullets` parameter from analyze()
- `backend/schemas.py` - Remove unused `ErrorResponse` class

**Files to deprecate (create `deprecated/` folder):**
- `src/linkedin_scraper.py` - ToS-violating, rarely used

---

### Phase 6: Testing & Verification
**Goal:** Verify all components work correctly

**Tests:**
1. Test new clustering command:
   ```bash
   python main.py cluster-vacancies
   ```
2. Generate variants:
   ```bash
   python main.py generate
   ```
3. Test matching against vacancies from each category:
   ```bash
   python main.py tailor vacancies/google.txt      # research_ml
   python main.py tailor vacancies/expedia.txt     # applied_production
   python main.py tailor vacancies/docker.txt      # genai_llm
   ```
4. Start web server and test API:
   ```bash
   python main.py serve
   curl -X POST http://localhost:8000/api/analyze -d '{"job_description": "..."}'
   ```
5. Verify variant PDFs open correctly

---

## File Changes Summary

| File | Action | Changes |
|------|--------|---------|
| `src/vacancy_clustering.py` | **CREATE** | New hybrid clustering pipeline (~300 lines) |
| `src/keyword_engine.py` | Modify | Update TECH_TAXONOMY (3 categories), get_resume_themes() (3 themes), add utility functions |
| `src/resume_generator.py` | Modify | Update summary templates for 3 new themes |
| `main.py` | Modify | Add `cluster-vacancies` command, remove legacy `match` command |
| `src/semantic_matcher.py` | Minor | Verify variant discovery works with new names |
| `backend/services.py` | Modify | Remove deprecated `rewrite_bullets` parameter |
| `backend/schemas.py` | Modify | Remove unused `ErrorResponse` class |
| `src/matcher.py` | Delete | Legacy code (368 lines), replaced by semantic_matcher |
| `src/linkedin_scraper.py` | Move | Move to `deprecated/` folder |
| `.hybrid_embeddings_cache.json` | Delete | Force cache regeneration |
| `output/resume_*.tex` | Delete | Remove old 5 variants, generate 3 new ones |

---

## Verification Checklist

**Phase 1 - Clustering Pipeline:**
- [ ] `cluster-vacancies` CLI command runs without errors
- [ ] Stage 1: TF-IDF + taxonomy extracts keywords from all 36 vacancies
- [ ] Stage 2: GPT categorizes and deduplicates keywords (or skips with `--no-gpt`)
- [ ] Stage 3: Local embeddings cluster similar keywords
- [ ] Stage 4: Vacancies assigned to 3 clusters correctly
- [ ] JSON output saved with `--output clusters.json`

**Phase 2-3 - Resume Generation:**
- [ ] 3 new resume variants generated in `output/`
- [ ] All 3 PDFs compile correctly
- [ ] Each variant has distinct summary and skills ordering

**Phase 4 - Matching System:**
- [ ] Semantic matcher selects correct variant per job type:
  - [ ] google.txt → research_ml
  - [ ] expedia.txt → applied_production
  - [ ] docker.txt → genai_llm
- [ ] Web API `/api/analyze` returns proper results

**Phase 5 - Code Cleanup:**
- [ ] Legacy `match` command removed from CLI
- [ ] `src/matcher.py` deleted
- [ ] `src/linkedin_scraper.py` moved to deprecated/
- [ ] `rewrite_bullets` parameter removed from services.py
