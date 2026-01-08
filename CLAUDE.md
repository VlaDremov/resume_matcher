# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Resume Keyword Matcher v2.0** - A GPT-5 powered resume optimization tool that generates 5 specialized resume variants and matches them to job descriptions using hybrid AI/ML techniques (semantic embeddings + keyword extraction).

**Tech Stack:**
- Backend: Python 3.10+, FastAPI, OpenAI API (gpt-5-mini, text-embedding-3-small)
- Frontend: React 18, TypeScript, Vite
- ML/NLP: Sentence Transformers (all-MiniLM-L6-v2), spaCy, scikit-learn
- Document: LaTeX (pdflatex), pdfplumber

## Development Commands

### Backend (Python)

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set up environment
export OPENAI_API_KEY="sk-..."  # Required for GPT features
# Or create .env file with OPENAI_API_KEY=sk-...

# Run web server (with auto-reload)
python main.py serve
python main.py serve --port 8080 --no-reload

# Generate resume variants (creates 5 LaTeX + PDF files)
python main.py generate
python main.py generate --no-compile-pdf  # Skip PDF compilation

# Match job to best variant (legacy keyword-based)
python main.py match vacancies/google.txt --explain

# Tailor resume with GPT-5 (semantic + keyword analysis)
python main.py tailor vacancies/google.txt
python main.py tailor --text "Job description..." --preview

# Analyze vacancy keywords
python main.py analyze --top 30

# Extract resume data
python main.py info
```

### Frontend (React)

```bash
cd frontend

# Install dependencies
npm install

# Development server (localhost:5173)
npm run dev

# Build for production
npm run build

# Lint
npm run lint

# Preview production build
npm run preview
```

### LaTeX Compilation

```bash
# macOS
brew install --cask mactex-no-gui

# Ubuntu
sudo apt-get install texlive-latex-base texlive-latex-extra
```

## Architecture

### Core Matching Pipeline

The system uses a **hybrid approach** that combines multiple techniques:

```
Job Description Input
  │
  ├─ Semantic Matching (parallel async)
  │  ├─ Local Embeddings (Sentence Transformers: all-MiniLM-L6-v2)
  │  ├─ OpenAI Embeddings (text-embedding-3-small)
  │  └─ Weighted combination (default: 0.4 local + 0.6 OpenAI)
  │
  ├─ Keyword Extraction (parallel async)
  │  ├─ Local: TF-IDF + Taxonomy (5 predefined tech categories)
  │  ├─ GPT: Structured JSON extraction (gpt-5-mini)
  │  └─ Merged & deduplicated results
  │
  └─ Output: Best variant + scores + matched/missing keywords
```

### Key Architectural Patterns

1. **Hybrid Embeddings** (`src/hybrid_embeddings.py`)
   - Runs local + OpenAI embeddings in parallel via `asyncio.gather()`
   - Local embedding runs in thread pool (CPU-bound)
   - OpenAI embedding is async (I/O-bound)
   - Gracefully falls back to local-only if OpenAI unavailable
   - Configurable weights via environment variables

2. **Lazy Loading** (`backend/services.py`)
   - Services instantiate expensive modules (SemanticMatcher, KeywordExtractor) only on first use
   - Reduces startup time and memory footprint

3. **Embedding Caching**
   - Resume variant embeddings cached as JSON (`.hybrid_embeddings_cache.json`)
   - Speeds up subsequent analyses by ~10x
   - Cache invalidated if variant files change

4. **Async-First API** (`backend/api.py`)
   - All endpoints use async handlers
   - Enables parallel execution of:
     - Local + OpenAI embeddings
     - Local + GPT keyword extraction
   - Reduces latency by 50-70% vs sequential execution

5. **Technology Taxonomy** (`src/keyword_engine.py`)
   - Hardcoded 5-category taxonomy:
     - MLOps & Platform Engineering
     - NLP & LLM Engineering
     - Cloud & AWS Infrastructure
     - Data Engineering & Pipelines
     - Classical ML & Analytics
   - Domain-specific keywords outperform generic NLP for job matching

### Module Responsibilities

**Entry Points:**
- `main.py` - CLI with click commands (generate, match, tailor, analyze, serve, etc.)
- `backend/api.py` - FastAPI REST API (CORS enabled for localhost:3000, :5173)

**Core Matching:**
- `src/semantic_matcher.py` - Hybrid semantic similarity (local + OpenAI embeddings)
- `src/hybrid_embeddings.py` - Orchestrates dual embedding sources with async parallelism
- `src/local_embeddings.py` - Sentence Transformers wrapper
- `src/hybrid_keywords.py` - Combines TF-IDF + GPT keyword extraction (async parallel)
- `src/keyword_engine.py` - Baseline TF-IDF + spaCy NER + taxonomy matching

**Resume Generation:**
- `src/resume_generator.py` - Creates 5 LaTeX variants by:
  - Reordering skills section per theme
  - Enhancing bullets with theme keywords
  - Adjusting professional summary
- `src/bullet_rewriter.py` - GPT-5 powered bullet point optimization

**Utilities:**
- `src/llm_client.py` - OpenAI API wrapper with retries, structured JSON, token logging
- `src/data_extraction.py` - LaTeX/PDF parsers
- `src/latex_compiler.py` - pdflatex wrapper

**Backend Services:**
- `backend/services.py` - Three service singletons:
  - `ResumeAnalysisService` - Lazy-loads matcher & extractor
  - `VacancyService` - File I/O for job descriptions
  - `ResumeVariantService` - Lists available variants
- `backend/schemas.py` - Pydantic models for request/response validation

### Data Flow Example: Web Analysis

```
React UI (POST /api/analyze with job description)
  ↓
backend/api.py: analyze_job()
  ↓
services.py: analysis_service.analyze_async()
  ↓ (parallel execution via asyncio.gather)
  ├─ semantic_matcher.match_async()
  │  ├─ hybrid_embeddings: local + OpenAI (parallel)
  │  └─ cosine similarity with 5 cached variant embeddings
  │
  ├─ hybrid_keywords.extract_keywords_hybrid_async()
  │  ├─ keyword_engine: TF-IDF + taxonomy (in thread pool)
  │  └─ llm_client: GPT structured extraction (async)
  │
  └─ Merge results into AnalyzeResponse
     ├─ best_variant (e.g., "nlp_llm")
     ├─ category_scores (all 5 variants ranked)
     ├─ key_matches (job keywords in resume)
     └─ missing_keywords (important gaps)
```

## Configuration

### Environment Variables (.env)

```bash
# Required for GPT features
OPENAI_API_KEY=sk-...

# Embedding weights (default: 0.4 local + 0.6 OpenAI)
LOCAL_EMBEDDING_WEIGHT=0.4
OPENAI_EMBEDDING_WEIGHT=0.6

# Feature flags
USE_GPT_KEYWORDS=true  # Enable/disable GPT keyword extraction
```

### Model Specifications

- Chat: `gpt-5-mini` (with `reasoning_effort: low`)
- Embedding: `text-embedding-3-small`
- Local embedding: `all-MiniLM-L6-v2` (Sentence Transformers)
- NER: `en_core_web_sm` (spaCy)

## File Structure

```
resume_matcher/
├── main.py                      # CLI entry point
├── resume.tex                   # Original resume (LaTeX)
├── .env                         # API keys & config
├── requirements.txt             # Python dependencies
│
├── src/                         # Core logic
│   ├── semantic_matcher.py      # Hybrid semantic matching
│   ├── hybrid_embeddings.py     # Local + OpenAI embeddings (async)
│   ├── hybrid_keywords.py       # Local + GPT keywords (async)
│   ├── keyword_engine.py        # TF-IDF + taxonomy baseline
│   ├── resume_generator.py      # LaTeX variant generation
│   ├── bullet_rewriter.py       # GPT-5 bullet optimization
│   ├── llm_client.py            # OpenAI API wrapper
│   ├── local_embeddings.py      # Sentence Transformers
│   ├── data_extraction.py       # PDF/LaTeX parsing
│   ├── latex_compiler.py        # pdflatex wrapper
│   ├── matcher.py               # Legacy keyword-based matching
│   └── linkedin_scraper.py      # LinkedIn scraping (optional)
│
├── backend/                     # FastAPI REST API
│   ├── api.py                   # Endpoints + CORS
│   ├── services.py              # Business logic (lazy loading)
│   └── schemas.py               # Pydantic models
│
├── frontend/                    # React + TypeScript
│   ├── src/
│   │   ├── App.tsx              # Main component
│   │   └── api.ts               # Fetch-based API client
│   └── package.json
│
├── output/                      # Generated variants (LaTeX + PDF)
├── vacancies/                   # Job description files
└── .gitignore                   # Excludes __pycache__, node_modules, cache
```

## Important Implementation Notes

### When Adding New Features

1. **Async Execution** - If calling OpenAI API or local embeddings, use async patterns from `hybrid_embeddings.py` or `hybrid_keywords.py` as reference. Run independent operations in parallel with `asyncio.gather()`.

2. **Caching** - Expensive computations (embeddings, GPT responses) should be cached. See `.hybrid_embeddings_cache.json` pattern in `semantic_matcher.py`.

3. **Graceful Degradation** - Always provide fallback for missing OpenAI key. Check `os.getenv("OPENAI_API_KEY")` and fall back to local-only mode.

4. **Structured LLM Output** - Use Pydantic models for GPT responses (see `llm_client.py` structured output pattern). This ensures reliable JSON parsing.

5. **Logging** - Use structured logging with request IDs (see `backend/api.py`). Include timing info for performance monitoring.

### When Modifying Resume Generation

- The 5 variants are **theme-based**, not dynamically generated per job
- Themes: `mlops_platform`, `nlp_llm`, `cloud_aws`, `data_engineering`, `classical_ml`
- Variant generation (`resume_generator.py`) reorders skills and enhances bullets, but maintains the original experience structure
- To add a new theme, update:
  1. `keyword_engine.py` - Add to `KEYWORD_TAXONOMY`
  2. `resume_generator.py` - Add to variant generation logic

### When Working with LaTeX

- Resume variants are `.tex` files compiled to PDF via `pdflatex`
- Text extraction for embeddings strips LaTeX commands (see `_extract_text_content()` in `semantic_matcher.py`)
- Bullet points use `\item` tags (parsed by `extract_bullets_from_latex()` in `bullet_rewriter.py`)

### When Debugging

- Check `backend/api.py` logs for request IDs and timing
- Embedding cache may be stale - delete `.hybrid_embeddings_cache.json` to regenerate
- If OpenAI calls fail, system falls back to local-only (check logs for warnings)
- spaCy model must be downloaded: `python -m spacy download en_core_web_sm`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/analyze` | POST | Analyze job, return best variant + keywords |
| `POST /api/save-vacancy` | POST | Save job description to `vacancies/` |
| `GET /api/vacancies` | GET | List saved vacancies |
| `GET /api/variants` | GET | List resume variants |
| `GET /api/resume/{variant}/pdf` | GET | Download variant PDF |
| `GET /api/resume/{variant}/tex` | GET | Download variant LaTeX |

## Testing

Currently no automated tests. When adding tests:
- Use `pytest` for Python tests
- Test async functions with `pytest-asyncio`
- Mock OpenAI API calls to avoid costs
- Focus on: keyword extraction, embedding caching, variant generation
