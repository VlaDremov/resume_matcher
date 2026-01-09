"""
FastAPI Backend for Resume Matcher.

Provides REST API endpoints for:
- Analyzing job descriptions
- Matching to resume variants
- Saving vacancies to database
- Serving resume PDFs
"""

import logging
import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    SaveVacancyRequest,
    SaveVacancyResponse,
    VacanciesListResponse,
    VariantsListResponse,
)
from backend.services import (
    analysis_service,
    vacancy_service,
    variant_service,
)

# * Load environment variables
load_dotenv()

# * Configure logging with timestamps
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

logger = logging.getLogger("resume_matcher.api")

# * Create FastAPI app
app = FastAPI(
    title="Resume Keyword Matcher API",
    description="API for analyzing job descriptions and matching to optimized resume variants",
    version="2.0.0",
)

# * Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Resume Keyword Matcher API", "version": "2.0.0"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_job(request: AnalyzeRequest):
    """
    Analyze a job description and find the best resume match.

    Uses hybrid matching with parallel async execution:
    - Local + OpenAI embeddings for semantic similarity (parallel)
    - TF-IDF + GPT for keyword extraction (parallel)
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")

    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    # * Log warning if OpenAI key not set (hybrid mode will use local-only)
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set - using local embeddings only")

    logger.info(
        "Analyze request id=%s semantic=%s job_chars=%s",
        request_id,
        request.use_semantic,
        len(request.job_description),
    )

    try:
        # * Use async analyze for parallel execution of LLM calls
        result = await analysis_service.analyze_async(
            job_description=request.job_description,
            use_semantic=request.use_semantic,
            include_market_trends=request.include_market_trends,
        )
        duration = time.perf_counter() - start
        logger.info(
            "Analyze response id=%s variant=%s matches=%s missing=%s duration=%.3fs",
            request_id,
            result.best_variant,
            len(result.key_matches),
            len(result.missing_keywords),
            duration,
        )
        return result
    except Exception as e:
        duration = time.perf_counter() - start
        logger.error(
            "Analyze failed id=%s duration=%.3fs error=%s",
            request_id,
            duration,
            e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save-vacancy", response_model=SaveVacancyResponse)
async def save_vacancy(request: SaveVacancyRequest):
    """
    Save a job description to the vacancies database.
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")

    if not request.filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    logger.info(
        "Save vacancy request id=%s filename=%s company=%s position=%s chars=%s",
        request_id,
        request.filename,
        request.company,
        request.position,
        len(request.job_description),
    )

    result = vacancy_service.save_vacancy(
        job_description=request.job_description,
        filename=request.filename,
        company=request.company,
        position=request.position,
    )

    if not result.success:
        duration = time.perf_counter() - start
        logger.error(
            "Save vacancy failed id=%s duration=%.3fs message=%s",
            request_id,
            duration,
            result.message,
        )
        raise HTTPException(status_code=500, detail=result.message)

    duration = time.perf_counter() - start
    logger.info(
        "Save vacancy success id=%s duration=%.3fs path=%s",
        request_id,
        duration,
        result.filepath,
    )
    return result


@app.get("/api/vacancies", response_model=VacanciesListResponse)
async def list_vacancies():
    """
    List all saved vacancies.
    """
    vacancies = vacancy_service.list_vacancies()
    logger.info("Vacancies listed count=%s", len(vacancies))
    return VacanciesListResponse(vacancies=vacancies, count=len(vacancies))


@app.get("/api/variants", response_model=VariantsListResponse)
async def list_variants():
    """
    List all available resume variants.
    """
    variants = variant_service.list_variants()
    logger.info("Variants listed count=%s", len(variants))
    return VariantsListResponse(variants=variants, count=len(variants))


@app.get("/api/resume/{variant_name}/pdf")
async def get_resume_pdf(variant_name: str):
    """
    Get the PDF file for a resume variant.
    """
    filepath = variant_service.get_variant_file(variant_name, "pdf")

    if not filepath:
        logger.warning("PDF not found variant=%s", variant_name)
        raise HTTPException(status_code=404, detail=f"PDF not found for variant: {variant_name}")

    logger.info("Serving PDF variant=%s path=%s", variant_name, filepath)
    return FileResponse(
        filepath,
        media_type="application/pdf",
        filename=f"resume_{variant_name}.pdf",
    )


@app.get("/api/resume/{variant_name}/tex")
async def get_resume_tex(variant_name: str):
    """
    Get the LaTeX file for a resume variant.
    """
    filepath = variant_service.get_variant_file(variant_name, "tex")

    if not filepath:
        logger.warning("LaTeX not found variant=%s", variant_name)
        raise HTTPException(status_code=404, detail=f"LaTeX not found for variant: {variant_name}")

    logger.info("Serving LaTeX variant=%s path=%s", variant_name, filepath)
    return FileResponse(
        filepath,
        media_type="text/plain",
        filename=f"resume_{variant_name}.tex",
    )


@app.get("/api/vacancy/{filename}")
async def get_vacancy(filename: str):
    """
    Get content of a specific vacancy file.
    """
    vacancies = vacancy_service.list_vacancies()

    for vacancy in vacancies:
        if vacancy.filename == filename:
            with open(vacancy.filepath, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info("Vacancy served filename=%s bytes=%s", filename, len(content))
            return {
                "filename": vacancy.filename,
                "company": vacancy.company,
                "position": vacancy.position,
                "content": content,
            }

    logger.warning("Vacancy not found filename=%s", filename)
    raise HTTPException(status_code=404, detail=f"Vacancy not found: {filename}")


# * Run with: uvicorn backend.api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
