"""
FastAPI Backend for Resume Matcher.

Provides REST API endpoints for:
- Analyzing job descriptions
- Matching to resume variants
- Saving vacancies to database
- Serving resume PDFs
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
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

    Returns relevancy score, best variant, and optionally rewritten bullets.
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")

    # * Check if OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured. Set it in .env file or environment.",
        )

    try:
        result = analysis_service.analyze(
            job_description=request.job_description,
            use_semantic=request.use_semantic,
            rewrite_bullets=request.rewrite_bullets,
        )
        return result
    except Exception as e:
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

    result = vacancy_service.save_vacancy(
        job_description=request.job_description,
        filename=request.filename,
        company=request.company,
        position=request.position,
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)

    return result


@app.get("/api/vacancies", response_model=VacanciesListResponse)
async def list_vacancies():
    """
    List all saved vacancies.
    """
    vacancies = vacancy_service.list_vacancies()
    return VacanciesListResponse(vacancies=vacancies, count=len(vacancies))


@app.get("/api/variants", response_model=VariantsListResponse)
async def list_variants():
    """
    List all available resume variants.
    """
    variants = variant_service.list_variants()
    return VariantsListResponse(variants=variants, count=len(variants))


@app.get("/api/resume/{variant_name}/pdf")
async def get_resume_pdf(variant_name: str):
    """
    Get the PDF file for a resume variant.
    """
    filepath = variant_service.get_variant_file(variant_name, "pdf")

    if not filepath:
        raise HTTPException(status_code=404, detail=f"PDF not found for variant: {variant_name}")

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
        raise HTTPException(status_code=404, detail=f"LaTeX not found for variant: {variant_name}")

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
            return {
                "filename": vacancy.filename,
                "company": vacancy.company,
                "position": vacancy.position,
                "content": content,
            }

    raise HTTPException(status_code=404, detail=f"Vacancy not found: {filename}")


# * Run with: uvicorn backend.api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

