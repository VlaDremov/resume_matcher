"""
Pydantic Schemas for API Request/Response Models.
"""

from typing import Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for the analyze endpoint."""

    job_description: str = Field(..., description="The job description text to analyze")
    use_semantic: bool = Field(default=True, description="Use semantic matching with embeddings")
    rewrite_bullets: bool = Field(default=True, description="Rewrite bullets using GPT-5")


class CategoryScore(BaseModel):
    """Score for a single category."""

    category: str
    score: float
    display_name: str


class RewrittenBullet(BaseModel):
    """A rewritten bullet point."""

    original: str
    rewritten: str
    keywords_added: list[str]
    confidence: float


class AnalyzeResponse(BaseModel):
    """Response from the analyze endpoint."""

    relevancy_score: int = Field(..., ge=0, le=100, description="Overall relevancy score 0-100")
    best_variant: str = Field(..., description="Name of the best matching variant")
    best_variant_display: str = Field(..., description="Display name of the best variant")
    category_scores: list[CategoryScore] = Field(..., description="Scores by category")
    key_matches: list[str] = Field(..., description="Keywords matched from job")
    missing_keywords: list[str] = Field(default=[], description="Important missing keywords")
    rewritten_bullets: list[RewrittenBullet] = Field(default=[], description="Rewritten bullets")
    reasoning: str = Field(default="", description="Analysis reasoning")
    tex_path: Optional[str] = Field(None, description="Path to LaTeX file")
    pdf_path: Optional[str] = Field(None, description="Path to PDF file")


class SaveVacancyRequest(BaseModel):
    """Request body for saving a vacancy."""

    job_description: str = Field(..., description="The job description text")
    filename: str = Field(..., description="Filename to save as (without extension)")
    company: Optional[str] = Field(None, description="Company name")
    position: Optional[str] = Field(None, description="Position title")


class SaveVacancyResponse(BaseModel):
    """Response from save vacancy endpoint."""

    success: bool
    filepath: str
    message: str


class VacancyInfo(BaseModel):
    """Information about a saved vacancy."""

    filename: str
    filepath: str
    company: Optional[str] = None
    position: Optional[str] = None
    preview: str = Field(..., description="First 200 chars of content")


class VacanciesListResponse(BaseModel):
    """Response listing all vacancies."""

    vacancies: list[VacancyInfo]
    count: int


class ResumeVariantInfo(BaseModel):
    """Information about a resume variant."""

    name: str
    display_name: str
    description: str
    tex_exists: bool
    pdf_exists: bool
    tex_path: Optional[str] = None
    pdf_path: Optional[str] = None


class VariantsListResponse(BaseModel):
    """Response listing all resume variants."""

    variants: list[ResumeVariantInfo]
    count: int


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None

