"""
Pydantic Schemas for API Request/Response Models.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# * Keyword category type
KeywordCategory = Literal[
    "mlops", "nlp_llm", "cloud_aws", "data_engineering", "classical_ml", "other"
]

# * Keyword importance level
KeywordImportance = Literal["critical", "important", "nice_to_have"]


class KeywordWithMetadata(BaseModel):
    """A keyword with category and importance metadata."""

    keyword: str = Field(..., description="The keyword text")
    category: KeywordCategory = Field(..., description="Tech category")
    importance: KeywordImportance = Field(..., description="Importance level")
    is_matched: bool = Field(default=False, description="Whether keyword is in resume")
    is_trending: bool = Field(default=False, description="Whether skill is trending in market")
    demand_level: Optional[str] = Field(None, description="Market demand: high/medium/low")


class CategorizedKeywords(BaseModel):
    """Keywords grouped by tech category."""

    mlops: list[KeywordWithMetadata] = Field(default_factory=list)
    nlp_llm: list[KeywordWithMetadata] = Field(default_factory=list)
    cloud_aws: list[KeywordWithMetadata] = Field(default_factory=list)
    data_engineering: list[KeywordWithMetadata] = Field(default_factory=list)
    classical_ml: list[KeywordWithMetadata] = Field(default_factory=list)
    other: list[KeywordWithMetadata] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    """Request body for the analyze endpoint."""

    job_description: str = Field(..., description="The job description text to analyze")
    use_semantic: bool = Field(default=True, description="Use semantic matching with embeddings")
    include_market_trends: bool = Field(default=False, description="Include current job market trends in response")


class CategoryScore(BaseModel):
    """Score for a single resume category/variant."""

    category: str
    score: float
    display_name: str


class TrendingSkillInfo(BaseModel):
    """A skill with market demand metadata."""

    skill: str
    category: str
    demand_level: str
    trend: str


class MarketTrendsInfo(BaseModel):
    """Market trends summary for display."""

    trending_skills: list[TrendingSkillInfo] = Field(default_factory=list)
    emerging_technologies: list[str] = Field(default_factory=list)
    industry_insights: str = ""
    last_updated: str = ""


class AnalyzeResponse(BaseModel):
    """Response from the analyze endpoint."""

    best_variant: str = Field(..., description="Name of the best matching variant")
    best_variant_display: str = Field(..., description="Display name of the best variant")
    category_scores: list[CategoryScore] = Field(..., description="Scores by category")

    # * Rich keyword data (new)
    categorized_matches: CategorizedKeywords = Field(
        default_factory=CategorizedKeywords,
        description="Matched keywords grouped by category with importance",
    )
    categorized_missing: CategorizedKeywords = Field(
        default_factory=CategorizedKeywords,
        description="Missing keywords grouped by category with importance",
    )

    # * Market trends (optional)
    market_trends: Optional[MarketTrendsInfo] = Field(
        default=None,
        description="Current job market trends for context",
    )

    # * Legacy fields (kept for backward compatibility)
    key_matches: list[str] = Field(default=[], description="Keywords matched from job (deprecated)")
    missing_keywords: list[str] = Field(default=[], description="Important missing keywords (deprecated)")


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
