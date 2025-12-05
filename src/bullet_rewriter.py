"""
Bullet Point Rewriter Module.

Uses GPT-5 to intelligently rewrite resume bullet points
to emphasize job-relevant keywords while preserving factual accuracy.
"""

import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.llm_client import LLMClient, get_client


class RewrittenBullet(BaseModel):
    """A single rewritten bullet point."""

    original: str = Field(description="The original bullet point text")
    rewritten: str = Field(description="The rewritten bullet point optimized for the job")
    keywords_added: list[str] = Field(description="Keywords from job description incorporated")
    confidence: float = Field(description="Confidence score 0-1 that rewrite is accurate")


class AnalysisResult(BaseModel):
    """Complete analysis result with relevancy scoring."""

    relevancy_score: int = Field(description="Overall relevancy score 0-100", ge=0, le=100)
    best_variant: str = Field(description="Recommended resume variant name")
    category_scores: dict[str, float] = Field(description="Scores for each category")
    key_matches: list[str] = Field(description="Key skills/keywords matched from job")
    missing_keywords: list[str] = Field(description="Important keywords not in resume")
    rewritten_bullets: list[RewrittenBullet] = Field(description="Rewritten bullet points")
    reasoning: str = Field(description="Explanation of the analysis")


class BulletRewriter:
    """
    GPT-5 powered bullet point rewriter.

    Analyzes job descriptions and rewrites resume bullets to
    naturally incorporate relevant keywords while maintaining
    factual accuracy.
    """

    def __init__(self, client: Optional[LLMClient] = None):
        """
        Initialize the bullet rewriter.

        Args:
            client: Optional LLM client. Creates one if not provided.
        """
        self.client = client or get_client()

    def analyze_and_rewrite(
        self,
        job_description: str,
        resume_bullets: list[str],
        resume_variant: str = "general",
    ) -> AnalysisResult:
        """
        Analyze job description and rewrite bullets with keyword emphasis.

        Args:
            job_description: Full job description text.
            resume_bullets: List of current resume bullet points.
            resume_variant: Name of the resume variant being used.

        Returns:
            AnalysisResult with scores and rewritten bullets.
        """
        system_prompt = """You are an expert resume optimizer specializing in ML Engineer, 
Data Scientist, Applied Scientist, and AI Engineer positions.

Your task is to:
1. Analyze the job description to identify key requirements and keywords
2. Score how relevant the resume content is to this job (0-100)
3. Identify which category best matches: mlops, nlp_llm, cloud_aws, data_engineering, classical_ml
4. Rewrite bullet points to naturally incorporate relevant keywords

IMPORTANT RULES:
- NEVER fabricate achievements or skills - only rephrase existing content
- Preserve all quantified metrics (percentages, dollar amounts, timeframes)
- Keep the same meaning while emphasizing job-relevant aspects
- Use action verbs and professional language
- Each rewritten bullet should be similar length to original"""

        prompt = f"""Job Description:
{job_description}

Current Resume Bullets:
{self._format_bullets(resume_bullets)}

Resume Variant: {resume_variant}

Analyze this job and rewrite the bullets to better match the job requirements.
Return your analysis as structured JSON."""

        try:
            result = self.client.chat_structured(
                prompt=prompt,
                response_model=AnalysisResult,
                system_prompt=system_prompt,
                temperature=0.3,
            )
            return result
        except Exception as e:
            print(f"! Analysis failed: {e}")
            # * Return a default result on failure
            return AnalysisResult(
                relevancy_score=50,
                best_variant=resume_variant,
                category_scores={},
                key_matches=[],
                missing_keywords=[],
                rewritten_bullets=[],
                reasoning=f"Analysis failed: {e}",
            )

    def _format_bullets(self, bullets: list[str]) -> str:
        """Format bullet list for prompt."""
        return "\n".join(f"- {bullet}" for bullet in bullets)

    def rewrite_single_bullet(
        self,
        bullet: str,
        job_keywords: list[str],
    ) -> RewrittenBullet:
        """
        Rewrite a single bullet point to incorporate keywords.

        Args:
            bullet: Original bullet point text.
            job_keywords: Keywords to incorporate from job description.

        Returns:
            RewrittenBullet with original and rewritten versions.
        """
        system_prompt = """You are a resume optimization expert.
Rewrite the bullet point to naturally incorporate relevant keywords.

RULES:
- Preserve all factual content and metrics
- Only rephrase, never fabricate
- Keep similar length
- Use professional action verbs"""

        prompt = f"""Original bullet: {bullet}

Keywords to emphasize if relevant: {', '.join(job_keywords)}

Rewrite this bullet to better match these keywords while preserving accuracy."""

        try:
            result = self.client.chat_structured(
                prompt=prompt,
                response_model=RewrittenBullet,
                system_prompt=system_prompt,
                temperature=0.3,
            )
            return result
        except Exception as e:
            print(f"! Bullet rewrite failed: {e}")
            return RewrittenBullet(
                original=bullet,
                rewritten=bullet,
                keywords_added=[],
                confidence=0.0,
            )

    def extract_keywords_from_job(self, job_description: str) -> list[str]:
        """
        Extract key technical keywords from a job description.

        Args:
            job_description: Job description text.

        Returns:
            List of important keywords.
        """

        class KeywordExtraction(BaseModel):
            technical_skills: list[str] = Field(description="Technical skills mentioned")
            tools_and_frameworks: list[str] = Field(description="Tools and frameworks")
            soft_skills: list[str] = Field(description="Soft skills and competencies")
            domain_keywords: list[str] = Field(description="Domain-specific terms")

        system_prompt = """Extract the most important keywords from this job description.
Focus on:
- Technical skills (Python, ML, etc.)
- Tools and frameworks (TensorFlow, Docker, etc.)
- Methodologies (Agile, MLOps, etc.)
- Domain terms (NLP, Computer Vision, etc.)"""

        try:
            result = self.client.chat_structured(
                prompt=f"Job Description:\n{job_description}",
                response_model=KeywordExtraction,
                system_prompt=system_prompt,
                temperature=0.2,
            )

            # * Combine all keywords
            all_keywords = (
                result.technical_skills
                + result.tools_and_frameworks
                + result.soft_skills
                + result.domain_keywords
            )

            # * Remove duplicates while preserving order
            seen = set()
            unique = []
            for kw in all_keywords:
                kw_lower = kw.lower()
                if kw_lower not in seen:
                    seen.add(kw_lower)
                    unique.append(kw)

            return unique

        except Exception as e:
            print(f"! Keyword extraction failed: {e}")
            return []


def extract_bullets_from_latex(latex_content: str) -> list[str]:
    """
    Extract bullet points from LaTeX resume content.

    Args:
        latex_content: Raw LaTeX content.

    Returns:
        List of bullet point texts.
    """
    # * Pattern to match \resumeItem{...}
    pattern = re.compile(r"\\resumeItem\{(.+?)\}(?=\s*\\resumeItem|\s*\\resumeItemListEnd)", re.DOTALL)

    bullets = []
    for match in pattern.finditer(latex_content):
        bullet_text = match.group(1).strip()
        # * Clean up LaTeX formatting
        bullet_text = _clean_latex_text(bullet_text)
        if bullet_text:
            bullets.append(bullet_text)

    return bullets


def _clean_latex_text(text: str) -> str:
    """Remove LaTeX formatting from text."""
    # * Remove href commands but keep display text
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)

    # * Remove textbf
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)

    # * Remove emph/textit
    text = re.sub(r"\\(?:emph|textit)\{([^}]*)\}", r"\1", text)

    # * Remove percentage escapes
    text = text.replace("\\%", "%")
    text = text.replace("\\$", "$")

    # * Clean up multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def apply_rewrites_to_latex(
    latex_content: str,
    rewrites: list[RewrittenBullet],
) -> str:
    """
    Apply rewritten bullets back to LaTeX content.

    Args:
        latex_content: Original LaTeX content.
        rewrites: List of RewrittenBullet objects.

    Returns:
        Updated LaTeX content with rewritten bullets.
    """
    result = latex_content

    for rewrite in rewrites:
        if rewrite.confidence < 0.5:
            # * Skip low-confidence rewrites
            continue

        # * Find and replace the original bullet
        # * Need to handle LaTeX escaping
        original_escaped = _escape_for_latex(rewrite.original)
        rewritten_escaped = _escape_for_latex(rewrite.rewritten)

        # * Try to find the original in the content
        # * This is approximate since we cleaned the text
        pattern = re.compile(
            r"(\\resumeItem\{)([^}]*" + re.escape(rewrite.original[:30]) + r"[^}]*)(\})",
            re.DOTALL,
        )

        match = pattern.search(result)
        if match:
            # * Replace with rewritten version
            result = result[:match.start()] + match.group(1) + rewritten_escaped + match.group(3) + result[match.end():]

    return result


def _escape_for_latex(text: str) -> str:
    """Escape special characters for LaTeX."""
    # * Escape special LaTeX characters
    text = text.replace("%", "\\%")
    text = text.replace("$", "\\$")
    text = text.replace("&", "\\&")
    text = text.replace("#", "\\#")

    return text

