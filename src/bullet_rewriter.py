"""
GPT-Powered Bullet Point Rewriter.

Uses GPT to genuinely rewrite resume content for each theme,
creating meaningfully different variants instead of just reordering.
"""

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("resume_matcher.bullet_rewriter")


# * Pydantic models for structured GPT output


class RewrittenBullet(BaseModel):
    """Single rewritten bullet point."""

    original: str = Field(description="Original bullet text")
    rewritten: str = Field(description="Theme-optimized rewritten bullet")
    keywords_added: list[str] = Field(
        default_factory=list,
        description="Theme-relevant keywords naturally incorporated",
    )


class BulletRewriteResponse(BaseModel):
    """GPT response for bullet rewriting."""

    bullets: list[RewrittenBullet] = Field(description="List of rewritten bullets")


class SummaryRewriteResponse(BaseModel):
    """GPT response for professional summary rewriting."""

    summary: str = Field(description="Theme-optimized professional summary")
    emphasis_points: list[str] = Field(
        default_factory=list,
        description="Key themes emphasized in the summary",
    )


# * Cache file path
CACHE_PATH = Path("output/.bullet_rewrite_cache.json")


class BulletRewriter:
    """
    GPT-powered bullet point and content rewriter.

    Transforms generic resume content into theme-specific variants
    by rewriting bullets, summaries, and skill emphasis.
    """

    def __init__(self, llm_client=None, use_cache: bool = True):
        """
        Initialize the bullet rewriter.

        Args:
            llm_client: Optional pre-configured LLM client.
            use_cache: Whether to read/write cached rewrites.
        """
        self._llm_client = llm_client
        self._cache: dict[str, dict] = {}
        self._use_cache = use_cache
        self._load_cache()

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            try:
                from src.llm_client import get_client

                self._llm_client = get_client()
            except Exception as e:
                logger.warning("LLM client not available: %s", e)
        return self._llm_client

    def _load_cache(self):
        """Load rewrite cache from disk."""
        if not self._use_cache:
            return
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info("Loaded bullet rewrite cache: %d entries", len(self._cache))
            except Exception as e:
                logger.warning("Could not load cache: %s", e)
                self._cache = {}

    def _save_cache(self):
        """Save rewrite cache to disk."""
        if not self._use_cache:
            return
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
            logger.debug("Saved bullet rewrite cache: %d entries", len(self._cache))
        except Exception as e:
            logger.warning("Could not save cache: %s", e)

    def _get_cache_key(self, content: str, theme: str) -> str:
        """Generate cache key from content and theme."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"{theme}:{content_hash}"

    def _build_bullet_prompt(
        self,
        bullets: list[str],
        theme_name: str,
        theme_config: dict,
    ) -> tuple[str, str]:
        """Build prompts for bullet rewriting."""
        theme_keywords = theme_config.get("keywords", [])
        experience_keywords = theme_config.get("experience_keywords", [])

        system_prompt = f"""You are an expert resume writer specializing in {theme_config.get('name', theme_name)} roles.

Your task is to rewrite resume bullet points to better target {theme_config.get('name', theme_name)} positions.

Theme focus keywords: {', '.join(theme_keywords[:15])}
Experience keywords: {', '.join(experience_keywords[:10])}

Guidelines:
1. PRESERVE the core achievement and metrics (numbers, percentages, dollar amounts, timeframes)
2. Naturally incorporate 1-2 theme-relevant keywords where appropriate
3. Emphasize aspects most relevant to {theme_config.get('name', theme_name)}
4. Keep bullets concise (1-2 lines)
5. Use strong action verbs
6. Do NOT fabricate or exaggerate achievements
7. Do NOT change the fundamental meaning of any bullet
8. If a bullet is not relevant to the theme, make minimal changes
9. IMPORTANT: Avoid special characters that break LaTeX: &, %, $, #, _, {{, }}. Use "and" instead of "&"."""

        bullets_text = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(bullets))
        user_prompt = f"""Rewrite these resume bullets for a {theme_config.get('name', theme_name)} position.

IMPORTANT: Return a JSON object with a "bullets" array. Each bullet must be an object with:
- "original": the original bullet text
- "rewritten": your rewritten version
- "keywords_added": list of theme keywords you incorporated

Example format:
{{
  "bullets": [
    {{"original": "Built X", "rewritten": "Engineered X for MLOps", "keywords_added": ["MLOps"]}},
    {{"original": "Led Y", "rewritten": "Led Y deployment", "keywords_added": []}}
  ]
}}

Original bullets to rewrite:
{bullets_text}

Rewrite each bullet to better target {theme_config.get('name', theme_name)} roles while preserving accuracy."""

        return system_prompt, user_prompt

    def _build_summary_prompt(
        self,
        summary: str,
        theme_name: str,
        theme_config: dict,
    ) -> tuple[str, str]:
        """Build prompts for summary rewriting."""
        theme_keywords = theme_config.get("keywords", [])
        system_prompt = f"""You are an expert resume writer specializing in {theme_config.get('name', theme_name)}.

Rewrite the professional summary to target {theme_config.get('name', theme_name)} positions.

Guidelines:
1. Keep it 2-3 concise sentences
2. Emphasize relevant experience and skills for {theme_config.get('name', theme_name)}
3. Maintain a professional tone
4. Do NOT fabricate experience
5. Highlight transferable skills where relevant
6. Where natural, incorporate 1-2 of these focus keywords: {', '.join(theme_keywords[:8])}"""

        user_prompt = f"""Original summary:
{summary}

Rewrite for {theme_config.get('name', theme_name)} roles."""

        return system_prompt, user_prompt

    async def rewrite_bullets_async(
        self,
        bullets: list[str],
        theme_name: str,
        theme_config: dict,
    ) -> list[RewrittenBullet]:
        """
        Rewrite bullets for a theme (async).

        Args:
            bullets: Original bullet texts (without LaTeX formatting).
            theme_name: Theme identifier.
            theme_config: Theme configuration derived from a cluster artifact.

        Returns:
            List of RewrittenBullet with original and rewritten text.
        """
        if not bullets:
            return []

        # * Check cache first
        cache_key = self._get_cache_key("\n".join(bullets), theme_name)
        if self._use_cache and cache_key in self._cache:
            logger.info("Using cached bullet rewrites for theme %s", theme_name)
            cached = self._cache[cache_key]
            return [RewrittenBullet(**item) for item in cached]

        if self.llm_client is None:
            logger.warning("LLM client not available, returning original bullets")
            return [
                RewrittenBullet(original=b, rewritten=b, keywords_added=[])
                for b in bullets
            ]

        system_prompt, user_prompt = self._build_bullet_prompt(
            bullets, theme_name, theme_config
        )

        try:
            result = await self.llm_client.chat_structured_async(
                prompt=user_prompt,
                response_model=BulletRewriteResponse,
                system_prompt=system_prompt,
            )

            # * Cache the results
            if self._use_cache:
                self._cache[cache_key] = [item.model_dump() for item in result.bullets]
                self._save_cache()

            logger.info(
                "Rewrote %d bullets for theme %s",
                len(result.bullets),
                theme_name,
            )
            return result.bullets

        except Exception as e:
            logger.error("Bullet rewriting failed: %s", e)
            # * Fallback: return originals unchanged
            return [
                RewrittenBullet(original=b, rewritten=b, keywords_added=[])
                for b in bullets
            ]

    def rewrite_bullets(
        self,
        bullets: list[str],
        theme_name: str,
        theme_config: dict,
    ) -> list[RewrittenBullet]:
        """Sync wrapper for rewrite_bullets_async."""
        return asyncio.run(
            self.rewrite_bullets_async(bullets, theme_name, theme_config)
        )

    async def rewrite_summary_async(
        self,
        summary: str,
        theme_name: str,
        theme_config: dict,
    ) -> SummaryRewriteResponse:
        """
        Rewrite professional summary for a theme (async).

        Args:
            summary: Original summary text.
            theme_name: Theme identifier.
            theme_config: Theme configuration.

        Returns:
            SummaryRewriteResponse with rewritten summary.
        """
        # * Check cache
        cache_key = self._get_cache_key(f"summary:{summary}", theme_name)
        if self._use_cache and cache_key in self._cache:
            logger.info("Using cached summary rewrite for theme %s", theme_name)
            return SummaryRewriteResponse(**self._cache[cache_key])

        if self.llm_client is None:
            logger.warning("LLM client not available, returning original summary")
            return SummaryRewriteResponse(summary=summary, emphasis_points=[])

        system_prompt, user_prompt = self._build_summary_prompt(
            summary, theme_name, theme_config
        )

        try:
            result = await self.llm_client.chat_structured_async(
                prompt=user_prompt,
                response_model=SummaryRewriteResponse,
                system_prompt=system_prompt,
            )

            # * Cache the result
            if self._use_cache:
                self._cache[cache_key] = result.model_dump()
                self._save_cache()

            logger.info("Rewrote summary for theme %s", theme_name)
            return result

        except Exception as e:
            logger.error("Summary rewriting failed: %s", e)
            return SummaryRewriteResponse(summary=summary, emphasis_points=[])

    def rewrite_summary(
        self,
        summary: str,
        theme_name: str,
        theme_config: dict,
    ) -> SummaryRewriteResponse:
        """Sync wrapper for rewrite_summary_async."""
        return asyncio.run(
            self.rewrite_summary_async(summary, theme_name, theme_config)
        )

    def clear_cache(self):
        """Clear the rewrite cache."""
        self._cache = {}
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
        logger.info("Cleared bullet rewrite cache")


def escape_latex_text(text: str) -> str:
    """
    Escape LaTeX special characters in text.

    Args:
        text: Text that may contain LaTeX special characters.

    Returns:
        Text with special characters properly escaped for LaTeX.
    """
    # * Escape LaTeX special characters in order
    # * Important: Do these replacements first (before braces)
    # * & must become \&
    text = text.replace("&", "\\&")
    # * % must become \%
    text = text.replace("%", "\\%")
    # * $ must become \$
    text = text.replace("$", "\\$")
    # * # must become \#
    text = text.replace("#", "\\#")
    # * _ must become \_
    text = text.replace("_", "\\_")
    
    # * Escape { and } - only if not already escaped
    # * Since GPT is instructed to avoid LaTeX special characters, 
    # * the text should be clean, but we escape to be safe
    # * Replace { with \{ (but not \{)
    text = text.replace("{", "\\{")
    # * Replace } with \} (but not \})
    text = text.replace("}", "\\}")
    
    return text


def extract_bullets_from_latex(content: str) -> list[str]:
    """
    Extract bullet point texts from LaTeX resume content.

    Args:
        content: LaTeX resume content.

    Returns:
        List of bullet point texts (without LaTeX formatting).
    """
    bullets = []

    # * Match \resumeItem{...} patterns
    item_pattern = re.compile(r"\\resumeItem\{(.+?)\}", re.DOTALL)

    for match in item_pattern.finditer(content):
        bullet_text = match.group(1).strip()
        # * Clean up LaTeX artifacts
        bullet_text = re.sub(r"\\textbf\{([^}]+)\}", r"\1", bullet_text)
        bullet_text = re.sub(r"\\emph\{([^}]+)\}", r"\1", bullet_text)
        bullet_text = re.sub(r"\\href\{[^}]+\}\{([^}]+)\}", r"\1", bullet_text)
        # * Also unescape any LaTeX special characters that were escaped
        bullet_text = bullet_text.replace("\\&", "&")
        bullet_text = bullet_text.replace("\\%", "%")
        bullet_text = bullet_text.replace("\\$", "$")
        bullet_text = bullet_text.replace("\\#", "#")
        bullet_text = bullet_text.replace("\\_", "_")
        bullet_text = bullet_text.replace("\\{", "{")
        bullet_text = bullet_text.replace("\\}", "}")
        bullets.append(bullet_text)

    return bullets


def extract_summary_from_latex(content: str) -> str:
    """
    Extract professional summary from LaTeX resume content.

    Args:
        content: LaTeX resume content.

    Returns:
        Summary text (without LaTeX formatting).
    """
    # * Find summary text after \end{center} and before first \section
    summary_pattern = re.compile(
        r"\\end\{center\}\s*\n\s*\\textbf\{([^}]+)\}([^\n]*)",
        re.DOTALL,
    )

    match = summary_pattern.search(content)
    if match:
        title = match.group(1).strip()
        rest = match.group(2).strip()
        return f"{title} {rest}".strip()

    return ""


def apply_rewritten_bullets_to_latex(
    content: str,
    original_bullets: list[str],
    rewritten_bullets: list[RewrittenBullet],
) -> str:
    """
    Apply rewritten bullets back to LaTeX content.

    Args:
        content: Original LaTeX content.
        original_bullets: Original bullet texts for matching.
        rewritten_bullets: Rewritten bullet data.

    Returns:
        Modified LaTeX content with rewritten bullets.
    """
    # * Build mapping from original to rewritten
    rewrite_map = {
        rw.original.strip(): rw.rewritten.strip() for rw in rewritten_bullets
    }

    def replace_item(match):
        original_text = match.group(1).strip()
        # * Clean for matching - remove LaTeX formatting and unescape
        clean_original = re.sub(r"\\textbf\{([^}]+)\}", r"\1", original_text)
        clean_original = re.sub(r"\\emph\{([^}]+)\}", r"\1", clean_original)
        clean_original = re.sub(r"\\href\{[^}]+\}\{([^}]+)\}", r"\1", clean_original)
        # * Unescape LaTeX special characters for matching
        clean_original = clean_original.replace("\\&", "&")
        clean_original = clean_original.replace("\\%", "%")
        clean_original = clean_original.replace("\\$", "$")
        clean_original = clean_original.replace("\\#", "#")
        clean_original = clean_original.replace("\\_", "_")
        clean_original = clean_original.replace("\\{", "{")
        clean_original = clean_original.replace("\\}", "}")
        clean_original = clean_original.strip()

        # * Try to find matching rewrite
        if clean_original in rewrite_map:
            rewritten = rewrite_map[clean_original]
            # * Escape LaTeX special characters in the rewritten text
            rewritten_escaped = escape_latex_text(rewritten)
            return f"\\resumeItem{{{rewritten_escaped}}}"

        # * No match found, return original
        return match.group(0)

    # * Replace bullet items
    item_pattern = re.compile(r"\\resumeItem\{(.+?)\}", re.DOTALL)
    content = item_pattern.sub(replace_item, content)

    return content


def apply_rewritten_summary_to_latex(
    content: str,
    rewritten_summary: SummaryRewriteResponse,
) -> str:
    """
    Apply rewritten summary to LaTeX content.

    Args:
        content: Original LaTeX content.
        rewritten_summary: Rewritten summary data.

    Returns:
        Modified LaTeX content with rewritten summary.
    """
    # * Find and replace summary
    summary_pattern = re.compile(
        r"(\\end\{center\}\s*\n\s*\\textbf\{)[^}]+(}[^\n]*)",
        re.DOTALL,
    )

    # * Extract the rewritten summary parts
    summary_text = rewritten_summary.summary

    # * Escape LaTeX special characters in the summary text
    summary_text_escaped = escape_latex_text(summary_text)
    
    # * Try to split into bold title and rest
    # * Typical format: "Senior ML Engineer with 5+ years..."
    if " with " in summary_text_escaped:
        parts = summary_text_escaped.split(" with ", 1)
        title = parts[0]
        rest = f" with {parts[1]}" if len(parts) > 1 else ""
    elif ". " in summary_text_escaped:
        parts = summary_text_escaped.split(". ", 1)
        title = parts[0]
        rest = f". {parts[1]}" if len(parts) > 1 else ""
    else:
        title = summary_text_escaped[:50] if len(summary_text_escaped) > 50 else summary_text_escaped
        rest = summary_text_escaped[50:] if len(summary_text_escaped) > 50 else ""

    def replace_summary(match):
        return f"{match.group(1)}{title}{match.group(2)}"

    content = summary_pattern.sub(replace_summary, content, count=1)

    return content


# * Global instance
_bullet_rewriter: Optional[BulletRewriter] = None


def get_bullet_rewriter() -> BulletRewriter:
    """Get shared BulletRewriter instance."""
    global _bullet_rewriter
    if _bullet_rewriter is None:
        _bullet_rewriter = BulletRewriter()
    return _bullet_rewriter
