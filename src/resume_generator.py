"""
Resume Variant Generator.

Generates 5 keyword-optimized LaTeX resume versions based on
different focus areas (MLOps, NLP/LLM, Cloud, Data Engineering, Classical ML).

Supports two modes:
- Basic: Reorder bullets and skills (fast, no API calls)
- GPT Rewrite: Use GPT to genuinely rewrite content (slower, better differentiation)
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from src.data_extraction import parse_latex_resume
from src.keyword_engine import get_resume_themes

logger = logging.getLogger("resume_matcher.generator")


def generate_all_variants(
    source_resume_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Generate all 5 resume variants from a source LaTeX file.

    Args:
        source_resume_path: Path to the source resume.tex file.
        output_dir: Directory to write generated variants.

    Returns:
        Dictionary mapping theme name to output file path.
    """
    source_resume_path = Path(source_resume_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # * Read source resume
    with open(source_resume_path, "r", encoding="utf-8") as f:
        source_content = f.read()

    themes = get_resume_themes()
    generated_files = {}

    for theme_name, theme_config in themes.items():
        output_path = output_dir / f"resume_{theme_name}.tex"

        # * Generate variant
        variant_content = _generate_variant(
            source_content,
            theme_name,
            theme_config,
        )

        # * Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(variant_content)

        generated_files[theme_name] = output_path
        print(f"Generated: {output_path}")

    return generated_files


def _generate_variant(
    source_content: str,
    theme_name: str,
    theme_config: dict,
) -> dict:
    """
    Generate a single resume variant with theme-specific optimizations.

    Args:
        source_content: Original LaTeX content.
        theme_name: Name of the theme.
        theme_config: Theme configuration dictionary.

    Returns:
        Modified LaTeX content.
    """
    content = source_content

    # * 1. Reorder Technical Skills section
    content = _reorder_skills_section(content, theme_config)

    # * 2. Add theme-specific comment at top
    theme_comment = f"% Resume variant: {theme_config['name']}\n% Theme: {theme_name}\n"
    content = theme_comment + content

    # * 3. Enhance experience bullets with theme keywords
    content = _enhance_experience_bullets(content, theme_config)

    # * 4. Adjust summary to emphasize theme
    content = _enhance_summary(content, theme_config)

    return content


def _reorder_skills_section(content: str, theme_config: dict) -> str:
    """
    Reorder the Technical Skills section to prioritize theme-relevant skills.

    Args:
        content: LaTeX content.
        theme_config: Theme configuration.

    Returns:
        Modified LaTeX content with reordered skills.
    """
    # * Find the Technical Skills section
    skills_pattern = re.compile(
        r"(\\section\{Technical Skills\}.*?\\begin\{itemize\}.*?\\small\{\\item\{)(.*?)(\\end\{itemize\})",
        re.DOTALL,
    )

    match = skills_pattern.search(content)
    if not match:
        return content

    prefix = match.group(1)
    skills_content = match.group(2)
    suffix = match.group(3)

    # * Parse skill categories
    category_pattern = re.compile(
        r"\\textbf\{([^}]+)\}\{:\s*([^}\\]+)\}",
        re.DOTALL,
    )

    categories = {}
    for cat_match in category_pattern.finditer(skills_content):
        cat_name = cat_match.group(1).strip()
        cat_skills = cat_match.group(2).strip()
        categories[cat_name] = cat_skills

    if not categories:
        return content

    # * Reorder skills within each category based on theme priority
    priority_skills = [s.lower() for s in theme_config.get("skills_priority", [])]

    for cat_name, cat_skills in categories.items():
        skills_list = [s.strip() for s in cat_skills.split(",")]

        # * Sort: priority skills first, then alphabetically
        def skill_sort_key(skill):
            skill_lower = skill.lower()
            for idx, priority in enumerate(priority_skills):
                if priority in skill_lower or skill_lower in priority:
                    return (0, idx, skill)
            return (1, 0, skill)

        sorted_skills = sorted(skills_list, key=skill_sort_key)
        categories[cat_name] = ", ".join(sorted_skills)

    # * Rebuild skills section with preferred category order
    # * Order categories based on theme
    category_order = _get_category_order_for_theme(theme_config, list(categories.keys()))

    new_skills_lines = []
    for cat_name in category_order:
        if cat_name in categories:
            new_skills_lines.append(f"     \\textbf{{{cat_name}}}{{: {categories[cat_name]}}} \\\\")

    # * Remove trailing \\ from last line
    if new_skills_lines:
        new_skills_lines[-1] = new_skills_lines[-1].rstrip(" \\\\")

    new_skills_content = "\n".join(new_skills_lines) + "\n    "

    # * Replace in content
    new_section = prefix + new_skills_content + suffix
    content = skills_pattern.sub(new_section.replace("\\", "\\\\"), content, count=1)

    return content


def _get_category_order_for_theme(theme_config: dict, available_categories: list[str]) -> list[str]:
    """
    Determine the optimal category order for a theme.

    Args:
        theme_config: Theme configuration.
        available_categories: List of available category names.

    Returns:
        Ordered list of category names.
    """
    # * Default priority mapping based on theme
    theme_primary = theme_config.get("primary_category", "")

    # * Define preferred order for each theme
    order_preferences = {
        "mlops": ["Frameworks and Libraries", "Developer Tools", "Cloud Platforms", "Languages"],
        "nlp_llm": ["Frameworks and Libraries", "Languages", "Cloud Platforms", "Developer Tools"],
        "cloud_aws": ["Cloud Platforms", "Developer Tools", "Frameworks and Libraries", "Languages"],
        "data_engineering": ["Developer Tools", "Languages", "Cloud Platforms", "Frameworks and Libraries"],
        "classical_ml": ["Frameworks and Libraries", "Languages", "Cloud Platforms", "Developer Tools"],
    }

    preferred = order_preferences.get(theme_primary, [])

    # * Build final order: preferred categories first, then remaining
    result = []
    for cat in preferred:
        if cat in available_categories and cat not in result:
            result.append(cat)

    for cat in available_categories:
        if cat not in result:
            result.append(cat)

    return result


def _enhance_experience_bullets(content: str, theme_config: dict) -> str:
    """
    Reorder experience bullet points to prioritize theme-relevant items.

    Bullets are scored based on keyword matches and reordered within each
    position so that most relevant bullets appear first.

    Args:
        content: LaTeX content.
        theme_config: Theme configuration.

    Returns:
        Modified LaTeX content with reordered bullets.
    """
    experience_keywords = theme_config.get("experience_keywords", [])
    primary_category = theme_config.get("primary_category", "")

    # * Get additional keywords from taxonomy
    from src.keyword_engine import TECH_TAXONOMY
    taxonomy_keywords = TECH_TAXONOMY.get(primary_category, [])

    # * Combine all relevant keywords
    all_keywords = set(kw.lower() for kw in experience_keywords)
    all_keywords.update(kw.lower() for kw in taxonomy_keywords)

    if not all_keywords:
        return content

    # * Find each resumeItemListStart ... resumeItemListEnd block
    item_list_pattern = re.compile(
        r"(\\resumeItemListStart\s*\n)(.*?)(\\resumeItemListEnd)",
        re.DOTALL,
    )

    def reorder_bullet_list(match):
        prefix = match.group(1)
        items_block = match.group(2)
        suffix = match.group(3)

        # * Extract individual bullet items with their line structure
        # * Match lines that start with \resumeItem (with leading whitespace)
        item_pattern = re.compile(
            r"^(\s*\\resumeItem\{.*?\})[ \t]*$",
            re.MULTILINE,
        )

        items = item_pattern.findall(items_block)

        if len(items) <= 1:
            return match.group(0)  # * Nothing to reorder

        # * Score each bullet based on keyword relevance
        def score_bullet(bullet_text: str) -> tuple[int, int]:
            """
            Return (primary_score, secondary_score) for sorting.
            Higher score = more relevant to theme.
            """
            text_lower = bullet_text.lower()

            primary_matches = 0
            secondary_matches = 0

            for keyword in all_keywords:
                if keyword in text_lower:
                    # * Count exact matches (more specific = higher score)
                    count = text_lower.count(keyword)
                    if keyword in [kw.lower() for kw in experience_keywords]:
                        primary_matches += count * len(keyword)  # * Weight by keyword length
                    else:
                        secondary_matches += count

            return (primary_matches, secondary_matches)

        # * Sort bullets by relevance (descending)
        scored_items = [(score_bullet(item), idx, item) for idx, item in enumerate(items)]
        scored_items.sort(key=lambda x: (-x[0][0], -x[0][1], x[1]))  # * Stable sort by original index

        # * Rebuild the items block with proper newlines
        reordered_items = [item.strip() for _, _, item in scored_items]
        new_items_block = "\n    ".join(reordered_items) + "\n  "

        return prefix + new_items_block + suffix

    content = item_list_pattern.sub(reorder_bullet_list, content)

    return content


def _enhance_summary(content: str, theme_config: dict) -> str:
    """
    Enhance the professional summary to emphasize theme-relevant skills.

    Args:
        content: LaTeX content.
        theme_config: Theme configuration.

    Returns:
        Modified LaTeX content.
    """
    theme_name = theme_config.get("name", "")
    primary_category = theme_config.get("primary_category", "")

    # * Define summary enhancements for each theme
    summary_additions = {
        "mlops": "Experienced in building and deploying production ML systems with robust CI/CD pipelines. ",
        "nlp_llm": "Specialized in NLP and Large Language Model applications, including RAG systems and conversational AI. ",
        "cloud_aws": "Expert in cloud-native ML solutions with extensive AWS experience including Sagemaker and Bedrock. ",
        "data_engineering": "Strong background in data engineering, ETL pipelines, and distributed data processing. ",
        "classical_ml": "Proficient in classical machine learning with focus on model performance optimization and A/B testing. ",
    }

    addition = summary_additions.get(primary_category, "")

    if not addition:
        return content

    # * Find the summary (text after \end{center} and before first \section)
    summary_pattern = re.compile(
        r"(\\end\{center\}\s*\n\s*)(\\textbf\{[^}]+\}[^\n]+)",
        re.DOTALL,
    )

    match = summary_pattern.search(content)
    if match:
        prefix = match.group(1)
        existing_summary = match.group(2)

        # * Insert addition at appropriate point in summary
        # * Find a good insertion point (after first sentence or role description)
        insertion_point = existing_summary.find(". ")
        if insertion_point > 0:
            new_summary = existing_summary[:insertion_point + 2] + addition + existing_summary[insertion_point + 2:]
        else:
            new_summary = existing_summary + " " + addition

        content = content.replace(
            prefix + existing_summary,
            prefix + new_summary,
            1,
        )

    return content


def generate_variant_for_job(
    source_resume_path: str | Path,
    job_description: str,
    output_path: Optional[str | Path] = None,
) -> tuple[str, str]:
    """
    Generate a resume variant optimized for a specific job description.

    Args:
        source_resume_path: Path to the source resume.tex file.
        job_description: Job description text.
        output_path: Optional path to write the output file.

    Returns:
        Tuple of (theme_name, generated_content).
    """
    from src.keyword_engine import find_best_theme_for_job

    # * Find best theme
    theme_name, score = find_best_theme_for_job(job_description)
    themes = get_resume_themes()
    theme_config = themes[theme_name]

    # * Read source resume
    with open(source_resume_path, "r", encoding="utf-8") as f:
        source_content = f.read()

    # * Generate variant
    variant_content = _generate_variant(source_content, theme_name, theme_config)

    # * Write if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(variant_content)

    return theme_name, variant_content


def get_variant_metadata(theme_name: str) -> dict:
    """
    Get metadata for a resume variant.

    Args:
        theme_name: Name of the theme.

    Returns:
        Dictionary with variant metadata.
    """
    themes = get_resume_themes()

    if theme_name not in themes:
        return {}

    theme = themes[theme_name]

    return {
        "theme_name": theme_name,
        "display_name": theme["name"],
        "primary_focus": theme["primary_category"],
        "key_skills": theme["skills_priority"][:5],
        "description": f"Resume optimized for {theme['name']} positions",
    }


def list_available_variants() -> list[dict]:
    """
    List all available resume variants with their metadata.

    Returns:
        List of variant metadata dictionaries.
    """
    themes = get_resume_themes()
    return [get_variant_metadata(name) for name in themes]


# ===== GPT-Powered Variant Generation =====


async def generate_all_variants_with_gpt_async(
    source_resume_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Generate all 5 resume variants with GPT-powered content rewriting (async).

    This creates genuinely different variants by rewriting bullet points
    and summaries for each theme, rather than just reordering.

    Args:
        source_resume_path: Path to the source resume.tex file.
        output_dir: Directory to write generated variants.

    Returns:
        Dictionary mapping theme name to output file path.
    """
    from src.bullet_rewriter import (
        BulletRewriter,
        apply_rewritten_bullets_to_latex,
        apply_rewritten_summary_to_latex,
        extract_bullets_from_latex,
        extract_summary_from_latex,
    )

    source_resume_path = Path(source_resume_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # * Read source resume
    with open(source_resume_path, "r", encoding="utf-8") as f:
        source_content = f.read()

    themes = get_resume_themes()
    generated_files = {}

    # * Extract bullets and summary once
    original_bullets = extract_bullets_from_latex(source_content)
    original_summary = extract_summary_from_latex(source_content)

    logger.info(
        "Extracted %d bullets and summary for GPT rewriting",
        len(original_bullets),
    )

    # * Initialize rewriter
    rewriter = BulletRewriter()

    # * Rewrite bullets for all themes in parallel
    rewrite_tasks = []
    for theme_name, theme_config in themes.items():
        bullet_task = rewriter.rewrite_bullets_async(
            original_bullets, theme_name, theme_config
        )
        summary_task = rewriter.rewrite_summary_async(
            original_summary, theme_name, theme_config
        )
        rewrite_tasks.append((theme_name, bullet_task, summary_task))

    # * Wait for all rewrites to complete
    logger.info("Starting GPT rewriting for %d themes...", len(themes))

    results_by_theme = {}
    for theme_name, bullet_task, summary_task in rewrite_tasks:
        try:
            bullets_result = await bullet_task
            summary_result = await summary_task
            results_by_theme[theme_name] = (bullets_result, summary_result)
            logger.info("Completed rewriting for theme: %s", theme_name)
        except Exception as e:
            logger.error("Failed to rewrite for theme %s: %s", theme_name, e)
            results_by_theme[theme_name] = None

    # * Generate each variant with rewritten content
    for theme_name, theme_config in themes.items():
        output_path = output_dir / f"resume_{theme_name}.tex"

        # * Start with basic variant (reordering)
        variant_content = _generate_variant(source_content, theme_name, theme_config)

        # * Apply GPT rewrites if available
        rewrite_result = results_by_theme.get(theme_name)
        if rewrite_result:
            bullets_result, summary_result = rewrite_result

            # * Apply rewritten bullets
            if bullets_result:
                variant_content = apply_rewritten_bullets_to_latex(
                    variant_content,
                    original_bullets,
                    bullets_result,
                )

            # * Apply rewritten summary
            if summary_result and summary_result.summary != original_summary:
                variant_content = apply_rewritten_summary_to_latex(
                    variant_content,
                    summary_result,
                )

        # * Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(variant_content)

        generated_files[theme_name] = output_path
        logger.info("Generated with GPT rewriting: %s", output_path)

    return generated_files


def generate_all_variants_with_gpt(
    source_resume_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Generate all 5 resume variants with GPT-powered content rewriting (sync).

    Wrapper for async version.

    Args:
        source_resume_path: Path to the source resume.tex file.
        output_dir: Directory to write generated variants.

    Returns:
        Dictionary mapping theme name to output file path.
    """
    return asyncio.run(
        generate_all_variants_with_gpt_async(source_resume_path, output_dir)
    )

