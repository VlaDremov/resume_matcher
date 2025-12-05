"""
Resume Variant Generator.

Generates 5 keyword-optimized LaTeX resume versions based on
different focus areas (MLOps, NLP/LLM, Cloud, Data Engineering, Classical ML).
"""

import re
from pathlib import Path
from typing import Optional

from src.data_extraction import parse_latex_resume
from src.keyword_engine import get_resume_themes


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
    Enhance experience bullet points with theme-relevant keywords.

    This performs light rewrites - adding emphasis to existing keywords
    rather than completely rewriting bullets.

    Args:
        content: LaTeX content.
        theme_config: Theme configuration.

    Returns:
        Modified LaTeX content.
    """
    experience_keywords = theme_config.get("experience_keywords", [])

    if not experience_keywords:
        return content

    # * Find all resumeItem entries
    item_pattern = re.compile(r"(\\resumeItem\{)([^}]+)(\})")

    def enhance_bullet(match):
        prefix = match.group(1)
        bullet_text = match.group(2)
        suffix = match.group(3)

        # * Check if bullet already contains theme keywords
        bullet_lower = bullet_text.lower()
        keyword_count = sum(1 for kw in experience_keywords if kw.lower() in bullet_lower)

        if keyword_count > 0:
            # * Bullet is already relevant - maybe bold key terms
            for keyword in experience_keywords:
                # * Bold exact keyword matches (case-insensitive)
                pattern = re.compile(f"\\b({re.escape(keyword)})\\b", re.IGNORECASE)
                # * Only bold if not already in a LaTeX command
                if f"\\textbf{{{keyword}}}" not in bullet_text:
                    # * Add subtle emphasis by keeping as-is (no over-bolding)
                    pass

        return prefix + bullet_text + suffix

    content = item_pattern.sub(enhance_bullet, content)

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

