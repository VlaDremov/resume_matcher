"""
Resume Variant Generator.

Generates cluster-driven LaTeX resume variants based on
vacancy cluster artifacts.

Supports two modes:
- Basic: Reorder bullets and skills (fast, no API calls)
- GPT Rewrite: Use GPT to genuinely rewrite content (slower, better differentiation)
"""

import asyncio
import logging
import re
from pathlib import Path

from src.cluster_artifacts import ClusterArtifact, load_cluster_artifact

logger = logging.getLogger("resume_matcher.generator")


def _dedupe_keywords(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result


def build_theme_config_from_cluster(cluster: ClusterArtifact.Cluster) -> dict:
    """Build a theme configuration dictionary from a cluster."""
    keyword_pool = _dedupe_keywords(
        cluster.top_keywords
        + cluster.defining_technologies
        + cluster.defining_skills
    )
    skills_priority = _dedupe_keywords(
        cluster.defining_technologies
        + cluster.defining_skills
        + cluster.top_keywords
    )

    summary_bits = []
    if cluster.summary:
        summary_bits.append(cluster.summary.rstrip("."))
    if keyword_pool:
        summary_bits.append(f"Core strengths include {', '.join(keyword_pool[:4])}")
    summary_detail = ". ".join(summary_bits) + "." if summary_bits else ""
    summary_template = "ML Engineer with 6+ years experience."
    if summary_detail:
        summary_template = f"{summary_template} {summary_detail}"

    return {
        "name": cluster.name,
        "slug": cluster.slug,
        "keywords": keyword_pool,
        "skills_priority": skills_priority,
        "experience_keywords": keyword_pool,
        "summary_template": summary_template,
    }


def generate_variants_from_clusters(
    source_resume_path: str | Path,
    output_dir: str | Path,
    artifact_path: str | Path,
    use_gpt_rewrite: bool = False,
) -> dict[str, Path]:
    """
    Generate resume variants for each cluster in the artifact.

    Args:
        source_resume_path: Path to the source resume.tex file.
        output_dir: Directory to write generated variants.
        artifact_path: Path to the vacancy cluster artifact JSON.
        use_gpt_rewrite: Whether to rewrite bullets via GPT.

    Returns:
        Dictionary mapping cluster slug to output file path.
    """
    if use_gpt_rewrite:
        return asyncio.run(
            _generate_variants_from_clusters_with_gpt_async(
                source_resume_path,
                output_dir,
                artifact_path,
            )
        )

    source_resume_path = Path(source_resume_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(source_resume_path, "r", encoding="utf-8") as f:
        source_content = f.read()

    artifact = load_cluster_artifact(artifact_path)
    themes = {
        cluster.slug: build_theme_config_from_cluster(cluster)
        for cluster in artifact.clusters
    }
    generated_files = {}

    for theme_name, theme_config in themes.items():
        output_path = output_dir / f"resume_{theme_name}.tex"
        variant_content = _generate_variant(
            source_content,
            theme_name,
            theme_config,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(variant_content)

        generated_files[theme_name] = output_path
        print(f"Generated: {output_path}")

    return generated_files


def _generate_variant(
    source_content: str,
    theme_name: str,
    theme_config: dict,
) -> str:
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
    theme_label = theme_config.get("name", theme_name)
    theme_comment = f"% Resume variant: {theme_label}\n% Cluster: {theme_name}\n"
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
        r"(\\section\{Technical Skills\}.*?\\begin\{itemize\}.*?\\small\{\\item\{)(.*?)(\\}\\}\\s*\\end\{itemize\})",
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
        skills_list = _split_skills_list(cat_skills)

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
    content = skills_pattern.sub(lambda _: new_section, content, count=1)

    return content


def _split_skills_list(skills_text: str) -> list[str]:
    """
    Split a comma-separated skill list, preserving commas inside parentheses.
    """
    skills = []
    current = []
    paren_depth = 0

    for ch in skills_text:
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            if paren_depth > 0:
                paren_depth -= 1

        if ch == "," and paren_depth == 0:
            item = "".join(current).strip()
            if item:
                skills.append(item)
            current = []
            continue

        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        skills.append(tail)

    return skills


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
    explicit_order = theme_config.get("category_order", [])

    # * Define preferred order for each theme
    order_preferences = {
        "research_ml": ["Frameworks and Libraries", "Languages", "Developer Tools", "Cloud Platforms"],
        "applied_production": ["Developer Tools", "Cloud Platforms", "Frameworks and Libraries", "Languages"],
        "genai_llm": ["Frameworks and Libraries", "Languages", "Developer Tools", "Cloud Platforms"],
    }

    preferred = explicit_order or order_preferences.get(theme_primary, [])

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
    cluster_keywords = theme_config.get("keywords", [])

    # * Combine all relevant keywords
    all_keywords = set(kw.lower() for kw in experience_keywords)
    all_keywords.update(kw.lower() for kw in cluster_keywords)

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
    summary_template = theme_config.get("summary_template", "")
    if not summary_template:
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

        template_text = summary_template
        if "ML Engineer" in template_text:
            template_text = template_text.replace(
                "ML Engineer",
                r"\textbf{ML Engineer}",
                1,
            )
        new_summary = template_text

        content = content.replace(
            prefix + existing_summary,
            prefix + new_summary,
            1,
        )

    return content


def list_available_variants(artifact_path: str | Path) -> list[dict]:
    """
    List all available resume variants from the cluster artifact.

    Returns:
        List of variant metadata dictionaries.
    """
    artifact = load_cluster_artifact(artifact_path)
    variants = []
    for cluster in artifact.clusters:
        variants.append({
            "theme_name": cluster.slug,
            "display_name": cluster.name,
            "description": cluster.summary or f"Resume optimized for {cluster.name} roles",
        })
    return variants


# ===== GPT-Powered Variant Generation =====


async def _generate_variants_from_clusters_with_gpt_async(
    source_resume_path: str | Path,
    output_dir: str | Path,
    artifact_path: str | Path,
) -> dict[str, Path]:
    """
    Generate cluster-driven resume variants with GPT-powered content rewriting (async).

    This creates genuinely different variants by rewriting bullet points
    and summaries for each cluster, rather than just reordering.
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

    artifact = load_cluster_artifact(artifact_path)
    themes = {
        cluster.slug: build_theme_config_from_cluster(cluster)
        for cluster in artifact.clusters
    }
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
    logger.info("Starting GPT rewriting for %d clusters...", len(themes))

    results_by_theme = {}
    for theme_name, bullet_task, summary_task in rewrite_tasks:
        try:
            bullets_result = await bullet_task
            summary_result = await summary_task
            results_by_theme[theme_name] = (bullets_result, summary_result)
            logger.info("Completed rewriting for cluster: %s", theme_name)
        except Exception as e:
            logger.error("Failed to rewrite for cluster %s: %s", theme_name, e)
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
