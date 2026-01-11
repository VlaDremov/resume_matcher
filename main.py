#!/usr/bin/env python3
"""
Resume Keyword Matcher CLI.

A tool to generate keyword-optimized resume variants and match them
to job descriptions for improved ATS (Applicant Tracking System) compatibility.

Usage:
    python main.py cluster-vacancies # Cluster vacancies into categories
    python main.py generate          # Generate resume variants per cluster
    python main.py analyze           # Analyze keywords in vacancies
"""

import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from tqdm import tqdm

from src.logging_config import configure_logging

# * Load environment variables from .env file
load_dotenv()
# * Configure logging with timestamps and filenames
configure_logging()

# * Configuration - adjust these paths as needed
DEFAULT_RESUME_PATH = Path(__file__).parent / "resume.tex"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_VACANCIES_DIR = Path(__file__).parent / "vacancies"
DEFAULT_CLUSTER_ARTIFACT = DEFAULT_OUTPUT_DIR / "vacancy_clusters.json"


@click.group()
@click.version_option(version="2.0.0", prog_name="Resume Keyword Matcher")
def cli():
    """
    Resume Keyword Matcher - Optimize your resume for ATS systems.

    Generate multiple keyword-optimized versions of your resume and
    match them to job descriptions.
    """
    pass


@cli.command()
@click.option(
    "--resume",
    "-r",
    type=click.Path(exists=True),
    default=str(DEFAULT_RESUME_PATH),
    help="Path to source resume.tex file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=str(DEFAULT_OUTPUT_DIR),
    help="Output directory for generated variants",
)
@click.option(
    "--clusters-artifact",
    "-c",
    type=click.Path(),
    default=str(DEFAULT_CLUSTER_ARTIFACT),
    help="Path to cluster artifact JSON",
)
@click.option(
    "--compile-pdf/--no-compile-pdf",
    default=True,
    help="Compile LaTeX to PDF (requires pdflatex)",
)
@click.option(
    "--gpt-cache/--no-gpt-cache",
    default=False,
    help="Use cached GPT rewrites when generating variants",
)
@click.option(
    "--use-gpt-rewrite/--no-gpt-rewrite",
    default=True,
    help="Use GPT to genuinely rewrite content per theme (costs ~$0.05, creates meaningfully different variants)",
)
def generate(
    resume: str,
    output: str,
    clusters_artifact: str,
    compile_pdf: bool,
    gpt_cache: bool,
    use_gpt_rewrite: bool,
):
    """
    Generate resume variants for each vacancy cluster.

    Use --use-gpt-rewrite for genuinely different variants (recommended).
    """
    import os

    from src.latex_compiler import check_pdflatex_installed, compile_latex_to_pdf
    from src.resume_generator import (
        generate_variants_from_clusters,
        list_available_variants,
    )

    resume_path = Path(resume)
    output_dir = Path(output)
    artifact_path = Path(clusters_artifact)

    click.echo(f"Source resume: {resume_path}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Cluster artifact: {artifact_path}")

    if use_gpt_rewrite:
        if not os.getenv("OPENAI_API_KEY"):
            click.echo("Error: OPENAI_API_KEY not set for GPT rewriting.", err=True)
            click.echo("Set it via: export OPENAI_API_KEY='sk-...'")
            click.echo("Or use --no-gpt-rewrite for basic generation.")
            sys.exit(1)
        click.echo("Mode: GPT-powered content rewriting (creates unique variants)")
    else:
        click.echo("Mode: Basic reordering (use --use-gpt-rewrite for better differentiation)")

    click.echo()

    if not artifact_path.exists():
        click.echo(f"Error: cluster artifact not found at {artifact_path}", err=True)
        click.echo("Run: python main.py cluster-vacancies --clusters 4")
        sys.exit(1)

    # * List variants to be generated
    click.echo("Generating resume variants:")
    for variant in list_available_variants(artifact_path):
        click.echo(f"  • {variant['display_name']}")
    click.echo()

    # * Generate variants
    try:
        if use_gpt_rewrite:
            click.echo("Using GPT to rewrite content (this may take 30-60 seconds)...")
            if not gpt_cache:
                click.echo("GPT rewrite cache: disabled")
        generated = generate_variants_from_clusters(
            resume_path,
            output_dir,
            artifact_path,
            use_gpt_rewrite=use_gpt_rewrite,
            use_gpt_cache=gpt_cache,
        )
        click.echo(f"\n✓ Generated {len(generated)} LaTeX variants")
    except Exception as e:
        click.echo(f"✗ Error generating variants: {e}", err=True)
        sys.exit(1)

    # * Compile to PDF
    if compile_pdf:
        click.echo()
        if not check_pdflatex_installed():
            click.echo("! pdflatex not found - skipping PDF compilation", err=True)
            click.echo("  Install a LaTeX distribution to enable PDF output:")
            click.echo("    macOS: brew install --cask mactex-no-gui")
            click.echo("    Ubuntu: sudo apt-get install texlive-latex-base")
        else:
            click.echo("Compiling to PDF...")
            success_count = 0

            for theme_name, tex_path in tqdm(generated.items(), desc="Compiling"):
                pdf_path = compile_latex_to_pdf(tex_path, output_dir)
                if pdf_path:
                    success_count += 1

            click.echo(f"\n✓ Compiled {success_count}/{len(generated)} PDFs")

    if use_gpt_rewrite:
        from src.llm_client import format_usage_summary, get_usage_summary

        usage_summary = get_usage_summary()
        click.echo()
        click.echo("LLM usage summary:")
        lines = format_usage_summary(usage_summary)
        if lines:
            for line in lines:
                click.echo(line)
        else:
            click.echo("total_requests=0")

    click.echo()
    click.echo(f"Output files are in: {output_dir}")


@cli.command()
@click.option(
    "--vacancies",
    "-v",
    type=click.Path(exists=True),
    default=str(DEFAULT_VACANCIES_DIR),
    help="Directory containing vacancy files",
)
@click.option(
    "--top",
    "-n",
    type=int,
    default=30,
    help="Number of top keywords to show",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Optional path to save keyword report as JSON",
)
def analyze(vacancies: str, top: int, output: str | None):
    """
    Analyze keywords from job descriptions in the vacancies folder.

    Shows keyword frequency and categorization to help understand
    what skills are most in demand. Use --output to save JSON for reuse.
    """
    import json

    from src.keyword_engine import analyze_vacancies, serialize_keyword_report

    vacancies_dir = Path(vacancies)

    click.echo(f"Analyzing vacancies in: {vacancies_dir}")
    click.echo()

    result = analyze_vacancies(vacancies_dir, top_n=top)

    if not result["keywords"]:
        click.echo("No keywords found. Check that vacancy files contain text.")
        return

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = serialize_keyword_report(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        click.echo(f"Saved keyword report to: {output_path}")
        click.echo()

    # * Show top keywords
    click.echo(f"Top {top} Keywords:")
    click.echo("-" * 50)

    for keyword, score in result["keywords"][:top]:
        bar = "█" * min(int(score / 2), 30)
        click.echo(f"  {keyword:30} {bar} ({score:.0f})")

    # * Show by category
    click.echo()
    click.echo("Keywords by Category:")
    click.echo("-" * 50)

    for category, keywords in result["by_category"].items():
        if keywords:
            category_display = category.replace("_", " ").title()
            click.echo(f"\n{category_display} ({len(keywords)}):")
            click.echo(f"  {', '.join(keywords[:15])}")
            if len(keywords) > 15:
                click.echo(f"  ... and {len(keywords) - 15} more")


@cli.command("cluster-vacancies")
@click.option(
    "--vacancies",
    "-d",
    type=click.Path(exists=True),
    default=str(DEFAULT_VACANCIES_DIR),
    help="Directory containing vacancy files",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=str(DEFAULT_CLUSTER_ARTIFACT),
    help="Save cluster artifact JSON to file",
)
@click.option(
    "--clusters",
    "-n",
    type=int,
    default=3,
    help="Number of clusters (0=auto)",
)
@click.option(
    "--gpt/--no-gpt",
    "use_gpt",
    default=True,
    help="Use GPT for enhancement",
)
@click.option(
    "--refresh/--no-refresh",
    default=True,
    help="Bypass cached clustering results",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed output",
)
def cluster_vacancies(
    vacancies: str,
    output: str | None,
    clusters: int,
    use_gpt: bool,
    refresh: bool,
    verbose: bool,
):
    """Analyze and cluster all vacancies by keyword similarity."""
    from src.cluster_artifacts import build_cluster_artifact, save_cluster_artifact
    from src.vacancy_clustering import VacancyClusteringPipeline

    vacancies_dir = Path(vacancies)
    pipeline = VacancyClusteringPipeline(
        vacancies_dir=vacancies_dir,
        use_gpt=use_gpt,
        refresh_cache=refresh,
    )

    click.echo(f"Clustering vacancies in: {vacancies_dir}")
    result = pipeline.cluster(num_clusters=clusters)

    stats = result.pipeline_stats or {}
    click.echo()
    click.echo(f"Clustering {result.total_vacancies} vacancies into {len(result.clusters)} categories...")
    click.echo()

    if stats:
        gpt_used = bool(stats.get("gpt_used"))
        click.echo(
            f"Stage 1: Extracting keywords (TF-IDF + taxonomy)... \u2713 {stats.get('raw_keywords', 0)} unique keywords"
        )
        if gpt_used:
            click.echo(
                f"Stage 2: Enriching keywords (LLM)... \u2713 {stats.get('canonical_keywords', 0)} canonical keywords"
            )
        elif use_gpt:
            click.echo("Stage 2: Enriching keywords (LLM)... skipped (no API key)")
        else:
            click.echo("Stage 2: Enriching keywords (LLM)... skipped (--no-gpt)")
        click.echo(
            f"Stage 3: Deduplicating keywords (embeddings)... \u2713 Merged to {stats.get('embedding_merged', 0)} keyword groups"
        )
        vector_source = stats.get("vacancy_vector_source", "unknown")
        cluster_count = stats.get("cluster_count", len(result.clusters))
        selection = stats.get("cluster_selection", "requested")
        click.echo(
            f"Stage 4: Clustering vacancies ({vector_source})... \u2713 {cluster_count} clusters ({selection})"
        )
        label_source = stats.get("label_source", "fallback")
        click.echo(f"Stage 5: Naming clusters ({label_source})... \u2713")
        if stats.get("cache_hit"):
            click.echo("Cache: hit")

    def format_keyword_list(items: list[str], counts: dict[str, int], limit: int = 6) -> str:
        if not items:
            return "None"
        parts = []
        for keyword in items[:limit]:
            parts.append(f"{keyword} ({counts.get(keyword, 0)})")
        return ", ".join(parts)

    for cluster in result.clusters.values():
        click.echo()
        click.echo("═" * 63)
        click.echo(f"CLUSTER: {cluster.name} ({cluster.slug}) ({len(cluster.vacancies)} vacancies)")
        click.echo("─" * 63)

        if cluster.summary:
            click.echo(f"Summary: {cluster.summary}")

        vacancies_list = ", ".join(cluster.vacancies[:10])
        if len(cluster.vacancies) > 10:
            vacancies_list += "..."
        click.echo(f"Vacancies: {vacancies_list or 'None'}")

        click.echo(
            f"Technologies: {format_keyword_list(cluster.defining_technologies, cluster.keyword_counts)}"
        )
        click.echo(
            f"Skills: {format_keyword_list(cluster.defining_skills, cluster.keyword_counts)}"
        )
        click.echo(f"Top Keywords: {', '.join(cluster.top_keywords[:8]) or 'None'}")

        if verbose and cluster.keyword_counts:
            top_counts = ", ".join(
                f"{kw} ({count})"
                for kw, count in sorted(cluster.keyword_counts.items(), key=lambda x: -x[1])[:10]
            )
            click.echo(f"Keyword Counts: {top_counts}")

    output_path = Path(output) if output else DEFAULT_CLUSTER_ARTIFACT
    artifact = build_cluster_artifact(result, vacancies_dir, clusters)
    save_cluster_artifact(output_path, artifact)
    click.echo()
    click.echo(f"Saved clustering report to: {output_path}")


@cli.command()
@click.option(
    "--resume",
    "-r",
    type=click.Path(exists=True),
    default=str(DEFAULT_RESUME_PATH),
    help="Path to resume.tex file",
)
@click.option(
    "--pdf",
    "-p",
    type=click.Path(exists=True),
    default=None,
    help="Path to LinkedIn PDF export",
)
def info(resume: str, pdf: str):
    """
    Display information extracted from resume and LinkedIn data.

    Useful for verifying that the parser correctly extracts your information.
    """
    from src.data_extraction import extract_text_from_pdf, parse_latex_resume

    if resume:
        resume_path = Path(resume)
        click.echo(f"Resume: {resume_path}")
        click.echo("=" * 50)

        parsed = parse_latex_resume(resume_path)

        # * Header
        header = parsed["header"]
        click.echo(f"Name: {header.get('name', 'N/A')}")
        click.echo(f"Email: {header.get('email', 'N/A')}")
        click.echo(f"Location: {header.get('location', 'N/A')}")
        click.echo()

        # * Summary
        if parsed["summary"]:
            click.echo("Summary:")
            # * Wrap long text
            summary = parsed["summary"][:200]
            if len(parsed["summary"]) > 200:
                summary += "..."
            click.echo(f"  {summary}")
            click.echo()

        # * Experience
        click.echo(f"Experience ({len(parsed['experience'])} entries):")
        for job in parsed["experience"]:
            click.echo(f"  • {job['title']} at {job['company']}")
            click.echo(f"    {job['dates']}")

        click.echo()

        # * Skills
        click.echo("Technical Skills:")
        for category, skills in parsed["skills"].items():
            click.echo(f"  {category}: {', '.join(skills[:5])}")
            if len(skills) > 5:
                click.echo(f"    ... and {len(skills) - 5} more")

    # * LinkedIn PDF
    if pdf:
        pdf_path = Path(pdf)
        click.echo()
        click.echo(f"LinkedIn PDF: {pdf_path}")
        click.echo("=" * 50)

        try:
            text = extract_text_from_pdf(pdf_path)
            # * Show first 500 characters
            preview = text[:500]
            if len(text) > 500:
                preview += "..."
            click.echo(preview)
        except Exception as e:
            click.echo(f"Error reading PDF: {e}")


@cli.command()
@click.argument("job_file", type=click.Path(exists=True), required=False)
@click.option(
    "--text",
    "-t",
    type=str,
    default=None,
    help="Job description text (alternative to file)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for tailored resume",
)
@click.option(
    "--clusters-artifact",
    "-c",
    type=click.Path(),
    default=str(DEFAULT_CLUSTER_ARTIFACT),
    help="Path to cluster artifact JSON",
)
@click.option(
    "--preview/--no-preview",
    default=False,
    help="Preview changes without saving",
)
def tailor(job_file: str, text: str, output: str, clusters_artifact: str, preview: bool):
    """
    Generate a tailored resume using GPT-5 analysis.

    Analyzes the job description and rewrites bullet points to
    emphasize relevant keywords.

    Example:
        python main.py tailor vacancies/google.txt
        python main.py tailor --text "Senior ML Engineer..."
    """
    import os
    from pathlib import Path

    # * Check API key
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("Error: OPENAI_API_KEY not set.", err=True)
        click.echo("Set it via: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    # * Get job description
    if job_file:
        job_path = Path(job_file)
        with open(job_path, "r", encoding="utf-8") as f:
            job_text = f.read()
        click.echo(f"Job description: {job_path.name}")
    elif text:
        job_text = text
        click.echo("Job description: (provided via --text)")
    else:
        click.echo("Error: Provide either JOB_FILE or --text", err=True)
        sys.exit(1)

    click.echo()
    click.echo("Analyzing with GPT-5...")

    try:
        from src.bullet_rewriter import (
            BulletRewriter,
            apply_rewritten_bullets_to_latex,
            apply_rewritten_summary_to_latex,
            extract_bullets_from_latex,
            extract_summary_from_latex,
        )
        from src.cluster_artifacts import load_cluster_artifact
        from src.cluster_matcher import ClusterMatcher
        from src.resume_generator import build_theme_config_from_cluster

        artifact_path = Path(clusters_artifact)
        if not artifact_path.exists():
            click.echo(f"Error: cluster artifact not found at {artifact_path}", err=True)
            click.echo("Run: python main.py cluster-vacancies --clusters 4")
            sys.exit(1)

        matcher = ClusterMatcher(artifact_path)
        match_result = matcher.match(job_text)

        best_variant = match_result["best_cluster"]
        similarity = match_result["best_score"]
        if not best_variant:
            click.echo("Error: no clusters available for matching.", err=True)
            sys.exit(1)

        click.echo(f"Best variant: {best_variant} (similarity: {similarity:.2%})")

        artifact = load_cluster_artifact(artifact_path)
        cluster = next((c for c in artifact.clusters if c.slug == best_variant), None)
        if not cluster:
            click.echo(f"Error: cluster '{best_variant}' not found in artifact.", err=True)
            sys.exit(1)

        theme_config = build_theme_config_from_cluster(cluster)

        # * Load variant and extract bullets
        variant_path = DEFAULT_OUTPUT_DIR / f"resume_{best_variant}.tex"
        if not variant_path.exists():
            click.echo(f"Error: resume variant not found at {variant_path}", err=True)
            sys.exit(1)

        with open(variant_path, "r", encoding="utf-8") as f:
            latex_content = f.read()

        bullets = extract_bullets_from_latex(latex_content)
        summary = extract_summary_from_latex(latex_content)
        click.echo(f"Found {len(bullets)} bullet points")

        click.echo()
        click.echo("Rewriting bullets and summary...")

        rewriter = BulletRewriter()
        rewritten_bullets = rewriter.rewrite_bullets(
            bullets,
            best_variant,
            theme_config,
        )
        rewritten_summary = rewriter.rewrite_summary(
            summary,
            best_variant,
            theme_config,
        )

        rewritten_content = apply_rewritten_bullets_to_latex(
            latex_content,
            bullets,
            rewritten_bullets,
        )
        if rewritten_summary and rewritten_summary.summary != summary:
            rewritten_content = apply_rewritten_summary_to_latex(
                rewritten_content,
                rewritten_summary,
            )

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rewritten_content)
            click.echo()
            click.echo(f"Saved tailored resume to: {output_path}")

        if preview and rewritten_bullets:
            click.echo()
            click.echo("Rewritten Bullets (preview):")
            click.echo("-" * 50)
            for rb in rewritten_bullets[:5]:
                click.echo(f"Original: {rb.original[:80]}...")
                click.echo(f"Rewritten: {rb.rewritten[:80]}...")
                click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--host",
    "-h",
    default="127.0.0.1",
    help="Host to bind to",
)
@click.option(
    "--port",
    "-p",
    default=8000,
    help="Port to bind to",
)
@click.option(
    "--reload/--no-reload",
    default=True,
    help="Enable auto-reload for development",
)
def serve(host: str, port: int, reload: bool):
    """
    Start the web interface server.

    Launches the FastAPI backend that serves the React frontend.

    Example:
        python main.py serve
        python main.py serve --port 8080
    """
    import os

    # * Check API key
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("Warning: OPENAI_API_KEY not set.", err=True)
        click.echo("GPT-5 features will not work without it.")
        click.echo()

    click.echo(f"Starting Resume Matcher API on http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")
    click.echo()

    try:
        import uvicorn
        uvicorn.run(
            "backend.api:app",
            host=host,
            port=port,
            reload=reload,
        )
    except ImportError:
        click.echo("Error: uvicorn not installed. Run: pip install uvicorn", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
