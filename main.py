#!/usr/bin/env python3
"""
Resume Keyword Matcher CLI.

A tool to generate keyword-optimized resume variants and match them
to job descriptions for improved ATS (Applicant Tracking System) compatibility.

Usage:
    python main.py generate          # Generate all 3 resume variants
    python main.py cluster-vacancies # Cluster vacancies into categories
    python main.py analyze           # Analyze keywords in vacancies
"""

import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from tqdm import tqdm

# * Load environment variables from .env file
load_dotenv()

# * Configuration - adjust these paths as needed
DEFAULT_RESUME_PATH = Path(__file__).parent / "resume.tex"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_VACANCIES_DIR = Path(__file__).parent / "vacancies"


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
    "--compile-pdf/--no-compile-pdf",
    default=True,
    help="Compile LaTeX to PDF (requires pdflatex)",
)
@click.option(
    "--use-gpt-rewrite/--no-gpt-rewrite",
    default=False,
    help="Use GPT to genuinely rewrite content per theme (costs ~$0.05, creates meaningfully different variants)",
)
def generate(resume: str, output: str, compile_pdf: bool, use_gpt_rewrite: bool):
    """
    Generate all 3 keyword-optimized resume variants.

    Creates resume variants focused on:
    - Research & Advanced ML
    - Applied ML & Production Systems
    - Generative AI & LLM Engineering

    Use --use-gpt-rewrite for genuinely different variants (recommended).
    """
    import os

    from src.latex_compiler import check_pdflatex_installed, compile_latex_to_pdf
    from src.resume_generator import (
        generate_all_variants,
        generate_all_variants_with_gpt,
        list_available_variants,
    )

    resume_path = Path(resume)
    output_dir = Path(output)

    click.echo(f"Source resume: {resume_path}")
    click.echo(f"Output directory: {output_dir}")

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

    # * List variants to be generated
    click.echo("Generating resume variants:")
    for variant in list_available_variants():
        click.echo(f"  • {variant['display_name']}")
    click.echo()

    # * Generate variants
    try:
        if use_gpt_rewrite:
            click.echo("Using GPT to rewrite content (this may take 30-60 seconds)...")
            generated = generate_all_variants_with_gpt(resume_path, output_dir)
            click.echo(f"\n✓ Generated {len(generated)} GPT-rewritten LaTeX variants")
        else:
            generated = generate_all_variants(resume_path, output_dir)
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
    default=None,
    help="Save JSON to file",
)
@click.option(
    "--clusters",
    "-n",
    type=int,
    default=3,
    help="Number of clusters",
)
@click.option(
    "--gpt/--no-gpt",
    "use_gpt",
    default=True,
    help="Use GPT for enhancement",
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
    verbose: bool,
):
    """Analyze and cluster all vacancies by keyword similarity."""
    import json

    from src.vacancy_clustering import VacancyClusteringPipeline

    vacancies_dir = Path(vacancies)
    pipeline = VacancyClusteringPipeline(vacancies_dir=vacancies_dir, use_gpt=use_gpt)

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
                f"Stage 2: Enhancing with GPT... \u2713 Categorized {stats.get('raw_keywords', 0)} \u2192 {stats.get('gpt_categorized', 0)} canonical keywords"
            )
        elif use_gpt:
            click.echo("Stage 2: Enhancing with GPT... skipped (no API key)")
        else:
            click.echo("Stage 2: Enhancing with GPT... skipped (--no-gpt)")
        click.echo(
            f"Stage 3: Clustering by embeddings... \u2713 Merged to {stats.get('embedding_merged', 0)} keyword groups"
        )
        click.echo("Stage 4: Assigning vacancies... \u2713")

    def format_keyword_list(items: list[str], counts: dict[str, int], limit: int = 6) -> str:
        if not items:
            return "None"
        parts = []
        for keyword in items[:limit]:
            parts.append(f"{keyword} ({counts.get(keyword, 0)})")
        return ", ".join(parts)

    for cluster_key, cluster in result.clusters.items():
        click.echo()
        click.echo("═" * 63)
        click.echo(f"CLUSTER: {cluster_key} ({len(cluster.vacancies)} vacancies)")
        click.echo("─" * 63)

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

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
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
@click.argument("url")
@click.option(
    "--cookies",
    "-c",
    type=click.Path(),
    default=None,
    help="Path to LinkedIn cookies file (JSON)",
)
@click.option(
    "--save-cookies",
    "-s",
    type=click.Path(),
    default=None,
    help="Save cookies after login",
)
def scrape(url: str, cookies: str, save_cookies: str):
    """
    Scrape a LinkedIn profile (requires manual login).

    WARNING: This violates LinkedIn's Terms of Service.
    Use the PDF export method instead when possible.

    Example:
        python main.py scrape https://www.linkedin.com/in/username/
    """
    from deprecated.linkedin_scraper import scrape_linkedin_profile

    click.echo("! Warning: Scraping LinkedIn violates their Terms of Service.")
    click.echo("  Consider using LinkedIn's PDF export feature instead.")
    click.echo()

    if not click.confirm("Do you want to continue?"):
        return

    click.echo()
    click.echo("A browser window will open. Log in to LinkedIn manually.")
    click.echo()

    try:
        data = scrape_linkedin_profile(
            url,
            cookies_path=cookies,
            save_cookies_path=save_cookies,
        )

        if data:
            click.echo()
            click.echo("Scraped Profile Data:")
            click.echo("=" * 50)

            for key, value in data.items():
                if isinstance(value, list):
                    click.echo(f"{key}: {len(value)} items")
                elif isinstance(value, str) and len(value) > 100:
                    click.echo(f"{key}: {value[:100]}...")
                else:
                    click.echo(f"{key}: {value}")
        else:
            click.echo("Failed to scrape profile data.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


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
    "--preview/--no-preview",
    default=False,
    help="Preview changes without saving",
)
def tailor(job_file: str, text: str, output: str, preview: bool):
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
        from src.bullet_rewriter import BulletRewriter, extract_bullets_from_latex
        from src.semantic_matcher import SemanticMatcher

        # * Semantic matching
        matcher = SemanticMatcher(DEFAULT_OUTPUT_DIR)
        match_result = matcher.match(job_text)

        best_variant = match_result["best_variant"]
        similarity = match_result["similarity_score"]

        click.echo(f"Best variant: {best_variant} (similarity: {similarity:.2%})")

        # * Load variant and extract bullets
        variant_path = DEFAULT_OUTPUT_DIR / f"resume_{best_variant}.tex"
        with open(variant_path, "r") as f:
            latex_content = f.read()

        bullets = extract_bullets_from_latex(latex_content)
        click.echo(f"Found {len(bullets)} bullet points")

        # * Analyze and rewrite
        click.echo()
        click.echo("Rewriting bullets...")

        rewriter = BulletRewriter()
        result = rewriter.analyze_and_rewrite(
            job_description=job_text,
            resume_bullets=bullets[:10],
            resume_variant=best_variant,
        )

        # * Display results
        click.echo()
        click.echo("═" * 50)
        click.echo(f"  RELEVANCY SCORE: {result.relevancy_score}/100")
        click.echo("═" * 50)

        if result.key_matches:
            click.echo()
            click.echo("Matched Keywords:")
            click.echo(f"  {', '.join(result.key_matches[:10])}")

        if result.missing_keywords:
            click.echo()
            click.echo("Missing Keywords:")
            click.echo(f"  {', '.join(result.missing_keywords[:10])}")

        if preview and result.rewritten_bullets:
            click.echo()
            click.echo("Rewritten Bullets:")
            click.echo("-" * 50)
            for rb in result.rewritten_bullets[:5]:
                click.echo(f"Original: {rb.original[:80]}...")
                click.echo(f"Rewritten: {rb.rewritten[:80]}...")
                click.echo()

        if result.reasoning:
            click.echo()
            click.echo("Analysis:")
            click.echo(f"  {result.reasoning}")

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
