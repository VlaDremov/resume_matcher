#!/usr/bin/env python3
"""
Resume Keyword Matcher CLI.

A tool to generate keyword-optimized resume variants and match them
to job descriptions for improved ATS (Applicant Tracking System) compatibility.

Usage:
    python main.py generate          # Generate all 5 resume variants
    python main.py match JOB_FILE    # Find best variant for a job
    python main.py analyze           # Analyze keywords in vacancies
"""

import sys
from pathlib import Path

import click
from tqdm import tqdm

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
def generate(resume: str, output: str, compile_pdf: bool):
    """
    Generate all 5 keyword-optimized resume variants.

    Creates resume variants focused on:
    - MLOps & Platform Engineering
    - NLP & LLM Engineering
    - Cloud & AWS Infrastructure
    - Data Engineering & Pipelines
    - Classical ML & Analytics
    """
    from src.latex_compiler import check_pdflatex_installed, compile_latex_to_pdf
    from src.resume_generator import generate_all_variants, list_available_variants

    resume_path = Path(resume)
    output_dir = Path(output)

    click.echo(f"Source resume: {resume_path}")
    click.echo(f"Output directory: {output_dir}")
    click.echo()

    # * List variants to be generated
    click.echo("Generating resume variants:")
    for variant in list_available_variants():
        click.echo(f"  • {variant['display_name']}")
    click.echo()

    # * Generate variants
    try:
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

    click.echo()
    click.echo(f"Output files are in: {output_dir}")


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
    "--variants",
    "-v",
    type=click.Path(exists=True),
    default=str(DEFAULT_OUTPUT_DIR),
    help="Directory containing resume variants",
)
@click.option(
    "--explain/--no-explain",
    default=False,
    help="Show detailed match explanation",
)
@click.option(
    "--all/--best",
    "show_all",
    default=False,
    help="Show all variants ranked (not just the best)",
)
def match(job_file: str, text: str, variants: str, explain: bool, show_all: bool):
    """
    Match a job description to the best resume variant.

    Provide either a JOB_FILE path or use --text for direct input.

    Example:
        python main.py match vacancies/asos.txt
        python main.py match --text "Senior ML Engineer with MLOps..."
    """
    from src.matcher import ResumeMatcher, rank_all_variants

    # * Get job description text
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

    variants_dir = Path(variants)

    if not variants_dir.exists():
        click.echo(f"Error: Variants directory not found: {variants_dir}", err=True)
        click.echo("Run 'python main.py generate' first to create variants.")
        sys.exit(1)

    click.echo()

    # * Perform matching
    matcher = ResumeMatcher(variants_dir)

    if not matcher.variants:
        click.echo("Error: No resume variants found.", err=True)
        click.echo("Run 'python main.py generate' first to create variants.")
        sys.exit(1)

    result = matcher.match(job_text)

    # * Display results
    if show_all:
        click.echo("All variants ranked by match score:")
        click.echo("-" * 50)

        ranked = rank_all_variants(job_text, variants_dir)
        for i, item in enumerate(ranked, 1):
            score_bar = "█" * int(item["score"] * 20)
            variant_display = item["variant"].replace("_", " ").title()

            click.echo(f"{i}. {variant_display:25} {score_bar:20} {item['score']:.3f}")

            if item["pdf_path"]:
                click.echo(f"   PDF: {item['pdf_path']}")

        click.echo()
    else:
        best = result["best_variant"]
        confidence = result["confidence"]

        click.echo("═" * 50)
        click.echo(f"  BEST MATCH: {best.replace('_', ' ').title()}")
        click.echo(f"  Confidence: {confidence:.1%}")
        click.echo("═" * 50)

        # * Show file paths
        tex_path = matcher.get_variant_path(best, "tex")
        pdf_path = matcher.get_variant_path(best, "pdf")

        click.echo()
        if tex_path:
            click.echo(f"  LaTeX: {tex_path}")
        if pdf_path:
            click.echo(f"  PDF:   {pdf_path}")

    # * Show explanation if requested
    if explain:
        click.echo()
        click.echo("Match Explanation:")
        click.echo("-" * 50)
        explanation = matcher.explain_match(job_text, result["best_variant"])
        click.echo(explanation)


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
def analyze(vacancies: str, top: int):
    """
    Analyze keywords from job descriptions in the vacancies folder.

    Shows keyword frequency and categorization to help understand
    what skills are most in demand.
    """
    from src.keyword_engine import analyze_vacancies

    vacancies_dir = Path(vacancies)

    click.echo(f"Analyzing vacancies in: {vacancies_dir}")
    click.echo()

    result = analyze_vacancies(vacancies_dir, top_n=top)

    if not result["keywords"]:
        click.echo("No keywords found. Check that vacancy files contain text.")
        return

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
    from src.linkedin_scraper import scrape_linkedin_profile

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

