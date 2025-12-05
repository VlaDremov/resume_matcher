# Resume Keyword Matcher

A Python tool that generates keyword-optimized resume variants and matches them to job descriptions for improved ATS (Applicant Tracking System) compatibility.

## Features

- **Generate 5 Resume Variants**: Creates keyword-focused versions of your resume:
  - MLOps & Platform Engineering
  - NLP & LLM Engineering  
  - Cloud & AWS Infrastructure
  - Data Engineering & Pipelines
  - Classical ML & Analytics

- **Smart Job Matching**: Analyzes job descriptions and recommends the best resume variant based on:
  - Category-based keyword matching
  - TF-IDF similarity scoring
  - Keyword overlap analysis

- **Keyword Analysis**: Extracts and clusters keywords from job descriptions to understand market demands

- **LaTeX to PDF**: Compiles generated variants to PDF (requires pdflatex)

## Installation

```bash
# Clone or navigate to the project
cd resume_matcher

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Optional: LaTeX Compilation

To compile resume variants to PDF, install a LaTeX distribution:

- **macOS**: `brew install --cask mactex-no-gui`
- **Ubuntu**: `sudo apt-get install texlive-latex-base texlive-latex-extra`
- **Windows**: Install [MiKTeX](https://miktex.org/)

## Usage

### Generate Resume Variants

```bash
# Generate all 5 variants (LaTeX + PDF if pdflatex available)
python main.py generate

# Generate LaTeX only (skip PDF compilation)
python main.py generate --no-compile-pdf

# Use custom paths
python main.py generate --resume path/to/resume.tex --output path/to/output/
```

### Match Job Description

```bash
# Match a job description file to the best resume
python main.py match vacancies/asos.txt

# With detailed explanation
python main.py match vacancies/asos.txt --explain

# Show all variants ranked
python main.py match vacancies/asos.txt --all

# Match with direct text input
python main.py match --text "Senior ML Engineer with MLOps experience..."
```

### Analyze Keywords

```bash
# Analyze keywords from all job descriptions
python main.py analyze

# Show more keywords
python main.py analyze --top 50
```

### View Resume Info

```bash
# Display parsed resume information
python main.py info

# Include LinkedIn PDF
python main.py info --pdf Profile.pdf
```

## Project Structure

```
resume_matcher/
├── resume.tex              # Your original resume
├── Profile.pdf             # LinkedIn export (optional)
├── vacancies/              # Job description files
│   ├── asos.txt
│   ├── strava.txt
│   └── ...
├── output/                 # Generated variants
│   ├── resume_mlops.tex
│   ├── resume_mlops.pdf
│   ├── resume_nlp_llm.tex
│   └── ...
├── src/
│   ├── data_extraction.py  # PDF and LaTeX parsing
│   ├── linkedin_scraper.py # LinkedIn scraping (optional)
│   ├── keyword_engine.py   # Keyword extraction and clustering
│   ├── resume_generator.py # Resume variant generation
│   ├── latex_compiler.py   # LaTeX to PDF compilation
│   └── matcher.py          # Job-to-resume matching
├── main.py                 # CLI entry point
├── requirements.txt
└── README.md
```

## How It Works

### 1. Keyword Extraction

The tool uses multiple techniques to identify relevant keywords:
- **TF-IDF Analysis**: Statistical importance of terms
- **spaCy NLP**: Named entity recognition and noun phrase extraction
- **Technology Taxonomy**: Predefined list of ML/tech keywords

### 2. Resume Variant Generation

Each variant emphasizes different skills by:
- Reordering the Technical Skills section
- Prioritizing theme-relevant technologies
- Adding theme-specific summary enhancements

### 3. Job Matching

The matcher combines three scoring methods:
- **Category Score (40%)**: How well the job matches predefined categories
- **TF-IDF Similarity (35%)**: Textual similarity to resume content
- **Keyword Overlap (25%)**: Direct keyword matching

## Configuration

Edit the taxonomy in `src/keyword_engine.py` to customize keyword categories:

```python
TECH_TAXONOMY = {
    "mlops": ["docker", "kubernetes", "mlflow", ...],
    "nlp_llm": ["langchain", "transformers", "rag", ...],
    # Add your own categories
}
```

## Dependencies

- `pdfplumber`: PDF text extraction
- `spacy`: NLP and keyword extraction
- `scikit-learn`: TF-IDF and similarity scoring
- `click`: CLI framework
- `tqdm`: Progress bars
- `selenium` (optional): LinkedIn scraping

## License

MIT License

