"""
Data Extraction Module.

Provides functionality to extract text from PDF files (LinkedIn exports)
and parse LaTeX resume files.
"""

import re
from pathlib import Path
from typing import Optional

import pdfplumber


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text content as a single string.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a valid PDF.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    text_content = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)

    return "\n\n".join(text_content)


def parse_latex_resume(latex_path: str | Path) -> dict:
    """
    Parse a LaTeX resume file and extract structured content.

    Args:
        latex_path: Path to the LaTeX file.

    Returns:
        Dictionary containing parsed resume sections:
        - header: Contact information
        - summary: Professional summary
        - experience: List of job entries
        - education: Education entries
        - skills: Technical skills by category

    Raises:
        FileNotFoundError: If the LaTeX file does not exist.
    """
    latex_path = Path(latex_path)

    if not latex_path.exists():
        raise FileNotFoundError(f"LaTeX file not found: {latex_path}")

    with open(latex_path, "r", encoding="utf-8") as f:
        content = f.read()

    parsed = {
        "raw_content": content,
        "header": _extract_header(content),
        "summary": _extract_summary(content),
        "experience": _extract_experience(content),
        "education": _extract_education(content),
        "skills": _extract_skills(content),
    }

    return parsed


def _extract_header(content: str) -> dict:
    """Extract header/contact information from LaTeX content."""
    header = {
        "name": "",
        "phone": "",
        "location": "",
        "email": "",
        "linkedin": "",
    }

    # * Extract name from \textbf{\Huge \scshape Name}
    name_match = re.search(r"\\textbf\{\\Huge\\s*\\scshape\s+([^}]+)\}", content)
    if name_match:
        header["name"] = name_match.group(1).strip()

    # * Extract phone number
    phone_match = re.search(r"\\small\s*(\+[\d\s]+)", content)
    if phone_match:
        header["phone"] = phone_match.group(1).strip()

    # * Extract location
    location_match = re.search(r"\$\|\$\s*\\small\s+([^$]+)\s*\$\|\$", content)
    if location_match:
        header["location"] = location_match.group(1).strip()

    # * Extract email
    email_match = re.search(r"\\href\{mailto:[^}]+\}\{\\underline\{([^}]+)\}\}", content)
    if email_match:
        header["email"] = email_match.group(1).strip()

    # * Extract LinkedIn
    linkedin_match = re.search(r"\\href\{(https://linkedin\.com/[^}]+)\}", content)
    if linkedin_match:
        header["linkedin"] = linkedin_match.group(1).strip()

    return header


def _extract_summary(content: str) -> str:
    """Extract professional summary from LaTeX content."""
    # * Summary is typically between header and first section
    summary_match = re.search(
        r"\\end\{center\}\s*\n\s*(.+?)(?=\\section|%-----------)",
        content,
        re.DOTALL,
    )

    if summary_match:
        summary = summary_match.group(1).strip()
        # * Clean up LaTeX formatting
        summary = re.sub(r"\\textbf\{([^}]+)\}", r"\1", summary)
        summary = summary.replace("\\", "").strip()
        return summary

    return ""


def _extract_experience(content: str) -> list[dict]:
    """Extract experience entries from LaTeX content."""
    experience = []

    # * Find the Experience section
    exp_section_match = re.search(
        r"\\section\{Experience\}(.*?)(?=\\section|\\end\{document\})",
        content,
        re.DOTALL,
    )

    if not exp_section_match:
        return experience

    exp_content = exp_section_match.group(1)

    # * Pattern to match resumeSubheading entries
    subheading_pattern = re.compile(
        r"\\resumeSubheading\s*\{([^}]*)\}\{([^}]*)\}\s*\{([^}]*)\}\{([^}]*)\}",
        re.DOTALL,
    )

    # * Find all job entries
    for match in subheading_pattern.finditer(exp_content):
        title = match.group(1).strip()
        dates = match.group(2).strip()
        company_raw = match.group(3).strip()
        location = match.group(4).strip()

        # * Extract company name from href if present
        company_match = re.search(r"\\href\{[^}]*\}\{([^}]*)\}", company_raw)
        if company_match:
            company = company_match.group(1).strip()
        else:
            company = company_raw

        # * Extract bullets for this job
        # * Find content between this subheading and the next one (or end)
        start_pos = match.end()
        next_match = subheading_pattern.search(exp_content, start_pos)
        end_pos = next_match.start() if next_match else len(exp_content)

        bullets_content = exp_content[start_pos:end_pos]
        bullets = _extract_bullets(bullets_content)

        experience.append({
            "title": title,
            "company": company,
            "dates": dates,
            "location": location,
            "bullets": bullets,
        })

    return experience


def _extract_bullets(content: str) -> list[str]:
    """Extract bullet points from LaTeX content."""
    bullets = []

    # * Pattern to match \resumeItem{...}
    item_pattern = re.compile(r"\\resumeItem\{(.+?)\}(?=\s*\\resumeItem|\s*\\resumeItemListEnd)", re.DOTALL)

    for match in item_pattern.finditer(content):
        bullet_text = match.group(1).strip()
        # * Clean up LaTeX formatting
        bullet_text = _clean_latex_text(bullet_text)
        if bullet_text:
            bullets.append(bullet_text)

    return bullets


def _extract_education(content: str) -> list[dict]:
    """Extract education entries from LaTeX content."""
    education = []

    # * Find the Education section
    edu_section_match = re.search(
        r"\\section\{Education\}(.*?)(?=\\section|\\end\{document\})",
        content,
        re.DOTALL,
    )

    if not edu_section_match:
        return education

    edu_content = edu_section_match.group(1)

    # * Pattern to match resumeSubheading entries for education
    subheading_pattern = re.compile(
        r"\\resumeSubheading\s*\{([^}]*)\}\{([^}]*)\}\s*\{([^}]*)\}\{([^}]*)\}",
        re.DOTALL,
    )

    for match in subheading_pattern.finditer(edu_content):
        institution = match.group(1).strip()
        location = match.group(2).strip()
        degree = match.group(3).strip()
        dates = match.group(4).strip()

        education.append({
            "institution": institution,
            "location": location,
            "degree": degree,
            "dates": dates,
        })

    return education


def _extract_skills(content: str) -> dict[str, list[str]]:
    """Extract technical skills from LaTeX content."""
    skills = {}

    # * Find the Technical Skills section
    skills_section_match = re.search(
        r"\\section\{Technical Skills\}(.*?)(?=\\section|\\end\{document\})",
        content,
        re.DOTALL,
    )

    if not skills_section_match:
        return skills

    skills_content = skills_section_match.group(1)

    # * Pattern to match skill categories: \textbf{Category}{: skill1, skill2}
    category_pattern = re.compile(r"\\textbf\{([^}]+)\}\{:\s*([^}\\]+)\}", re.DOTALL)

    for match in category_pattern.finditer(skills_content):
        category = match.group(1).strip()
        skills_list_raw = match.group(2).strip()

        # * Split by comma and clean up
        skills_list = [s.strip() for s in skills_list_raw.split(",") if s.strip()]

        # * Clean up LaTeX formatting in skill names
        skills_list = [_clean_latex_text(s) for s in skills_list]

        skills[category] = skills_list

    return skills


def _clean_latex_text(text: str) -> str:
    """Remove LaTeX formatting from text."""
    # * Remove href commands but keep the display text
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)

    # * Remove underline
    text = re.sub(r"\\underline\{([^}]*)\}", r"\1", text)

    # * Remove textbf
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)

    # * Remove emph/textit
    text = re.sub(r"\\(?:emph|textit)\{([^}]*)\}", r"\1", text)

    # * Remove percentage escapes
    text = text.replace("\\%", "%")

    # * Remove dollar signs used for spacing
    text = text.replace("$|$", "|")

    # * Remove remaining backslashes before common characters
    text = re.sub(r"\\([#$%&_{}])", r"\1", text)

    # * Clean up multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_vacancy_files(vacancies_dir: str | Path) -> dict[str, str]:
    """
    Load all vacancy/job description files from a directory.

    Args:
        vacancies_dir: Path to the directory containing vacancy files.

    Returns:
        Dictionary mapping filename to file content.
    """
    vacancies_dir = Path(vacancies_dir)

    if not vacancies_dir.exists():
        raise FileNotFoundError(f"Vacancies directory not found: {vacancies_dir}")

    vacancies = {}

    for file_path in vacancies_dir.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  # * Only include non-empty files
                    vacancies[file_path.stem] = content
        except (IOError, UnicodeDecodeError) as e:
            print(f"! Warning: Could not read {file_path}: {e}")

    return vacancies


def get_all_text_for_analysis(
    resume_path: Optional[str | Path] = None,
    pdf_path: Optional[str | Path] = None,
) -> str:
    """
    Combine text from resume and LinkedIn PDF for analysis.

    Args:
        resume_path: Path to LaTeX resume file.
        pdf_path: Path to LinkedIn PDF export.

    Returns:
        Combined text content.
    """
    text_parts = []

    if resume_path:
        parsed = parse_latex_resume(resume_path)

        # * Add summary
        if parsed["summary"]:
            text_parts.append(parsed["summary"])

        # * Add experience bullets
        for job in parsed["experience"]:
            text_parts.append(f"{job['title']} at {job['company']}")
            text_parts.extend(job["bullets"])

        # * Add skills
        for category, skills in parsed["skills"].items():
            text_parts.append(f"{category}: {', '.join(skills)}")

    if pdf_path and Path(pdf_path).exists():
        pdf_text = extract_text_from_pdf(pdf_path)
        text_parts.append(pdf_text)

    return "\n".join(text_parts)

