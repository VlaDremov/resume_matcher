"""
LaTeX Compiler Module.

Provides functionality to compile LaTeX files to PDF using pdflatex.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def check_pdflatex_installed() -> bool:
    """
    Check if pdflatex is installed and available in PATH.

    Returns:
        True if pdflatex is available, False otherwise.
    """
    return shutil.which("pdflatex") is not None


def compile_latex_to_pdf(
    tex_file: str | Path,
    output_dir: Optional[str | Path] = None,
    clean_aux: bool = True,
) -> Optional[Path]:
    """
    Compile a LaTeX file to PDF using pdflatex.

    Args:
        tex_file: Path to the .tex file to compile.
        output_dir: Directory to place the output PDF.
                   If None, uses the same directory as the tex file.
        clean_aux: Whether to clean up auxiliary files after compilation.

    Returns:
        Path to the generated PDF file, or None if compilation failed.

    Raises:
        FileNotFoundError: If the tex file doesn't exist.
        RuntimeError: If pdflatex is not installed.
    """
    tex_file = Path(tex_file).resolve()

    if not tex_file.exists():
        raise FileNotFoundError(f"LaTeX file not found: {tex_file}")

    if not check_pdflatex_installed():
        raise RuntimeError(
            "pdflatex is not installed. Please install a LaTeX distribution:\n"
            "  - macOS: brew install --cask mactex-no-gui\n"
            "  - Ubuntu: sudo apt-get install texlive-latex-base\n"
            "  - Windows: Install MiKTeX from https://miktex.org/"
        )

    # * Determine output directory
    if output_dir is None:
        output_dir = tex_file.parent
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # * Run pdflatex
    # * We run it twice to resolve references properly
    success = _run_pdflatex(tex_file, output_dir)

    if not success:
        return None

    # * Run again for proper reference resolution
    success = _run_pdflatex(tex_file, output_dir)

    if not success:
        return None

    # * Check if PDF was created
    pdf_name = tex_file.stem + ".pdf"
    pdf_path = output_dir / pdf_name

    if not pdf_path.exists():
        print(f"! PDF was not created: {pdf_path}")
        return None

    # * Clean auxiliary files
    if clean_aux:
        _clean_auxiliary_files(output_dir, tex_file.stem)

    return pdf_path


def _run_pdflatex(tex_file: Path, output_dir: Path) -> bool:
    """
    Run pdflatex on a tex file.

    Args:
        tex_file: Path to the .tex file.
        output_dir: Output directory for the PDF.

    Returns:
        True if compilation succeeded, False otherwise.
    """
    try:
        # * Build command
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",  # * Don't stop on errors
            "-halt-on-error",  # * Stop on first error
            f"-output-directory={output_dir}",
            str(tex_file),
        ]

        # * Run pdflatex
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # * 60 second timeout
            cwd=tex_file.parent,  # * Run from tex file's directory
        )

        if result.returncode != 0:
            print(f"! pdflatex failed with return code {result.returncode}")
            # * Print last few lines of output for debugging
            error_lines = result.stdout.split("\n")[-20:]
            for line in error_lines:
                if line.strip():
                    print(f"  {line}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print("! pdflatex timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"! Error running pdflatex: {e}")
        return False


def _clean_auxiliary_files(directory: Path, base_name: str) -> None:
    """
    Clean up LaTeX auxiliary files.

    Args:
        directory: Directory containing auxiliary files.
        base_name: Base name of the tex file (without extension).
    """
    aux_extensions = [".aux", ".log", ".out", ".toc", ".lof", ".lot", ".fls", ".fdb_latexmk", ".synctex.gz"]

    for ext in aux_extensions:
        aux_file = directory / (base_name + ext)
        if aux_file.exists():
            try:
                aux_file.unlink()
            except Exception:
                pass  # * Ignore cleanup errors


def compile_all_variants(
    variants_dir: str | Path,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Optional[Path]]:
    """
    Compile all LaTeX resume variants in a directory.

    Args:
        variants_dir: Directory containing .tex files.
        output_dir: Directory to place output PDFs.
                   If None, uses the same directory as variants.

    Returns:
        Dictionary mapping tex filename to output PDF path (or None if failed).
    """
    variants_dir = Path(variants_dir)

    if not variants_dir.exists():
        raise FileNotFoundError(f"Variants directory not found: {variants_dir}")

    if output_dir is None:
        output_dir = variants_dir
    else:
        output_dir = Path(output_dir)

    results = {}

    tex_files = list(variants_dir.glob("*.tex"))

    if not tex_files:
        print(f"No .tex files found in {variants_dir}")
        return results

    print(f"Compiling {len(tex_files)} LaTeX files...")

    for tex_file in tex_files:
        print(f"  Compiling {tex_file.name}...", end=" ")

        try:
            pdf_path = compile_latex_to_pdf(tex_file, output_dir)
            if pdf_path:
                print(f"OK -> {pdf_path.name}")
            else:
                print("FAILED")
            results[tex_file.name] = pdf_path
        except Exception as e:
            print(f"ERROR: {e}")
            results[tex_file.name] = None

    # * Summary
    success_count = sum(1 for p in results.values() if p is not None)
    print(f"\nCompilation complete: {success_count}/{len(results)} succeeded")

    return results


def compile_with_fallback(
    tex_file: str | Path,
    output_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    """
    Compile LaTeX with fallback to temporary directory if output dir has issues.

    Args:
        tex_file: Path to the .tex file.
        output_dir: Desired output directory.

    Returns:
        Path to the generated PDF, or None if failed.
    """
    tex_file = Path(tex_file).resolve()

    # * Try normal compilation first
    try:
        pdf_path = compile_latex_to_pdf(tex_file, output_dir)
        if pdf_path:
            return pdf_path
    except Exception as e:
        print(f"? Normal compilation failed: {e}")

    # * Try with temporary directory
    print("? Attempting compilation in temporary directory...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # * Copy tex file to temp dir
            temp_tex = temp_dir / tex_file.name
            shutil.copy2(tex_file, temp_tex)

            # * Also copy any input files (like glyphtounicode)
            for input_file in tex_file.parent.glob("*.sty"):
                shutil.copy2(input_file, temp_dir / input_file.name)

            # * Compile in temp dir
            pdf_path = compile_latex_to_pdf(temp_tex, temp_dir, clean_aux=False)

            if pdf_path and pdf_path.exists():
                # * Copy result to desired output
                final_output_dir = Path(output_dir) if output_dir else tex_file.parent
                final_output_dir.mkdir(parents=True, exist_ok=True)
                final_pdf = final_output_dir / pdf_path.name
                shutil.copy2(pdf_path, final_pdf)
                return final_pdf

    except Exception as e:
        print(f"! Fallback compilation also failed: {e}")

    return None


def get_latex_version() -> Optional[str]:
    """
    Get the version of pdflatex installed.

    Returns:
        Version string, or None if pdflatex is not installed.
    """
    if not check_pdflatex_installed():
        return None

    try:
        result = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            # * First line usually contains version
            first_line = result.stdout.split("\n")[0]
            return first_line.strip()

    except Exception:
        pass

    return None

