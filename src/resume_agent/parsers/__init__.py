from __future__ import annotations

from pathlib import Path

from resume_agent.parsers.docx_parser import extract_text_from_docx
from resume_agent.parsers.pdf import extract_text_from_pdf


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md"}


def extract_resume_text(path: str | Path) -> str:
    """Load resume text from PDF, Word, or plain text."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Resume not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif suffix in {".docx", ".doc"}:
        if suffix == ".doc":
            raise ValueError(
                "Legacy .doc is not supported. Save as .docx or PDF, then retry."
            )
        text = extract_text_from_docx(file_path)
    elif suffix in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(
            f"Unsupported resume format '{suffix}'. "
            f"Use one of: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    cleaned = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not cleaned:
        raise ValueError(f"No extractable text found in {file_path.name}")
    return cleaned


def load_job_text(path_or_text: str) -> str:
    """Accept a job description file path or raw text."""
    candidate = Path(path_or_text)
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8").strip()
    return path_or_text.strip()