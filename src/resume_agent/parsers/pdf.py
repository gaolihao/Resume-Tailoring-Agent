from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n\n".join(pages).strip()