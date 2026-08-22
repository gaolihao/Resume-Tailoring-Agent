from pathlib import Path


def extract_text_from_pdf(path: Path) -> str:
    """Extract plain text from a PDF using pymupdf, with pypdf as fallback."""
    errors: list[str] = []

    try:
        import fitz  # pymupdf

        doc = fitz.open(str(path))
        try:
            pages = [page.get_text("text") for page in doc]
        finally:
            doc.close()
        text = "\n\n".join(pages).strip()
        if text:
            return text
        errors.append("pymupdf returned no text")
    except Exception as exc:
        errors.append(f"pymupdf: {exc}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n\n".join(pages).strip()
        if text:
            return text
        errors.append("pypdf returned no text")
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    detail = "; ".join(errors) if errors else "unknown error"
    raise ValueError(
        f"Could not extract text from PDF '{path.name}'. "
        f"The file may be scanned/image-only or encrypted. ({detail})"
    )
