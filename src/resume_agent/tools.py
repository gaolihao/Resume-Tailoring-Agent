from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool

from resume_agent.parsers import extract_resume_text, load_job_text

_USER_AGENT = "resume-agent/0.1 (+https://github.com/local/resume-agent)"


@tool
def parse_resume(path: str) -> str:
    """Parse a local resume file (.pdf, .docx, .txt, .md) and return plain text."""
    return extract_resume_text(path)


@tool
def load_job_file(path: str) -> str:
    """Load a local job-description text file and return its contents."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return json.dumps({"error": f"Job file not found: {path}"})
    return load_job_text(str(file_path))


@tool
def fetch_job_from_url(url: str) -> str:
    """Fetch a job posting from an http(s) URL and return cleaned plain text."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return json.dumps({"error": f"Invalid URL: {url}"})

    try:
        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=20.0,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return json.dumps({"error": f"Failed to fetch URL: {exc}"})

    text = _html_to_text(response.text)
    if not text:
        return json.dumps({"error": "Fetched page had no extractable text"})
    # Keep tool outputs bounded for the model context.
    return text[:20000]


@tool
def evidence_check(claim: str, resume_text: str) -> str:
    """
    Deterministically check whether a claim appears supported by resume text.

    Uses token overlap heuristics (not another LLM). Returns a JSON string with
    supported/partial/unsupported status and matched terms.
    """
    claim_tokens = _tokens(claim)
    resume_tokens = set(_tokens(resume_text))
    if not claim_tokens:
        return json.dumps(
            {
                "claim": claim,
                "status": "unsupported",
                "matched_terms": [],
                "coverage": 0.0,
                "note": "Empty claim",
            }
        )

    matched = [t for t in claim_tokens if t in resume_tokens]
    coverage = len(matched) / len(claim_tokens)

    if coverage >= 0.6:
        status = "supported"
    elif coverage >= 0.3:
        status = "partial"
    else:
        status = "unsupported"

    return json.dumps(
        {
            "claim": claim,
            "status": status,
            "matched_terms": matched[:20],
            "coverage": round(coverage, 3),
            "note": (
                "Heuristic overlap only — does not prove the candidate did the work, "
                "only that related wording exists in the resume."
            ),
        }
    )


RESEARCH_TOOLS = [
    parse_resume,
    load_job_file,
    fetch_job_from_url,
    evidence_check,
]


def _tokens(text: str) -> list[str]:
    stop = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "be",
        "as",
        "at",
        "from",
        "that",
        "this",
        "it",
        "your",
        "you",
        "our",
        "we",
    }
    raw = re.findall(r"[a-z0-9][a-z0-9+.#/-]*", text.lower())
    return [t for t in raw if t not in stop and len(t) > 1]


def _html_to_text(html: str) -> str:
    # Lightweight cleanup without pulling in BeautifulSoup.
    cleaned = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    cleaned = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", cleaned)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"&amp;", "&", cleaned)
    cleaned = re.sub(r"&lt;", "<", cleaned)
    cleaned = re.sub(r"&gt;", ">", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
