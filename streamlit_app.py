"""Streamlit chat UI for the resume tailoring LangGraph agent."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from streamlit.elements.widgets.chat import ChatInputValue

from resume_agent.graph import run_agent

ROOT = Path(__file__).resolve().parent
SAMPLE_RESUME = ROOT / "examples" / "sample_resume.txt"
SAMPLE_JOB = ROOT / "examples" / "sample_job.txt"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "resume_agent_uploads"

RESUME_TYPES = ["pdf", "docx", "txt", "md"]
WELCOME = (
    "Hi — I tailor resumes to job postings using a LangGraph agent with tool-calling "
    "and guardrails.\n\n"
    "**Step 1:** Attach your resume here (PDF, DOCX, TXT) or type `sample`.\n\n"
    "**Step 2:** Paste a job description or posting URL."
)


def _init_session() -> None:
    defaults = {
        "messages": [{"role": "assistant", "content": WELCOME}],
        "resume_path": None,
        "resume_label": None,
        "last_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _has_resume() -> bool:
    return bool(st.session_state.get("resume_path"))


def _save_upload(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def _append(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def _set_resume(path: Path, label: str) -> None:
    st.session_state.resume_path = str(path)
    st.session_state.resume_label = label
    _append(
        "assistant",
        f"Resume loaded: **{label}**. Now paste a job description or URL.",
    )


def _use_sample_resume() -> None:
    _set_resume(SAMPLE_RESUME, "sample_resume.txt")


def _format_result_markdown(result: dict) -> str:
    parts: list[str] = []

    trace = result.get("tool_trace") or []
    if trace:
        parts.append("### Tool-calling trace\n")
        parts.extend(f"- `{line}`" for line in trace[:15])
        parts.append("")

    evidence = result.get("evidence_notes")
    if evidence:
        parts.append("### Evidence checks\n")
        parts.append(evidence[:2000])
        parts.append("")

    gaps = result.get("gap_analysis")
    if gaps:
        parts.append("### Gap analysis\n")
        parts.append(f"**Match score:** {gaps.match_score}/100\n")
        parts.append(
            f"**Matched:** {', '.join(gaps.matched_keywords[:12]) or '—'}\n"
        )
        parts.append(
            f"**Missing:** {', '.join(gaps.missing_keywords[:12]) or '—'}\n"
        )
        parts.append(f"**Positioning:** {gaps.positioning_advice}\n")

    review = result.get("quality_review")
    if review:
        status = "Approved" if review.approved else "Revised with notes"
        parts.append(f"### Quality review: {status}\n")
        if review.issues:
            parts.extend(f"- {issue}" for issue in review.issues[:6])
        parts.append("")

    tailored_md = result.get("tailored_markdown")
    if tailored_md:
        parts.append("### Tailored resume\n")
        parts.append(tailored_md)

    return "\n".join(parts).strip()


def _run_pipeline(resume_path: Path, job_input: str) -> dict:
    out_dir = UPLOAD_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_agent(
        resume_path=str(resume_path),
        job_path_or_text=job_input.strip(),
        output_path=str(out_dir),
    )


def _read_uploaded_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".txt", ".md"}:
        return uploaded_file.getvalue().decode("utf-8", errors="replace").strip()
    raise ValueError("Job file must be .txt or .md — paste the description as text instead.")


def _handle_resume_file(uploaded_file) -> None:
    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix not in RESUME_TYPES:
        _append(
            "assistant",
            f"Unsupported file type. Please upload: {', '.join(RESUME_TYPES)}.",
        )
        return
    path = _save_upload(uploaded_file)
    _set_resume(path, uploaded_file.name)


def _handle_job_submission(job_text: str) -> None:
    if not _has_resume():
        _append(
            "assistant",
            "Attach your resume first (or type `sample`), then send the job description.",
        )
        return

    job_text = job_text.strip()
    if not job_text:
        return

    _append("user", job_text)

    with st.spinner("Running LangGraph agent..."):
        try:
            result = _run_pipeline(Path(st.session_state.resume_path), job_text)
            st.session_state.last_result = result
            reply = _format_result_markdown(result)
        except Exception as exc:
            reply = f"**Error:** {exc}\n\nCheck that `GOOGLE_API_KEY` is set and valid."
            st.session_state.last_result = None

    _append("assistant", reply)


def _handle_chat_input(prompt: ChatInputValue) -> None:
    text = (prompt.text or "").strip()
    files = prompt.files or []

    if text.lower() in {"sample", "demo"} and not _has_resume():
        _append("user", text)
        _use_sample_resume()
        if text.lower() == "demo":
            _handle_job_submission(SAMPLE_JOB.read_text(encoding="utf-8"))
        return

    if not _has_resume():
        if files:
            _append("user", f"📎 {files[0].name}")
            _handle_resume_file(files[0])
            if text:
                _append("user", text)
                _handle_job_submission(text)
            return
        if text.lower() == "sample":
            _append("user", text)
            _use_sample_resume()
            return
        if text:
            _append("user", text)
        _append(
            "assistant",
            "Attach your resume file here, or type `sample` to use the demo resume.",
        )
        return

    # Resume is loaded — accept job text and/or a JD text file.
    if files:
        _append("user", f"📎 {files[0].name}")
        try:
            job_from_file = _read_uploaded_text(files[0])
        except ValueError as exc:
            _append("assistant", str(exc))
            return
        _handle_job_submission(job_from_file)
        return

    if text:
        _handle_job_submission(text)


def _chat_placeholder() -> str:
    if not _has_resume():
        return "Attach resume (PDF, DOCX, TXT) or type sample..."
    return "Paste job description or URL..."


def main() -> None:
    st.set_page_config(page_title="Resume Agent", page_icon="📄", layout="centered")
    _init_session()

    st.title("Resume Tailoring Agent")
    st.caption("LangGraph · Gemini · tool-calling · guardrails")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        _chat_placeholder(),
        accept_file=True,
        file_type=RESUME_TYPES,
    )
    if prompt:
        _handle_chat_input(prompt)
        st.rerun()

    result = st.session_state.get("last_result")
    if result:
        out_dir = Path(result.get("output_path", ""))
        if out_dir.is_dir():
            md_files = sorted(out_dir.glob("resume_*.md"))
            docx_files = sorted(out_dir.glob("resume_*.docx"))
            cols = st.columns(2)
            if md_files:
                cols[0].download_button(
                    "Download Markdown",
                    data=md_files[-1].read_bytes(),
                    file_name=md_files[-1].name,
                    mime="text/markdown",
                )
            if docx_files:
                cols[1].download_button(
                    "Download DOCX",
                    data=docx_files[-1].read_bytes(),
                    file_name=docx_files[-1].name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )


if __name__ == "__main__":
    main()
