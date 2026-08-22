"""Streamlit chat UI for the resume tailoring LangGraph agent."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from resume_agent.graph import run_agent

ROOT = Path(__file__).resolve().parent
SAMPLE_RESUME = ROOT / "examples" / "sample_resume.txt"
SAMPLE_JOB = ROOT / "examples" / "sample_job.txt"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "resume_agent_uploads"


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Upload a resume in the sidebar (or use the sample), then paste a "
                    "job description here. I'll run the LangGraph agent — tool-calling "
                    "research, gap analysis, tailor, and quality review — and return a "
                    "truthful tailored resume."
                ),
            }
        ]
    if "resume_path" not in st.session_state:
        st.session_state.resume_path = None
    if "resume_label" not in st.session_state:
        st.session_state.resume_label = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def _save_upload(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


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


def _sidebar() -> None:
    st.sidebar.header("Setup")

    api_key = st.sidebar.text_input(
        "Gemini API key (optional)",
        type="password",
        help="Leave blank if GOOGLE_API_KEY is set in the environment (Render).",
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key.strip()

    if st.sidebar.button("Use sample resume"):
        st.session_state.resume_path = str(SAMPLE_RESUME)
        st.session_state.resume_label = "sample_resume.txt"
        st.sidebar.success("Sample resume loaded.")

    uploaded = st.sidebar.file_uploader(
        "Resume (.pdf, .docx, .txt, .md)",
        type=["pdf", "docx", "txt", "md"],
    )
    if uploaded is not None:
        path = _save_upload(uploaded)
        st.session_state.resume_path = str(path)
        st.session_state.resume_label = uploaded.name
        st.sidebar.success(f"Loaded `{uploaded.name}`")

    if st.session_state.resume_label:
        st.sidebar.caption(f"Active resume: **{st.session_state.resume_label}**")
    else:
        st.sidebar.warning("Upload a resume or use the sample.")

    st.sidebar.divider()
    st.sidebar.subheader("Or run sample end-to-end")
    if st.sidebar.button("Run with sample job posting"):
        if not st.session_state.resume_path:
            st.session_state.resume_path = str(SAMPLE_RESUME)
            st.session_state.resume_label = "sample_resume.txt"
        st.session_state.pending_job = SAMPLE_JOB.read_text(encoding="utf-8")


def _handle_job_submission(job_text: str) -> None:
    resume_path = st.session_state.get("resume_path")
    if not resume_path:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Please upload a resume in the sidebar first (or click **Use sample resume**).",
            }
        )
        return

    if not job_text.strip():
        return

    st.session_state.messages.append({"role": "user", "content": job_text.strip()})

    with st.spinner("Running LangGraph agent (research tools → analyze → tailor → review)..."):
        try:
            result = _run_pipeline(Path(resume_path), job_text)
            st.session_state.last_result = result
            reply = _format_result_markdown(result)
        except Exception as exc:
            reply = f"**Error:** {exc}\n\nCheck that `GOOGLE_API_KEY` is set and valid."
            st.session_state.last_result = None

    st.session_state.messages.append({"role": "assistant", "content": reply})


def main() -> None:
    st.set_page_config(
        page_title="Resume Agent",
        page_icon="📄",
        layout="wide",
    )
    _init_session()
    _sidebar()

    st.title("Resume Tailoring Agent")
    st.caption("LangGraph · Gemini · tool-calling research · guardrails")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = st.session_state.pop("pending_job", None)
    if pending:
        _handle_job_submission(pending)

    if prompt := st.chat_input("Paste a job description (or URL on its own line)..."):
        _handle_job_submission(prompt)
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
