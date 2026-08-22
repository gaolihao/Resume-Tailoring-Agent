"""Streamlit chat UI for the resume tailoring LangGraph agent."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from streamlit.elements.widgets.chat import ChatInputValue

from resume_agent.streaming import format_agent_event, stream_agent

ROOT = Path(__file__).resolve().parent
SAMPLE_RESUME = ROOT / "examples" / "sample_resume.txt"
SAMPLE_JOB = ROOT / "examples" / "sample_job.txt"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "resume_agent_uploads"

RESUME_TYPES = ["pdf", "docx", "txt", "md"]

WELCOME = (
    "Hi! I tailor resumes to job postings using a LangGraph agent — with "
    "tool-calling research and guardrails so nothing gets invented.\n\n"
    "First, attach your resume below (PDF, DOCX, or TXT). "
    "You can also type `sample` to try the demo resume."
)

PROMPT_NEED_RESUME = (
    "Please attach your resume file here, or type `sample` to use the demo resume."
)

PROMPT_NEED_JOB = (
    "Now paste the job description you're applying to, send a link to the posting, "
    "or type `sample` to use the demo job."
)


def _init_session() -> None:
    defaults = {
        "messages": [{"role": "assistant", "content": WELCOME}],
        "resume_path": None,
        "resume_label": None,
        "last_result": None,
        "pending_job": None,
        "job_run_phase": None,
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
        f"Got it — I've loaded **{label}**.\n\n{PROMPT_NEED_JOB}",
    )


def _use_sample_resume() -> None:
    _set_resume(SAMPLE_RESUME, "sample_resume.txt")


def _format_result_markdown(result: dict) -> str:
    parts: list[str] = ["### Results", ""]

    gaps = result.get("gap_analysis")
    if gaps:
        parts.append("#### Screening fit")
        parts.append(f"- **Match score:** {gaps.match_score}/100")
        parts.append(f"- **Matched:** {', '.join(gaps.matched_keywords[:12]) or '—'}")
        parts.append(f"- **Missing:** {', '.join(gaps.missing_keywords[:12]) or '—'}")
        parts.append(f"- **Positioning:** {gaps.positioning_advice}")
        parts.append("")

    tailored = result.get("tailored_resume")
    if tailored:
        parts.append("#### Changes made")
        parts.extend(f"- {change}" for change in tailored.changes_made[:8])
        parts.append("")

    review = result.get("quality_review")
    if review:
        status = "Approved" if review.approved else "Revised with notes"
        parts.append(f"#### Quality review: {status}")
        if review.issues:
            parts.extend(f"- {issue}" for issue in review.issues[:6])
        parts.append("")

    parts.append("Download your tailored resume below (same format as your upload).")
    return "\n".join(parts).strip()


def _read_uploaded_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".txt", ".md"}:
        return uploaded_file.getvalue().decode("utf-8", errors="replace").strip()
    raise ValueError(
        "For the job description, paste the text directly or upload a .txt / .md file."
    )


def _handle_resume_file(uploaded_file) -> None:
    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix not in RESUME_TYPES:
        _append(
            "assistant",
            f"That file type isn't supported. Please upload: {', '.join(RESUME_TYPES)}.",
        )
        return
    path = _save_upload(uploaded_file)
    _set_resume(path, uploaded_file.name)


def _format_job_preview(job_text: str) -> str:
    return f"**Job description:**\n\n{job_text.strip()}"


def _queue_job_run(
    job_text: str,
    *,
    user_display: str,
    show_job_preview: bool = False,
    append_user: bool = True,
) -> None:
    """Show the user's input (and job text when using sample) before running."""
    if append_user:
        _append("user", user_display)
    if show_job_preview:
        _append("assistant", _format_job_preview(job_text))
    st.session_state.pending_job = job_text.strip()
    st.session_state.job_run_phase = "preview"


def _process_pending_job() -> None:
    job_text = st.session_state.get("pending_job")
    if not job_text or not _has_resume():
        return

    phase = st.session_state.get("job_run_phase")

    if phase == "preview":
        st.session_state.job_run_phase = "running"
        st.rerun()

    if phase != "running":
        return

    job_text = st.session_state.pop("pending_job")
    st.session_state.job_run_phase = None

    progress_placeholder = st.empty()
    resume_path = Path(st.session_state.resume_path)
    out_dir = UPLOAD_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    final: dict = {}

    def trace_stream():
        for event in stream_agent(
            resume_path=str(resume_path),
            job_path_or_text=job_text.strip(),
            output_path=str(out_dir),
        ):
            if event.get("event") == "complete":
                final["result"] = event.get("state")
                continue
            line = format_agent_event(event)
            if line:
                yield line + "\n"

        result = final.get("result")
        if result is None:
            raise RuntimeError("Agent finished without returning final state.")

        yield "\n"
        yield _format_result_markdown(result)

    with progress_placeholder.container():
        with st.chat_message("assistant"):
            try:
                full_trace = st.write_stream(trace_stream)
                st.session_state.last_result = final["result"]
                _append("assistant", full_trace.strip())
            except Exception as exc:
                _append(
                    "assistant",
                    f"**Error:** {exc}\n\nCheck that `GOOGLE_API_KEY` is set and valid.",
                )
                st.session_state.last_result = None

    progress_placeholder.empty()
    st.rerun()


def _handle_job_submission(
    job_text: str,
    *,
    user_display: str | None = None,
    show_job_preview: bool = False,
) -> None:
    if not _has_resume():
        _append("assistant", PROMPT_NEED_RESUME)
        return

    job_text = job_text.strip()
    if not job_text:
        return

    display = user_display if user_display is not None else job_text
    _queue_job_run(
        job_text,
        user_display=display,
        show_job_preview=show_job_preview,
    )
    st.rerun()


def _handle_resume_step(text: str, files: list) -> None:
    """Step 1: accept only a resume upload or `sample`."""
    if text.lower() == "demo":
        _append("user", text)
        _use_sample_resume()
        sample_job = SAMPLE_JOB.read_text(encoding="utf-8")
        _queue_job_run(
            sample_job,
            user_display=text,
            show_job_preview=True,
            append_user=False,
        )
        st.rerun()
        return

    if text.lower() == "sample":
        _append("user", text)
        _use_sample_resume()
        return

    if files:
        _append("user", f"📎 {files[0].name}")
        _handle_resume_file(files[0])
        if text:
            st.session_state.messages[-1]["content"] += (
                "\n\nSend the job description in your next message."
            )
        return

    if text:
        _append("user", text)
    _append("assistant", PROMPT_NEED_RESUME)


def _handle_job_step(text: str, files: list) -> None:
    """Step 2: accept job description text, URL, JD file, or `sample`."""
    if files:
        try:
            job_from_file = _read_uploaded_text(files[0])
        except ValueError as exc:
            _append("assistant", str(exc))
            return
        _handle_job_submission(
            job_from_file,
            user_display=f"📎 {files[0].name}",
            show_job_preview=True,
        )
        return

    if text.lower() == "sample":
        sample_job = SAMPLE_JOB.read_text(encoding="utf-8")
        _handle_job_submission(
            sample_job,
            user_display=text,
            show_job_preview=True,
        )
        return

    if text:
        _handle_job_submission(text)
        return

    _append("assistant", PROMPT_NEED_JOB)


def _handle_chat_input(prompt: ChatInputValue) -> None:
    text = (prompt.text or "").strip()
    files = prompt.files or []

    if not _has_resume():
        _handle_resume_step(text, files)
        return

    _handle_job_step(text, files)


def _chat_placeholder() -> str:
    if not _has_resume():
        return "Attach your resume, or type sample..."
    return "Paste the job description, URL, or type sample..."


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

    _process_pending_job()

    result = st.session_state.get("last_result")
    if result:
        primary = result.get("output_file_path")
        report = result.get("output_report_path")
        cols = st.columns(2)

        if primary and Path(primary).exists():
            primary_path = Path(primary)
            mime_by_suffix = {
                ".pdf": "application/pdf",
                ".docx": (
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
                ".txt": "text/plain",
                ".md": "text/markdown",
            }
            download_name = primary_path.name
            if st.session_state.get("resume_label"):
                stem = Path(st.session_state.resume_label).stem
                download_name = f"{stem}_tailored{primary_path.suffix}"

            cols[0].download_button(
                f"Download tailored resume ({primary_path.suffix.lstrip('.').upper()})",
                data=primary_path.read_bytes(),
                file_name=download_name,
                mime=mime_by_suffix.get(primary_path.suffix.lower(), "application/octet-stream"),
            )

        if report and Path(report).exists():
            cols[1].download_button(
                "Download change report (Markdown)",
                data=Path(report).read_bytes(),
                file_name=Path(report).name,
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
