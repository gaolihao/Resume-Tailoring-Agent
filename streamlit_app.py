"""Streamlit chat UI for the resume tailoring LangGraph agent."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from streamlit.elements.widgets.chat import ChatInputValue

from resume_agent.streaming import StepwiseAgent

ROOT = Path(__file__).resolve().parent
SAMPLE_RESUME = ROOT / "examples" / "sample_resume.txt"
SAMPLE_JOB = ROOT / "examples" / "sample_job.txt"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "resume_agent_uploads"

RESUME_TYPES = ["pdf", "docx", "txt", "md"]

WELCOME = (
    "Hi! I tailor resumes to job postings using a LangGraph agent — with "
    "tool-calling research and guardrails so nothing gets invented.\n\n"
    "First, attach your resume below (PDF, DOCX, or TXT), or click "
    "**Use sample resume** to try the demo."
)

PROMPT_NEED_RESUME = (
    "Please attach your resume file here, or click **Use sample resume** below."
)

PROMPT_NEED_JOB = (
    "Now paste the job description you're applying to, send a link to the posting, "
    "or click **Use sample job** below."
)


def _init_session() -> None:
    defaults = {
        "messages": [{"role": "assistant", "content": WELCOME}],
        "resume_path": None,
        "resume_label": None,
        "last_result": None,
        "pending_job": None,
        "job_run_phase": None,
        "agent_runner": None,
        "agent_executing_step": False,
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


def _submit_sample_resume(user_display: str = "Use sample resume") -> None:
    _append("user", user_display)
    _use_sample_resume()


def _submit_sample_job(user_display: str = "Use sample job") -> None:
    sample_job = SAMPLE_JOB.read_text(encoding="utf-8")
    _handle_job_submission(
        sample_job,
        user_display=user_display,
        show_job_preview=True,
    )


def _agent_busy() -> bool:
    return bool(
        st.session_state.get("pending_job") or st.session_state.get("agent_runner")
    )


def _render_quick_actions() -> None:
    if _agent_busy():
        return

    _, center, _ = st.columns([1, 2, 1])
    with center:
        if not _has_resume():
            if st.button("Use sample resume", use_container_width=True, key="sample_resume"):
                _submit_sample_resume()
                st.rerun()
        elif not st.session_state.get("last_result"):
            if st.button("Use sample job", use_container_width=True, key="sample_job"):
                _submit_sample_job()


PROGRESS_MARKER = "<!--agent-progress-->"


def _progress_html(label: str) -> str:
    safe_label = label.rstrip(".")
    return f"""{PROGRESS_MARKER}
<style>
@keyframes agentDotPulse {{
  0%, 80%, 100% {{ opacity: 0.2; }}
  40% {{ opacity: 1; }}
}}
.agent-dot {{
  animation: agentDotPulse 1.4s infinite ease-in-out both;
  font-weight: bold;
}}
.agent-dot:nth-child(2) {{ animation-delay: 0.2s; }}
.agent-dot:nth-child(3) {{ animation-delay: 0.4s; }}
</style>
<p><strong>{safe_label}<span aria-hidden="true"><span class="agent-dot">.</span><span class="agent-dot">.</span><span class="agent-dot">.</span></span></strong></p>"""


def _is_progress_message(content: str) -> bool:
    return content.startswith(PROGRESS_MARKER)


def _render_message(content: str) -> None:
    if _is_progress_message(content):
        st.html(content[len(PROGRESS_MARKER) :].strip(), unsafe_allow_javascript=False)
    else:
        st.markdown(content)


def _format_result_markdown(result: dict) -> str:
    return "**All done.** Use the buttons below to preview or download your tailored resume (TXT)."


def _preview_content(result: dict, primary_path: Path) -> str:
    markdown = result.get("tailored_markdown")
    if markdown:
        return markdown.strip()
    return primary_path.read_text(encoding="utf-8", errors="replace").strip()


@st.dialog("Tailored resume preview", width="large")
def _preview_dialog(content: str) -> None:
    st.markdown(content)


def _download_name(primary_path: Path) -> str:
    if st.session_state.get("resume_label"):
        stem = Path(st.session_state.resume_label).stem
        return f"{stem}_tailored.txt"
    return primary_path.with_suffix(".txt").name


def _render_output_actions(result: dict) -> None:
    primary = result.get("output_file_path")
    if not primary or not Path(primary).exists():
        return

    primary_path = Path(primary)
    download_name = _download_name(primary_path)
    file_data = primary_path.read_bytes()

    st.divider()
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.download_button(
            "Download tailored resume (TXT)",
            data=file_data,
            file_name=download_name,
            mime="text/plain",
            use_container_width=True,
            type="primary",
        )
        if st.button("Preview tailored resume", use_container_width=True):
            _preview_dialog(_preview_content(result, primary_path))


def _append_step(content: str) -> None:
    """Append an assistant message and render it as its own chat bubble."""
    _append("assistant", content)
    with st.chat_message("assistant"):
        _render_message(content)


def _replace_last_assistant(content: str) -> None:
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.messages[-1]["content"] = content
    else:
        _append("assistant", content)


def _append_progress_step(label: str) -> None:
    _append_step(_progress_html(label))


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
        out_dir = UPLOAD_DIR / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.agent_runner = StepwiseAgent(
            resume_path=st.session_state.resume_path,
            job_path_or_text=job_text.strip(),
            output_path=str(out_dir),
        )
        st.rerun()

    runner = st.session_state.get("agent_runner")
    if phase != "running" or runner is None:
        return

    try:
        if st.session_state.get("agent_executing_step"):
            message, done = runner.run_next()
            if message:
                _replace_last_assistant(message)
            st.session_state.agent_executing_step = False
        else:
            progress_label = runner.current_progress_label()
            if progress_label:
                _append_progress_step(progress_label)
                st.session_state.agent_executing_step = True
                st.rerun()
                return

            message, done = runner.run_next()
            if message:
                _append_step(message)

        if done:
            st.session_state.pop("pending_job", None)
            st.session_state.job_run_phase = None
            st.session_state.agent_runner = None
            st.session_state.agent_executing_step = False
            st.session_state.last_result = runner.state
            _append_step(_format_result_markdown(runner.state))
            st.rerun()
            return

        st.rerun()
    except Exception as exc:
        st.session_state.pop("pending_job", None)
        st.session_state.job_run_phase = None
        st.session_state.agent_runner = None
        st.session_state.agent_executing_step = False
        st.session_state.last_result = None
        _append_step(
            f"**Error:** {exc}\n\nCheck that `GOOGLE_API_KEY` is set and valid."
        )
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
    if text.lower() == "sample":
        _submit_sample_resume(text)
        st.rerun()
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
        _submit_sample_job(text)
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
        return "Attach your resume..."
    return "Paste the job description or URL..."


def main() -> None:
    st.set_page_config(page_title="Resume Agent", page_icon="📄", layout="centered")
    _init_session()

    st.title("Resume Tailoring Agent")
    st.caption("LangGraph · Gemini · tool-calling · guardrails")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            _render_message(message["content"])

    _render_quick_actions()

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
        _render_output_actions(result)


if __name__ == "__main__":
    main()
