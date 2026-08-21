from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from resume_agent.llm import get_llm
from resume_agent.models import GapAnalysis, JobAnalysis, QualityReview, TailoredResume
from resume_agent.parsers import extract_resume_text, load_job_text
from resume_agent.state import AgentState
from resume_agent.writers import tailored_to_markdown, write_docx, write_markdown

MAX_REVISIONS = 2

TRUTH_RULES = """
CRITICAL TRUTH RULES:
- Never invent jobs, employers, dates, degrees, certifications, metrics, or tools.
- Only rephrase, reorder, and emphasize experience that already appears in the resume.
- You may map synonymous wording (e.g. "built APIs" → "designed REST APIs") only when
  the underlying work is clearly present.
- If a requirement is missing, do not claim it; note the gap instead.
- Prefer quantified bullets only when numbers already exist in the source resume.
""".strip()


def load_inputs(state: AgentState) -> dict:
    resume_text = extract_resume_text(state["resume_path"])
    job_text = load_job_text(state["job_path_or_text"])
    return {
        "resume_text": resume_text,
        "job_text": job_text,
        "revision_count": state.get("revision_count", 0),
    }


def analyze_job(state: AgentState) -> dict:
    llm = get_llm().with_structured_output(JobAnalysis)
    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You extract screening-relevant requirements from job postings. "
                    "Be specific and prefer exact phrases employers/ATS systems look for."
                )
            ),
            HumanMessage(
                content=f"Analyze this job posting:\n\n{state['job_text']}"
            ),
        ]
    )
    return {"job_analysis": result}


def analyze_gaps(state: AgentState) -> dict:
    llm = get_llm().with_structured_output(GapAnalysis)
    job = state["job_analysis"]
    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Compare a candidate resume to a job analysis for ATS/human screening. "
                    "Be honest about gaps. Suggest reframes only when evidence exists.\n\n"
                    f"{TRUTH_RULES}"
                )
            ),
            HumanMessage(
                content=(
                    f"## Job analysis (JSON)\n{job.model_dump_json(indent=2)}\n\n"
                    f"## Resume\n{state['resume_text']}"
                )
            ),
        ]
    )
    return {"gap_analysis": result}


def tailor_resume(state: AgentState) -> dict:
    llm = get_llm().with_structured_output(TailoredResume)
    revision_notes = ""
    review = state.get("quality_review")
    if review and not review.approved:
        revision_notes = (
            f"\n\n## Revision notes from quality review\n{review.revision_notes}\n"
            f"Issues: {', '.join(review.issues)}"
        )

    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You rewrite resumes to improve screening match for a specific job. "
                    "Mirror the posting's language where truthful. Keep a clean professional tone.\n\n"
                    f"{TRUTH_RULES}"
                )
            ),
            HumanMessage(
                content=(
                    f"## Job analysis\n{state['job_analysis'].model_dump_json(indent=2)}\n\n"
                    f"## Gap analysis\n{state['gap_analysis'].model_dump_json(indent=2)}\n\n"
                    f"## Original resume\n{state['resume_text']}"
                    f"{revision_notes}\n\n"
                    "Produce a tailored resume that maximizes legitimate keyword and "
                    "responsibility alignment."
                )
            ),
        ]
    )
    return {
        "tailored_resume": result,
        "revision_count": state.get("revision_count", 0) + 1,
    }


def review_quality(state: AgentState) -> dict:
    llm = get_llm().with_structured_output(QualityReview)
    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict resume QA reviewer. Approve only if the tailored "
                    "resume is truthful, screening-optimized, and free of invented claims.\n\n"
                    f"{TRUTH_RULES}\n\n"
                    "Reject if: fabricated experience, ignored must-have matches that "
                    "exist in the source, or awkward keyword stuffing."
                )
            ),
            HumanMessage(
                content=(
                    f"## Original resume\n{state['resume_text']}\n\n"
                    f"## Job analysis\n{state['job_analysis'].model_dump_json(indent=2)}\n\n"
                    f"## Tailored resume\n{state['tailored_resume'].model_dump_json(indent=2)}"
                )
            ),
        ]
    )
    return {"quality_review": result}


def should_revise(state: AgentState) -> str:
    review = state.get("quality_review")
    revisions = state.get("revision_count", 0)
    if review and not review.approved and revisions < MAX_REVISIONS:
        return "tailor_resume"
    return "export_resume"


def export_resume(state: AgentState) -> dict:
    tailored = state["tailored_resume"]
    markdown = tailored_to_markdown(tailored, state.get("job_analysis"))

    out_dir = Path(state.get("output_path") or "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    job = state.get("job_analysis")
    slug = _slug(job.title if job else "role")
    md_path = out_dir / f"resume_{slug}.md"
    docx_path = out_dir / f"resume_{slug}.docx"

    write_markdown(markdown, md_path)
    write_docx(tailored, docx_path, job)

    return {"tailored_markdown": markdown, "output_path": str(out_dir)}


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:60] or "tailored"