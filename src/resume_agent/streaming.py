from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from resume_agent.nodes import (
    analyze_gaps,
    analyze_job,
    export_resume,
    review_quality,
    should_revise,
    tailor_resume,
)
from resume_agent.research_agent import stream_research_agent
from resume_agent.state import AgentState


def format_agent_event(event: dict[str, Any]) -> str | None:
    """Return a single plain status line for the UI, or None to skip."""
    if event.get("event") == "status":
        return event.get("message", "").strip() or None
    return None


def stream_agent(
    resume_path: str,
    job_path_or_text: str,
    output_path: str = "output",
) -> Iterator[dict[str, Any]]:
    """Yield simple status events, ending with event=complete and final state."""
    state: AgentState = {
        "resume_path": resume_path,
        "job_path_or_text": job_path_or_text,
        "output_path": output_path,
        "revision_count": 0,
        "messages": [],
        "tool_trace": [],
        "evidence_notes": "",
    }

    yield {
        "event": "status",
        "message": "Planning tool calls to load the resume and job posting.",
    }

    research_update = stream_research_agent(
        resume_path=resume_path,
        job_path_or_text=job_path_or_text,
    )
    state.update(research_update)

    yield {"event": "status", "message": "Analyzing the resume and job posting."}

    yield {"event": "status", "message": "Analyzing the job requirements."}
    state.update(analyze_job(state))

    yield {"event": "status", "message": "Checking how your experience matches."}
    state.update(analyze_gaps(state))

    while True:
        yield {"event": "status", "message": "Tailoring your resume."}
        state.update(tailor_resume(state))

        yield {"event": "status", "message": "Reviewing for accuracy."}
        state.update(review_quality(state))

        if should_revise(state) == "tailor_resume":
            yield {"event": "status", "message": "Making a few revisions."}
            continue
        break

    yield {"event": "status", "message": "Exporting your tailored resume."}
    state.update(export_resume(state))

    yield {"event": "complete", "state": state}
