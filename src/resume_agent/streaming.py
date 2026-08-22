from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from resume_agent.models import GapAnalysis, JobAnalysis, QualityReview, TailoredResume
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


def _step_message(title: str, feedback: str) -> str:
    feedback = feedback.strip()
    if feedback:
        return f"**{title}**\n\n{feedback}"
    return f"**{title}**"


def _first_resume_line(resume_text: str) -> str:
    for line in resume_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return "Resume loaded"


def _format_research_feedback(state: AgentState) -> str:
    resume_text = state.get("resume_text") or ""
    job_text = state.get("job_text") or ""
    lines = [
        f"- Parsed resume ({len(resume_text.splitlines())} lines) — "
        f"starting with *{_first_resume_line(resume_text)}*",
        f"- Loaded job description ({len(job_text.split())} words)",
    ]

    evidence = (state.get("evidence_notes") or "").strip()
    if evidence:
        evidence_lines = [
            ln.strip()
            for ln in evidence.splitlines()
            if ln.strip() and not ln.strip().lower().startswith("evidence checks:")
        ]
        if evidence_lines:
            lines.append("- Evidence checks:")
            for ln in evidence_lines[:5]:
                cleaned = ln.lstrip("- ").strip()
                lines.append(f"  - {cleaned}")

    trace = state.get("tool_trace") or []
    if trace:
        lines.append(f"- Ran {len(trace)} tool step(s)")

    return "\n".join(lines)


def _format_job_feedback(job: JobAnalysis) -> str:
    company = f" at **{job.company}**" if job.company else ""
    seniority = f" ({job.seniority})" if job.seniority else ""
    lines = [
        f"- Role: **{job.title}**{company}{seniority}",
        f"- Must-haves: {', '.join(job.must_have_skills[:6]) or '—'}",
        f"- Key keywords: {', '.join(job.keywords[:8]) or '—'}",
        f"- {job.summary}",
    ]
    return "\n".join(lines)


def _format_gaps_feedback(gaps: GapAnalysis) -> str:
    lines = [
        f"- Match score: **{gaps.match_score}/100**",
        f"- Strong matches: {', '.join(gaps.matched_keywords[:8]) or '—'}",
        f"- Gaps to address: {', '.join(gaps.missing_keywords[:8]) or '—'}",
    ]
    for gap in gaps.gaps[:3]:
        lines.append(f"- [{gap.status.replace('_', ' ')}] {gap.requirement} — {gap.suggestion}")
    lines.append(f"- Strategy: {gaps.positioning_advice}")
    return "\n".join(lines)


def _format_tailor_feedback(tailored: TailoredResume) -> str:
    lines = [
        f"- Updated summary and **{len(tailored.experience_entries)}** experience entries",
    ]
    if tailored.keywords_injected:
        lines.append(
            f"- Keywords woven in: {', '.join(tailored.keywords_injected[:8])}"
        )
    if tailored.changes_made:
        lines.append("- Main changes:")
        lines.extend(f"  - {change}" for change in tailored.changes_made[:4])
    return "\n".join(lines)


def _format_review_feedback(review: QualityReview, *, revised: bool = False) -> str:
    if review.approved:
        lines = ["- Draft looks accurate — no invented claims flagged."]
    else:
        lines = ["- Found issues to fix before export:"]
        lines.extend(f"  - {issue}" for issue in review.issues[:4])
        if review.revision_notes:
            lines.append(f"- Next pass: {review.revision_notes}")

    if revised:
        lines.insert(0, "- Sending the draft back for another tailoring pass.")
    return "\n".join(lines)


def _format_export_feedback(state: AgentState) -> str:
    path = state.get("output_file_path") or ""
    name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if path else "tailored resume"
    return (
        f"- Saved as **{name}** (.txt).\n"
        "- Preview or download your file using the buttons below."
    )


def _new_agent_state(
    resume_path: str,
    job_path_or_text: str,
    output_path: str,
) -> AgentState:
    return {
        "resume_path": resume_path,
        "job_path_or_text": job_path_or_text,
        "output_path": output_path,
        "revision_count": 0,
        "messages": [],
        "tool_trace": [],
        "evidence_notes": "",
    }


class StepwiseAgent:
    """Run the agent one step at a time for incremental chat UI updates."""

    _PROGRESS_LABELS = {
        1: "Analyzing the resume and job posting.",
        2: "Analyzing the job requirements.",
        3: "Checking how your experience matches.",
        4: "Tailoring your resume.",
        5: "Reviewing for accuracy.",
        6: "Exporting your tailored resume.",
    }

    def __init__(
        self,
        resume_path: str,
        job_path_or_text: str,
        output_path: str = "output",
    ) -> None:
        self.state = _new_agent_state(resume_path, job_path_or_text, output_path)
        self._phase = 0

    def current_progress_label(self) -> str | None:
        """Label to show with ... while the current step runs."""
        return self._PROGRESS_LABELS.get(self._phase)

    def run_next(self) -> tuple[str | None, bool]:
        """Run the next step. Returns (chat message with feedback, is_complete)."""
        if self._phase == 0:
            self._phase = 1
            return _step_message(
                "Planning tool calls to load the resume and job posting.",
                "I'll parse your resume, load the job description, and verify claims "
                "against your experience before rewriting anything.",
            ), False

        if self._phase == 1:
            self.state.update(
                stream_research_agent(
                    resume_path=self.state["resume_path"],
                    job_path_or_text=self.state["job_path_or_text"],
                )
            )
            self._phase = 2
            return _step_message(
                "Analyzing the resume and job posting.",
                _format_research_feedback(self.state),
            ), False

        if self._phase == 2:
            self.state.update(analyze_job(self.state))
            job = self.state["job_analysis"]
            self._phase = 3
            return _step_message(
                "Analyzing the job requirements.",
                _format_job_feedback(job),
            ), False

        if self._phase == 3:
            self.state.update(analyze_gaps(self.state))
            gaps = self.state["gap_analysis"]
            self._phase = 4
            return _step_message(
                "Checking how your experience matches.",
                _format_gaps_feedback(gaps),
            ), False

        if self._phase == 4:
            self.state.update(tailor_resume(self.state))
            tailored = self.state["tailored_resume"]
            self._phase = 5
            return _step_message(
                "Tailoring your resume.",
                _format_tailor_feedback(tailored),
            ), False

        if self._phase == 5:
            self.state.update(review_quality(self.state))
            review = self.state["quality_review"]
            if should_revise(self.state) == "tailor_resume":
                self._phase = 4
                return _step_message(
                    "Making a few revisions.",
                    _format_review_feedback(review, revised=True),
                ), False
            self._phase = 6
            return _step_message(
                "Reviewing for accuracy.",
                _format_review_feedback(review),
            ), False

        if self._phase == 6:
            self.state.update(export_resume(self.state))
            self._phase = 7
            return _step_message(
                "Exporting your tailored resume.",
                _format_export_feedback(self.state),
            ), False

        return None, True


def stream_agent(
    resume_path: str,
    job_path_or_text: str,
    output_path: str = "output",
) -> Iterator[dict[str, Any]]:
    """Yield simple status events, ending with event=complete and final state."""
    runner = StepwiseAgent(resume_path, job_path_or_text, output_path)
    while True:
        message, done = runner.run_next()
        if message:
            yield {"event": "status", "message": message}
        if done:
            break
    yield {"event": "complete", "state": runner.state}
