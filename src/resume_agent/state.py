from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from resume_agent.models import GapAnalysis, JobAnalysis, QualityReview, TailoredResume


class AgentState(TypedDict, total=False):
    """Shared state flowing through the resume-tailoring graph."""

    resume_path: str
    job_path_or_text: str
    resume_text: str
    job_text: str
    evidence_notes: str
    tool_trace: list[str]
    job_analysis: JobAnalysis
    gap_analysis: GapAnalysis
    tailored_resume: TailoredResume
    quality_review: QualityReview
    tailored_markdown: str
    output_path: str
    output_file_path: str
    output_report_path: str
    revision_count: int
    messages: Annotated[list, add_messages]