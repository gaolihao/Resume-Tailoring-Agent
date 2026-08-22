from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JobAnalysis(BaseModel):
    """Structured read of a job posting for ATS/screening alignment."""

    title: str = Field(description="Job title from the posting")
    company: str | None = Field(default=None, description="Company name if present")
    must_have_skills: list[str] = Field(
        default_factory=list,
        description="Hard requirements and must-have skills",
    )
    nice_to_have_skills: list[str] = Field(
        default_factory=list,
        description="Preferred or nice-to-have skills",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="ATS-relevant keywords and phrases from the posting",
    )
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Core responsibilities",
    )
    seniority: str | None = Field(
        default=None,
        description="Seniority level if implied (e.g. junior, mid, senior)",
    )
    summary: str = Field(
        description="2-4 sentence summary of what the role needs"
    )


class GapItem(BaseModel):
    requirement: str
    status: Literal["strong_match", "partial_match", "missing", "transferable"]
    evidence: str = Field(
        description="Where this shows up in the resume, or why it is missing"
    )
    suggestion: str = Field(
        description="How to emphasize or reframe without inventing experience"
    )


class GapAnalysis(BaseModel):
    match_score: int = Field(
        ge=0,
        le=100,
        description="Estimated screening match score before tailoring",
    )
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)
    positioning_advice: str = Field(
        description="Overall strategy for how to position this candidate"
    )


class ExperienceEntry(BaseModel):
    """One role block, preserving the original resume layout."""

    role_lines: list[str] = Field(
        description="1-2 header lines for the role (title/company and dates)"
    )
    bullets: list[str] = Field(
        description="Achievement bullets only — no role prefix, no leading dash"
    )


class TailoredResume(BaseModel):
    """Truthful rewrite of the resume for a specific posting."""

    professional_summary: str
    skills_section: list[str]
    experience_entries: list[ExperienceEntry] = Field(
        description="Experience section mirroring the original role + bullet structure"
    )
    education: list[str] = Field(default_factory=list)
    additional_sections: list[str] = Field(
        default_factory=list,
        description="Certifications, projects, awards, etc.",
    )
    changes_made: list[str] = Field(
        description="Bullet list of what changed and why"
    )
    keywords_injected: list[str] = Field(
        default_factory=list,
        description="Job keywords naturally incorporated from real experience",
    )

    @property
    def experience_bullets(self) -> list[str]:
        """Flattened bullets for summaries and legacy callers."""
        lines: list[str] = []
        for entry in self.experience_entries:
            prefix = " — ".join(entry.role_lines) if entry.role_lines else "Experience"
            for bullet in entry.bullets:
                lines.append(f"{prefix} — {bullet}")
        return lines


class QualityReview(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    revision_notes: str = Field(
        default="",
        description="Concrete fix instructions if not approved",
    )