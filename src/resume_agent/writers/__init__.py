from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Pt

from resume_agent.models import JobAnalysis, TailoredResume


def tailored_to_markdown(
    tailored: TailoredResume,
    job: JobAnalysis | None = None,
) -> str:
    lines: list[str] = []
    if job:
        target = job.title
        if job.company:
            target = f"{job.title} @ {job.company}"
        lines.append(f"# Tailored Resume — {target}")
        lines.append("")

    lines.extend(
        [
            "## Professional Summary",
            tailored.professional_summary,
            "",
            "## Skills",
            ", ".join(tailored.skills_section),
            "",
            "## Experience",
        ]
    )
    for bullet in tailored.experience_bullets:
        lines.append(f"- {bullet}")

    if tailored.education:
        lines.extend(["", "## Education"])
        for item in tailored.education:
            lines.append(f"- {item}")

    if tailored.additional_sections:
        lines.extend(["", "## Additional"])
        for item in tailored.additional_sections:
            lines.append(f"- {item}")

    lines.extend(["", "## Changes Made"])
    for change in tailored.changes_made:
        lines.append(f"- {change}")

    if tailored.keywords_injected:
        lines.extend(
            [
                "",
                "## Keywords Incorporated",
                ", ".join(tailored.keywords_injected),
            ]
        )

    return "\n".join(lines).strip() + "\n"


def write_markdown(content: str, path: Path) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def write_docx(
    tailored: TailoredResume,
    path: Path,
    job: JobAnalysis | None = None,
) -> Path:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    if job:
        heading = doc.add_heading(job.title, level=1)
        heading.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
        if job.company:
            doc.add_paragraph(f"Target: {job.company}")

    doc.add_heading("Professional Summary", level=2)
    doc.add_paragraph(tailored.professional_summary)

    doc.add_heading("Skills", level=2)
    doc.add_paragraph(", ".join(tailored.skills_section))

    doc.add_heading("Experience", level=2)
    for bullet in tailored.experience_bullets:
        doc.add_paragraph(bullet, style="List Bullet")

    if tailored.education:
        doc.add_heading("Education", level=2)
        for item in tailored.education:
            doc.add_paragraph(item, style="List Bullet")

    if tailored.additional_sections:
        doc.add_heading("Additional", level=2)
        for item in tailored.additional_sections:
            doc.add_paragraph(item, style="List Bullet")

    doc.save(str(path))
    return path