from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from resume_agent.graph import run_agent

app = typer.Typer(
    add_completion=False,
    help="Tailor a resume to a job posting with a LangGraph agent.",
)
console = Console()


@app.command()
def tailor(
    resume: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to resume (.pdf, .docx, .txt, .md)",
    ),
    job: Path = typer.Option(
        None,
        "--job",
        "-j",
        help="Path to a job description text file",
    ),
    job_url: Optional[str] = typer.Option(
        None,
        "--job-url",
        help="Job posting URL (research agent fetches via tool)",
    ),
    job_text: Optional[str] = typer.Option(
        None,
        "--job-text",
        help="Job description as a raw string",
    ),
    output: Path = typer.Option(
        Path("output"),
        "--output",
        "-o",
        help="Directory for tailored markdown/docx output",
    ),
    show: bool = typer.Option(
        True,
        "--show/--no-show",
        help="Print a summary of tools, gaps, and changes",
    ),
) -> None:
    """Run the resume-tailoring LangGraph agent (includes tool-calling research)."""
    provided = [x for x in (job, job_url, job_text) if x]
    if len(provided) != 1:
        raise typer.BadParameter(
            "Provide exactly one of --job PATH, --job-url URL, or --job-text '...'"
        )

    if job is not None:
        job_input = str(job)
    elif job_url:
        job_input = job_url
    else:
        job_input = job_text or ""

    with console.status("[bold]Running resume agent (research tools + tailor)..."):
        result = run_agent(
            resume_path=str(resume),
            job_path_or_text=job_input,
            output_path=str(output),
        )

    if show:
        _print_summary(result)

    console.print(
        Panel.fit(
            f"Wrote tailored resume files to [bold]{result.get('output_path', output)}[/bold]",
            title="Done",
            border_style="green",
        )
    )


def _print_summary(result: dict) -> None:
    trace = result.get("tool_trace") or []
    if trace:
        lines = "\n".join(f"- {t}" for t in trace[:12])
        console.print(Panel(lines, title="Tool-calling trace", border_style="yellow"))

    evidence = result.get("evidence_notes")
    if evidence:
        console.print(Panel(evidence[:1200], title="Evidence notes", border_style="blue"))

    gaps = result.get("gap_analysis")
    tailored = result.get("tailored_resume")
    review = result.get("quality_review")

    if gaps:
        table = Table(title="Screening Gap Analysis")
        table.add_column("Match score")
        table.add_column("Matched keywords")
        table.add_column("Missing keywords")
        table.add_row(
            f"{gaps.match_score}/100",
            ", ".join(gaps.matched_keywords[:12]) or "—",
            ", ".join(gaps.missing_keywords[:12]) or "—",
        )
        console.print(table)
        console.print(
            Panel(
                gaps.positioning_advice,
                title="Positioning advice",
                border_style="cyan",
            )
        )

    if tailored:
        changes = "\n".join(f"- {c}" for c in tailored.changes_made)
        console.print(Panel(Markdown(changes), title="Changes made", border_style="magenta"))

    if review:
        status = "Approved" if review.approved else "Needs revision notes applied"
        console.print(f"Quality review: [bold]{status}[/bold]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
