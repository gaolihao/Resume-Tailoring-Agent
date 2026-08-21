# Resume Tailoring Agent (LangGraph)

LangGraph agent that rewrites your resume against a job posting so it screens better — without inventing experience.

## What it does

```
load resume (PDF / DOCX / TXT)
        ↓
analyze job posting (skills, keywords, responsibilities)
        ↓
gap analysis (matches, misses, positioning)
        ↓
tailor resume  ⇄  quality review (max 2 revision loops)
        ↓
export Markdown + DOCX
```

**Truth rules baked in:** the agent may rephrase, reorder, and emphasize what’s already on your resume. It will not fabricate jobs, metrics, tools, or credentials.

## Setup

```bash
cd JobApplicationProject
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -e .
copy .env.example .env   # then set OPENAI_API_KEY
```

## Usage

```bash
# Resume + job description file
resume-agent tailor path\to\resume.pdf --job path\to\job.txt

# Or paste the posting inline
resume-agent tailor path\to\resume.docx --job-text "We are hiring a Senior Backend Engineer..."

# Custom output folder
resume-agent tailor examples\sample_resume.txt --job examples\sample_job.txt -o output
```

Also:

```bash
python -m resume_agent tailor examples\sample_resume.txt --job examples\sample_job.txt
```

Outputs land in `output/`:

- `resume_<role>.md` — tailored content + change log
- `resume_<role>.docx` — Word version for applications

## Supported resume formats

| Format | Support |
|--------|---------|
| `.pdf` | Yes (`pypdf`) |
| `.docx` | Yes (`python-docx`) |
| `.txt` / `.md` | Yes |
| `.doc` (legacy) | Convert to `.docx` or `.pdf` first |

## Configuration

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required |
| `OPENAI_MODEL` | Default `gpt-4o-mini` |
| `OPENAI_BASE_URL` | Optional (Azure, OpenRouter, local gateway) |

## Project layout

```
src/resume_agent/
  graph.py      # LangGraph StateGraph wiring
  nodes.py      # load → analyze → gap → tailor → review → export
  models.py     # structured LLM outputs
  parsers/      # PDF / DOCX / text extraction
  writers/      # Markdown + DOCX export
  cli.py        # Typer CLI
```

## Notes

- Best results when the source resume already has real overlap with the role.
- Always review the tailored file before submitting — especially numbers and titles.
- Keyword stuffing is intentionally discouraged by the quality-review node.
# Resume-Tailoring-Agent
