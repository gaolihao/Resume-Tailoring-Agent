# Resume Tailoring Agent (LangGraph)

LangGraph agent that rewrites your resume against a job posting so it screens better — without inventing experience.

Includes a **real tool-calling research agent** (model ⇄ tools loop) before the tailor/review pipeline.

## What it does

```
research_inputs  (tool-calling ReAct agent)
  tools: parse_resume | load_job_file | fetch_job_from_url | evidence_check
        ↓
analyze job posting (skills, keywords, responsibilities)
        ↓
gap analysis (matches, misses, positioning)
        ↓
tailor resume  ⇄  quality review (max 2 revision loops)
        ↓
export Markdown + DOCX
```

**Truth rules baked in:** the agent may rephrase, reorder, and emphasize what’s already on your resume. It will not fabricate jobs, metrics, tools, or credentials. `evidence_check` is a deterministic overlap tool (not another LLM).

## Setup

```bash
cd JobApplicationProject
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -e .
copy .env.example .env   # then set GOOGLE_API_KEY
```

Get a Gemini key from [Google AI Studio](https://aistudio.google.com/apikey).

## Usage

```bash
# Activate the venv (Windows)
.\.venv\Scripts\activate

# Resume + job description file
resume-agent examples\sample_resume.txt --job examples\sample_job.txt

# Resume + job posting URL (research agent calls fetch_job_from_url)
resume-agent examples\sample_resume.txt --job-url "https://example.com/jobs/123"

# Or paste the posting inline
resume-agent path\to\resume.docx --job-text "We are hiring a Senior Backend Engineer..."

# Custom output folder
resume-agent examples\sample_resume.txt --job examples\sample_job.txt -o output
```

Also:

```bash
python -m resume_agent examples\sample_resume.txt --job examples\sample_job.txt
```

Outputs land in `output/`:

- `resume_<role>.md` — tailored content + change log
- `resume_<role>.docx` — Word version for applications

## Tool-calling research agent

The first graph node is a ReAct subgraph:

1. LLM decides which tools to call (`bind_tools`)
2. `ToolNode` executes them
3. Results return as `ToolMessage`s
4. Loop until the model stops requesting tools

| Tool | Purpose |
|------|---------|
| `parse_resume` | Extract text from PDF/DOCX/TXT |
| `load_job_file` | Read a local JD file |
| `fetch_job_from_url` | Download + clean a job posting URL |
| `evidence_check` | Heuristic claim-vs-resume support check |

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
| `GOOGLE_API_KEY` | Required (Gemini / Google AI Studio key) |
| `GEMINI_API_KEY` | Optional alias for the same key |
| `GEMINI_MODEL` | Default `gemini-3.5-flash-lite` |
| `GEMINI_THINKING_LEVEL` | `minimal` (default), `low`, `medium`, or `high` |

## Project layout

```
src/resume_agent/
  graph.py           # Outer StateGraph wiring
  research_agent.py  # Tool-calling ReAct subgraph
  tools.py           # parse_resume, fetch_job_from_url, evidence_check, ...
  nodes.py           # research → analyze → gap → tailor → review → export
  models.py          # structured LLM outputs
  parsers/           # PDF / DOCX / text extraction
  writers/           # Markdown + DOCX export
  cli.py             # Typer CLI
```

## Notes

- Best results when the source resume already has real overlap with the role.
- Always review the tailored file before submitting — especially numbers and titles.
- Keyword stuffing is intentionally discouraged by the quality-review node.
