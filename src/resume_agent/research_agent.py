from __future__ import annotations

import json
from typing import Annotated, TypedDict
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from resume_agent.llm import get_llm
from resume_agent.parsers import extract_resume_text, load_job_text
from resume_agent.tools import RESEARCH_TOOLS

RESEARCH_SYSTEM = """
You are a research agent that gathers source material before resume tailoring.

You MUST use tools — do not invent resume or job content.

Required workflow:
1. Call parse_resume with the provided resume path.
2. Resolve the job source:
   - If it looks like an http(s) URL, call fetch_job_from_url.
   - If it looks like a local file path, call load_job_file.
   - If it is raw job-description text, use that text directly (no tool needed).
3. From the job text, pick 3-6 concrete must-have skills/requirements and call
   evidence_check once per claim against the resume text from parse_resume.
4. After tools finish, write a short plain-text summary of what you gathered and
   which claims were supported / partial / unsupported. Do not call more tools
   after that summary.
""".strip()


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]


def build_research_agent():
    """ReAct-style tool-calling subgraph: model ⇄ tools until done."""
    model = get_llm().bind_tools(RESEARCH_TOOLS)

    def call_model(state: ResearchState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ResearchState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(RESEARCH_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_research_agent(resume_path: str, job_path_or_text: str) -> dict:
    """
    Run the tool-calling research agent and normalize outputs for the outer graph.

    Returns resume_text, job_text, evidence_notes, and tool_trace.
    Falls back to direct parsers if the model skipped required tools.
    """
    agent = build_research_agent()
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(
                    content=(
                        f"Resume path: {resume_path}\n"
                        f"Job source (path, URL, or raw text):\n{job_path_or_text}"
                    )
                ),
            ]
        }
    )
    messages = result["messages"]
    tool_outputs = _collect_tool_outputs(messages)
    tool_trace = [
        f"{name}: {preview}"
        for name, preview in (
            (n, (o[:160] + "…") if len(o) > 160 else o) for n, o in tool_outputs
        )
    ]

    resume_text = _first_tool_output(tool_outputs, "parse_resume")
    job_text = _resolve_job_text(tool_outputs, job_path_or_text)

    if not resume_text:
        resume_text = extract_resume_text(resume_path)
        tool_trace.append("fallback: parse_resume via direct parser")

    if not job_text:
        job_text = load_job_text(job_path_or_text)
        tool_trace.append("fallback: job text via direct loader")

    evidence_notes = _evidence_notes(tool_outputs)
    summary = _last_ai_text(messages)
    if summary:
        evidence_notes = (evidence_notes + "\n\n" + summary).strip()

    return {
        "resume_text": resume_text,
        "job_text": job_text,
        "evidence_notes": evidence_notes,
        "tool_trace": tool_trace,
        "messages": messages,
        "revision_count": 0,
    }


def _collect_tool_outputs(messages: list) -> list[tuple[str, str]]:
    """Map ToolMessages back to tool names using prior AIMessage tool_calls."""
    call_names: dict[str, str] = {}
    outputs: list[tuple[str, str]] = []

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                call_names[call["id"]] = call["name"]
        elif isinstance(message, ToolMessage):
            name = call_names.get(message.tool_call_id, message.name or "unknown_tool")
            content = message.content if isinstance(message.content, str) else str(message.content)
            outputs.append((name, content))
    return outputs


def _first_tool_output(outputs: list[tuple[str, str]], tool_name: str) -> str | None:
    for name, content in outputs:
        if name == tool_name and content and not _is_error_payload(content):
            return content
    return None


def _resolve_job_text(outputs: list[tuple[str, str]], original: str) -> str | None:
    for name in ("fetch_job_from_url", "load_job_file"):
        text = _first_tool_output(outputs, name)
        if text:
            return text

    if not _looks_like_path_or_url(original):
        return original.strip() or None
    return None


def _evidence_notes(outputs: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for name, content in outputs:
        if name != "evidence_check":
            continue
        try:
            payload = json.loads(content)
            claim = payload.get("claim", "?")
            status = payload.get("status", "?")
            matched = ", ".join(payload.get("matched_terms", [])[:8]) or "—"
            lines.append(f"- [{status}] {claim} (matched: {matched})")
        except json.JSONDecodeError:
            lines.append(f"- {content[:240]}")
    return "Evidence checks:\n" + "\n".join(lines) if lines else ""


def _last_ai_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()
    return ""


def _looks_like_path_or_url(value: str) -> bool:
    stripped = value.strip()
    if "\n" in stripped:
        return False
    parsed = urlparse(stripped)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return True
    return len(stripped) < 260 and (
        "/" in stripped or "\\" in stripped or stripped.lower().endswith((".txt", ".md"))
    )


def _is_error_payload(content: str) -> bool:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and "error" in payload
