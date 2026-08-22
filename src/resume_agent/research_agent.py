from __future__ import annotations

import json
from typing import Annotated, Any, Callable, TypedDict
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


EventCallback = Callable[[dict[str, Any]], None]


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


def _emit(callback: EventCallback | None, event: dict[str, Any]) -> None:
    if callback:
        callback(event)


def _preview_tool_args(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "parse_resume":
        return f"path=`{args.get('path', '')}`"
    if tool_name == "load_job_file":
        return f"path=`{args.get('path', '')}`"
    if tool_name == "fetch_job_from_url":
        return f"url=`{args.get('url', '')}`"
    if tool_name == "evidence_check":
        claim = str(args.get("claim", ""))[:80]
        return f"claim=\"{claim}\""
    return json.dumps(args, ensure_ascii=False)[:120]


def _preview_tool_result(tool_name: str, content: str) -> str:
    if tool_name == "evidence_check":
        try:
            payload = json.loads(content)
            return (
                f"status={payload.get('status')} "
                f"coverage={payload.get('coverage')} "
                f"claim=\"{str(payload.get('claim', ''))[:60]}\""
            )
        except json.JSONDecodeError:
            pass
    if _is_error_payload(content):
        try:
            return f"error: {json.loads(content).get('error', content)[:120]}"
        except json.JSONDecodeError:
            return content[:120]
    one_line = " ".join(content.split())
    return (one_line[:180] + "…") if len(one_line) > 180 else one_line


def _finalize_research_result(
    messages: list,
    resume_path: str,
    job_path_or_text: str,
    on_event: EventCallback | None = None,
) -> dict:
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
        _emit(
            on_event,
            {
                "event": "fallback",
                "node": "research_inputs",
                "detail": "Used direct resume parser fallback.",
            },
        )

    if not job_text:
        job_text = load_job_text(job_path_or_text)
        tool_trace.append("fallback: job text via direct loader")
        _emit(
            on_event,
            {
                "event": "fallback",
                "node": "research_inputs",
                "detail": "Used direct job text loader fallback.",
            },
        )

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


def stream_research_agent(
    resume_path: str,
    job_path_or_text: str,
    on_event: EventCallback | None = None,
) -> dict:
    """Run the research ReAct agent, emitting tool/thought events step by step."""
    agent = build_research_agent()
    input_state: ResearchState = {
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

    _emit(
        on_event,
        {
            "event": "node_start",
            "node": "research_inputs",
            "title": "Research agent",
            "detail": "Planning tool calls to load the resume and job posting.",
        },
    )

    call_names: dict[str, str] = {}
    messages: list = list(input_state["messages"])

    for chunk in agent.stream(input_state, stream_mode="updates"):
        for update in chunk.values():
            for message in update.get("messages", []):
                messages.append(message)
                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        for call in message.tool_calls:
                            call_names[call["id"]] = call["name"]
                            _emit(
                                on_event,
                                {
                                    "event": "tool_call",
                                    "node": "research_inputs",
                                    "tool": call["name"],
                                    "args_preview": _preview_tool_args(
                                        call["name"], call.get("args", {})
                                    ),
                                    "detail": f"Calling `{call['name']}`",
                                },
                            )
                    elif message.content:
                        text = (
                            message.content
                            if isinstance(message.content, str)
                            else str(message.content)
                        ).strip()
                        if text:
                            _emit(
                                on_event,
                                {
                                    "event": "thought",
                                    "node": "research_inputs",
                                    "content": text[:600],
                                    "detail": "Research agent summary",
                                },
                            )
                elif isinstance(message, ToolMessage):
                    tool_name = call_names.get(
                        message.tool_call_id, message.name or "tool"
                    )
                    content = (
                        message.content
                        if isinstance(message.content, str)
                        else str(message.content)
                    )
                    _emit(
                        on_event,
                        {
                            "event": "tool_result",
                            "node": "research_inputs",
                            "tool": tool_name,
                            "preview": _preview_tool_result(tool_name, content),
                            "detail": f"`{tool_name}` finished",
                        },
                    )

    result = _finalize_research_result(
        messages, resume_path, job_path_or_text, on_event=on_event
    )
    _emit(
        on_event,
        {
            "event": "node_done",
            "node": "research_inputs",
            "detail": f"Research complete ({len(result['tool_trace'])} tool steps).",
        },
    )
    return result


def run_research_agent(resume_path: str, job_path_or_text: str) -> dict:
    """Run research agent without streaming (CLI / tests)."""
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
    return _finalize_research_result(
        result["messages"], resume_path, job_path_or_text
    )


def _collect_tool_outputs(messages: list) -> list[tuple[str, str]]:
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
