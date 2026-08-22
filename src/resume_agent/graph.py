from langgraph.graph import END, START, StateGraph

from resume_agent.nodes import (
    analyze_gaps,
    analyze_job,
    export_resume,
    research_inputs,
    review_quality,
    should_revise,
    tailor_resume,
)
from resume_agent.state import AgentState


def build_graph():
    """
    Document adaptation graph with a real tool-calling research agent:

    research_inputs (model ⇄ tools)
      → analyze_job → analyze_gaps → tailor_resume
      → review_quality ⇄ tailor_resume (max 2 revisions)
      → export_resume
    """
    graph = StateGraph(AgentState)

    graph.add_node("research_inputs", research_inputs)
    graph.add_node("analyze_job", analyze_job)
    graph.add_node("analyze_gaps", analyze_gaps)
    graph.add_node("tailor_resume", tailor_resume)
    graph.add_node("review_quality", review_quality)
    graph.add_node("export_resume", export_resume)

    graph.add_edge(START, "research_inputs")
    graph.add_edge("research_inputs", "analyze_job")
    graph.add_edge("analyze_job", "analyze_gaps")
    graph.add_edge("analyze_gaps", "tailor_resume")
    graph.add_edge("tailor_resume", "review_quality")
    graph.add_conditional_edges(
        "review_quality",
        should_revise,
        {
            "tailor_resume": "tailor_resume",
            "export_resume": "export_resume",
        },
    )
    graph.add_edge("export_resume", END)

    return graph.compile()


def run_agent(
    resume_path: str,
    job_path_or_text: str,
    output_path: str = "output",
) -> AgentState:
    app = build_graph()
    return app.invoke(
        {
            "resume_path": resume_path,
            "job_path_or_text": job_path_or_text,
            "output_path": output_path,
            "revision_count": 0,
            "messages": [],
            "tool_trace": [],
            "evidence_notes": "",
        }
    )
