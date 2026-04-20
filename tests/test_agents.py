from enterprise_ai.agents.graph import run_agent_workflow


def test_agent_workflow_runs():
    out = run_agent_workflow("Explain MCP and RAG together.")
    assert "context" in out
    assert out["context"]
    assert out.get("messages")
