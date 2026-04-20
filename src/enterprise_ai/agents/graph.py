"""Multi-agent style workflow in LangGraph: retrieve → analyst → editor."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from enterprise_ai.llm.factory import get_chat_model
from enterprise_ai.rag.retriever import TfidfRetriever


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    query: str
    context: str


def build_multi_agent_graph(retriever: TfidfRetriever | None = None):
    retriever = retriever or TfidfRetriever()
    llm = get_chat_model()

    def retrieve_node(state: AgentState) -> AgentState:
        hits = retriever.retrieve(state["query"], top_k=3)
        ctx = "\n\n".join(f"- ({score:.3f}) {doc}" for doc, score in hits) or "(no hits)"
        return {"context": ctx}

    def analyst_node(state: AgentState) -> AgentState:
        msgs = [
            SystemMessage(
                content=(
                    "You are a research analyst. Use ONLY the CONTEXT to reason. "
                    "Produce bullet findings."
                )
            ),
            HumanMessage(
                content=f"QUESTION:\n{state['query']}\n\nCONTEXT:\n{state['context']}"
            ),
        ]
        out = llm.invoke(msgs)
        assert isinstance(out, AIMessage)
        return {"messages": [out]}

    def editor_node(state: AgentState) -> AgentState:
        prior = state["messages"][-1].content if state["messages"] else ""
        msgs = [
            SystemMessage(
                content=(
                    "You are an editor. Turn analyst notes into a concise final answer "
                    "for a technical stakeholder."
                )
            ),
            HumanMessage(
                content=(
                    f"QUESTION:\n{state['query']}\n\n"
                    f"ANALYST NOTES:\n{prior}\n\n"
                    f"CONTEXT (for fidelity):\n{state['context']}"
                )
            ),
        ]
        out = llm.invoke(msgs)
        assert isinstance(out, AIMessage)
        return {"messages": [out]}

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("analyst", analyst_node)
    g.add_node("editor", editor_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "analyst")
    g.add_edge("analyst", "editor")
    g.add_edge("editor", END)
    return g.compile()


def run_agent_workflow(user_query: str) -> dict:
    graph = build_multi_agent_graph()
    init: AgentState = {"messages": [], "query": user_query, "context": ""}
    return graph.invoke(init)
