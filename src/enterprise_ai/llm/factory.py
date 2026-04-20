from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI

from enterprise_ai.config import get_settings


def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    if settings.openai_active and settings.openai_api_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return FakeListChatModel(
        responses=[
            (
                "[Offline demo LLM] Analyst: Retrieved chunks support themes of RAG, "
                "LangGraph orchestration, and MCP-style tool exposure. "
                "Recommend citing sources and logging agent steps."
            ),
            (
                "[Offline demo LLM] Editor: Consolidated answer — use TF-IDF or dense "
                "retrieval for grounding, route via a supervisor graph, and validate "
                "with Neo4j or SQL checks where applicable."
            ),
        ]
    )
