from enterprise_ai.rag.retriever import TfidfRetriever


def test_retriever_returns_scored_chunks():
    r = TfidfRetriever()
    hits = r.retrieve("LangGraph orchestration", top_k=2)
    assert len(hits) >= 1
    assert all(isinstance(t[0], str) and t[1] > 0 for t in hits)
