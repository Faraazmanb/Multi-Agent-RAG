from fastapi.testclient import TestClient

from enterprise_ai.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_rag_search():
    r = client.post("/rag/search", json={"query": "RAG architecture"})
    assert r.status_code == 200
    data = r.json()
    assert "chunks" in data
    assert len(data["chunks"]) >= 1
