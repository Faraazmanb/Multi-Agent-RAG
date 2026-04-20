"""REST API: RAG, LangGraph agents, classical ML, ETL, Neo4j health."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from enterprise_ai.agents.graph import run_agent_workflow
from enterprise_ai.etl.pipeline import run_sample_etl
from enterprise_ai.graph_db.neo4j_client import ping_neo4j, seed_demo_graph
from enterprise_ai.ml.traditional import iris_classification_demo, iris_clustering_demo
from enterprise_ai.rag.retriever import TfidfRetriever

app = FastAPI(
    title="Enterprise AI Assistant",
    version="0.1.0",
    description="Portfolio API: LangGraph multi-step agents, TF-IDF RAG, ML, SQL/ETL, Neo4j.",
)
_retriever = TfidfRetriever()


class QueryBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class RagResponse(BaseModel):
    chunks: list[dict[str, str | float]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rag/search", response_model=RagResponse)
def rag_search(body: QueryBody):
    hits = _retriever.retrieve(body.query, top_k=5)
    return {
        "chunks": [{"text": doc, "score": score} for doc, score in hits],
    }


@app.post("/agents/run")
def agents_run(body: QueryBody):
    state = run_agent_workflow(body.query)
    msgs = state.get("messages", [])
    last = msgs[-1].content if msgs else ""
    return {
        "context": state.get("context", ""),
        "final_answer": last,
        "steps": len(msgs),
    }


@app.get("/ml/classification")
def ml_classification():
    report = iris_classification_demo()
    return {
        "accuracy": report.accuracy,
        "feature_names": report.feature_names,
        "coefficients_class0": report.coefficients,
    }


@app.get("/ml/clustering")
def ml_clustering():
    report = iris_clustering_demo()
    return {"silhouette": report.silhouette, "cluster_sizes": report.cluster_sizes}


@app.post("/etl/run")
def etl_run():
    data_dir = Path.cwd() / "data"
    summary = run_sample_etl(data_dir / "portfolio_etl.sqlite3")
    return {"database_path": summary.database_path, "row_counts": summary.row_counts}


@app.get("/graph/neo4j/ping")
def neo4j_ping():
    return ping_neo4j().__dict__


@app.post("/graph/neo4j/seed")
def neo4j_seed():
    return seed_demo_graph().__dict__


def create_app() -> FastAPI:
    return app
