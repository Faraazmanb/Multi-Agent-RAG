"""MCP server exposing retrieval + ETL helpers (stdio transport)."""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from enterprise_ai.etl.pipeline import run_sample_etl
from enterprise_ai.rag.retriever import TfidfRetriever

mcp = FastMCP("Enterprise AI Assistant")

_retriever = TfidfRetriever()


@mcp.tool()
def retrieve_context(query: str, top_k: int = 3) -> str:
    """Return top TF-IDF chunks for a query (offline RAG context)."""
    hits = _retriever.retrieve(query, top_k=top_k)
    lines = [f"[{score:.3f}] {doc}" for doc, score in hits]
    return "\n".join(lines) if lines else "(no hits)"


@mcp.tool()
def run_sqlite_etl_demo(database_filename: str = "mcp_etl.sqlite3") -> str:
    """Build a small SQLite warehouse under ./data for SQL/ETL demonstration."""
    root = Path.cwd() / "data"
    summary = run_sample_etl(root / database_filename)
    return f"{summary.database_path} | rows: {summary.row_counts}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
