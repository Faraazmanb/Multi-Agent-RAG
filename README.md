# Enterprise AI Assistant

This is a hands-on portfolio project I built to show practical AI engineering skills across LLM workflows and traditional data/ML work.

It combines:
- Python + FastAPI service development
- RAG-style retrieval
- Multi-step orchestration with LangGraph
- Classical ML (classification and clustering)
- SQL/ETL pipeline design
- Optional Neo4j graph integration
- MCP-compatible tool exposure

## Tech stack

- **Backend**: FastAPI, Pydantic
- **Agent flow**: LangGraph (`retrieve -> analyst -> editor`)
- **Retrieval**: TF-IDF + cosine similarity (offline corpus)
- **LLM mode**:
  - Live mode: set `OPENAI_ACTIVE=true` and a valid `OPENAI_API_KEY`
  - Offline/demo mode: deterministic `FakeListChatModel` responses
- **ML**: scikit-learn (Iris classification + KMeans clustering)
- **ETL/SQL**: SQLite staging/dimension/fact example
- **Graph DB**: Neo4j (optional)
- **Quality**: pytest + ruff + GitHub Actions CI

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
python -m enterprise_ai
```

Open API docs at: `http://127.0.0.1:8000/docs`

## Available API endpoints

- `GET /health`
- `POST /rag/search`
- `POST /agents/run`
- `GET /ml/classification`
- `GET /ml/clustering`
- `POST /etl/run`
- `GET /graph/neo4j/ping`
- `POST /graph/neo4j/seed`

## Neo4j setup (optional)

```bash
docker compose up -d neo4j
copy .env.example .env
```

Then set:
- `NEO4J_URI=bolt://localhost:7687`
- `NEO4J_USER=neo4j`
- `NEO4J_PASSWORD=portfolio-demo`

## MCP server (optional)

```bash
python -m enterprise_ai.mcp_tools.server
```

This exposes two tools:
- `retrieve_context`
- `run_sqlite_etl_demo`

## Notes

- The retrieval layer is intentionally lightweight and offline-friendly.
- In a production setting, this can be replaced with embedding-based retrieval and a vector database.
- Neo4j and MCP pieces are optional add-ons to demonstrate interoperability patterns.
