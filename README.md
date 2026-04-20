# Enterprise AI Assistant (portfolio)

End-to-end demo aimed at **AI Developer** roles that expect **Python**, **LLM tooling**, **multi-agent orchestration**, **RAG**, **classical ML**, **SQL/ETL**, **Neo4j**, and **MCP**.

## What’s inside

| Area | Implementation |
|------|----------------|
| RAG | TF–IDF + cosine similarity over an in-repo corpus (offline; easy to swap for Chroma / Azure AI Search) |
| Agents | **LangGraph** graph: retrieve → analyst → editor |
| LLM | `OPENAI_ACTIVE=true` and `OPENAI_API_KEY` → `gpt-4o-mini`; otherwise **LangChain `FakeListChatModel`** (avoids failed calls when a stale key is in the environment) |
| ML | scikit-learn: **Iris** classification + **KMeans** clustering + metrics |
| ETL / SQL | **SQLite** pipeline: staging → dimensions → facts |
| Graph | **Neo4j** optional (`docker-compose`); seed + ping endpoints |
| MCP | `mcp.server.fastmcp` stdio server with `retrieve_context` + `run_sqlite_etl_demo` tools |
| CI | GitHub Actions: **ruff** + **pytest** |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
python -m enterprise_ai
```

API: `http://127.0.0.1:8000/docs`

### Neo4j (optional)

```bash
docker compose up -d neo4j
copy .env.example .env
```

Set `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`, `NEO4J_PASSWORD=portfolio-demo`.  
`GET /graph/neo4j/ping` · `POST /graph/neo4j/seed`

### MCP (stdio)

```bash
python -m enterprise_ai.mcp_tools.server
```

Wire this server in Cursor / Claude Desktop / any MCP host and call `retrieve_context` or `run_sqlite_etl_demo`.

## Interview narrative

1. **Grounding**: TF–IDF RAG proves retrieval design; mention upgrading to embeddings + vector DB for production.  
2. **Orchestration**: LangGraph shows explicit steps and state — extend with conditional routing (supervisor) or tool-calling agents.  
3. **Interop**: MCP tools mirror how enterprise stacks expose **governed** capabilities to models (**A2A** / agent handoffs pair naturally with the same boundaries).  
4. **Data**: SQLite ETL + optional Neo4j show **SQL modeling** and **graph** reasoning side-by-side with GenAI.  
5. **ML**: Classical models complement LLMs for structured labels and segmentation.

## API sketch

- `POST /rag/search` — lexical retrieval  
- `POST /agents/run` — full LangGraph run  
- `GET /ml/classification`, `GET /ml/clustering`  
- `POST /etl/run` — builds `./data/portfolio_etl.sqlite3`

---

*This repository is an independent portfolio sample and is not affiliated with or endorsed by Cognizant.*
