"""Small in-repo knowledge base for RAG demos (no external downloads)."""

DEFAULT_CHUNKS: list[str] = [
    (
        "Retrieval-Augmented Generation (RAG) combines retrieval from a knowledge base "
        "with an LLM to ground answers and reduce hallucinations."
    ),
    (
        "LangGraph models agent workflows as graphs: nodes are steps, edges define "
        "control flow including cycles for multi-agent orchestration."
    ),
    (
        "Multi-agent systems route work between specialized agents; a supervisor "
        "coordinates intent detection, tool use, and handoffs between agents."
    ),
    (
        "MCP (Model Context Protocol) exposes tools and resources to models in a "
        "standard way; A2A (agent-to-agent) patterns describe interoperable agent messaging."
    ),
    (
        "Neo4j stores knowledge as property graphs; Cypher queries traverse relationships "
        "for explainable retrieval and entity-centric analytics."
    ),
    (
        "ETL pipelines extract from sources, transform with SQL or Python, and load "
        "into warehouses; idempotent loads and data quality checks are common patterns."
    ),
    (
        "Traditional ML includes classification, clustering, and regression; these "
        "complement LLM workflows for structured prediction and segmentation."
    ),
]
