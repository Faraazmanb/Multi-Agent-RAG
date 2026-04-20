"""Optional Neo4j integration — graph context for RAG or analytics."""

from __future__ import annotations

from dataclasses import dataclass

from enterprise_ai.config import get_settings


@dataclass(frozen=True)
class Neo4jPing:
    ok: bool
    detail: str


def ping_neo4j() -> Neo4jPing:
    settings = get_settings()
    if not settings.neo4j_uri or not settings.neo4j_user or not settings.neo4j_password:
        return Neo4jPing(
            ok=False,
            detail="Neo4j not configured (set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).",
        )
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        return Neo4jPing(ok=False, detail=f"neo4j driver import failed: {e}")

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        driver.verify_connectivity()
        return Neo4jPing(ok=True, detail="Connected to Neo4j.")
    except Exception as e:
        return Neo4jPing(ok=False, detail=f"Neo4j connection failed: {e}")
    finally:
        driver.close()


def seed_demo_graph() -> Neo4jPing:
    """Creates a tiny (:Topic)-[:RELATES_TO]->(:Topic) pattern if DB is reachable."""
    settings = get_settings()
    if not settings.neo4j_uri or not settings.neo4j_user or not settings.neo4j_password:
        return Neo4jPing(ok=False, detail="Neo4j not configured.")
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        return Neo4jPing(ok=False, detail=f"neo4j driver import failed: {e}")

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        cypher = """
        MERGE (a:Topic {name: $a})
        MERGE (b:Topic {name: $b})
        MERGE (a)-[:RELATES_TO {weight: 0.9}]->(b)
        """
        with driver.session() as session:
            session.run(cypher, a="RAG", b="LangGraph")
            session.run(cypher, a="LangGraph", b="MCP")
        return Neo4jPing(ok=True, detail="Seeded demo Topic nodes and RELATES_TO edges.")
    except Exception as e:
        return Neo4jPing(ok=False, detail=f"Neo4j seed failed: {e}")
    finally:
        driver.close()
