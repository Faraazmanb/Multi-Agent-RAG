"""SQLite ETL example: extract → transform → load with SQL-friendly shapes."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EtlSummary:
    database_path: str
    row_counts: dict[str, int]


def run_sample_etl(db_path: str | Path) -> EtlSummary:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS stg_events;
            DROP TABLE IF EXISTS dim_topic;
            DROP TABLE IF EXISTS fact_mentions;

            CREATE TABLE stg_events (
              id INTEGER PRIMARY KEY,
              raw_text TEXT NOT NULL,
              source TEXT NOT NULL
            );

            CREATE TABLE dim_topic (
              topic_id INTEGER PRIMARY KEY,
              name TEXT NOT NULL
            );

            CREATE TABLE fact_mentions (
              mention_id INTEGER PRIMARY KEY,
              topic_id INTEGER NOT NULL REFERENCES dim_topic(topic_id),
              weight REAL NOT NULL
            );

            INSERT INTO stg_events (raw_text, source) VALUES
              ('RAG with LangGraph for enterprise', 'doc_a'),
              ('Neo4j graph analytics for entities', 'doc_b'),
              ('ETL and SQL pipelines to warehouse', 'doc_c');

            INSERT INTO dim_topic (name) VALUES
              ('rag'), ('graph'), ('etl');

            INSERT INTO fact_mentions (topic_id, weight) VALUES
              (1, 1.0),
              (2, 0.8),
              (3, 0.9);
            """
        )
        conn.commit()
        counts = {}
        for table in ("stg_events", "dim_topic", "fact_mentions"):
            cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = int(cur.fetchone()[0])
        return EtlSummary(database_path=str(path.resolve()), row_counts=counts)
    finally:
        conn.close()
