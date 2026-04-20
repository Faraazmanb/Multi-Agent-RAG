import tempfile
from pathlib import Path

from enterprise_ai.etl.pipeline import run_sample_etl


def test_etl_creates_tables():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.sqlite3"
        s = run_sample_etl(p)
        assert s.row_counts["stg_events"] == 3
        assert s.row_counts["fact_mentions"] == 3
