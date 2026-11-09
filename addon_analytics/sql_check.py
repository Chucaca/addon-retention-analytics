"""Optional: run sql/segmentation.sql in DuckDB and compare with the pandas segments.

duckdb is imported lazily; when it is missing an ImportError is raised and the CLI skips the check.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def run_sql_checks(sql_dir: Path, messages: pd.DataFrame, segments: pd.DataFrame, as_of: str = "2025-01-01") -> str:
    import duckdb  # noqa: F401  (optional dependency)

    con = duckdb.connect()
    con.register("messages", messages)
    query = (Path(sql_dir) / "segmentation.sql").read_text(encoding="utf-8").replace("{as_of}", as_of)
    sql_seg = con.execute(query).df()
    merged = sql_seg.merge(segments[["merchant_id", "segment"]], on="merchant_id", suffixes=("_sql", "_pandas"))
    agree = (merged["segment_sql"] == merged["segment_pandas"]).mean() if len(merged) else float("nan")
    return f"duckdb segmentation.sql vs pandas: {len(merged)} merchants, {agree:.1%} identical segments"
