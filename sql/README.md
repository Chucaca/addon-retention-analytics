# SQL reference queries

Equivalent SQL for the core pandas logic. They are written in DuckDB dialect; the
production version of this kind of analysis ran as Spark SQL / HiveQL, and the
comments note the Spark spelling where it differs.

Placeholders such as `{as_of}` are substituted by the caller. Table names match
the synthetic frames: `messages`, `topups`, `merchants`.

If `duckdb` is installed, `python -m addon_analytics.cli all` executes
`segmentation.sql` against the in-memory synthetic data and checks that it
reproduces the pandas segments; otherwise the check is skipped.
