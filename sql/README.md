# SQL Layer

SQL-based alternative to the pandas feature pipeline, demonstrating production-ready data engineering patterns.

## Files

| File | Purpose |
|---|---|
| `schema.sql` | DDL — creates the `support_tickets` table with typed columns and indexes |
| `feature_extraction.sql` | Feature query — extracts model-ready features using CTEs and window functions |

## SQL techniques demonstrated

- **CTE (`WITH` clause)** — `ticket_enriched` computes resolution hours, SLA targets, and time-of-day features in a readable intermediate step before the final `SELECT`
- **Window function 1 — `AVG() OVER (PARTITION BY ... ROWS BETWEEN ...)`** — rolling average resolution time per team over the last 10 tickets; captures team-level workload pressure as a breach risk signal
- **Window function 2 — `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`** — ticket sequence within each team, useful for detecting early-vs-late handling patterns
- **Computed columns** — `sla_breach` (derived label), `sla_time_consumed_pct` (% of SLA window consumed; values above 100 mean a breach is already in progress)

## How to run

From the project root with the virtual environment active:

```python
from src.data_cleaning import clean_data
from src.db_loader import load_to_sqlite, extract_features_sql

df = clean_data("data/raw/support_tickets.csv")
load_to_sqlite(df, "support_tickets.db")
features = extract_features_sql("support_tickets.db")
print(features.head())
```

Or run the SQL directly in any SQLite client:

```bash
sqlite3 support_tickets.db < sql/feature_extraction.sql
```
