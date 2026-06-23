import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
_QUERY_PATH = PROJECT_ROOT / "sql" / "feature_extraction.sql"

_RAW_COLUMNS = [
    "ticket_id",
    "created_date",
    "resolved_date",
    "priority",
    "category",
    "channel",
    "team",
    "status",
    "escalated",
    "customer_satisfaction",
    "customer_message",
]


def load_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    """
    Insert a cleaned ticket DataFrame into a SQLite database using the schema
    defined in sql/schema.sql.

    Only the raw schema columns are inserted; derived feature columns produced
    by data_cleaning.py are excluded so that sql/feature_extraction.sql can
    recompute them from the raw data.

    Args:
        df: Cleaned DataFrame returned by src.data_cleaning.clean_data().
        db_path: File path to the SQLite database (created if it does not exist).

    Returns:
        None
    """
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")

    insert_df = df[[c for c in _RAW_COLUMNS if c in df.columns]].copy()

    for col in _RAW_COLUMNS:
        if col not in insert_df.columns:
            insert_df[col] = None

    for col in ("created_date", "resolved_date"):
        if col in insert_df.columns:
            insert_df[col] = insert_df[col].astype(str).replace("NaT", None)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        insert_df.to_sql("support_tickets", conn, if_exists="replace", index=False)

    print(f"Loaded {len(insert_df):,} tickets into {db_path}")


def extract_features_sql(db_path: str) -> pd.DataFrame:
    """
    Run the feature extraction SQL query against a SQLite database and return
    the result as a DataFrame.

    The query uses a CTE and window functions to compute resolution hours, SLA
    breach labels, rolling team averages, and other model-ready features.

    Args:
        db_path: File path to the SQLite database populated by load_to_sqlite().

    Returns:
        pd.DataFrame: Feature table with one row per ticket, ready for
                      inspection or model input.
    """
    query = _QUERY_PATH.read_text(encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
