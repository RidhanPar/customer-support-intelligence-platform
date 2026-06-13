from pathlib import Path

import pandas as pd

from src.data_cleaning import clean_data, standardize_columns
from src.feature_engineering import create_features
from src.model import FEATURES_NUMERIC


def test_standardize_columns_handles_source_aliases():
    frame = pd.DataFrame({"Ticket ID": ["T-1"], "Ticket Priority": ["High"]})

    result = standardize_columns(frame)

    assert list(result.columns) == ["ticket_id", "priority"]


def test_clean_data_derives_sla_breach_and_features(tmp_path: Path):
    source = tmp_path / "tickets.csv"
    pd.DataFrame(
        {
            "Ticket ID": ["T-1", "T-2"],
            "Ticket Priority": ["Critical", "Low"],
            "Ticket Description": ["Urgent outage", "Question"],
            "Date of Purchase": ["2026-01-01", "2026-01-02"],
            "Time to Resolution": [8, 12],
        }
    ).to_csv(source, index=False)

    result = clean_data(str(source))

    assert result["sla_breach"].tolist() == [1, 0]
    assert result["message_length"].tolist() == [13, 8]
    assert result["ticket_id"].tolist() == ["T-1", "T-2"]


def test_create_features_encodes_priority_and_weekend():
    frame = pd.DataFrame(
        {
            "priority": ["Critical", "Unknown"],
            "day_of_week": ["Saturday", "Monday"],
        }
    )

    result = create_features(frame)

    assert result["priority_encoded"].tolist() == [4, 2]
    assert result["is_critical"].tolist() == [1, 0]
    assert result["is_weekend"].tolist() == [1, 0]


def test_model_features_exclude_target_defining_resolution_time():
    assert "resolution_hours" not in FEATURES_NUMERIC
