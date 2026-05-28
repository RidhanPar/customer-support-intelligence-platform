import pandas as pd
from pathlib import Path

SLA_TARGET_HOURS = {
    "Critical": 6,
    "High": 12,
    "Medium": 24,
    "Low": 48,
    "Unknown": 24,
}

COLUMN_ALIASES = {
    "Ticket ID": "ticket_id",
    "Ticket Subject": "subject",
    "Ticket Description": "customer_message",
    "Ticket Status": "status",
    "Ticket Priority": "priority",
    "Ticket Channel": "channel",
    "First Response Time": "first_response_time",
    "Time to Resolution": "resolution_hours",
    "Customer Satisfaction Rating": "customer_satisfaction",
    "Date of Purchase": "created_date",
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = standardize_columns(df)
    df = df.drop_duplicates()

    required_defaults = {
        "priority": "Unknown",
        "category": "Unknown",
        "channel": "Unknown",
        "team": "Unknown",
        "status": "Unknown",
        "customer_message": "No message provided",
        "escalated": 0,
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"TKT-{i+1:05d}" for i in range(len(df))]

    if "created_date" in df.columns:
        df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    else:
        df["created_date"] = pd.Timestamp.today()

    if "resolved_date" in df.columns:
        df["resolved_date"] = pd.to_datetime(df["resolved_date"], errors="coerce")
    else:
        df["resolved_date"] = pd.NaT

    if "resolution_hours" not in df.columns:
        df["resolution_hours"] = (df["resolved_date"] - df["created_date"]).dt.total_seconds() / 3600

    df["resolution_hours"] = pd.to_numeric(df["resolution_hours"], errors="coerce")
    median_resolution = df["resolution_hours"].median() if df["resolution_hours"].notna().any() else 24
    df["resolution_hours"] = df["resolution_hours"].fillna(median_resolution).clip(lower=0)

    df["priority"] = df["priority"].astype(str).str.title()
    df["sla_target_hours"] = df["priority"].map(SLA_TARGET_HOURS).fillna(24)
    df["sla_breach"] = (df["resolution_hours"] > df["sla_target_hours"]).astype(int)
    df["ticket_age_hours"] = (pd.Timestamp.now() - df["created_date"]).dt.total_seconds() / 3600
    df["hour_created"] = df["created_date"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["created_date"].dt.day_name().fillna("Unknown")
    df["message_length"] = df["customer_message"].astype(str).str.len()
    df["escalated"] = pd.to_numeric(df["escalated"], errors="coerce").fillna(0).astype(int)

    return df

if __name__ == "__main__":
    input_path = Path("data/raw/support_tickets.csv")
    output_path = Path("data/processed/cleaned_tickets.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = clean_data(str(input_path))
    cleaned.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
