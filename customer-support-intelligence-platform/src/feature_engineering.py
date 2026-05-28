import pandas as pd

PRIORITY_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4, "Unknown": 2}

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["priority_encoded"] = df["priority"].map(PRIORITY_ORDER).fillna(2).astype(int)
    df["is_critical"] = (df["priority"] == "Critical").astype(int)
    df["is_high_priority"] = df["priority"].isin(["High", "Critical"]).astype(int)
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return df
