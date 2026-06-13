from pathlib import Path
import sys
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.feature_engineering import create_features

DATA_PATH = PROJECT_ROOT / "data/processed/cleaned_tickets.csv"
MODEL_PATH = PROJECT_ROOT / "models/sla_breach_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models/model_metrics.txt"

FEATURES_NUMERIC = [
    "ticket_age_hours",
    "hour_created",
    "message_length",
    "escalated",
    "priority_encoded",
    "is_critical",
    "is_high_priority",
    "is_weekend",
]

FEATURES_CATEGORICAL = [
    "priority",
    "category",
    "channel",
    "team",
    "status",
    "day_of_week",
]

TARGET = "sla_breach"


def train_model():
    df = pd.read_csv(DATA_PATH)
    df = create_features(df)

    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df[TARGET]

    stratify_value = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value,
    )

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, FEATURES_NUMERIC),
            ("cat", categorical_pipe, FEATURES_CATEGORICAL),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "numeric": FEATURES_NUMERIC,
            "categorical": FEATURES_CATEGORICAL,
        },
        MODEL_PATH,
    )

    report = f"""
Accuracy: {accuracy:.4f}
F1-score: {f1:.4f}
ROC AUC: {roc_auc:.4f}

Classification Report:
{classification_report(y_test, preds, zero_division=0)}

Confusion Matrix:
{confusion_matrix(y_test, preds)}
"""

    METRICS_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
