from pathlib import Path
import sys
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.feature_engineering import create_features

DATA_PATH = PROJECT_ROOT / "data/processed/cleaned_tickets.csv"
MODEL_PATH = PROJECT_ROOT / "models/sla_breach_model.pkl"
XGB_MODEL_PATH = PROJECT_ROOT / "models/xgb_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models/model_metrics.txt"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "assets/screenshots/feature_importance.png"

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


def _build_preprocessor() -> ColumnTransformer:
    """Return a fresh ColumnTransformer for numeric and categorical features."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, FEATURES_NUMERIC),
        ("cat", categorical_pipe, FEATURES_CATEGORICAL),
    ])


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a fitted sklearn Pipeline on the test set and return a metrics dict.

    Args:
        model: Fitted sklearn Pipeline with a 'classifier' step.
        X_test: Test feature DataFrame.
        y_test: True labels for the test set.

    Returns:
        dict: Keys — roc_auc, precision, recall, f1, accuracy, tn, fp, fn, tp,
              report (str), confusion_matrix (ndarray).
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "report": classification_report(y_test, preds, zero_division=0),
        "confusion_matrix": cm,
    }


def compare_models(results: dict) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame of model evaluation metrics.

    Args:
        results: Dict mapping model name (str) to a metrics dict returned by
                 evaluate_model().

    Returns:
        pd.DataFrame: One row per model with columns ROC-AUC, Precision, Recall,
                      F1, Accuracy, TP, FP, TN, FN.
    """
    rows = []
    for model_name, m in results.items():
        rows.append({
            "Model": model_name,
            "ROC-AUC": round(m["roc_auc"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4),
            "Accuracy": round(m["accuracy"], 4),
            "TP": m["tp"],
            "FP": m["fp"],
            "TN": m["tn"],
            "FN": m["fn"],
        })
    return pd.DataFrame(rows).set_index("Model")


def plot_feature_importance(
    model: Pipeline, feature_names: list, output_path: str
) -> None:
    """
    Generate and save a horizontal bar chart of the top 10 Random Forest feature
    importances (Gini impurity reduction).

    Args:
        model: Fitted sklearn Pipeline containing a RandomForestClassifier as the
               'classifier' step.
        feature_names: Full ordered list of feature names after preprocessing
                       (numeric features first, then OHE-expanded categoricals).
        output_path: File path to save the PNG chart.

    Returns:
        None
    """
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#dc2626")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top 10 Features — SLA Breach Predictor")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance chart saved to {output_path}")


def train_model():
    df = pd.read_csv(DATA_PATH)
    df = create_features(df)

    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df[TARGET]

    stratify_value = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_value,
    )

    # --- Random Forest ---
    rf_model = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced",
        )),
    ])
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    # --- XGBoost ---
    # scale_pos_weight balances classes the same way class_weight="balanced" does for RF
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    xgb_model = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("classifier", XGBClassifier(
            n_estimators=250,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            verbosity=0,
        )),
    ])
    xgb_model.fit(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    # --- Save models ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": rf_model, "numeric": FEATURES_NUMERIC, "categorical": FEATURES_CATEGORICAL},
        MODEL_PATH,
    )
    joblib.dump(
        {"model": xgb_model, "numeric": FEATURES_NUMERIC, "categorical": FEATURES_CATEGORICAL},
        XGB_MODEL_PATH,
    )

    # --- Model comparison ---
    comparison = compare_models({"Random Forest": rf_metrics, "XGBoost": xgb_metrics})
    print("\nModel Comparison:")
    print(comparison.to_string())

    # --- Feature importance chart (Random Forest) ---
    rf_preprocessor = rf_model.named_steps["preprocessor"]
    cat_encoder = rf_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_features = cat_encoder.get_feature_names_out(FEATURES_CATEGORICAL).tolist()
    all_feature_names = FEATURES_NUMERIC + encoded_cat_features

    plot_feature_importance(rf_model, all_feature_names, str(FEATURE_IMPORTANCE_PATH))

    # --- Metrics file (format kept backward-compatible with streamlit_app.py parsers) ---
    report = f"""Random Forest:
Accuracy: {rf_metrics['accuracy']:.4f}
F1-score: {rf_metrics['f1']:.4f}
ROC AUC: {rf_metrics['roc_auc']:.4f}

Classification Report:
{rf_metrics['report']}

Confusion Matrix:
{rf_metrics['confusion_matrix']}

XGBoost:
Accuracy: {xgb_metrics['accuracy']:.4f}
F1-score (XGBoost): {xgb_metrics['f1']:.4f}
ROC AUC (XGBoost): {xgb_metrics['roc_auc']:.4f}

Classification Report:
{xgb_metrics['report']}

Confusion Matrix:
{xgb_metrics['confusion_matrix']}
"""

    METRICS_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Random Forest model saved to {MODEL_PATH}")
    print(f"XGBoost model saved to {XGB_MODEL_PATH}")

    return rf_metrics, xgb_metrics


if __name__ == "__main__":
    train_model()
