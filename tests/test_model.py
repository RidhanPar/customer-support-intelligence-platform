"""Unit tests for the SLA-breach modelling layer in ``src/model.py``.

These cover the framework-independent pieces of the training pipeline:

* the preprocessing ``ColumnTransformer`` (imputation, scaling, OHE),
* leakage safety — preprocessing is fit on training rows only,
* ``evaluate_model`` metric computation, and
* ``compare_models`` table assembly.

They build a small synthetic ticket dataset rather than reading the CSV
artifact, so the suite runs anywhere (including CI) without trained models.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.model import (
    FEATURES_CATEGORICAL,
    FEATURES_NUMERIC,
    _build_preprocessor,
    compare_models,
    evaluate_model,
)


def _make_tickets(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic ticket frame with a learnable SLA-breach signal.

    Breach probability rises with ticket age and critical priority, giving the
    classifier real signal so evaluation metrics are meaningful (not degenerate).
    """
    rng = np.random.default_rng(seed)
    priority = rng.choice(["Low", "Medium", "High", "Critical"], size=n)
    age = rng.uniform(0, 72, size=n)

    priority_rank = pd.Series(priority).map(
        {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    ).to_numpy()
    logit = -3.0 + 0.05 * age + 0.9 * priority_rank
    breach = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

    day = rng.choice(
        ["Monday", "Tuesday", "Wednesday", "Saturday", "Sunday"], size=n
    )
    df = pd.DataFrame(
        {
            "ticket_age_hours": age,
            "hour_created": rng.integers(0, 24, size=n),
            "message_length": rng.integers(5, 500, size=n),
            "escalated": rng.integers(0, 2, size=n),
            "priority_encoded": priority_rank + 1,
            "is_critical": (priority == "Critical").astype(int),
            "is_high_priority": np.isin(priority, ["High", "Critical"]).astype(int),
            "is_weekend": np.isin(day, ["Saturday", "Sunday"]).astype(int),
            "priority": priority,
            "category": rng.choice(["Billing", "Technical", "Account"], size=n),
            "channel": rng.choice(["Email", "Chat", "Phone"], size=n),
            "team": rng.choice(["Tier1", "Tier2"], size=n),
            "status": rng.choice(["Open", "Closed"], size=n),
            "day_of_week": day,
            "sla_breach": breach,
        }
    )
    return df


def _fit_pipeline(df: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    model.fit(X, df["sla_breach"])
    return model


def test_preprocessor_covers_every_declared_feature():
    pre = _build_preprocessor()
    handled = [col for _, _, cols in pre.transformers for col in cols]
    assert sorted(handled) == sorted(FEATURES_NUMERIC + FEATURES_CATEGORICAL)


def test_preprocessor_handles_unseen_category_without_error():
    """OneHotEncoder(handle_unknown='ignore') must not raise on new categories."""
    train = _make_tickets(seed=1)
    pre = _build_preprocessor().fit(
        train[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    )

    novel = train.iloc[:1].copy()
    novel["channel"] = "CarrierPigeon"  # category never seen during fit
    transformed = pre.transform(novel[FEATURES_NUMERIC + FEATURES_CATEGORICAL])

    assert transformed.shape[0] == 1


def test_scaler_is_fit_on_training_rows_only():
    """Guards against data leakage: scaler statistics come from train, not test.

    The numeric scaler living inside a fitted Pipeline must reproduce the
    training set's per-feature means, proving the test split never influenced
    preprocessing.
    """
    df = _make_tickets(seed=2)
    train = df.iloc[:80]
    model = _fit_pipeline(train)

    scaler = (
        model.named_steps["preprocessor"]
        .named_transformers_["num"]
        .named_steps["scaler"]
    )
    expected_means = train[FEATURES_NUMERIC].median().where(
        train[FEATURES_NUMERIC].notna().all(), train[FEATURES_NUMERIC].mean()
    )
    # No NaNs in synthetic data, so median imputation is a no-op and the
    # scaler should have learned the raw training-column means.
    np.testing.assert_allclose(
        scaler.mean_, train[FEATURES_NUMERIC].mean().to_numpy(), rtol=1e-6
    )
    assert len(scaler.mean_) == len(FEATURES_NUMERIC)
    assert expected_means.notna().all()


def test_evaluate_model_returns_consistent_metrics():
    df = _make_tickets(seed=3)
    train, test = df.iloc[:90], df.iloc[90:]
    model = _fit_pipeline(train)

    metrics = evaluate_model(
        model, test[FEATURES_NUMERIC + FEATURES_CATEGORICAL], test["sla_breach"]
    )

    expected_keys = {
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "tn", "fp", "fn", "tp", "report", "confusion_matrix",
    }
    assert expected_keys.issubset(metrics)

    # Confusion-matrix cells account for every test row.
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == len(test)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0
    for key in ("tn", "fp", "fn", "tp"):
        assert isinstance(metrics[key], int)


def test_evaluate_model_handles_single_class_test_set():
    """ROC-AUC is undefined for one class; the function returns 0.0, not a crash."""
    df = _make_tickets(seed=4)
    model = _fit_pipeline(df)

    single = df[df["sla_breach"] == 0].iloc[:15].copy()
    assert single["sla_breach"].nunique() == 1

    metrics = evaluate_model(
        model,
        single[FEATURES_NUMERIC + FEATURES_CATEGORICAL],
        single["sla_breach"],
    )
    assert metrics["roc_auc"] == 0.0


def test_compare_models_builds_indexed_table():
    df = _make_tickets(seed=5)
    train, test = df.iloc[:90], df.iloc[90:]
    model = _fit_pipeline(train)
    metrics = evaluate_model(
        model, test[FEATURES_NUMERIC + FEATURES_CATEGORICAL], test["sla_breach"]
    )

    table = compare_models({"Logistic": metrics, "LogisticCopy": metrics})

    assert list(table.index) == ["Logistic", "LogisticCopy"]
    assert {"ROC-AUC", "Precision", "Recall", "F1", "Accuracy"}.issubset(table.columns)
    # Identical inputs produce identical rows.
    assert table.loc["Logistic"].equals(table.loc["LogisticCopy"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
