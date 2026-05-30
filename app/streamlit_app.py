import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_cleaning import clean_data
from src.feature_engineering import create_features
from src.model import train_model

# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="Customer Support Intelligence Platform",
    page_icon="📊",
    layout="wide",
)
def inject_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f8fafc;
        }

        h1, h2, h3 {
            color: #0f172a;
            font-weight: 700;
        }

        .stMetric {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            border-bottom: 1px solid #e5e7eb;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #ffffff;
            border-radius: 10px 10px 0 0;
            padding: 10px 18px;
            border: 1px solid #e5e7eb;
            color: #334155;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background-color: #fee2e2;
            color: #dc2626;
            border-bottom: 2px solid #ef4444;
        }

        section[data-testid="stSidebar"] {
            background-color: #f1f5f9;
            border-right: 1px solid #e2e8f0;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

# ===============================
# Paths
# ===============================

MODEL_PATH = PROJECT_ROOT / "models/sla_breach_model.pkl"
DEMO_DATA_PATH = PROJECT_ROOT / "data/raw/support_tickets.csv"
MODEL_METRICS_PATH = PROJECT_ROOT / "models/model_metrics.txt"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data/processed/cleaned_tickets.csv"
# ===============================
# Styling
# ===============================

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.5rem;
}

.big-title {
    font-size: 36px;
    font-weight: 800;
    color: #18212f;
}

.subtitle {
    font-size: 16px;
    color: #5f6b7a;
    margin-bottom: 10px;
}

.info-card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06);
    border: 1px solid #edf0f5;
}

.risk-high {
    background:#ffecec;
    padding:14px;
    border-radius:14px;
    color:#9b111e;
    font-weight:700;
    border-left: 6px solid #d62828;
}

.risk-medium {
    background:#fff6df;
    padding:14px;
    border-radius:14px;
    color:#8a5a00;
    font-weight:700;
    border-left: 6px solid #f4a261;
}

.risk-low {
    background:#eaf8ef;
    padding:14px;
    border-radius:14px;
    color:#166534;
    font-weight:700;
    border-left: 6px solid #2a9d8f;
}
</style>
""",
    unsafe_allow_html=True,
)


# ===============================
# Header
# ===============================

st.markdown(
    '<div class="big-title">Customer Support Intelligence Platform</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">SLA breach prediction, operational analytics, and support risk monitoring.</div>',
    unsafe_allow_html=True,
)

st.divider()


# ===============================
# Data and Model Loading
# ===============================

@st.cache_data
def load_and_prepare(uploaded_file=None):
    if uploaded_file is not None:
        temp_path = PROJECT_ROOT / "data/raw/uploaded_support_tickets.csv"
        temp_path.write_bytes(uploaded_file.getvalue())
        df = clean_data(str(temp_path))
    else:
        df = clean_data(str(DEMO_DATA_PATH))

    df = create_features(df)
    return df


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None
def train_demo_model_from_app():
    """
    Trains the demo SLA breach prediction model from the included sample dataset.
    This is used by the Streamlit app when the model file is missing or needs refresh.
    """
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df = clean_data(str(DEMO_DATA_PATH))
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    train_model()

    load_model.clear()
    load_model_metrics.clear()
@st.cache_data
def load_model_metrics():
    if not MODEL_METRICS_PATH.exists():
        return {}, ""

    metrics_text = MODEL_METRICS_PATH.read_text(encoding="utf-8")

    accuracy_match = re.search(r"Accuracy:\s*([0-9.]+)", metrics_text)
    f1_match = re.search(r"F1-score:\s*([0-9.]+)", metrics_text)
    roc_auc_match = re.search(r"ROC AUC:\s*([0-9.]+)", metrics_text)
    test_records_match = re.search(r"accuracy\s+[0-9.]+\s+(\d+)", metrics_text)

    metrics = {
        "accuracy": float(accuracy_match.group(1)) if accuracy_match else None,
        "f1_score": float(f1_match.group(1)) if f1_match else None,
        "roc_auc": float(roc_auc_match.group(1)) if roc_auc_match else None,
        "test_records": int(test_records_match.group(1)) if test_records_match else None,
    }

    return metrics, metrics_text
def get_feature_importance(model_bundle):
    if model_bundle is None:
        return pd.DataFrame()

    try:
        model_pipeline = model_bundle["model"]
        numeric_features = model_bundle["numeric"]
        categorical_features = model_bundle["categorical"]

        preprocessor = model_pipeline.named_steps["preprocessor"]
        classifier = model_pipeline.named_steps["classifier"]

        categorical_pipeline = preprocessor.named_transformers_["cat"]
        encoder = categorical_pipeline.named_steps["encoder"]

        encoded_categorical_features = encoder.get_feature_names_out(
            categorical_features
        ).tolist()

        feature_names = numeric_features + encoded_categorical_features
        importances = classifier.feature_importances_

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )

        importance_df["importance_percentage"] = (
            importance_df["importance"] / importance_df["importance"].sum() * 100
        )

        return importance_df.sort_values(
            "importance_percentage", ascending=False
        ).head(15)

    except Exception:
        return pd.DataFrame()
# ===============================
# Helper Functions
# ===============================

def assign_risk_level(score, critical_threshold, high_threshold, medium_threshold):
    if score >= critical_threshold:
        return "Critical"
    elif score >= high_threshold:
        return "High"
    elif score >= medium_threshold:
        return "Medium"
    elif score >= 30:
        return "Watch"
    return "Low"


def generate_recommendation(row):
    risk_level = row.get("risk_level", "Low")
    priority = str(row.get("priority", "")).lower()
    status = str(row.get("status", "")).lower()
    age = row.get("ticket_age_hours", 0)
    resolution = row.get("resolution_hours", 0)
    escalated = row.get("escalated", 0)

    if risk_level == "Critical":
        if priority in ["critical", "high"]:
            return "Immediate senior review"
        return "Fast-track before SLA breach"

    if risk_level == "High":
        if escalated == 1:
            return "Check escalation progress"
        if age > 18:
            return "Prioritize in next queue review"
        return "Assign experienced agent"

    if risk_level == "Medium":
        if status in ["pending", "open"]:
            return "Follow up with ticket owner"
        if resolution > 12:
            return "Monitor resolution delay"
        return "Keep under active monitoring"

    if risk_level == "Watch":
        if priority in ["critical", "high"]:
            return "Keep priority watch"
        return "Normal queue with periodic check"

    return "Low risk, normal handling"


def recommendation_group(recommendation):
    text = str(recommendation).lower()

    if "immediate" in text or "fast-track" in text:
        return "Urgent Action"

    if "prioritize" in text or "assign" in text or "escalation" in text:
        return "Priority Handling"

    if "follow" in text or "monitor" in text or "watch" in text:
        return "Monitoring"

    return "Normal Handling"


def style_risk_level(value):
    if value == "Critical":
        return "background-color: #ffecec; color: #9b111e; font-weight: bold;"

    if value == "High":
        return "background-color: #fff0e6; color: #b45309; font-weight: bold;"

    if value == "Medium":
        return "background-color: #fff6df; color: #8a5a00; font-weight: bold;"

    if value == "Watch":
        return "background-color: #eef4ff; color: #1d4ed8; font-weight: bold;"

    return "background-color: #eaf8ef; color: #166534; font-weight: bold;"


def safe_mean(series):
    if len(series) == 0:
        return 0
    return series.mean()


# ===============================
# Sidebar
# ===============================

with st.sidebar:
    st.header("Controls")

    uploaded = st.file_uploader("Upload ticket CSV", type=["csv"])
    use_demo = st.toggle("Use demo dataset", value=True)

    st.divider()

    st.subheader("Risk Settings")

    critical_threshold = st.slider(
        "Critical risk threshold",
        min_value=75,
        max_value=95,
        value=85,
    )

    high_threshold = st.slider(
        "High risk threshold",
        min_value=50,
        max_value=85,
        value=70,
    )

    medium_threshold = st.slider(
        "Medium risk threshold",
        min_value=30,
        max_value=70,
        value=50,
    )

    top_n = st.slider(
        "Show top risky tickets",
        min_value=5,
        max_value=100,
        value=20,
    )

    st.markdown("### Model Controls")

    if st.button("Train / Refresh Model", use_container_width=True):
        with st.spinner("Training model using the demo support ticket dataset..."):
            try:
                train_demo_model_from_app()
                st.success("Model trained successfully. Refreshing dashboard...")
                st.rerun()
            except Exception as e:
                st.error(f"Model training failed: {e}")

    if MODEL_PATH.exists():
        st.success("Model is ready")
    else:
        st.warning("Model is not trained yet")


# ===============================
# Load Dataset
# ===============================

try:
    if uploaded is not None:
        df = load_and_prepare(uploaded)
    elif use_demo:
        df = load_and_prepare()
    else:
        df = pd.DataFrame()

except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()


if df.empty:
    st.info("Upload a CSV or enable the demo dataset.")
    st.stop()


# ===============================
# Global Filters
# ===============================

st.subheader("Filters")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    priorities = st.multiselect(
        "Priority",
        sorted(df["priority"].dropna().unique()),
        default=sorted(df["priority"].dropna().unique()),
    )

with filter_col2:
    categories = st.multiselect(
        "Category",
        sorted(df["category"].dropna().unique()),
        default=sorted(df["category"].dropna().unique()),
    )

with filter_col3:
    channels = st.multiselect(
        "Channel",
        sorted(df["channel"].dropna().unique()),
        default=sorted(df["channel"].dropna().unique()),
    )

filtered = df[
    df["priority"].isin(priorities)
    & df["category"].isin(categories)
    & df["channel"].isin(channels)
].copy()

if filtered.empty:
    st.warning("No records found for the selected filters.")
    st.stop()


# ===============================
# Tabs
# ===============================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Dashboard", "Risk Monitor", "Dataset Explorer", "Model Performance"]
)


# ===============================
# Tab 1: Executive Dashboard
# ===============================

with tab1:
    st.subheader("Operational Overview")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total Tickets", f"{len(filtered):,}")
    kpi2.metric("SLA Breach Rate", f"{safe_mean(filtered['sla_breach']) * 100:.1f}%")
    kpi3.metric("Avg Resolution Hours", f"{safe_mean(filtered['resolution_hours']):.1f}")
    kpi4.metric("Escalation Rate", f"{safe_mean(filtered['escalated']) * 100:.1f}%")

    breach_rate = safe_mean(filtered["sla_breach"]) * 100
    avg_resolution = safe_mean(filtered["resolution_hours"])
    escalation_rate = safe_mean(filtered["escalated"]) * 100

    top_risk_category = (
        filtered.groupby("category")["sla_breach"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
    )

    top_category_name = top_risk_category.index[0] if not top_risk_category.empty else "N/A"
    top_category_breach = top_risk_category.iloc[0] * 100 if not top_risk_category.empty else 0

    st.markdown("### Automated Executive Insights")

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    with insight_col1:
        if breach_rate >= 70:
            st.error(
                f"SLA breach rate is high at {breach_rate:.1f}%. Immediate queue review is recommended."
            )
        elif breach_rate >= 40:
            st.warning(
                f"SLA breach rate is moderate at {breach_rate:.1f}%. Monitor high-priority queues closely."
            )
        else:
            st.success(
                f"SLA breach rate is controlled at {breach_rate:.1f}%. Current support flow looks stable."
            )

    with insight_col2:
        st.info(
            f"The highest-risk category is {top_category_name}, with a breach rate of {top_category_breach:.1f}%."
        )

    with insight_col3:
        if escalation_rate >= 30:
            st.warning(
                f"Escalation rate is {escalation_rate:.1f}%, suggesting possible complexity or backlog pressure."
            )
        else:
            st.success(
                f"Escalation rate is {escalation_rate:.1f}%, indicating manageable escalation volume."
            )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered,
            x="priority",
            color="sla_breach",
            barmode="group",
            title="SLA Breach by Priority",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        category_breach = (
            filtered.groupby("category", as_index=False)["sla_breach"]
            .mean()
            .sort_values("sla_breach", ascending=False)
        )

        fig = px.bar(
            category_breach,
            x="category",
            y="sla_breach",
            title="Breach Rate by Category",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        daily = filtered.copy()
        daily["created_day"] = pd.to_datetime(daily["created_date"]).dt.date

        daily_volume = daily.groupby("created_day", as_index=False).size()

        fig = px.line(
            daily_volume,
            x="created_day",
            y="size",
            title="Ticket Volume Trend",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(
            filtered,
            x="priority",
            y="resolution_hours",
            title="Resolution Time Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)


# ===============================
# Tab 2: Risk Monitor
# ===============================

with tab2:
    st.subheader("SLA Breach Prediction and Dynamic Recommendations")

    model_bundle = load_model()

    if model_bundle is None:
        st.warning(
            "Model not found. Run `python src/data_cleaning.py` and `python src/model.py` first."
        )

    else:
        model = model_bundle["model"]
        numeric = model_bundle["numeric"]
        categorical = model_bundle["categorical"]

        required_columns = numeric + categorical
        missing_columns = [col for col in required_columns if col not in filtered.columns]

        if missing_columns:
            st.error(f"Missing columns for prediction: {missing_columns}")
            st.stop()

        prediction_data = filtered[required_columns].copy()

        risk_scores = model.predict_proba(prediction_data)[:, 1]

        output_columns = [
            "ticket_id",
            "priority",
            "category",
            "channel",
            "team",
            "status",
            "resolution_hours",
            "ticket_age_hours",
            "escalated",
            "sla_breach",
        ]

        available_output_columns = [
            col for col in output_columns if col in filtered.columns
        ]

        output = filtered[available_output_columns].copy()

        output["predicted_risk_score"] = np.round(risk_scores * 100, 2)

        output["risk_level"] = output["predicted_risk_score"].apply(
            lambda score: assign_risk_level(
                score,
                critical_threshold,
                high_threshold,
                medium_threshold,
            )
        )

        output["recommendation"] = output.apply(generate_recommendation, axis=1)
        output["action_group"] = output["recommendation"].apply(recommendation_group)

        avg_risk = output["predicted_risk_score"].mean()
        critical_count = (output["risk_level"] == "Critical").sum()
        high_count = (output["risk_level"] == "High").sum()
        medium_count = (output["risk_level"] == "Medium").sum()

        r1, r2, r3, r4 = st.columns(4)

        r1.metric("Average Risk", f"{avg_risk:.1f}%")
        r2.metric("Critical Tickets", critical_count)
        r3.metric("High Risk Tickets", high_count)
        r4.metric("Medium Risk Tickets", medium_count)

        if avg_risk >= critical_threshold:
            st.markdown(
                f'<div class="risk-high">Average Risk: {avg_risk:.1f}% — Critical queue pressure detected.</div>',
                unsafe_allow_html=True,
            )

        elif avg_risk >= high_threshold:
            st.markdown(
                f'<div class="risk-medium">Average Risk: {avg_risk:.1f}% — High-risk tickets need close attention.</div>',
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                f'<div class="risk-low">Average Risk: {avg_risk:.1f}% — Current support flow looks manageable.</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        c1, c2 = st.columns(2)

        with c1:
            risk_order = ["Critical", "High", "Medium", "Watch", "Low"]

            risk_summary = (
                output["risk_level"]
                .value_counts()
                .reindex(risk_order)
                .fillna(0)
                .reset_index()
            )

            risk_summary.columns = ["risk_level", "count"]

            fig = px.bar(
                risk_summary,
                x="risk_level",
                y="count",
                title="Tickets by Risk Level",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            action_summary = output["action_group"].value_counts().reset_index()
            action_summary.columns = ["action_group", "count"]

            fig = px.pie(
                action_summary,
                names="action_group",
                values="count",
                title="Recommended Action Mix",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        selected_risk = st.multiselect(
            "Filter by risk level",
            ["Critical", "High", "Medium", "Watch", "Low"],
            default=["Critical", "High", "Medium", "Watch", "Low"],
        )

        selected_actions = st.multiselect(
            "Filter by recommendation group",
            sorted(output["action_group"].unique()),
            default=sorted(output["action_group"].unique()),
        )

        risk_table = output[
            output["risk_level"].isin(selected_risk)
            & output["action_group"].isin(selected_actions)
        ].sort_values("predicted_risk_score", ascending=False)

        st.dataframe(
            risk_table.head(top_n).style.map(
                style_risk_level,
                subset=["risk_level"],
            ),
            use_container_width=True,
        )

        csv = risk_table.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download risk prediction results",
            csv,
            "sla_risk_predictions.csv",
            "text/csv",
        )


# ===============================
# Tab 3: Dataset Explorer
# ===============================

with tab3:
    st.subheader("Dataset Explorer")

    search_text = st.text_input("Search by ticket ID, category, team, or status")

    explorer_df = filtered.copy()

    if search_text:
        search_text = search_text.lower()

        explorer_df = explorer_df[
            explorer_df.astype(str)
            .apply(
                lambda row: row.str.lower().str.contains(search_text).any(),
                axis=1,
            )
        ]

    st.dataframe(explorer_df.head(100), use_container_width=True)

    csv = explorer_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download filtered dataset",
        csv,
        "filtered_support_tickets.csv",
        "text/csv",
    )
# ===============================
# Tab 4: Model Performance
# ===============================
with tab4:
    st.subheader("Model Performance and Validation")

    st.markdown(
        """
        This section explains how the SLA breach prediction model performs on the test dataset.
        These results are useful for understanding whether the model is reliable enough for
        decision-support use cases such as ticket prioritization and operational risk monitoring.
        """
    )

    metrics, metrics_text = load_model_metrics()

    m1, m2, m3, m4 = st.columns(4)

    if metrics:
        accuracy = metrics.get("accuracy")
        f1_score = metrics.get("f1_score")
        roc_auc = metrics.get("roc_auc")
        test_records = metrics.get("test_records")

        m1.metric("Accuracy", f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A")
        m2.metric("F1-score", f"{f1_score:.4f}" if f1_score is not None else "N/A")
        m3.metric("ROC AUC", f"{roc_auc:.4f}" if roc_auc is not None else "N/A")
        m4.metric("Test Records", f"{test_records:,}" if test_records is not None else "N/A")
    else:
        st.warning("Model metrics file was not found.")

    st.info(
        "The current model uses fictional demo data. The results demonstrate the machine learning workflow, "
        "but should not be treated as production-level performance without validation on real operational data."
    )

    st.markdown("### Classification Report and Confusion Matrix")

    if metrics_text:
        st.code(metrics_text, language="text")
    else:
        st.warning("Model metrics file was not found.")

    st.markdown("### Top Model Drivers")

    model_bundle = load_model()
    feature_importance = get_feature_importance(model_bundle)

    if not feature_importance.empty:
        fig = px.bar(
            feature_importance.sort_values("importance_percentage"),
            x="importance_percentage",
            y="feature",
            orientation="h",
            title="Top Features Influencing SLA Breach Prediction",
            labels={
                "importance_percentage": "Importance (%)",
                "feature": "Feature",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

        display_importance = feature_importance.copy()
        display_importance["importance_percentage"] = display_importance[
            "importance_percentage"
        ].round(2)

        st.dataframe(
            display_importance[["feature", "importance_percentage"]],
            use_container_width=True,
        )
    else:
        st.info(
            "Feature importance is not available. Train the model first using `python src/model.py`."
        )

    st.markdown(
        """
        ### Interpretation

        - The model performs strongly on the demo test dataset.
        - The F1-score is useful because SLA breach prediction is a classification problem.
        - ROC AUC helps evaluate how well the model separates breach and non-breach cases.
        - The confusion matrix helps identify false positives and false negatives.
        - Feature importance shows which operational factors influence the model prediction most.
        - In a real business environment, the model should be retrained and validated using live ticket data.
        """
    )