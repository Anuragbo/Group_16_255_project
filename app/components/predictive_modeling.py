import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, fbeta_score

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
CURATED_DATA_PATH = ROOT / "data" / "curated.parquet"
BASELINE_MODEL = "LogisticRegression"


@st.cache_resource
def load_model_bundle():
    import joblib

    with MANIFEST_PATH.open() as handle:
        manifest = json.load(handle)
    model_meta = manifest[BASELINE_MODEL]
    model = joblib.load(MODELS_DIR / model_meta["path"])
    return model, model_meta


@st.cache_data
def score_dataset(model_name: str = BASELINE_MODEL):
    model, meta = load_model_bundle()
    df = pd.read_parquet(CURATED_DATA_PATH)
    feature_order = meta["feature_order"]
    X = df.drop(columns=["Churn"])[feature_order].astype(float)
    probs = model.predict_proba(X)[:, 1]
    return (
        pd.DataFrame(
            {
                "p_churn": probs,
                "Churn_actual": df["Churn"].values,
            },
            index=df.index,
        ),
        X,
    )


def _header(meta: dict) -> float:
    st.title("Predictive Modeling")
    st.caption("Baseline workflow powered by Logistic Regression.")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(meta["default_threshold"]),
        step=0.005,
        format="%.3f",
    )
    return threshold


def _kpis(meta: dict, threshold: float, df_scored: pd.DataFrame) -> None:
    y_true = df_scored["Churn_actual"].values
    y_pred = (df_scored["p_churn"] >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    precision = tp / max(1, (y_pred == 1).sum())
    recall = tp / max(1, (y_true == 1).sum())
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual Churn Rate", f"{df_scored['Churn_actual'].mean():.1%}")
    c2.metric("CV ROC-AUC", f"{meta['metrics']['cv_roc_auc']:.3f}")
    c3.metric("Recall", f"{recall:.1%}")
    c4.metric("Precision / F2", f"{precision:.1%} / {f2:.3f}")


def _confusion_matrix(threshold: float, df_scored: pd.DataFrame) -> None:
    y_true = df_scored["Churn_actual"].values
    y_prob = df_scored["p_churn"].values
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    counts = cm.astype(int)
    text = np.vectorize(lambda label, count: f"{label}<br>{count}")(labels, counts)
    color_codes = np.array([[0, 1], [2, 0]])
    colorscale = [
        [0.0, "#22C55E"],
        [0.33, "#22C55E"],
        [0.34, "#EF4444"],
        [0.66, "#EF4444"],
        [0.67, "#B91C1C"],
        [1.0, "#B91C1C"],
    ]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=color_codes,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color="white", size=14),
            colorscale=colorscale,
            showscale=False,
            zmin=0,
            zmax=2,
        )
    )
    fig_cm.update_layout(
        height=320,
        margin=dict(t=50, b=0),
        xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Predicted: Stay", "Predicted: Churn"]),
        yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Actual: Stay", "Actual: Churn"]),
        title=f"Confusion Matrix at threshold {threshold:.3f}",
    )
    st.plotly_chart(fig_cm, use_container_width=True)


def _prediction_table(threshold: float, df_scored: pd.DataFrame) -> None:
    st.subheader("Prediction Table")
    df_display = df_scored.copy()
    df_display["customer_index"] = df_display.index
    df_display["Predicted"] = np.where(df_display["p_churn"] >= threshold, "Churn", "Stay")
    df_display["Actual"] = df_display["Churn_actual"].map({1: "Churn", 0: "Stay"})
    df_display["p_churn_pct"] = df_display["p_churn"] * 100
    df_display["prob_rank"] = df_display["p_churn"].rank(method="min", ascending=False).astype(int)
    df_display = df_display.sort_values("p_churn", ascending=False)

    if "prediction_filter" in st.session_state:
        filter_label = st.session_state.get("prediction_filter_label", "segment filter active")
        c1, c2 = st.columns([4, 1])
        with c1:
            st.info(f"Showing: **{filter_label}**")
        with c2:
            if st.button("Clear filter"):
                del st.session_state["prediction_filter"]
                del st.session_state["prediction_filter_label"]
                st.rerun()
        df_display = df_display[df_display["customer_index"].isin(st.session_state["prediction_filter"])]

    st.dataframe(
        df_display[["customer_index", "prob_rank", "p_churn_pct", "Predicted", "Actual"]],
        use_container_width=True,
        height=420,
        column_config={
            "customer_index": st.column_config.NumberColumn("Customer ID"),
            "prob_rank": st.column_config.NumberColumn("Risk Rank"),
            "p_churn_pct": st.column_config.NumberColumn("Churn Probability %", format="%.2f"),
        },
    )


def render() -> None:
    model, meta = load_model_bundle()
    del model
    df_scored, _ = score_dataset()
    threshold = _header(meta)
    _kpis(meta, threshold, df_scored)
    st.divider()
    _confusion_matrix(threshold, df_scored)
    st.divider()
    _prediction_table(threshold, df_scored)
