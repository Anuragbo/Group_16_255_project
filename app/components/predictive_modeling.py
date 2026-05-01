import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, fbeta_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
CURATED_DATA_PATH = ROOT / "data" / "curated.parquet"


@st.cache_resource
def load_models():
    import joblib

    with MANIFEST_PATH.open() as handle:
        manifest = json.load(handle)
    models, thresholds = {}, {}
    for name, meta in manifest.items():
        models[name] = joblib.load(MODELS_DIR / meta["path"])
        thresholds[name] = meta["default_threshold"]
    return models, thresholds, manifest


@st.cache_data
def score_dataset(model_name: str):
    models, _, manifest = load_models()
    df = pd.read_parquet(CURATED_DATA_PATH)
    feature_order = manifest[model_name]["feature_order"]
    X = df.drop(columns=["Churn"])[feature_order].astype(float)
    probs = models[model_name].predict_proba(X)[:, 1]
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


@st.cache_data
def score_all_models() -> pd.DataFrame:
    models, thresholds, manifest = load_models()
    df = pd.read_parquet(CURATED_DATA_PATH)
    result_df = pd.DataFrame({"Churn_actual": df["Churn"].values}, index=df.index)
    for name, model in models.items():
        X = df.drop(columns=["Churn"])[manifest[name]["feature_order"]].astype(float)
        probs = model.predict_proba(X)[:, 1]
        result_df[f"p_{name}"] = probs
        result_df[f"pred_{name}"] = (probs >= thresholds[name]).astype(int)
    pred_cols = [col for col in result_df.columns if col.startswith("pred_")]
    result_df["churn_pred_count"] = result_df[pred_cols].sum(axis=1)
    result_df["non_churn_pred_count"] = len(pred_cols) - result_df["churn_pred_count"]
    result_df["consensus_label"] = np.where(
        result_df["churn_pred_count"] > result_df["non_churn_pred_count"],
        "Churn",
        np.where(
            result_df["churn_pred_count"] < result_df["non_churn_pred_count"],
            "Stay",
            "Tie",
        ),
    )
    return result_df


@st.cache_data
def model_comparison_metrics() -> pd.DataFrame:
    _, thresholds, manifest = load_models()
    rows = []
    for model_name in manifest:
        scored_df, _ = score_dataset(model_name)
        y_true = scored_df["Churn_actual"].values
        y_prob = scored_df["p_churn"].values
        threshold = thresholds[model_name]
        y_pred = (y_prob >= threshold).astype(int)
        precision = ((y_pred == 1) & (y_true == 1)).sum() / max(1, (y_pred == 1).sum())
        recall = ((y_pred == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        rows.append(
            {
                "Model": model_name,
                "CV ROC-AUC": manifest[model_name]["metrics"]["cv_roc_auc"],
                "Default Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F2": f2,
            }
        )
    return pd.DataFrame(rows).sort_values("CV ROC-AUC", ascending=False)


def _section1_header() -> tuple[str, dict]:
    st.title("Predictive Modeling")

    _, _, manifest = load_models()
    model_names = list(manifest.keys())

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = model_names[0]
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = manifest[st.session_state["selected_model"]]["default_threshold"]

    selected_model = st.selectbox(
        "Model",
        model_names,
        index=model_names.index(st.session_state["selected_model"]),
        key="model_selector",
    )

    if selected_model != st.session_state.get("selected_model"):
        st.session_state["threshold"] = manifest[selected_model]["default_threshold"]
        st.session_state["selected_model"] = selected_model

    return selected_model, manifest


def _section2_kpis(selected_model: str, manifest: dict, df_scored: pd.DataFrame) -> None:
    auc = manifest[selected_model]["metrics"]["cv_roc_auc"]
    default_t = manifest[selected_model]["default_threshold"]
    churn_total = int(df_scored["Churn_actual"].sum())
    total = int(df_scored["Churn_actual"].shape[0])
    non_churn_total = total - churn_total
    churn_rate = churn_total / total if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual Churn Rate", f"{churn_rate:.1%}")
    c2.metric("Churn / Non-Churn", f"{churn_total} / {non_churn_total}")
    c3.metric("CV ROC-AUC", f"{auc:.3f}")
    c4.metric("Default Threshold (F2)", f"{default_t:.3f}")


def _section2_threshold_kpis(threshold: float, df_scored: pd.DataFrame) -> None:
    y_true = df_scored["Churn_actual"].values
    y_pred = (df_scored["p_churn"] >= threshold).astype(int)
    predicted_rate = y_pred.mean() if y_pred.size > 0 else 0.0
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    precision = tp / max(1, (y_pred == 1).sum())
    recall = tp / max(1, (y_true == 1).sum())
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Churn Rate", f"{predicted_rate:.1%}")
    c2.metric("Recall", f"{recall:.1%}")
    c3.metric("Precision", f"{precision:.1%}")
    c4.metric("F2", f"{f2:.3f}")


def _section3_threshold(selected_model: str, manifest: dict) -> float:
    default_t = manifest[selected_model]["default_threshold"]
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("threshold", default_t),
        step=0.005,
        format="%.3f",
        help="Lower thresholds increase recall at the cost of precision.",
        key="threshold_slider",
    )
    st.session_state["threshold"] = threshold
    return threshold


def _section4_eval(selected_model: str, threshold: float, df_scored: pd.DataFrame) -> None:
    y_true = df_scored["Churn_actual"].values
    y_prob = df_scored["p_churn"].values
    col_l, col_m, col_r = st.columns([1, 1, 1])

    with col_l:
        st.subheader("PR Curve")
        precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_prob)
        idx = np.searchsorted(thresholds_pr, threshold, side="left")
        precision_t = precisions[min(idx, len(precisions) - 1)]
        recall_t = recalls[min(idx, len(recalls) - 1)]

        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=recalls,
                y=precisions,
                mode="lines",
                name="PR Curve",
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
            )
        )
        fig_pr.add_trace(
            go.Scatter(
                x=[recall_t],
                y=[precision_t],
                mode="markers",
                name=f"t={threshold:.3f}",
                marker=dict(size=10, color="#EF4444"),
            )
        )
        fig_pr.add_vline(x=recall_t, line_dash="dash", line_color="grey")
        fig_pr.add_hline(y=precision_t, line_dash="dash", line_color="grey")
        fig_pr.update_layout(
            height=300,
            margin=dict(t=50, b=0),
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_m:
        st.subheader("Confusion Matrix")
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        labels = np.array([["TN", "FP"], ["FN", "TP"]])
        descriptions = np.array(
            [
                ["True Negative", "False Alarm"],
                ["Missed Churn", "True Positive"],
            ]
        )
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
                customdata=np.dstack([labels, descriptions, counts]),
                hovertemplate="%{customdata[1]}<br>%{customdata[0]}: %{customdata[2]}<extra></extra>",
                colorscale=colorscale,
                showscale=False,
                zmin=0,
                zmax=2,
            )
        )
        fig_cm.update_layout(
            height=300,
            margin=dict(t=50, b=0),
            xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Predicted: Stay", "Predicted: Churn"]),
            yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Actual: Stay", "Actual: Churn"]),
            title=f"Threshold = {threshold:.3f}",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_r:
        st.subheader("Metrics vs Threshold")
        thresholds_sweep = np.linspace(0.0, 1.0, 201)
        y_mat = y_prob[:, None] >= thresholds_sweep[None, :]
        y_true_col = y_true[:, None] == 1
        tp = (y_mat & y_true_col).sum(axis=0).astype(float)
        fp = (y_mat & ~y_true_col).sum(axis=0).astype(float)
        fn = (~y_mat & y_true_col).sum(axis=0).astype(float)
        precisions = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recalls = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f2s = np.divide(5 * tp, 5 * tp + 4 * fn + fp, out=np.zeros_like(tp), where=(5 * tp + 4 * fn + fp) > 0)

        sens_df = pd.DataFrame(
            {
                "Threshold": thresholds_sweep,
                "Precision": precisions,
                "Recall": recalls,
                "F2": f2s,
            }
        )
        fig_sens = px.line(
            sens_df.melt(id_vars="Threshold", var_name="Metric", value_name="Score"),
            x="Threshold",
            y="Score",
            color="Metric",
        )
        fig_sens.add_vline(x=threshold, line_dash="dash", line_color="grey", annotation_text=f"t={threshold:.3f}")
        fig_sens.update_layout(height=300, margin=dict(t=50, b=0))
        st.plotly_chart(fig_sens, use_container_width=True)


def _section4b_model_comparison() -> None:
    st.subheader("Model Comparison")
    comparison_df = model_comparison_metrics()

    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(
            comparison_df.style.format(
                {
                    "CV ROC-AUC": "{:.3f}",
                    "Default Threshold": "{:.3f}",
                    "Precision": "{:.1%}",
                    "Recall": "{:.1%}",
                    "F2": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        fig = px.bar(
            comparison_df.melt(
                id_vars="Model",
                value_vars=["CV ROC-AUC", "Precision", "Recall", "F2"],
                var_name="Metric",
                value_name="Score",
            ),
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Default-threshold performance across available models",
        )
        fig.update_layout(height=320, margin=dict(t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)


def _section5_prediction_table(selected_model: str, threshold: float, df_scored: pd.DataFrame, X_full: pd.DataFrame):
    del selected_model, X_full
    st.subheader("Ranked Customer Predictions")

    df_display = df_scored.copy()
    df_display["customer_index"] = df_display.index
    df_display["Predicted"] = np.where(df_display["p_churn"] >= threshold, "Churn", "Stay")
    df_display["prob_rank"] = df_display["p_churn"].rank(method="min", ascending=False).astype(int)
    df_display["distance_to_threshold_pct"] = (df_display["p_churn"] - threshold).abs() * 100
    df_display["result_label"] = np.select(
        [
            (df_display["Churn_actual"] == 1) & (df_display["Predicted"] == "Churn"),
            (df_display["Churn_actual"] == 0) & (df_display["Predicted"] == "Churn"),
            (df_display["Churn_actual"] == 1) & (df_display["Predicted"] == "Stay"),
            (df_display["Churn_actual"] == 0) & (df_display["Predicted"] == "Stay"),
        ],
        ["TP", "FP", "FN", "TN"],
        default="",
    )
    df_display["Actual"] = df_display["Churn_actual"].map({1: "Churn", 0: "Stay"})
    df_display["p_churn_pct"] = df_display["p_churn"] * 100

    agreement_df = score_all_models()[["churn_pred_count", "non_churn_pred_count", "consensus_label"]]
    df_display = df_display.join(agreement_df)
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

    selection = st.dataframe(
        df_display[
            [
                "customer_index",
                "prob_rank",
                "p_churn_pct",
                "distance_to_threshold_pct",
                "Predicted",
                "Actual",
                "result_label",
                "consensus_label",
            ]
        ].reset_index(drop=True),
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
        height=400,
        column_config={
            "customer_index": st.column_config.NumberColumn("Customer ID"),
            "prob_rank": st.column_config.NumberColumn("Risk Rank"),
            "p_churn_pct": st.column_config.NumberColumn("Churn Probability %", format="%.2f"),
            "distance_to_threshold_pct": st.column_config.NumberColumn("Distance to Threshold %", format="%.2f"),
            "result_label": st.column_config.TextColumn("Result"),
            "Actual": st.column_config.TextColumn("Actual"),
            "consensus_label": st.column_config.TextColumn("Model Consensus"),
        },
    )
    return selection


def render() -> None:
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] input {
            opacity: 0 !important;
            pointer-events: none !important;
            caret-color: transparent !important;
        }
        div[data-baseweb="select"] * { cursor: pointer !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    selected_model, manifest = _section1_header()
    df_scored, X_full = score_dataset(selected_model)

    _section2_kpis(selected_model, manifest, df_scored)
    st.divider()
    threshold = _section3_threshold(selected_model, manifest)
    _section2_threshold_kpis(threshold, df_scored)
    st.divider()
    _section4_eval(selected_model, threshold, df_scored)
    st.divider()
    _section4b_model_comparison()
    st.divider()
    _section5_prediction_table(selected_model, threshold, df_scored, X_full)
