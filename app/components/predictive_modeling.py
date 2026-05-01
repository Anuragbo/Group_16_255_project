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


@st.cache_resource
def load_shap_explainers():
    import shap

    models, _, manifest = load_models()
    df = pd.read_parquet(CURATED_DATA_PATH)
    explainers = {}
    for name, model in models.items():
        feature_order = manifest[name]["feature_order"]
        X = df.drop(columns=["Churn"])[feature_order].astype(float)
        if name == "LogisticRegression":
            background = X.sample(n=min(200, len(X)), random_state=42)
            explainers[name] = shap.LinearExplainer(
                model,
                background,
                feature_perturbation="interventional",
            )
        else:
            explainers[name] = shap.TreeExplainer(model)
    return explainers


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

    col_table, col_export = st.columns([3, 1])
    with col_table:
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
    with col_export:
        n_top = st.number_input("Export top-N by risk", min_value=10, max_value=int(len(df_display) or 10), value=min(100, max(10, int(len(df_display) or 10))))
        st.download_button(
            "Download Intervention List",
            df_display.head(int(n_top)).to_csv(index=False),
            "intervention_list.csv",
        )
    return selection, df_display, X_full


def _section6_local_explainability(selected_model: str, threshold: float, selection, df_display: pd.DataFrame, X_full: pd.DataFrame) -> None:
    if not selection.selection.rows:
        return

    selected_pos = selection.selection.rows[0]
    selected_row = df_display.reset_index(drop=True).iloc[selected_pos]
    selected_idx = int(selected_row["customer_index"])
    p = float(selected_row["p_churn"])
    actual = int(selected_row["Churn_actual"])
    predicted = "Churn" if p >= threshold else "Stay"
    correct = (predicted == "Churn" and actual == 1) or (predicted == "Stay" and actual == 0)

    st.divider()
    st.subheader(f"Customer #{selected_idx} Local Explainability")

    explainers = load_shap_explainers()
    row = X_full.loc[[selected_idx]]
    shap_values = explainers[selected_model].shap_values(row)
    expected_value = explainers[selected_model].expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    expected_value = float(np.asarray(expected_value))
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    if hasattr(shap_values, "ndim") and shap_values.ndim == 2:
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values).ravel()

    feature_names = row.columns.tolist()
    feature_values = row.iloc[0].values
    if shap_values.size != len(feature_names):
        n = min(shap_values.size, len(feature_names))
        shap_values = shap_values[:n]
        feature_names = feature_names[:n]
        feature_values = feature_values[:n]

    contributions = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "value": feature_values,
                "contribution": shap_values,
            }
        )
        .sort_values("contribution", key=abs, ascending=False)
        .head(15)
    )
    contributions["color"] = contributions["contribution"].apply(lambda value: "#EF4444" if value > 0 else "#22C55E")

    fig = px.bar(
        contributions,
        x="contribution",
        y="feature",
        orientation="h",
        color="color",
        color_discrete_map="identity",
        hover_data={"value": ":.3f", "color": False},
        title="SHAP contributions (red increases churn risk, green decreases it)",
    )
    fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"}, height=420, margin=dict(t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)
    final_logodds = expected_value + float(shap_values.sum())
    st.caption(f"Base log-odds: {expected_value:.2f} -> final log-odds: {final_logodds:.2f} -> probability: {p:.1%}")
    correctness_label = "Correct" if correct else "Incorrect"
    st.caption(f"Predicted: {predicted} (p={p:.3f}, t={threshold:.3f}) | Actual: {actual} | {correctness_label}")


def _section7_feature_impact_analysis(selected_model: str) -> None:
    st.subheader("Feature Impact Analysis")
    interp_dir = ROOT / "outputs" / "interpretability"
    if selected_model == "LogisticRegression":
        coef_df = pd.read_csv(interp_dir / "lr_coefficients.csv")
        top20 = coef_df.sort_values("coefficient", key=abs, ascending=False).head(20).sort_values("coefficient")
        top20["color"] = top20["coefficient"].apply(lambda value: "#EF4444" if value > 0 else "#22C55E")
        top20["OR"] = top20["odds_ratio"].round(2)

        fig = px.bar(
            top20,
            x="coefficient",
            y="feature",
            orientation="h",
            color="color",
            color_discrete_map="identity",
            hover_data={"OR": True, "color": False},
            title="Top 20 logistic regression coefficients",
            labels={"OR": "Odds Ratio"},
        )
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"}, height=500, margin=dict(t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
    elif selected_model == "RandomForest":
        st.image(str(interp_dir / "rf_shap_summary.png"), use_container_width=True)
    elif selected_model == "XGBoost":
        st.image(str(interp_dir / "xgb_shap_summary.png"), use_container_width=True)
    else:
        st.info("Global interpretability is available only for the serialized models in this dashboard.")


def _section8_whatif(selected_model: str, threshold: float) -> None:
    with st.expander("What-If Scoring", expanded=False):
        st.caption("Adjust a hypothetical customer profile and score it with the selected model.")

        c1, c2, c3 = st.columns(3)
        with c1:
            tenure_wi = st.slider("Tenure (months)", 0, 72, 12, key="wi_tenure")
            monthly_wi = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5, key="wi_monthly")
            total_wi = st.number_input("Total Charges ($)", value=float(tenure_wi * monthly_wi), min_value=0.0, max_value=9000.0, key="wi_total")
        with c2:
            contract_wi = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="wi_contract")
            internet_wi = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key="wi_internet")
            payment_wi = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="wi_payment",
            )
        with c3:
            senior_wi = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True, key="wi_senior")
            paperless_wi = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True, key="wi_paperless")
            partner_wi = st.radio("Partner", ["Yes", "No"], horizontal=True, key="wi_partner")

        if st.button("Predict churn probability", key="wi_predict"):
            models, _, manifest = load_models()
            feature_order = manifest[selected_model]["feature_order"]
            raw_row = {
                "tenure": tenure_wi,
                "MonthlyCharges": monthly_wi,
                "TotalCharges": total_wi,
                "SeniorCitizen": 1 if senior_wi == "Yes" else 0,
                "Partner": 1 if partner_wi == "Yes" else 0,
                "PaperlessBilling": 1 if paperless_wi == "Yes" else 0,
                "Contract_Month_to_month": 1 if contract_wi == "Month-to-month" else 0,
                "Contract_One_year": 1 if contract_wi == "One year" else 0,
                "Contract_Two_year": 1 if contract_wi == "Two year" else 0,
                "InternetService_Fiber_optic": 1 if internet_wi == "Fiber optic" else 0,
                "InternetService_DSL": 1 if internet_wi == "DSL" else 0,
                "InternetService_No": 1 if internet_wi == "No" else 0,
                "PaymentMethod_Electronic_check": 1 if payment_wi == "Electronic check" else 0,
                "PaymentMethod_Mailed_check": 1 if payment_wi == "Mailed check" else 0,
                "PaymentMethod_Bank_transfer_automatic": 1 if payment_wi == "Bank transfer (automatic)" else 0,
                "PaymentMethod_Credit_card_automatic": 1 if payment_wi == "Credit card (automatic)" else 0,
            }
            row_df = pd.DataFrame([{feature: raw_row.get(feature, 0) for feature in feature_order}])[feature_order].astype(float)
            p_wi = models[selected_model].predict_proba(row_df)[0, 1]
            color = "#EF4444" if p_wi >= threshold else "#22C55E"
            label = "Churn" if p_wi >= threshold else "Stay"
            st.markdown(f"<h3 style='color:{color};'>Predicted: {label} ({p_wi:.1%})</h3>", unsafe_allow_html=True)
            st.caption(f"Using {selected_model} at threshold {threshold:.3f}.")


def _section9_retention_simulator(selected_model: str, threshold: float) -> None:
    with st.expander("Retention Impact Simulator", expanded=False):
        st.caption("Re-score the dataset after applying a simple retention policy scenario.")

        sim_policy = st.selectbox(
            "Retention policy scenario",
            [
                "Month-to-month -> One year contract",
                "Manual payment -> Auto-pay",
                "No online security -> Add online security",
            ],
            key="sim_policy",
        )
        sim_scope = st.radio(
            "Apply to",
            ["All customers matching the condition", "Top-N highest-risk only"],
            horizontal=True,
            key="sim_scope",
        )
        sim_n = None
        if sim_scope == "Top-N highest-risk only":
            sim_n = st.slider("N", 50, 1000, 200, step=50, key="sim_n")

        if st.button("Run simulation", key="run_sim"):
            models, _, manifest = load_models()
            df_sim = pd.read_parquet(CURATED_DATA_PATH)
            feature_order = manifest[selected_model]["feature_order"]
            X_sim = df_sim.drop(columns=["Churn"])[feature_order].astype(float)

            p_base = models[selected_model].predict_proba(X_sim)[:, 1]
            X_scenario = X_sim.copy()

            if sim_policy == "Month-to-month -> One year contract":
                mask = X_scenario["Contract_Month_to_month"] == 1
                X_scenario.loc[mask, "Contract_Month_to_month"] = 0
                X_scenario.loc[mask, "Contract_One_year"] = 1
            elif sim_policy == "Manual payment -> Auto-pay":
                manual_cols = ["PaymentMethod_Electronic_check", "PaymentMethod_Mailed_check"]
                mask = X_scenario[manual_cols].sum(axis=1) > 0
                X_scenario.loc[mask, manual_cols] = 0
                if "PaymentMethod_Bank_transfer_automatic" in X_scenario.columns:
                    X_scenario.loc[mask, "PaymentMethod_Bank_transfer_automatic"] = 1
            elif sim_policy == "No online security -> Add online security" and "OnlineSecurity" in X_scenario.columns:
                mask = X_scenario["OnlineSecurity"] == 0
                X_scenario.loc[mask, "OnlineSecurity"] = 1

            if sim_scope == "Top-N highest-risk only" and sim_n:
                top_n_idx = pd.Series(p_base, index=X_scenario.index).nlargest(sim_n).index
                p_modified = p_base.copy()
                p_scenario_sub = models[selected_model].predict_proba(X_scenario.loc[top_n_idx])[:, 1]
                for position, idx in enumerate(top_n_idx):
                    p_modified[idx] = p_scenario_sub[position]
            else:
                p_modified = models[selected_model].predict_proba(X_scenario)[:, 1]

            base_rate = (p_base >= threshold).mean()
            scenario_rate = (p_modified >= threshold).mean()
            delta_pp = (scenario_rate - base_rate) * 100
            affected = int((p_base != p_modified).sum())

            col_b, col_s, col_d = st.columns(3)
            col_b.metric("Current predicted churn rate", f"{base_rate:.1%}")
            col_s.metric("Scenario predicted churn rate", f"{scenario_rate:.1%}", delta=f"{delta_pp:+.1f}pp", delta_color="inverse")
            col_d.metric("Customers affected", affected)
            st.caption(f"Scenario: **{sim_policy}** re-scored with {selected_model} at threshold {threshold:.3f}.")


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
    selection, df_display, X_full = _section5_prediction_table(selected_model, threshold, df_scored, X_full)
    _section6_local_explainability(selected_model, threshold, selection, df_display, X_full)
    st.divider()
    _section7_feature_impact_analysis(selected_model)
    st.divider()
    _section8_whatif(selected_model, threshold)
    _section9_retention_simulator(selected_model, threshold)
