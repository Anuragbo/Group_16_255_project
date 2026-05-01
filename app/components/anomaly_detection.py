from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
OUTLIER_OUTPUT_DIR = ROOT / "outputs" / "outlier_detection"
CURATED_DATA_PATH = ROOT / "data" / "curated.parquet"
CLUSTERED_DATA_PATH = ROOT / "data" / "clustered.parquet"

OUTLIER_MAP = {
    "Isolation Forest": {
        "score_col": "IsoForest_Anomaly_Score",
        "top_100_file": "top_100_isolation_forest.csv",
        "corr_val": -0.273,
        "corr_pval": "<0.001",
        "significant": True,
    },
    "Local Outlier Factor (LOF)": {
        "score_col": "LOF_Anomaly_Score",
        "top_100_file": "top_100_lof.csv",
        "corr_val": 0.008,
        "corr_pval": "0.483",
        "significant": False,
    },
}

CLUSTER_COLS_DISPLAY = {
    "K-Means": "Cluster_KMeans",
    "GMM": "Cluster_GMM",
    "HDBSCAN (PCA)": "Cluster_HDBSCAN_PCA",
    "HDBSCAN (Gower)": "Cluster_HDBSCAN_Gower",
}

SCORE_COLS_IN_CSV = {"Churn", "IsoForest_Anomaly_Score", "LOF_Anomaly_Score"}
CLUSTER_COLS_IN_CSV = set(CLUSTER_COLS_DISPLAY.values())
PCA_COLS_IN_CSV = {"PCA1", "PCA2"}


@st.cache_data
def load_all_scores() -> pd.DataFrame:
    return pd.read_csv(OUTLIER_OUTPUT_DIR / "all_scores.csv")


@st.cache_data
def load_top100(filename: str) -> pd.DataFrame:
    return pd.read_csv(OUTLIER_OUTPUT_DIR / filename)


@st.cache_data
def load_global_means() -> pd.Series:
    df = pd.read_parquet(CURATED_DATA_PATH)
    return df.drop(columns=["Churn"]).mean()


@st.cache_data
def load_clustered() -> pd.DataFrame:
    return pd.read_parquet(CLUSTERED_DATA_PATH)


@st.cache_data
def build_anomaly_churn_scatter(model_name: str) -> pd.DataFrame:
    from app.components.predictive_modeling import score_dataset

    scores_df = load_all_scores().reset_index().rename(columns={"index": "customer_index"})
    pred_df, _ = score_dataset(model_name)
    pred_df = pred_df.reset_index().rename(columns={"index": "customer_index"})
    return scores_df.merge(pred_df, on="customer_index", how="left")


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if filters["contract_types"]:
        contract_map = {
            "Month-to-month": "Contract_Month_to_month",
            "One year": "Contract_One_year",
            "Two year": "Contract_Two_year",
        }
        masks = [df[contract_map[c]] == 1 for c in filters["contract_types"] if contract_map.get(c) in df.columns]
        if masks:
            combined = masks[0]
            for mask in masks[1:]:
                combined = combined | mask
            df = df[combined]
    if filters["internet_services"]:
        internet_map = {
            "Fiber optic": "InternetService_Fiber_optic",
            "DSL": "InternetService_DSL",
            "No": "InternetService_No",
        }
        masks = [df[internet_map[s]] == 1 for s in filters["internet_services"] if internet_map.get(s) in df.columns]
        if masks:
            combined = masks[0]
            for mask in masks[1:]:
                combined = combined | mask
            df = df[combined]
    if filters["senior"] == "Senior":
        df = df[df["SeniorCitizen"] == 1]
    elif filters["senior"] == "Non-Senior":
        df = df[df["SeniorCitizen"] == 0]
    return df


def _get_filtered_score_frame(filters: dict) -> pd.DataFrame:
    clustered = load_clustered().reset_index().rename(columns={"index": "customer_index"})
    filtered_clustered = _apply_filters(clustered.copy(), filters)
    all_scores = load_all_scores().reset_index().rename(columns={"index": "customer_index"})
    return filtered_clustered[["customer_index"]].merge(all_scores, on="customer_index", how="left")


def _get_filtered_top100(top_100_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    return _apply_filters(top_100_df.copy(), filters)


def _get_x_feature_cols(top_100_df: pd.DataFrame) -> list[str]:
    excluded = SCORE_COLS_IN_CSV | CLUSTER_COLS_IN_CSV | PCA_COLS_IN_CSV | {"customer_index"}
    return [col for col in top_100_df.columns if col not in excluded]


def _section1_kpis(algo_info: dict, all_scores: pd.DataFrame, top_100_df: pd.DataFrame) -> None:
    churn_in_scope = top_100_df["Churn"].mean() if len(top_100_df) else 0.0
    global_churn = 26.54 / 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered Customers", f"{len(all_scores):,}")
    c2.metric("Visible Top Anomalies", f"{len(top_100_df):,}", help="Subset of the global top-100 after sidebar filters.")
    c3.metric(
        "Churn Rate in Visible Anomalies",
        f"{churn_in_scope:.1%}",
        delta=f"{(churn_in_scope - global_churn) * 100:+.1f}pp vs {global_churn:.1%} global",
        delta_color="off",
    )
    badge_text = (
        f"Significant (p {algo_info['corr_pval']})"
        if algo_info["significant"]
        else f"Not significant (p = {algo_info['corr_pval']})"
    )
    c4.metric("Point-Biserial Correlation", f"r = {algo_info['corr_val']:+.3f}", delta=badge_text, delta_color="off")


def _section2_distributions(algo_info: dict, all_scores: pd.DataFrame, top_100_df: pd.DataFrame) -> None:
    score_col = algo_info["score_col"]
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Score Distribution by Churn")
        plot_df = all_scores[[score_col, "Churn"]].copy()
        plot_df["Churn Status"] = plot_df["Churn"].map({0: "No Churn", 1: "Churned"})
        fig = px.violin(
            plot_df,
            y=score_col,
            x="Churn Status",
            color="Churn Status",
            box=True,
            points=False,
            color_discrete_map={"No Churn": "#3B82F6", "Churned": "#EF4444"},
            title=f"{score_col} by Churn Status",
        )
        fig.update_layout(height=380, showlegend=False, margin=dict(t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
        if score_col == "IsoForest_Anomaly_Score":
            st.caption(
                "Isolation Forest preserves the project’s counter-intuitive finding: churners often look more typical than long-tenure non-churners."
            )

    with col_right:
        st.subheader("Cluster Overlap")
        cluster_display = st.selectbox("Clustering method", list(CLUSTER_COLS_DISPLAY.keys()), key=f"cluster_overlap_{score_col}")
        cluster_col = CLUSTER_COLS_DISPLAY[cluster_display]
        overlap_df = (
            top_100_df.groupby(cluster_col)
            .size()
            .reset_index(name="Outlier_Count")
            .sort_values("Outlier_Count", ascending=False)
        )
        fig_bar = px.bar(
            overlap_df,
            x=cluster_col,
            y="Outlier_Count",
            color="Outlier_Count",
            color_continuous_scale="Reds",
            title=f"Visible Top-100 Anomalies by {cluster_display} Cluster",
            labels={cluster_col: "Cluster ID", "Outlier_Count": "Anomaly Count"},
        )
        fig_bar.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=50, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)


def _section3_explorer(algo_info: dict, top_100_df: pd.DataFrame, global_means: pd.Series):
    st.subheader("Anomaly Explorer")
    score_col = algo_info["score_col"]
    x_feature_cols = _get_x_feature_cols(top_100_df)

    if top_100_df.empty:
        st.info("No anomalies remain after the current sidebar filters.")
        return None

    display_cols = [
        "customer_index",
        score_col,
        "Churn",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract_Month_to_month",
        "Contract_One_year",
        "Contract_Two_year",
    ]
    display_cols = [col for col in display_cols if col in top_100_df.columns]

    selection = st.dataframe(
        top_100_df[display_cols].reset_index(drop=True),
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
        height=280,
        column_config={
            score_col: st.column_config.NumberColumn("Anomaly Score", format="%.4f"),
            "Churn": st.column_config.NumberColumn("Churn (1=Yes)"),
        },
    )

    if selection.selection.rows:
        idx = selection.selection.rows[0]
        row = top_100_df.iloc[idx][x_feature_cols]
        global_mean = global_means.reindex(x_feature_cols)
        deviation = ((row - global_mean) / global_mean.replace(0, 1e-9)) * 100

        colors = [
            "#EF4444" if abs(value) > 50 else "#F97316" if abs(value) > 20 else "#9CA3AF"
            for value in deviation.values
        ]
        fig = go.Figure(
            go.Bar(
                x=deviation.values,
                y=deviation.index.tolist(),
                orientation="h",
                marker_color=colors,
            )
        )
        customer_index = int(top_100_df.iloc[idx]["customer_index"]) if "customer_index" in top_100_df.columns else idx
        fig.update_layout(
            title=f"Why is Customer #{customer_index} anomalous? (% deviation from global mean)",
            height=max(300, len(x_feature_cols) * 18),
            margin=dict(t=50, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Red bars show the strongest deviations from the global customer profile. Orange bars are secondary anomaly drivers."
        )
        selected_customer = int(top_100_df.iloc[idx]["customer_index"]) if "customer_index" in top_100_df.columns else idx
        if st.button("View this anomaly in Predictive Modeling", key=f"drill_anomaly_{selected_customer}"):
            st.session_state["prediction_filter"] = [selected_customer]
            st.session_state["prediction_filter_label"] = f"Selected anomaly customer #{selected_customer}"
            st.success("Predictive Modeling has been primed with this customer.")
    return selection


def _section4_scatter(algo_info: dict, filtered_indices: pd.Series) -> None:
    st.subheader("Anomaly Score vs Predicted Churn Probability")
    score_col = algo_info["score_col"]

    scatter_model = st.selectbox(
        "Churn model for scatter",
        ["XGBoost", "RandomForest", "LogisticRegression"],
        key="anomaly_scatter_model",
    )

    merged = build_anomaly_churn_scatter(scatter_model)
    merged = merged[merged["customer_index"].isin(filtered_indices)]

    fig = px.scatter(
        merged,
        x=score_col,
        y="p_churn",
        color=merged["Churn"].map({0: "No Churn", 1: "Churned"}),
        color_discrete_map={"No Churn": "#22C55E", "Churned": "#EF4444"},
        opacity=0.45,
        labels={score_col: "Anomaly Score", "p_churn": "Predicted Churn Probability"},
        title=f"{score_col} vs churn probability using {scatter_model}",
    )
    fig.add_vline(x=merged[score_col].median(), line_dash="dot", line_color="grey")
    fig.add_hline(y=merged["p_churn"].median(), line_dash="dot", line_color="grey")
    fig.update_layout(height=500, margin=dict(t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Customers in the top-right quadrant combine unusual behavior with high churn probability, making them strong intervention candidates."
    )


def render(filters: dict | None = None) -> None:
    filters = filters or {"contract_types": [], "internet_services": [], "senior": "All"}
    st.title("Anomaly Detection")

    filtered_scores = _get_filtered_score_frame(filters)
    global_means = load_global_means()

    selected_algo = st.selectbox("Algorithm", list(OUTLIER_MAP.keys()))
    algo_info = OUTLIER_MAP[selected_algo]
    top_100_df = load_top100(algo_info["top_100_file"]).reset_index().rename(columns={"index": "customer_index"})
    filtered_top100 = _get_filtered_top100(top_100_df, filters)

    st.caption(f"Showing **{len(filtered_scores):,}** customers after filters.")
    st.divider()
    _section1_kpis(algo_info, filtered_scores, filtered_top100)
    st.divider()
    _section2_distributions(algo_info, filtered_scores, filtered_top100)
    st.divider()
    _section3_explorer(algo_info, filtered_top100, global_means)
    if not filtered_top100.empty and st.button("View visible anomaly cohort in Predictive Modeling", key=f"drill_all_{selected_algo}"):
        st.session_state["prediction_filter"] = filtered_top100["customer_index"].tolist()
        st.session_state["prediction_filter_label"] = f"{selected_algo} visible anomaly cohort ({len(filtered_top100)} customers)"
        st.success("Predictive Modeling has been primed with the current anomaly cohort.")
    st.divider()
    _section4_scatter(algo_info, filtered_scores["customer_index"])
