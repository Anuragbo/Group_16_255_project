from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
CLUSTERED_DATA_PATH = ROOT / "data" / "clustered.parquet"
SEGMENTATION_OUTPUT_DIR = ROOT / "outputs" / "segmentation"

ALGO_MAP = {
    "K-Means": ("Cluster_KMeans", "kmeans_profile.csv"),
    "GMM": ("Cluster_GMM", "gmm_profile.csv"),
    "HDBSCAN (PCA)": ("Cluster_HDBSCAN_PCA", "hdbscan_pca_profile.csv"),
    "HDBSCAN (Gower)": ("Cluster_HDBSCAN_Gower", "hdbscan_gower_profile.csv"),
}


@st.cache_data
def load_clustered() -> pd.DataFrame:
    return pd.read_parquet(CLUSTERED_DATA_PATH)


@st.cache_data
def load_profile(filename: str) -> pd.DataFrame:
    return pd.read_csv(SEGMENTATION_OUTPUT_DIR / filename, index_col=0)


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


def _cluster_kpis(profile: pd.DataFrame) -> None:
    valid_profile = profile[profile.index != -1]
    n_clusters = len(valid_profile)

    c1, c2, c3 = st.columns(3)
    c1.metric("Number of Clusters", n_clusters, help="Noise cluster -1 excluded where present.")
    if len(valid_profile) == 0:
        c2.metric("Highest-Churn Cluster", "N/A")
        c3.metric("Lowest-Churn Cluster", "N/A")
        return

    high_churn_id = valid_profile["Churn"].idxmax()
    high_churn_rate = valid_profile.loc[high_churn_id, "Churn"]
    low_churn_id = valid_profile["Churn"].idxmin()
    low_churn_rate = valid_profile.loc[low_churn_id, "Churn"]
    c2.metric("Highest-Churn Cluster", f"Cluster {high_churn_id}: {high_churn_rate:.1%}")
    c3.metric("Lowest-Churn Cluster", f"Cluster {low_churn_id}: {low_churn_rate:.1%}")


def _global_view(clustered_df: pd.DataFrame, cluster_col: str, profile: pd.DataFrame, algo_name: str) -> None:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("PCA Scatter")
        plot_df = clustered_df.copy()
        plot_df[cluster_col] = plot_df[cluster_col].astype(str)
        color_map = {"-1": "grey"} if "-1" in plot_df[cluster_col].values else None

        fig = px.scatter(
            plot_df,
            x="PCA1",
            y="PCA2",
            color=cluster_col,
            color_discrete_map=color_map,
            hover_data={
                "tenure": True,
                "MonthlyCharges": True,
                "Churn": True,
                "PCA1": False,
                "PCA2": False,
            },
            title=f"{algo_name} Cluster Assignments",
            labels={cluster_col: "Cluster"},
            opacity=0.65,
        )
        fig.update_layout(height=420, margin=dict(t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

        if algo_name == "HDBSCAN (Gower)":
            st.caption(
                "HDBSCAN (Gower) uses the original mixed feature space. The PCA view is an approximate projection for comparison only."
            )

    with col_right:
        st.subheader("Cluster Profile Summary")
        display_profile = profile.rename(columns={"Churn": "Churn Rate"})
        st.dataframe(
            display_profile.style.format(
                {"Churn Rate": "{:.1%}", "tenure": "{:.1f}", "MonthlyCharges": "{:.2f}"}
            ).background_gradient(cmap="Reds", subset=["Churn Rate"]),
            use_container_width=True,
            height=420,
        )


def _deep_dive(clustered_df: pd.DataFrame, cluster_col: str, profile: pd.DataFrame, algo_name: str) -> None:
    st.subheader("Cluster Deep Dive")

    valid_ids = sorted([cluster_id for cluster_id in profile.index if cluster_id != -1])
    if not valid_ids:
        st.info("No valid clusters available to inspect.")
        return

    selected_cluster_id = st.selectbox("Select Cluster", valid_ids, key=f"cluster_sel_{algo_name}")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Radar: Cluster vs Global")
        radar_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        available_features = [feature for feature in radar_features if feature in clustered_df.columns]

        cluster_mask = clustered_df[cluster_col] == selected_cluster_id
        cluster_means = clustered_df.loc[cluster_mask, available_features].mean()
        global_means = clustered_df[available_features].mean()

        norm_cluster = cluster_means.copy()
        norm_global = global_means.copy()
        for feature in available_features:
            feat_min = clustered_df[feature].min()
            feat_max = clustered_df[feature].max()
            rng = feat_max - feat_min if feat_max != feat_min else 1e-9
            norm_cluster[feature] = (cluster_means[feature] - feat_min) / rng
            norm_global[feature] = (global_means[feature] - feat_min) / rng

        radar_df = pd.DataFrame(
            {
                "Feature": available_features * 2,
                "Value": list(norm_cluster.values) + list(norm_global.values),
                "Group": [f"Cluster {selected_cluster_id}"] * len(available_features)
                + ["Dataset Average"] * len(available_features),
            }
        )
        fig_radar = px.line_polar(
            radar_df,
            r="Value",
            theta="Feature",
            color="Group",
            line_close=True,
            title=f"Cluster {selected_cluster_id} vs Global (normalized 0-1)",
        )
        fig_radar.update_traces(fill="toself", opacity=0.5)
        fig_radar.update_layout(height=350)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r:
        st.subheader("Contract Distribution")
        contract_cols = [col for col in profile.columns if "Contract_" in col]
        if contract_cols:
            cluster_row = profile.loc[selected_cluster_id, contract_cols]
            global_row = clustered_df[contract_cols].mean()

            bar_data = []
            for col in contract_cols:
                label = col.replace("Contract_", "").replace("_", " ")
                bar_data.append({"Contract": label, "Value": cluster_row[col], "Group": f"Cluster {selected_cluster_id}"})
                bar_data.append({"Contract": label, "Value": global_row[col], "Group": "Dataset Average"})
            bar_df = pd.DataFrame(bar_data)
            fig_bar = px.bar(
                bar_df,
                x="Contract",
                y="Value",
                color="Group",
                barmode="group",
                title="Contract Type Proportions",
                labels={"Value": "Proportion"},
            )
            fig_bar.update_layout(height=350, margin=dict(t=50, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Contract proportion columns are not available in the profile export.")

    if algo_name == "HDBSCAN (Gower)":
        st.caption("Highest churn separation appears in the HDBSCAN (Gower) segmentation.")
    elif algo_name == "GMM":
        st.caption("GMM surfaces a large month-to-month cluster with elevated churn compared with the dataset baseline.")
    elif algo_name == "K-Means":
        st.caption("K-Means mostly separates by tenure and spend, with weaker churn separation than HDBSCAN.")


def render(filters: dict | None = None) -> None:
    filters = filters or {"contract_types": [], "internet_services": [], "senior": "All"}

    st.title("Customer Segments")
    clustered_df = load_clustered()
    filtered_df = _apply_filters(clustered_df.copy(), filters)

    algo_name = st.selectbox("Clustering Algorithm", list(ALGO_MAP.keys()))
    cluster_col, profile_file = ALGO_MAP[algo_name]
    profile = load_profile(profile_file)

    st.caption(f"Showing **{len(filtered_df):,}** customers after filters.")
    _cluster_kpis(profile)
    st.divider()
    _global_view(filtered_df, cluster_col, profile, algo_name)
    st.divider()
    _deep_dive(filtered_df, cluster_col, profile, algo_name)
    st.divider()
    st.subheader("Silhouette Comparison")
    st.image(str(SEGMENTATION_OUTPUT_DIR / "silhouette_scores_bar.png"), use_container_width=True)
