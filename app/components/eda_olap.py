from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT / "data" / "raw.parquet"

BASELINES = {
    "n": 7043,
    "churn_rate": 26.54,
    "monthly_charges": 64.76,
    "tenure": 32.37,
}

ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


@st.cache_data
def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if filters["contract_types"]:
        df = df[df["Contract"].isin(filters["contract_types"])]
    if filters["internet_services"]:
        df = df[df["InternetService"].isin(filters["internet_services"])]
    if filters["senior"] == "Senior":
        df = df[df["SeniorCitizen"] == 1]
    elif filters["senior"] == "Non-Senior":
        df = df[df["SeniorCitizen"] == 0]
    return df


def _kpi_cards(df: pd.DataFrame) -> None:
    n = len(df)
    churn_rate = (df["Churn"] == "Yes").mean() * 100 if n else 0.0
    avg_monthly = df["MonthlyCharges"].mean() if n else 0.0
    avg_tenure = df["tenure"].mean() if n else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Customers",
        f"{n:,}",
        delta=f"{n - BASELINES['n']:+,} vs {BASELINES['n']:,}" if n != BASELINES["n"] else None,
    )
    c2.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%",
        delta=f"{churn_rate - BASELINES['churn_rate']:+.2f}pp vs {BASELINES['churn_rate']}%",
        delta_color="inverse",
    )
    c3.metric(
        "Avg Monthly Charges",
        f"${avg_monthly:.2f}",
        delta=f"{avg_monthly - BASELINES['monthly_charges']:+.2f} vs ${BASELINES['monthly_charges']}",
    )
    c4.metric(
        "Avg Tenure",
        f"{avg_tenure:.1f} mo",
        delta=f"{avg_tenure - BASELINES['tenure']:+.1f} vs {BASELINES['tenure']} mo",
    )


def _view_churn_overview(df: pd.DataFrame) -> None:
    col_left, col_right = st.columns([2, 3])

    with col_left:
        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Churn", "Count"]
        fig = px.pie(
            counts,
            names="Churn",
            values="Count",
            color="Churn",
            color_discrete_map={"Yes": "#EF4444", "No": "#22C55E"},
            hole=0.55,
            title="Overall Churn Distribution",
        )
        pct = (df["Churn"] == "Yes").mean() * 100 if len(df) else 0.0
        fig.add_annotation(
            text=f"<b>{pct:.1f}%</b><br>Churn",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(height=350, margin=dict(t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        overall_churn = (df["Churn"] == "Yes").mean() * 100 if len(df) else 0.0
        segments = {}
        for gender_val, label in [("Male", "Male"), ("Female", "Female")]:
            sub = df[df["gender"] == gender_val]
            if len(sub):
                segments[label] = (sub["Churn"] == "Yes").mean() * 100
        for senior_val, label in [(1, "Senior"), (0, "Non-Senior")]:
            sub = df[df["SeniorCitizen"] == senior_val]
            if len(sub):
                segments[label] = (sub["Churn"] == "Yes").mean() * 100
        for partner_val, label in [("Yes", "Has Partner"), ("No", "No Partner")]:
            sub = df[df["Partner"] == partner_val]
            if len(sub):
                segments[label] = (sub["Churn"] == "Yes").mean() * 100
        for dep_val, label in [("Yes", "Has Dependents"), ("No", "No Dependents")]:
            sub = df[df["Dependents"] == dep_val]
            if len(sub):
                segments[label] = (sub["Churn"] == "Yes").mean() * 100

        seg_df = pd.DataFrame(
            {"Segment": list(segments.keys()), "Churn Rate (%)": list(segments.values())}
        ).sort_values("Churn Rate (%)")

        fig2 = px.bar(
            seg_df,
            x="Churn Rate (%)",
            y="Segment",
            orientation="h",
            color="Churn Rate (%)",
            color_continuous_scale="Reds",
            title="Churn Rate by Demographic Segment",
        )
        fig2.add_vline(
            x=overall_churn,
            line_dash="dash",
            line_color="grey",
            annotation_text=f"Overall {overall_churn:.1f}%",
            annotation_position="top right",
        )
        fig2.update_layout(height=350, coloraxis_showscale=False, margin=dict(t=50, b=0))
        st.plotly_chart(fig2, use_container_width=True)


def _view_numeric_distributions(df: pd.DataFrame) -> None:
    cols = st.columns(3)
    features = [
        ("tenure", "Tenure (months)"),
        ("MonthlyCharges", "Monthly Charges ($)"),
        ("TotalCharges", "Total Charges ($)"),
    ]

    for col_st, (feat, title) in zip(cols, features):
        with col_st:
            df_no = df[df["Churn"] == "No"][feat].dropna()
            df_yes = df[df["Churn"] == "Yes"][feat].dropna()

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.65, 0.35],
                vertical_spacing=0.08,
            )
            fig.add_trace(
                go.Histogram(
                    x=df_no,
                    name="No Churn",
                    opacity=0.6,
                    marker_color="#3B82F6",
                    nbinsx=30,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Histogram(
                    x=df_yes,
                    name="Churn",
                    opacity=0.6,
                    marker_color="#EF4444",
                    nbinsx=30,
                ),
                row=1,
                col=1,
            )
            fig.update_layout(barmode="overlay")

            fig.add_trace(
                go.Box(x=df_no, name="No Churn", marker_color="#3B82F6", boxmean=True, orientation="h"),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Box(x=df_yes, name="Churn", marker_color="#EF4444", boxmean=True, orientation="h"),
                row=2,
                col=1,
            )

            fig.update_layout(
                title=title,
                height=350,
                showlegend=False,
                margin=dict(t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Distribution Statistics"):
        stat_rows = []
        for feat, title in features:
            for churn_val, label in [("No", "No Churn"), ("Yes", "Churn")]:
                s = df[df["Churn"] == churn_val][feat].dropna()
                stat_rows.append(
                    {
                        "Feature": title,
                        "Churn": label,
                        "Mean": round(s.mean(), 2),
                        "Median": round(s.median(), 2),
                        "Std": round(s.std(), 2),
                        "Min": round(s.min(), 2),
                        "Max": round(s.max(), 2),
                    }
                )
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)


def _view_service_contract(df: pd.DataFrame) -> None:
    st.subheader("Add-on Service Adoption by Churn Status")
    adoption = df.groupby("Churn")[ADDON_COLS].apply(lambda g: (g == "Yes").mean() * 100)
    fig_heat = px.imshow(
        adoption,
        color_continuous_scale="RdYlGn",
        text_auto=".1f",
        title="Add-on Adoption Rate (%) by Churn Status",
        labels=dict(x="Add-on Service", y="Churn", color="Adoption %"),
    )
    fig_heat.update_layout(height=250, margin=dict(t=50, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

    stab_a, stab_b, stab_c = st.tabs(["Contract & Billing", "Internet & Phone", "Demographics Detail"])

    def churn_rate_bar(group_col: str, orientation: str = "v", title: str | None = None):
        tmp = (
            df.groupby(group_col)["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate (%)")
        )
        if orientation == "h":
            fig = px.bar(
                tmp,
                x="Churn Rate (%)",
                y=group_col,
                orientation="h",
                color="Churn Rate (%)",
                color_continuous_scale="Reds",
                title=title or f"Churn by {group_col}",
            )
        else:
            fig = px.bar(
                tmp,
                x=group_col,
                y="Churn Rate (%)",
                color="Churn Rate (%)",
                color_continuous_scale="Reds",
                title=title or f"Churn by {group_col}",
            )
        fig.update_layout(height=300, coloraxis_showscale=False, margin=dict(t=50, b=0))
        return fig

    with stab_a:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                churn_rate_bar("Contract", title="Churn by Contract Type"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                churn_rate_bar("PaymentMethod", orientation="h", title="Churn by Payment Method"),
                use_container_width=True,
            )

    with stab_b:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                churn_rate_bar("InternetService", title="Churn by Internet Service"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                churn_rate_bar("MultipleLines", title="Churn by Multiple Lines"),
                use_container_width=True,
            )

    with stab_c:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                churn_rate_bar("PaperlessBilling", title="Churn by Paperless Billing"),
                use_container_width=True,
            )
        with c2:
            df2 = df.copy()
            df2["ServiceCount"] = df2[ADDON_COLS].apply(lambda col: col == "Yes").sum(axis=1)
            tmp = (
                df2.groupby("ServiceCount")["Churn"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .reset_index(name="Churn Rate (%)")
            )
            fig_sc = px.bar(
                tmp,
                x="ServiceCount",
                y="Churn Rate (%)",
                color="Churn Rate (%)",
                color_continuous_scale="Reds",
                title="Churn by Service Count (0-6 add-ons)",
            )
            fig_sc.update_layout(height=300, coloraxis_showscale=False, margin=dict(t=50, b=0))
            st.plotly_chart(fig_sc, use_container_width=True)


@st.cache_data
def compute_olap_series(focus_col: str | None, focus_val: str | None) -> dict:
    df = pd.read_parquet(RAW_DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if focus_col and focus_val:
        df = df[df[focus_col] == focus_val]

    df["TenureCohort"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
        include_lowest=True,
    )
    df["ChargesCohort"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 90, float("inf")],
        labels=["$0-35", "$35-65", "$65-90", "$90+"],
    )
    df["PayType"] = df["PaymentMethod"].apply(
        lambda value: "Auto-pay" if "automatic" in value.lower() else "Manual"
    )

    def churn_rate(group_col: str) -> pd.DataFrame:
        return (
            df.groupby(group_col, observed=True)["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="ChurnRate")
        )

    cross_tab = (
        df.groupby(["Contract", "InternetService"])["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .unstack("InternetService")
        .round(1)
    )

    return {
        "contract": churn_rate("Contract"),
        "tenure": churn_rate("TenureCohort"),
        "payment": churn_rate("PaymentMethod"),
        "internet": churn_rate("InternetService"),
        "charges": churn_rate("ChargesCohort"),
        "pay_type": churn_rate("PayType"),
        "cross_tab": cross_tab,
        "n": len(df),
    }


def _view_olap() -> None:
    st.caption("Showing all 7,043 customers. Sidebar filters do not apply here.")

    focus_options = {"None (full dataset)": (None, None)}
    for contract in ["Month-to-month", "One year", "Two year"]:
        focus_options[f"Contract: {contract}"] = ("Contract", contract)
    for internet in ["Fiber optic", "DSL", "No"]:
        focus_options[f"Internet: {internet}"] = ("InternetService", internet)

    focus_label = st.selectbox("Focus Dimension", list(focus_options.keys()))
    focus_col, focus_val = focus_options[focus_label]
    series = compute_olap_series(focus_col, focus_val)
    st.caption(
        f"Showing {series['n']:,} customers"
        + (f" filtered to {focus_val}" if focus_val else " (full dataset)")
    )

    def make_bar(
        data: pd.DataFrame,
        x_col: str,
        y_col: str = "ChurnRate",
        orientation: str = "v",
        annotation: str | None = None,
        title: str = "",
    ):
        if orientation == "h":
            fig = px.bar(
                data,
                x=y_col,
                y=x_col,
                orientation="h",
                color=y_col,
                color_continuous_scale="Reds",
                title=title,
            )
        else:
            fig = px.bar(
                data,
                x=x_col,
                y=y_col,
                color=y_col,
                color_continuous_scale="Reds",
                title=title,
            )
        fig.update_layout(height=280, coloraxis_showscale=False, margin=dict(t=50, b=0))
        if annotation:
            fig.add_annotation(
                text=annotation,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.0,
                showarrow=False,
                font=dict(size=10, color="grey"),
            )
        return fig

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(
            make_bar(
                series["contract"],
                "Contract",
                annotation="Full dataset: month-to-month is the highest-risk contract" if not focus_val else None,
                title="Churn by Contract Type",
            ),
            use_container_width=True,
        )
    with r1c2:
        st.plotly_chart(
            make_bar(
                series["tenure"],
                "TenureCohort",
                annotation="0-12 months is the highest-risk tenure cohort" if not focus_val else None,
                title="Churn by Tenure Cohort",
            ),
            use_container_width=True,
        )

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(
            make_bar(
                series["payment"],
                "PaymentMethod",
                orientation="h",
                annotation="Electronic check customers churn the most in the full dataset" if not focus_val else None,
                title="Churn by Payment Method",
            ),
            use_container_width=True,
        )
    with r2c2:
        st.plotly_chart(
            make_bar(
                series["internet"],
                "InternetService",
                annotation="Fiber optic customers show the highest churn" if not focus_val else None,
                title="Churn by Internet Service",
            ),
            use_container_width=True,
        )

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(
            make_bar(
                series["charges"],
                "ChargesCohort",
                annotation="Mid-to-high monthly charge cohorts have the highest churn" if not focus_val else None,
                title="Churn by Monthly Charges Cohort",
            ),
            use_container_width=True,
        )
    with r3c2:
        st.plotly_chart(
            make_bar(
                series["pay_type"],
                "PayType",
                annotation="Manual payment customers churn more than auto-pay customers" if not focus_val else None,
                title="Auto-pay vs Manual Payment",
            ),
            use_container_width=True,
        )

    st.subheader("Cross-Dimensional: Contract x Internet Service")
    cross_tab = series["cross_tab"]
    fig_ht = px.imshow(
        cross_tab,
        color_continuous_scale="Reds",
        text_auto=".1f",
        title="Churn Rate (%) by Contract x Internet Service",
        labels=dict(x="Internet Service", y="Contract", color="Churn %"),
    )
    fig_ht.update_layout(height=300, margin=dict(t=50, b=20))
    st.plotly_chart(fig_ht, use_container_width=True)
    if not focus_val:
        st.caption(
            "The month-to-month and fiber optic cell is the highest-risk cohort in the full dataset."
        )
    else:
        peak = cross_tab.max().max()
        st.caption(f"Peak cell in the filtered view: **{peak:.1f}% churn**.")


def render(filters: dict | None = None) -> None:
    filters = filters or {"contract_types": [], "internet_services": [], "senior": "All"}
    raw_full = load_raw()
    df = _apply_filters(raw_full.copy(), filters)

    st.title("EDA & OLAP")
    st.caption(f"Showing **{len(df):,}** customers after filters.")

    _kpi_cards(df)

    view = st.radio(
        "View",
        ["Churn Overview", "Numeric Distributions", "Service & Contract", "OLAP Drill-Down"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    if view == "Churn Overview":
        _view_churn_overview(df)
    elif view == "Numeric Distributions":
        _view_numeric_distributions(df)
    elif view == "Service & Contract":
        _view_service_contract(df)
    else:
        _view_olap()
