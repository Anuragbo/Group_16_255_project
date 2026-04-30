from pathlib import Path

import streamlit as st

from app.components import anomaly_detection, customer_segments, eda_olap, predictive_modeling

ROOT = Path(__file__).resolve().parents[1]


def _sidebar() -> dict:
    st.sidebar.title("ChurnCube")
    st.sidebar.markdown("Telco Customer Churn Analysis")
    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    contract_types = st.sidebar.multiselect(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        default=[],
    )
    internet_services = st.sidebar.multiselect(
        "Internet Service",
        ["Fiber optic", "DSL", "No"],
        default=[],
    )
    senior = st.sidebar.radio("Senior Citizen", ["All", "Senior", "Non-Senior"])
    st.sidebar.divider()
    st.sidebar.caption(
        "Filters apply to EDA, customer segments, and anomaly views. "
        "Predictive modeling always uses the full dataset."
    )
    st.sidebar.caption(f"Workspace: `{ROOT.name}`")
    return {
        "contract_types": contract_types,
        "internet_services": internet_services,
        "senior": senior,
    }


def main() -> None:
    st.set_page_config(page_title="ChurnCube", page_icon="📊", layout="wide")
    st.info(
        "**Convergence banner:** the highest-risk churn archetype in this project is "
        "customers with month-to-month contracts, fiber optic internet, electronic check "
        "payments, and short tenure. The dashboard ties the EDA, segmentation, anomaly, "
        "and predictive views back to that same pattern."
    )

    filters = _sidebar()
    tabs = st.tabs(
        ["EDA & OLAP", "Customer Segments", "Anomaly Detection", "Predictive Modeling"]
    )
    with tabs[0]:
        eda_olap.render(filters)
    with tabs[1]:
        customer_segments.render(filters)
    with tabs[2]:
        anomaly_detection.render(filters)
    with tabs[3]:
        predictive_modeling.render()


if __name__ == "__main__":
    main()
