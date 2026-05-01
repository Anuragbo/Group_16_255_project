# ChurnCube: Churn Prediction, Segmentation, Anomaly Detection, and Dashboarding

**CMPE 255: Data Mining** · San José State University · Spring 2026

---

## Team

| Name | SJSU ID | Responsibility |
|------|---------|----------------|
| Anurag Bodapathy | 019094218 | Initial setup, baseline Logistic Regression model, dashboard integration |
| Dhruv Verma | 018561309 | ETL pipeline, OLAP analytics, data layers |
| Shubham Baid | 018221333 | Ensemble models (RF + XGBoost), threshold tuning, SHAP interpretability |
| Siddarth Vuppunahalli | 019157203 | Customer segmentation (KMeans / GMM / HDBSCAN), outlier detection |

---

## Project Overview

**ChurnCube** is a reproducible end-to-end data mining pipeline applied to the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM/Kaggle, *n* = 7,043 customers, 21 features). The pipeline addresses three complementary analytical tasks:

1. **Supervised Churn Prediction** — Logistic Regression, Random Forest, and XGBoost trained with class-imbalance correction, 5-fold cross-validation, and $F_2$-score threshold tuning.
2. **Customer Segmentation** — K-Means, Gaussian Mixture Models (GMM), and HDBSCAN (Euclidean and Gower distance) with silhouette-based cluster selection and churn-rate segment profiling.
3. **Anomaly Detection** — Isolation Forest and Local Outlier Factor for surfacing anomalous customer profiles.
4. **Interactive Dashboard** — Streamlit + Plotly experience that connects EDA/OLAP, segmentation, anomaly detection, and predictive modeling with shared filters and drill-through workflows.

Key dashboard finding: the same high-risk churn archetype appears across all four analytical layers, especially customers with month-to-month contracts, fiber optic internet, electronic check payments, and short tenure.

---

## Repository Structure

```
Group_16_255_project/
│
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw source data (IBM/Kaggle)
│
├── data/
│   ├── raw.parquet                              # 7,043 × 21 — TotalCharges fixed
│   ├── staging.parquet                          # 7,043 × 28 — encoded, unscaled
│   ├── curated.parquet                          # 7,043 × 27 — encoded model features
│   └── clustered.parquet                        # curated data plus clustering labels and PCA coordinates
│
├── app/
│   ├── ui.py                                    # Streamlit entrypoint
│   └── components/
│       ├── eda_olap.py
│       ├── customer_segments.py
│       ├── anomaly_detection.py
│       └── predictive_modeling.py
│
├── models/
│   ├── manifest.json                            # model metadata, feature order, thresholds
│   ├── logisticregression.pkl
│   ├── randomforest.pkl
│   └── xgboost.pkl
│
├── notebooks/
│   ├── 01_etl_pipeline.ipynb                    # ETL: clean → encode → normalize
│   ├── 02_olap_analytics.ipynb                  # OLAP: churn rate pivots & heatmap
│   └── 03_churn_models.ipynb                    # LR + RF + XGBoost + SHAP
│
├── src/
│   ├── baseline_model.py                        # Logistic Regression baseline
│   ├── train_churn.py                           # Ensemble models + threshold tuning
│   └── customer_segmentation.py                 # KMeans / GMM / HDBSCAN pipeline
│
├── outputs/
│   ├── olap/                                    # 4 bar charts + cross-dim heatmap (PNG + CSV)
│   ├── churn_models/                            # PR curves for LR, RF, XGBoost
│   ├── interpretability/                        # LR coefficients CSV, RF/XGB SHAP plots
│   ├── segmentation/                            # Cluster profiles (CSV), silhouette charts
│   ├── outlier_detection/                       # anomaly scores, overlaps, top-anomaly exports
│   ├── classification_report.txt
│   └── precision_recall_curve.png
│
├── EDA_Report_with_Visualizations.ipynb         # Exploratory data analysis
├── EDA_Conclusions.md
├── CHECKIN2_HANDOFF.md
└── requirements.txt
```

---

## Results Summary

### Supervised Learning

| Model | CV ROC-AUC | Tuned Threshold | Churn Recall | Churn F1 |
|-------|-----------|----------------|-------------|---------|
| Logistic Regression | 0.847 | 0.305 | 0.928 | 0.591 |
| Random Forest | 0.830 | 0.115 | 0.898 | 0.560 |
| XGBoost | 0.833 | 0.230 | 0.877 | 0.582 |

Threshold tuned on validation set by maximizing $F_2$-score ($\beta=2$, recall-weighted).

### Customer Segmentation

| Algorithm | Clusters | Churn-Rate Variance | Highest-Risk Segment |
|-----------|----------|--------------------|--------------------|
| K-Means | 2 | 0.0052 | Cluster 0 — 31.6% churn |
| GMM | 4 | 0.0293 | Cluster 2 — 48.3% churn (100% M2M) |
| HDBSCAN (PCA) | 11 | 0.0560 | Cluster 2 — 66.8% churn |
| HDBSCAN (Gower) | 10 | **0.0572** | Cluster 2 — **69.8% churn** |

---

## Dashboard Overview

The Streamlit dashboard is organized around a shared convergence story rather than four disconnected views. A shared sidebar lets you filter by:

- Contract Type
- Internet Service
- Senior Citizen

The main surface includes four tabs:

1. **EDA & OLAP** — KPI cards, churn overview, numeric distributions, service analysis, OLAP drill-down, and contract × internet churn heatmap.
2. **Customer Segments** — algorithm selector, cluster KPIs, PCA scatter, profile table, cluster deep dive, and silhouette comparison.
3. **Anomaly Detection** — anomaly selector, top-anomaly KPIs, score distributions by churn, overlap analysis, anomaly explorer, and score-vs-probability scatter.
4. **Predictive Modeling** — model selector, threshold controls, confusion matrix, PR curve, threshold analytics, model comparison, ranked predictions, export, local explainability, what-if scoring, and retention simulation.

Cross-tab drill-through is supported from segmentation and anomaly views into the predictive table.

---

## Setup and Reproduction

### Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `pyarrow`, `matplotlib`, `plotly`, `streamlit`, `joblib`

### Run Order

```bash
# 1. Build data layers
jupyter nbconvert --to notebook --execute notebooks/01_etl_pipeline.ipynb

# 2. OLAP analytics
jupyter nbconvert --to notebook --execute notebooks/02_olap_analytics.ipynb

# 3. Churn prediction models
python src/train_churn.py          # or run notebooks/03_churn_models.ipynb

# 4. Customer segmentation
python src/customer_segmentation.py

# 5. Outlier detection
python src/outlier_detection.py

# 6. Streamlit dashboard
streamlit run app/ui.py
```

All analytical outputs are written to `outputs/`. The serialized dashboard models live under `models/`.

---

## Quick Verification Checklist

After `streamlit run app/ui.py`, verify:

1. The sidebar shows filters for contract type, internet service, and senior citizen status.
2. The convergence banner appears above all four tabs.
3. The EDA & OLAP tab shows KPI cards plus a contract × internet heatmap in the OLAP view.
4. The Customer Segments tab loads PCA scatter plots and allows drill-through into Predictive Modeling.
5. The Anomaly Detection tab shows score distributions and can pass a selected anomaly into Predictive Modeling.
6. The Predictive Modeling tab loads all available models, updates threshold metrics interactively, and supports CSV export.

---

## Deliverable Status

| Deliverable | Status |
|-------------|--------|
| ETL pipeline (3-layer Parquet architecture) | ✅ Complete |
| OLAP analytics (4 dimensions + heatmap) | ✅ Complete |
| Supervised learning (LR + RF + XGBoost + SHAP) | ✅ Complete |
| Customer segmentation (KMeans + GMM + HDBSCAN) | ✅ Complete |
| Outlier detection (Isolation Forest + LOF) | ✅ Complete |
| Streamlit dashboard | ✅ Complete |

---

## Dataset

> IBM Telco Customer Churn Dataset — available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
> 7,043 customer records · 21 features · 26.54% churn rate · ~3:1 class imbalance.
