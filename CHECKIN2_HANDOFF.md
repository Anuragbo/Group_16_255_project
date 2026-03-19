# Check-in 2 Handoff — ETL & OLAP (Dhruv)

## Task 1: ETL Pipeline
**Notebook:** `notebooks/01_etl_pipeline.ipynb`

| Stage | Rows | Cols | File |
|-------|------|------|------|
| Raw CSV | 7,043 | 21 | *(source)* |
| Raw Parquet | 7,043 | 21 | `data/raw.parquet` |
| Staging Parquet | 7,043 | 28 | `data/staging.parquet` |
| Curated Parquet | 7,043 | 27 | `data/curated.parquet` |

**Steps performed:**
- **TotalCharges fix:** 11 blank entries (all tenure=0, new customers) → imputed with `MonthlyCharges` value; cast to float
- **Binary encoding (0/1):** `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`
- **Add-on service encoding (0/1):** `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` — "No internet service" and "No phone service" both treated as 0
- **One-hot encoding:** `InternetService` (3 values), `Contract` (3 values), `PaymentMethod` (4 values) → added 7 new columns
- **StandardScaler normalization:** `tenure`, `MonthlyCharges`, `TotalCharges`
- **Dropped:** `customerID` (not a feature)

**Use `data/curated.parquet` for all modeling.** Use `data/raw.parquet` for OLAP (readable labels).

---

## Task 2: OLAP Analytics
**Notebook:** `notebooks/02_olap_analytics.ipynb`
**Charts + CSVs:** `outputs/olap/`

### Key findings for report (Section 5):

| Dimension | Highest churn | Rate | Lowest churn | Rate |
|-----------|--------------|------|-------------|------|
| Contract type | Month-to-month | **42.7%** | Two year | 2.8% |
| Tenure cohort | 0–12 months | **47.4%** | 49–72 months | 9.5% |
| Payment method | Electronic check | **45.3%** | Credit card (auto) | 15.2% |
| Internet service | Fiber optic | **41.9%** | No internet | 7.4% |

**Business insight for report:** Churn is concentrated in a clear high-risk profile — month-to-month customers in their first year paying with electronic check on fiber optic service churn at nearly 5× the rate of long-tenure, two-year contract holders. The cross-dimensional heatmap (`churn_contract_x_internet_heatmap.png`) shows month-to-month + fiber optic is the highest-risk combination. These OLAP findings directly motivate the need for both supervised prediction (who will churn) and segmentation (which group they belong to).

### Output files:
- `outputs/olap/churn_by_contract.csv` / `.png`
- `outputs/olap/churn_by_tenure_cohort.csv` / `.png`
- `outputs/olap/churn_by_payment_method.csv` / `.png`
- `outputs/olap/churn_by_internet_service.csv` / `.png`
- `outputs/olap/churn_contract_x_internet.csv` / `_heatmap.png`
