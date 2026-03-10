# Exploratory Data Analysis: Telco Customer Churn

## 1. Dataset Overview

### Basic Information
- **Total Records**: 7,043 customers
- **Total Features**: 21 attributes
- **Data Quality**: No missing values detected
- **Time Period**: Cross-sectional snapshot of customer data

### Data Composition
- **Demographic Features**: Gender, Senior Citizen status, Partner, Dependents
- **Service Features**: Internet service type, various add-on services (security, backup, tech support, streaming)
- **Contract Information**: Contract type, billing method, payment method
- **Financial Metrics**: Monthly charges, total charges, tenure
- **Target Variable**: Churn status (Yes/No)

---

## 2. Key Findings

### 2.1 Churn Distribution
- **Non-Churned Customers**: 5,174 (73.46%)
- **Churned Customers**: 1,869 (26.54%)
- **Churn Rate**: 26.54% - Significant proportion of customers are leaving the company
- **Class Imbalance**: Moderate imbalance (3:1 ratio) - consider for modeling

### 2.2 Demographic Characteristics

**Gender Distribution:**
- Male: 3,555 (50.46%)
- Female: 3,488 (49.54%)
- **Conclusion**: Nearly equal gender distribution in customer base

**Senior Citizen Status:**
- Non-Senior: 5,900 (83.79%)
- Senior Citizens: 1,143 (16.21%)
- **Conclusion**: Customer base is predominantly younger

**Family Status:**
- No Partner: 3,641 (51.68%)
- With Partner: 3,402 (48.32%)
- **Conclusion**: Slight majority without partners

**Dependents:**
- No Dependents: 4,933 (70.01%)
- With Dependents: 2,110 (29.99%)
- **Conclusion**: Most customers do not have dependents

### 2.3 Service Subscriptions

**Internet Service Types:**
- Fiber Optic: 3,096 (43.94%) - Most popular
- DSL: 2,421 (34.39%)
- No Internet Service: 1,526 (21.66%)
- **Insight**: Majority of customers have internet service; Fiber optic is preferred technology

**Phone Service:**
- With Phone Service: 6,361 (90.3%)
- Without Phone Service: 682 (9.7%)
- **Insight**: Phone service adoption is very high

**Add-on Services (for Internet customers):**
- Online Security: 2,019 users (28.65% of total)
- Online Backup: 2,429 users (34.49% of total)
- Device Protection: 2,422 users (34.39% of total)
- Tech Support: 2,044 users (29.01% of total)
- Streaming TV: 2,707 users (38.41% of total)
- Streaming Movies: 2,732 users (38.79% of total)
- **Conclusion**: About 30-39% adoption rates for add-on services; Streaming services are most popular

### 2.4 Contract and Billing Information

**Contract Types:**
- Month-to-month: 3,875 (55.02%) - Largest segment
- Two-year: 1,695 (24.07%)
- One-year: 1,473 (20.91%)
- **Insight**: More than half of customers are on short-term contracts (month-to-month), which may correlate with higher churn risk

**Paperless Billing:**
- Paperless: 4,171 (59.23%)
- Paper: 2,872 (40.77%)
- **Insight**: Majority prefer paperless billing

**Payment Methods:**
- Electronic Check: 2,365 (33.58%) - Most common
- Mailed Check: 1,612 (22.88%)
- Bank Transfer (Automatic): 1,544 (21.91%)
- Credit Card (Automatic): 1,522 (21.60%)
- **Insight**: Largest segment uses electronic checks (potentially less committed); only ~43.5% use automatic payments

---

## 3. Numerical Features Analysis

### 3.1 Tenure (Customer Lifetime)

**Statistics:**
- Mean: 32.37 months (~2.7 years)
- Median: 29 months (~2.4 years)
- Min: 0 months (new customers)
- Max: 72 months (6 years)
- Std Dev: 24.56 months

**Distribution Pattern:**
- Wide range indicates customers across all lifecycle stages
- Presence of brand new customers (0 months) suggests ongoing acquisition

### 3.2 Monthly Charges

**Statistics:**
- Mean: $64.76
- Median: $70.35
- Min: $18.25
- Max: $118.75
- Std Dev: $30.09

**Distribution Pattern:**
- Relatively uniform distribution with mean slightly lower than median
- Range suggests significant variation in service packages and bundles
- Higher-priced services likely bundled with multiple add-ons

### 3.3 Total Charges

**Observations:**
- Highly diverse values (6,531 unique values across 7,043 records)
- Calculated as: Monthly Charges × Tenure
- **Data Quality Issue**: 11 blank/empty values detected (represented as spaces) - requires data cleaning before modeling

### 3.4 Senior Citizen (Binary)

- 16.21% are senior citizens
- Relatively balanced distribution for a demographic trait

---

## 4. Correlation Analysis

### Correlation Matrix Results

```
                    SeniorCitizen    tenure  MonthlyCharges
SeniorCitizen       1.000000         0.0166   0.2202
tenure              0.0166           1.0000   0.2479
MonthlyCharges      0.2202           0.2479   1.0000
```

### Key Insights:

1. **Senior Citizen & Tenure**: Near-zero correlation (0.0166)
   - Age and customer loyalty are independent
   - Senior citizens don't necessarily have longer tenure

2. **Monthly Charges & Senior Citizen**: Weak correlation (0.2202)
   - Senior citizens tend to pay slightly more
   - May indicate different service preferences or discounts

3. **Monthly Charges & Tenure**: Weak positive correlation (0.2479)
   - Longer-tenured customers tend to pay slightly more
   - Could indicate service plan inflation or bundle additions over time

4. **Overall**: Low correlations suggest these numeric features are relatively independent
   - Churn likely driven by factors beyond these three numeric features
   - Categorical features may be more predictive

---


## 8. Summary Statistics Table

| Metric | Value |
|--------|-------|
| Total Customers | 7,043 |
| Churn Rate | 26.54% |
| Avg Tenure (months) | 32.37 |
| Avg Monthly Charge | $64.76 |
| Median Monthly Charge | $70.35 |
| Month-to-month Contracts | 55.02% |
| Internet Service Users | 78.34% |
| Customers with Add-on Services | 30-39% |
| Automatic Payment Users | 43.51% |

---


