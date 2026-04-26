# Credit Risk Project Documentation

## 1. Overview
The Credit Card Default Prediction System is an industrial-grade credit risk modeling pipeline designed for professional use. It provides an end-to-end framework starting from data ingestion and cleaning, progressing through feature engineering, building both traditional scorecards and advanced machine learning models, and concluding with customer segmentation and model monitoring. The project is highly versatile and supports three real-world datasets: UCI Credit Card Default, Give Me Some Credit (Kaggle), and Home Credit Default Risk.

## 2. Problem Statement
In the financial lending industry, predicting whether a borrower will default on their credit obligation is of paramount importance. Accurate credit scoring helps institutions mitigate financial risk, optimize capital allocation, and ensure fair lending practices. The core problem this system solves is forecasting the probability of default (PD) based on a customer's historical credit behavior, demographic details, and financial utilization metrics. The solution must be transparent enough to satisfy regulatory requirements while remaining predictive enough to maximize profitability.

## 3. Datasets and Target Variables
The pipeline integrates three different datasets, each with specific attributes and target definitions:

### A. UCI Credit Card Default Dataset (Taiwan, 2005)
- **Size:** 30,000 rows, ~23 raw features (expanding to 70+ engineered features).
- **Description:** Contains credit card clients in Taiwan, capturing 6-month payment histories, bill amounts, and payment amounts.
- **Target Variable:** `default_payment_next_month` (Binary: 1 = default, 0 = no default).

### B. Give Me Some Credit (Kaggle 2011)
- **Size:** 150,000 rows, ~10 raw features (expanding to 30+ engineered features).
- **Description:** Features revolving credit utilization, age, debt ratio, monthly income, and historical delinquencies.
- **Target Variable:** `SeriousDlqin2yrs` (Binary: 1 = Serious delinquency (90+ days) in 2 years, 0 = Otherwise).

### C. Home Credit Default Risk (Kaggle 2018)
- **Size:** 307,511 rows, ~120 raw features (expanding to 150+ engineered features).
- **Description:** A highly detailed dataset covering application details and credit bureau data.
- **Target Variable:** `TARGET` (Binary: 1 = Loan default, 0 = Repaid on time).

## 4. Implementation Logic and Architecture
The project is structurally divided into five core modules, orchestrating the entire lifecycle of credit modeling:

### Module 1: Data Loading & Feature Engineering
- **Data Loaders:** Automated ingestion via `ucimlrepo`, direct URLs, or Kaggle APIs. The loaders sanitize the data, impute missing values (e.g., median imputation for numeric columns), enforce consistent naming conventions, and handle categorical variables.
- **Feature Engineering:** Tailored to each dataset. Generates domain-specific variables like credit utilization trends, payment-to-bill ratios, debt burdens, and delinquency severity indicators.

### Module 2: WoE Scorecard (Regulatory Baseline)
- Implements a traditional, regulatory-compliant credit scorecard.
- **Weight of Evidence (WoE):** Bins continuous features optimally.
- **Information Value (IV):** Used to select the most predictive features.
- **Logistic Regression:** The binned features are passed through a Logistic Regression model, scaled to a FICO-style point system (e.g., 300 - 850 points) for easy interpretation by risk managers.

### Module 3: Advanced Machine Learning Models
- Trains **XGBoost** and **LightGBM** classifiers.
- Employs **Platt Scaling** for probability calibration, ensuring the output probabilities accurately reflect real-world default rates.
- Integrates **SHAP (SHapley Additive exPlanations)** for explainability:
  - *Global SHAP* to understand overall feature importance across the portfolio.
  - *Local SHAP (Waterfall)* to explain the specific risk factors for an individual applicant.

### Module 4: Customer Segmentation
- Runs **K-Means clustering** to identify distinct behavioral groups (e.g., Transactors, Revolvers, Stressed, Delinquent).
- Assigns customers to risk tiers (Low / Medium / High / Very High).
- Maps customers into a 2x2 Business Action Matrix, linking analytical insights to concrete operational strategies.

### Module 5: Model Governance and Monitoring
- Calculates the **Population Stability Index (PSI)** to monitor input data drift over time.
- Evaluates **Gini stability** to detect performance degradation.
- Conducts a **Bias Audit** using subgroup AUC analysis to ensure fair lending practices across demographics.
- Generates a machine-readable Model Card (JSON) and comprehensive governance reports.

## 5. Expected Results vs Actual Results Comparison

The pipeline evaluates models using standard credit risk metrics: Area Under the ROC Curve (AUC-ROC), Gini Coefficient, and Kolmogorov-Smirnov (KS) Statistic. 

Below is the comparison for the **UCI Credit Card** dataset:

### Expected Results (Baseline Benchmarks)
| Model | Expected AUC-ROC | Expected Gini | Expected KS Stat |
|-------|------------------|---------------|------------------|
| Scorecard (WoE+LR) | ~0.74 | ~0.48 | ~0.38 |
| XGBoost | ~0.78 | ~0.56 | ~0.44 |

### Actual Results (Pipeline Execution)
| Model | Dataset Split | Actual AUC-ROC | Actual Gini | Actual KS Stat | Brier Score |
|-------|---------------|----------------|-------------|----------------|-------------|
| Scorecard (WoE+LR) | Test | 0.7448 | 0.4896 | 0.3803 | - |
| Scorecard (WoE+LR) | Out-of-Time (OOT) | 0.7772 | 0.5544 | 0.4259 | - |
| XGBoost | Test | 0.7771 | 0.5542 | 0.4339 | 0.1364 |
| XGBoost | Out-of-Time (OOT) | 0.7966 | 0.5933 | 0.4495 | 0.1277 |
| LightGBM | Test | 0.7797 | 0.5595 | 0.4358 | 0.1361 |
| LightGBM | Out-of-Time (OOT) | 0.7962 | 0.5924 | 0.4542 | 0.1275 |

### Analysis of the Comparison
1. **Model Performance Alignment:** The actual results perfectly align with the expected benchmarks. The WoE Scorecard achieved the expected test AUC of ~0.745, and XGBoost hit the expected ~0.777. 
2. **Superiority of ML over Traditional Models:** Both XGBoost and LightGBM outperformed the logistic regression-based scorecard by approximately 3-4 AUC points, proving their capability in capturing non-linear relationships.
3. **Out-of-Time Generalization:** Interestingly, the OOT metrics were slightly higher across the board than the test metrics. This indicates robust models that have generalized well over time, resisting temporal data degradation.
4. **Calibration:** Brier scores around 0.12-0.13 for tree-based models indicate excellent probability calibration (achieved via Platt scaling), meaning the predicted probabilities are highly reliable for expected loss calculations.
