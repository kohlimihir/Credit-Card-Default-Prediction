# Credit Card Default Prediction System
### Industrial-Grade Credit Risk Data Science | 3 Real Datasets

### [🚀 Live Dashboard](https://credit-card-default-prediction-dashboard.streamlit.app/)

---

## Quick Start

```bash
pip install -r requirements.txt
python pipeline.py                     # uses UCI dataset (auto-download)
python pipeline.py --dataset gmc       # Give Me Some Credit
python pipeline.py --dataset homecredit  # Home Credit Default Risk
```

---

## Supported Real Datasets

| Dataset | Rows | Features | Source | Auto-Download |
|---------|------|----------|--------|---------------|
| **UCI Credit Card Default** | 30,000 | 23 raw → 64 engineered → 32 after VIF | UCI ML Repo | ✅ `ucimlrepo` package |
| **Give Me Some Credit** | 150,000 | 10 raw → 30+ engineered | Kaggle 2011 | ✅ Kaggle API |
| **Home Credit Default Risk** | 307,511 | 120 raw → 150+ engineered | Kaggle 2018 | ✅ Kaggle API |

---

## Dataset Setup

### Option 1 — UCI (Recommended for first run)

Auto-download via Python package:
```bash
pip install ucimlrepo
python pipeline.py --dataset uci
```

Manual download:
```
1. Go to: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
2. Click Download → save file to data/
3. python pipeline.py
```

### Option 2 — Give Me Some Credit (150k rows, richer)

```bash
# Setup Kaggle API
pip install kaggle
# Download kaggle.json from: https://www.kaggle.com/settings → API
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle competitions download -c GiveMeSomeCredit -p data/
unzip data/GiveMeSomeCredit.zip -d data/

# Run
python pipeline.py --dataset gmc
```

Manual: https://www.kaggle.com/c/GiveMeSomeCredit/data → download `cs-training.csv` → save as `data/cs-training.csv`

### Option 3 — Home Credit (307k rows, most industrial)

```bash
kaggle competitions download -c home-credit-default-risk -p data/
unzip data/home-credit-default-risk.zip -d data/
python pipeline.py --dataset homecredit
```

Manual: https://www.kaggle.com/c/home-credit-default-risk/data → download `application_train.csv` → save to `data/`

---

## Architecture (5 Modules)

```
Module 1: Data Loading + Feature Engineering
    ├── UCILoader          → ucimlrepo / direct URL / GitHub mirror
    ├── GiveMeSomeCreditLoader → Kaggle API / manual
    └── HomeCreditLoader   → Kaggle API / manual

    ├── UCIFeatureEngineer      → utilization, payment ratio, delinquency trend
    ├── GMCFeatureEngineer      → delinquency severity, debt burden, income ratios
    └── HomeCreditFeatureEngineer → credit ratios, bureau scores, employment

    Feature Selection (Multicollinearity Removal)
    ├── Pairwise correlation filter (|r| > 0.85)
    └── Iterative VIF elimination (VIF > 10)

Module 2: WoE Scorecard (Regulatory Baseline)
    └── WoE binning → IV selection → Logistic Regression → FICO-style 300-850 points

Module 3: ML Models + SHAP
    ├── XGBoost + LightGBM with Platt probability calibration
    ├── Global SHAP (beeswarm) — which features drive default risk?
    ├── Local SHAP (waterfall)  — why is THIS customer high risk?
    └── Head-to-head model comparison

Module 4: Customer Segmentation
    ├── K-Means behavioural clustering (Transactors / Revolvers / Stressed / Delinquent)
    ├── Risk tier assignment (Low / Medium / High / Very High)
    └── 2×2 Business Action Matrix

Module 5: Model Governance
    ├── Population Stability Index (PSI) — input drift detection
    ├── Gini stability over time — discrimination degradation
    ├── Bias audit — subgroup AUC analysis (fair lending)
    └── Model Card (JSON + text report)
```

---

## Results (UCI Dataset — Latest Run)

| Model | Dataset | AUC-ROC | Gini | KS Stat | Brier Score |
|-------|---------|---------|------|---------|-------------|
| Scorecard (WoE+LR) | Test | 0.7262 | 0.4524 | 0.3535 | 0.2087 |
| Scorecard (WoE+LR) | OOT  | 0.7268 | 0.4536 | 0.3431 | 0.2044 |
| XGBoost | Test | 0.7407 | 0.4815 | 0.3619 | 0.1481 |
| XGBoost | OOT  | 0.7379 | 0.4758 | 0.3458 | 0.1475 |
| LightGBM | Test | 0.7391 | 0.4782 | 0.3565 | 0.1479 |
| LightGBM | OOT  | 0.7406 | 0.4812 | 0.3478 | 0.1474 |

**Governance:** Score PSI = 0.0007 🟢 | Gini range = 0.658–0.710 | All 20 monitored features PSI 🟢 STABLE

---

## Output Files

All saved to `reports/`:

| File | Description |
|------|-------------|
| `scorecard_evaluation.png` | 4-panel: ROC, KS plot, score distribution, calibration |
| `iv_chart.png` | Information Value per feature |
| `scorecard_points_table.csv` | Points-based scorecard table |
| `xgb_shap_summary.png` | Global SHAP beeswarm |
| `xgb_shap_waterfall_customer0.png` | Single customer explanation |
| `model_comparison.png` | ROC + Gini bar: all models |
| `calibration_comparison.png` | Probability calibration quality |
| `multicollinearity_drop_report.csv` | Correlation + VIF feature drops |
| `cluster_profiles.png` | Segment heatmap |
| `pca_clusters.png` | 2D PCA scatter |
| `monitoring_dashboard.png` | Gini stability + PSI + bias audit |
| `model_card.json` | Machine-readable governance doc |
| `model_governance_report.txt` | Human-readable governance report |
| `results_summary.csv` | All model metrics |
| `bias_audit.csv` | AUC by demographic subgroup |

---

## Project Structure

```
credit_default_system/
├── pipeline.py                  # Master orchestrator
├── config.py                    # All settings (change ACTIVE_DATASET here)
├── requirements.txt
├── README.md
├── data/                        # Downloaded datasets (auto-created)
├── reports/                     # All output files (auto-created)
├── models/                      # Saved model artifacts (auto-created)
└── src/
    ├── data_loader.py           # Multi-dataset loader (UCI / GMC / HomeCredit)
    ├── feature_engineering.py   # Dataset-aware feature builder
    ├── feature_selection.py     # Correlation + VIF multicollinearity filter
    ├── woe_scorecard.py         # WoE binning + logistic scorecard
    ├── ml_models.py             # XGBoost + LightGBM + SHAP
    ├── segmentation.py          # K-Means + risk tiers + action matrix
    └── monitoring.py            # PSI + Gini stability + bias + model card
```

---