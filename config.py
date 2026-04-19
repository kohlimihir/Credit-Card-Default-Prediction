"""
config.py — Central configuration for Credit Default Prediction System
=======================================================================
Change ACTIVE_DATASET to switch between the 3 supported real datasets.
All other settings auto-adjust per dataset.
"""

import os

# ─────────────────────────────────────────────
# DATASET SELECTION  ← change this to switch datasets
# ─────────────────────────────────────────────
#   "uci"   → UCI Credit Card Default (Taiwan, 2005) — 30,000 rows, auto-download
#   "gmc"   → Give Me Some Credit (Kaggle 2011)      — 150,000 rows, Kaggle API
#   "homecredit" → Home Credit Default Risk (Kaggle) — 307,511 rows, Kaggle API
ACTIVE_DATASET = "uci"

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

for d in [DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# DATASET METADATA  (used by loader + feature engineering)
# ─────────────────────────────────────────────
DATASET_CONFIGS = {
    "uci": {
        "name":        "UCI Credit Card Default (Taiwan)",
        "target_col":  "default_payment_next_month",
        "id_col":      "ID",
        "n_rows":      30_000,
        "event_rate":  0.2212,
        "description": "30,000 credit card customers. 6-month payment history. Binary: default next month.",
        # Download sources — tried in order
        "download_sources": [
            {
                "method": "ucimlrepo",
                "dataset_id": 350,
                "description": "Official UCI ML Repo Python package",
            },
            {
                "method": "direct_url",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
                "filename": "uci_raw.xls",
                "file_format": "xls",
                "description": "Direct download from UCI repository",
            },
            {
                "method": "github_mirror",
                "url": "https://raw.githubusercontent.com/dsavg/capstone_credit_card_default_prediction/master/data/UCI_Credit_Card.csv",
                "filename": "uci_raw_mirror.csv",
                "file_format": "csv",
                "description": "GitHub mirror (CSV format)",
            },
        ],
        "manual_instructions": """
MANUAL DOWNLOAD — UCI Credit Card Default:
  1. Go to: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
  2. Click 'Download' → save as data/uci_raw.xls
  3. Re-run pipeline.py
        """,
    },

    "gmc": {
        "name":        "Give Me Some Credit (Kaggle)",
        "target_col":  "SeriousDlqin2yrs",
        "id_col":      "ID",
        "n_rows":      150_000,
        "event_rate":  0.0667,
        "description": "150,000 US borrowers. Delinquency history, income, utilisation. Binary: 90+ day delinquency.",
        "kaggle_competition": "GiveMeSomeCredit",
        "kaggle_file": "cs-training.csv",
        "manual_instructions": """
MANUAL DOWNLOAD — Give Me Some Credit:
  Option A (Kaggle API):
    1. Install:  pip install kaggle
    2. Set up:   ~/.kaggle/kaggle.json  (from https://www.kaggle.com/settings → API)
    3. Run:      kaggle competitions download -c GiveMeSomeCredit -p data/
    4. Unzip:    unzip data/GiveMeSomeCredit.zip -d data/

  Option B (Browser):
    1. Go to: https://www.kaggle.com/c/GiveMeSomeCredit/data
    2. Download cs-training.csv → save as data/cs-training.csv
    3. Re-run pipeline.py
        """,
    },

    "homecredit": {
        "name":        "Home Credit Default Risk (Kaggle)",
        "target_col":  "TARGET",
        "id_col":      "SK_ID_CURR",
        "n_rows":      307_511,
        "event_rate":  0.0817,
        "description": "307,511 loan applications. Rich bureau + application data. Binary: default.",
        "kaggle_competition": "home-credit-default-risk",
        "kaggle_file": "application_train.csv",
        "manual_instructions": """
MANUAL DOWNLOAD — Home Credit Default Risk:
  Option A (Kaggle API):
    1. Install:  pip install kaggle
    2. Set up:   ~/.kaggle/kaggle.json
    3. Run:      kaggle competitions download -c home-credit-default-risk -p data/
    4. Unzip:    unzip data/home-credit-default-risk.zip -d data/

  Option B (Browser):
    1. Go to: https://www.kaggle.com/c/home-credit-default-risk/data
    2. Download application_train.csv → save as data/application_train.csv
    3. Re-run pipeline.py
        """,
    },
}

# Active dataset shorthand (used throughout codebase)
DATASET_CFG    = DATASET_CONFIGS[ACTIVE_DATASET]
TARGET_COLUMN  = DATASET_CFG["target_col"]
ID_COLUMN      = DATASET_CFG["id_col"]

# ─────────────────────────────────────────────
# UCI COLUMN DEFINITIONS (used in feature_engineering)
# ─────────────────────────────────────────────
UCI_DEMOGRAPHIC_COLS    = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
UCI_CREDIT_COLS         = ["LIMIT_BAL"]
UCI_PAYMENT_STATUS_COLS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
UCI_BILL_AMT_COLS       = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
UCI_PAY_AMT_COLS        = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

# ─────────────────────────────────────────────
# GMC COLUMN DEFINITIONS
# ─────────────────────────────────────────────
GMC_DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
GMC_FINANCIAL_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
]

# ─────────────────────────────────────────────
# HOME CREDIT KEY COLUMNS
# ─────────────────────────────────────────────
HC_NUMERIC_COLS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "DAYS_LAST_PHONE_CHANGE", "OWN_CAR_AGE",
    "AMT_REQ_CREDIT_BUREAU_YEAR", "AMT_REQ_CREDIT_BUREAU_MON",
    "BUREAU_ACTIVE_LOANS", "BUREAU_CLOSED_LOANS",
]
HC_CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
UTILIZATION_CLIP    = (0.0, 2.0)
PAYMENT_RATIO_CLIP  = (0.0, 5.0)

# ─────────────────────────────────────────────
# WoE / SCORECARD
# ─────────────────────────────────────────────
WOE_MAX_BINS            = 10
IV_THRESHOLD_DROP       = 0.02
IV_THRESHOLD_SUSPICIOUS = 0.5
REGULARIZATION_C        = 0.1

# ─────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": round((1 - DATASET_CFG["event_rate"]) / DATASET_CFG["event_rate"], 2),
    "eval_metric":      "auc",
    "random_state":     42,
    "n_jobs":           -1,
}

LGBM_PARAMS = {
    "n_estimators":     500,
    "max_depth":        5,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_samples":50,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "class_weight":     "balanced",
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

CV_FOLDS            = 5
TEST_SIZE           = 0.20
OOT_CUTOFF_QUANTILE = 0.80      # top 20% = out-of-time holdout

# ─────────────────────────────────────────────
# SEGMENTATION
# ─────────────────────────────────────────────
N_CLUSTERS           = 4
CLUSTER_RANDOM_STATE = 42

RISK_TIER_THRESHOLDS = {
    "Low Risk":       (0.00, 0.10),
    "Medium Risk":    (0.10, 0.25),
    "High Risk":      (0.25, 0.50),
    "Very High Risk": (0.50, 1.00),
}

# ─────────────────────────────────────────────
# MODEL GOVERNANCE
# ─────────────────────────────────────────────
PSI_THRESHOLDS = {"stable": 0.10, "moderate": 0.25}
GINI_DEGRADATION_ALERT = 0.05
MIN_ACCEPTABLE_GINI    = 0.40

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
