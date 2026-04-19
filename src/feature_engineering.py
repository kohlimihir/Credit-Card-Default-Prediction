"""
src/feature_engineering.py
===========================
Dataset-aware feature engineering pipeline.

Supports all 3 real datasets:
  UCI         → 6-month statement features: utilization, payment ratio, delinquency trend
  GMC         → Delinquency severity, income ratios, debt burden features
  Home Credit → Application ratios, credit bureau aggregates, annuity features

Entry point:
    fe = FeatureEngineer(dataset="uci")
    df_features = fe.fit_transform(df_raw)
    df_new_feat  = fe.transform(df_new_raw)

All feature builders return a DataFrame where:
  - Original raw columns are preserved (needed for bias audit etc.)
  - Engineered features are ADDED (not replacing originals)
  - Target and ID columns are carried through unchanged
  - Infinity / NaN values are cleaned before return
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ACTIVE_DATASET, TARGET_COLUMN, ID_COLUMN,
    UCI_BILL_AMT_COLS, UCI_PAY_AMT_COLS, UCI_PAYMENT_STATUS_COLS,
    UTILIZATION_CLIP, PAYMENT_RATIO_CLIP,
    GMC_DELINQUENCY_COLS, GMC_FINANCIAL_COLS,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Factory — returns correct engineer for active dataset
# ═════════════════════════════════════════════════════════════════════════════

def FeatureEngineer(dataset: str = None):
    """
    Factory function. Returns the correct FeatureEngineer subclass.

    Usage:
        fe = FeatureEngineer()              # uses config.ACTIVE_DATASET
        fe = FeatureEngineer("uci")
        fe = FeatureEngineer("gmc")
        fe = FeatureEngineer("homecredit")
    """
    dataset = dataset or ACTIVE_DATASET
    engineers = {
        "uci":        UCIFeatureEngineer,
        "gmc":        GMCFeatureEngineer,
        "homecredit": HomeCreditFeatureEngineer,
    }
    if dataset not in engineers:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose: {list(engineers.keys())}")
    return engineers[dataset]()


# ═════════════════════════════════════════════════════════════════════════════
# Base class
# ═════════════════════════════════════════════════════════════════════════════

class BaseFeatureEngineer:
    """Shared infrastructure for all feature engineers."""

    def __init__(self):
        self.feature_names_: list = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._build_features(df)
        df = self._clean_infinities(df)
        self.feature_names_ = [
            c for c in df.columns
            if c not in [ID_COLUMN, TARGET_COLUMN]
        ]
        self._fitted = True
        logger.info(
            f"[{self.__class__.__name__}] Features built | "
            f"Raw cols: {len(df.columns) - len(self.feature_names_)} metadata | "
            f"Feature cols: {len(self.feature_names_)}"
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        df = df.copy()
        df = self._build_features(df)
        df = self._clean_infinities(df)
        return df

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def get_feature_names(self) -> list:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        return self.feature_names_

    def get_feature_groups(self) -> dict:
        raise NotImplementedError

    def _clean_infinities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/-inf with NaN then fill with column median."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaNs with median
        for col in num_cols:
            if df[col].isnull().any():
                med = df[col].median()
                df[col] = df[col].fillna(med)
        return df


# ═════════════════════════════════════════════════════════════════════════════
# UCI Feature Engineer
# ═════════════════════════════════════════════════════════════════════════════

class UCIFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineering for UCI Credit Card Default dataset.

    Raw features available:
      LIMIT_BAL                   — credit limit
      SEX, EDUCATION, MARRIAGE, AGE — demographics
      PAY_1 … PAY_6               — payment status (most recent = PAY_1)
      BILL_AMT1 … BILL_AMT6       — monthly statement amounts
      PAY_AMT1 … PAY_AMT6         — monthly payments made

    Engineered feature families:
      1. Utilization              — bill/limit ratio, trend, volatility
      2. Payment behaviour        — payment ratio, consistency, zero-payment flag
      3. Delinquency              — severity, recency, chronic flag
      4. Balance trajectory       — is debt growing?
      5. Velocity                 — month-over-month change rates
      6. Interaction              — stress score, combined signals
    """

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._utilization(df)
        df = self._payment_behaviour(df)
        df = self._delinquency(df)
        df = self._balance_trajectory(df)
        df = self._velocity(df)
        df = self._interactions(df)
        return df

    # ── Feature families ─────────────────────────────────────────────────

    def _utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Credit utilization = Bill Amount / Credit Limit."""
        limit = df["LIMIT_BAL"].replace(0, np.nan)

        for i, col in enumerate(UCI_BILL_AMT_COLS, 1):
            df[f"UTIL_{i}"] = (df[col] / limit).clip(*UTILIZATION_CLIP)

        util_cols = [f"UTIL_{i}" for i in range(1, 7)]
        df["UTIL_MEAN"]   = df[util_cols].mean(axis=1)
        df["UTIL_MAX"]    = df[util_cols].max(axis=1)
        df["UTIL_STD"]    = df[util_cols].std(axis=1)
        df["UTIL_LATEST"] = df["UTIL_1"]
        # Positive = rising utilization (deteriorating)
        df["UTIL_TREND"]  = df["UTIL_1"] - df["UTIL_6"]
        return df

    def _payment_behaviour(self, df: pd.DataFrame) -> pd.DataFrame:
        """Payment ratio = Amount Paid / Bill Amount."""
        for i, (pay, bill) in enumerate(zip(UCI_PAY_AMT_COLS, UCI_BILL_AMT_COLS), 1):
            bill_val = df[bill].replace(0, np.nan)
            df[f"PAY_RATIO_{i}"] = (df[pay] / bill_val).clip(*PAYMENT_RATIO_CLIP).fillna(0)

        pr_cols = [f"PAY_RATIO_{i}" for i in range(1, 7)]
        df["PAY_RATIO_MEAN"]  = df[pr_cols].mean(axis=1)
        df["PAY_RATIO_MIN"]   = df[pr_cols].min(axis=1)
        df["PAY_RATIO_STD"]   = df[pr_cols].std(axis=1)
        df["MONTHS_PAID_FULL"]= (df[pr_cols] >= 1.0).sum(axis=1).astype(int)
        df["MONTHS_PAID_ZERO"]= (df[[f"PAY_AMT{i}" for i in range(1,7)]] == 0).sum(axis=1).astype(int)

        total_paid   = df[UCI_PAY_AMT_COLS].sum(axis=1)
        total_billed = df[UCI_BILL_AMT_COLS].sum(axis=1).replace(0, np.nan)
        df["TOTAL_PAY_RATIO"] = (total_paid / total_billed).clip(0, 5).fillna(0)
        return df

    def _delinquency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PAY_* columns: -2 = no consumption, -1 = paid in full,
        0 = minimum payment, 1-9 = months delayed.
        """
        pay = df[UCI_PAYMENT_STATUS_COLS]
        df["N_MONTHS_DELINQUENT"]    = (pay > 0).sum(axis=1)
        df["MAX_DELINQUENCY"]        = pay.max(axis=1)
        df["RECENT_DELINQUENCY"]     = df["PAY_1"].clip(lower=0)
        df["TOTAL_DELINQUENCY_SCORE"]= pay.clip(lower=0).sum(axis=1)
        df["RECENT_2M_DELINQUENT"]   = ((df["PAY_1"] > 0) | (df["PAY_2"] > 0)).astype(int)
        df["DELINQUENCY_TREND"]      = df["PAY_1"] - df["PAY_6"]   # positive = worsening
        df["CHRONIC_DELINQUENT"]     = (df["N_MONTHS_DELINQUENT"] >= 4).astype(int)
        return df

    def _balance_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Is the customer's outstanding debt growing or shrinking?"""
        df["BALANCE_CHANGE_1M"] = df["BILL_AMT1"] - df["BILL_AMT2"]
        df["BALANCE_CHANGE_3M"] = df["BILL_AMT1"] - df["BILL_AMT3"]
        df["BALANCE_CHANGE_6M"] = df["BILL_AMT1"] - df["BILL_AMT6"]

        limit = df["LIMIT_BAL"].replace(0, np.nan)
        df["BALANCE_GROWTH_RATE"] = (df["BALANCE_CHANGE_6M"] / limit).clip(-2, 2).fillna(0)
        df["BALANCE_TO_LIMIT"]    = (df["BILL_AMT1"] / limit).clip(0, 2).fillna(0)

        df["NET_DEBT_ACCUMULATION"] = (
            df[UCI_BILL_AMT_COLS].sum(axis=1) - df[UCI_PAY_AMT_COLS].sum(axis=1)
        )
        return df

    def _velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Month-over-month acceleration/deceleration signals."""
        df["PAY_AMT_VELOCITY"]  = df["PAY_AMT1"] - df["PAY_AMT2"]
        df["PAY_AMT_ACCEL"]     = (df["PAY_AMT1"] - df["PAY_AMT2"]) - (df["PAY_AMT2"] - df["PAY_AMT3"])
        df["BILL_AMT_VELOCITY"] = df["BILL_AMT1"] - df["BILL_AMT2"]
        df["UTIL_VELOCITY"]     = df["UTIL_1"] - df["UTIL_2"]
        df["AVG_MONTHLY_PAYMENT"]= df[UCI_PAY_AMT_COLS].mean(axis=1)
        df["STD_MONTHLY_PAYMENT"]= df[UCI_PAY_AMT_COLS].std(axis=1)
        return df

    def _interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-feature signals with incremental predictive power."""
        df["UTIL_X_DELINQUENCY"] = df["UTIL_MEAN"] * df["N_MONTHS_DELINQUENT"]
        df["STRESS_SCORE"]       = (1 - df["PAY_RATIO_MEAN"].clip(0, 1)) * df["UTIL_MEAN"]

        if "AGE" in df.columns:
            df["AGE_LIMIT_RATIO"] = df["AGE"] / (df["LIMIT_BAL"] / 10_000 + 1)

        avg_bill = df[UCI_BILL_AMT_COLS].mean(axis=1).replace(0, np.nan)
        df["LIMIT_TO_AVG_BILL"] = (df["LIMIT_BAL"] / avg_bill).clip(0, 20).fillna(10)
        return df

    def get_feature_groups(self) -> dict:
        return {
            "Utilization":   [c for c in self.feature_names_ if "UTIL" in c],
            "Payment":       [c for c in self.feature_names_ if "PAY_RATIO" in c or "MONTHS_PAID" in c or "TOTAL_PAY" in c],
            "Delinquency":   [c for c in self.feature_names_ if "DELINQUENT" in c or "DELINQUENCY" in c],
            "Balance":       [c for c in self.feature_names_ if "BALANCE" in c or "DEBT" in c],
            "Velocity":      [c for c in self.feature_names_ if "VELOCITY" in c or "ACCEL" in c],
            "Interaction":   [c for c in self.feature_names_ if "STRESS" in c or "X_" in c or "LIMIT_TO" in c],
            "Original":      [c for c in self.feature_names_ if c in ["LIMIT_BAL","AGE","SEX","EDUCATION","MARRIAGE"]],
        }


# ═════════════════════════════════════════════════════════════════════════════
# GMC Feature Engineer
# ═════════════════════════════════════════════════════════════════════════════

class GMCFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineering for Give Me Some Credit (Kaggle).

    Raw features:
      RevolvingUtilizationOfUnsecuredLines — revolving credit utilization (0–1)
      age                                  — borrower age
      NumberOfTime30-59DaysPastDueNotWorse — count of 30-59 day late payments
      DebtRatio                            — monthly obligations / monthly income
      MonthlyIncome                        — monthly gross income
      NumberOfOpenCreditLinesAndLoans      — total open credit lines
      NumberOfTimes90DaysLate              — count of 90+ day late payments
      NumberRealEstateLoansOrLines         — real estate credit lines
      NumberOfTime60-89DaysPastDueNotWorse — count of 60-89 day late payments
      NumberOfDependents                   — number of dependents

    Engineered feature families:
      1. Delinquency severity   — total, weighted, recency of late payments
      2. Debt burden            — income-adjusted ratios
      3. Credit complexity      — number and type of credit lines
      4. Life-stage features    — age + dependents + income
      5. Interaction signals    — combined risk indicators
    """

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._delinquency_severity(df)
        df = self._debt_burden(df)
        df = self._credit_complexity(df)
        df = self._lifestage(df)
        df = self._interactions(df)
        return df

    def _delinquency_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weighted delinquency score — more weight for longer delays."""
        c30  = "NumberOfTime30-59DaysPastDueNotWorse"
        c60  = "NumberOfTime60-89DaysPastDueNotWorse"
        c90  = "NumberOfTimes90DaysLate"

        df["TOTAL_DELINQUENCIES"]   = df[c30] + df[c60] + df[c90]
        # Weighted: 90+ day delinquencies are 3× more severe than 30-59
        df["WEIGHTED_DELINQUENCY"]  = df[c30] * 1 + df[c60] * 2 + df[c90] * 3
        df["EVER_90_LATE"]          = (df[c90] > 0).astype(int)
        df["EVER_LATE"]             = (df["TOTAL_DELINQUENCIES"] > 0).astype(int)
        df["CHRONIC_LATE"]          = (df["TOTAL_DELINQUENCIES"] >= 5).astype(int)
        # 90+day as share of all delinquencies (severity ratio)
        total = df["TOTAL_DELINQUENCIES"].replace(0, np.nan)
        df["SEVERE_DELINQUENCY_RATIO"] = (df[c90] / total).fillna(0)
        return df

    def _debt_burden(self, df: pd.DataFrame) -> pd.DataFrame:
        """Income-adjusted debt and affordability ratios."""
        income = df["MonthlyIncome"].replace(0, np.nan)

        # DebtRatio is already monthly debt / monthly income
        df["DEBT_RATIO"] = df["DebtRatio"].clip(0, 5)

        # Monthly debt obligation in dollar terms (if income is available)
        df["MONTHLY_DEBT_OBLIGATION"] = (df["DebtRatio"] * df["MonthlyIncome"]).clip(0, 1e6)

        # Residual income after debt service (negative = underwater)
        df["RESIDUAL_INCOME"] = (df["MonthlyIncome"] - df["MONTHLY_DEBT_OBLIGATION"]).clip(-1e5, 1e6)

        # Log income (reduces skewness)
        df["LOG_INCOME"] = np.log1p(df["MonthlyIncome"].fillna(0))

        # Affordability: debt per dependent
        deps = (df["NumberOfDependents"] + 1)  # +1 to include borrower
        df["DEBT_PER_DEPENDENT"] = (df["MONTHLY_DEBT_OBLIGATION"] / deps).clip(0, 1e5)

        return df

    def _credit_complexity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Credit line count and structure features."""
        df["TOTAL_CREDIT_LINES"] = (
            df["NumberOfOpenCreditLinesAndLoans"] +
            df["NumberRealEstateLoansOrLines"]
        )
        # Real estate share of total credit (more real estate = more stable historically)
        total = df["TOTAL_CREDIT_LINES"].replace(0, np.nan)
        df["RE_SHARE_OF_CREDIT"]  = (df["NumberRealEstateLoansOrLines"] / total).fillna(0)

        # High credit line count + high utilization = stressed
        df["UTIL_PER_CREDIT_LINE"] = (
            df["RevolvingUtilizationOfUnsecuredLines"] /
            (df["NumberOfOpenCreditLinesAndLoans"] + 1)
        )
        df["HIGH_UTILIZATION"]    = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.75).astype(int)
        df["OVER_LIMIT"]          = (df["RevolvingUtilizationOfUnsecuredLines"] > 1.0).astype(int)
        return df

    def _lifestage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Age and life-stage features."""
        df["AGE_BUCKET"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(float).fillna(2)

        df["INCOME_PER_DEPENDENT"] = (
            df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
        ).clip(0, 1e5)

        # Young + low income + high debt = high risk profile
        df["YOUNG_HIGH_DEBT"] = (
            (df["age"] < 35) & (df["DebtRatio"] > 0.5)
        ).astype(int)

        # Older with many dependents (financial stress indicator)
        df["SENIOR_DEPENDENTS"] = (
            (df["age"] > 50) & (df["NumberOfDependents"] > 2)
        ).astype(int)
        return df

    def _interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combined risk signals."""
        # High utilization AND delinquency = very high risk
        df["UTIL_X_DELINQUENCY"]  = (
            df["RevolvingUtilizationOfUnsecuredLines"] *
            df["TOTAL_DELINQUENCIES"]
        )
        # Debt ratio AND delinquency
        df["DEBT_X_DELINQUENCY"]  = df["DEBT_RATIO"] * df["WEIGHTED_DELINQUENCY"]
        # Composite stress score (0–1 normalised risk signal)
        df["STRESS_SCORE"] = (
            df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1) * 0.4 +
            (df["WEIGHTED_DELINQUENCY"] / (df["WEIGHTED_DELINQUENCY"].max() + 1e-6)) * 0.4 +
            df["DEBT_RATIO"].clip(0, 1) * 0.2
        )
        return df

    def get_feature_groups(self) -> dict:
        return {
            "Delinquency":     [c for c in self.feature_names_ if "DELINQUENT" in c or "DELINQUENCY" in c or "LATE" in c],
            "Debt Burden":     [c for c in self.feature_names_ if "DEBT" in c or "INCOME" in c or "RESIDUAL" in c],
            "Credit Lines":    [c for c in self.feature_names_ if "CREDIT" in c or "UTIL" in c or "LIMIT" in c],
            "Life Stage":      [c for c in self.feature_names_ if "AGE" in c or "DEPENDENT" in c or "YOUNG" in c or "SENIOR" in c],
            "Interaction":     [c for c in self.feature_names_ if "STRESS" in c or "X_" in c],
            "Original":        [c for c in self.feature_names_ if c in [
                "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
                "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
                "NumberRealEstateLoansOrLines", "NumberOfDependents",
            ]],
        }


# ═════════════════════════════════════════════════════════════════════════════
# Home Credit Feature Engineer
# ═════════════════════════════════════════════════════════════════════════════

class HomeCreditFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineering for Home Credit Default Risk.

    Focuses on application_train.csv (main table).
    Key feature families:
      1. Credit ratios    — annuity/income, credit/income, goods price/credit
      2. Age features     — DAYS_BIRTH → age, employment length
      3. External scores  — EXT_SOURCE_1/2/3 ensemble
      4. Credit request   — recent bureau enquiries
      5. Interaction      — cross-feature risk signals
    """

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._credit_ratios(df)
        df = self._age_employment(df)
        df = self._external_scores(df)
        df = self._bureau_enquiries(df)
        df = self._interactions(df)
        return df

    def _credit_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loan-to-income and affordability ratios."""
        inc = df["AMT_INCOME_TOTAL"].replace(0, np.nan)

        if "AMT_CREDIT" in df.columns:
            df["CREDIT_TO_INCOME"]  = (df["AMT_CREDIT"] / inc).clip(0, 20).fillna(0)
        if "AMT_ANNUITY" in df.columns:
            df["ANNUITY_TO_INCOME"] = (df["AMT_ANNUITY"] / inc).clip(0, 1).fillna(0)
        if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
            credit = df["AMT_CREDIT"].replace(0, np.nan)
            df["GOODS_TO_CREDIT"]   = (df["AMT_GOODS_PRICE"] / credit).clip(0, 2).fillna(1)
        if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
            credit = df["AMT_CREDIT"].replace(0, np.nan)
            df["ANNUITY_TO_CREDIT"] = (df["AMT_ANNUITY"] / credit).clip(0, 0.2).fillna(0)
        return df

    def _age_employment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Age and employment stability features."""
        if "DAYS_BIRTH" in df.columns:
            df["AGE_YEARS"]        = (df["DAYS_BIRTH"] / 365.25).astype(int)
            df["YOUNG_BORROWER"]   = (df["AGE_YEARS"] < 28).astype(int)
            df["SENIOR_BORROWER"]  = (df["AGE_YEARS"] > 60).astype(int)

        if "DAYS_EMPLOYED" in df.columns:
            df["EMPLOY_YEARS"]     = (df["DAYS_EMPLOYED"] / 365.25).clip(0, 40)
            df["SHORT_EMPLOYMENT"] = (df["EMPLOY_YEARS"] < 1).astype(int)

        if "DAYS_BIRTH" in df.columns and "DAYS_EMPLOYED" in df.columns:
            df["EMPLOY_TO_AGE_RATIO"] = (
                df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)
            ).clip(0, 1)
        return df

    def _external_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """EXT_SOURCE scores (external credit bureau / scoring sources)."""
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
                    if c in df.columns]
        if ext_cols:
            df["EXT_SOURCE_MEAN"]   = df[ext_cols].mean(axis=1)
            df["EXT_SOURCE_MIN"]    = df[ext_cols].min(axis=1)
            df["EXT_SOURCE_PRODUCT"]= df[ext_cols].prod(axis=1)
            # Number of available EXT_SOURCE values (missing = not in bureau)
            df["EXT_SOURCE_COUNT"]  = df[ext_cols].notna().sum(axis=1)
        return df

    def _bureau_enquiries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recent credit bureau enquiry count (proxy for credit-seeking behaviour)."""
        enquiry_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
        if enquiry_cols:
            df["TOTAL_ENQUIRIES_1Y"]  = df[[c for c in enquiry_cols if "YEAR" in c]].sum(axis=1) if any("YEAR" in c for c in enquiry_cols) else 0
            df["TOTAL_ENQUIRIES_ALL"] = df[enquiry_cols].sum(axis=1)
            df["RECENT_ENQUIRY"]      = (df["TOTAL_ENQUIRIES_1Y"] > 3).astype(int)
        return df

    def _interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-variable risk signals."""
        if "EXT_SOURCE_MEAN" in df.columns and "CREDIT_TO_INCOME" in df.columns:
            # Low bureau score + high credit = risk
            df["BUREAU_X_CREDIT_RATIO"] = (
                (1 - df["EXT_SOURCE_MEAN"].fillna(0.5)) * df["CREDIT_TO_INCOME"]
            )
        if "ANNUITY_TO_INCOME" in df.columns and "EXT_SOURCE_MEAN" in df.columns:
            df["STRESS_SCORE"] = (
                df["ANNUITY_TO_INCOME"].clip(0, 1) * 0.5 +
                (1 - df["EXT_SOURCE_MEAN"].fillna(0.5)) * 0.5
            )
        return df

    def get_feature_groups(self) -> dict:
        return {
            "Credit Ratios":   [c for c in self.feature_names_ if "RATIO" in c or "CREDIT_TO" in c],
            "Age/Employment":  [c for c in self.feature_names_ if "AGE" in c or "EMPLOY" in c or "YOUNG" in c or "SENIOR" in c],
            "External Scores": [c for c in self.feature_names_ if "EXT_SOURCE" in c],
            "Enquiries":       [c for c in self.feature_names_ if "ENQUIR" in c],
            "Interaction":     [c for c in self.feature_names_ if "STRESS" in c or "X_" in c],
            "Original":        [c for c in self.feature_names_ if c in [
                "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS",
            ]],
        }


# ─── Standalone test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=ACTIVE_DATASET)
    args = parser.parse_args()

    from src.data_loader import load_dataset
    df = load_dataset(args.dataset)

    fe = FeatureEngineer(args.dataset)
    df_feat = fe.fit_transform(df)

    print(f"\n=== Feature Engineering: {args.dataset} ===")
    print(f"Input columns:  {len(df.columns)}")
    print(f"Output columns: {len(df_feat.columns)}")
    print(f"Feature count:  {len(fe.get_feature_names())}")
    print(f"\nFeature groups:")
    for grp, cols in fe.get_feature_groups().items():
        print(f"  {grp:<20}: {len(cols)} features")
    print(f"\nSample feature stats:\n{df_feat[fe.get_feature_names()[:6]].describe().round(3)}")
