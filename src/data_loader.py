"""
src/data_loader.py
==================
Multi-dataset loader supporting 3 real publicly available credit datasets.

Datasets:
  1. UCI Credit Card Default  — auto-download (ucimlrepo → direct URL → GitHub mirror)
  2. Give Me Some Credit      — Kaggle API or manual download
  3. Home Credit Default Risk — Kaggle API or manual download

Each loader returns a validated, cleaned DataFrame with a standardised schema:
  - All columns renamed to consistent names
  - Target column always = config.TARGET_COLUMN (value 0/1)
  - ID column always = config.ID_COLUMN
  - Known data quality issues fixed
  - Missing value report logged

Usage:
    from src.data_loader import load_dataset
    df = load_dataset()          # uses ACTIVE_DATASET from config.py
    df = load_dataset("uci")     # explicit dataset selection
    df = load_dataset("gmc")
    df = load_dataset("homecredit")
"""

import io
import logging
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ACTIVE_DATASET, DATA_DIR, DATASET_CONFIGS,
    TARGET_COLUMN, ID_COLUMN,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset: str = None, force_download: bool = False) -> pd.DataFrame:
    """
    Load and return a real credit default dataset.

    Parameters
    ----------
    dataset : str, optional
        One of 'uci', 'gmc', 'homecredit'.
        Defaults to config.ACTIVE_DATASET.
    force_download : bool
        Re-download even if cached CSV exists.

    Returns
    -------
    pd.DataFrame
        Cleaned, validated DataFrame ready for feature engineering.
    """
    dataset = dataset or ACTIVE_DATASET
    cfg     = DATASET_CONFIGS.get(dataset)
    if cfg is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose: {list(DATASET_CONFIGS.keys())}")

    logger.info(f"{'='*60}")
    logger.info(f"Dataset : {cfg['name']}")
    logger.info(f"Target  : {cfg['target_col']}  |  Rows: {cfg['n_rows']:,}")
    logger.info(f"{'='*60}")

    loaders = {
        "uci":         UCILoader,
        "gmc":         GiveMeSomeCreditLoader,
        "homecredit":  HomeCreditLoader,
    }
    loader_cls = loaders[dataset]
    loader     = loader_cls(force_download=force_download)
    df         = loader.load()
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Base Loader
# ═════════════════════════════════════════════════════════════════════════════

class BaseLoader:
    """Common validation and logging for all loaders."""

    def __init__(self, force_download: bool = False):
        self.force_download = force_download

    def _validate(self, df: pd.DataFrame, target_col: str, min_rows: int = 1000) -> None:
        """Assert data contract."""
        assert len(df) >= min_rows, f"Only {len(df)} rows — too small."
        assert target_col in df.columns, f"Target '{target_col}' not found. Got: {df.columns.tolist()}"
        assert set(df[target_col].dropna().unique()).issubset({0, 1}), \
            f"Target must be binary 0/1. Found: {df[target_col].unique()}"
        null_pct = df.isnull().mean()
        high_null = null_pct[null_pct > 0.30]
        if len(high_null) > 0:
            logger.warning(f"High null rate columns (>30%):\n{high_null.round(3)}")
        logger.info("✓ Data validation passed.")

    def _log_summary(self, df: pd.DataFrame, target_col: str) -> None:
        n_def  = df[target_col].sum()
        rate   = n_def / len(df)
        nulls  = df.isnull().sum().sum()
        logger.info(
            f"Loaded | Rows: {len(df):,} | Cols: {len(df.columns)} | "
            f"Defaults: {n_def:,} ({rate:.2%}) | Total NaNs: {nulls:,}"
        )

    def _cached_csv(self, name: str) -> str:
        return os.path.join(DATA_DIR, f"{name}_clean.csv")

    def _try_kaggle_api(self, competition: str, output_dir: str) -> bool:
        """Attempt Kaggle CLI download. Returns True on success."""
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "download", "-c", competition, "-p", output_dir],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                logger.info(f"Kaggle API download succeeded: {competition}")
                return True
            else:
                logger.warning(f"Kaggle API failed: {result.stderr.strip()}")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Kaggle CLI not available: {e}")
            return False

    def _unzip_in_dir(self, directory: str) -> None:
        """Unzip all zip files found in a directory."""
        for zf in Path(directory).glob("*.zip"):
            logger.info(f"Unzipping {zf.name} ...")
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(directory)
            logger.info(f"Extracted to {directory}")


# ═════════════════════════════════════════════════════════════════════════════
# Dataset 1: UCI Credit Card Default
# ═════════════════════════════════════════════════════════════════════════════

class UCILoader(BaseLoader):
    """
    UCI Credit Card Default Dataset (Taiwan, 2005).
    Yeh, I. C., & Lien, C. H. (2009). Knowledge-Based Systems.

    30,000 credit card customers from a Taiwanese bank.
    Features: 6-month payment status, bill amounts, payment amounts.
    Target:   default_payment_next_month (1 = default, 0 = no default).

    Auto-download sources (tried in order):
      1. ucimlrepo Python package  (pip install ucimlrepo)
      2. Direct URL from UCI repository
      3. GitHub CSV mirror

    Manual fallback:
      https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
    """

    CACHE_CSV = os.path.join(DATA_DIR, "uci_credit_default_clean.csv")

    # Multiple download sources — tried in order
    DOWNLOAD_SOURCES = [
        {
            "method":      "ucimlrepo",
            "dataset_id":  350,
        },
        {
            "method":      "direct_url",
            "url":         "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
            "local_file":  os.path.join(DATA_DIR, "uci_raw.xls"),
            "format":      "xls",
        },
        {
            "method":      "github_csv",
            "url":         "https://raw.githubusercontent.com/dsavg/capstone_credit_card_default_prediction/master/data/UCI_Credit_Card.csv",
            "local_file":  os.path.join(DATA_DIR, "uci_raw_mirror.csv"),
            "format":      "csv",
        },
    ]

    def load(self) -> pd.DataFrame:
        if os.path.exists(self.CACHE_CSV) and not self.force_download:
            logger.info(f"Loading cached UCI dataset from {self.CACHE_CSV}")
            df = pd.read_csv(self.CACHE_CSV)
            self._validate(df, "default_payment_next_month")
            self._log_summary(df, "default_payment_next_month")
            return df

        df = self._download()
        df = self._clean(df)
        df.to_csv(self.CACHE_CSV, index=False)
        logger.info(f"UCI dataset cached at {self.CACHE_CSV}")
        self._validate(df, "default_payment_next_month")
        self._log_summary(df, "default_payment_next_month")
        return df

    def _download(self) -> pd.DataFrame:
        """Try each download source in order."""
        for source in self.DOWNLOAD_SOURCES:
            method = source["method"]
            logger.info(f"Trying download method: {method} ...")
            try:
                if method == "ucimlrepo":
                    df = self._download_ucimlrepo(source["dataset_id"])
                elif method == "direct_url":
                    df = self._download_url(source["url"], source["local_file"], source["format"])
                elif method == "github_csv":
                    df = self._download_url(source["url"], source["local_file"], source["format"])
                if df is not None and len(df) > 1000:
                    logger.info(f"✓ Download succeeded via: {method}")
                    return df
            except Exception as e:
                logger.warning(f"Method '{method}' failed: {e}")

        # All sources failed — check for existing manual download
        for filename in ["uci_raw.xls", "uci_raw_mirror.csv", "UCI_Credit_Card.csv",
                         "default of credit card clients.xls"]:
            path = os.path.join(DATA_DIR, filename)
            if os.path.exists(path):
                logger.info(f"Found manual download: {path}")
                return self._read_file(path)

        raise FileNotFoundError(
            "\n\n" + "="*60 + "\n"
            "UCI DATASET NOT FOUND — MANUAL DOWNLOAD REQUIRED\n"
            "="*60 + "\n"
            "1. Go to: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients\n"
            "2. Click the Download button\n"
            "3. Save the file to: data/\n"
            "4. Re-run: python pipeline.py\n"
            "="*60 + "\n\n"
            "OR install ucimlrepo for auto-download:\n"
            "   pip install ucimlrepo\n"
        )

    def _download_ucimlrepo(self, dataset_id: int) -> Optional[pd.DataFrame]:
        """Download via official ucimlrepo package."""
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            logger.info("ucimlrepo not installed. Try: pip install ucimlrepo")
            return None

        logger.info(f"Fetching UCI dataset id={dataset_id} via ucimlrepo ...")
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        logger.info(f"ucimlrepo: fetched {len(df)} rows, {len(df.columns)} columns")
        return df

    def _download_url(self, url: str, local_file: str, fmt: str) -> Optional[pd.DataFrame]:
        """Download from a direct URL."""
        logger.info(f"Downloading from URL: {url[:80]}...")
        urllib.request.urlretrieve(url, local_file)
        logger.info(f"Saved to {local_file}")
        return self._read_file(local_file)

    def _read_file(self, path: str) -> pd.DataFrame:
        """Read XLS or CSV into DataFrame."""
        ext = Path(path).suffix.lower()
        if ext in [".xls", ".xlsx"]:
            # UCI XLS has a double header — row 0 is description, row 1 is actual headers
            try:
                df = pd.read_excel(path, header=1, engine="xlrd")
                if len(df) < 1000:
                    df = pd.read_excel(path, header=0, engine="xlrd")
            except Exception:
                df = pd.read_excel(path, header=1)
        else:
            df = pd.read_csv(path)
        logger.info(f"Read {len(df)} rows from {path}")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names and fix known data quality issues."""
        df.columns = [str(c).strip() for c in df.columns]

        # ── Handle ucimlrepo generic column names (X1..X23, Y) ────────
        # The ucimlrepo package returns features as X1-X23 and target as Y
        # instead of the real UCI column names. Map them back.
        if "X1" in df.columns and "Y" in df.columns and len(df.columns) <= 25:
            ucimlrepo_rename = {
                "X1":  "LIMIT_BAL",
                "X2":  "SEX",
                "X3":  "EDUCATION",
                "X4":  "MARRIAGE",
                "X5":  "AGE",
                "X6":  "PAY_1",      # Most recent month (Sep)
                "X7":  "PAY_2",
                "X8":  "PAY_3",
                "X9":  "PAY_4",
                "X10": "PAY_5",
                "X11": "PAY_6",
                "X12": "BILL_AMT1",
                "X13": "BILL_AMT2",
                "X14": "BILL_AMT3",
                "X15": "BILL_AMT4",
                "X16": "BILL_AMT5",
                "X17": "BILL_AMT6",
                "X18": "PAY_AMT1",
                "X19": "PAY_AMT2",
                "X20": "PAY_AMT3",
                "X21": "PAY_AMT4",
                "X22": "PAY_AMT5",
                "X23": "PAY_AMT6",
                "Y":   "default_payment_next_month",
            }
            df = df.rename(columns=ucimlrepo_rename)
            logger.info("Mapped ucimlrepo generic column names (X1..X23, Y) → real UCI names")

        # Unify various naming conventions seen across download sources
        rename = {
            "default.payment.next.month": "default_payment_next_month",
            "default payment next month":  "default_payment_next_month",
            "PAY_0":  "PAY_1",    # UCI incorrectly names most recent month PAY_0
        }
        df = df.rename(columns=rename)

        # Ensure ID column exists
        if "ID" not in df.columns:
            df.insert(0, "ID", range(1, len(df) + 1))

        # ── Fix known categorical encoding issues ──────────────────────
        # EDUCATION: 0, 5, 6 are undocumented categories → map to 4 (Other)
        if "EDUCATION" in df.columns:
            df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

        # MARRIAGE: 0 is undocumented → map to 3 (Other)
        if "MARRIAGE" in df.columns:
            df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

        # ── Remove known duplicate/index columns ──────────────────────
        drop_cols = [c for c in df.columns if c in ["Unnamed: 0", "index"]]
        df = df.drop(columns=drop_cols, errors="ignore")

        # ── Type enforcement ──────────────────────────────────────────
        df["default_payment_next_month"] = df["default_payment_next_month"].astype(int)

        # ── Drop rows with all-NaN values ─────────────────────────────
        df = df.dropna(how="all").reset_index(drop=True)

        logger.info(f"UCI cleaning complete | {len(df):,} rows | {len(df.columns)} columns")
        return df


# ═════════════════════════════════════════════════════════════════════════════
# Dataset 2: Give Me Some Credit (Kaggle)
# ═════════════════════════════════════════════════════════════════════════════

class GiveMeSomeCreditLoader(BaseLoader):
    """
    Give Me Some Credit — Kaggle Competition 2011.
    https://www.kaggle.com/c/GiveMeSomeCredit

    150,000 US borrowers.
    Target: SeriousDlqin2yrs — serious delinquency (90+ days) in 2 years.

    Features:
      RevolvingUtilizationOfUnsecuredLines — revolving credit utilization
      age                                  — age of borrower
      NumberOfTime30-59DaysPastDueNotWorse — 30-59 day delinquencies
      DebtRatio                            — monthly debt / monthly income
      MonthlyIncome                        — self-reported monthly income
      NumberOfOpenCreditLinesAndLoans      — open credit lines
      NumberOfTimes90DaysLate              — 90+ day delinquencies
      NumberRealEstateLoansOrLines         — real estate loans
      NumberOfTime60-89DaysPastDueNotWorse — 60-89 day delinquencies
      NumberOfDependents                   — number of dependents

    Download:
      Auto: kaggle competitions download -c GiveMeSomeCredit
      Manual: https://www.kaggle.com/c/GiveMeSomeCredit/data → cs-training.csv
    """

    CACHE_CSV     = os.path.join(DATA_DIR, "gmc_clean.csv")
    RAW_FILENAMES = ["cs-training.csv", "GiveMeSomeCredit/cs-training.csv"]

    def load(self) -> pd.DataFrame:
        if os.path.exists(self.CACHE_CSV) and not self.force_download:
            logger.info(f"Loading cached GMC dataset from {self.CACHE_CSV}")
            df = pd.read_csv(self.CACHE_CSV)
            self._validate(df, "SeriousDlqin2yrs")
            self._log_summary(df, "SeriousDlqin2yrs")
            return df

        raw_df = self._get_raw()
        df     = self._clean(raw_df)
        df.to_csv(self.CACHE_CSV, index=False)
        logger.info(f"GMC dataset cached at {self.CACHE_CSV}")
        self._validate(df, "SeriousDlqin2yrs")
        self._log_summary(df, "SeriousDlqin2yrs")
        return df

    def _get_raw(self) -> pd.DataFrame:
        """Try Kaggle API → look for existing files → raise with instructions."""
        # 1. Try Kaggle API
        cfg = DATASET_CONFIGS["gmc"]
        if self._try_kaggle_api(cfg["kaggle_competition"], DATA_DIR):
            self._unzip_in_dir(DATA_DIR)

        # 2. Look for existing files
        for filename in self.RAW_FILENAMES:
            path = os.path.join(DATA_DIR, filename)
            if os.path.exists(path):
                logger.info(f"Found raw file: {path}")
                return pd.read_csv(path, index_col=0)

        raise FileNotFoundError(
            "\n\n" + DATASET_CONFIGS["gmc"]["manual_instructions"]
        )

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean GMC-specific issues."""
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # Ensure ID
        if "ID" not in df.columns:
            df.insert(0, "ID", range(1, len(df) + 1))

        # ── Handle known outliers ──────────────────────────────────────
        # RevolvingUtilizationOfUnsecuredLines: cap at 1.5 (>1 means over-limit)
        if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
            out = (df["RevolvingUtilizationOfUnsecuredLines"] > 1.5).sum()
            df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1.5)
            logger.info(f"Clipped {out:,} extreme utilization values (>1.5)")

        # DebtRatio: cap at 10 (>1 already means insolvent; extreme values are data errors)
        if "DebtRatio" in df.columns:
            out = (df["DebtRatio"] > 10).sum()
            df["DebtRatio"] = df["DebtRatio"].clip(0, 10)
            logger.info(f"Clipped {out:,} extreme debt ratio values (>10)")

        # age: 0 is clearly erroneous; fill with median
        if "age" in df.columns:
            age_zero = (df["age"] == 0).sum()
            if age_zero > 0:
                df.loc[df["age"] == 0, "age"] = np.nan
                df["age"] = df["age"].fillna(df["age"].median())
                logger.info(f"Replaced {age_zero} zero-age values with median")

        # MonthlyIncome: has NaNs — fill with median
        if "MonthlyIncome" in df.columns:
            n_null = df["MonthlyIncome"].isnull().sum()
            df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
            logger.info(f"Imputed {n_null:,} missing MonthlyIncome values with median")

        # NumberOfDependents: has NaNs — fill with 0 (common assumption)
        if "NumberOfDependents" in df.columns:
            n_null = df["NumberOfDependents"].isnull().sum()
            df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)
            logger.info(f"Imputed {n_null:,} missing NumberOfDependents with 0")

        # ── Type enforcement ───────────────────────────────────────────
        df["SeriousDlqin2yrs"] = df["SeriousDlqin2yrs"].astype(int)

        logger.info(f"GMC cleaning complete | {len(df):,} rows | {len(df.columns)} columns")
        return df


# ═════════════════════════════════════════════════════════════════════════════
# Dataset 3: Home Credit Default Risk (Kaggle)
# ═════════════════════════════════════════════════════════════════════════════

class HomeCreditLoader(BaseLoader):
    """
    Home Credit Default Risk — Kaggle Competition 2018.
    https://www.kaggle.com/c/home-credit-default-risk

    307,511 loan applications (application_train.csv).
    Target: TARGET — loan default (1) or not (0).
    122 features covering application details, credit bureau data.

    NOTE: For a richer multi-table analysis, the competition also includes
    bureau.csv, previous_application.csv etc. This loader uses application_train.csv
    (the main table) which is sufficient for a strong model on its own.

    Download:
      Auto: kaggle competitions download -c home-credit-default-risk
      Manual: https://www.kaggle.com/c/home-credit-default-risk/data
    """

    CACHE_CSV     = os.path.join(DATA_DIR, "homecredit_clean.csv")
    RAW_FILENAMES = [
        "application_train.csv",
        "home-credit-default-risk/application_train.csv",
    ]

    def load(self) -> pd.DataFrame:
        if os.path.exists(self.CACHE_CSV) and not self.force_download:
            logger.info(f"Loading cached Home Credit dataset from {self.CACHE_CSV}")
            df = pd.read_csv(self.CACHE_CSV)
            self._validate(df, "TARGET")
            self._log_summary(df, "TARGET")
            return df

        raw_df = self._get_raw()
        df     = self._clean(raw_df)
        df.to_csv(self.CACHE_CSV, index=False)
        logger.info(f"Home Credit dataset cached at {self.CACHE_CSV}")
        self._validate(df, "TARGET")
        self._log_summary(df, "TARGET")
        return df

    def _get_raw(self) -> pd.DataFrame:
        cfg = DATASET_CONFIGS["homecredit"]
        if self._try_kaggle_api(cfg["kaggle_competition"], DATA_DIR):
            self._unzip_in_dir(DATA_DIR)

        for filename in self.RAW_FILENAMES:
            path = os.path.join(DATA_DIR, filename)
            if os.path.exists(path):
                logger.info(f"Found raw file: {path} — loading (307k rows, may take ~10s)")
                return pd.read_csv(path)

        raise FileNotFoundError(
            "\n\n" + DATASET_CONFIGS["homecredit"]["manual_instructions"]
        )

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Home Credit application_train.csv."""
        df = df.copy()

        # Rename ID for consistency
        df = df.rename(columns={"SK_ID_CURR": "ID"})

        # ── Days columns: convert negative days to positive years ──────
        # DAYS_BIRTH, DAYS_EMPLOYED etc are stored as negative integers
        days_cols = [c for c in df.columns if c.startswith("DAYS_")]
        for col in days_cols:
            df[col] = df[col].abs()

        # DAYS_EMPLOYED: 365243 is a sentinel for "pensioner/retired"
        if "DAYS_EMPLOYED" in df.columns:
            sentinel = (df["DAYS_EMPLOYED"] == 365243).sum()
            df["DAYS_EMPLOYED_PENSIONER"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].fillna(df["DAYS_EMPLOYED"].median())
            logger.info(f"DAYS_EMPLOYED: handled {sentinel:,} pensioner sentinels")

        # ── One-hot encode key categorical columns ─────────────────────
        cat_cols = [c for c in df.columns if df[c].dtype == "object"]
        low_cardinality = [c for c in cat_cols if df[c].nunique() <= 10]
        df = pd.get_dummies(df, columns=low_cardinality, drop_first=True, dtype=int)
        logger.info(f"One-hot encoded {len(low_cardinality)} categorical columns")

        # ── Fill numeric NaNs with median ──────────────────────────────
        num_cols  = df.select_dtypes(include=[np.number]).columns
        null_cols = [c for c in num_cols if df[c].isnull().any()]
        for col in null_cols:
            df[col] = df[col].fillna(df[col].median())
        logger.info(f"Imputed {len(null_cols)} columns with median")

        # ── Type enforcement ───────────────────────────────────────────
        df["TARGET"] = df["TARGET"].astype(int)

        logger.info(f"Home Credit cleaning complete | {len(df):,} rows | {len(df.columns)} columns")
        return df


# ═════════════════════════════════════════════════════════════════════════════
# Standalone test
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Test data loader")
    parser.add_argument("--dataset", default=ACTIVE_DATASET,
                        choices=["uci", "gmc", "homecredit"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = load_dataset(args.dataset, force_download=args.force)
    print(f"\n--- Dataset: {args.dataset} ---")
    print(f"Shape  : {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    print(f"\nNull summary:\n{df.isnull().sum()[df.isnull().sum()>0]}")
