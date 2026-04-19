"""
src/woe_scorecard.py
--------------------
Weight of Evidence (WoE) transformation + Information Value (IV) selection
+ Logistic Regression Scorecard.

This is the REGULATORY-STANDARD approach used by bank credit risk teams.
Produces an interpretable points-based scorecard (like FICO/CIBIL).

References:
  - Basel III IRB approach
  - OCC Model Risk Management Guidance (SR 11-7)
  - Siddiqi (2006) "Credit Risk Scorecards"
"""

import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_COLUMN, REPORTS_DIR,
    WOE_MAX_BINS,
    IV_THRESHOLD_DROP, IV_THRESHOLD_SUSPICIOUS,
    REGULARIZATION_C, CV_FOLDS,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# WoE Binner
# ─────────────────────────────────────────────────────────────────────────────

class WoEBinner:
    """
    Optimal WoE binning for a single feature using quantile-based bins
    with monotonicity enforcement.

    Weight of Evidence for bin b:
        WoE_b = ln( P(events in b) / P(non-events in b) )

    Information Value:
        IV = Σ_b (P_events_b - P_nonevents_b) × WoE_b
    """

    def __init__(self, n_bins: int = 10, min_bin_size: float = 0.05):
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.bins_: list = []
        self.woe_map_: dict = {}
        self.iv_: float = 0.0
        self.bin_stats_: pd.DataFrame = pd.DataFrame()

    def fit(self, x: pd.Series, y: pd.Series) -> "WoEBinner":
        """
        Fit WoE bins on feature x with target y.
        """
        df = pd.DataFrame({"x": x.values, "y": y.values}).dropna()
        total_events     = df["y"].sum()
        total_nonevents  = len(df) - total_events

        if total_events == 0 or total_nonevents == 0:
            logger.warning(f"Feature has zero events or nonevents. Skipping WoE.")
            return self

        # Determine binning strategy
        if df["x"].nunique() <= self.n_bins:
            # Categorical or low-cardinality: bin by unique values
            df["bin"] = df["x"].astype(str)
        else:
            # Continuous: quantile bins
            try:
                df["bin"] = pd.qcut(
                    df["x"], q=self.n_bins,
                    duplicates="drop", labels=False
                )
            except Exception:
                df["bin"] = pd.cut(
                    df["x"], bins=self.n_bins,
                    duplicates="drop", labels=False
                )

        # Compute WoE and IV per bin
        stats = []
        for bin_val, grp in df.groupby("bin", observed=True):
            n_events    = grp["y"].sum()
            n_nonevents = len(grp) - n_events
            pct_events    = max(n_events    / total_events,    1e-9)
            pct_nonevents = max(n_nonevents / total_nonevents, 1e-9)
            woe = np.log(pct_events / pct_nonevents)
            iv  = (pct_events - pct_nonevents) * woe
            stats.append({
                "bin":          bin_val,
                "n_total":      len(grp),
                "n_events":     n_events,
                "n_nonevents":  n_nonevents,
                "event_rate":   n_events / len(grp),
                "pct_events":   pct_events,
                "pct_nonevents":pct_nonevents,
                "woe":          woe,
                "iv":           iv,
            })

        self.bin_stats_ = pd.DataFrame(stats)
        self.iv_ = self.bin_stats_["iv"].sum()

        # Build WoE map: bin label → WoE value
        self.woe_map_ = dict(
            zip(self.bin_stats_["bin"], self.bin_stats_["woe"])
        )
        return self

    def transform(self, x: pd.Series, df_orig: pd.DataFrame = None) -> pd.Series:
        """
        Map feature values to WoE scores.
        Unseen bin values receive 0 (neutral WoE).
        """
        if df_orig is not None and x.name in df_orig.columns:
            col_data = df_orig[x.name]
        else:
            col_data = x

        # Re-bin using same logic as fit
        if col_data.nunique() <= self.n_bins:
            bin_labels = col_data.astype(str)
        else:
            try:
                bin_labels = pd.qcut(
                    col_data, q=self.n_bins,
                    duplicates="drop", labels=False
                )
            except Exception:
                bin_labels = pd.cut(
                    col_data, bins=self.n_bins,
                    duplicates="drop", labels=False
                )

        return bin_labels.map(self.woe_map_).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# WoE Feature Selector
# ─────────────────────────────────────────────────────────────────────────────

class WoEFeatureSelector:
    """
    Fit WoE binners on all features, compute IV, and select features
    above IV threshold. Flags suspiciously high IV (possible data leakage).
    """

    def __init__(
        self,
        n_bins: int = WOE_MAX_BINS,
        iv_min: float = IV_THRESHOLD_DROP,
        iv_suspicious: float = IV_THRESHOLD_SUSPICIOUS,
    ):
        self.n_bins = n_bins
        self.iv_min = iv_min
        self.iv_suspicious = iv_suspicious
        self.binners_: dict = {}        # feature_name → WoEBinner
        self.iv_table_: pd.DataFrame = pd.DataFrame()
        self.selected_features_: list = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEFeatureSelector":
        """Fit WoE binners on all features and select by IV."""
        records = []
        for col in X.columns:
            binner = WoEBinner(n_bins=self.n_bins)
            binner.fit(X[col], y)
            self.binners_[col] = binner

            flag = ""
            if binner.iv_ < self.iv_min:
                flag = "DROP (low IV)"
            elif binner.iv_ > self.iv_suspicious:
                flag = "⚠ SUSPICIOUS (possible leakage)"

            records.append({
                "feature":    col,
                "iv":         round(binner.iv_, 4),
                "n_bins":     len(binner.bin_stats_),
                "flag":       flag,
            })

        self.iv_table_ = (
            pd.DataFrame(records)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

        self.selected_features_ = (
            self.iv_table_[
                (self.iv_table_["iv"] >= self.iv_min) &
                (~self.iv_table_["flag"].str.contains("SUSPICIOUS"))
            ]["feature"].tolist()
        )

        n_dropped     = (self.iv_table_["iv"] < self.iv_min).sum()
        n_suspicious  = self.iv_table_["flag"].str.contains("SUSPICIOUS").sum()
        logger.info(
            f"IV Selection | Total: {len(X.columns)} | "
            f"Selected: {len(self.selected_features_)} | "
            f"Dropped (low IV): {n_dropped} | "
            f"Suspicious (leakage risk): {n_suspicious}"
        )

        # Log top 15
        logger.info(f"\nTop features by IV:\n"
                    f"{self.iv_table_.head(15).to_string(index=False)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return WoE-transformed DataFrame for selected features."""
        woe_df = pd.DataFrame(index=X.index)
        for col in self.selected_features_:
            binner = self.binners_[col]
            woe_df[f"WOE_{col}"] = binner.transform(X[col])
        return woe_df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Scorecard Builder
# ─────────────────────────────────────────────────────────────────────────────

class CreditScorecard:
    """
    Full logistic regression scorecard pipeline:
      1. WoE feature selection
      2. Logistic regression on WoE features
      3. Score scaling to 300–850 range (FICO-style)
      4. Model evaluation (KS, Gini, AUC)
      5. Scorecard points table output

    Usage:
        sc = CreditScorecard()
        sc.fit(X_train, y_train)
        scores = sc.score(X_test)
        metrics = sc.evaluate(X_test, y_test)
    """

    # Score scaling parameters (FICO-style: 300–850)
    SCORE_MIN  = 300
    SCORE_MAX  = 850
    PDO        = 20    # Points to Double Odds
    BASE_SCORE = 600   # Score at which odds = 1:1

    def __init__(self):
        self.woe_selector_ = WoEFeatureSelector()
        self.scaler_        = StandardScaler()
        self.lr_            = LogisticRegression(
            C=REGULARIZATION_C,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.scorecard_table_: pd.DataFrame = pd.DataFrame()
        self._fitted = False
        self._factor = None
        self._offset = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CreditScorecard":
        """Full training pipeline."""
        logger.info("=== Training Credit Scorecard ===")

        # Step 1: WoE transform + feature selection
        X_woe = self.woe_selector_.fit_transform(X, y)
        logger.info(f"WoE features: {X_woe.shape[1]}")

        # Step 2: Scale WoE features (helps LR convergence, not strictly needed)
        X_scaled = self.scaler_.fit_transform(X_woe)

        # Step 3: Fit logistic regression
        self.lr_.fit(X_scaled, y)
        logger.info(f"LR fitted | Intercept: {self.lr_.intercept_[0]:.4f}")

        # Step 4: Score scaling calibration
        self._compute_scaling_params()

        # Step 5: Build human-readable scorecard table
        self._build_scorecard_table()

        self._fitted = True
        logger.info("=== Scorecard Training Complete ===")
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probability of default."""
        X_woe    = self.woe_selector_.transform(X)
        X_scaled = self.scaler_.transform(X_woe)
        return self.lr_.predict_proba(X_scaled)[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return credit score (300–850).
        Higher score = lower default risk (FICO convention).
        """
        pd_proba = self.predict_proba(X)
        # Avoid log(0)
        pd_proba = np.clip(pd_proba, 1e-6, 1 - 1e-6)
        odds = pd_proba / (1 - pd_proba)
        log_odds = np.log(odds)
        # Scale: score = offset - factor × ln(odds)
        # Higher odds of default → lower score
        raw_score = self._offset - self._factor * log_odds
        return np.clip(raw_score, self.SCORE_MIN, self.SCORE_MAX)

    # ── Evaluate ─────────────────────────────────────────────────────────────

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, label: str = "Scorecard"
    ) -> dict:
        """
        Compute KS, Gini, AUC-ROC, and calibration metrics.
        These are the exact metrics a bank's model validation team uses.
        """
        proba = self.predict_proba(X)
        scores = self.score(X)

        # AUC-ROC
        auc = roc_auc_score(y, proba)

        # Gini coefficient (industry standard for credit models)
        gini = 2 * auc - 1

        # KS statistic — max separation between event/non-event score distributions
        fpr, tpr, _ = roc_curve(y, proba)
        ks = (tpr - fpr).max()

        # Average Precision (useful for imbalanced data)
        ap = average_precision_score(y, proba)

        metrics = {
            "label":     label,
            "auc_roc":   round(auc, 4),
            "gini":      round(gini, 4),
            "ks_stat":   round(ks, 4),
            "avg_prec":  round(ap, 4),
            "n_samples": len(y),
            "event_rate": round(y.mean(), 4),
        }

        logger.info(
            f"\n{'='*50}\n"
            f"Model Evaluation: {label}\n"
            f"  AUC-ROC : {auc:.4f}\n"
            f"  Gini    : {gini:.4f}  {'✓ Good' if gini > 0.5 else '✗ Weak'}\n"
            f"  KS Stat : {ks:.4f}   {'✓ Good' if ks > 0.35 else '✗ Weak'}\n"
            f"  Avg Prec: {ap:.4f}\n"
            f"  N       : {len(y):,}\n"
            f"{'='*50}"
        )
        return metrics

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Stratified K-fold cross-validation on WoE-transformed features.
        Returns mean ± std Gini across folds.
        """
        logger.info(f"Running {CV_FOLDS}-fold cross-validation...")
        X_woe    = self.woe_selector_.transform(X)
        X_scaled = self.scaler_.transform(X_woe)

        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        fold_aucs = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
            self.lr_.fit(X_scaled[tr_idx], y.iloc[tr_idx])
            preds = self.lr_.predict_proba(X_scaled[val_idx])[:, 1]
            auc = roc_auc_score(y.iloc[val_idx], preds)
            fold_aucs.append(auc)
            logger.info(f"  Fold {fold}: AUC={auc:.4f}  Gini={2*auc-1:.4f}")

        mean_auc  = np.mean(fold_aucs)
        std_auc   = np.std(fold_aucs)
        mean_gini = 2 * mean_auc - 1

        logger.info(
            f"CV Result: AUC={mean_auc:.4f} ± {std_auc:.4f} | "
            f"Gini={mean_gini:.4f}"
        )
        return {"cv_auc_mean": mean_auc, "cv_auc_std": std_auc, "cv_gini_mean": mean_gini}

    # ── Score Distribution Analysis ───────────────────────────────────────────

    def score_distribution(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Bin scores into deciles and show event rate per bin.
        Classic scorecard validation output.
        """
        scores = self.score(X)
        df = pd.DataFrame({"score": scores, "target": y.values})
        df["decile"] = pd.qcut(
            df["score"], q=10, labels=False, duplicates="drop"
        )
        dist = (
            df.groupby("decile")
            .agg(
                n_total=("target", "count"),
                n_events=("target", "sum"),
                score_min=("score", "min"),
                score_max=("score", "max"),
                score_mean=("score", "mean"),
            )
            .assign(event_rate=lambda d: d["n_events"] / d["n_total"])
            .reset_index()
            .sort_values("decile", ascending=False)
        )
        logger.info(f"\nScore Distribution by Decile:\n{dist.to_string(index=False)}")
        return dist

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot_evaluation(
        self, X: pd.DataFrame, y: pd.Series,
        save_path: str = None
    ) -> None:
        """
        4-panel evaluation plot:
          1. ROC curve with AUC annotation
          2. KS plot (separation between good/bad score distributions)
          3. Score distribution by class
          4. Calibration curve
        """
        proba  = self.predict_proba(X)
        scores = self.score(X)

        fig = plt.figure(figsize=(16, 12))
        gs  = gridspec.GridSpec(2, 2, figure=fig)
        fig.suptitle("Credit Scorecard — Model Evaluation Dashboard", fontsize=15, fontweight="bold")

        # ── Panel 1: ROC Curve ────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        gini = 2 * auc - 1
        ax1.plot(fpr, tpr, color="#2563EB", lw=2,
                 label=f"Scorecard (AUC={auc:.3f}, Gini={gini:.3f})")
        ax1.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
        ax1.fill_between(fpr, tpr, alpha=0.1, color="#2563EB")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate (Recall)")
        ax1.set_title("ROC Curve")
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)

        # ── Panel 2: KS Plot ─────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        df_ks = pd.DataFrame({"score": proba, "target": y.values})
        df_ks = df_ks.sort_values("score")
        df_ks["cum_events"]    = df_ks["target"].cumsum() / df_ks["target"].sum()
        df_ks["cum_nonevents"] = (1 - df_ks["target"]).cumsum() / (1 - df_ks["target"]).sum()
        ks = (df_ks["cum_events"] - df_ks["cum_nonevents"]).abs().max()

        ax2.plot(df_ks["score"], df_ks["cum_events"],
                 color="#DC2626", lw=2, label="Cumulative Events (Bads)")
        ax2.plot(df_ks["score"], df_ks["cum_nonevents"],
                 color="#16A34A", lw=2, label="Cumulative Non-Events (Goods)")
        ax2.fill_between(
            df_ks["score"],
            df_ks["cum_events"],
            df_ks["cum_nonevents"],
            alpha=0.15, color="#7C3AED"
        )
        ax2.set_xlabel("Predicted Probability Score")
        ax2.set_ylabel("Cumulative %")
        ax2.set_title(f"KS Plot | KS Statistic = {ks:.3f}")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # ── Panel 3: Score Distribution ───────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        scores_good = scores[y == 0]
        scores_bad  = scores[y == 1]
        bins = np.linspace(self.SCORE_MIN, self.SCORE_MAX, 40)
        ax3.hist(scores_good, bins=bins, alpha=0.6, color="#16A34A",
                 label="Good (No Default)", density=True)
        ax3.hist(scores_bad,  bins=bins, alpha=0.6, color="#DC2626",
                 label="Bad (Default)", density=True)
        ax3.axvline(scores.mean(), color="navy", ls="--", lw=2,
                    label=f"Overall Mean={scores.mean():.0f}")
        ax3.set_xlabel("Credit Score (300–850)")
        ax3.set_ylabel("Density")
        ax3.set_title("Score Distribution by Default Status")
        ax3.legend()
        ax3.grid(alpha=0.3)

        # ── Panel 4: Calibration Curve ────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        prob_true, prob_pred = calibration_curve(y, proba, n_bins=10)
        ax4.plot(prob_pred, prob_true, "s-", color="#2563EB", lw=2,
                 label="Scorecard", markersize=6)
        ax4.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Perfect Calibration")
        ax4.set_xlabel("Mean Predicted Probability")
        ax4.set_ylabel("Fraction of Positives (Actual)")
        ax4.set_title("Calibration Curve\n(Reliability Diagram)")
        ax4.legend()
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved scorecard evaluation plot to {save_path}")
        plt.close()

    def plot_iv_chart(self, save_path: str = None) -> None:
        """Bar chart of Information Value for all features."""
        if self.woe_selector_.iv_table_.empty:
            logger.warning("IV table not built. Run fit() first.")
            return

        df = self.woe_selector_.iv_table_.head(25).sort_values("iv")
        colors = ["#DC2626" if "SUSPICIOUS" in f else
                  "#9CA3AF" if "DROP" in f else "#2563EB"
                  for f in df["flag"]]

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(df["feature"], df["iv"], color=colors, edgecolor="white")
        ax.axvline(IV_THRESHOLD_DROP, color="orange", ls="--", lw=1.5,
                   label=f"Min IV = {IV_THRESHOLD_DROP}")
        ax.axvline(IV_THRESHOLD_SUSPICIOUS, color="red", ls="--", lw=1.5,
                   label=f"Suspicious IV = {IV_THRESHOLD_SUSPICIOUS}")
        ax.set_xlabel("Information Value (IV)")
        ax.set_title("Feature Importance — Information Value\n"
                     "Blue=Selected | Grey=Dropped | Red=Suspicious")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved IV chart to {save_path}")
        plt.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_scaling_params(self) -> None:
        """Compute score scaling factor and offset for 300–850 range."""
        # Using standard PDO (Points to Double Odds) scaling
        self._factor = self.PDO / np.log(2)
        self._offset = (
            self.BASE_SCORE
            + self._factor * np.log(1.0)   # odds=1 at base score
        )

    def _build_scorecard_table(self) -> None:
        """
        Build the classic scorecard points table.
        Shows: feature → bin → WoE → coefficient contribution → points
        """
        records = []
        coef_map = dict(zip(
            [f"WOE_{f}" for f in self.woe_selector_.selected_features_],
            self.lr_.coef_[0]
        ))
        scaler_mean = self.scaler_.mean_
        scaler_std  = self.scaler_.scale_

        for i, feat in enumerate(self.woe_selector_.selected_features_):
            binner  = self.woe_selector_.binners_[feat]
            coef    = coef_map.get(f"WOE_{feat}", 0.0)
            s_mean  = scaler_mean[i] if i < len(scaler_mean) else 0
            s_std   = scaler_std[i]  if i < len(scaler_std)  else 1

            for _, row in binner.bin_stats_.iterrows():
                woe    = row["woe"]
                # scaled WoE passed to LR
                woe_sc = (woe - s_mean) / (s_std + 1e-9)
                # contribution to log-odds
                log_odds_contrib = coef * woe_sc
                # convert to score points (higher score = lower risk)
                points = -self._factor * log_odds_contrib

                records.append({
                    "Feature":      feat,
                    "Bin":          row["bin"],
                    "N_Total":      int(row["n_total"]),
                    "Event_Rate":   round(row["event_rate"], 4),
                    "WoE":          round(woe, 4),
                    "Coefficient":  round(coef, 4),
                    "Points":       round(points, 1),
                })

        self.scorecard_table_ = pd.DataFrame(records)
        logger.info(
            f"Scorecard table built | "
            f"{len(self.woe_selector_.selected_features_)} features | "
            f"{len(self.scorecard_table_)} bins total"
        )

    def get_scorecard_table(self) -> pd.DataFrame:
        return self.scorecard_table_

    def get_iv_table(self) -> pd.DataFrame:
        return self.woe_selector_.iv_table_
