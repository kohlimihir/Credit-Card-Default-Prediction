"""
src/monitoring.py
------------------
Model Governance & Monitoring — the industrial differentiator.

Implements:
  1. Population Stability Index (PSI) — is the input data drifting?
  2. Gini stability over time — is model discrimination degrading?
  3. Out-of-Time (OOT) backtesting — how does the model hold up on future data?
  4. Score distribution shift detection
  5. Model Card generation (structured documentation)
  6. Bias audit — does the model perform equally across demographic groups?

These are the exact checks a Model Risk Management (MRM) team runs
under Basel III and OCC SR 11-7 guidance.
"""

import logging
import os
import sys
import json
import warnings
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_COLUMN, REPORTS_DIR,
    PSI_THRESHOLDS, GINI_DEGRADATION_ALERT, MIN_ACCEPTABLE_GINI,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Population Stability Index (PSI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    feature_name: str = "feature",
) -> dict:
    """
    Compute PSI between a reference (training) distribution
    and a monitoring (production) distribution.

    PSI Formula:
        PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

    Thresholds (industry standard):
        PSI < 0.10  → No significant shift (stable)
        PSI 0.10–0.25 → Moderate shift (monitor closely)
        PSI > 0.25  → Significant shift (trigger model review / rebuild)
    """
    # Build bins from training data
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # Remove duplicate bin edges
    bins[0]  = -np.inf
    bins[-1] =  np.inf

    # Count observations per bin
    exp_counts = np.histogram(expected, bins=bins)[0]
    act_counts = np.histogram(actual,   bins=bins)[0]

    # Convert to proportions (avoid division by zero)
    exp_pct = np.where(exp_counts > 0, exp_counts / len(expected), 1e-6)
    act_pct = np.where(act_counts > 0, act_counts / len(actual),   1e-6)

    # PSI per bin
    psi_bins = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    psi      = psi_bins.sum()

    # Interpret
    if psi < PSI_THRESHOLDS["stable"]:
        status = "🟢 STABLE"
    elif psi < PSI_THRESHOLDS["moderate"]:
        status = "🟡 MODERATE — Monitor"
    else:
        status = "🔴 SIGNIFICANT — Review Model"

    result = {
        "feature":   feature_name,
        "psi":       round(psi, 5),
        "status":    status,
        "n_bins":    len(bins) - 1,
        "n_expected": len(expected),
        "n_actual":  len(actual),
        "bin_details": pd.DataFrame({
            "bin":       range(1, len(exp_pct) + 1),
            "exp_pct":   exp_pct.round(4),
            "act_pct":   act_pct.round(4),
            "psi_bin":   psi_bins.round(5),
        }),
    }

    logger.info(f"PSI [{feature_name}]: {psi:.4f}  →  {status}")
    return result


def compute_feature_psi_report(
    X_train: pd.DataFrame,
    X_monitor: pd.DataFrame,
    feature_list: List[str] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute PSI for all features between training and monitoring data.
    Returns a sorted report table.
    """
    feature_list = feature_list or list(X_train.select_dtypes(include=[np.number]).columns)
    records = []

    for feat in feature_list:
        if feat not in X_monitor.columns:
            continue
        result = compute_psi(
            X_train[feat].values,
            X_monitor[feat].values,
            n_bins=n_bins,
            feature_name=feat,
        )
        records.append({
            "feature": feat,
            "psi":     result["psi"],
            "status":  result["status"],
        })

    report = pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)
    n_flag = (report["psi"] >= PSI_THRESHOLDS["moderate"]).sum()
    logger.info(
        f"PSI Report Complete | {len(report)} features | "
        f"{n_flag} flagged (PSI ≥ {PSI_THRESHOLDS['moderate']})"
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Gini Stability Over Time
# ─────────────────────────────────────────────────────────────────────────────

def compute_gini_stability(
    df: pd.DataFrame,
    pd_score_col: str,
    target_col: str = TARGET_COLUMN,
    time_col: str = None,
    n_periods: int = 5,
) -> pd.DataFrame:
    """
    Track Gini coefficient across time periods (monthly/quarterly cohorts).
    Used to detect model degradation before it becomes a business problem.

    If no time_col is available, splits data into n_periods equal chunks
    (simulating time periods using index order).
    """
    df = df.copy()

    if time_col and time_col in df.columns:
        df["_period"] = df[time_col]
        periods = sorted(df["_period"].unique())
    else:
        # Simulate time by splitting into equal chunks
        df["_period"] = pd.qcut(df.index, q=n_periods, labels=False, duplicates="drop")
        periods = sorted(df["_period"].unique())

    records = []
    for period in periods:
        subset = df[df["_period"] == period]
        if subset[target_col].nunique() < 2 or len(subset) < 50:
            logger.warning(f"Period {period} skipped: insufficient data.")
            continue
        auc  = roc_auc_score(subset[target_col], subset[pd_score_col])
        gini = 2 * auc - 1
        records.append({
            "period":       period,
            "n_customers":  len(subset),
            "event_rate":   round(subset[target_col].mean(), 4),
            "gini":         round(gini, 4),
            "auc":          round(auc, 4),
        })

    results = pd.DataFrame(records)

    # Flag degradation
    if len(results) >= 2:
        first_gini = results.iloc[0]["gini"]
        results["gini_delta_from_baseline"] = results["gini"] - first_gini
        results["alert"] = results["gini_delta_from_baseline"].apply(
            lambda d: "🔴 ALERT" if d < -GINI_DEGRADATION_ALERT
            else "🟡 WATCH" if d < -GINI_DEGRADATION_ALERT / 2
            else "🟢 OK"
        )

    logger.info(f"\nGini Stability Over Time:\n{results.to_string(index=False)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Bias Audit
# ─────────────────────────────────────────────────────────────────────────────

def bias_audit(
    df: pd.DataFrame,
    pd_score_col: str,
    target_col: str = TARGET_COLUMN,
    group_cols: List[str] = None,
    min_group_size: int = 100,
) -> pd.DataFrame:
    """
    Check if model performance is consistent across demographic subgroups.
    Flags groups where AUC differs materially from overall performance.

    Relevant for fair lending compliance (ECOA, FCRA in the US context).
    """
    group_cols = group_cols or ["SEX", "EDUCATION", "MARRIAGE"]
    group_cols = [c for c in group_cols if c in df.columns]

    overall_auc = roc_auc_score(df[target_col], df[pd_score_col])
    records     = [{"group_col": "OVERALL", "group_val": "All", "n": len(df),
                    "auc": overall_auc, "gini": 2*overall_auc-1,
                    "event_rate": df[target_col].mean(), "auc_delta": 0.0}]

    for col in group_cols:
        for val, grp in df.groupby(col):
            if len(grp) < min_group_size:
                continue
            if grp[target_col].nunique() < 2:
                continue
            g_auc  = roc_auc_score(grp[target_col], grp[pd_score_col])
            records.append({
                "group_col":  col,
                "group_val":  val,
                "n":          len(grp),
                "auc":        round(g_auc, 4),
                "gini":       round(2*g_auc-1, 4),
                "event_rate": round(grp[target_col].mean(), 4),
                "auc_delta":  round(g_auc - overall_auc, 4),
            })

    report = pd.DataFrame(records)
    n_flagged = (report["auc_delta"].abs() > 0.05).sum()
    logger.info(
        f"\nBias Audit Complete | {len(report)-1} subgroups | "
        f"{n_flagged} groups with AUC delta > 0.05\n"
        f"{report.to_string(index=False)}"
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Monitoring Dashboard Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_monitoring_dashboard(
    gini_stability: pd.DataFrame,
    psi_report: pd.DataFrame,
    bias_report: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    3-panel monitoring dashboard:
      1. Gini stability over time
      2. PSI distribution (top features)
      3. Bias audit — AUC by subgroup
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle(
        "Model Governance Dashboard\nMonitoring Report",
        fontsize=14, fontweight="bold"
    )

    # ── Panel 1: Gini Over Time ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])  # full width top row
    gini_vals = gini_stability["gini"]
    ax1.plot(
        gini_stability["period"].astype(str),
        gini_vals,
        "o-", color="#2563EB", lw=2.5, markersize=8,
        label="Gini per Period"
    )
    ax1.axhline(
        gini_vals.iloc[0], color="gray", ls="--", lw=1.5,
        label=f"Baseline Gini ({gini_vals.iloc[0]:.3f})"
    )
    ax1.axhline(
        MIN_ACCEPTABLE_GINI, color="#DC2626", ls="--", lw=1.5,
        label=f"Minimum Acceptable ({MIN_ACCEPTABLE_GINI:.2f})"
    )
    ax1.fill_between(
        gini_stability["period"].astype(str),
        gini_vals.iloc[0] - GINI_DEGRADATION_ALERT,
        gini_vals.iloc[0],
        alpha=0.08, color="#F59E0B",
        label=f"Alert Zone (drop > {GINI_DEGRADATION_ALERT:.0%})"
    )
    ax1.set_xlabel("Time Period")
    ax1.set_ylabel("Gini Coefficient")
    ax1.set_title("Gini Stability Over Time — Model Discrimination Health")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(max(0, gini_vals.min() - 0.1), min(1, gini_vals.max() + 0.1))

    # ── Panel 2: PSI Distribution ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    top_psi = psi_report.head(20).sort_values("psi")
    colors  = [
        "#DC2626" if p >= PSI_THRESHOLDS["moderate"] else
        "#F59E0B" if p >= PSI_THRESHOLDS["stable"] else
        "#16A34A"
        for p in top_psi["psi"]
    ]
    ax2.barh(top_psi["feature"], top_psi["psi"], color=colors, edgecolor="white")
    ax2.axvline(PSI_THRESHOLDS["stable"],   color="orange", ls="--", lw=1.5,
                label=f"Stable threshold ({PSI_THRESHOLDS['stable']})")
    ax2.axvline(PSI_THRESHOLDS["moderate"], color="red",    ls="--", lw=1.5,
                label=f"Action threshold ({PSI_THRESHOLDS['moderate']})")
    ax2.set_xlabel("PSI")
    ax2.set_title("Population Stability Index\nTop Features")
    ax2.legend(fontsize=8)
    ax2.grid(axis="x", alpha=0.3)

    # ── Panel 3: Bias Audit ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    bias_sub = bias_report[bias_report["group_col"] != "OVERALL"].copy()
    if not bias_sub.empty:
        bias_sub["label"] = bias_sub["group_col"] + "=" + bias_sub["group_val"].astype(str)
        bias_sub = bias_sub.sort_values("auc")
        bar_colors = [
            "#DC2626" if abs(d) > 0.05 else "#16A34A"
            for d in bias_sub["auc_delta"]
        ]
        ax3.barh(bias_sub["label"], bias_sub["auc"], color=bar_colors, edgecolor="white")
        overall_auc = bias_report.loc[bias_report["group_col"]=="OVERALL", "auc"].values
        if len(overall_auc):
            ax3.axvline(overall_auc[0], color="navy", ls="--", lw=1.5,
                        label=f"Overall AUC ({overall_auc[0]:.3f})")
        ax3.set_xlabel("AUC-ROC")
        ax3.set_title("Bias Audit — AUC by Subgroup\nRed = >0.05 delta from overall")
        ax3.legend(fontsize=8)
        ax3.grid(axis="x", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No subgroup data available",
                 ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Bias Audit")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved monitoring dashboard to {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Model Card Generator
# ─────────────────────────────────────────────────────────────────────────────

class ModelCard:
    """
    Generates a structured Model Card document.

    Inspired by:
      - OCC SR 11-7 Model Risk Management Guidance
      - Google Model Cards (Mitchell et al., 2019)
      - EU AI Act documentation requirements

    This is the document a model validation team reviews
    before a model goes into production.
    """

    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        self.model_name    = model_name
        self.model_version = model_version
        self.created_at    = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.card_data: dict = {}

    def add_section(self, section: str, data: dict) -> None:
        self.card_data[section] = data

    def build(
        self,
        train_metrics:   dict,
        test_metrics:    dict,
        oot_metrics:     dict,
        cv_metrics:      dict,
        feature_list:    list,
        top_features:    pd.DataFrame,
        gini_stability:  pd.DataFrame,
        psi_summary:     dict,
        bias_report:     pd.DataFrame,
        n_train:         int,
        n_test:          int,
        n_oot:           int,
    ) -> dict:
        """Assemble the complete model card."""

        self.card_data = {

            "model_overview": {
                "name":               self.model_name,
                "version":            self.model_version,
                "created":            self.created_at,
                "purpose":            "Predict probability of customer credit card default within next billing cycle",
                "intended_use":       "Credit limit decisions, collections triggers, risk-based pricing",
                "not_intended_for":   "Sole basis of adverse action without human review",
                "regulatory_context": "Aligned with OCC SR 11-7 Model Risk Management guidance",
            },

            "data": {
                "training_source":     "UCI Credit Card Default Dataset (Taiwan, 2005)",
                "n_train":             n_train,
                "n_test":              n_test,
                "n_oot":               n_oot,
                "target_definition":  "Binary: 1 = default on next payment, 0 = no default",
                "train_event_rate":    train_metrics.get("event_rate", "N/A"),
                "n_input_features":    len(feature_list),
                "feature_engineering": "Velocity, utilization, payment ratio, delinquency trend features engineered from raw data",
            },

            "model_architecture": {
                "type":               "Gradient Boosted Trees (XGBoost) with Platt probability calibration",
                "baseline_model":     "Logistic Regression Scorecard with WoE binning (regulatory standard)",
                "key_hyperparameters":{
                    "n_estimators": "500 (with early stopping)",
                    "max_depth":    "5",
                    "learning_rate":"0.05",
                    "regularization": "L1 + L2",
                },
                "imbalance_handling": "scale_pos_weight = 3.5 (event rate ~22%)",
                "calibration":        "Platt sigmoid scaling (3-fold CV)",
            },

            "performance": {
                "cross_validation": {
                    "folds":          5,
                    "oof_gini":       cv_metrics.get("oof_gini", cv_metrics.get("cv_gini_mean", "N/A")),
                    "mean_gini":      cv_metrics.get("cv_gini_mean", "N/A"),
                    "std_gini":       cv_metrics.get("cv_auc_std", "N/A"),
                },
                "in_sample_test": {
                    "gini":       test_metrics.get("gini", "N/A"),
                    "ks_stat":    test_metrics.get("ks_stat", "N/A"),
                    "auc_roc":    test_metrics.get("auc_roc", "N/A"),
                },
                "out_of_time_test": {
                    "gini":       oot_metrics.get("gini", "N/A"),
                    "ks_stat":    oot_metrics.get("ks_stat", "N/A"),
                    "auc_roc":    oot_metrics.get("auc_roc", "N/A"),
                    "note":       "OOT cohort represents most recent 20% of data by index order",
                },
                "gini_stability": {
                    "periods_tested": len(gini_stability),
                    "min_gini":       float(gini_stability["gini"].min()),
                    "max_gini":       float(gini_stability["gini"].max()),
                    "status":         "STABLE" if gini_stability["gini"].min() >= MIN_ACCEPTABLE_GINI else "DEGRADED",
                },
            },

            "top_predictive_features": (
                top_features[["feature", "mean_abs_shap"]].head(10).to_dict("records")
                if top_features is not None and not top_features.empty else []
            ),

            "fairness_bias_audit": {
                "groups_tested":  list(bias_report["group_col"].unique()),
                "max_auc_delta":  float(bias_report["auc_delta"].abs().max()),
                "status":         (
                    "PASS — No subgroup AUC delta > 0.05"
                    if bias_report["auc_delta"].abs().max() <= 0.05
                    else "REVIEW — Subgroup performance gap detected"
                ),
                "details":        bias_report.to_dict("records"),
            },

            "psi_monitoring_thresholds": {
                "green_threshold":  PSI_THRESHOLDS["stable"],
                "amber_threshold":  PSI_THRESHOLDS["moderate"],
                "monitoring_cadence": "Monthly",
                "trigger_action":   "PSI > 0.25 on score distribution triggers model review",
                "current_status":   psi_summary,
            },

            "known_limitations": [
                "Trained on Taiwan credit card data (2005) — may not fully generalise to other markets",
                "Does not capture macroeconomic regime changes (COVID-type shocks)",
                "Static model — does not update on new observations in real-time",
                "Demographic features (SEX, EDUCATION) included for monitoring only — not decision input",
                "Probability outputs are calibrated but not true Bayesian posterior probabilities",
            ],

            "recommended_monitoring": {
                "score_psi":          "Monthly — trigger review if PSI > 0.25",
                "feature_psi":        "Monthly — flag top-10 features with PSI > 0.10",
                "gini_tracking":      "Quarterly — alert if Gini drops > 5 points from baseline",
                "bias_audit":         "Semi-annual — verify no material AUC gap across subgroups",
                "full_rebuild_trigger": "Gini < 0.40 OR score PSI > 0.40 OR significant policy change",
            },

        }
        return self.card_data

    def save_json(self, path: str = None) -> str:
        """Save model card as JSON."""
        path = path or os.path.join(REPORTS_DIR, f"model_card_{self.model_version}.json")
        with open(path, "w") as f:
            json.dump(self.card_data, f, indent=2, default=str)
        logger.info(f"Model card saved to {path}")
        return path

    def save_text_report(self, path: str = None) -> str:
        """Save model card as a readable text report."""
        path = path or os.path.join(REPORTS_DIR, f"model_governance_report_{self.model_version}.txt")
        lines = []
        lines.append("=" * 70)
        lines.append(f"  MODEL GOVERNANCE REPORT")
        lines.append(f"  {self.model_name} | v{self.model_version} | {self.created_at}")
        lines.append("=" * 70)

        for section, content in self.card_data.items():
            lines.append(f"\n{'─'*70}")
            lines.append(f"  {section.upper().replace('_', ' ')}")
            lines.append(f"{'─'*70}")
            if isinstance(content, dict):
                for k, v in content.items():
                    if isinstance(v, dict):
                        lines.append(f"  {k}:")
                        for kk, vv in v.items():
                            lines.append(f"    {kk}: {vv}")
                    elif isinstance(v, list):
                        lines.append(f"  {k}:")
                        for item in v:
                            lines.append(f"    - {item}")
                    else:
                        lines.append(f"  {k}: {v}")
            elif isinstance(content, list):
                for item in content:
                    lines.append(f"  - {item}")

        lines.append("\n" + "=" * 70)
        lines.append("  END OF REPORT")
        lines.append("=" * 70)

        with open(path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Governance report saved to {path}")
        return path
