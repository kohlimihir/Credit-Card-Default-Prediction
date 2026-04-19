"""
pipeline.py
===========
Master orchestration pipeline — runs all 5 modules end-to-end
using ONLY real downloaded data (no synthetic generation anywhere).

Supports 3 real datasets controlled via config.ACTIVE_DATASET:
  "uci"        → UCI Credit Card Default (auto-download)
  "gmc"        → Give Me Some Credit (Kaggle download / manual)
  "homecredit" → Home Credit Default Risk (Kaggle download / manual)

Usage:
    python pipeline.py                    # uses config.ACTIVE_DATASET
    python pipeline.py --dataset gmc      # override dataset
    python pipeline.py --dataset uci --force-download

Outputs (reports/):
  *.png plots, *.csv tables, model_card.json, model_governance_report.txt
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ACTIVE_DATASET, DATASET_CONFIGS,
    TARGET_COLUMN, ID_COLUMN,
    REPORTS_DIR, MODELS_DIR,
    TEST_SIZE, OOT_CUTOFF_QUANTILE,
    PSI_THRESHOLDS,
    LOG_LEVEL, LOG_FORMAT,
)
from src.data_loader         import load_dataset
from src.feature_engineering import FeatureEngineer
from src.woe_scorecard       import CreditScorecard
from src.ml_models           import MLCreditModel
from src.segmentation        import CustomerSegmenter, build_risk_tier_summary
from src.monitoring          import (
    compute_psi,
    compute_feature_psi_report,
    compute_gini_stability,
    bias_audit,
    plot_monitoring_dashboard,
    ModelCard,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rpt(filename: str) -> str:
    return os.path.join(REPORTS_DIR, filename)


def sep(title: str) -> None:
    logger.info(f"\n{'═'*65}")
    logger.info(f"  {title}")
    logger.info(f"{'═'*65}")


# ─────────────────────────────────────────────────────────────────────────────
# Data Split Helper
# ─────────────────────────────────────────────────────────────────────────────

def make_splits(X: pd.DataFrame, y: pd.Series, oot_quantile: float = OOT_CUTOFF_QUANTILE):
    """
    Create Train / Test / Out-of-Time (OOT) splits.

    OOT = the last (1 - oot_quantile) fraction of the data by row index.
    This simulates real deployment: train on historical, evaluate on future.
    """
    oot_cutoff = int(len(X) * oot_quantile)

    X_dev, y_dev = X.iloc[:oot_cutoff],  y.iloc[:oot_cutoff]
    X_oot, y_oot = X.iloc[oot_cutoff:],  y.iloc[oot_cutoff:]

    X_train, X_test, y_train, y_test = train_test_split(
        X_dev, y_dev,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y_dev,
    )
    logger.info(
        f"Data splits:\n"
        f"  Train : {len(X_train):>7,} rows  ({y_train.mean():.2%} default rate)\n"
        f"  Test  : {len(X_test):>7,} rows  ({y_test.mean():.2%} default rate)\n"
        f"  OOT   : {len(X_oot):>7,} rows  ({y_oot.mean():.2%} default rate)\n"
        f"  OOT cutoff at row {oot_cutoff:,} (simulates temporal holdout)"
    )
    return X_train, X_test, y_train, y_test, X_dev, y_dev, X_oot, y_oot


# ─────────────────────────────────────────────────────────────────────────────
# Bias audit column map per dataset
# ─────────────────────────────────────────────────────────────────────────────

BIAS_COLS = {
    "uci":        ["SEX", "EDUCATION", "MARRIAGE"],
    "gmc":        ["age"],         # will be bucketed inside bias_audit
    "homecredit": ["CODE_GENDER_M", "NAME_EDUCATION_TYPE_Higher education"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(dataset: str = None, force_download: bool = False) -> dict:
    """Execute all 5 modules. Returns results dict."""
    dataset     = dataset or ACTIVE_DATASET
    dataset_cfg = DATASET_CONFIGS[dataset]
    target_col  = dataset_cfg["target_col"]

    all_results = {}

    # ══════════════════════════════════════════════════════════════════════
    # MODULE 1 — Data Loading + Feature Engineering
    # ══════════════════════════════════════════════════════════════════════
    sep("MODULE 1 — Data Loading & Feature Engineering")
    logger.info(f"Dataset: {dataset_cfg['name']}")
    logger.info(f"Description: {dataset_cfg['description']}")

    # ── Real data load ────────────────────────────────────────────────
    df_raw = load_dataset(dataset, force_download=force_download)

    # ── Feature engineering ────────────────────────────────────────────
    fe = FeatureEngineer(dataset)
    df = fe.fit_transform(df_raw)

    feature_cols = fe.get_feature_names()
    X_all = df[feature_cols]
    y_all = df[target_col]

    logger.info(f"\nFeature groups:")
    for grp, cols in fe.get_feature_groups().items():
        logger.info(f"  {grp:<20}: {len(cols):>3} features")

    # ── Splits ────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, X_dev, y_dev, X_oot, y_oot = make_splits(X_all, y_all)

    all_results["data"] = {
        "dataset":      dataset,
        "n_total":      len(df),
        "n_train":      len(X_train),
        "n_test":       len(X_test),
        "n_oot":        len(X_oot),
        "n_features":   len(feature_cols),
        "event_rate_train": round(float(y_train.mean()), 4),
    }

    # ══════════════════════════════════════════════════════════════════════
    # MODULE 2 — WoE Scorecard (Regulatory Baseline)
    # ══════════════════════════════════════════════════════════════════════
    sep("MODULE 2 — WoE Scorecard (Regulatory Baseline)")

    scorecard = CreditScorecard()
    scorecard.fit(X_train, y_train)

    sc_cv    = scorecard.cross_validate(X_train, y_train)
    sc_test  = scorecard.evaluate(X_test, y_test,  label="Scorecard (Test)")
    sc_oot   = scorecard.evaluate(X_oot,  y_oot,   label="Scorecard (OOT)")

    _   = scorecard.score_distribution(X_test, y_test)

    scorecard.plot_evaluation(X_test, y_test,  save_path=rpt("scorecard_evaluation.png"))
    scorecard.plot_iv_chart(save_path=rpt("iv_chart.png"))

    scorecard.get_scorecard_table().to_csv(rpt("scorecard_points_table.csv"), index=False)
    scorecard.get_iv_table().to_csv(rpt("iv_table.csv"), index=False)

    sc_proba_test = scorecard.predict_proba(X_test)
    sc_proba_oot  = scorecard.predict_proba(X_oot)

    all_results["scorecard"] = {"cv": sc_cv, "test": sc_test, "oot": sc_oot}

    # ══════════════════════════════════════════════════════════════════════
    # MODULE 3 — ML Models + SHAP
    # ══════════════════════════════════════════════════════════════════════
    sep("MODULE 3 — ML Models (XGBoost + LightGBM) + SHAP")

    # ── XGBoost ───────────────────────────────────────────────────────
    xgb_model = MLCreditModel(model_type="xgboost")
    xgb_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    xgb_cv   = xgb_model.cross_validate(X_train, y_train)
    xgb_test = xgb_model.evaluate(X_test, y_test, label="XGBoost (Test)")
    xgb_oot  = xgb_model.evaluate(X_oot,  y_oot,  label="XGBoost (OOT)")

    xgb_proba_test = xgb_model.predict_proba(X_test)
    xgb_proba_oot  = xgb_model.predict_proba(X_oot)
    xgb_proba_dev  = xgb_model.predict_proba(X_dev)

    # ── LightGBM ──────────────────────────────────────────────────────
    lgbm_model = MLCreditModel(model_type="lightgbm")
    lgbm_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    lgbm_test  = lgbm_model.evaluate(X_test, y_test, label="LightGBM (Test)")
    lgbm_oot   = lgbm_model.evaluate(X_oot,  y_oot,  label="LightGBM (OOT)")
    lgbm_proba_test = lgbm_model.predict_proba(X_test)

    # ── Model comparison plot ─────────────────────────────────────────
    all_probas = {
        "Scorecard (WoE+LR)": sc_proba_test,
        "XGBoost":            xgb_proba_test,
        "LightGBM":           lgbm_proba_test,
    }
    xgb_model.plot_model_comparison(all_probas, y_test,
                                     save_path=rpt("model_comparison.png"))
    xgb_model.plot_calibration_comparison(all_probas, y_test,
                                           save_path=rpt("calibration_comparison.png"))

    # ── SHAP ──────────────────────────────────────────────────────────
    logger.info("Building SHAP explanations ...")
    # Use a background sample of 200 rows from training data (speeds up computation)
    background = X_train.sample(min(200, len(X_train)), random_state=42)
    xgb_model.build_shap_explainer(background)

    # SHAP on test set (use up to 2000 rows for speed)
    X_shap = X_test.sample(min(2000, len(X_test)), random_state=42)
    xgb_model.plot_shap_summary(X_shap,  save_path=rpt("xgb_shap_summary.png"))
    xgb_model.plot_shap_waterfall(X_shap, row_idx=0,
                                   save_path=rpt("xgb_shap_waterfall_customer0.png"))

    top_shap = xgb_model.get_top_shap_features(X_shap, n=20)
    top_shap.to_csv(rpt("top_shap_features.csv"), index=False)

    # Dependence plot for the top SHAP feature
    top_feat = top_shap.iloc[0]["feature"]
    xgb_model.plot_shap_dependence(X_shap, feature=top_feat,
                                    save_path=rpt(f"shap_dependence_{top_feat}.png"))

    logger.info(f"\nTop 10 SHAP features:\n{top_shap.head(10).to_string(index=False)}")

    all_results["xgboost"]  = {"cv": xgb_cv, "test": xgb_test, "oot": xgb_oot}
    all_results["lightgbm"] = {"test": lgbm_test, "oot": lgbm_oot}

    # ══════════════════════════════════════════════════════════════════════
    # MODULE 4 — Customer Segmentation
    # ══════════════════════════════════════════════════════════════════════
    sep("MODULE 4 — Customer Risk Segmentation")

    segmenter = CustomerSegmenter(n_clusters=4)

    # Elbow plot to justify k=4
    segmenter.plot_elbow(X_train, save_path=rpt("elbow_plot.png"))

    # Fit on training data with XGBoost PD scores
    xgb_proba_train = xgb_model.predict_proba(X_train)
    segmenter.fit(X_train, xgb_proba_train, y_true=y_train)

    # Predict and summarise dev set
    segment_df = segmenter.predict(X_dev, pd_scores=xgb_proba_dev)

    tier_summary = build_risk_tier_summary(xgb_proba_dev, y_dev)
    tier_summary.to_csv(rpt("risk_tier_summary.csv"))

    action_matrix = segmenter.build_action_matrix(X_dev, xgb_proba_dev)
    action_matrix.to_csv(rpt("business_action_matrix.csv"), index=False)

    segmenter.plot_cluster_profiles(save_path=rpt("cluster_profiles.png"))
    segmenter.plot_pca_clusters(X_dev, xgb_proba_dev, save_path=rpt("pca_clusters.png"))

    logger.info(f"\nSegment distribution:\n{segment_df['segment_name'].value_counts().to_string()}")
    logger.info(f"\nRisk Tier Summary:\n{tier_summary.round(4).to_string()}")
    logger.info(f"\nBusiness Action Matrix:\n{action_matrix.to_string(index=False)}")

    all_results["segmentation"] = {
        "n_segments":    4,
        "segment_names": list(segmenter.label_map_.values()),
    }

    # ══════════════════════════════════════════════════════════════════════
    # MODULE 5 — Model Governance
    # ══════════════════════════════════════════════════════════════════════
    sep("MODULE 5 — Model Governance & Monitoring")

    # ── Score PSI: Test vs OOT ────────────────────────────────────────
    score_psi = compute_psi(
        expected=xgb_proba_test,
        actual=xgb_proba_oot,
        feature_name="XGBoost_PD_Score",
    )

    # ── Feature PSI on top SHAP features ─────────────────────────────
    top_feat_names = top_shap["feature"].head(20).tolist()
    psi_report = compute_feature_psi_report(
        X_train=X_train[top_feat_names],
        X_monitor=X_oot[top_feat_names],
        n_bins=10,
    )
    psi_report.to_csv(rpt("psi_report.csv"), index=False)

    # ── Gini Stability (5 time periods across dev set) ────────────────
    df_dev_scores = X_dev.copy()
    df_dev_scores["pd_score"]  = xgb_proba_dev
    df_dev_scores[target_col]  = y_dev.values

    gini_stability = compute_gini_stability(
        df=df_dev_scores,
        pd_score_col="pd_score",
        target_col=target_col,
        n_periods=5,
    )
    gini_stability.to_csv(rpt("gini_stability.csv"), index=False)

    # ── Bias Audit ────────────────────────────────────────────────────
    # Attach raw demographic columns to dev scores for audit
    raw_dev = df_raw.iloc[:len(X_dev)].copy().reset_index(drop=True)
    raw_dev["pd_score"] = xgb_proba_dev

    bias_group_cols = BIAS_COLS.get(dataset, [])
    # Filter to columns that actually exist in raw data
    bias_group_cols = [c for c in bias_group_cols if c in raw_dev.columns]

    bias_report = bias_audit(
        df=raw_dev,
        pd_score_col="pd_score",
        target_col=target_col,
        group_cols=bias_group_cols if bias_group_cols else None,
    )
    bias_report.to_csv(rpt("bias_audit.csv"), index=False)

    # ── Monitoring Dashboard ──────────────────────────────────────────
    plot_monitoring_dashboard(
        gini_stability=gini_stability,
        psi_report=psi_report,
        bias_report=bias_report,
        save_path=rpt("monitoring_dashboard.png"),
    )

    # ── Model Card ────────────────────────────────────────────────────
    model_card = ModelCard(
        model_name=f"Credit Default Prediction — {dataset_cfg['name']}",
        model_version="1.0.0",
    )
    model_card.build(
        train_metrics=xgb_test,
        test_metrics=xgb_test,
        oot_metrics=xgb_oot,
        cv_metrics=xgb_cv,
        feature_list=feature_cols,
        top_features=top_shap,
        gini_stability=gini_stability,
        psi_summary={
            "score_psi":           score_psi["psi"],
            "score_status":        score_psi["status"],
            "n_features_flagged":  int((psi_report["psi"] >= PSI_THRESHOLDS["moderate"]).sum()),
        },
        bias_report=bias_report,
        n_train=len(X_train),
        n_test=len(X_test),
        n_oot=len(X_oot),
    )
    model_card.save_json(rpt("model_card.json"))
    model_card.save_text_report(rpt("model_governance_report.txt"))

    all_results["governance"] = {
        "score_psi":          round(score_psi["psi"], 4),
        "score_psi_status":   score_psi["status"],
        "gini_periods_tested":len(gini_stability),
        "gini_min":           float(gini_stability["gini"].min()),
        "bias_max_auc_delta": float(bias_report["auc_delta"].abs().max()),
    }

    # ══════════════════════════════════════════════════════════════════════
    # FINAL RESULTS SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    sep("PIPELINE COMPLETE — RESULTS SUMMARY")

    summary_rows = [
        {"Model": "Scorecard (WoE+LR)", "Dataset": "Test", **sc_test},
        {"Model": "Scorecard (WoE+LR)", "Dataset": "OOT",  **sc_oot},
        {"Model": "XGBoost",            "Dataset": "Test", **xgb_test},
        {"Model": "XGBoost",            "Dataset": "OOT",  **xgb_oot},
        {"Model": "LightGBM",           "Dataset": "Test", **lgbm_test},
        {"Model": "LightGBM",           "Dataset": "OOT",  **lgbm_oot},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[["Model", "Dataset", "auc_roc", "gini", "ks_stat", "brier_score"]].round(4)
    summary_df.to_csv(rpt("results_summary.csv"), index=False)

    logger.info(f"\n{summary_df.to_string(index=False)}")

    logger.info(f"\n{'─'*65}")
    logger.info(f"Governance Summary:")
    logger.info(f"  Score PSI (Train→OOT): {score_psi['psi']:.4f}  {score_psi['status']}")
    logger.info(f"  Gini range across time: {gini_stability['gini'].min():.3f} – {gini_stability['gini'].max():.3f}")
    logger.info(f"  Bias max AUC delta:     {bias_report['auc_delta'].abs().max():.4f}")
    logger.info(f"{'─'*65}")

    logger.info(f"\nAll output files saved to: {REPORTS_DIR}")
    _list_outputs()

    all_results["summary"] = summary_df.to_dict("records")
    return all_results


def _list_outputs() -> None:
    files = sorted(Path(REPORTS_DIR).glob("*"))
    if not files:
        return
    logger.info("\nGenerated output files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {f.name:<55} {size_kb:>7.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Default Prediction Pipeline")
    parser.add_argument(
        "--dataset",
        choices=["uci", "gmc", "homecredit"],
        default=None,
        help="Dataset to use (overrides config.ACTIVE_DATASET)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if cached data exists",
    )
    args = parser.parse_args()

    results = run_pipeline(
        dataset=args.dataset,
        force_download=args.force_download,
    )
