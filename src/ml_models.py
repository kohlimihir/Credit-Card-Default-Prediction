"""
src/ml_models.py
----------------
XGBoost + LightGBM ensemble with:
  - Stratified cross-validation
  - Probability calibration (Platt scaling)
  - SHAP global + local explanations
  - Head-to-head comparison with Scorecard
  - Feature importance plots
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
import shap

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_COLUMN, REPORTS_DIR,
    XGBOOST_PARAMS, LGBM_PARAMS,
    CV_FOLDS,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
shap.initjs()


# ─────────────────────────────────────────────────────────────────────────────
# Base ML Model Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MLCreditModel:
    """
    Wrapper for XGBoost / LightGBM with:
      - Cross-validation
      - Probability calibration
      - SHAP explanations
      - Standardised evaluation metrics
    """

    SUPPORTED_MODELS = {"xgboost", "lightgbm"}

    def __init__(self, model_type: str = "xgboost"):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")
        self.model_type = model_type
        self.model_ = None
        self.calibrated_model_ = None
        self.feature_names_: list = []
        self.shap_explainer_ = None
        self.cv_results_: dict = {}
        self._fitted = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_base_model(self):
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(**XGBOOST_PARAMS)
        else:
            return lgb.LGBMClassifier(**LGBM_PARAMS)

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self, X: pd.DataFrame, y: pd.Series,
        X_val: pd.DataFrame = None, y_val: pd.Series = None
    ) -> "MLCreditModel":
        """
        Train model with optional early stopping on validation set.
        Calibrates probability output using Platt scaling.
        """
        logger.info(f"=== Training {self.model_type.upper()} Model ===")
        self.feature_names_ = list(X.columns)

        base_model = self._build_base_model()

        if X_val is not None and y_val is not None:
            if self.model_type == "xgboost":
                base_model.set_params(early_stopping_rounds=30)
                base_model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    verbose=100,
                )
            else:  # lightgbm
                base_model.set_params(early_stopping_rounds=30, callbacks=[lgb.log_evaluation(100)])
                base_model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                )
        else:
            base_model.fit(X, y)

        self.model_ = base_model

        # Calibrate probabilities using Platt scaling (sigmoid)
        logger.info("Calibrating probability output with Platt scaling...")
        self.calibrated_model_ = CalibratedClassifierCV(
            estimator=self._build_base_model(),
            method="sigmoid",
            cv=3,
        )
        self.calibrated_model_.fit(X, y)

        self._fitted = True
        logger.info(f"=== {self.model_type.upper()} Training Complete ===")
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probability of default."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.calibrated_model_.predict_proba(X)[:, 1]

    def predict_proba_uncalibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Raw (uncalibrated) probability for comparison."""
        return self.model_.predict_proba(X)[:, 1]

    # ── Cross-Validation ─────────────────────────────────────────────────────

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Stratified K-fold CV.
        Returns Gini, KS, AUC per fold + summary statistics.
        """
        logger.info(f"Running {CV_FOLDS}-fold stratified CV for {self.model_type}...")
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

        fold_metrics = []
        oof_proba = np.zeros(len(y))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model = self._build_base_model()
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_val)[:, 1]
            oof_proba[val_idx] = preds

            auc  = roc_auc_score(y_val, preds)
            gini = 2 * auc - 1
            fpr, tpr, _ = roc_curve(y_val, preds)
            ks   = (tpr - fpr).max()

            fold_metrics.append({"fold": fold, "auc": auc, "gini": gini, "ks": ks})
            logger.info(f"  Fold {fold}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

        # OOF (Out-of-Fold) aggregate
        oof_auc  = roc_auc_score(y, oof_proba)
        oof_gini = 2 * oof_auc - 1

        df_folds = pd.DataFrame(fold_metrics)
        self.cv_results_ = {
            "model":          self.model_type,
            "cv_auc_mean":    round(df_folds["auc"].mean(), 4),
            "cv_auc_std":     round(df_folds["auc"].std(), 4),
            "cv_gini_mean":   round(df_folds["gini"].mean(), 4),
            "cv_ks_mean":     round(df_folds["ks"].mean(), 4),
            "oof_auc":        round(oof_auc, 4),
            "oof_gini":       round(oof_gini, 4),
            "fold_details":   df_folds,
        }

        logger.info(
            f"CV Summary | OOF AUC={oof_auc:.4f} | OOF Gini={oof_gini:.4f} | "
            f"Mean±Std Gini={df_folds['gini'].mean():.4f}±{df_folds['gini'].std():.4f}"
        )
        return self.cv_results_

    # ── Evaluate ─────────────────────────────────────────────────────────────

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, label: str = None
    ) -> dict:
        """Compute evaluation metrics on a holdout set."""
        label = label or self.model_type.upper()
        proba = self.predict_proba(X)

        auc  = roc_auc_score(y, proba)
        gini = 2 * auc - 1
        fpr, tpr, _ = roc_curve(y, proba)
        ks   = (tpr - fpr).max()
        ap   = average_precision_score(y, proba)
        brier = brier_score_loss(y, proba)

        metrics = {
            "label":      label,
            "auc_roc":    round(auc, 4),
            "gini":       round(gini, 4),
            "ks_stat":    round(ks, 4),
            "avg_prec":   round(ap, 4),
            "brier_score":round(brier, 4),
            "n_samples":  len(y),
            "event_rate": round(float(y.mean()), 4),
        }

        logger.info(
            f"\n{'='*50}\n"
            f"Evaluation: {label}\n"
            f"  AUC-ROC     : {auc:.4f}\n"
            f"  Gini        : {gini:.4f}\n"
            f"  KS Statistic: {ks:.4f}\n"
            f"  Brier Score : {brier:.4f}  (lower=better, 0=perfect)\n"
            f"{'='*50}"
        )
        return metrics

    # ── SHAP ─────────────────────────────────────────────────────────────────

    def build_shap_explainer(self, X_background: pd.DataFrame) -> None:
        """
        Build SHAP TreeExplainer (fast, exact for tree models).
        X_background: a sample (200–500 rows) for baseline reference.
        """
        logger.info("Building SHAP TreeExplainer...")
        self.shap_explainer_ = shap.TreeExplainer(
            self.model_,
            data=X_background.sample(
                min(200, len(X_background)), random_state=42
            ),
            feature_perturbation="interventional",
        )
        logger.info("SHAP explainer ready.")

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for a dataset."""
        if self.shap_explainer_ is None:
            raise RuntimeError("Call build_shap_explainer() first.")
        shap_values = self.shap_explainer_.shap_values(X)
        # For binary classifiers, shap_values may be list of 2 arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return shap_values

    def plot_shap_summary(
        self, X: pd.DataFrame, n_display: int = 20,
        save_path: str = None
    ) -> None:
        """
        SHAP beeswarm summary plot: global feature importance +
        direction of effect for each feature.
        """
        shap_values = self.compute_shap_values(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            max_display=n_display,
            show=False,
            plot_size=None,
        )
        plt.title(
            f"SHAP Feature Importance — {self.model_type.upper()}\n"
            "Red=increases default risk | Blue=decreases default risk",
            fontsize=11
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved SHAP summary plot to {save_path}")
        plt.close()

    def plot_shap_waterfall(
        self, X: pd.DataFrame, row_idx: int = 0,
        save_path: str = None
    ) -> None:
        """
        SHAP waterfall plot for a single customer.
        Shows exactly why this customer has a high/low default probability.
        This is the 'local explanation' — the key to regulatory compliance.
        """
        if self.shap_explainer_ is None:
            raise RuntimeError("Call build_shap_explainer() first.")

        shap_values = self.shap_explainer_(X.iloc[[row_idx]])

        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values.values[0],
                base_values=shap_values.base_values[0],
                data=X.iloc[row_idx].values,
                feature_names=list(X.columns),
            ),
            show=False,
        )
        plt.title(f"SHAP Waterfall — Customer #{row_idx} Default Explanation", fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved SHAP waterfall to {save_path}")
        plt.close()

    def plot_shap_dependence(
        self, X: pd.DataFrame, feature: str,
        interaction_feature: str = None,
        save_path: str = None,
    ) -> None:
        """SHAP dependence plot: non-linear relationship for a feature."""
        shap_values = self.compute_shap_values(X)
        plt.figure(figsize=(9, 6))
        shap.dependence_plot(
            feature, shap_values, X,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.title(f"SHAP Dependence: {feature}", fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved SHAP dependence plot to {save_path}")
        plt.close()

    def get_top_shap_features(
        self, X: pd.DataFrame, n: int = 20
    ) -> pd.DataFrame:
        """Return mean absolute SHAP value per feature — global importance."""
        shap_values = self.compute_shap_values(X)
        importance = pd.DataFrame({
            "feature":    X.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).head(n)
        return importance

    # ── Comparison Plots ─────────────────────────────────────────────────────

    def plot_model_comparison(
        self,
        models_proba: dict,     # {"Model Name": proba_array}
        y_true: pd.Series,
        save_path: str = None,
    ) -> None:
        """
        ROC curve comparison across multiple models.
        Pass in dict of {model_name: predicted_probabilities}.
        """
        colors = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Model Comparison — Credit Default Prediction", fontsize=13, fontweight="bold")

        # ROC comparison
        for i, (name, proba) in enumerate(models_proba.items()):
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc  = roc_auc_score(y_true, proba)
            gini = 2 * auc - 1
            axes[0].plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                         label=f"{name} (AUC={auc:.3f}, Gini={gini:.3f})")
        axes[0].plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve Comparison")
        axes[0].legend(loc="lower right", fontsize=9)
        axes[0].grid(alpha=0.3)

        # Gini bar chart
        names  = list(models_proba.keys())
        ginis  = [2 * roc_auc_score(y_true, p) - 1 for p in models_proba.values()]
        bars = axes[1].bar(names, ginis,
                           color=colors[:len(names)], edgecolor="white", width=0.5)
        axes[1].set_ylim(0, max(ginis) * 1.3)
        axes[1].set_ylabel("Gini Coefficient")
        axes[1].set_title("Gini Comparison")
        for bar, g in zip(bars, ginis):
            axes[1].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{g:.3f}", ha="center", va="bottom", fontweight="bold"
            )
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved model comparison plot to {save_path}")
        plt.close()

    def plot_calibration_comparison(
        self,
        models_proba: dict,
        y_true: pd.Series,
        save_path: str = None,
    ) -> None:
        """Calibration comparison: does predicted PD match actual default rate?"""
        colors = ["#2563EB", "#DC2626", "#16A34A", "#D97706"]
        fig, ax = plt.subplots(figsize=(8, 7))

        for i, (name, proba) in enumerate(models_proba.items()):
            prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10)
            brier = brier_score_loss(y_true, proba)
            ax.plot(prob_pred, prob_true, "s-",
                    color=colors[i % len(colors)], lw=2, markersize=6,
                    label=f"{name} (Brier={brier:.3f})")

        ax.plot([0,1],[0,1], "k--", lw=1.5, label="Perfect Calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Actual Default Rate")
        ax.set_title("Calibration Comparison\n(closer to diagonal = better calibrated PD)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved calibration comparison to {save_path}")
        plt.close()
