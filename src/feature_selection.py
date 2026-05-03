"""
src/feature_selection.py
------------------------
Correlation + VIF based feature selection to remove multicollinearity.

Pipeline:
  1. Drop features with near-zero variance
  2. Drop one of each highly correlated pair (|r| > threshold)
  3. Iteratively drop the highest-VIF feature until all VIF < threshold

Fit on training data only → apply the same column mask to test/OOT.
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


class MulticollinearityFilter:
    """
    Two-stage multicollinearity removal:
      Stage 1 — pairwise Pearson correlation (fast, removes obvious duplicates)
      Stage 2 — iterative VIF elimination   (catches linear combos that pairwise misses)
    """

    def __init__(
        self,
        corr_threshold: float = 0.85,
        vif_threshold: float = 10.0,
        min_variance: float = 1e-6,
    ):
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.min_variance = min_variance

        self.dropped_zero_var_: list = []
        self.dropped_corr_: list = []
        self.dropped_vif_: list = []
        self.selected_features_: list = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "MulticollinearityFilter":
        """Identify features to keep. y is used to prefer the more IV-correlated feature."""
        cols = list(X.columns)

        # Stage 0: drop near-zero variance
        variances = X[cols].var()
        zero_var = variances[variances < self.min_variance].index.tolist()
        if zero_var:
            cols = [c for c in cols if c not in zero_var]
            self.dropped_zero_var_ = zero_var
            logger.info(f"VIF filter | Dropped {len(zero_var)} near-zero variance features")

        # Stage 1: pairwise correlation
        corr_matrix = X[cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        target_corr = {}
        if y is not None:
            for c in cols:
                target_corr[c] = abs(X[c].corr(y))

        to_drop_corr = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] > self.corr_threshold].tolist()
            for paired in high_corr:
                if col in to_drop_corr or paired in to_drop_corr:
                    continue
                # keep the one with higher target correlation
                if y is not None:
                    drop = paired if target_corr.get(col, 0) >= target_corr.get(paired, 0) else col
                else:
                    drop = paired
                to_drop_corr.add(drop)

        self.dropped_corr_ = sorted(to_drop_corr)
        cols = [c for c in cols if c not in to_drop_corr]
        logger.info(
            f"VIF filter | Correlation stage: dropped {len(self.dropped_corr_)} features "
            f"(|r| > {self.corr_threshold})"
        )

        # Stage 2: iterative VIF elimination
        self.dropped_vif_ = []
        working = cols.copy()

        while len(working) > 1:
            try:
                X_vif = X[working].values.astype(np.float64)
                vifs = [variance_inflation_factor(X_vif, i) for i in range(len(working))]
            except Exception:
                break

            max_vif = max(vifs)
            if max_vif <= self.vif_threshold:
                break

            worst_idx = vifs.index(max_vif)
            worst_feat = working[worst_idx]
            self.dropped_vif_.append((worst_feat, round(max_vif, 1)))
            working.pop(worst_idx)

        logger.info(
            f"VIF filter | VIF stage: dropped {len(self.dropped_vif_)} features "
            f"(VIF > {self.vif_threshold})"
        )

        self.selected_features_ = working
        self._fitted = True

        total_dropped = len(self.dropped_zero_var_) + len(self.dropped_corr_) + len(self.dropped_vif_)
        logger.info(
            f"VIF filter | Final: {len(self.selected_features_)} features kept, "
            f"{total_dropped} removed"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_drop_report(self) -> pd.DataFrame:
        """Summary of all dropped features and reasons."""
        rows = []
        for f in self.dropped_zero_var_:
            rows.append({"feature": f, "reason": "near-zero variance", "detail": ""})
        for f in self.dropped_corr_:
            rows.append({"feature": f, "reason": "high correlation", "detail": f"|r| > {self.corr_threshold}"})
        for f, vif in self.dropped_vif_:
            rows.append({"feature": f, "reason": "high VIF", "detail": f"VIF = {vif}"})
        return pd.DataFrame(rows)
