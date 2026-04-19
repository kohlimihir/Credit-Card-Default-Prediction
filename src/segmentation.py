"""
src/segmentation.py
--------------------
Customer Risk Segmentation:
  1. Risk tier assignment based on predicted PD
  2. Behavioral clustering using K-Means
  3. Business action matrix (2x2: Risk × Trend)
  4. Segment profiling and visualisation

The output answers: "Who are these customers and what should we DO about it?"
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
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_COLUMN, REPORTS_DIR,
    N_CLUSTERS, CLUSTER_RANDOM_STATE,
    RISK_TIER_THRESHOLDS,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Risk Tier Classifier
# ─────────────────────────────────────────────────────────────────────────────

def assign_risk_tiers(
    pd_scores: np.ndarray,
    thresholds: dict = None,
) -> pd.Series:
    """
    Map predicted probability of default to labelled risk tiers.
    Used for credit limit decisions, collections triggers, and reporting.
    """
    thresholds = thresholds or RISK_TIER_THRESHOLDS
    tiers = pd.Series(index=range(len(pd_scores)), dtype=str)

    for tier_name, (low, high) in thresholds.items():
        mask = (pd_scores >= low) & (pd_scores < high)
        tiers[mask] = tier_name

    tiers = tiers.fillna("Unknown")
    return tiers


def build_risk_tier_summary(
    pd_scores: np.ndarray,
    y_true: pd.Series,
) -> pd.DataFrame:
    """
    Summary table: risk tier → count, default rate, avg PD.
    Validates that the tiers are well-separated (rank ordering check).
    """
    tiers = assign_risk_tiers(pd_scores)
    df = pd.DataFrame({
        "pd_score": pd_scores,
        "risk_tier": tiers,
        "actual_default": y_true.values,
    })

    summary = (
        df.groupby("risk_tier")
        .agg(
            n_customers=("pd_score", "count"),
            avg_pd=("pd_score", "mean"),
            actual_default_rate=("actual_default", "mean"),
            pct_of_portfolio=("pd_score", "count"),
        )
        .assign(pct_of_portfolio=lambda d: d["pct_of_portfolio"] / len(df) * 100)
        .round(4)
    )

    # Force correct tier ordering
    tier_order = list(RISK_TIER_THRESHOLDS.keys())
    existing_tiers = [t for t in tier_order if t in summary.index]
    summary = summary.reindex(existing_tiers)

    logger.info(f"\nRisk Tier Summary:\n{summary.to_string()}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Behavioural Cluster Model
# ─────────────────────────────────────────────────────────────────────────────

class CustomerSegmenter:
    """
    Behavioural segmentation via K-Means clustering.

    Clusters customers into groups with similar risk profiles:
      - Transactors (low risk, pay in full)
      - Revolvers (medium risk, carry balance)
      - Stressed Revolvers (high risk, declining payments)
      - Delinquency-Prone (very high risk, chronic late payment)

    Usage:
        seg = CustomerSegmenter(n_clusters=4)
        seg.fit(X_features, pd_scores)
        labels = seg.predict(X_new, pd_new)
    """

    CLUSTER_LABEL_MAP = {
        0: "Transactors",
        1: "Revolvers",
        2: "Stressed Revolvers",
        3: "Delinquency-Prone",
    }

    # Clustering features — interpretable behavioral signals
    CLUSTER_FEATURES = [
        "UTIL_MEAN",
        "PAY_RATIO_MEAN",
        "N_MONTHS_DELINQUENT",
        "MAX_DELINQUENCY",
        "BALANCE_GROWTH_RATE",
        "MONTHS_PAID_FULL",
        "STRESS_SCORE",
        "TOTAL_PAY_RATIO",
    ]

    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters
        self.scaler_    = StandardScaler()
        self.kmeans_    = KMeans(
            n_clusters=n_clusters,
            random_state=CLUSTER_RANDOM_STATE,
            n_init=10,
            max_iter=300,
        )
        self.cluster_profiles_: pd.DataFrame = pd.DataFrame()
        self.label_map_: dict = {}
        self._fitted = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        pd_scores: np.ndarray,
        y_true: pd.Series = None,
    ) -> "CustomerSegmenter":
        """
        Fit K-Means on available clustering features.
        Uses predicted PD to label clusters in risk order.
        """
        X_cluster = self._extract_cluster_features(X)
        X_scaled  = self.scaler_.fit_transform(X_cluster)

        logger.info(f"Fitting K-Means with k={self.n_clusters}...")
        self.kmeans_.fit(X_scaled)
        cluster_labels = self.kmeans_.labels_

        # Silhouette score to validate cluster quality
        sil_score = silhouette_score(X_scaled, cluster_labels, sample_size=2000)
        logger.info(f"Silhouette Score: {sil_score:.4f}  "
                    f"({'Good' if sil_score > 0.3 else 'Moderate' if sil_score > 0.1 else 'Weak'})")

        # Build cluster profiles
        df_profile = X_cluster.copy()
        df_profile["cluster_id"]   = cluster_labels
        df_profile["pd_score"]     = pd_scores
        if y_true is not None:
            df_profile["actual_default"] = y_true.values

        # Assign business labels: rank clusters by avg PD
        cluster_pd_avg = df_profile.groupby("cluster_id")["pd_score"].mean().sort_values()
        self.label_map_ = {
            cluster_id: self.CLUSTER_LABEL_MAP.get(i, f"Segment_{i}")
            for i, cluster_id in enumerate(cluster_pd_avg.index)
        }

        # Build profiles table
        agg_dict = {
            "UTIL_MEAN":         "mean",
            "PAY_RATIO_MEAN":    "mean",
            "N_MONTHS_DELINQUENT":"mean",
            "BALANCE_GROWTH_RATE":"mean",
            "MONTHS_PAID_FULL":  "mean",
            "pd_score":          ["mean", "count"],
        }
        if y_true is not None:
            agg_dict["actual_default"] = "mean"

        # Flatten multiindex
        profiles = df_profile.groupby("cluster_id").agg(agg_dict)
        profiles.columns = ["_".join(c).strip("_") for c in profiles.columns]
        profiles["segment_name"] = profiles.index.map(self.label_map_)
        profiles = profiles.rename(columns={"pd_score_count": "n_customers"})

        # Sort by risk level
        risk_order = list(self.label_map_.values())
        profiles["_sort"] = profiles["segment_name"].map(
            {name: i for i, name in enumerate(risk_order)}
        )
        self.cluster_profiles_ = profiles.sort_values("_sort").drop("_sort", axis=1)

        logger.info(f"\nCluster Profiles:\n{self.cluster_profiles_.round(3).to_string()}")
        self._fitted = True
        return self

    def predict(
        self, X: pd.DataFrame, pd_scores: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Assign cluster labels and risk tiers to new customers.
        Returns DataFrame with cluster_id, segment_name, risk_tier, pd_score.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        X_cluster = self._extract_cluster_features(X)
        X_scaled  = self.scaler_.transform(X_cluster)
        cluster_ids = self.kmeans_.predict(X_scaled)

        result = pd.DataFrame({
            "cluster_id":   cluster_ids,
            "segment_name": [self.label_map_.get(c, "Unknown") for c in cluster_ids],
        }, index=X.index)

        if pd_scores is not None:
            result["pd_score"]  = pd_scores
            result["risk_tier"] = assign_risk_tiers(pd_scores)

        return result

    # ── Business Action Matrix ────────────────────────────────────────────────

    def build_action_matrix(
        self,
        X: pd.DataFrame,
        pd_scores: np.ndarray,
    ) -> pd.DataFrame:
        """
        2×2 Business Action Matrix:
          X-axis: Risk Level (High PD vs Low PD)
          Y-axis: Trend Direction (Deteriorating vs Stable/Improving)

        Each cell = a recommended business action for that customer group.
        """
        median_pd = np.median(pd_scores)

        # Trend = recent payment direction
        if "PAY_RATIO_MEAN" in X.columns and "PAY_RATIO_MIN" in X.columns:
            trend_feature = X["PAY_RATIO_MEAN"] - X["PAY_RATIO_MIN"]
            trend_label   = trend_feature.apply(
                lambda x: "Deteriorating" if x < -0.1 else "Stable/Improving"
            )
        else:
            trend_label = pd.Series(["Unknown"] * len(pd_scores), index=X.index)

        risk_label = pd.Series(
            ["High Risk" if p >= median_pd else "Low Risk" for p in pd_scores],
            index=X.index
        )

        df = pd.DataFrame({
            "risk_level":  risk_label,
            "trend":       trend_label,
            "pd_score":    pd_scores,
        })

        ACTIONS = {
            ("Low Risk",  "Stable/Improving"): "Limit Increase / Upsell",
            ("Low Risk",  "Deteriorating"):    "Monitor / Early Warning",
            ("High Risk", "Stable/Improving"): "Hold / Maintain Limit",
            ("High Risk", "Deteriorating"):    "Limit Decrease / Collections",
        }

        df["recommended_action"] = df.apply(
            lambda r: ACTIONS.get((r["risk_level"], r["trend"]), "Review"), axis=1
        )

        matrix = df.groupby(["risk_level", "trend"]).agg(
            n_customers=("pd_score", "count"),
            avg_pd=("pd_score", "mean"),
            recommended_action=("recommended_action", "first"),
        ).reset_index()

        logger.info(f"\nBusiness Action Matrix:\n{matrix.to_string(index=False)}")
        return matrix

    # ── Elbow Plot ────────────────────────────────────────────────────────────

    def plot_elbow(
        self, X: pd.DataFrame, k_range=(2, 9),
        save_path: str = None
    ) -> None:
        """
        Elbow plot to choose optimal number of clusters.
        Shows both inertia (elbow) and silhouette score.
        """
        X_cluster = self._extract_cluster_features(X)
        X_scaled  = self.scaler_.fit_transform(X_cluster)

        ks, inertias, sil_scores = [], [], []
        for k in range(k_range[0], k_range[1]):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            ks.append(k)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels, sample_size=2000))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Optimal Number of Clusters", fontsize=12, fontweight="bold")

        ax1.plot(ks, inertias, "o-", color="#2563EB", lw=2, markersize=7)
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia (Within-Cluster SSE)")
        ax1.set_title("Elbow Method")
        ax1.grid(alpha=0.3)

        ax2.plot(ks, sil_scores, "s-", color="#DC2626", lw=2, markersize=7)
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Silhouette Score (higher = better)")
        ax2.set_title("Silhouette Analysis")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved elbow plot to {save_path}")
        plt.close()

    def plot_cluster_profiles(
        self, save_path: str = None
    ) -> None:
        """Radar chart / heatmap of cluster feature profiles."""
        if self.cluster_profiles_.empty:
            logger.warning("No cluster profiles. Run fit() first.")
            return

        # Select numeric profile columns
        profile_cols = [
            c for c in self.cluster_profiles_.columns
            if c not in ["segment_name", "n_customers", "actual_default_mean"]
            and "pd_score" not in c
        ]
        df_plot = self.cluster_profiles_[profile_cols + ["segment_name"]].set_index("segment_name")

        # Normalize each feature to 0-1 for radar-style comparison
        df_norm = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min() + 1e-9)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            df_norm.T,
            annot=df_plot.T.round(2),
            fmt=".2f",
            cmap="RdYlGn_r",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Normalised Value (0=best, 1=worst)"},
        )
        ax.set_title("Customer Segment Profiles\n(Raw values annotated | Colour = normalised risk)", fontsize=11)
        ax.set_xlabel("Customer Segment")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved cluster profiles to {save_path}")
        plt.close()

    def plot_pca_clusters(
        self, X: pd.DataFrame, pd_scores: np.ndarray,
        save_path: str = None
    ) -> None:
        """PCA scatter plot coloured by cluster segment."""
        X_cluster = self._extract_cluster_features(X)
        X_scaled  = self.scaler_.transform(X_cluster)

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        var_exp = pca.explained_variance_ratio_

        cluster_ids = self.kmeans_.predict(X_scaled)
        labels      = [self.label_map_.get(c, "Unknown") for c in cluster_ids]

        colors_map = {
            "Transactors":       "#16A34A",
            "Revolvers":         "#2563EB",
            "Stressed Revolvers":"#D97706",
            "Delinquency-Prone": "#DC2626",
        }

        fig, ax = plt.subplots(figsize=(10, 8))
        for seg_name in set(labels):
            mask = np.array(labels) == seg_name
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors_map.get(seg_name, "#6B7280"),
                alpha=0.4, s=15,
                label=f"{seg_name} (n={mask.sum():,})",
            )
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
        ax.set_title("Customer Segments — PCA Projection\n(2D reduction of behavioural features)")
        ax.legend(markerscale=2, fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved PCA cluster plot to {save_path}")
        plt.close()

    # ── Utility ──────────────────────────────────────────────────────────────

    def _extract_cluster_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract available clustering features (handles missing columns gracefully)."""
        available = [c for c in self.CLUSTER_FEATURES if c in X.columns]
        if len(available) < 3:
            raise ValueError(
                f"Need at least 3 clustering features. Found: {available}. "
                "Run FeatureEngineer.fit_transform() before segmentation."
            )
        if len(available) < len(self.CLUSTER_FEATURES):
            missing = set(self.CLUSTER_FEATURES) - set(available)
            logger.warning(f"Clustering features missing (using subset): {missing}")
        return X[available].copy()
