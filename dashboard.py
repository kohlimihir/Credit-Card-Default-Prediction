"""
dashboard.py
============
Streamlit Credit Risk Monitoring Dashboard.

Displays all pipeline outputs — model performance, SHAP explainability,
scorecard analysis, risk segmentation, PSI/Gini stability, bias audit,
and governance report — using interactive Plotly charts.

Usage:
    streamlit run dashboard.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Monitoring Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def rpt(filename: str) -> str:
    return os.path.join(REPORTS_DIR, filename)


# ─────────────────────────────────────────────
# Custom Theme / CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a4a 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-good { color: #34d399; }
    .delta-bad  { color: #f87171; }
    .delta-neutral { color: #94a3b8; }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-green {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .badge-yellow {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    .badge-red {
        background: rgba(248, 113, 113, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(139, 92, 246, 0.3);
    }

    /* Governance text report */
    .governance-report {
        background: #0f0f1a;
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 24px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.82rem;
        line-height: 1.6;
        white-space: pre-wrap;
        max-height: 600px;
        overflow-y: auto;
    }

    /* Hide default streamlit metric styling for cleaner look */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Plotly Theme
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=50, r=30, t=50, b=50),
    hoverlabel=dict(
        bgcolor="#1e1e2f",
        font_size=13,
        font_family="Inter, sans-serif",
        bordercolor="rgba(139,92,246,0.3)",
    ),
)

COLOR_PALETTE = [
    "#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444",
    "#34d399", "#ec4899", "#3b82f6", "#f97316",
]

GRADIENT_COLORSCALE = [
    [0.0, "#06b6d4"],
    [0.5, "#8b5cf6"],
    [1.0, "#ef4444"],
]


def apply_layout(fig, title="", height=450):
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=16, color="#e2e8f0"), x=0.02),
        height=height,
        xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
    )
    return fig


# ─────────────────────────────────────────────
# Data Loaders (cached)
# ─────────────────────────────────────────────

@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = rpt(filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_json(filename: str) -> dict:
    path = rpt(filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_text(filename: str) -> str:
    path = rpt(filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# ─────────────────────────────────────────────
# Helper: Metric Card
# ─────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str = None, delta_type: str = "neutral"):
    delta_html = ""
    if delta:
        css_cls = f"delta-{delta_type}"
        delta_html = f'<div class="metric-delta {css_cls}">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    status_lower = status.lower()
    if "stable" in status_lower or "ok" in status_lower or "pass" in status_lower:
        return f'<span class="badge badge-green">● {status}</span>'
    elif "moderate" in status_lower or "warn" in status_lower:
        return f'<span class="badge badge-yellow">⚠ {status}</span>'
    else:
        return f'<span class="badge badge-red">✕ {status}</span>'


# ═════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏦 Credit Risk Monitor")
    st.markdown("---")

    model_card = load_json("model_card.json")

    if model_card:
        overview = model_card.get("model_overview", {})
        st.markdown(f"**Model:** {overview.get('name', 'N/A')}")
        st.markdown(f"**Version:** {overview.get('version', 'N/A')}")
        st.markdown(f"**Created:** {overview.get('created', 'N/A')}")
        st.markdown("---")

    page = st.radio(
        "📊 Navigation",
        [
            "🏠 Executive Summary",
            "📈 Model Performance",
            "🔍 SHAP Explainability",
            "📋 Scorecard Analysis",
            "👥 Risk Segmentation",
            "🛡️ Monitoring & Stability",
            "⚖️ Bias & Fairness",
            "📄 Governance Report",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        "Built with Streamlit + Plotly<br>"
        "Aligned with OCC SR 11-7"
        "</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════
# PAGE: Executive Summary
# ═════════════════════════════════════════════

if page == "🏠 Executive Summary":
    st.markdown("# 🏦 Credit Risk Model — Executive Summary")
    st.markdown("Real-time monitoring of model performance, stability, and governance metrics.")
    st.markdown("")

    results = load_csv("results_summary.csv")
    gini_df = load_csv("gini_stability.csv")
    psi_df = load_csv("psi_report.csv")
    bias_df = load_csv("bias_audit.csv")
    risk_df = load_csv("risk_tier_summary.csv")

    # ── Top KPI Row ──
    if not results.empty:
        xgb_test = results[(results["Model"] == "XGBoost") & (results["Dataset"] == "Test")]
        xgb_oot  = results[(results["Model"] == "XGBoost") & (results["Dataset"] == "OOT")]

        best_auc  = xgb_test["auc_roc"].values[0] if len(xgb_test) > 0 else 0
        best_gini = xgb_test["gini"].values[0] if len(xgb_test) > 0 else 0
        best_ks   = xgb_test["ks_stat"].values[0] if len(xgb_test) > 0 else 0
        oot_auc   = xgb_oot["auc_roc"].values[0] if len(xgb_oot) > 0 else 0

        score_psi_val = psi_df["psi"].max() if not psi_df.empty else 0
        gini_min = gini_df["gini"].min() if not gini_df.empty else 0
        bias_max = bias_df["auc_delta"].abs().max() if not bias_df.empty else 0

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            metric_card("AUC-ROC (Test)", f"{best_auc:.4f}",
                        "✓ Strong" if best_auc > 0.75 else "Weak", "good" if best_auc > 0.75 else "bad")
        with col2:
            metric_card("Gini (Test)", f"{best_gini:.4f}",
                        "✓ Good" if best_gini > 0.5 else "Weak", "good" if best_gini > 0.5 else "bad")
        with col3:
            metric_card("KS Stat (Test)", f"{best_ks:.4f}",
                        "✓ Good sep." if best_ks > 0.3 else "Weak", "good" if best_ks > 0.3 else "bad")
        with col4:
            metric_card("AUC-ROC (OOT)", f"{oot_auc:.4f}",
                        f"Δ {oot_auc - best_auc:+.4f}", "good" if abs(oot_auc - best_auc) < 0.03 else "bad")
        with col5:
            metric_card("Max Feature PSI", f"{score_psi_val:.4f}",
                        "🟢 STABLE" if score_psi_val < 0.1 else "⚠ DRIFT", "good" if score_psi_val < 0.1 else "bad")
        with col6:
            metric_card("Bias Max Δ AUC", f"{bias_max:.4f}",
                        "✓ Fair" if bias_max < 0.05 else "⚠ Check", "good" if bias_max < 0.05 else "bad")

    st.markdown("")

    # ── Model Comparison Chart ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        if not results.empty:
            fig = go.Figure()
            metrics_to_show = ["auc_roc", "gini", "ks_stat"]
            metric_labels = {"auc_roc": "AUC-ROC", "gini": "Gini", "ks_stat": "KS Stat"}

            for i, row in results.iterrows():
                model_label = f"{row['Model']} ({row['Dataset']})"
                vals = [row.get(m, 0) for m in metrics_to_show]
                fig.add_trace(go.Bar(
                    name=model_label,
                    x=[metric_labels[m] for m in metrics_to_show],
                    y=vals,
                    marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                    text=[f"{v:.4f}" for v in vals],
                    textposition="outside",
                    textfont=dict(size=11),
                ))
            fig = apply_layout(fig, "Model Performance Comparison", height=420)
            fig.update_layout(barmode="group", yaxis_range=[0, 1.05])
            st.plotly_chart(fig, width="stretch")

    with col_right:
        if not risk_df.empty:
            fig = go.Figure(go.Pie(
                labels=risk_df["risk_tier"],
                values=risk_df["n_customers"],
                marker=dict(colors=["#34d399", "#f59e0b", "#ef4444", "#dc2626"]),
                hole=0.55,
                textinfo="label+percent",
                textfont=dict(size=12),
                hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
            ))
            fig = apply_layout(fig, "Portfolio Risk Distribution", height=420)
            st.plotly_chart(fig, width="stretch")

    # ── Gini Stability + PSI Feature ──
    col_g, col_p = st.columns(2)

    with col_g:
        if not gini_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=gini_df["period"], y=gini_df["gini"],
                mode="lines+markers",
                name="Gini",
                line=dict(color="#8b5cf6", width=3),
                marker=dict(size=10, color="#8b5cf6", line=dict(color="#fff", width=2)),
                fill="tozeroy",
                fillcolor="rgba(139,92,246,0.1)",
            ))
            fig.add_trace(go.Scatter(
                x=gini_df["period"], y=gini_df["auc"],
                mode="lines+markers",
                name="AUC",
                line=dict(color="#06b6d4", width=3),
                marker=dict(size=10, color="#06b6d4", line=dict(color="#fff", width=2)),
            ))
            fig = apply_layout(fig, "Gini & AUC Stability Over Time", height=380)
            fig.update_layout(
                xaxis_title="Time Period",
                yaxis_title="Metric Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, width="stretch")

    with col_p:
        if not psi_df.empty:
            psi_sorted = psi_df.sort_values("psi", ascending=True).tail(15)
            colors = ["#34d399" if v < 0.1 else "#f59e0b" if v < 0.25 else "#ef4444"
                       for v in psi_sorted["psi"]]
            fig = go.Figure(go.Bar(
                x=psi_sorted["psi"],
                y=psi_sorted["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.4f}" for v in psi_sorted["psi"]],
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig.add_vline(x=0.1, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Moderate", annotation_position="top")
            fig.add_vline(x=0.25, line_dash="dash", line_color="#ef4444",
                          annotation_text="High Drift", annotation_position="top")
            fig = apply_layout(fig, "Feature PSI (Population Stability Index)", height=380)
            fig.update_layout(xaxis_title="PSI Value", yaxis_title="")
            st.plotly_chart(fig, width="stretch")


# ═════════════════════════════════════════════
# PAGE: Model Performance
# ═════════════════════════════════════════════

elif page == "📈 Model Performance":
    st.markdown("# 📈 Model Performance")

    results = load_csv("results_summary.csv")

    if not results.empty:
        st.markdown("### Performance Metrics — All Models")
        st.dataframe(
            results.style.format({
                "auc_roc": "{:.4f}", "gini": "{:.4f}",
                "ks_stat": "{:.4f}", "brier_score": "{:.4f}",
            }).background_gradient(subset=["auc_roc", "gini"], cmap="viridis"),
            width="stretch",
            height=250,
        )

        # ── Radar Chart ──
        st.markdown("### Model Comparison — Radar View")
        models_to_plot = results[results["Dataset"] == "Test"].reset_index(drop=True)

        if len(models_to_plot) > 0:
            metrics_radar = ["auc_roc", "gini", "ks_stat"]
            fig = go.Figure()

            for i, row in models_to_plot.iterrows():
                vals = [row[m] for m in metrics_radar] + [row[metrics_radar[0]]]
                cats = ["AUC-ROC", "Gini", "KS Stat", "AUC-ROC"]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=cats,
                    fill="toself",
                    name=row["Model"],
                    line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=2),
                    fillcolor=f"rgba({int(COLOR_PALETTE[i % len(COLOR_PALETTE)][1:3],16)},"
                              f"{int(COLOR_PALETTE[i % len(COLOR_PALETTE)][3:5],16)},"
                              f"{int(COLOR_PALETTE[i % len(COLOR_PALETTE)][5:7],16)},0.1)",
                ))
            fig = apply_layout(fig, "", height=480)
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[0, 1], gridcolor="rgba(148,163,184,0.1)"),
                    angularaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                ),
            )
            st.plotly_chart(fig, width="stretch")

    # ── Pipeline-generated plots ──
    st.markdown("### Detailed Evaluation Plots")

    tabs = st.tabs(["📊 Model Comparison", "📏 Calibration", "🎯 Scorecard Evaluation"])

    with tabs[0]:
        img_path = rpt("model_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, width="stretch")
        else:
            st.info("Run pipeline first to generate this plot.")

    with tabs[1]:
        img_path = rpt("calibration_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, width="stretch")

    with tabs[2]:
        img_path = rpt("scorecard_evaluation.png")
        if os.path.exists(img_path):
            st.image(img_path, width="stretch")


# ═════════════════════════════════════════════
# PAGE: SHAP Explainability
# ═════════════════════════════════════════════

elif page == "🔍 SHAP Explainability":
    st.markdown("# 🔍 SHAP Feature Explainability")
    st.markdown("Understanding what drives the model's predictions using SHapley Additive exPlanations.")

    shap_df = load_csv("top_shap_features.csv")

    if not shap_df.empty:
        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            top_n = st.slider("Top N features to display", 5, 20, 15)
            shap_plot = shap_df.head(top_n).sort_values("mean_abs_shap", ascending=True)

            fig = go.Figure(go.Bar(
                x=shap_plot["mean_abs_shap"],
                y=shap_plot["feature"],
                orientation="h",
                marker=dict(
                    color=shap_plot["mean_abs_shap"],
                    colorscale=GRADIENT_COLORSCALE,
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                ),
                text=[f"{v:.4f}" for v in shap_plot["mean_abs_shap"]],
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
            ))
            fig = apply_layout(fig, "Top SHAP Features — Mean |SHAP| Value", height=max(350, top_n * 28))
            fig.update_layout(xaxis_title="Mean |SHAP| Value", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        with col_table:
            st.markdown("### Feature Importance Table")
            styled_df = shap_df.head(top_n).reset_index(drop=True)
            styled_df.index = styled_df.index + 1
            styled_df.columns = ["Feature", "Mean |SHAP|"]
            st.dataframe(
                styled_df.style.format({"Mean |SHAP|": "{:.6f}"})
                    .bar(subset=["Mean |SHAP|"], color="rgba(139,92,246,0.3)"),
                width="stretch",
                height=min(600, top_n * 38 + 50),
            )

    # ── SHAP Plots from Pipeline ──
    st.markdown("---")
    st.markdown("### SHAP Visualizations")

    tabs = st.tabs(["🌡️ Summary Plot", "💧 Waterfall (Sample)", "📈 Dependence Plot"])

    with tabs[0]:
        img = rpt("xgb_shap_summary.png")
        if os.path.exists(img):
            st.image(img, width="stretch")

    with tabs[1]:
        img = rpt("xgb_shap_waterfall_customer0.png")
        if os.path.exists(img):
            st.image(img, width="stretch")

    with tabs[2]:
        # Find all dependence plot files
        dep_plots = sorted(Path(REPORTS_DIR).glob("shap_dependence_*.png"))
        if dep_plots:
            for dp in dep_plots:
                feat_name = dp.stem.replace("shap_dependence_", "")
                st.markdown(f"**Feature:** `{feat_name}`")
                st.image(str(dp), width="stretch")
        else:
            st.info("No SHAP dependence plots found.")


# ═════════════════════════════════════════════
# PAGE: Scorecard Analysis
# ═════════════════════════════════════════════

elif page == "📋 Scorecard Analysis":
    st.markdown("# 📋 Scorecard Analysis (WoE + Logistic Regression)")

    iv_df = load_csv("iv_table.csv")
    sc_df = load_csv("scorecard_points_table.csv")

    # ── IV Chart ──
    if not iv_df.empty:
        st.markdown("### Information Value (IV) — Feature Selection")

        col_iv, col_summary = st.columns([3, 1])

        with col_iv:
            iv_sorted = iv_df.sort_values("iv", ascending=True).tail(25)

            colors = []
            for _, row in iv_sorted.iterrows():
                flag = str(row.get("flag", ""))
                if "SUSPICIOUS" in flag:
                    colors.append("#ef4444")
                elif row["iv"] >= 0.3:
                    colors.append("#8b5cf6")
                elif row["iv"] >= 0.1:
                    colors.append("#06b6d4")
                elif row["iv"] >= 0.02:
                    colors.append("#34d399")
                else:
                    colors.append("#64748b")

            fig = go.Figure(go.Bar(
                x=iv_sorted["iv"],
                y=iv_sorted["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in iv_sorted["iv"]],
                textposition="outside",
                textfont=dict(size=10),
                hovertemplate="<b>%{y}</b><br>IV: %{x:.4f}<extra></extra>",
            ))
            fig.add_vline(x=0.02, line_dash="dot", line_color="#64748b",
                          annotation_text="Weak (0.02)", annotation_position="top right")
            fig.add_vline(x=0.1, line_dash="dot", line_color="#06b6d4",
                          annotation_text="Medium (0.1)", annotation_position="top right")
            fig.add_vline(x=0.3, line_dash="dot", line_color="#8b5cf6",
                          annotation_text="Strong (0.3)", annotation_position="top right")
            fig.add_vline(x=0.5, line_dash="dash", line_color="#ef4444",
                          annotation_text="Suspicious (0.5)", annotation_position="top right")
            fig = apply_layout(fig, "Feature Information Value (IV)", height=650)
            fig.update_layout(xaxis_title="IV", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        with col_summary:
            st.markdown("#### IV Interpretation")
            st.markdown("""
            | IV Range | Predictive Power |
            |----------|:---:|
            | < 0.02 | ❌ Useless |
            | 0.02 – 0.1 | 🟡 Weak |
            | 0.1 – 0.3 | 🟢 Medium |
            | 0.3 – 0.5 | 🔵 Strong |
            | > 0.5 | 🔴 Suspicious |
            """)

            n_suspicious = iv_df[iv_df["flag"].str.contains("SUSPICIOUS", na=False)].shape[0]
            n_selected = iv_df[iv_df["iv"] >= 0.02].shape[0]
            n_dropped = iv_df[iv_df["iv"] < 0.02].shape[0]

            st.metric("Total Features", len(iv_df))
            st.metric("Selected (IV ≥ 0.02)", n_selected)
            st.metric("Dropped (low IV)", n_dropped)
            st.metric("⚠ Suspicious (IV > 0.5)", n_suspicious)

    # ── Scorecard Points Table ──
    if not sc_df.empty:
        st.markdown("---")
        st.markdown("### Scorecard Points Table")

        features_in_sc = sc_df["Feature"].unique().tolist()
        selected_feat = st.selectbox("Select feature to view bins:", features_in_sc)

        feat_data = sc_df[sc_df["Feature"] == selected_feat].reset_index(drop=True)

        col_tbl, col_viz = st.columns([2, 2])

        with col_tbl:
            st.dataframe(
                feat_data.style.format({
                    "Event_Rate": "{:.4f}", "WoE": "{:.4f}",
                    "Coefficient": "{:.4f}", "Points": "{:.1f}",
                }).bar(subset=["Points"], color=["rgba(239,68,68,0.3)", "rgba(52,211,153,0.3)"]),
                width="stretch",
            )

        with col_viz:
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("WoE by Bin", "Points by Bin"),
                                horizontal_spacing=0.15)
            bin_labels = [str(b) for b in feat_data["Bin"]]

            fig.add_trace(go.Bar(
                x=bin_labels, y=feat_data["WoE"],
                marker_color=["#ef4444" if w > 0 else "#34d399" for w in feat_data["WoE"]],
                name="WoE", showlegend=False,
            ), row=1, col=1)

            fig.add_trace(go.Bar(
                x=bin_labels, y=feat_data["Points"],
                marker_color=["#34d399" if p > 0 else "#ef4444" for p in feat_data["Points"]],
                name="Points", showlegend=False,
            ), row=1, col=2)

            fig = apply_layout(fig, "", height=350)
            st.plotly_chart(fig, width="stretch")

    # ── Pipeline IV chart image ──
    img = rpt("iv_chart.png")
    if os.path.exists(img):
        with st.expander("📊 Pipeline-generated IV Chart"):
            st.image(img, width="stretch")


# ═════════════════════════════════════════════
# PAGE: Risk Segmentation
# ═════════════════════════════════════════════

elif page == "👥 Risk Segmentation":
    st.markdown("# 👥 Customer Risk Segmentation")

    risk_df = load_csv("risk_tier_summary.csv")
    action_df = load_csv("business_action_matrix.csv")

    if not risk_df.empty:
        st.markdown("### Risk Tier Summary")

        # ── Risk tier KPIs ──
        cols = st.columns(len(risk_df))
        tier_colors = {
            "Low Risk": "#34d399", "Medium Risk": "#f59e0b",
            "High Risk": "#ef4444", "Very High Risk": "#dc2626"
        }
        for i, (_, row) in enumerate(risk_df.iterrows()):
            with cols[i]:
                tier_name = row["risk_tier"]
                color = tier_colors.get(tier_name, "#8b5cf6")
                st.markdown(f"""
                <div class="metric-card" style="border-color: {color}40;">
                    <div class="metric-label">{tier_name}</div>
                    <div class="metric-value" style="background: {color}; -webkit-background-clip: text;">{row['n_customers']:,}</div>
                    <div class="metric-delta" style="color: {color}">
                        PD: {row['avg_pd']:.2%} &nbsp;|&nbsp; Default: {row['actual_default_rate']:.2%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Dual charts ──
        col_bar, col_scatter = st.columns(2)

        with col_bar:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=risk_df["risk_tier"],
                y=risk_df["avg_pd"],
                name="Avg Predicted PD",
                marker_color="#8b5cf6",
                text=[f"{v:.2%}" for v in risk_df["avg_pd"]],
                textposition="outside",
            ))
            fig.add_trace(go.Bar(
                x=risk_df["risk_tier"],
                y=risk_df["actual_default_rate"],
                name="Actual Default Rate",
                marker_color="#06b6d4",
                text=[f"{v:.2%}" for v in risk_df["actual_default_rate"]],
                textposition="outside",
            ))
            fig = apply_layout(fig, "Predicted PD vs Actual Default Rate", height=420)
            fig.update_layout(barmode="group", yaxis_tickformat=".0%",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, width="stretch")

        with col_scatter:
            fig = go.Figure(go.Scatter(
                x=risk_df["avg_pd"],
                y=risk_df["actual_default_rate"],
                mode="markers+text",
                text=risk_df["risk_tier"],
                textposition="top center",
                marker=dict(
                    size=risk_df["n_customers"] / risk_df["n_customers"].max() * 60 + 15,
                    color=[tier_colors.get(t, "#8b5cf6") for t in risk_df["risk_tier"]],
                    line=dict(color="#fff", width=2),
                ),
                hovertemplate="<b>%{text}</b><br>Avg PD: %{x:.2%}<br>Actual: %{y:.2%}<extra></extra>",
            ))
            # 45° line
            max_val = max(risk_df["avg_pd"].max(), risk_df["actual_default_rate"].max()) * 1.1
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", line=dict(dash="dash", color="#64748b"),
                showlegend=False,
            ))
            fig = apply_layout(fig, "Calibration: PD vs Actual (Bubble = Pop.)", height=420)
            fig.update_layout(xaxis_title="Avg Predicted PD", yaxis_title="Actual Default Rate",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

    # ── Business Action Matrix ──
    if not action_df.empty:
        st.markdown("### 🎯 Business Action Matrix")
        st.dataframe(
            action_df.style.applymap(
                lambda v: "color: #34d399" if "Upsell" in str(v) or "Increase" in str(v)
                else ("color: #ef4444" if "Reduce" in str(v) or "Review" in str(v) else ""),
                subset=["recommended_action"]
            ),
            width="stretch",
        )

    # ── Pipeline cluster images ──
    st.markdown("---")
    tabs = st.tabs(["📊 Cluster Profiles", "🎯 PCA Clusters", "📉 Elbow Plot"])

    with tabs[0]:
        img = rpt("cluster_profiles.png")
        if os.path.exists(img):
            st.image(img, width="stretch")

    with tabs[1]:
        img = rpt("pca_clusters.png")
        if os.path.exists(img):
            st.image(img, width="stretch")

    with tabs[2]:
        img = rpt("elbow_plot.png")
        if os.path.exists(img):
            st.image(img, width="stretch")


# ═════════════════════════════════════════════
# PAGE: Monitoring & Stability
# ═════════════════════════════════════════════

elif page == "🛡️ Monitoring & Stability":
    st.markdown("# 🛡️ Model Monitoring & Stability")

    gini_df = load_csv("gini_stability.csv")
    psi_df = load_csv("psi_report.csv")

    # ── Gini Stability ──
    if not gini_df.empty:
        st.markdown("### Gini Coefficient Stability Over Time")

        col_chart, col_detail = st.columns([3, 1])

        with col_chart:
            fig = go.Figure()

            # Gini line
            fig.add_trace(go.Scatter(
                x=gini_df["period"], y=gini_df["gini"],
                mode="lines+markers+text",
                name="Gini",
                line=dict(color="#8b5cf6", width=3, shape="spline"),
                marker=dict(size=12, color="#8b5cf6", symbol="circle",
                            line=dict(color="#fff", width=2)),
                text=[f"{v:.3f}" for v in gini_df["gini"]],
                textposition="top center",
                textfont=dict(size=11),
                fill="tozeroy",
                fillcolor="rgba(139,92,246,0.08)",
            ))

            # AUC line
            fig.add_trace(go.Scatter(
                x=gini_df["period"], y=gini_df["auc"],
                mode="lines+markers",
                name="AUC-ROC",
                line=dict(color="#06b6d4", width=3, shape="spline"),
                marker=dict(size=12, color="#06b6d4", symbol="diamond",
                            line=dict(color="#fff", width=2)),
            ))

            # Event rate on secondary axis
            fig.add_trace(go.Bar(
                x=gini_df["period"], y=gini_df["event_rate"],
                name="Event Rate",
                marker_color="rgba(249,115,22,0.3)",
                yaxis="y2",
            ))

            fig = apply_layout(fig, "Gini, AUC & Event Rate by Time Period", height=450)
            fig.update_layout(
                xaxis_title="Time Period",
                yaxis=dict(title="Gini / AUC", range=[0.4, 1.0]),
                yaxis2=dict(title="Event Rate", overlaying="y", side="right",
                            range=[0, 0.5], tickformat=".0%",
                            gridcolor="rgba(0,0,0,0)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, width="stretch")

        with col_detail:
            st.markdown("#### Period Details")
            for _, row in gini_df.iterrows():
                alert = row.get("alert", "")
                badge = status_badge("STABLE" if "OK" in str(alert) else "DRIFT")
                st.markdown(
                    f"**Period {int(row['period'])}** {badge}<br>"
                    f"Gini: `{row['gini']:.4f}` | Δ: `{row['gini_delta_from_baseline']:+.4f}`",
                    unsafe_allow_html=True,
                )
                st.markdown("")

    # ── PSI Report ──
    if not psi_df.empty:
        st.markdown("---")
        st.markdown("### Population Stability Index (PSI) — Feature Drift")

        col_psi_chart, col_psi_tbl = st.columns([2, 1])

        with col_psi_chart:
            psi_sorted = psi_df.sort_values("psi", ascending=False)

            fig = go.Figure()
            colors = ["#34d399" if v < 0.1 else "#f59e0b" if v < 0.25 else "#ef4444"
                       for v in psi_sorted["psi"]]

            fig.add_trace(go.Bar(
                x=psi_sorted["feature"],
                y=psi_sorted["psi"],
                marker_color=colors,
                text=[f"{v:.4f}" for v in psi_sorted["psi"]],
                textposition="outside",
                textfont=dict(size=10),
                hovertemplate="<b>%{x}</b><br>PSI: %{y:.4f}<extra></extra>",
            ))
            fig.add_hline(y=0.1, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Moderate (0.10)")
            fig.add_hline(y=0.25, line_dash="dash", line_color="#ef4444",
                          annotation_text="High Drift (0.25)")
            fig = apply_layout(fig, "PSI per Feature (Train → OOT)", height=450)
            fig.update_layout(xaxis_tickangle=-45, xaxis_title="", yaxis_title="PSI")
            st.plotly_chart(fig, width="stretch")

        with col_psi_tbl:
            st.markdown("#### PSI Thresholds")
            st.markdown("""
            | PSI Range | Interpretation |
            |-----------|:---:|
            | < 0.10 | 🟢 Stable |
            | 0.10 – 0.25 | 🟡 Moderate shift |
            | > 0.25 | 🔴 Significant drift |
            """)

            n_stable = (psi_df["psi"] < 0.1).sum()
            n_moderate = ((psi_df["psi"] >= 0.1) & (psi_df["psi"] < 0.25)).sum()
            n_high = (psi_df["psi"] >= 0.25).sum()

            st.metric("🟢 Stable", n_stable)
            st.metric("🟡 Moderate", n_moderate)
            st.metric("🔴 High Drift", n_high)

    # ── Pipeline monitoring dashboard image ──
    img = rpt("monitoring_dashboard.png")
    if os.path.exists(img):
        with st.expander("📊 Full Pipeline Monitoring Dashboard"):
            st.image(img, width="stretch")


# ═════════════════════════════════════════════
# PAGE: Bias & Fairness
# ═════════════════════════════════════════════

elif page == "⚖️ Bias & Fairness":
    st.markdown("# ⚖️ Bias & Fairness Audit")
    st.markdown("Evaluating model fairness across protected demographic groups.")

    bias_df = load_csv("bias_audit.csv")

    if not bias_df.empty:
        # Separate overall from subgroups
        overall = bias_df[bias_df["group_col"] == "OVERALL"]
        subgroups = bias_df[bias_df["group_col"] != "OVERALL"].copy()

        # ── Overall metrics ──
        if len(overall) > 0:
            ov = overall.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_card("Overall AUC", f"{ov['auc']:.4f}")
            with col2:
                metric_card("Overall Gini", f"{ov['gini']:.4f}")
            with col3:
                metric_card("Overall Event Rate", f"{ov['event_rate']:.2%}")
            with col4:
                max_delta = subgroups["auc_delta"].abs().max() if len(subgroups) > 0 else 0
                metric_card("Max AUC Δ", f"{max_delta:.4f}",
                            "✓ Fair" if max_delta < 0.05 else "⚠ Review",
                            "good" if max_delta < 0.05 else "bad")

        st.markdown("")

        if len(subgroups) > 0:
            # ── AUC Delta Chart ──
            col_auc, col_er = st.columns(2)

            with col_auc:
                fig = go.Figure()
                group_labels = subgroups.apply(
                    lambda r: f"{r['group_col']}: {r['group_val']}", axis=1
                )
                colors = ["#34d399" if abs(d) < 0.03 else "#f59e0b" if abs(d) < 0.05 else "#ef4444"
                           for d in subgroups["auc_delta"]]

                fig.add_trace(go.Bar(
                    x=group_labels,
                    y=subgroups["auc_delta"],
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in subgroups["auc_delta"]],
                    textposition="outside",
                    textfont=dict(size=11),
                    hovertemplate="<b>%{x}</b><br>AUC Δ: %{y:+.4f}<extra></extra>",
                ))
                fig.add_hline(y=0.05, line_dash="dash", line_color="#ef4444", annotation_text="+0.05 threshold")
                fig.add_hline(y=-0.05, line_dash="dash", line_color="#ef4444", annotation_text="-0.05 threshold")
                fig.add_hline(y=0, line_color="#64748b", line_width=1)
                fig = apply_layout(fig, "AUC Delta by Demographic Subgroup", height=420)
                fig.update_layout(xaxis_tickangle=-30, xaxis_title="",
                                  yaxis_title="AUC Δ from Overall")
                st.plotly_chart(fig, width="stretch")

            with col_er:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=group_labels,
                    y=subgroups["event_rate"],
                    marker=dict(
                        color=subgroups["event_rate"],
                        colorscale=GRADIENT_COLORSCALE,
                    ),
                    text=[f"{v:.2%}" for v in subgroups["event_rate"]],
                    textposition="outside",
                    textfont=dict(size=11),
                    hovertemplate="<b>%{x}</b><br>Event Rate: %{y:.2%}<br>N: %{customdata:,}<extra></extra>",
                    customdata=subgroups["n"],
                ))
                if len(overall) > 0:
                    fig.add_hline(y=ov["event_rate"], line_dash="dash", line_color="#8b5cf6",
                                  annotation_text=f"Overall: {ov['event_rate']:.2%}")
                fig = apply_layout(fig, "Event Rate by Demographic Subgroup", height=420)
                fig.update_layout(xaxis_tickangle=-30, xaxis_title="",
                                  yaxis_title="Event Rate", yaxis_tickformat=".0%")
                st.plotly_chart(fig, width="stretch")

            # ── Detailed table ──
            st.markdown("### Detailed Bias Audit Table")
            display_df = bias_df.copy()
            display_df.columns = ["Group", "Value", "N", "AUC", "Gini", "Event Rate", "AUC Δ"]
            st.dataframe(
                display_df.style.format({
                    "N": "{:,}", "AUC": "{:.4f}", "Gini": "{:.4f}",
                    "Event Rate": "{:.4f}", "AUC Δ": "{:+.4f}",
                }).applymap(
                    lambda v: "background-color: rgba(248,113,113,0.2)" if isinstance(v, (int, float)) and abs(v) > 0.05 else "",
                    subset=["AUC Δ"]
                ),
                width="stretch",
            )

            # ── Population distribution ──
            st.markdown("### Population Distribution by Group")
            group_cols_unique = subgroups["group_col"].unique()
            tabs = st.tabs([f"📊 {g}" for g in group_cols_unique])

            for tab, grp in zip(tabs, group_cols_unique):
                with tab:
                    grp_data = subgroups[subgroups["group_col"] == grp]
                    fig = go.Figure(go.Pie(
                        labels=[str(v) for v in grp_data["group_val"]],
                        values=grp_data["n"],
                        marker=dict(colors=COLOR_PALETTE[:len(grp_data)]),
                        hole=0.5,
                        textinfo="label+percent",
                        hovertemplate="<b>%{label}</b><br>N: %{value:,}<br>%{percent}<extra></extra>",
                    ))
                    fig = apply_layout(fig, f"{grp} — Population Distribution", height=350)
                    st.plotly_chart(fig, width="stretch")


# ═════════════════════════════════════════════
# PAGE: Governance Report
# ═════════════════════════════════════════════

elif page == "📄 Governance Report":
    st.markdown("# 📄 Model Governance Report")

    model_card = load_json("model_card.json")
    report_text = load_text("model_governance_report.txt")

    if model_card:
        # ── Structured Model Card view ──
        st.markdown("### 📇 Model Card")

        tabs = st.tabs(["🏗️ Overview", "📊 Performance", "🔒 Governance", "📋 Full Report"])

        with tabs[0]:
            overview = model_card.get("model_overview", {})
            data_info = model_card.get("data", {})
            arch = model_card.get("model_architecture", {})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Model Overview")
                for k, v in overview.items():
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

            with col2:
                st.markdown("#### Data")
                for k, v in data_info.items():
                    if isinstance(v, dict):
                        continue
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

            st.markdown("---")
            st.markdown("#### Model Architecture")
            for k, v in arch.items():
                if isinstance(v, dict):
                    st.markdown(f"**{k.replace('_', ' ').title()}:**")
                    for kk, vv in v.items():
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {kk}: `{vv}`")
                else:
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

        with tabs[1]:
            perf = model_card.get("performance", {})
            if perf:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### Cross-Validation")
                    cv = perf.get("cross_validation", {})
                    for k, v in cv.items():
                        st.metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v))

                with col2:
                    st.markdown("#### In-Sample Test")
                    ist = perf.get("in_sample_test", {})
                    for k, v in ist.items():
                        st.metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v))

                with col3:
                    st.markdown("#### Out-of-Time Test")
                    oot = perf.get("out_of_time_test", {})
                    for k, v in oot.items():
                        if isinstance(v, float):
                            st.metric(k.replace("_", " ").title(), f"{v:.4f}")
                        else:
                            st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

            # Top features from model card
            top_feats = model_card.get("top_predictive_features", [])
            if top_feats:
                st.markdown("---")
                st.markdown("#### Top 10 Predictive Features")
                feat_df = pd.DataFrame(top_feats)
                st.dataframe(feat_df.style.format({"mean_abs_shap": "{:.6f}"}),
                             width="stretch")

        with tabs[2]:
            st.markdown("#### Fairness & Bias")
            fairness = model_card.get("fairness_bias_audit", {})
            if fairness:
                for k, v in fairness.items():
                    if isinstance(v, list):
                        st.markdown(f"**{k.replace('_', ' ').title()}:** {', '.join(str(x) for x in v)}")
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            st.markdown(f"**{kk.replace('_', ' ').title()}:** {vv}")
                    else:
                        st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

            st.markdown("---")
            st.markdown("#### Stability & Monitoring")
            stab = model_card.get("performance", {}).get("gini_stability", {})
            if stab:
                for k, v in stab.items():
                    if isinstance(v, float):
                        st.metric(k.replace("_", " ").title(), f"{v:.4f}")
                    else:
                        st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

            limits = model_card.get("limitations_risks", {})
            if limits:
                st.markdown("---")
                st.markdown("#### Limitations & Risks")
                for k, v in limits.items():
                    if isinstance(v, list):
                        for item in v:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

        with tabs[3]:
            if report_text:
                st.markdown(
                    f'<div class="governance-report">{report_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Governance report text file not found.")

    # ── Download section ──
    st.markdown("---")
    st.markdown("### 📥 Download Reports")

    col1, col2, col3 = st.columns(3)

    with col1:
        if model_card:
            st.download_button(
                "⬇️ Model Card (JSON)",
                json.dumps(model_card, indent=2),
                "model_card.json",
                "application/json",
            )
    with col2:
        if report_text:
            st.download_button(
                "⬇️ Governance Report (TXT)",
                report_text,
                "model_governance_report.txt",
                "text/plain",
            )
    with col3:
        results = load_csv("results_summary.csv")
        if not results.empty:
            st.download_button(
                "⬇️ Results Summary (CSV)",
                results.to_csv(index=False),
                "results_summary.csv",
                "text/csv",
            )


# ═════════════════════════════════════════════
# Footer
# ═════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.8rem; padding:16px;'>"
    "🏦 Credit Risk Monitoring Dashboard &nbsp;|&nbsp; "
    "Aligned with OCC SR 11-7 Model Risk Management &nbsp;|&nbsp; "
    "Powered by Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)

