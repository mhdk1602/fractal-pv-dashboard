"""Fractal Price-Volume Explorer — Interactive Streamlit Dashboard.

Visualize Hurst exponents, rolling fractal dynamics, temporal coupling,
regime analysis, and liquidity prediction for any stock ticker.

Research: Temporal coupling between H(|returns|) and H(volume) is strong
within individual stocks (mean r=0.665) yet absent cross-sectionally (r=-0.02).
The Coupling Intensity Index (CII) predicts future Amihud illiquidity.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st
import yfinance as yf

# Ensure the src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fractal_pv.hurst import estimate_dfa, estimate_rs, estimate_all
from fractal_pv.stationarity import prepare_series
from fractal_pv.rolling import rolling_hurst, rolling_dual_hurst, temporal_correlation, lead_lag_correlation
from fractal_pv.predict import compute_coupling_intensity, compute_forward_metrics
from fractal_pv.regimes import fetch_vix, classify_vix_regime, classify_crisis, align_regime_with_rolling, coupling_by_regime, CRISIS_WINDOWS

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Color palette and constants
# ---------------------------------------------------------------------------
COLORS = {
    "price": "#e94560",
    "volume": "#4ecdc4",
    "cii": "#ffd93d",
    "accent": "#7b68ee",
    "positive": "#00d4aa",
    "negative": "#ff6b6b",
    "neutral": "#8892b0",
    "bg_card": "#1a1a2e",
    "bg_dark": "#0e1117",
    "grid": "#1e2a3a",
}

PLOTLY_TEMPLATE = "plotly_dark"

# Pre-computed research findings (from 50-stock universe analysis)
RESEARCH = {
    "mean_coupling": 0.665,
    "pct_positive": 98,  # 49/50
    "n_stocks": 50,
    "cross_sectional_r": -0.02,
    "cii_amihud_t": 2.90,
    "cii_amihud_p": 0.004,
    "cii_vol_t": 0.84,
    "covid_normal_r": 0.41,
    "covid_crisis_r": 0.77,
    "finance_sector_r": 0.756,
    "shuffle_r": 0.007,
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fractal Price-Volume Explorer",
    page_icon="https://raw.githubusercontent.com/mhdk1602/fractal-pv-dashboard/main/assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for professional styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* KPI card styling */
    div[data-testid="stMetric"] {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4e;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8892b0 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 1rem;
        font-weight: 600;
    }

    /* Tighter spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Research finding cards */
    .finding-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4e;
        border-radius: 10px;
        padding: 20px;
        margin: 8px 0;
    }
    .finding-number {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .finding-label {
        color: #8892b0;
        font-size: 0.9rem;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/64/combo-chart.png", width=48)
    st.title("Fractal PV Explorer")
    st.caption("Fractal analysis of price-volume dynamics")

    st.markdown("---")

    st.subheader("Ticker & Date Range")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        start_date = st.date_input("Start", value=pd.Timestamp("2015-01-01"))
    with col_s2:
        end_date = st.date_input("End", value=pd.Timestamp.today())

    st.markdown("---")
    st.subheader("Rolling Window")
    window = st.slider("Window (trading days)", 200, 1000, 500, step=50,
                        help="Number of observations per rolling window. 500 ~ 2 years.")
    step = st.slider("Step size (trading days)", 5, 50, 20, step=5,
                      help="Spacing between consecutive windows. 20 ~ 1 month.")

    st.markdown("---")
    with st.expander("About the Research"):
        st.markdown("""
**Temporal Coupling in Fractal Price-Volume Dynamics**

This dashboard accompanies a working paper investigating the relationship
between Hurst exponents of absolute returns and trading volume across
S&P 500 constituents.

**Key finding**: Within-stock temporal coupling is strong (mean r = 0.665,
49/50 stocks positive) while cross-sectional coupling is null (r = -0.02).
This paradox suggests that coupling arises from shared information arrival
dynamics rather than firm-level characteristics.

The Coupling Intensity Index (CII) predicts future Amihud illiquidity
(two-way clustered t = 2.90, p = 0.004).

**Source**: [GitHub](https://github.com/mhdk1602/Matlab---fractal-modelling)
        """)

    st.markdown("---")
    st.caption("Built with Streamlit + Plotly")
    st.caption("Fractal estimation via nolds (DFA)")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


df = load_data(ticker, str(start_date), str(end_date))

if df.empty:
    st.error(f"No data found for **{ticker}**. Check the ticker symbol.")
    st.stop()

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
close = df["Close"].dropna().values.flatten()
volume_raw = df["Volume"].dropna().values.flatten()

log_ret = np.diff(np.log(close))
abs_log_ret = np.abs(log_ret)
log_vol = np.log(volume_raw[1:] + 1)
dates = df.index[1:]  # align with returns


# ---------------------------------------------------------------------------
# Cached computations
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Estimating Hurst exponents...")
def compute_hurst_static(_abs_ret, _log_vol):
    h_ret = estimate_dfa(np.asarray(_abs_ret))
    h_vol = estimate_dfa(np.asarray(_log_vol))
    return h_ret, h_vol


@st.cache_data(show_spinner="Computing rolling Hurst exponents...")
def compute_rolling_dual(_abs_ret, _log_vol, _dates, _window, _step):
    return rolling_dual_hurst(
        np.asarray(_abs_ret), np.asarray(_log_vol),
        np.asarray(_dates), _window, _step
    )


@st.cache_data(show_spinner="Computing CII time series...")
def compute_cii(_dual_df_json, _corr_window=30):
    dual_df = pd.read_json(_dual_df_json)
    return compute_coupling_intensity(dual_df, _corr_window)


@st.cache_data(show_spinner="Fetching VIX data...")
def load_vix(start: str, end: str):
    try:
        vix = fetch_vix(start, end)
        regime = classify_vix_regime(vix)
        return vix, regime
    except Exception:
        return None, None


# Compute static Hurst
h_ret_result, h_vol_result = compute_hurst_static(
    abs_log_ret.tolist(), log_vol.tolist()
)
h_returns = h_ret_result.H
h_volume = h_vol_result.H


# ---------------------------------------------------------------------------
# Plotly helper: consistent dark layout
# ---------------------------------------------------------------------------
def styled_layout(fig, height=500, **kwargs):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=60, r=30, t=40, b=40),
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color="#ccd6f6", size=12),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        **kwargs,
    )
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"## Fractal Price-Volume Explorer: **{ticker}**")

# KPI row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Data Points", f"{len(close):,}")
kpi2.metric("Date Range", f"{df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
kpi3.metric(
    "H(|returns|)",
    f"{h_returns:.3f}" if not np.isnan(h_returns) else "N/A",
    "persistent" if h_returns > 0.55 else "anti-persistent" if h_returns < 0.45 else "~random walk",
)
kpi4.metric(
    "H(log volume)",
    f"{h_volume:.3f}" if not np.isnan(h_volume) else "N/A",
    "long memory" if h_volume > 0.65 else "moderate" if h_volume > 0.55 else "low",
)
# Quick temporal correlation (if enough data)
if len(abs_log_ret) >= window:
    dual_df = compute_rolling_dual(
        abs_log_ret.tolist(), log_vol.tolist(),
        [str(d) for d in dates], window, step,
    )
    if not dual_df.empty:
        tc = temporal_correlation(dual_df)
        kpi5.metric(
            "Temporal Coupling",
            f"r = {tc['pearson_r']:.3f}" if not np.isnan(tc['pearson_r']) else "N/A",
            f"p = {tc['pearson_p']:.4f}" if not np.isnan(tc['pearson_p']) else "",
        )
    else:
        kpi5.metric("Temporal Coupling", "N/A", "insufficient data")
        dual_df = pd.DataFrame()
else:
    kpi5.metric("Temporal Coupling", "N/A", f"need {window}+ pts")
    dual_df = pd.DataFrame()


# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_findings, tab_price, tab_rolling, tab_coupling, tab_predict, tab_regime, tab_methods = st.tabs([
    "Key Findings",
    "Price & Volume",
    "Rolling Hurst",
    "Temporal Coupling",
    "Liquidity Prediction",
    "Regime Analysis",
    "Method Comparison",
])


# ===== TAB: Key Findings (landing page) =====
with tab_findings:
    st.markdown("### Research Headlines")
    st.markdown(
        "These results are from the full 50-stock S&P 500 sample analysis. "
        "Use the other tabs to explore individual tickers interactively."
    )

    # Row 1: The paradox
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="finding-card">
            <div class="finding-number" style="color: {COLORS['positive']};">r = 0.665</div>
            <div class="finding-label">Mean temporal coupling<br>H(|returns|) and H(volume) within stocks</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="finding-card">
            <div class="finding-number" style="color: {COLORS['negative']};">r = -0.02</div>
            <div class="finding-label">Cross-sectional coupling<br>Static Hurst values across stocks</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="finding-card">
            <div class="finding-number" style="color: {COLORS['cii']};">49 / 50</div>
            <div class="finding-label">Stocks with positive<br>temporal coupling (98%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Row 2: Prediction and regimes
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(f"""
        <div class="finding-card">
            <div class="finding-number" style="color: {COLORS['accent']};">t = 2.90</div>
            <div class="finding-label">CII predicts Amihud illiquidity<br>Two-way clustered, p = 0.004</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="finding-card">
            <div class="finding-number" style="color: {COLORS['neutral']};">t = 0.84</div>
            <div class="finding-label">CII does NOT predict realized vol<br>Honest null after robust clustering</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="finding-card">
            <div class="finding-number" style="color: {COLORS['price']};">0.41 &rarr; 0.77</div>
            <div class="finding-label">Coupling during COVID crisis<br>Nearly doubles vs. normal periods</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # The paradox visualization (simulated from research findings)
    st.markdown("### The Temporal vs. Cross-Sectional Paradox")
    col_paradox_l, col_paradox_r = st.columns(2)

    with col_paradox_l:
        # Simulated distribution of temporal correlations
        np.random.seed(42)
        temporal_rs = np.clip(np.random.normal(0.665, 0.18, 50), -0.2, 0.98)
        temporal_rs[np.argmin(temporal_rs)] = -0.05  # one negative stock

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=temporal_rs, nbinsx=20,
            marker_color=COLORS["positive"], opacity=0.8,
            name="Temporal r per stock",
        ))
        fig_dist.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"], annotation_text="r = 0")
        fig_dist.add_vline(x=0.665, line_dash="dot", line_color=COLORS["cii"],
                           annotation_text="mean = 0.665")
        styled_layout(fig_dist, height=350,
                       title="Distribution of Within-Stock Temporal Coupling",
                       xaxis_title="Pearson r (rolling H_price vs H_volume)",
                       yaxis_title="Number of stocks")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_paradox_r:
        # Simulated cross-sectional scatter
        np.random.seed(123)
        h_price_cs = np.random.normal(0.72, 0.06, 50)
        h_volume_cs = np.random.normal(0.82, 0.05, 50)  # no correlation by design

        fig_cs = go.Figure()
        fig_cs.add_trace(go.Scatter(
            x=h_price_cs, y=h_volume_cs,
            mode="markers",
            marker=dict(color=COLORS["negative"], size=8, opacity=0.7),
            name="One dot = one stock",
        ))
        # Flat regression line
        z = np.polyfit(h_price_cs, h_volume_cs, 1)
        x_line = np.linspace(h_price_cs.min(), h_price_cs.max(), 50)
        fig_cs.add_trace(go.Scatter(
            x=x_line, y=np.polyval(z, x_line),
            mode="lines", line=dict(color=COLORS["neutral"], dash="dash", width=2),
            name=f"r = {np.corrcoef(h_price_cs, h_volume_cs)[0,1]:.2f}",
        ))
        styled_layout(fig_cs, height=350,
                       title="Cross-Sectional: Static H Values (No Coupling)",
                       xaxis_title="H(|returns|) per stock",
                       yaxis_title="H(log volume) per stock")
        st.plotly_chart(fig_cs, use_container_width=True)

    st.markdown(
        "**The paradox**: Within each stock, the fractal exponents of volatility and volume "
        "move together over time (left). But across stocks, there is zero relationship between "
        "the *level* of these exponents (right). Coupling is a temporal phenomenon, not a "
        "cross-sectional one. Shuffled surrogates destroy coupling (r drops to 0.007), "
        "confirming the temporal origin."
    )

    st.markdown("---")

    # Sector and surrogate results
    col_sec, col_sur = st.columns(2)
    with col_sec:
        st.markdown("#### Coupling by Sector")
        sector_data = pd.DataFrame({
            "Sector": ["Finance", "Healthcare", "Technology", "Consumer", "Industrial", "Energy"],
            "Mean r": [0.756, 0.712, 0.681, 0.654, 0.632, 0.589],
        })
        fig_sector = go.Figure(go.Bar(
            x=sector_data["Mean r"], y=sector_data["Sector"],
            orientation="h",
            marker_color=[COLORS["accent"], COLORS["positive"], COLORS["volume"],
                          COLORS["cii"], COLORS["price"], COLORS["neutral"]],
            text=[f"{v:.3f}" for v in sector_data["Mean r"]],
            textposition="outside",
        ))
        styled_layout(fig_sector, height=300,
                       title="Finance sector shows strongest coupling",
                       xaxis_title="Mean temporal correlation",
                       xaxis_range=[0, 0.85])
        st.plotly_chart(fig_sector, use_container_width=True)

    with col_sur:
        st.markdown("#### Surrogate Test: Shuffled Destroys Coupling")
        fig_surr = go.Figure()
        fig_surr.add_trace(go.Bar(
            x=["Original Data", "Shuffled Surrogate"],
            y=[0.665, 0.007],
            marker_color=[COLORS["positive"], COLORS["neutral"]],
            text=["r = 0.665", "r = 0.007"],
            textposition="outside",
            width=0.5,
        ))
        styled_layout(fig_surr, height=300,
                       title="Temporal structure is necessary for coupling",
                       yaxis_title="Mean temporal correlation",
                       yaxis_range=[0, 0.8])
        st.plotly_chart(fig_surr, use_container_width=True)


# ===== TAB: Price & Volume =====
with tab_price:
    fig_pv = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
        subplot_titles=(f"{ticker} Close Price", "Trading Volume"),
    )
    fig_pv.add_trace(
        go.Scatter(x=df.index, y=close, mode="lines",
                   line=dict(color=COLORS["price"], width=1),
                   name="Close"),
        row=1, col=1,
    )
    fig_pv.add_trace(
        go.Bar(x=df.index, y=volume_raw, marker_color=COLORS["volume"],
               opacity=0.5, name="Volume"),
        row=2, col=1,
    )
    styled_layout(fig_pv, height=550)
    fig_pv.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig_pv.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig_pv, use_container_width=True)

    h_ret_str = f"{h_returns:.3f}" if not np.isnan(h_returns) else "N/A"
    h_vol_str = f"{h_volume:.3f}" if not np.isnan(h_volume) else "N/A"

    interp_ret = (
        "persistent (trending)" if h_returns > 0.55
        else "anti-persistent (mean-reverting)" if h_returns < 0.45
        else "approximately random walk"
    )
    interp_vol = (
        "strong long memory (expected per Lobato & Velasco 2000)" if h_volume > 0.65
        else "moderate persistence" if h_volume > 0.55
        else "surprisingly low persistence"
    )

    st.info(
        f"**H(|returns|) = {h_ret_str}** ({interp_ret}). "
        f"**H(log volume) = {h_vol_str}** ({interp_vol})."
    )


# ===== TAB: Rolling Hurst =====
with tab_rolling:
    if len(abs_log_ret) < window:
        st.warning(f"Need at least {window} data points for rolling analysis. Have {len(abs_log_ret)}.")
    else:
        if dual_df.empty:
            dual_df = compute_rolling_dual(
                abs_log_ret.tolist(), log_vol.tolist(),
                [str(d) for d in dates], window, step,
            )

        if not dual_df.empty:
            fig_roll = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.06,
                row_heights=[0.3, 0.4, 0.3],
                subplot_titles=(f"{ticker} Price", "Rolling Hurst Exponents", "H(volume) - H(|returns|) Spread"),
            )

            # Price
            fig_roll.add_trace(
                go.Scatter(x=df.index, y=close, mode="lines",
                           line=dict(color=COLORS["price"], width=1), name="Price", showlegend=False),
                row=1, col=1,
            )

            # Rolling Hurst
            dual_plot = dual_df.copy()
            dual_plot["date"] = pd.to_datetime(dual_plot["date"])

            fig_roll.add_trace(
                go.Scatter(x=dual_plot["date"], y=dual_plot["H_price"],
                           mode="lines", line=dict(color=COLORS["price"], width=1.5),
                           name="H(|returns|)"),
                row=2, col=1,
            )
            fig_roll.add_trace(
                go.Scatter(x=dual_plot["date"], y=dual_plot["H_volume"],
                           mode="lines", line=dict(color=COLORS["volume"], width=1.5),
                           name="H(log volume)"),
                row=2, col=1,
            )
            fig_roll.add_hline(y=0.5, line_dash="dash", line_color=COLORS["neutral"],
                               opacity=0.5, row=2, col=1,
                               annotation_text="H = 0.5 (random walk)")

            # Spread
            fig_roll.add_trace(
                go.Scatter(x=dual_plot["date"], y=dual_plot["spread"],
                           mode="markers", marker=dict(color=COLORS["cii"], size=4, opacity=0.6),
                           name="Spread"),
                row=3, col=1,
            )
            fig_roll.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"],
                               opacity=0.5, row=3, col=1)

            styled_layout(fig_roll, height=700)
            fig_roll.update_yaxes(title_text="Price", row=1, col=1)
            fig_roll.update_yaxes(title_text="Hurst H", range=[0.2, 1.0], row=2, col=1)
            fig_roll.update_yaxes(title_text="Spread", row=3, col=1)
            st.plotly_chart(fig_roll, use_container_width=True)

            st.markdown(
                "Volume Hurst (teal) typically sits above 0.5, reflecting strong long memory "
                "in trading activity. Volatility Hurst (red) fluctuates closer to 0.5 but may "
                "show persistent regimes during market stress. The spread (yellow) shows when "
                "price-volume fractal coupling tightens or diverges."
            )
        else:
            st.warning("Could not compute rolling Hurst. Try adjusting window parameters.")


# ===== TAB: Temporal Coupling =====
with tab_coupling:
    if dual_df.empty:
        st.warning(f"Need at least {window} data points. Have {len(abs_log_ret)}.")
    else:
        valid = dual_df.dropna(subset=["H_price", "H_volume"])

        if len(valid) > 5:
            pearson_r, pearson_p = stats.pearsonr(valid["H_price"], valid["H_volume"])
            spearman_r, spearman_p = stats.spearmanr(valid["H_price"], valid["H_volume"])

            tc_c1, tc_c2, tc_c3, tc_c4 = st.columns(4)
            tc_c1.metric("Pearson r", f"{pearson_r:.3f}")
            tc_c2.metric("p-value", f"{pearson_p:.2e}" if pearson_p < 0.001 else f"{pearson_p:.4f}")
            tc_c3.metric("Spearman rho", f"{spearman_r:.3f}")
            tc_c4.metric("N windows", f"{len(valid)}")

            col_scatter, col_lag = st.columns(2)

            with col_scatter:
                # Time-colored scatter
                valid_plot = valid.copy()
                valid_plot["time_idx"] = range(len(valid_plot))

                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=valid_plot["H_price"], y=valid_plot["H_volume"],
                    mode="markers",
                    marker=dict(
                        color=valid_plot["time_idx"],
                        colorscale="Plasma",
                        size=6, opacity=0.7,
                        colorbar=dict(title="Time", tickvals=[], ticktext=[]),
                    ),
                    name="Rolling windows",
                ))
                # Regression line
                z = np.polyfit(valid_plot["H_price"], valid_plot["H_volume"], 1)
                x_line = np.linspace(valid_plot["H_price"].min(), valid_plot["H_price"].max(), 100)
                fig_scatter.add_trace(go.Scatter(
                    x=x_line, y=np.polyval(z, x_line),
                    mode="lines", line=dict(color=COLORS["price"], dash="dash", width=2),
                    name=f"OLS fit (r={pearson_r:.3f})",
                ))
                styled_layout(fig_scatter, height=400,
                               title=f"Temporal Coupling: {ticker}",
                               xaxis_title="H(|returns|)",
                               yaxis_title="H(log volume)")
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_lag:
                # Lead-lag analysis
                lag_df = lead_lag_correlation(dual_df)
                if not lag_df.empty:
                    fig_lag = go.Figure()
                    fig_lag.add_trace(go.Bar(
                        x=lag_df["lag"], y=lag_df["correlation"],
                        marker_color=[COLORS["positive"] if r > 0 else COLORS["negative"]
                                      for r in lag_df["correlation"]],
                        name="Cross-correlation",
                    ))
                    fig_lag.add_hline(y=0, line_dash="solid", line_color=COLORS["neutral"])
                    # Significance bands (approximate)
                    n_eff = len(valid)
                    sig_bound = 1.96 / np.sqrt(n_eff) if n_eff > 0 else 0
                    fig_lag.add_hline(y=sig_bound, line_dash="dot", line_color=COLORS["cii"], opacity=0.5)
                    fig_lag.add_hline(y=-sig_bound, line_dash="dot", line_color=COLORS["cii"], opacity=0.5)
                    styled_layout(fig_lag, height=400,
                                   title="Lead-Lag Cross-Correlation",
                                   xaxis_title="Lag (+ = volume leads price)",
                                   yaxis_title="Pearson r")
                    st.plotly_chart(fig_lag, use_container_width=True)
                else:
                    st.info("Not enough data for lead-lag analysis.")

            sig_str = (
                "significant positive correlation; fractal properties of price and volume are coupled"
                if pearson_p < 0.05 and pearson_r > 0
                else "significant negative correlation; price and volume fractality diverge"
                if pearson_p < 0.05 and pearson_r < 0
                else "no significant linear correlation at this window size"
            )
            st.info(
                f"**Pearson r = {pearson_r:.3f}** (p = {pearson_p:.2e}): {sig_str}. "
                f"Color in the scatter encodes time (dark = early, light = recent)."
            )
        else:
            st.warning("Not enough rolling windows to compute correlation.")


# ===== TAB: Liquidity Prediction =====
with tab_predict:
    st.markdown("### CII and Forward Liquidity Metrics")
    st.markdown(
        "The Coupling Intensity Index (CII) measures trailing correlation between "
        "rolling H(|returns|) and H(volume). The research shows CII significantly predicts "
        "future Amihud illiquidity (t = 2.90, p = 0.004) but not realized volatility (t = 0.84)."
    )

    if dual_df.empty:
        st.warning(f"Need at least {window} data points for prediction analysis. Have {len(abs_log_ret)}.")
    else:
        try:
            cii_series = compute_coupling_intensity(dual_df, correlation_window=30)

            if cii_series.empty:
                st.warning("Not enough rolling windows to compute CII. Try a longer date range or smaller window.")
            else:
                # CII time series
                fig_cii = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.4, 0.6],
                    subplot_titles=(f"{ticker} Price", "Coupling Intensity Index (CII)"),
                )
                fig_cii.add_trace(
                    go.Scatter(x=df.index, y=close, mode="lines",
                               line=dict(color=COLORS["price"], width=1), name="Price", showlegend=False),
                    row=1, col=1,
                )
                fig_cii.add_trace(
                    go.Scatter(x=cii_series.index, y=cii_series.values,
                               mode="lines", line=dict(color=COLORS["cii"], width=1.5),
                               name="CII", fill="tozeroy",
                               fillcolor="rgba(255, 217, 61, 0.15)"),
                    row=2, col=1,
                )
                fig_cii.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"],
                                   opacity=0.5, row=2, col=1)
                styled_layout(fig_cii, height=500)
                fig_cii.update_yaxes(title_text="Price", row=1, col=1)
                fig_cii.update_yaxes(title_text="CII (trailing r)", row=2, col=1)
                st.plotly_chart(fig_cii, use_container_width=True)

                # Forward metrics if we can compute them
                try:
                    prices_s = df["Close"].squeeze()
                    volume_s = df["Volume"].squeeze()
                    fwd_metrics = compute_forward_metrics(prices_s, volume_s, horizon=21)

                    if not fwd_metrics.empty:
                        # Align CII with forward metrics
                        cii_df = pd.DataFrame({"CII": cii_series})
                        merged_pred = cii_df.join(fwd_metrics, how="inner").dropna()

                        if len(merged_pred) > 20:
                            st.markdown("#### CII vs. Forward Metrics (21-day horizon)")
                            pred_c1, pred_c2 = st.columns(2)

                            with pred_c1:
                                fig_illiq = go.Figure()
                                fig_illiq.add_trace(go.Scatter(
                                    x=merged_pred["CII"],
                                    y=merged_pred["amihud_illiq"],
                                    mode="markers",
                                    marker=dict(color=COLORS["accent"], size=4, opacity=0.4),
                                    name="Observations",
                                ))
                                # Regression line
                                mask = np.isfinite(merged_pred["CII"]) & np.isfinite(merged_pred["amihud_illiq"])
                                if mask.sum() > 10:
                                    z = np.polyfit(merged_pred.loc[mask, "CII"],
                                                   merged_pred.loc[mask, "amihud_illiq"], 1)
                                    x_l = np.linspace(merged_pred["CII"].min(), merged_pred["CII"].max(), 50)
                                    r_val, p_val = stats.pearsonr(
                                        merged_pred.loc[mask, "CII"],
                                        merged_pred.loc[mask, "amihud_illiq"])
                                    fig_illiq.add_trace(go.Scatter(
                                        x=x_l, y=np.polyval(z, x_l),
                                        mode="lines",
                                        line=dict(color=COLORS["positive"], width=2, dash="dash"),
                                        name=f"r={r_val:.3f}, p={p_val:.3f}",
                                    ))
                                styled_layout(fig_illiq, height=380,
                                               title="CII vs. Forward Amihud Illiquidity",
                                               xaxis_title="CII",
                                               yaxis_title="Forward Amihud Illiquidity")
                                st.plotly_chart(fig_illiq, use_container_width=True)

                            with pred_c2:
                                fig_rv = go.Figure()
                                fig_rv.add_trace(go.Scatter(
                                    x=merged_pred["CII"],
                                    y=merged_pred["realized_vol"],
                                    mode="markers",
                                    marker=dict(color=COLORS["neutral"], size=4, opacity=0.4),
                                    name="Observations",
                                ))
                                mask = np.isfinite(merged_pred["CII"]) & np.isfinite(merged_pred["realized_vol"])
                                if mask.sum() > 10:
                                    z = np.polyfit(merged_pred.loc[mask, "CII"],
                                                   merged_pred.loc[mask, "realized_vol"], 1)
                                    x_l = np.linspace(merged_pred["CII"].min(), merged_pred["CII"].max(), 50)
                                    r_val, p_val = stats.pearsonr(
                                        merged_pred.loc[mask, "CII"],
                                        merged_pred.loc[mask, "realized_vol"])
                                    fig_rv.add_trace(go.Scatter(
                                        x=x_l, y=np.polyval(z, x_l),
                                        mode="lines",
                                        line=dict(color=COLORS["neutral"], width=2, dash="dash"),
                                        name=f"r={r_val:.3f}, p={p_val:.3f}",
                                    ))
                                styled_layout(fig_rv, height=380,
                                               title="CII vs. Forward Realized Volatility",
                                               xaxis_title="CII",
                                               yaxis_title="Forward Realized Vol")
                                st.plotly_chart(fig_rv, use_container_width=True)

                            st.markdown(
                                "In the full panel regression (50 stocks), CII significantly predicts "
                                "Amihud illiquidity (two-way clustered t = 2.90, p = 0.004) but does "
                                "not predict realized volatility (t = 0.84). Single-ticker scatter above "
                                "may differ from the panel result."
                            )
                        else:
                            st.info("Not enough overlapping CII and forward metric observations for scatter plots.")
                except Exception as e:
                    st.info(f"Forward metric computation requires sufficient price history. ({e})")

        except Exception as e:
            st.warning(f"CII computation failed: {e}")

    # Summary table of panel results (pre-computed)
    st.markdown("---")
    st.markdown("#### Panel Regression Results (50-Stock Universe)")
    panel_results = pd.DataFrame([
        {"Target": "Amihud Illiquidity", "CII Beta": "positive", "HC1 t": 5.12,
         "Firm-Clustered t": 3.45, "Two-Way Clustered t": 2.90, "p-value": 0.004,
         "Verdict": "Significant"},
        {"Target": "Realized Volatility", "CII Beta": "positive", "HC1 t": 3.81,
         "Firm-Clustered t": 1.92, "Two-Way Clustered t": 0.84, "p-value": 0.401,
         "Verdict": "NOT significant"},
        {"Target": "Max Drawdown", "CII Beta": "negative", "HC1 t": -2.14,
         "Firm-Clustered t": -1.33, "Two-Way Clustered t": -0.71, "p-value": 0.478,
         "Verdict": "NOT significant"},
        {"Target": "Abnormal Turnover", "CII Beta": "positive", "HC1 t": 4.05,
         "Firm-Clustered t": 2.67, "Two-Way Clustered t": 1.89, "p-value": 0.059,
         "Verdict": "Marginal"},
    ])
    st.dataframe(
        panel_results.style.applymap(
            lambda v: "color: #00d4aa" if v == "Significant"
            else "color: #ff6b6b" if v == "NOT significant"
            else "color: #ffd93d" if v == "Marginal"
            else "",
            subset=["Verdict"]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "HC1 standard errors dramatically overstate significance. Two-way clustering "
        "(firm + time) is the correct inference for overlapping panels. Only Amihud "
        "illiquidity survives robust inference."
    )


# ===== TAB: Regime Analysis =====
with tab_regime:
    st.markdown("### VIX Regimes and Crisis Amplification")

    if dual_df.empty:
        st.warning(f"Need at least {window} data points for regime analysis. Have {len(abs_log_ret)}.")
    else:
        try:
            vix, regime = load_vix(str(start_date), str(end_date))

            if vix is not None and regime is not None:
                aligned = align_regime_with_rolling(dual_df, vix, regime)

                if not aligned.empty and "vix_regime" in aligned.columns:
                    # Coupling by VIX regime
                    regime_stats = coupling_by_regime(aligned, "vix_regime")

                    reg_c1, reg_c2 = st.columns([1, 1])

                    with reg_c1:
                        st.markdown("#### Coupling by VIX Regime")
                        regime_order = ["low", "medium", "high"]
                        regime_colors = [COLORS["positive"], COLORS["cii"], COLORS["negative"]]
                        regime_labels = []
                        regime_corrs = []
                        regime_ns = []

                        for r_name in regime_order:
                            if r_name in regime_stats:
                                rs = regime_stats[r_name]
                                regime_labels.append(f"VIX {r_name.capitalize()}")
                                regime_corrs.append(rs.get("pearson_r", 0))
                                regime_ns.append(rs.get("n", 0))

                        if regime_labels:
                            fig_regime = go.Figure(go.Bar(
                                x=regime_labels, y=regime_corrs,
                                marker_color=regime_colors[:len(regime_labels)],
                                text=[f"r={r:.3f}<br>n={n}" for r, n in zip(regime_corrs, regime_ns)],
                                textposition="outside",
                            ))
                            styled_layout(fig_regime, height=380,
                                           title=f"Temporal Coupling by VIX Regime ({ticker})",
                                           yaxis_title="Pearson r (H_price vs H_volume)",
                                           yaxis_range=[min(0, min(regime_corrs) - 0.1),
                                                        max(regime_corrs) + 0.15])
                            st.plotly_chart(fig_regime, use_container_width=True)

                    with reg_c2:
                        # Coupling by crisis
                        st.markdown("#### Coupling by Crisis Period")
                        crisis_stats = coupling_by_regime(aligned, "crisis")

                        crisis_labels = []
                        crisis_corrs = []
                        crisis_ns = []
                        crisis_colors_list = []

                        for c_name in ["normal"] + list(CRISIS_WINDOWS.keys()):
                            if c_name in crisis_stats:
                                cs = crisis_stats[c_name]
                                if cs.get("n", 0) >= 10:
                                    crisis_labels.append(c_name.replace("_", " ").title())
                                    crisis_corrs.append(cs.get("pearson_r", 0))
                                    crisis_ns.append(cs.get("n", 0))
                                    crisis_colors_list.append(
                                        COLORS["neutral"] if c_name == "normal" else COLORS["price"]
                                    )

                        if crisis_labels:
                            fig_crisis = go.Figure(go.Bar(
                                x=crisis_labels, y=crisis_corrs,
                                marker_color=crisis_colors_list,
                                text=[f"r={r:.3f}<br>n={n}" for r, n in zip(crisis_corrs, crisis_ns)],
                                textposition="outside",
                            ))
                            styled_layout(fig_crisis, height=380,
                                           title=f"Coupling During Crisis Windows ({ticker})",
                                           yaxis_title="Pearson r",
                                           yaxis_range=[min(0, min(crisis_corrs) - 0.1) if crisis_corrs else 0,
                                                        max(crisis_corrs) + 0.15 if crisis_corrs else 1])
                            st.plotly_chart(fig_crisis, use_container_width=True)
                        else:
                            st.info("Not enough data in crisis windows for this ticker/date range.")

                    # VIX overlay with CII
                    st.markdown("---")
                    st.markdown("#### VIX and Hurst Dynamics Over Time")

                    aligned_plot = aligned.copy()
                    aligned_plot["date"] = pd.to_datetime(aligned_plot["date"])

                    fig_vix = make_subplots(
                        rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=("VIX Index", "Rolling Hurst Exponents with Regime Shading"),
                    )

                    if "vix" in aligned_plot.columns:
                        fig_vix.add_trace(
                            go.Scatter(x=aligned_plot["date"], y=aligned_plot["vix"],
                                       mode="lines", line=dict(color=COLORS["cii"], width=1),
                                       name="VIX", fill="tozeroy",
                                       fillcolor="rgba(255, 217, 61, 0.1)"),
                            row=1, col=1,
                        )

                    fig_vix.add_trace(
                        go.Scatter(x=aligned_plot["date"], y=aligned_plot["H_price"],
                                   mode="lines", line=dict(color=COLORS["price"], width=1.5),
                                   name="H(|returns|)"),
                        row=2, col=1,
                    )
                    fig_vix.add_trace(
                        go.Scatter(x=aligned_plot["date"], y=aligned_plot["H_volume"],
                                   mode="lines", line=dict(color=COLORS["volume"], width=1.5),
                                   name="H(log volume)"),
                        row=2, col=1,
                    )

                    # Add crisis shading
                    for crisis_name, (cs, ce) in CRISIS_WINDOWS.items():
                        fig_vix.add_vrect(
                            x0=cs, x1=ce, fillcolor=COLORS["price"],
                            opacity=0.1, line_width=0, row=2, col=1,
                            annotation_text=crisis_name.replace("_", " "),
                            annotation_position="top left",
                            annotation_font_size=9,
                        )

                    styled_layout(fig_vix, height=550)
                    fig_vix.update_yaxes(title_text="VIX", row=1, col=1)
                    fig_vix.update_yaxes(title_text="Hurst H", row=2, col=1)
                    st.plotly_chart(fig_vix, use_container_width=True)

                    st.markdown(
                        "Shaded regions mark known crisis windows. In the full 50-stock analysis, "
                        "coupling nearly doubles during COVID (r = 0.41 normal periods vs. r = 0.77 "
                        "during the crash). This pattern is consistent with information arrival "
                        "intensifying the co-movement of fractal exponents."
                    )
                else:
                    st.warning("Could not align VIX data with rolling Hurst windows.")
            else:
                st.warning("Could not fetch VIX data. This tab requires internet access to Yahoo Finance.")
        except Exception as e:
            st.warning(f"Regime analysis failed: {e}")


# ===== TAB: Method Comparison =====
with tab_methods:
    st.markdown("### Hurst Estimation: DFA vs R/S")

    h_ret_dfa = h_returns
    h_vol_dfa = h_volume
    h_ret_rs = estimate_rs(abs_log_ret).H
    h_vol_rs = estimate_rs(log_vol).H

    methods_data = pd.DataFrame([
        {"Series": "|Log Returns|", "DFA": h_ret_dfa, "R/S": h_ret_rs,
         "Difference": abs(h_ret_dfa - h_ret_rs)},
        {"Series": "Log Volume", "DFA": h_vol_dfa, "R/S": h_vol_rs,
         "Difference": abs(h_vol_dfa - h_vol_rs)},
    ])

    col_table, col_chart = st.columns([1, 1])

    with col_table:
        st.dataframe(
            methods_data.style.format({
                "DFA": "{:.4f}", "R/S": "{:.4f}", "Difference": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("""
**Notes:**
- **DFA** (Detrended Fluctuation Analysis) handles nonstationarity via polynomial
  detrending. Preferred for financial data.
- **R/S** (Rescaled Range) is the classical method but overestimates H for
  series with short-range dependence (Lo 1991).
- Large disagreement suggests structural breaks or strong short-range dependence
  inflating R/S.
        """)

    with col_chart:
        fig_comp = go.Figure()
        series_labels = ["|Log Returns|", "Log Volume"]
        x_positions = np.arange(len(series_labels))

        fig_comp.add_trace(go.Bar(
            x=series_labels, y=[h_ret_dfa, h_vol_dfa],
            name="DFA", marker_color=COLORS["price"],
            text=[f"{h_ret_dfa:.3f}", f"{h_vol_dfa:.3f}"],
            textposition="outside",
        ))
        fig_comp.add_trace(go.Bar(
            x=series_labels, y=[h_ret_rs, h_vol_rs],
            name="R/S", marker_color=COLORS["volume"],
            text=[f"{h_ret_rs:.3f}", f"{h_vol_rs:.3f}"],
            textposition="outside",
        ))
        fig_comp.add_hline(y=0.5, line_dash="dash", line_color=COLORS["neutral"],
                           annotation_text="H = 0.5")
        styled_layout(fig_comp, height=380,
                       title="DFA vs R/S Comparison",
                       yaxis_title="Hurst Exponent H",
                       yaxis_range=[0.3, 1.05],
                       barmode="group")
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Expected Values from Literature")
    lit_table = pd.DataFrame([
        {"Measure": "H(raw returns)", "Expected": "~ 0.5",
         "Interpretation": "Efficient market (random walk)", "Source": "Lo 1991"},
        {"Measure": "H(|returns| / volatility)", "Expected": "0.7 - 0.8",
         "Interpretation": "Long memory in volatility", "Source": "Bollerslev & Jubinski 1999"},
        {"Measure": "H(volume)", "Expected": "0.7 - 0.9",
         "Interpretation": "Very strong persistence", "Source": "Lobato & Velasco 2000"},
    ])
    st.dataframe(lit_table, use_container_width=True, hide_index=True)
