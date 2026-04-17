"""Predictive content analysis — tests whether coupling predicts finance-relevant outcomes.

This is the module that determines whether the paper targets a finance journal
or stays in econophysics. If the Coupling Intensity Index (CII) predicts
future realized volatility, drawdowns, or abnormal turnover, the paper
has economic significance beyond pattern documentation.

Hypotheses tested:
  H4a: CII predicts future realized volatility
  H4b: CII predicts future abnormal turnover
  H4c: CII predicts future maximum drawdown
  H4d: CII predicts future Amihud illiquidity
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_coupling_intensity(
    rolling_df: pd.DataFrame,
    correlation_window: int = 30,
) -> pd.Series:
    """Compute the Coupling Intensity Index (CII) as a time series.

    CII_t = rolling Pearson r between H_{|r|}(τ) and H_{vol}(τ)
    for τ in [t - correlation_window, t].

    Parameters
    ----------
    rolling_df : DataFrame
        Must have columns 'date', 'H_price', 'H_volume' (from rolling_dual_hurst).
    correlation_window : int
        Number of rolling Hurst observations to use for the trailing correlation.
        30 observations × 20-day step = ~600 trading days trailing.

    Returns
    -------
    pd.Series indexed by date with CII values.
    """
    df = rolling_df.dropna(subset=["H_price", "H_volume"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    cii = pd.Series(np.nan, index=df.index, name="CII")

    for i in range(correlation_window, len(df)):
        window = df.iloc[i - correlation_window : i]
        if len(window) >= 10:
            r, _ = stats.pearsonr(window["H_price"], window["H_volume"])
            cii.iloc[i] = r

    result = pd.Series(cii.values, index=pd.to_datetime(df["date"]), name="CII")
    return result.dropna()


def compute_forward_metrics(
    prices: pd.Series,
    volume: pd.Series,
    horizon: int = 21,
) -> pd.DataFrame:
    """Compute forward-looking finance metrics at each date.

    Parameters
    ----------
    prices : Series
        Daily close prices indexed by date.
    volume : Series
        Daily volume indexed by date.
    horizon : int
        Forward-looking window in trading days. Default 21 (~1 month).

    Returns
    -------
    DataFrame with columns:
        realized_vol: sqrt(sum(r^2)) over [t+1, t+horizon]
        abnormal_turnover: mean(volume / trailing_60d_mean_volume) over [t+1, t+horizon]
        max_drawdown: max peak-to-trough loss in [t+1, t+horizon]
        amihud_illiq: mean(|r| / volume) over [t+1, t+horizon]
    """
    prices = prices.dropna().sort_index()
    volume = volume.dropna().sort_index()

    # Align
    common = prices.index.intersection(volume.index)
    prices = prices.loc[common]
    volume = volume.loc[common]

    log_returns = np.log(prices / prices.shift(1)).dropna()
    trailing_vol_mean = volume.rolling(60).mean()

    dates = log_returns.index
    n = len(dates)
    results = []

    for i in range(n - horizon):
        t = dates[i]
        fwd_slice = slice(i + 1, i + 1 + horizon)
        fwd_returns = log_returns.iloc[fwd_slice]
        fwd_volume = volume.iloc[fwd_slice]
        fwd_prices = prices.iloc[fwd_slice]

        if len(fwd_returns) < horizon * 0.8:
            continue

        # Realized volatility
        rv = np.sqrt(np.sum(fwd_returns.values ** 2))

        # Abnormal turnover
        trailing_mean = trailing_vol_mean.iloc[i]
        if trailing_mean > 0 and not np.isnan(trailing_mean):
            abn_turnover = np.mean(fwd_volume.values / trailing_mean)
        else:
            abn_turnover = np.nan

        # Maximum drawdown
        cum_ret = (1 + fwd_returns).cumprod()
        running_max = cum_ret.cummax()
        drawdowns = (cum_ret - running_max) / running_max
        max_dd = float(drawdowns.min())

        # Amihud illiquidity
        abs_ret = np.abs(fwd_returns.values).astype(float)
        vol_safe = fwd_volume.values.astype(float).copy()
        vol_safe[vol_safe == 0] = np.nan
        amihud = np.nanmean(abs_ret / vol_safe) * 1e9  # scale for readability

        results.append({
            "date": t,
            "realized_vol": rv,
            "abnormal_turnover": abn_turnover,
            "max_drawdown": max_dd,
            "amihud_illiq": amihud,
        })

    return pd.DataFrame(results).set_index("date")


def build_prediction_panel(
    tickers_data: dict,
    rolling_results: dict,
    horizon: int = 21,
    correlation_window: int = 30,
) -> pd.DataFrame:
    """Build panel dataset for predictive regressions.

    Parameters
    ----------
    tickers_data : dict
        {ticker: {'df': DataFrame, 'series': dict, 'dates': array}}
    rolling_results : dict
        {ticker: {'dual': DataFrame, 'tc': dict}}
    horizon : int
        Forward-looking window.
    correlation_window : int
        Trailing window for CII computation.

    Returns
    -------
    DataFrame with columns: ticker, date, CII, H_price, H_volume,
        realized_vol, abnormal_turnover, max_drawdown, amihud_illiq
    """
    panels = []

    for ticker, data in tickers_data.items():
        if ticker not in rolling_results:
            continue

        dual = rolling_results[ticker]["dual"]
        if dual.empty:
            continue

        # CII
        cii = compute_coupling_intensity(dual, correlation_window)
        if cii.empty:
            continue

        # Forward metrics
        df = data["df"]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        prices = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        fwd = compute_forward_metrics(prices, volume, horizon)
        if fwd.empty:
            continue

        # Current Hurst values
        dual_dated = dual.copy()
        dual_dated["date"] = pd.to_datetime(dual_dated["date"])
        dual_dated = dual_dated.set_index("date")

        # Merge all on date
        merged = pd.DataFrame({"CII": cii})
        merged = merged.join(dual_dated[["H_price", "H_volume"]], how="inner")
        merged = merged.join(fwd, how="inner")
        merged["ticker"] = ticker
        merged = merged.dropna()

        if len(merged) > 20:
            panels.append(merged.reset_index())

    if not panels:
        return pd.DataFrame()

    panel = pd.concat(panels, ignore_index=True)
    return panel


def run_predictive_regression(
    panel: pd.DataFrame,
    target: str = "realized_vol",
) -> dict:
    """Run pooled OLS with firm fixed effects.

    Model: target_{i,t+h} = α_i + β₁·CII_{i,t} + β₂·H_price_{i,t}
                            + β₃·H_volume_{i,t} + ε_{i,t}

    Parameters
    ----------
    panel : DataFrame
        From build_prediction_panel.
    target : str
        Column name for the dependent variable.

    Returns
    -------
    dict with regression results including coefficients, t-stats, R², n.
    """
    from numpy.linalg import lstsq

    df = panel.dropna(subset=["CII", "H_price", "H_volume", target]).copy()

    if len(df) < 50:
        return {"error": "Insufficient observations", "n": len(df)}

    # Demean by ticker (firm fixed effects)
    for col in ["CII", "H_price", "H_volume", target]:
        group_mean = df.groupby("ticker")[col].transform("mean")
        df[f"{col}_dm"] = df[col] - group_mean

    y = df[f"{target}_dm"].values
    X = df[["CII_dm", "H_price_dm", "H_volume_dm"]].values

    # Add constant for demeaned regression (should be ~0 but include for completeness)
    X = np.column_stack([np.ones(len(X)), X])
    labels = ["const", "CII", "H_price", "H_volume"]

    # OLS
    beta, residuals, rank, sv = lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n, k = X.shape

    # Standard errors (heteroskedasticity-robust, HC1)
    sigma2 = np.sum(resid ** 2) / (n - k)
    bread = np.linalg.inv(X.T @ X)

    # HC1 robust
    u2 = resid ** 2
    meat = X.T @ np.diag(u2) @ X
    V_hc1 = (n / (n - k)) * bread @ meat @ bread
    se_robust = np.sqrt(np.diag(V_hc1))

    t_stats = beta / se_robust
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    results = {
        "target": target,
        "n": n,
        "n_tickers": df["ticker"].nunique(),
        "r_squared": float(r_squared),
        "coefficients": {},
    }
    for i, label in enumerate(labels):
        results["coefficients"][label] = {
            "beta": float(beta[i]),
            "se": float(se_robust[i]),
            "t_stat": float(t_stats[i]),
            "p_value": float(p_values[i]),
        }

    return results


def run_all_predictions(
    panel: pd.DataFrame,
) -> dict:
    """Run predictive regressions for all target variables."""
    targets = ["realized_vol", "abnormal_turnover", "max_drawdown", "amihud_illiq"]
    results = {}
    for target in targets:
        if target in panel.columns:
            results[target] = run_predictive_regression(panel, target)
    return results
