"""Rolling window Hurst analysis and temporal cross-correlation.

Computes Hurst exponents in rolling windows to reveal how fractal
properties evolve over time. This is where regime-dependent behavior
becomes visible — e.g., does volatility persistence spike during crises?
Does price-volume coupling tighten in high-volatility periods?
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .hurst import estimate_dfa, HurstResult


def rolling_hurst(
    series: np.ndarray,
    dates: np.ndarray | None = None,
    window: int = 500,
    step: int = 20,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute Hurst exponent in rolling windows.

    Parameters
    ----------
    series : array-like
        1D time series.
    dates : array-like or None
        Date index aligned with series. If None, uses integer index.
    window : int
        Rolling window size in observations. 500 ≈ 2 years of daily data.
        Minimum recommended: 256 (Peng et al. 1994).
    step : int
        Step size between windows. 20 ≈ 1 month of daily data.
    n_jobs : int
        Parallel jobs. 1 = sequential (safer for small runs).

    Returns
    -------
    pd.DataFrame with columns: date, H, r_squared, window_start, window_end
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    if n < window:
        return pd.DataFrame(columns=["date", "H", "r_squared", "window_start", "window_end"])

    starts = list(range(0, n - window + 1, step))

    def _compute_one(start):
        segment = series[start : start + window]
        result = estimate_dfa(segment)
        mid = start + window // 2
        return {
            "idx": mid,
            "H": result.H,
            "r_squared": result.r_squared,
            "window_start": start,
            "window_end": start + window,
        }

    if n_jobs == 1:
        rows = [_compute_one(s) for s in starts]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_one)(s) for s in starts
        )

    df = pd.DataFrame(rows)

    if dates is not None and len(dates) > 0:
        dates = np.asarray(dates)
        df["date"] = df["idx"].apply(lambda i: dates[min(i, len(dates) - 1)])
    else:
        df["date"] = df["idx"]

    return df


def rolling_dual_hurst(
    abs_returns: np.ndarray,
    log_volume: np.ndarray,
    dates: np.ndarray | None = None,
    window: int = 500,
    step: int = 20,
) -> pd.DataFrame:
    """Compute rolling Hurst for both |returns| and volume in aligned windows.

    Returns a single DataFrame with H_price, H_volume, and their spread.
    This is the core data for temporal cross-correlation analysis.
    """
    min_len = min(len(abs_returns), len(log_volume))
    abs_returns = abs_returns[:min_len]
    log_volume = log_volume[:min_len]
    if dates is not None:
        dates = np.asarray(dates)[:min_len]

    roll_price = rolling_hurst(abs_returns, dates, window, step)
    roll_vol = rolling_hurst(log_volume, dates, window, step)

    if roll_price.empty or roll_vol.empty:
        return pd.DataFrame()

    merged = roll_price[["date", "H", "r_squared"]].merge(
        roll_vol[["date", "H", "r_squared"]],
        on="date",
        suffixes=("_price", "_volume"),
    )
    merged["spread"] = merged["H_volume"] - merged["H_price"]
    merged["H_diff_abs"] = np.abs(merged["spread"])

    return merged


def temporal_correlation(dual_df: pd.DataFrame) -> dict:
    """Compute temporal correlation statistics from rolling dual Hurst.

    Returns Pearson and Spearman correlations between rolling H(price)
    and H(volume), plus summary statistics on the spread.
    """
    from scipy import stats

    valid = dual_df.dropna(subset=["H_price", "H_volume"])

    if len(valid) < 10:
        return {
            "n_windows": len(valid),
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_r": np.nan, "spearman_p": np.nan,
            "mean_spread": np.nan, "std_spread": np.nan,
        }

    pr, pp = stats.pearsonr(valid["H_price"], valid["H_volume"])
    sr, sp = stats.spearmanr(valid["H_price"], valid["H_volume"])

    return {
        "n_windows": len(valid),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "mean_H_price": float(valid["H_price"].mean()),
        "std_H_price": float(valid["H_price"].std()),
        "mean_H_volume": float(valid["H_volume"].mean()),
        "std_H_volume": float(valid["H_volume"].std()),
        "mean_spread": float(valid["spread"].mean()),
        "std_spread": float(valid["spread"].std()),
        "min_spread": float(valid["spread"].min()),
        "max_spread": float(valid["spread"].max()),
    }


def lead_lag_correlation(
    dual_df: pd.DataFrame, max_lag: int = 10
) -> pd.DataFrame:
    """Test whether H(volume) leads H(price) or vice versa.

    Computes cross-correlation at lags -max_lag to +max_lag.
    Positive lag means volume leads price (H_volume at t predicts H_price at t+lag).
    Negative lag means price leads volume.
    """
    valid = dual_df.dropna(subset=["H_price", "H_volume"])

    if len(valid) < max_lag * 3:
        return pd.DataFrame(columns=["lag", "correlation", "p_value"])

    from scipy import stats

    results = []
    h_price = valid["H_price"].values
    h_volume = valid["H_volume"].values

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # volume leads: correlate volume[:-lag] with price[lag:]
            x = h_volume[: -lag] if lag > 0 else h_volume
            y = h_price[lag:] if lag > 0 else h_price
        elif lag < 0:
            # price leads: correlate price[:lag] with volume[-lag:]
            x = h_price[:lag]
            y = h_volume[-lag:]
        else:
            x = h_volume
            y = h_price

        if len(x) < 10:
            continue

        r, p = stats.pearsonr(x, y)
        results.append({"lag": lag, "correlation": r, "p_value": p})

    return pd.DataFrame(results)
