"""Regime identification and VIX-conditioned analysis.

Classifies market states by VIX level, crisis windows, and other
observable regime indicators. Enables testing H3: does temporal
coupling intensify during high-uncertainty regimes?
"""

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_vix(start: str = "2015-01-01", end: str | None = None) -> pd.Series:
    """Fetch daily VIX close from Yahoo Finance."""
    df = yf.download("^VIX", start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna().squeeze()


def classify_vix_regime(
    vix: pd.Series, low_quantile: float = 0.25, high_quantile: float = 0.75
) -> pd.Series:
    """Classify each date into low/medium/high VIX regime.

    Returns a Series with values 'low', 'medium', 'high'.
    """
    low_thresh = vix.quantile(low_quantile)
    high_thresh = vix.quantile(high_quantile)

    regime = pd.Series("medium", index=vix.index, name="vix_regime")
    regime[vix <= low_thresh] = "low"
    regime[vix >= high_thresh] = "high"
    return regime


CRISIS_WINDOWS = {
    "Volmageddon": ("2018-02-01", "2018-02-28"),
    "COVID_crash": ("2020-02-20", "2020-03-23"),
    "COVID_recovery": ("2020-03-24", "2020-06-30"),
    "Fed_tightening_2022": ("2022-01-01", "2022-10-31"),
    "SVB_crisis": ("2023-03-08", "2023-03-31"),
    "META_crisis": ("2022-02-01", "2022-11-30"),
}


def classify_crisis(dates) -> pd.Series:
    """Classify dates into crisis/normal periods.

    Returns Series with crisis name or 'normal'.
    """
    dates = pd.DatetimeIndex(dates)
    labels = ["normal"] * len(dates)
    for cname, (start, end) in CRISIS_WINDOWS.items():
        for i, d in enumerate(dates):
            if pd.Timestamp(start) <= d <= pd.Timestamp(end):
                labels[i] = cname
    return pd.Series(labels, index=dates, name="crisis")


def align_regime_with_rolling(
    rolling_df: pd.DataFrame,
    vix: pd.Series,
    regime: pd.Series,
) -> pd.DataFrame:
    """Merge regime classifications into rolling Hurst DataFrame.

    Parameters
    ----------
    rolling_df : DataFrame
        Must have 'date' column (from rolling_dual_hurst).
    vix : Series
        Daily VIX indexed by date.
    regime : Series
        VIX regime classification indexed by date.

    Returns
    -------
    DataFrame with added columns: vix, vix_regime, crisis.
    """
    df = rolling_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Align VIX to rolling dates (use nearest available)
    vix_df = vix.reset_index()
    vix_df.columns = ["date", "vix"]
    vix_df["date"] = pd.to_datetime(vix_df["date"])

    regime_df = regime.reset_index()
    regime_df.columns = ["date", "vix_regime"]
    regime_df["date"] = pd.to_datetime(regime_df["date"])

    # Normalize datetime resolution to avoid merge_asof dtype mismatch
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    regime_df["date"] = pd.to_datetime(regime_df["date"]).dt.tz_localize(None).astype("datetime64[ns]")

    df = pd.merge_asof(df.sort_values("date"), vix_df.sort_values("date"), on="date", direction="nearest")
    df = pd.merge_asof(df.sort_values("date"), regime_df.sort_values("date"), on="date", direction="nearest")
    crisis = classify_crisis(df["date"].values)
    df["crisis"] = crisis.values

    return df


def coupling_by_regime(
    aligned_df: pd.DataFrame,
    regime_col: str = "vix_regime",
) -> dict:
    """Compute temporal coupling statistics conditioned on regime.

    Returns dict of {regime_value: {pearson_r, n, mean_H_price, mean_H_volume}}.
    """
    from scipy import stats

    results = {}
    for regime_val, group in aligned_df.groupby(regime_col):
        valid = group.dropna(subset=["H_price", "H_volume"])
        if len(valid) < 10:
            results[regime_val] = {"pearson_r": np.nan, "n": len(valid)}
            continue

        r, p = stats.pearsonr(valid["H_price"], valid["H_volume"])
        results[regime_val] = {
            "pearson_r": float(r),
            "pearson_p": float(p),
            "n": len(valid),
            "mean_H_price": float(valid["H_price"].mean()),
            "mean_H_volume": float(valid["H_volume"].mean()),
            "mean_spread": float((valid["H_volume"] - valid["H_price"]).mean()),
        }

    return results
