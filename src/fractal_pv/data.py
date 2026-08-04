"""Data fetching and caching for fractal price-volume analysis.

Fetches OHLCV data from Yahoo Finance via yfinance, caches locally as parquet.
Daily data is the default — it gives the deepest history and is the literature
standard for Hurst exponent estimation (which needs 500+ data points minimum).
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_ticker(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker, caching as parquet.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL").
    start : str
        Start date in YYYY-MM-DD format.
    end : str or None
        End date. None means today.
    interval : str
        Data interval. "1d" for daily (recommended), "1h" for hourly.
    cache_dir : Path
        Directory for cached parquet files.
    force_refresh : bool
        If True, re-download even if cache exists.

    Returns
    -------
    pd.DataFrame
        OHLCV dataframe indexed by date.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}_{interval}_{start}_{end or 'latest'}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns if present (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_parquet(cache_path)
    return df


def fetch_universe(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    min_observations: int = 500,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple tickers, filtering by minimum length.

    Sequential per-ticker fetcher. Suitable for ~50 tickers. For larger
    universes (Phase 2 onward) prefer ``fetch_universe_batch`` which uses
    yfinance's multi-ticker mode and is roughly 10x faster while being
    less rate-limited.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start, end, interval, cache_dir
        Passed to fetch_ticker.
    min_observations : int
        Minimum number of rows required. Tickers with fewer are skipped.
        500 is the minimum for reliable DFA estimation (~2 years of daily data).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV dataframe, only for tickers meeting the minimum.
    """
    universe = {}
    failed = []

    for ticker in tqdm(tickers, desc="Fetching"):
        try:
            df = fetch_ticker(ticker, start=start, end=end, interval=interval, cache_dir=cache_dir)
            if len(df) >= min_observations:
                universe[ticker] = df
        except Exception as e:
            failed.append((ticker, str(e)))

    if failed:
        print(f"\nFailed to fetch {len(failed)} tickers:")
        for t, err in failed[:10]:
            print(f"  {t}: {err}")

    print(f"\nFetched {len(universe)}/{len(tickers)} tickers with >= {min_observations} observations")
    return universe


def fetch_universe_batch(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    min_observations: int = 600,
    batch_size: int = 50,
    max_retries: int = 3,
    retry_sleep: float = 5.0,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Batch-mode universe fetcher for Phase 2 (G ≈ 500) scale.

    Uses yfinance.download in multi-ticker mode, which is the documented
    path for >100 ticker pulls. Each batch produces a wide MultiIndex
    DataFrame; this function unpacks per-ticker frames, caches them as
    parquet, and returns the dict in the same shape as fetch_universe.

    Yahoo Finance ticker convention differs from CRSP/Wikipedia for share
    classes: BRK.B → BRK-B, BF.B → BF-B. The function normalizes.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols. Will be normalized for yfinance (dots → dashes).
    start, end, interval, cache_dir, min_observations
        As in fetch_universe.
    batch_size : int
        Tickers per yfinance.download call. 50 is a safe default; yfinance
        has been observed to silently truncate beyond ~100 in some
        environments.
    max_retries, retry_sleep
        Retry policy on transient HTTP errors.
    force_refresh : bool
        If True, bypass parquet cache and re-download.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of original ticker symbol → OHLCV DataFrame.
    """
    import time

    cache_dir.mkdir(parents=True, exist_ok=True)
    universe: dict[str, pd.DataFrame] = {}
    failed: list[tuple[str, str]] = []

    # Normalize tickers for yfinance (BRK.B -> BRK-B etc.)
    yf_tickers = [t.replace(".", "-") for t in tickers]
    original_of = dict(zip(yf_tickers, tickers))

    # Identify which tickers are already cached.
    to_fetch: list[str] = []
    for yf_t, orig in original_of.items():
        cache_path = cache_dir / f"{orig}_{interval}_{start}_{end or 'latest'}.parquet"
        if cache_path.exists() and not force_refresh:
            try:
                df = pd.read_parquet(cache_path)
                if len(df) >= min_observations:
                    universe[orig] = df
                    continue
            except Exception:
                pass
        to_fetch.append(yf_t)

    print(f"  cache hits: {len(universe)}; to fetch: {len(to_fetch)}")

    # Batch-fetch the rest.
    for i in tqdm(range(0, len(to_fetch), batch_size), desc="Batch fetch"):
        batch = to_fetch[i : i + batch_size]
        attempt = 0
        while attempt < max_retries:
            try:
                df_batch = yf.download(
                    tickers=batch,
                    start=start,
                    end=end,
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                    auto_adjust=False,
                )
                break
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    for t in batch:
                        failed.append((original_of[t], f"batch fetch failed after {max_retries}: {e}"))
                    df_batch = None
                    break
                time.sleep(retry_sleep * attempt)

        if df_batch is None or df_batch.empty:
            continue

        # yfinance returns columns as MultiIndex (ticker, field). Unpack.
        for yf_t in batch:
            orig = original_of[yf_t]
            try:
                if isinstance(df_batch.columns, pd.MultiIndex):
                    if yf_t in df_batch.columns.get_level_values(0):
                        sub = df_batch[yf_t].dropna(how="all")
                    else:
                        failed.append((orig, "ticker not in batch result"))
                        continue
                else:
                    # Single-ticker batch — yfinance returns a flat frame.
                    sub = df_batch.dropna(how="all")
                if sub.empty:
                    failed.append((orig, "empty after dropna"))
                    continue
                if len(sub) < min_observations:
                    failed.append((orig, f"only {len(sub)} obs, need {min_observations}"))
                    continue
                cache_path = cache_dir / f"{orig}_{interval}_{start}_{end or 'latest'}.parquet"
                sub.to_parquet(cache_path)
                universe[orig] = sub
            except Exception as e:
                failed.append((orig, f"unpack failed: {e}"))

    if failed:
        print(f"\nFailed: {len(failed)} tickers")
        for t, err in failed[:15]:
            print(f"  {t}: {err}")

    print(f"\nFetched {len(universe)}/{len(tickers)} tickers with >= {min_observations} observations")
    return universe


# Common ticker universes for analysis
SP500_SAMPLE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM", "JNJ", "V",
    "PG", "UNH", "HD", "MA", "DIS", "ADBE", "CRM", "NFLX", "CMCSA", "PFE",
    "KO", "PEP", "ABT", "TMO", "COST", "AVGO", "NKE", "MRK", "WMT", "CVX",
    "LLY", "MCD", "DHR", "TXN", "NEE", "BMY", "UPS", "MS", "SCHW", "RTX",
    "LOW", "INTC", "QCOM", "INTU", "AMAT", "GS", "BLK", "ISRG", "MDLZ", "ADP",
]
