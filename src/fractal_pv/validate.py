"""Validation harness — checks results against theoretical expectations.

This module is the "sanity check" layer. Every time we compute a Hurst
exponent or run an analysis, we can pass it through here to verify
consistency with established literature and flag anomalies.

Theoretical priors (from peer-reviewed sources):
  - H(raw returns) ≈ 0.5 for efficient markets           [Lo 1991]
  - H(|returns|)   ≈ 0.7–0.8 (long memory in volatility) [Bollerslev & Jubinski 1999]
  - H(volume)      ≈ 0.7–0.9 (very long memory)          [Lobato & Velasco 2000]
  - H(emerging)    > H(developed) for raw returns         [Di Matteo et al. 2005]
  - Multifractal spectrum width Δα ≈ 0.3–0.8             [Kantelhardt et al. 2002]
  - R/S overestimates H relative to DFA                   [Lo 1991]

Design: each check returns a ValidationResult with pass/warn/fail status,
the observed value, the expected range, and a plain-English interpretation
suitable for display in the Streamlit app or a notebook.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Status(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class ValidationResult:
    """Single validation check result."""

    check_name: str
    status: Status
    observed: float
    expected_range: tuple[float, float]
    interpretation: str
    reference: str
    detail: str = ""

    def __str__(self) -> str:
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}[self.status.value]
        return (
            f"{icon} {self.check_name}: observed={self.observed:.4f}, "
            f"expected=[{self.expected_range[0]:.2f}, {self.expected_range[1]:.2f}] "
            f"→ {self.interpretation}"
        )


@dataclass
class ValidationReport:
    """Collection of validation results for a single analysis run."""

    ticker: str
    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    @property
    def passes(self) -> list[ValidationResult]:
        return [r for r in self.results if r.status == Status.PASS]

    @property
    def warnings(self) -> list[ValidationResult]:
        return [r for r in self.results if r.status == Status.WARN]

    @property
    def failures(self) -> list[ValidationResult]:
        return [r for r in self.results if r.status == Status.FAIL]

    @property
    def summary(self) -> str:
        return (
            f"{self.ticker}: {len(self.passes)} pass, "
            f"{len(self.warnings)} warn, {len(self.failures)} fail"
        )

    def __str__(self) -> str:
        lines = [f"=== Validation Report: {self.ticker} ==="]
        for r in self.results:
            lines.append(str(r))
        lines.append(f"\n{self.summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_returns_hurst(H: float, method: str = "DFA") -> ValidationResult:
    """H(raw returns) should be ≈ 0.5 for efficient markets.

    A value significantly above 0.5 suggests trending (momentum).
    A value significantly below 0.5 suggests mean reversion.
    Both are interesting findings but need careful interpretation.
    """
    if 0.45 <= H <= 0.55:
        status = Status.PASS
        interp = "Consistent with efficient market hypothesis (random walk)."
    elif 0.40 <= H <= 0.60:
        status = Status.WARN
        interp = (
            "Mild departure from random walk. Could be genuine long memory "
            "or artifact of short-range dependence (see Lo 1991). "
            "Verify with modified R/S or DFA at multiple scales."
        )
    else:
        status = Status.FAIL
        interp = (
            f"H={H:.3f} is a strong departure from 0.5. "
            f"{'Persistent (momentum) signal.' if H > 0.5 else 'Anti-persistent (mean-reverting) signal.'} "
            "Check for: nonstationarity in the input, structural breaks, "
            "or incorrect preprocessing (did you use returns, not prices?)."
        )

    return ValidationResult(
        check_name=f"H(returns) [{method}]",
        status=status,
        observed=H,
        expected_range=(0.45, 0.55),
        interpretation=interp,
        reference="Lo (1991), Di Matteo et al. (2005)",
    )


def check_volatility_hurst(H: float, method: str = "DFA") -> ValidationResult:
    """H(|returns|) should be ≈ 0.7–0.8 (long memory in volatility).

    This is one of the most robust findings in financial econometrics.
    If we don't see it, something is wrong with our pipeline.
    """
    if 0.60 <= H <= 0.90:
        status = Status.PASS
        interp = "Consistent with long-memory volatility (expected)."
    elif 0.50 <= H <= 0.95:
        status = Status.WARN
        interp = (
            f"H={H:.3f} is at the edge of expected range. "
            "Could reflect short sample, regime change, or atypical asset."
        )
    else:
        status = Status.FAIL
        interp = (
            f"H={H:.3f} is unexpected for volatility. "
            "Expected 0.7–0.8. Check: are you using absolute returns (not raw returns)? "
            "Is the series long enough (need 500+ observations)?"
        )

    return ValidationResult(
        check_name=f"H(|returns|) [{method}]",
        status=status,
        observed=H,
        expected_range=(0.65, 0.85),
        interpretation=interp,
        reference="Bollerslev & Jubinski (1999)",
    )


def check_volume_hurst(H: float, method: str = "DFA") -> ValidationResult:
    """H(volume) should be ≈ 0.7–0.9 (very long memory).

    Volume persistence is even stronger than volatility persistence.
    This is robust across markets, time periods, and frequencies.
    """
    if 0.65 <= H <= 0.95:
        status = Status.PASS
        interp = "Consistent with long-memory volume (expected)."
    elif 0.55 <= H <= 1.0:
        status = Status.WARN
        interp = (
            f"H={H:.3f} is slightly outside typical range. "
            "Volume persistence varies by asset class and liquidity."
        )
    else:
        status = Status.FAIL
        interp = (
            f"H={H:.3f} is unexpected for volume. "
            "Expected 0.7–0.9. Check: are you using log(volume)? "
            "Raw volume can have nonstationarity issues."
        )

    return ValidationResult(
        check_name=f"H(volume) [{method}]",
        status=status,
        observed=H,
        expected_range=(0.70, 0.90),
        interpretation=interp,
        reference="Lobato & Velasco (2000)",
    )


def check_method_agreement(
    H_dfa: float, H_rs: float, tolerance: float = 0.10
) -> ValidationResult:
    """DFA and R/S should roughly agree, but R/S typically overestimates.

    Large disagreement suggests structural breaks, trends, or
    short-range dependence inflating R/S (Lo 1991).
    """
    diff = abs(H_dfa - H_rs)
    rs_higher = H_rs > H_dfa

    if diff <= tolerance:
        status = Status.PASS
        interp = f"Methods agree within {tolerance} (diff={diff:.3f}). Estimates are stable."
    elif diff <= 0.15:
        status = Status.WARN
        interp = (
            f"Moderate disagreement (diff={diff:.3f}). "
            f"{'R/S higher than DFA — typical, reflects short-range dependence inflation.' if rs_higher else 'DFA higher than R/S — unusual, check for trends in data.'}"
        )
    else:
        status = Status.FAIL
        interp = (
            f"Large disagreement (diff={diff:.3f}). "
            "Suggests structural breaks or strong short-range dependence. "
            "DFA estimate is more reliable. R/S result should be treated with caution."
        )

    return ValidationResult(
        check_name="DFA vs R/S agreement",
        status=status,
        observed=diff,
        expected_range=(0.0, tolerance),
        interpretation=interp,
        reference="Lo (1991)",
        detail=f"H_DFA={H_dfa:.4f}, H_RS={H_rs:.4f}",
    )


def check_stationarity_consistency(
    adf_stationary: bool, kpss_stationary: bool, series_name: str
) -> ValidationResult:
    """ADF and KPSS should agree on stationarity status.

    - ADF rejects + KPSS doesn't reject → stationary (good)
    - ADF doesn't reject + KPSS rejects → nonstationary (transform needed)
    - Both reject → trend-stationary (difference first)
    - Neither rejects → inconclusive (low power, need more data)
    """
    if adf_stationary and kpss_stationary:
        status = Status.PASS
        interp = "Both tests agree: series is stationary. Safe for Hurst estimation."
    elif not adf_stationary and not kpss_stationary:
        status = Status.WARN
        interp = (
            "Neither test conclusive. May need more data or the series is "
            "near the boundary. Proceed with caution."
        )
    elif not adf_stationary and kpss_stationary is False:
        status = Status.FAIL
        interp = (
            "Series is nonstationary. Hurst estimate on raw series will be "
            "inflated. Transform to returns or differences first."
        )
    else:
        # Both reject — trend-stationary
        status = Status.WARN
        interp = "Trend-stationary. Consider detrending or differencing before estimation."

    return ValidationResult(
        check_name=f"Stationarity ({series_name})",
        status=status,
        observed=float(adf_stationary),
        expected_range=(1.0, 1.0),
        interpretation=interp,
        reference="ADF: Dickey & Fuller (1979), KPSS: Kwiatkowski et al. (1992)",
    )


def check_sample_size(n: int, method: str = "DFA") -> ValidationResult:
    """Minimum sample size for reliable Hurst estimation.

    DFA needs at least 256 points (preferably 512+).
    Below that, the log-log regression has too few points.
    """
    if n >= 512:
        status = Status.PASS
        interp = f"n={n} is sufficient for reliable {method} estimation."
    elif n >= 256:
        status = Status.WARN
        interp = (
            f"n={n} is marginal. Estimates will have wider CIs. "
            "Consider using a longer time window if available."
        )
    else:
        status = Status.FAIL
        interp = (
            f"n={n} is too short for reliable {method}. "
            "Need at least 256, preferably 512+. Results may be unreliable."
        )

    return ValidationResult(
        check_name=f"Sample size for {method}",
        status=status,
        observed=float(n),
        expected_range=(512.0, float("inf")),
        interpretation=interp,
        reference="Peng et al. (1994), Kantelhardt et al. (2002)",
    )


def check_loglog_fit(r_squared: float, method: str = "DFA") -> ValidationResult:
    """Log-log regression R² should be high (>0.95) for valid Hurst estimate.

    If the log-log relationship is not linear, the Hurst exponent is
    not well-defined. Low R² suggests multifractality or crossover behavior.
    """
    if r_squared >= 0.95:
        status = Status.PASS
        interp = "Excellent log-log linearity. Monofractal assumption holds."
    elif r_squared >= 0.90:
        status = Status.WARN
        interp = (
            "Decent fit but some deviation from linearity. "
            "May indicate mild multifractality or crossover at certain scales. "
            "Consider MFDFA for a fuller picture."
        )
    else:
        status = Status.FAIL
        interp = (
            f"R²={r_squared:.3f} is poor. The log-log relationship is not linear. "
            "A single Hurst exponent does not adequately describe this series. "
            "Use MFDFA to characterize the scale-dependent structure."
        )

    return ValidationResult(
        check_name=f"Log-log fit [{method}]",
        status=status,
        observed=r_squared,
        expected_range=(0.95, 1.0),
        interpretation=interp,
        reference="Kantelhardt et al. (2002)",
    )


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------


def validate_ticker(
    ticker: str,
    H_returns: float,
    H_abs_returns: float,
    H_volume: float,
    H_rs_returns: float | None = None,
    r_squared_returns: float | None = None,
    r_squared_abs_returns: float | None = None,
    r_squared_volume: float | None = None,
    adf_returns: bool | None = None,
    kpss_returns: bool | None = None,
    n_observations: int | None = None,
) -> ValidationReport:
    """Run full validation suite for a single ticker.

    Pass in whatever results you have; checks with None inputs are skipped.
    Returns a ValidationReport with all applicable checks.
    """
    report = ValidationReport(ticker=ticker)

    if n_observations is not None:
        report.add(check_sample_size(n_observations))

    if adf_returns is not None and kpss_returns is not None:
        report.add(check_stationarity_consistency(adf_returns, kpss_returns, "log_returns"))

    if not np.isnan(H_returns):
        report.add(check_returns_hurst(H_returns))

    if not np.isnan(H_abs_returns):
        report.add(check_volatility_hurst(H_abs_returns))

    if not np.isnan(H_volume):
        report.add(check_volume_hurst(H_volume))

    if H_rs_returns is not None and not np.isnan(H_rs_returns) and not np.isnan(H_returns):
        report.add(check_method_agreement(H_returns, H_rs_returns))

    if r_squared_returns is not None and not np.isnan(r_squared_returns):
        report.add(check_loglog_fit(r_squared_returns, "DFA(returns)"))

    if r_squared_abs_returns is not None and not np.isnan(r_squared_abs_returns):
        report.add(check_loglog_fit(r_squared_abs_returns, "DFA(|returns|)"))

    if r_squared_volume is not None and not np.isnan(r_squared_volume):
        report.add(check_loglog_fit(r_squared_volume, "DFA(volume)"))

    return report
