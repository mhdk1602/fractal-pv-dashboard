"""Inference engine — draws conclusions from Hurst analysis results.

Takes computed Hurst exponents, bootstrap CIs, and cross-correlation
results and produces structured findings with theoretical context.
Each finding cites the relevant literature and states confidence level.

This is what turns numbers into narrative — the "so what" layer.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Finding:
    """A single analytical finding with evidence and context."""

    claim: str
    evidence: str
    confidence: str  # "high", "moderate", "low"
    theoretical_context: str
    implication: str
    references: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        conf_icon = {"high": "●", "moderate": "◐", "low": "○"}[self.confidence]
        return (
            f"{conf_icon} [{self.confidence.upper()}] {self.claim}\n"
            f"  Evidence: {self.evidence}\n"
            f"  Context: {self.theoretical_context}\n"
            f"  Implication: {self.implication}\n"
            f"  Refs: {', '.join(self.references)}"
        )


@dataclass
class InferenceReport:
    """Collection of findings from an analysis run."""

    ticker: str
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def __str__(self) -> str:
        lines = [f"=== Inference Report: {self.ticker} ==="]
        for i, f in enumerate(self.findings, 1):
            lines.append(f"\n--- Finding {i} ---")
            lines.append(str(f))
        return "\n".join(lines)


def infer_market_efficiency(H_returns: float, p_value: float | None = None) -> Finding:
    """What does H(returns) tell us about market efficiency?"""
    if p_value is not None and p_value >= 0.05:
        return Finding(
            claim=f"Returns are consistent with the efficient market hypothesis (H={H_returns:.3f}, p={p_value:.3f}).",
            evidence=f"H(returns)={H_returns:.3f} is not significantly different from 0.5.",
            confidence="high" if p_value > 0.10 else "moderate",
            theoretical_context=(
                "Under the EMH, returns follow a random walk (H=0.5). "
                "This is the default expectation for liquid, developed-market equities."
            ),
            implication="No evidence of exploitable long-memory in returns. Standard risk models apply.",
            references=["Lo (1991)", "Fama (1970)"],
        )
    elif H_returns > 0.55:
        return Finding(
            claim=f"Returns show persistent behavior (H={H_returns:.3f}), suggesting momentum.",
            evidence=f"H(returns)={H_returns:.3f} > 0.5" + (f", p={p_value:.3f}" if p_value else "") + ".",
            confidence="moderate" if (p_value and p_value < 0.05) else "low",
            theoretical_context=(
                "Persistent returns (H>0.5) imply positive autocorrelation at long lags. "
                "This is more common in emerging markets or illiquid assets (Di Matteo 2005). "
                "For developed-market large caps, verify this isn't an artifact of short-range dependence."
            ),
            implication=(
                "If genuine, suggests momentum strategies may have statistical backing for this asset. "
                "But: Lo (1991) showed that naive R/S can mistake short-range dependence for long memory. "
                "Cross-check with DFA and bootstrap CI."
            ),
            references=["Di Matteo et al. (2005)", "Lo (1991)"],
        )
    else:
        return Finding(
            claim=f"Returns show anti-persistent behavior (H={H_returns:.3f}), suggesting mean reversion.",
            evidence=f"H(returns)={H_returns:.3f} < 0.5" + (f", p={p_value:.3f}" if p_value else "") + ".",
            confidence="moderate" if (p_value and p_value < 0.05) else "low",
            theoretical_context=(
                "Anti-persistent returns (H<0.5) imply negative autocorrelation: "
                "increases tend to be followed by decreases. This can arise from "
                "bid-ask bounce, market microstructure effects, or overreaction-correction dynamics."
            ),
            implication="Mean-reversion strategies may be applicable. Verify at multiple time scales.",
            references=["Lo (1991)", "Mandelbrot (1963)"],
        )


def infer_volatility_persistence(H_vol: float, p_value: float | None = None) -> Finding:
    """What does H(|returns|) tell us about volatility clustering?"""
    if 0.65 <= H_vol <= 0.85:
        return Finding(
            claim=f"Volatility exhibits expected long memory (H={H_vol:.3f}).",
            evidence=f"H(|returns|)={H_vol:.3f} is within the literature range of 0.7–0.8.",
            confidence="high",
            theoretical_context=(
                "Long memory in volatility is one of the most robust empirical facts in finance. "
                "It means that high-volatility periods tend to cluster and decay slowly — "
                "much slower than GARCH models predict."
            ),
            implication=(
                "Pipeline is producing expected results. Volatility forecasting models should "
                "account for long memory (FIGARCH or HAR-RV models, not standard GARCH)."
            ),
            references=["Bollerslev & Jubinski (1999)", "Ding et al. (1993)"],
        )
    elif H_vol > 0.85:
        return Finding(
            claim=f"Volatility persistence is unusually high (H={H_vol:.3f}).",
            evidence=f"H(|returns|)={H_vol:.3f} exceeds the typical 0.7–0.8 range.",
            confidence="moderate",
            theoretical_context=(
                "H > 0.85 for volatility is high even by financial standards. "
                "Could indicate: (a) structural breaks in volatility regime, "
                "(b) the analysis period spans a major crisis, or "
                "(c) the asset has unusually clustered volatility (e.g., biotech, crypto)."
            ),
            implication="Investigate whether this is driven by a specific regime (crisis period). Rolling Hurst may reveal the source.",
            references=["Bollerslev & Jubinski (1999)"],
        )
    else:
        return Finding(
            claim=f"Volatility persistence is lower than expected (H={H_vol:.3f}).",
            evidence=f"H(|returns|)={H_vol:.3f} is below the typical 0.7–0.8 range.",
            confidence="moderate",
            theoretical_context=(
                "Low volatility persistence could mean: (a) the asset is very liquid with "
                "rapid volatility mean-reversion, (b) the sample is too short, or "
                "(c) structural breaks are disrupting the long-memory estimate."
            ),
            implication="Check sample size (need 500+). If adequate, this asset may have genuinely different volatility dynamics.",
            references=["Bollerslev & Jubinski (1999)"],
        )


def infer_volume_memory(H_vol: float) -> Finding:
    """What does H(volume) tell us about trading activity persistence?"""
    if H_vol >= 0.70:
        return Finding(
            claim=f"Volume has strong long memory (H={H_vol:.3f}), as expected.",
            evidence=f"H(volume)={H_vol:.3f} is within the literature range of 0.7–0.9.",
            confidence="high",
            theoretical_context=(
                "Volume persistence is one of the strongest long-memory signals in financial data. "
                "High-volume periods beget more high-volume periods, reflecting sustained information "
                "arrival, herding, or liquidity provision patterns."
            ),
            implication="Volume dynamics follow expected fractal behavior. Cross-correlation with volatility is the next analytical step.",
            references=["Lobato & Velasco (2000)", "Plerou et al. (2003)"],
        )
    else:
        return Finding(
            claim=f"Volume memory is weaker than expected (H={H_vol:.3f}).",
            evidence=f"H(volume)={H_vol:.3f} is below the typical 0.7–0.9 range.",
            confidence="moderate",
            theoretical_context=(
                "Low volume persistence is unusual. Most assets show H(volume) > 0.7. "
                "Possible explanations: very liquid/heavily traded asset where volume "
                "reverts quickly, or sample period that excludes major trading events."
            ),
            implication="Investigate whether this is sample-specific. Compare across time windows.",
            references=["Lobato & Velasco (2000)"],
        )


def infer_price_volume_coupling(
    corr: float, p_value: float, H_vol_mean: float, H_price_mean: float
) -> Finding:
    """What does the cross-correlation of rolling Hurst values tell us?"""
    gap = H_vol_mean - H_price_mean

    if p_value >= 0.05:
        return Finding(
            claim=f"No significant linear coupling between price and volume fractality (r={corr:.3f}, p={p_value:.3f}).",
            evidence=f"Pearson r={corr:.3f}, p={p_value:.3f}. Mean gap: H(vol)-H(|ret|)={gap:.3f}.",
            confidence="moderate",
            theoretical_context=(
                "The absence of linear correlation doesn't mean the series are independent — "
                "they may be coupled at specific scales (detectable via DCCA) or in "
                "specific regimes (detectable via rolling analysis with regime conditioning)."
            ),
            implication="Proceed with DCCA for scale-dependent cross-correlation. Check for nonlinear coupling.",
            references=["Podobnik et al. (2010)", "Ardalankia et al. (2019)"],
        )
    elif corr > 0:
        return Finding(
            claim=f"Positive coupling: when price fractality increases, volume fractality increases (r={corr:.3f}).",
            evidence=f"Pearson r={corr:.3f}, p={p_value:.3f}. Mean gap: H(vol)-H(|ret|)={gap:.3f}.",
            confidence="high" if p_value < 0.01 else "moderate",
            theoretical_context=(
                "Positive coupling supports the mixture-of-distributions hypothesis (MDH): "
                "information arrival drives both price volatility and trading volume simultaneously. "
                "When the market is in a 'trending' regime (high H for prices), volume is also more persistent."
            ),
            implication=(
                "The MDH framework appears applicable. This suggests a common latent factor "
                "(information flow) drives both series. Investigate whether the coupling "
                "strengthens during crisis periods."
            ),
            references=["Bollerslev & Jubinski (1999)", "Plerou et al. (2003)"],
        )
    else:
        return Finding(
            claim=f"Negative coupling: price and volume fractality diverge (r={corr:.3f}).",
            evidence=f"Pearson r={corr:.3f}, p={p_value:.3f}. Mean gap: H(vol)-H(|ret|)={gap:.3f}.",
            confidence="high" if p_value < 0.01 else "moderate",
            theoretical_context=(
                "Negative coupling is less common but theoretically interesting. "
                "It could indicate: (a) high-volume periods are associated with efficient "
                "(H≈0.5) price discovery, while low-volume periods allow momentum to build, "
                "or (b) liquidity provision and price dynamics have opposing persistence structures."
            ),
            implication="This is a novel finding worth investigating further. Check whether it's robust across time windows and sectors.",
            references=["Ardalankia et al. (2019)"],
        )


def infer_bootstrap_significance(
    H_point: float, ci_low: float, ci_high: float, p_value: float, series_name: str
) -> Finding:
    """Interpret bootstrap CI and significance test."""
    ci_contains_05 = ci_low <= 0.5 <= ci_high

    if p_value < 0.01:
        confidence = "high"
    elif p_value < 0.05:
        confidence = "moderate"
    else:
        confidence = "low"

    if p_value < 0.05 and not ci_contains_05:
        return Finding(
            claim=f"H({series_name})={H_point:.3f} is significantly different from 0.5 (p={p_value:.3f}).",
            evidence=f"95% CI: [{ci_low:.3f}, {ci_high:.3f}] does not contain 0.5.",
            confidence=confidence,
            theoretical_context=(
                "A Hurst exponent significantly different from 0.5 indicates the series "
                "has genuine long-range dependence (if H>0.5) or anti-persistence (if H<0.5) "
                "that is not explained by random variation."
            ),
            implication=f"The {'persistence' if H_point > 0.5 else 'anti-persistence'} in {series_name} is statistically reliable.",
            references=["Politis & Romano (1994)"],
        )
    else:
        return Finding(
            claim=f"H({series_name})={H_point:.3f} is NOT significantly different from 0.5 (p={p_value:.3f}).",
            evidence=f"95% CI: [{ci_low:.3f}, {ci_high:.3f}] contains 0.5.",
            confidence=confidence,
            theoretical_context=(
                "Failure to reject H=0.5 means we cannot distinguish the observed Hurst "
                "exponent from what a random walk would produce. This does not prove "
                "the series IS a random walk — only that we lack evidence to say otherwise."
            ),
            implication=f"Cannot claim {series_name} has long memory or anti-persistence based on this sample.",
            references=["Politis & Romano (1994)"],
        )


def run_inference(
    ticker: str,
    H_returns: float,
    H_abs_returns: float,
    H_volume: float,
    p_value_returns: float | None = None,
    p_value_abs_returns: float | None = None,
    ci_returns: tuple[float, float] | None = None,
    ci_abs_returns: tuple[float, float] | None = None,
    rolling_corr: float | None = None,
    rolling_corr_p: float | None = None,
) -> InferenceReport:
    """Run full inference suite for a single ticker."""
    report = InferenceReport(ticker=ticker)

    report.add(infer_market_efficiency(H_returns, p_value_returns))
    report.add(infer_volatility_persistence(H_abs_returns, p_value_abs_returns))
    report.add(infer_volume_memory(H_volume))

    if ci_returns is not None and p_value_returns is not None:
        report.add(infer_bootstrap_significance(
            H_returns, ci_returns[0], ci_returns[1], p_value_returns, "returns"
        ))

    if ci_abs_returns is not None and p_value_abs_returns is not None:
        report.add(infer_bootstrap_significance(
            H_abs_returns, ci_abs_returns[0], ci_abs_returns[1], p_value_abs_returns, "|returns|"
        ))

    if rolling_corr is not None and rolling_corr_p is not None:
        report.add(infer_price_volume_coupling(
            rolling_corr, rolling_corr_p, H_volume, H_abs_returns
        ))

    return report
