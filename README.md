# Fractal Price-Volume Dashboard

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19611544.svg)](https://doi.org/10.5281/zenodo.19611544)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fractal-pv.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Interactive explorer for fractal coupling between price volatility and trading volume in S&P 500 equities.

**[Launch the dashboard](https://fractal-pv.streamlit.app)**

## What This Shows

Price volatility and trading volume both exhibit long-range memory (Hurst exponents > 0.5). This dashboard lets you explore how these two persistence structures co-evolve through time for individual stocks, and what that co-movement tells us about future market liquidity.

Key findings from the [accompanying research](https://doi.org/10.5281/zenodo.19611544):
- **Temporal coupling** is strong and positive in 49/50 equities (mean *r* = 0.665)
- **Static coupling** (cross-sectional) is null (*r* = -0.02)
- **CII predicts illiquidity**: two-way clustered *t* = 2.90, *p* = 0.004
- **CII does not predict volatility** once proper clustering is applied (*t* = 0.84)
- **Crisis amplification**: coupling nearly doubles during COVID-19

## Methods

- **DFA** (Detrended Fluctuation Analysis) for Hurst exponent estimation
- **Rolling dual-Hurst** analysis with aligned windows (*W* = 500, Δ = 20)
- **Block bootstrap** confidence intervals (Politis & Romano 1994)
- **50 S&P 500 stocks**, daily data 2015-2026 via Yahoo Finance

## Setup

```bash
pip install -e ".[dev,test]"
python -c "from fractal_pv.data import fetch_universe, SP500_SAMPLE; fetch_universe(SP500_SAMPLE)"
streamlit run app.py
```

## Citation

```bibtex
@misc{hari2026fractal,
  author={Hari, Dinesh},
  title={Static and Temporal Fractal Coupling Between Volatility and
         Trading Volume: Evidence from {S\&P}~500 Stocks, 2015--2026},
  year={2026},
  doi={10.5281/zenodo.19611544}
}
```

## License

MIT
