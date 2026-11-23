# CCR Exposure Engine

[![CI](https://github.com/aleandradewo-collab/ccr-exposure-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/aleandradewo-collab/ccr-exposure-engine/actions/workflows/ci.yml)

An open-source counterparty credit risk (CCR) exposure engine built from scratch in Python,
inspired by the Internal Model Method (IMM) used by banks under the EU Capital Requirements
Regulation (CRR).

The engine simulates future market scenarios with Monte Carlo, prices a portfolio of
derivatives on every scenario and date, and computes the exposure metrics used for risk
limits and regulatory capital. Every module is backed by the underlying theory and tested
against known analytical results.

> Built with public market data only (ECB yield curves) and public regulation (CRR, BCBS).
> Not affiliated with any financial institution.

![Hull-White scenarios](docs/hull_white_fan.png)

## Roadmap

- [x] Project structure, discount factors and CI
- [x] Yield curve from ECB data, log-linear interpolation and forward rates
- [x] Automated daily download of the ECB curve (GitHub Actions)
- [x] Pricing of interest rate swaps: legs, par rate, NPV and DV01
- [x] Hull-White 1F model: exact simulation and closed-form zero-coupon bonds
- [ ] Hull-White calibration
- [ ] Swap valuation on simulated scenarios (exposure cube)
- [ ] Exposure profiles: EE, PFE, EPE, Effective EE, EEPE and EAD
- [ ] Netting sets and collateral (CSA thresholds, MTA, margin period of risk)
- [ ] Backtesting of the exposure model
- [ ] SA-CCR implementation and IMM vs SA-CCR capital comparison
- [ ] FX forwards

## What has been built so far

### 1. Market data: the ECB yield curve
The engine uses the euro area AAA-rated government bond yield curve published daily by the
European Central Bank (zero-coupon spot rates, continuous compounding).

- A scheduled GitHub Actions workflow downloads the latest curve every business day and
  commits it to `data/`, building a historical dataset for future calibration and backtesting.
- Between pillars, the curve interpolates `r(t)·t` linearly (equivalent to piecewise-constant
  instantaneous forward rates), which guarantees monotonically decreasing discount factors.

![ECB curve](docs/ecb_curve.png)

### 2. Interest rate swap pricing
Fixed-vs-floating swaps priced from discount factors:

- Fixed leg: `N · K · A`, where `A = Σ τᵢ P(0,Tᵢ)` is the annuity
- Floating leg: projected with forward rates, `N · Σ τᵢ F(0;Tᵢ₋₁,Tᵢ) P(0,Tᵢ)`
- Par rate, NPV and DV01 (bump-and-revalue)

Tests check the results against theory, e.g. the floating leg telescopes to
`N · (P(0,T₀) − P(0,Tₙ))`, a swap struck at par has zero value and payer and receiver are
exact opposites.

### 3. Hull-White one-factor model
The short rate is modelled as `r(t) = x(t) + φ(t)`, where `x` is an Ornstein-Uhlenbeck
process and `φ` fits today's curve exactly.

- **Exact simulation** of `x(t)` (no discretisation error), with seeds for reproducibility
- **Closed-form zero-coupon bonds** `P(t,T)` on every scenario: the whole future curve
  depends on a single state variable, which keeps the simulation cheap
- **Martingale tests:** the Monte Carlo average of the stochastic discount factor matches
  today's market prices within 4 standard errors, which validates the model end to end

## Project structure

```
ccr-exposure-engine/
├── .github/workflows/
│   ├── ci.yml                  # lint + tests on every push
│   └── update-ecb-curve.yml    # daily download of the ECB curve
├── src/ccr_engine/
│   ├── curves.py               # YieldCurve: discounting, interpolation, forwards
│   ├── ecb.py                  # ECB data loading
│   ├── instruments.py          # InterestRateSwap
│   └── hull_white.py           # HullWhite1F model
├── tests/                      # unit and statistical tests
├── scripts/                    # plots and examples
├── data/                       # ECB curves (updated automatically)
└── docs/                       # figures
```

## Installation

```bash
git clone https://github.com/aleandradewo-collab/ccr-exposure-engine.git
cd ccr-exposure-engine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Usage

```bash
python scripts/plot_ecb_curve.py data/ecb_raw.csv   # plot the latest ECB curve
python scripts/price_swap.py                         # par rates and a sample swap valuation
python scripts/plot_hull_white.py                    # simulate and plot short-rate scenarios
```

## Modelling assumptions and limitations

This is an educational project; several simplifications are made on purpose and documented
here:

- **Single-curve framework:** the AAA government curve is used both to project and to
  discount cash flows. Banks use €STR (OIS) for discounting and Euribor curves for projection.
- **Simplified schedules:** equal accrual periods, no day-count conventions or business-day
  calendars, same frequency on both legs.
- **Hull-White 1F:** Gaussian (negative rates are possible) and one-factor (all maturities
  move together within a scenario).
- **Uncalibrated parameters:** mean reversion and volatility are currently illustrative
  (`a = 0.03`, `σ = 1%`). Calibration is on the roadmap.
- **Interpolation:** piecewise-constant forwards cause small jumps in the model drift at curve
  pillars; a smoother scheme is a possible improvement.

## References

- Brigo, D. and Mercurio, F. (2006). *Interest Rate Models: Theory and Practice*. Springer.
- Gregory, J. (2020). *The xVA Challenge*. Wiley.
- Regulation (EU) No 575/2013 (CRR), Articles 283–294: Internal Model Method.
- European Central Bank: euro area yield curves.