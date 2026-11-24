"""Efecto del colateral en la exposición de un swap a 5 años: varios CSA comparados."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ccr_engine.collateral import CSA, collateralized_values
from ccr_engine.ecb import curve_from_dataframe, fetch_ecb_spot_rates
from ccr_engine.exposure import exposure_profile, index_of, simulation_grid, swap_values
from ccr_engine.hull_white import HullWhite1F
from ccr_engine.instruments import InterestRateSwap
from ccr_engine.regulatory import regulatory_metrics

ROOT = Path(__file__).resolve().parents[1]
A, SIGMA, N_PATHS = 0.03, 0.01, 5_000

df = fetch_ecb_spot_rates(source=ROOT / "data" / "ecb_raw.csv")
curve = curve_from_dataframe(df)
model = HullWhite1F(curve, a=A, sigma=SIGMA)

k_par = InterestRateSwap(10_000_000, 0.0, 5, 2).par_rate(curve)
swap = InterestRateSwap(10_000_000, k_par, 5, 2, payer=True)

csas = {
    "Sin colateral": None,
    "Threshold 500 k€, MTA 50 k€": CSA(threshold=500_000, mta=50_000),
    "Threshold 0 (VM diario)": CSA(),
    "Threshold 0 + margen inicial 150 k€": CSA(independent_amount=150_000),
}

# Rejilla mensual más las fechas t - MPoR (todos los CSA usan 10 días).
exposure_times = np.linspace(0, 5, 5 * 12 + 1)
times = simulation_grid(exposure_times, [swap], mpor=CSA().mpor)
x = model.simulate_x(times, N_PATHS, seed=2026)
values = swap_values(swap, model, times, x)

fig, ax = plt.subplots(figsize=(9, 5))
print(f"{'Acuerdo':<38}{'EE máx':>12}{'EEPE':>12}{'EAD':>12}")
for name, csa in csas.items():
    if csa is None:
        idx = [index_of(times, t) for t in exposure_times]
        net = values[idx]
    else:
        net = collateralized_values(values, times, exposure_times, csa)
    prof = exposure_profile(exposure_times, net)
    m = regulatory_metrics(exposure_times, prof.ee)
    print(f"{name:<38}{prof.ee.max():>12,.0f}{m.eepe:>12,.0f}{m.ead:>12,.0f}")
    ax.plot(exposure_times, prof.ee / 1e3, linewidth=2, label=name)

ax.set_xlabel("Años")
ax.set_ylabel("EE (miles de €)")
ax.set_title("Payer swap 5A, 10 M€ al par: EE con distintos acuerdos de colateral (MPoR 10 días)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "docs" / "collateral_ee.png", dpi=150)
print("\nGráfico guardado en docs/collateral_ee.png")