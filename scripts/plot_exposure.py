"""Perfil de exposición (EE y PFE) de un swap a 5 años sobre la curva del BCE."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ccr_engine.ecb import curve_from_dataframe, fetch_ecb_spot_rates
from ccr_engine.exposure import exposure_profile, simulation_grid, swap_values
from ccr_engine.hull_white import HullWhite1F
from ccr_engine.instruments import InterestRateSwap

ROOT = Path(__file__).resolve().parents[1]
A, SIGMA, N_PATHS = 0.03, 0.01, 5_000
NOTIONAL, MATURITY, FREQ = 10_000_000, 5, 2

df = fetch_ecb_spot_rates(source=ROOT / "data" / "ecb_raw.csv")
curve = curve_from_dataframe(df)
model = HullWhite1F(curve, a=A, sigma=SIGMA)

# Payer swap contratado hoy al tipo par: vale 0 hoy.
k_par = InterestRateSwap(NOTIONAL, 0.0, MATURITY, FREQ).par_rate(curve)
swap = InterestRateSwap(NOTIONAL, k_par, MATURITY, FREQ, payer=True)

times = simulation_grid(np.linspace(0, MATURITY, MATURITY * 12 + 1), [swap])
x = model.simulate_x(times, N_PATHS, seed=2026)
values = swap_values(swap, model, times, x)
prof = exposure_profile(times, values)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(times, values[:, :20] / 1e3, color="grey", alpha=0.25, linewidth=0.8)
ax.plot(times, prof.pfe / 1e3, linewidth=2, label="PFE 97,5 %")
ax.plot(times, prof.ee / 1e3, linewidth=2, label="EE")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Años")
ax.set_ylabel("Miles de €")
ax.set_title(f"Payer swap {MATURITY}A, 10 M€ al par ({100 * k_par:.2f} %), Hull-White sobre curva BCE")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "docs" / "exposure_profile.png", dpi=150)

i_ee, i_pfe = np.argmax(prof.ee), np.argmax(prof.pfe)
print(f"Tipo par: {100 * k_par:.3f} %")
print(f"EE máxima:  {prof.ee[i_ee]:>12,.0f} € en t = {times[i_ee]:.2f} años")
print(f"PFE máxima: {prof.pfe[i_pfe]:>12,.0f} € en t = {times[i_pfe]:.2f} años")
print("Gráfico guardado en docs/exposure_profile.png")