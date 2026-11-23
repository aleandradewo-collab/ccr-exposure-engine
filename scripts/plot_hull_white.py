"""Simula el tipo corto con Hull-White sobre la curva del BCE y dibuja el abanico de escenarios."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ccr_engine.ecb import curve_from_dataframe, fetch_ecb_spot_rates
from ccr_engine.hull_white import HullWhite1F

ROOT = Path(__file__).resolve().parents[1]

# Parámetros ilustrativos (aún sin calibrar): reversión lenta y ~100 pb de volatilidad anual.
A, SIGMA = 0.03, 0.01
N_PATHS, HORIZON = 5_000, 10

df = fetch_ecb_spot_rates(source=ROOT / "data" / "ecb_raw.csv")
curve = curve_from_dataframe(df)
model = HullWhite1F(curve, a=A, sigma=SIGMA)

times = np.linspace(0, HORIZON, HORIZON * 12 + 1)  # rejilla mensual
x = model.simulate_x(times, N_PATHS, seed=2026)
r = np.stack([model.short_rate(t, x[i]) for i, t in enumerate(times)])

p2_5, p50, p97_5 = np.percentile(r, [2.5, 50, 97.5], axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(times, 100 * r[:, :30], color="grey", alpha=0.25, linewidth=0.8)
ax.fill_between(times, 100 * p2_5, 100 * p97_5, alpha=0.2, label="Intervalo 95 %")
ax.plot(times, 100 * p50, linewidth=2, label="Mediana")
ax.set_xlabel("Años")
ax.set_ylabel("Tipo corto r(t) (%)")
ax.set_title(f"Hull-White 1F sobre la curva BCE ({N_PATHS:,} escenarios, a={A}, σ={SIGMA:.0%})")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "docs" / "hull_white_fan.png", dpi=150)

print(f"Tipo corto hoy:           {100 * r[0, 0]:.2f} %")
for year in [1, 5, 10]:
    i = year * 12
    print(f"Año {year:>2}: mediana {100 * p50[i]:.2f} %, "
          f"intervalo 95 % [{100 * p2_5[i]:.2f} %, {100 * p97_5[i]:.2f} %]")
print("Gráfico guardado en docs/hull_white_fan.png")