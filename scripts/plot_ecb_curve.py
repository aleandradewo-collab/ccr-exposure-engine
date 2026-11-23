"""Descarga la última curva del BCE, la guarda en CSV y dibuja tipos cupón cero y forwards."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ccr_engine.ecb import curve_from_dataframe, fetch_ecb_spot_rates

ROOT = Path(__file__).resolve().parents[1]

df = fetch_ecb_spot_rates()
(ROOT / "data").mkdir(exist_ok=True)
fecha = df["date"].max().date()
df.to_csv(ROOT / "data" / f"ecb_curve_{fecha}.csv", index=False)
print(df)

curve = curve_from_dataframe(df)
t = np.linspace(0.25, 30, 300)
fwd = curve.forward_rate(t - 0.25, t)  # forwards a 3 meses

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t, 100 * curve.zero_rate(t), label="Cupón cero r(t)")
ax.plot(t, 100 * fwd, label="Forward 3M", linestyle="--")
ax.scatter(curve.tenors, 100 * curve.zero_rates, color="black", zorder=3, label="Pilares BCE")
ax.set_xlabel("Plazo (años)")
ax.set_ylabel("Tipo (%)")
ax.set_title(f"Curva soberana AAA zona euro (BCE), {fecha}")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
(ROOT / "docs").mkdir(exist_ok=True)
fig.savefig(ROOT / "docs" / "ecb_curve.png", dpi=150)
print("Gráfico guardado en docs/ecb_curve.png")