"""Netting set de tres swaps: beneficio del netting y métricas regulatorias (EEPE y EAD)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ccr_engine.ecb import curve_from_dataframe, fetch_ecb_spot_rates
from ccr_engine.exposure import (
    exposure_profile,
    netting_set_values,
    simulation_grid,
    swap_values,
)
from ccr_engine.hull_white import HullWhite1F
from ccr_engine.instruments import InterestRateSwap
from ccr_engine.regulatory import effective_ee, regulatory_metrics

ROOT = Path(__file__).resolve().parents[1]
A, SIGMA, N_PATHS = 0.03, 0.01, 5_000

df = fetch_ecb_spot_rates(source=ROOT / "data" / "ecb_raw.csv")
curve = curve_from_dataframe(df)
model = HullWhite1F(curve, a=A, sigma=SIGMA)


def at_par(notional, maturity, freq, payer):
    k = InterestRateSwap(notional, 0.0, maturity, freq).par_rate(curve)
    return InterestRateSwap(notional, k, maturity, freq, payer=payer)


# Cartera con una misma contraparte: posiciones en sentidos opuestos.
swaps = {
    "Payer 5A 10 M€": at_par(10_000_000, 5, 2, payer=True),
    "Receiver 10A 5 M€": at_par(5_000_000, 10, 1, payer=False),
    "Payer 2A 8 M€": at_par(8_000_000, 2, 4, payer=True),
}

times = simulation_grid(np.linspace(0, 10, 10 * 12 + 1), swaps.values())
x = model.simulate_x(times, N_PATHS, seed=2026)

profiles = {name: exposure_profile(times, swap_values(s, model, times, x))
            for name, s in swaps.items()}
netted = exposure_profile(times, netting_set_values(swaps.values(), model, times, x))
sum_standalone_ee = sum(p.ee for p in profiles.values())

# ---------- Métricas regulatorias ----------
print(f"{'Operación':<22}{'EPE':>12}{'EEPE':>12}{'EAD':>12}")
for name, p in profiles.items():
    m = regulatory_metrics(times, p.ee)
    print(f"{name:<22}{m.epe:>12,.0f}{m.eepe:>12,.0f}{m.ead:>12,.0f}")
standalone_ead = sum(regulatory_metrics(times, p.ee).ead for p in profiles.values())
m_net = regulatory_metrics(times, netted.ee)
print(f"{'Netting set':<22}{m_net.epe:>12,.0f}{m_net.eepe:>12,.0f}{m_net.ead:>12,.0f}")
print(f"\nSuma de EAD sin netting: {standalone_ead:,.0f} €")
print(f"EAD con netting:         {m_net.ead:,.0f} €")
print(f"Beneficio del netting:   {1 - m_net.ead / standalone_ead:.0%}")

# ---------- Gráficos ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

ax1.plot(times, sum_standalone_ee / 1e3, linewidth=2, label="Suma de EE sin netting")
ax1.plot(times, netted.ee / 1e3, linewidth=2, label="EE del netting set")
ax1.set_title("Beneficio del netting")
ax1.set_xlabel("Años")
ax1.set_ylabel("Miles de €")
ax1.grid(alpha=0.3)
ax1.legend()

first_year = times <= 1 + 1e-12
ax2.plot(times[first_year], netted.ee[first_year] / 1e3, linewidth=2, label="EE")
ax2.plot(times[first_year], effective_ee(netted.ee)[first_year] / 1e3, linewidth=2,
         linestyle="--", label="Effective EE")
ax2.axhline(m_net.eepe / 1e3, color="black", linewidth=1, label=f"EEPE = {m_net.eepe / 1e3:,.0f} k€")
ax2.set_title("Netting set: primer año (horizonte regulatorio)")
ax2.set_xlabel("Años")
ax2.grid(alpha=0.3)
ax2.legend()

fig.tight_layout()
fig.savefig(ROOT / "docs" / "netting_and_eepe.png", dpi=150)
print("\nGráfico guardado en docs/netting_and_eepe.png")