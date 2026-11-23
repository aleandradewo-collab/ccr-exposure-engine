"""Valora swaps con la última curva del BCE descargada en data/ecb_raw.csv."""

from pathlib import Path

from ccr_engine.ecb import curve_from_dataframe, fetch_ecb_spot_rates
from ccr_engine.instruments import InterestRateSwap

ROOT = Path(__file__).resolve().parents[1]

df = fetch_ecb_spot_rates(source=ROOT / "data" / "ecb_raw.csv")
curve = curve_from_dataframe(df)
print(f"Curva BCE del {df['date'].max().date()}\n")

print("Tipos swap par (pagos anuales):")
for maturity in [1, 2, 5, 10, 20, 30]:
    swap = InterestRateSwap(notional=1, fixed_rate=0.0, maturity=maturity)
    print(f"  {maturity:>2} años: {100 * swap.par_rate(curve):.3f} %")

swap = InterestRateSwap(notional=10_000_000, fixed_rate=0.025, maturity=5, payer=True)
print("\nPayer swap 5 años, nominal 10 M€, fijo 2,5 %:")
print(f"  Pata flotante: {swap.floating_leg_pv(curve):>14,.2f} €")
print(f"  Pata fija:     {swap.fixed_leg_pv(curve):>14,.2f} €")
print(f"  NPV:           {swap.npv(curve):>14,.2f} €")
print(f"  DV01:          {swap.dv01(curve):>14,.2f} € por punto básico")