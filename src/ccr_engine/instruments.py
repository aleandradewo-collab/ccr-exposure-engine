"""Instrumentos derivados y su valoración."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterestRateSwap:
    """Swap de tipos de interés (IRS) fijo contra flotante.

    Convenciones simplificadas: plazos en años, periodos todos iguales
    (1 / pay_frequency) y la misma frecuencia en las dos patas.
    payer=True significa que pagamos fijo y recibimos flotante.
    """

    notional: float
    fixed_rate: float
    maturity: float
    pay_frequency: int = 1
    payer: bool = True
    start: float = 0.0

    def __post_init__(self):
        if self.notional <= 0:
            raise ValueError("El nominal debe ser positivo")
        if self.pay_frequency not in (1, 2, 4, 12):
            raise ValueError("pay_frequency debe ser 1, 2, 4 o 12 pagos al año")
        if self.start < 0 or self.maturity <= self.start:
            raise ValueError("Debe cumplirse 0 <= start < maturity")
        n = (self.maturity - self.start) * self.pay_frequency
        if not np.isclose(n, round(n)):
            raise ValueError("La duración del swap debe ser un número entero de periodos")

    @property
    def n_periods(self):
        return round((self.maturity - self.start) * self.pay_frequency)

    def payment_times(self):
        """Fechas de pago T_1, ..., T_n (en años desde hoy)."""
        k = np.arange(1, self.n_periods + 1)
        return self.start + k / self.pay_frequency

    def accrual_fractions(self):
        """Fracción de año de cada periodo, tau_i."""
        return np.full(self.n_periods, 1 / self.pay_frequency)

    def annuity(self, curve):
        """Anualidad A = sum tau_i * P(0, T_i)."""
        return np.sum(self.accrual_fractions() * curve.discount(self.payment_times()))

    def fixed_leg_pv(self, curve):
        """Valor presente de la pata fija: N * K * A."""
        return self.notional * self.fixed_rate * self.annuity(curve)

    def floating_leg_pv(self, curve):
        """Valor presente de la pata flotante, flujo a flujo.

        Cada flujo esperado es N * tau_i * F(0; T_{i-1}, T_i), descontado con P(0, T_i).
        """
        ends = self.payment_times()
        starts = np.concatenate([[self.start], ends[:-1]])
        forwards = curve.forward_rate(starts, ends)
        tau = self.accrual_fractions()
        return self.notional * np.sum(tau * forwards * curve.discount(ends))

    def par_rate(self, curve):
        """Tipo fijo que hace que el swap valga cero hoy."""
        return self.floating_leg_pv(curve) / (self.notional * self.annuity(curve))

    def npv(self, curve):
        """Valor del swap para nosotros: flotante - fija si somos payer, al revés si no."""
        value = self.floating_leg_pv(curve) - self.fixed_leg_pv(curve)
        return value if self.payer else -value

    def dv01(self, curve):
        """Cambio de valor si toda la curva sube 1 punto básico (0,01 %)."""
        return self.npv(curve.shifted(1)) - self.npv(curve)