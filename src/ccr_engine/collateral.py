"""Colateral (CSA) y exposición colateralizada con margin period of risk (MPoR)."""

from dataclasses import dataclass

import numpy as np

from ccr_engine.exposure import index_of

BUSINESS_DAYS_PER_YEAR = 250


@dataclass(frozen=True)
class CSA:
    """Términos de un acuerdo de colateral (Credit Support Annex), simplificados.

    threshold: exposición que se tolera sin pedir colateral (simétrico para ambas partes).
    mta: importe mínimo de transferencia (Minimum Transfer Amount).
    independent_amount: margen inicial fijo que nos entrega la contraparte.
    mpor_days: margin period of risk, en días hábiles.
    """

    threshold: float = 0.0
    mta: float = 0.0
    independent_amount: float = 0.0
    mpor_days: int = 10

    def __post_init__(self):
        if self.threshold < 0 or self.mta < 0 or self.independent_amount < 0:
            raise ValueError("threshold, mta e independent_amount no pueden ser negativos")
        if self.mpor_days < 0:
            raise ValueError("El MPoR no puede ser negativo")

    @property
    def mpor(self):
        """MPoR en años."""
        return self.mpor_days / BUSINESS_DAYS_PER_YEAR

    def collateral(self, portfolio_value):
        """Colateral que tenemos (positivo) o hemos entregado (negativo) según el valor.

        Aproximación habitual: el MTA se suma al threshold. Así se evita simular
        la historia diaria de llamadas de margen, y el resultado es conservador.
        """
        v = np.asarray(portfolio_value, dtype=float)
        h = self.threshold + self.mta
        variation_margin = np.where(v > h, v - h, np.where(v < -h, v + h, 0.0))
        return variation_margin + self.independent_amount


def collateralized_values(values, times, exposure_times, csa):
    """Valor de la cartera neto de colateral, V(t) - C(t), en cada fecha de exposición.

    El colateral disponible en t se calculó con el valor de la cartera en t - MPoR:
    es la última llamada de margen que la contraparte atendió antes de quebrar.
    Devuelve un array (len(exposure_times), n_escenarios).
    """
    exposure_times = np.asarray(exposure_times, dtype=float)
    out = np.empty((len(exposure_times), values.shape[1]))
    for k, t in enumerate(exposure_times):
        v_now = values[index_of(times, t)]
        v_lag = values[index_of(times, max(t - csa.mpor, 0.0))]
        out[k] = v_now - csa.collateral(v_lag)
    return out