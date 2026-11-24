"""Métricas regulatorias del método de modelos internos (IMM), CRR artículo 284."""

from dataclasses import dataclass

import numpy as np

ALPHA = 1.4  # multiplicador regulatorio por defecto (CRR art. 284.4)


def effective_ee(ee):
    """Effective EE: EE que no puede bajar. EffEE(t_k) = max(EffEE(t_k-1), EE(t_k))."""
    return np.maximum.accumulate(np.asarray(ee, dtype=float))


def time_average(times, values, horizon):
    """Media ponderada por tiempo de values entre 0 y horizon.

    Usa el valor al final de cada intervalo (t_k-1, t_k], como la fórmula del CRR:
    sum(values_k · Δt_k) / horizon.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if not np.any(np.isclose(times, horizon)):
        raise ValueError(f"El horizonte {horizon} debe ser una fecha de la rejilla")
    mask = (times > 0) & (times <= horizon + 1e-12)
    dt = np.diff(times)[mask[1:]]
    return float(np.sum(values[mask] * dt) / horizon)


@dataclass(frozen=True)
class RegulatoryMetrics:
    horizon: float
    epe: float
    eepe: float
    ead: float


def regulatory_metrics(times, ee, netting_set_maturity=None, alpha=ALPHA):
    """Calcula EPE, EEPE y EAD = alpha · EEPE a partir de un perfil de EE.

    El horizonte es 1 año, o el vencimiento más largo del netting set si todas
    sus operaciones vencen antes.
    """
    horizon = 1.0 if netting_set_maturity is None else min(1.0, netting_set_maturity)
    eff = effective_ee(ee)
    epe = time_average(times, ee, horizon)
    eepe = time_average(times, eff, horizon)
    return RegulatoryMetrics(horizon=horizon, epe=epe, eepe=eepe, ead=alpha * eepe)