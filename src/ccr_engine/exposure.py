"""Valoración de operaciones sobre escenarios simulados y métricas de exposición."""

from dataclasses import dataclass

import numpy as np


def simulation_grid(exposure_times, swaps):
    """Rejilla de simulación: fechas de exposición más todas las fechas de fijación.

    Hace falta simular x(t) en cada fecha en la que se fija un cupón flotante,
    porque el valor del swap en fechas posteriores depende de ese tipo ya fijado.
    """
    fixing_dates = [np.concatenate([[s.start], s.payment_times()[:-1]]) for s in swaps]
    grid = np.concatenate([[0.0], np.asarray(exposure_times, dtype=float), *fixing_dates])
    return np.unique(np.round(grid, 10))


def _index_of(times, t):
    """Posición de la fecha t en la rejilla (error si no está)."""
    i = np.searchsorted(times, t - 1e-9)
    if i >= len(times) or not np.isclose(times[i], t):
        raise ValueError(f"La fecha {t} no está en la rejilla de simulación")
    return i


def swap_values(swap, model, times, x):
    """Valor del swap en cada fecha de la rejilla y en cada escenario.

    Devuelve un array (len(times), n_escenarios). En una fecha t:
      - Los pagos con fecha <= t ya se han hecho y no cuentan.
      - Si t está dentro de un periodo, su cupón flotante ya se fijó al inicio del
        periodo T_(i-1), con el tipo del mercado de ese escenario en ese momento.
    """
    times = np.asarray(times, dtype=float)
    pay = swap.payment_times()
    tau = swap.accrual_fractions()
    starts = np.concatenate([[swap.start], pay[:-1]])
    values = np.zeros_like(x)

    for k, t in enumerate(times):
        if t >= swap.maturity:
            continue  # el swap ya ha vencido: vale 0
        x_t = x[k]
        alive = pay > t

        # Pata fija: cupones pendientes descontados con la curva del escenario en t.
        fixed = swap.notional * swap.fixed_rate * sum(
            tau_j * model.zero_bond(t, T_j, x_t) for tau_j, T_j in zip(tau[alive], pay[alive])
        )

        # Pata flotante.
        if t < swap.start:
            # Swap aún no empezado: todos los cupones por fijar (fórmula telescópica).
            floating = swap.notional * (
                model.zero_bond(t, swap.start, x_t) - model.zero_bond(t, swap.maturity, x_t)
            )
        else:
            i = np.argmax(alive)  # primer pago pendiente
            fixing_date, next_pay = starts[i], pay[i]
            # El cupón en curso se fijó en fixing_date: 1/P(T_(i-1), T_i) en ese escenario.
            x_fix = x[_index_of(times, fixing_date)]
            fixed_coupon_factor = 1 / model.zero_bond(fixing_date, next_pay, x_fix)
            floating = swap.notional * (
                model.zero_bond(t, next_pay, x_t) * fixed_coupon_factor
                - model.zero_bond(t, swap.maturity, x_t)
            )

        value = floating - fixed
        values[k] = value if swap.payer else -value
    return values


def netting_set_values(swaps, model, times, x):
    """Valor neto de un netting set: suma de los valores de todas sus operaciones.

    Con un acuerdo de netting, si la contraparte quiebra se compensan todas las
    operaciones entre sí, así que la exposición es max(suma de V, 0) y no la suma
    de max(V, 0) de cada operación.
    """
    return sum(swap_values(s, model, times, x) for s in swaps)


@dataclass
class ExposureProfile:
    """Perfil de exposición en cada fecha de la rejilla."""

    times: np.ndarray
    expected_value: np.ndarray  # E[V(t)]
    ee: np.ndarray  # Expected Exposure: E[max(V(t), 0)]
    pfe: np.ndarray  # Potential Future Exposure: percentil de max(V(t), 0)


def exposure_profile(times, values, pfe_quantile=0.975):
    """Calcula E[V], EE y PFE a partir de la matriz de valores (fechas × escenarios)."""
    exposure = np.maximum(values, 0.0)
    return ExposureProfile(
        times=np.asarray(times),
        expected_value=values.mean(axis=1),
        ee=exposure.mean(axis=1),
        pfe=np.quantile(exposure, pfe_quantile, axis=1),
    )