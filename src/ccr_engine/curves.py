"""Curvas de tipos y factores de descuento."""

import numpy as np


def discount_factor(rate, t):
    """Factor de descuento P(0, t) = exp(-r * t), capitalización continua.

    Acepta números o arrays de numpy (se calcula elemento a elemento).
    """
    rate = np.asarray(rate, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("El plazo t no puede ser negativo")
    return np.exp(-rate * t)