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


class YieldCurve:
    """Curva cupón cero construida a partir de unos plazos (pilares) y sus tipos.

    Los tipos son cupón cero, en decimal y en capitalización continua.
    Entre pilares se interpola linealmente r(t)·t, que equivale a suponer
    un tipo forward instantáneo constante en cada tramo.
    """

    def __init__(self, tenors, zero_rates):
        tenors = np.asarray(tenors, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)

        if tenors.ndim != 1 or tenors.shape != zero_rates.shape:
            raise ValueError("tenors y zero_rates deben ser vectores de igual longitud")
        if len(tenors) < 2:
            raise ValueError("Hacen falta al menos dos pilares")
        if np.any(tenors <= 0):
            raise ValueError("Los plazos deben ser positivos")
        if np.any(np.diff(tenors) <= 0):
            raise ValueError("Los plazos deben estar ordenados de menor a mayor y sin repetir")

        self.tenors = tenors
        self.zero_rates = zero_rates
        # Puntos del interpolador: añadimos (0, 0) porque r·t = 0 en t = 0.
        self._t_nodes = np.concatenate([[0.0], tenors])
        self._rt_nodes = np.concatenate([[0.0], tenors * zero_rates])
        # Forward del último tramo, para extrapolar más allá del último pilar.
        self._last_forward = (self._rt_nodes[-1] - self._rt_nodes[-2]) / (
            self._t_nodes[-1] - self._t_nodes[-2]
        )

    def _rt(self, t):
        """Devuelve r(t)·t, es decir, -log P(0, t)."""
        t = np.asarray(t, dtype=float)
        if np.any(t < 0):
            raise ValueError("El plazo t no puede ser negativo")
        rt = np.interp(t, self._t_nodes, self._rt_nodes)
        beyond = t > self._t_nodes[-1]
        return np.where(
            beyond, self._rt_nodes[-1] + self._last_forward * (t - self._t_nodes[-1]), rt
        )

    def discount(self, t):
        """Factor de descuento P(0, t)."""
        return np.exp(-self._rt(t))

    def zero_rate(self, t):
        """Tipo cupón cero r(t) en capitalización continua."""
        t = np.asarray(t, dtype=float)
        safe_t = np.where(t > 0, t, self.tenors[0])
        return self._rt(safe_t) / safe_t

    def forward_rate(self, t1, t2):
        """Tipo forward simple entre t1 y t2: (P(0,t1)/P(0,t2) - 1) / (t2 - t1)."""
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        if np.any(t2 <= t1):
            raise ValueError("t2 debe ser mayor que t1")
        return (self.discount(t1) / self.discount(t2) - 1.0) / (t2 - t1)

    def shifted(self, bp):
        """Nueva curva con todos los tipos cupón cero desplazados bp puntos básicos."""
        return YieldCurve(self.tenors, self.zero_rates + bp / 10_000)