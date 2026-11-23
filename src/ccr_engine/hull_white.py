"""Modelo de tipos de interés Hull-White de un factor."""

import numpy as np


class HullWhite1F:
    """Modelo Hull-White 1F ajustado a una curva inicial.

    El tipo corto se descompone como r(t) = x(t) + phi(t), donde:
      - x(t) es un proceso Ornstein-Uhlenbeck: dx = -a·x·dt + sigma·dW, con x(0) = 0.
      - phi(t) es una función determinista que hace que el modelo reproduzca
        exactamente la curva de hoy. Nunca hace falta calcularla de forma explícita:
        va metida dentro de la fórmula de los bonos cupón cero.
    """

    def __init__(self, curve, a, sigma):
        if a <= 0:
            raise ValueError("La reversión a la media a debe ser positiva")
        if sigma <= 0:
            raise ValueError("La volatilidad sigma debe ser positiva")
        self.curve = curve
        self.a = a
        self.sigma = sigma

    # ---------- Simulación ----------

    def simulate_x(self, times, n_paths, seed=None):
        """Simula x(t) en las fechas pedidas con la solución exacta del proceso OU.

        Devuelve un array de forma (len(times), n_paths): fila = fecha, columna = escenario.
        """
        times = np.asarray(times, dtype=float)
        if times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times debe empezar en 0 y ser estrictamente creciente")

        rng = np.random.default_rng(seed)
        x = np.zeros((len(times), n_paths))
        for i, dt in enumerate(np.diff(times)):
            decay = np.exp(-self.a * dt)
            std = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))
            x[i + 1] = x[i] * decay + std * rng.standard_normal(n_paths)
        return x

    def x_variance(self, t):
        """Varianza teórica de x(t): sigma² (1 - e^{-2at}) / (2a)."""
        return self.sigma**2 * (1 - np.exp(-2 * self.a * np.asarray(t))) / (2 * self.a)

    # ---------- Bonos cupón cero ----------

    def _B(self, t, T):
        """B(t,T) = (1 - e^{-a(T-t)}) / a: sensibilidad del bono a x(t)."""
        return (1 - np.exp(-self.a * (T - t))) / self.a

    def _V(self, t, T):
        """Varianza de la integral de x entre t y T (término de convexidad)."""
        a, s, tau = self.a, self.sigma, T - t
        return (s**2 / a**2) * (
            tau + 2 / a * np.exp(-a * tau) - 1 / (2 * a) * np.exp(-2 * a * tau) - 3 / (2 * a)
        )

    def zero_bond(self, t, T, x_t):
        """Precio en t del bono cupón cero que paga 1 en T, dado x(t).

        P(t,T) = P(0,T)/P(0,t) · exp(0.5·[V(t,T) - V(0,T) + V(0,t)] - B(t,T)·x(t))
        x_t puede ser un array con todos los escenarios: devuelve un precio por escenario.
        """
        if T < t:
            raise ValueError("T debe ser mayor o igual que t")
        p0T = self.curve.discount(T)
        p0t = self.curve.discount(t)
        convexity = 0.5 * (self._V(t, T) - self._V(0, T) + self._V(0, t))
        return p0T / p0t * np.exp(convexity - self._B(t, T) * np.asarray(x_t))

    def short_rate(self, t, x_t):
        """Tipo corto r(t) = x(t) + phi(t), con phi(t) = f(0,t) + sigma²/(2a²)·(1-e^{-at})²."""
        f0t = self.curve.instantaneous_forward(t)
        phi = f0t + self.sigma**2 / (2 * self.a**2) * (1 - np.exp(-self.a * t)) ** 2
        return np.asarray(x_t) + phi

    def stochastic_discount(self, times, x):
        """Factor de descuento estocástico D(0,t) = exp(-∫ r) por escenario (trapecios).

        Necesita una rejilla de fechas fina para ser preciso.
        """
        r = np.stack([self.short_rate(t, x[i]) for i, t in enumerate(times)])
        dt = np.diff(times)[:, None]
        integral = np.concatenate(
            [np.zeros((1, x.shape[1])), np.cumsum(0.5 * (r[1:] + r[:-1]) * dt, axis=0)]
        )
        return np.exp(-integral)