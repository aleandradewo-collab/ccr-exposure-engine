import numpy as np
import pytest

from ccr_engine.curves import YieldCurve
from ccr_engine.hull_white import HullWhite1F


@pytest.fixture
def curve():
    return YieldCurve([0.5, 1, 2, 5, 10, 30], [0.029, 0.030, 0.031, 0.0315, 0.034, 0.036])


@pytest.fixture
def model(curve):
    return HullWhite1F(curve, a=0.05, sigma=0.01)


def test_reproduce_la_curva_de_hoy(model, curve):
    # En t = 0, x = 0: la fórmula del bono debe devolver exactamente la curva.
    for T in [0.5, 1, 3, 7, 20]:
        assert model.zero_bond(0.0, T, 0.0) == pytest.approx(curve.discount(T))


def test_bono_que_vence_ya_vale_uno(model):
    x = np.array([-0.02, 0.0, 0.03])
    assert model.zero_bond(2.0, 2.0, x) == pytest.approx(np.ones(3))


def test_tipos_altos_bonos_baratos(model):
    # Más x (tipos más altos) -> bono más barato.
    p = model.zero_bond(1.0, 5.0, np.array([-0.01, 0.0, 0.01]))
    assert np.all(np.diff(p) < 0)


def test_simulacion_reproducible(model):
    t = np.linspace(0, 5, 21)
    assert np.array_equal(model.simulate_x(t, 100, seed=1), model.simulate_x(t, 100, seed=1))


def test_media_y_varianza_de_x(model):
    t = np.linspace(0, 10, 11)
    x = model.simulate_x(t, 50_000, seed=42)
    # Media teórica 0 y varianza teórica sigma²(1-e^{-2at})/(2a).
    assert np.abs(x[-1].mean()) < 4 * np.sqrt(model.x_variance(10) / 50_000)
    assert x[1:].var(axis=1) == pytest.approx(model.x_variance(t[1:]), rel=0.03)


def test_mas_reversion_menos_dispersion(curve):
    lenta = HullWhite1F(curve, a=0.01, sigma=0.01)
    rapida = HullWhite1F(curve, a=0.5, sigma=0.01)
    assert rapida.x_variance(10) < lenta.x_variance(10)


def test_martingala_descuento(model, curve):
    # Bajo Q, la media de D(0,T) en todos los escenarios es el precio de hoy P(0,T).
    t = np.linspace(0, 5, 5 * 52 + 1)  # rejilla semanal
    n = 20_000
    x = model.simulate_x(t, n, seed=7)
    d = model.stochastic_discount(t, x)
    for i in [52, 156, 260]:  # 1, 3 y 5 años
        error_estandar = d[i].std() / np.sqrt(n)
        assert abs(d[i].mean() - curve.discount(t[i])) < 4 * error_estandar


def test_martingala_bono_futuro(model, curve):
    # Comprar hoy un bono a 10 años equivale a esperar 3 años, descontar, y comprarlo entonces.
    t = np.linspace(0, 3, 3 * 52 + 1)
    n = 20_000
    x = model.simulate_x(t, n, seed=11)
    valor = model.stochastic_discount(t, x)[-1] * model.zero_bond(3.0, 10.0, x[-1])
    error_estandar = valor.std() / np.sqrt(n)
    assert abs(valor.mean() - curve.discount(10.0)) < 4 * error_estandar


@pytest.mark.parametrize("a, sigma", [(0, 0.01), (0.05, 0), (-1, 0.01)])
def test_parametros_invalidos(curve, a, sigma):
    with pytest.raises(ValueError):
        HullWhite1F(curve, a=a, sigma=sigma)