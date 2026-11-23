import numpy as np
import pandas as pd
import pytest

from ccr_engine.curves import YieldCurve
from ccr_engine.ecb import curve_from_dataframe, parse_tenor


@pytest.fixture
def curve():
    # Curva inventada con pendiente positiva, parecida a la del euro.
    return YieldCurve([0.5, 1, 2, 5, 10], [0.020, 0.021, 0.022, 0.025, 0.028])


def test_reproduce_los_pilares(curve):
    assert curve.zero_rate(curve.tenors) == pytest.approx(curve.zero_rates)


def test_descuento_en_cero_vale_uno(curve):
    assert curve.discount(0.0) == pytest.approx(1.0)


def test_descuento_decreciente(curve):
    t = np.linspace(0, 30, 301)
    assert np.all(np.diff(curve.discount(t)) < 0)


def test_interpolacion_entre_pilares(curve):
    r = curve.zero_rate(3.0)
    assert 0.022 < r < 0.025


def test_forward_coherente_con_descuentos(curve):
    # Invertir de 0 a 5 debe dar lo mismo que de 0 a 2 y reinvertir al forward 2-5.
    f = curve.forward_rate(2.0, 5.0)
    assert curve.discount(2.0) / (1 + f * 3.0) == pytest.approx(curve.discount(5.0))


def test_curva_plana_forward_constante():
    flat = YieldCurve([1, 5, 10], [0.03, 0.03, 0.03])
    # Con capitalización continua plana, el forward simple anual es e^0.03 - 1.
    assert flat.forward_rate(4.0, 5.0) == pytest.approx(np.exp(0.03) - 1)


def test_extrapola_mas_alla_del_ultimo_pilar(curve):
    assert curve.discount(40.0) < curve.discount(10.0)


def test_plazos_desordenados_fallan():
    with pytest.raises(ValueError):
        YieldCurve([2, 1], [0.02, 0.02])


@pytest.mark.parametrize(
    "texto, anios", [("3M", 0.25), ("6M", 0.5), ("1Y", 1.0), ("1Y6M", 1.5), ("30Y", 30.0)]
)
def test_parse_tenor(texto, anios):
    assert parse_tenor(texto) == pytest.approx(anios)


def test_parse_tenor_invalido():
    with pytest.raises(ValueError):
        parse_tenor("diez años")


def test_curve_from_dataframe_usa_la_ultima_fecha():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-11-18"] * 2 + ["2026-11-19"] * 2),
            "tenor": ["1Y", "5Y", "1Y", "5Y"],
            "years": [1.0, 5.0, 1.0, 5.0],
            "rate": [0.020, 0.025, 0.021, 0.026],
        }
    )
    c = curve_from_dataframe(df)
    assert c.zero_rate(1.0) == pytest.approx(0.021)