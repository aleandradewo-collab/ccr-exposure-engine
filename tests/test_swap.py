import numpy as np
import pytest

from ccr_engine.curves import YieldCurve
from ccr_engine.instruments import InterestRateSwap


@pytest.fixture
def curve():
    return YieldCurve([0.5, 1, 2, 5, 10, 30], [0.020, 0.021, 0.022, 0.025, 0.028, 0.030])


def test_ejemplo_de_teoria():
    # P(0,1) = 0,97 y P(0,2) = 0,94 -> K_par = 0,06 / 1,91 ≈ 3,14 %
    t = np.array([1.0, 2.0])
    c = YieldCurve(t, -np.log([0.97, 0.94]) / t)
    swap = InterestRateSwap(notional=1, fixed_rate=0.03, maturity=2)
    assert swap.par_rate(c) == pytest.approx(0.06 / 1.91)


def test_swap_a_tipo_par_vale_cero(curve):
    base = InterestRateSwap(notional=10_000_000, fixed_rate=0.0, maturity=5, pay_frequency=2)
    k = base.par_rate(curve)
    swap = InterestRateSwap(notional=10_000_000, fixed_rate=k, maturity=5, pay_frequency=2)
    assert swap.npv(curve) == pytest.approx(0.0, abs=1e-6)


def test_pata_flotante_telescopica(curve):
    # La suma de forwards descontados se simplifica a N * (P(0,T0) - P(0,Tn)).
    swap = InterestRateSwap(notional=1_000_000, fixed_rate=0.03, maturity=7, pay_frequency=4)
    esperado = 1_000_000 * (curve.discount(0.0) - curve.discount(7.0))
    assert swap.floating_leg_pv(curve) == pytest.approx(esperado)


def test_payer_y_receiver_son_opuestos(curve):
    payer = InterestRateSwap(notional=1e6, fixed_rate=0.03, maturity=10, payer=True)
    receiver = InterestRateSwap(notional=1e6, fixed_rate=0.03, maturity=10, payer=False)
    assert payer.npv(curve) == pytest.approx(-receiver.npv(curve))


def test_fijo_por_debajo_del_par_favorece_al_payer(curve):
    swap = InterestRateSwap(notional=1e6, fixed_rate=0.01, maturity=5)
    assert swap.npv(curve) > 0


def test_payer_gana_si_suben_tipos(curve):
    payer = InterestRateSwap(notional=1e6, fixed_rate=0.025, maturity=5)
    receiver = InterestRateSwap(notional=1e6, fixed_rate=0.025, maturity=5, payer=False)
    assert payer.dv01(curve) > 0
    assert receiver.dv01(curve) < 0


def test_dv01_crece_con_el_plazo(curve):
    corto = InterestRateSwap(notional=1e6, fixed_rate=0.025, maturity=2)
    largo = InterestRateSwap(notional=1e6, fixed_rate=0.025, maturity=10)
    assert largo.dv01(curve) > corto.dv01(curve)


def test_curva_plana_par_rate():
    flat = YieldCurve([1, 10], [0.03, 0.03])
    swap = InterestRateSwap(notional=1, fixed_rate=0.0, maturity=5)
    assert swap.par_rate(flat) == pytest.approx(np.exp(0.03) - 1)


def test_swap_forward_starting(curve):
    swap = InterestRateSwap(notional=1, fixed_rate=0.03, maturity=5, start=2)
    assert swap.payment_times() == pytest.approx([3, 4, 5])
    esperado = curve.discount(2.0) - curve.discount(5.0)
    assert swap.floating_leg_pv(curve) == pytest.approx(esperado)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"notional": -1, "fixed_rate": 0.03, "maturity": 5},
        {"notional": 1, "fixed_rate": 0.03, "maturity": 5, "pay_frequency": 3},
        {"notional": 1, "fixed_rate": 0.03, "maturity": 2.3},
        {"notional": 1, "fixed_rate": 0.03, "maturity": 2, "start": 3},
    ],
)
def test_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        InterestRateSwap(**kwargs)