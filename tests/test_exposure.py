import numpy as np
import pytest

from ccr_engine.curves import YieldCurve
from ccr_engine.exposure import exposure_profile, simulation_grid, swap_values
from ccr_engine.hull_white import HullWhite1F
from ccr_engine.instruments import InterestRateSwap


@pytest.fixture
def curve():
    return YieldCurve([0.5, 1, 2, 5, 10, 30], [0.029, 0.030, 0.031, 0.0315, 0.034, 0.036])


@pytest.fixture
def model(curve):
    return HullWhite1F(curve, a=0.03, sigma=0.01)


def par_swap(curve, maturity=5, payer=True, freq=2, start=0.0):
    base = InterestRateSwap(notional=1e6, fixed_rate=0.0, maturity=maturity, start=start,
                            pay_frequency=freq)
    k = base.par_rate(curve)
    return InterestRateSwap(notional=1e6, fixed_rate=k, maturity=maturity, start=start,
                            pay_frequency=freq, payer=payer)


def test_rejilla_incluye_fijaciones():
    swap = InterestRateSwap(notional=1, fixed_rate=0.03, maturity=2, pay_frequency=2)
    grid = simulation_grid([0.25, 1.75], [swap])
    for t in [0.0, 0.25, 0.5, 1.0, 1.5, 1.75]:
        assert np.any(np.isclose(grid, t))


def test_valor_hoy_coincide_con_npv(model, curve):
    swap = InterestRateSwap(notional=1e6, fixed_rate=0.025, maturity=5, pay_frequency=2)
    times = simulation_grid([1.0], [swap])
    x = model.simulate_x(times, 10, seed=1)
    v = swap_values(swap, model, times, x)
    assert v[0] == pytest.approx(np.full(10, swap.npv(curve)))


def test_vale_cero_al_vencimiento(model):
    swap = InterestRateSwap(notional=1e6, fixed_rate=0.03, maturity=2, pay_frequency=2)
    times = simulation_grid([2.0, 2.5], [swap])
    x = model.simulate_x(times, 50, seed=2)
    v = swap_values(swap, model, times, x)
    assert np.all(v[times >= 2.0] == 0)


def test_falta_fecha_de_fijacion(model):
    swap = InterestRateSwap(notional=1e6, fixed_rate=0.03, maturity=2, pay_frequency=2)
    times = np.array([0.0, 0.75])  # falta la fijación de 0,5
    x = model.simulate_x(times, 5, seed=3)
    with pytest.raises(ValueError):
        swap_values(swap, model, times, x)


def _martingale_check(model, swap, t_check):
    # Bajo Q: E[D(0,t)·V(t)] = V(0) si entre 0 y t no hay pagos.
    times = np.unique(np.concatenate([np.linspace(0, t_check, 53), simulation_grid([], [swap])]))
    times = times[times <= t_check + 1e-12]
    n = 20_000
    x = model.simulate_x(times, n, seed=5)
    v = swap_values(swap, model, times, x)[-1]
    d = model.stochastic_discount(times, x)[-1]
    descontado = d * v
    error_estandar = descontado.std() / np.sqrt(n)
    assert abs(descontado.mean() - swap.npv(model.curve)) < 4 * error_estandar


def test_martingala_cupon_ya_fijado(model):
    # t = 0,4 está dentro del primer periodo: el cupón ya se fijó en 0.
    swap = InterestRateSwap(notional=1e6, fixed_rate=0.02, maturity=5, pay_frequency=2)
    _martingale_check(model, swap, 0.4)


def test_martingala_swap_no_empezado(model):
    swap = InterestRateSwap(notional=1e6, fixed_rate=0.04, maturity=6, start=2)
    _martingale_check(model, swap, 1.5)


def test_perfil_swap_a_la_par(model, curve):
    swap = par_swap(curve)
    times = simulation_grid(np.linspace(0, 5, 61), [swap])
    x = model.simulate_x(times, 5_000, seed=8)
    prof = exposure_profile(times, swap_values(swap, model, times, x))
    assert prof.ee[0] == pytest.approx(0.0, abs=1e-6)  # vale 0 al contratarlo
    assert prof.ee[-1] == 0.0  # y al vencer
    pico = np.argmax(prof.ee)
    assert 0.5 < times[pico] < 4.5  # la joroba: el máximo está en medio
    assert np.all(prof.pfe >= prof.ee - 1e-9)


def test_identidad_payer_receiver(model, curve):
    # max(V,0) - max(-V,0) = V  ->  EE_payer - EE_receiver = E[V_payer]
    payer = par_swap(curve, payer=True)
    receiver = par_swap(curve, payer=False)
    times = simulation_grid(np.linspace(0, 5, 21), [payer])
    x = model.simulate_x(times, 2_000, seed=9)
    p = exposure_profile(times, swap_values(payer, model, times, x))
    r = exposure_profile(times, swap_values(receiver, model, times, x))
    assert p.ee - r.ee == pytest.approx(p.expected_value)


def test_mas_volatilidad_mas_exposicion(curve):
    swap = par_swap(curve)
    times = simulation_grid(np.linspace(0, 5, 21), [swap])
    ee = []
    for sigma in [0.005, 0.01, 0.02]:
        m = HullWhite1F(curve, a=0.03, sigma=sigma)
        x = m.simulate_x(times, 3_000, seed=10)
        ee.append(exposure_profile(times, swap_values(swap, m, times, x)).ee.max())
    assert ee[0] < ee[1] < ee[2]