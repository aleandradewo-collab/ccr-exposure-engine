import numpy as np
import pytest

from ccr_engine.collateral import CSA, collateralized_values
from ccr_engine.curves import YieldCurve
from ccr_engine.exposure import exposure_profile, index_of, simulation_grid, swap_values
from ccr_engine.hull_white import HullWhite1F
from ccr_engine.instruments import InterestRateSwap

# ---------- Reglas del CSA ----------


def test_threshold_cero_colateral_igual_al_valor():
    v = np.array([-300.0, 0.0, 500.0])
    assert CSA().collateral(v) == pytest.approx(v)


def test_threshold_y_mta():
    csa = CSA(threshold=100, mta=20)
    v = np.array([-500.0, -110.0, 50.0, 120.0, 400.0])
    # Umbral efectivo 120: por debajo no se mueve colateral, por encima solo el exceso.
    assert csa.collateral(v) == pytest.approx([-380, 0, 0, 0, 280])


def test_independent_amount_se_suma():
    csa = CSA(independent_amount=50)
    assert csa.collateral(np.array([0.0, 100.0])) == pytest.approx([50, 150])


def test_mpor_en_anios():
    assert CSA(mpor_days=10).mpor == pytest.approx(0.04)


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        CSA(threshold=-1)


def test_rejilla_incluye_fechas_retrasadas():
    swap = InterestRateSwap(1, 0.03, 2, 1)
    grid = simulation_grid([1.0, 1.5], [swap], mpor=0.04)
    for t in [0.96, 1.0, 1.46, 1.5]:
        assert np.any(np.isclose(grid, t))


# ---------- Exposición colateralizada ----------


@pytest.fixture
def setup():
    curve = YieldCurve([0.5, 1, 2, 5, 10, 30], [0.029, 0.030, 0.031, 0.0315, 0.034, 0.036])
    model = HullWhite1F(curve, a=0.03, sigma=0.01)
    k = InterestRateSwap(1e7, 0.0, 5, 2).par_rate(curve)
    swap = InterestRateSwap(1e7, k, 5, 2, payer=True)
    exposure_times = np.linspace(0, 5, 61)
    return model, swap, exposure_times


def _ee(setup, csa, n=4_000):
    # Misma rejilla y misma semilla en todos los casos (con las fechas retrasadas de 10 y
    # 40 días): así todos los CSA se comparan sobre exactamente los mismos escenarios.
    model, swap, exposure_times = setup
    times = np.union1d(simulation_grid(exposure_times, [swap], mpor=10 / 250),
                       simulation_grid(exposure_times, [swap], mpor=40 / 250))
    x = model.simulate_x(times, n, seed=3)
    v = swap_values(swap, model, times, x)
    if csa is None:  # sin colateral: la exposición de siempre, max(V, 0)
        idx = [index_of(times, t) for t in exposure_times]
        return exposure_profile(exposure_times, v[idx]).ee
    return exposure_profile(exposure_times, collateralized_values(v, times, exposure_times, csa)).ee


def test_mpor_cero_sin_threshold_elimina_la_exposicion(setup):
    assert _ee(setup, CSA(mpor_days=0)) == pytest.approx(np.zeros(61), abs=1e-6)


def test_threshold_enorme_equivale_a_sin_colateral(setup):
    sin = _ee(setup, None)
    con = _ee(setup, CSA(threshold=1e12))
    assert con == pytest.approx(sin)


def test_colateral_reduce_mucho_la_exposicion(setup):
    sin = _ee(setup, None)
    con = _ee(setup, CSA(mpor_days=10))
    assert con.max() < 0.4 * sin.max()


def test_menos_threshold_menos_exposicion(setup):
    picos = [_ee(setup, CSA(threshold=h)).max() for h in [1e6, 2e5, 0.0]]
    assert picos[0] > picos[1] > picos[2]


def test_exposicion_escala_con_raiz_del_mpor(setup):
    # Con threshold 0, la exposición viene del movimiento durante el MPoR: ∝ σ·√MPoR.
    ee10 = _ee(setup, CSA(mpor_days=10))[12:48].mean()
    ee40 = _ee(setup, CSA(mpor_days=40))[12:48].mean()
    assert ee40 / ee10 == pytest.approx(2.0, rel=0.1)


def test_margen_inicial_reduce_la_exposicion(setup):
    sin_im = _ee(setup, CSA(mpor_days=10))
    con_im = _ee(setup, CSA(mpor_days=10, independent_amount=100_000))
    assert con_im.max() < sin_im.max()