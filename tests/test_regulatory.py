import numpy as np
import pytest

from ccr_engine.curves import YieldCurve
from ccr_engine.exposure import (
    exposure_profile,
    netting_set_values,
    simulation_grid,
    swap_values,
)
from ccr_engine.hull_white import HullWhite1F
from ccr_engine.instruments import InterestRateSwap
from ccr_engine.regulatory import effective_ee, regulatory_metrics, time_average

# ---------- Métricas regulatorias con un ejemplo hecho a mano ----------

TIMES = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25])
EE = np.array([0, 4, 2, 3, 1, 10])


def test_effective_ee_no_baja():
    assert effective_ee(EE) == pytest.approx([0, 4, 4, 4, 4, 10])


def test_ejemplo_a_mano():
    m = regulatory_metrics(TIMES, EE)
    # EPE  = (4 + 2 + 3 + 1) · 0,25 / 1 = 2,5
    # EEPE = (4 + 4 + 4 + 4) · 0,25 / 1 = 4      (el 10 de t = 1,25 queda fuera del año)
    assert m.epe == pytest.approx(2.5)
    assert m.eepe == pytest.approx(4.0)
    assert m.ead == pytest.approx(1.4 * 4.0)


def test_netting_set_corto_usa_su_vencimiento():
    m = regulatory_metrics(TIMES, EE, netting_set_maturity=0.5)
    # Solo cuentan 0,25 y 0,5: EEPE = (4 + 4) · 0,25 / 0,5 = 4 ; EPE = (4 + 2) · 0,25 / 0,5 = 3
    assert m.horizon == 0.5
    assert m.epe == pytest.approx(3.0)
    assert m.eepe == pytest.approx(4.0)


def test_eepe_mayor_o_igual_que_epe():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 1, 13)
    for _ in range(20):
        ee = np.abs(rng.normal(size=13))
        m = regulatory_metrics(times, ee)
        assert m.eepe >= m.epe - 1e-12


def test_horizonte_fuera_de_la_rejilla():
    with pytest.raises(ValueError):
        time_average(np.array([0, 0.3, 0.7]), np.ones(3), 1.0)


# ---------- Netting ----------

@pytest.fixture
def model():
    curve = YieldCurve([0.5, 1, 2, 5, 10, 30], [0.029, 0.030, 0.031, 0.0315, 0.034, 0.036])
    return HullWhite1F(curve, a=0.03, sigma=0.01)


def test_operaciones_opuestas_se_cancelan(model):
    payer = InterestRateSwap(1e6, 0.03, 5, 2, payer=True)
    receiver = InterestRateSwap(1e6, 0.03, 5, 2, payer=False)
    times = simulation_grid(np.linspace(0, 5, 21), [payer, receiver])
    x = model.simulate_x(times, 1_000, seed=1)
    neto = netting_set_values([payer, receiver], model, times, x)
    assert neto == pytest.approx(np.zeros_like(neto), abs=1e-6)
    # Sin netting, en cambio, la exposición sería la de cada swap por separado.
    sin_netting = sum(exposure_profile(times, swap_values(s, model, times, x)).ee
                      for s in [payer, receiver])
    assert sin_netting.max() > 0


def test_netting_nunca_aumenta_la_exposicion(model):
    swaps = [
        InterestRateSwap(1e6, 0.030, 5, 2, payer=True),
        InterestRateSwap(2e6, 0.034, 10, 1, payer=False),
        InterestRateSwap(5e5, 0.025, 3, 4, payer=True),
    ]
    times = simulation_grid(np.linspace(0, 10, 41), swaps)
    x = model.simulate_x(times, 2_000, seed=2)
    individuales = [swap_values(s, model, times, x) for s in swaps]
    neto = netting_set_values(swaps, model, times, x)
    assert neto == pytest.approx(sum(individuales))
    # max(suma V, 0) <= suma de max(V, 0), escenario a escenario.
    assert np.all(np.maximum(neto, 0) <= sum(np.maximum(v, 0) for v in individuales) + 1e-6)