import numpy as np
import pytest

from ccr_engine.curves import discount_factor


def test_ejemplo_teoria():
    # r = 3 %, T = 2 años -> e^(-0,06) ≈ 0,9418
    assert discount_factor(0.03, 2) == pytest.approx(0.9418, abs=1e-4)


def test_plazo_cero_vale_uno():
    assert discount_factor(0.05, 0) == 1.0


def test_tipos_suben_descuento_baja():
    assert discount_factor(0.04, 5) < discount_factor(0.02, 5)


def test_vectorizado():
    t = np.array([1.0, 2.0, 5.0])
    df = discount_factor(0.03, t)
    assert df.shape == (3,)
    assert np.all(np.diff(df) < 0)  # más plazo -> menor factor


def test_plazo_negativo_falla():
    with pytest.raises(ValueError):
        discount_factor(0.03, -1)