import numpy as np
from thermal_stack_rl.physics import k_eff_parallel, k_eff_series_from_layers

def test_k_eff_parallel_basic():
    k = [100.0, 200.0]
    A = [0.5, 0.5]
    t = 1.0e-3
    ke = k_eff_parallel(k, A, t)
    assert np.isclose(ke, 150.0, rtol=1e-6)

def test_k_eff_series_basic():
    ke_layers = [100.0, 200.0]
    ts = [1.0e-3, 1.0e-3]
    ke_series = k_eff_series_from_layers(ke_layers, ts)
    assert np.isclose(ke_series, 133.3333333, rtol=1e-6)
