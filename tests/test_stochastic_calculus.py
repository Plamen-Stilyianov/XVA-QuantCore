import numpy as np
from models.stochastic_calculus import simulate_gbm_paths_ai, simulate_3factor_hjm_paths_ai


def test_gbm_path_matrix_dimensions():
    """Verifies parallelized GBM path generation grid output array shapes."""
    num_paths, num_steps = 100, 50
    vol_curve = np.full(num_steps, 0.02)
    paths = simulate_gbm_paths_ai(s0=1.25, mu=0.01, vol_curve=vol_curve, T=1.0, num_steps=num_steps,
                                  num_paths=num_paths)

    assert paths.shape == (num_paths, num_steps + 1), "GBM destination path layout dimensions are mismatched."


def test_3factor_hjm_assembly_compilation():
    """Verifies multi-factor HJM Interest Rate curve path matrices shape compliance."""
    num_paths, num_steps = 50, 120
    vol_curve = np.full(num_steps, 0.015)
    rates = simulate_3factor_hjm_paths_ai(r0=0.04, vol_shift_curve=vol_curve, vol_twist=0.008, vol_bow=0.004,
                                          mean_reversion=0.05, T=1.0, num_steps=num_steps, num_paths=num_paths)

    assert rates.shape == (num_paths, num_steps + 1), "3-Factor HJM grid dimension mapping broken."
