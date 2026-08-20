import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def simulate_gbm_paths_ai(s0: float, mu: float, vol_curve: np.ndarray, T: float, num_steps: int,
                          num_paths: int) -> np.ndarray:
    """
    Generates standard FX paths using Geometric Brownian Motion.
    Streams a time-dependent, AI-forecasted volatility array across each step.
    """
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    paths = np.zeros((num_paths, num_steps + 1))

    for i in prange(num_paths):
        paths[i, 0] = s0

    for t in range(1, num_steps + 1):
        shocks = np.random.normal(0.0, 1.0, num_paths)
        sigma_t = vol_curve[t - 1]  # Extract specific volatility step
        drift = (mu - 0.5 * sigma_t ** 2) * dt

        for i in prange(num_paths):
            diffusion = sigma_t * sqrt_dt * shocks[i]
            paths[i, t] = paths[i, t - 1] * np.exp(drift + diffusion)

    return paths


@njit(parallel=True, fastmath=True)
def simulate_3factor_hjm_paths_ai(r0: float, vol_shift_curve: np.ndarray, vol_twist: float, vol_bow: float,
                                  mean_reversion: float, T: float, num_steps: int, num_paths: int) -> np.ndarray:
    """
    Generates interest rate forward curves under a 3-Factor HJM structure.
    Streams an AI-calibrated volatility curve directly into the primary Level Shift factor.
    """
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    short_rates = np.zeros((num_paths, num_steps + 1))

    for i in prange(num_paths):
        short_rates[i, 0] = r0

    for t in range(1, num_steps + 1):
        z1 = np.random.normal(0.0, 1.0, num_paths)
        z2 = np.random.normal(0.0, 1.0, num_paths)
        z3 = np.random.normal(0.0, 1.0, num_paths)
        t_tau = t / num_steps

        sigma_shift_t = vol_shift_curve[t - 1]  # Load specific vector element

        for i in prange(num_paths):
            drift = mean_reversion * (0.04 - short_rates[i, t - 1]) * dt
            shock_shift = sigma_shift_t * sqrt_dt * z1[i]
            shock_twist = vol_twist * t_tau * sqrt_dt * z2[i]
            shock_bow = vol_bow * np.sin(np.pi * t_tau) * sqrt_dt * z3[i]

            short_rates[i, t] = short_rates[i, t - 1] + drift + shock_shift + shock_twist + shock_bow

    return short_rates
