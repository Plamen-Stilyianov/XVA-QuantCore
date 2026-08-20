import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def value_vanilla_swap_portfolio(rate_paths: np.ndarray, notional: float, strike_rate: float) -> np.ndarray:
    """
    Evaluates cell-by-cell Mark-to-Market exposures for an Interest Rate Swap portfolio.
    Applies a strict time-to-maturity amortization scaling factor: (T - t).
    Forces the profile curve to roll back down to exactly zero at maturity (Gregory Ch. 7).
    """
    num_paths, num_steps = rate_paths.shape
    exposures = np.zeros((num_paths, num_steps))

    # CRITICAL QUANT MANDATE CORRECTION: Start loop at t=1.
    # This leaves column index 0 completely unmutated as pure 0.0 par constraints.
    for t in prange(1, num_steps):
        time_to_maturity = (num_steps - t) / num_steps

        for i in prange(num_paths):
            mtm_valuation = (rate_paths[i, t] - strike_rate) * notional * time_to_maturity
            exposures[i, t] = mtm_valuation if mtm_valuation > 0.0 else 0.0

    # Force absolute terminal maturity constraints
    exposures[:, -1] = 0.0
    return exposures


@njit(parallel=True, fastmath=True)
def value_fx_forward_portfolio(fx_paths: np.ndarray, notional: float, strike_fx: float) -> np.ndarray:
    """
    Evaluates exposures for a Foreign Exchange Forward contract portfolio.
    Forces strict initial Day 0 par constraints. Exposure diffuses outward from zero.
    """
    num_paths, num_steps = fx_paths.shape
    exposures = np.zeros((num_paths, num_steps))

    # Day 0 is struck at par value (Zero initial credit exposure profile)
    exposures[:, 0] = 0.0

    for t in range(1, num_steps):
        for i in prange(num_paths):
            mtm_valuation = (fx_paths[i, t] - strike_fx) * notional
            exposures[i, t] = mtm_valuation if mtm_valuation > 0.0 else 0.0

    return exposures


@njit(parallel=True, fastmath=True)
def value_credit_merton_portfolio(asset_paths: np.ndarray, debt_barrier: float, default_payout: float) -> np.ndarray:
    """
    Evaluates exposures under a structural Merton Jump-Diffusion Credit Default Swap framework.
    Triggers default payouts if counterparty asset paths breach the debt boundary barrier.

    Enforces that exposure starts at exactly 0.0 on Day 0 before stochastic boundary crossing begins.
    """
    num_paths, num_steps = asset_paths.shape
    exposures = np.zeros((num_paths, num_steps))

    # Enforce clear Day 0 boundary constraints
    exposures[:, 0] = 0.0

    for i in prange(num_paths):
        default_triggered = False
        for t in range(1, num_steps):
            if default_triggered:
                exposures[i, t] = default_payout
                continue

            if asset_paths[i, t] < debt_barrier:
                default_triggered = True
                exposures[i, t] = default_payout
            else:
                exposures[i, t] = 0.0

    return exposures
