import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def reduce_matrix_to_expected_exposure(exposures: np.ndarray) -> np.ndarray:
    """
    Collapses the 2D simulation grid vertically down Axis 0.
    Enforces a strict Day 0 boundary constraint: Expected Exposure at t=0 is exactly 0.0.
    """
    num_paths, num_steps = exposures.shape
    ee_profile = np.zeros(num_steps)

    for t in range(num_steps):
        total_exposure_at_step = 0.0
        for i in prange(num_paths):
            total_exposure_at_step += exposures[i, t]
        ee_profile[t] = total_exposure_at_step / num_paths

    # CRITICAL QUANT MANDATE: Day 0 initial credit exposure must be exactly zero
    ee_profile[0] = 0.0
    return ee_profile


@njit(fastmath=True)
def aggregate_lump_sum_cva(ee_profile: np.ndarray, recovery_rate: float, annualized_pd: float) -> float:
    """
    Performs a high-speed dot-product summation using Jon Gregory's Chapter 14 rules.
    Deducts the final CVA premium fee as an upfront lump sum charge from the day-one cash exchange.
    """
    num_steps = len(ee_profile)
    loss_given_default = 1.0 - recovery_rate
    pd_per_step = annualized_pd / num_steps

    time_step_summation = 0.0
    for t in range(num_steps):
        time_step_summation += ee_profile[t] * pd_per_step

    return loss_given_default * time_step_summation
