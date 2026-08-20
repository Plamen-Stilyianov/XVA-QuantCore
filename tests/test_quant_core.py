import os
import sys
import numpy as np
import pytest

# Ensure the root project path is pushed into the active Python tracking path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.machine_learning import forecast_vol_with_lstm, forecast_vol_with_svm, generate_probabilistic_synthetic_vol
from models.stochastic_calculus import simulate_gbm_paths_ai, simulate_3factor_hjm_paths_ai
from risk_engine.matrix_reduction import reduce_matrix_to_expected_exposure, aggregate_lump_sum_cva
from models.instrument_valuations import value_vanilla_swap_portfolio, value_fx_forward_portfolio, value_credit_merton_portfolio


# =====================================================================
# MODULE 1: TESTING THE AI DECOUPLED MAPPING LAYERS
# =====================================================================
def test_lstm_volatility_forecasting_bounds():
    """
    Verifies that the Numba-compiled LSTM forward pass handles memory layers
    and respects the hard volatility floor constraint of 0.20% (0.002).
    """
    num_steps = 360
    base_vol = 0.025

    vol_curve = forecast_vol_with_lstm(num_steps, base_vol)

    assert isinstance(vol_curve, np.ndarray), "LSTM output must be a standard NumPy array."
    assert len(vol_curve) == num_steps, "LSTM array length must strictly match time steps."
    assert np.all(vol_curve >= 0.002), "LSTM output violated the structural minimum volatility boundary floor."


def test_svm_rbf_kernel_volatility_generation():
    """
    Verifies that the SVM Support Vector Regressor generates non-linear RBF transformations
    and bounds the curve arrays within stable tracking boundaries.
    """
    num_steps = 100
    base_vol = 0.015

    vol_curve = forecast_vol_with_svm(num_steps, base_vol)

    assert len(vol_curve) == num_steps
    assert np.all(vol_curve > 0.0), "SVM volatility must strictly be an absolute non-negative domain."


def test_markov_switching_synthetic_variance():
    """
    Verifies that the Generative Probability Markov model alternates between regimes
    probabilistically and does not output flat or uncalibrated arrays.
    """
    num_steps = 360
    base_vol = 0.025

    vol_curve = generate_probabilistic_synthetic_vol(num_steps, base_vol)

    assert len(vol_curve) == num_steps
    # Ensure stochastic properties exist by checking standard deviation is non-zero
    assert np.std(vol_curve) > 0.0, "Generative switching model failed to construct stochastic regime variations."


def test_gbm_path_matrix_dimensions():
    """
    Verifies that the parallelised GBM path generator outputs a 2D matrix
    with exact dimensions matching: (num_paths, num_steps + 1).
    The extra column represents the initial Day 0 boundary value state.
    """
    num_paths = 100
    num_steps = 360
    T = 1.0
    s0 = 1.25
    mu = 0.01

    # Ingest a mocked 360-step volatility vector from our ML/AI layer
    mock_vol_curve = np.full(num_steps, 0.02)

    # Trigger the Numba JIT parallel execution loop
    paths = simulate_gbm_paths_ai(s0, mu, mock_vol_curve, T, num_steps, num_paths)

    assert isinstance(paths, np.ndarray), "GBM path output must be a standard NumPy array."
    assert paths.ndim == 2, "GBM output must be structured as a 2D matrix layout."
    assert paths.shape == (num_paths, num_steps + 1), "GBM destination grid dimensions are mismatched."
    assert np.all(paths[:, 0] == s0), "Boundary Condition Broken: Day 0 initial state must lock to s0 parameters."

# =====================================================================
# MODULE 2: TESTING QUANTITATIVE FINANCIAL BOUNDARY STATES
# =====================================================================
def test_3factor_hjm_multifactor_matrix_generation():
    """
    Verifies that the 3-Factor HJM engine simulates interest rate forward paths
    using our AI-calibrated volatility arrays while maintaining correct matrix boundaries.
    """
    num_paths = 50
    num_steps = 120
    T = 1.0
    r0 = 0.04
    vol_twist = 0.008
    vol_bow = 0.004
    mean_reversion = 0.05

    # Ingest a mocked AI shift vector representing our denoised PCA level factors
    mock_vol_shift_curve = np.full(num_steps, 0.015)

    # Trigger the 3-Factor interest rate simulation core
    rates = simulate_3factor_hjm_paths_ai(r0, mock_vol_shift_curve, vol_twist, vol_bow,
                                          mean_reversion, T, num_steps, num_paths)

    assert rates.ndim == 2
    assert rates.shape == (num_paths, num_steps + 1), "3-Factor HJM array grid dimension mapping broken."
    assert np.all(rates[:, 0] == r0), "HJM initial short rate state failed to anchor to r0 base."

    # Confirm that our stochastic calculus drift actually occurs by checking variance across time
    assert np.var(
        rates[:, -1]) > 0.0, "Stochastic diffusion failure: Final time step contains zero structural variance."


# =====================================================================
# MODULE 3: TESTING QUANTITATIVE FINANCIAL BOUNDARY STATES
# =====================================================================
def test_fixed_income_swap_amortisation_decay():
    """
    Verifies that Interest Rate Swap portfolios enforce the strict temporal boundary constraint.
    The final step must decay back to exactly zero exposure as future cash flows roll off.
    """
    num_paths = 100
    num_steps = 50
    rate_paths = np.full((num_paths, num_steps), 0.05)  # Static 5.00% rate environment
    notional = 10_000_000.0
    strike = 0.04

    exposures = value_vanilla_swap_portfolio(rate_paths, notional, strike)

    # Confirm Day 0 Par Constraint
    assert np.all(exposures[:, 0] == 0.0), "Initial condition broken: Swap exposure at t=0 must be exactly zero."
    # Confirm Amortisation Roll-Off Decay at maturity (final column slice)
    assert np.all(exposures[:, -1] == 0.0), "Quantitative validation failed: Swap exposure must roll back to zero at T."


def test_cross_sectional_matrix_axis_0_reduction():
    """
    Verifies that the parallel reduction engine collapses the 2D path matrix
    vertically down Axis 0 and cleanly enforces the global Day 0 Par constraint.
    """
    num_paths = 1000
    num_steps = 10
    # Simulate an active, volatile exposure matrix field
    mock_exposures = np.random.uniform(50000.0, 100000.0, (num_paths, num_steps))
    mock_exposures[:, 0] = 0.0  # Force par value configuration at t=0

    ee_profile = reduce_matrix_to_expected_exposure(mock_exposures)

    assert len(ee_profile) == num_steps, "Reduction engine distorted timeline dimensions."
    assert ee_profile[0] == 0.0, "Matrix reduction failed to enforce the global Day 0 Par constraint."


def test_swap_amortisation_roll_off():
    """Verifies Interest Rate Swap cash flow amortization decay constraints (Gregory Ch. 7)."""
    num_paths, num_steps = 100, 360
    rate_paths = np.full((num_paths, num_steps), 0.05)
    exposures = value_vanilla_swap_portfolio(rate_paths, notional=10000000.0, strike_rate=0.04)

    assert np.all(exposures[:, 0] == 0.0), "Swap exposure on Day 0 must be exactly zero."
    assert np.all(exposures[:, -1] == 0.0), "Swap exposure must decay to exactly zero at maturity."


def test_fx_forward_par_initialization_and_bounds():
    """
    Verifies that the FX Forward valuation engine strictly enforces Day 0 par constraints
    and calculates non-negative Credit Mark-to-Market option floor exposures (Ruiz Ch. 3).
    """
    num_paths = 1000
    num_steps = 360
    notional = 10000000.0
    strike_fx = 1.25

    # Generate mock FX paths where half are in-the-money (1.30) and half are out-of-the-money (1.20)
    fx_paths = np.zeros((num_paths, num_steps))
    fx_paths[:500, :] = 1.30  # In-the-money rows
    fx_paths[500:, :] = 1.20  # Out-of-the-money rows

    # Trigger the Numba parallel valuation loop
    exposures = value_fx_forward_portfolio(fx_paths, notional, strike_fx)

    # 1. Test Day 0 Boundary Constraint (Struck at par value)
    assert np.all(exposures[:, 0] == 0.0), "Initial condition broken: FX Forward exposure at t=0 must be exactly zero."

    # 2. Test the non-negative Credit MTM floor rule: max(Market_Value, 0.0)
    # The out-of-the-money rows must be cleanly floored at 0.0 instead of showing negative liability debt
    assert np.all(
        exposures[500:, 1:] == 0.0), "Credit Risk Floor Rule broken: Engine permitted negative exposure values."

    # 3. Test in-the-money cell calculations
    expected_itm_value = (1.30 - 1.25) * notional
    assert np.all(exposures[:500, 1:] == expected_itm_value), "FX valuation mathematical calculation mismatch."


def test_merton_credit_default_barrier():
    """Verifies binary boundary triggers inside your Merton structural default script."""
    num_paths, num_steps = 10, 100
    asset_paths = np.full((num_paths, num_steps), 65.0)  # Asset values initialized below debt barrier
    default_payout = 50000.0

    exposures = value_credit_merton_portfolio(asset_paths, debt_barrier=70.0, default_payout=default_payout)

    # 1. Enforce Day 0 Par Constraint (Column index 0 must be exactly zero across all paths)
    assert np.all(exposures[:, 0] == 0.0), "Day 0 barrier trigger constraint violated."

    # 2. Enforce Default Trigger Payouts for the remaining steps (Columns 1 through the end)
    assert np.all(
        exposures[:, 1:] == default_payout), "Merton model failed to trigger default payout at breached threshold node."


def test_cross_sectional_axis_0_reduction():
    """Verifies that the reduction module collapses a 2D matrix vertically down Axis 0."""
    num_paths, num_steps = 1000, 10
    mock_exposures = np.random.uniform(10000.0, 50000.0, (num_paths, num_steps))
    mock_exposures[:, 0] = 0.0

    ee_profile = reduce_matrix_to_expected_exposure(mock_exposures)

    assert len(ee_profile) == num_steps, "Reduction layer distorted timeline step shapes."
    assert ee_profile[0] == 0.0, "Cross-sectional array reduction failed to enforce the global Day 0 Par constraint."


def test_aggregate_lump_sum_cva_mathematical_precision():
    """
    Verifies that the CVA dot-product loop accurately integrates Expected Exposure profiles
    against survival thresholds to compute an upfront cash premium (Jon Gregory Ch. 14).
    """
    num_steps = 360
    recovery_rate = 0.40  # LGD = 1.0 - 0.40 = 0.60
    annualized_pd = 0.02  # PD per step = 0.02 / 360 = 5.555e-5

    # Profile A: Simulated portfolio with a flat constant exposure of £100,000
    flat_exposure = 100000.0
    ee_profile_flat = np.full(num_steps, flat_exposure)

    final_cva_flat = aggregate_lump_sum_cva(ee_profile_flat, recovery_rate, annualized_pd)

    # Expected: LGD * SUM(EE * delta_PD) -> 0.60 * (360 * 100000 * (0.02 / 360)) = 0.60 * 100000 * 0.02 = 1,200
    expected_cva = 0.60 * flat_exposure * annualized_pd
    assert final_cva_flat == pytest.approx(expected_cva, rel=1e-5), "CVA mathematical dot-product output mismatch."

    # Profile B: Zero exposure profile must structurally yield an exact zero CVA haircut charge
    ee_profile_zero = np.zeros(num_steps)
    final_cva_zero = aggregate_lump_sum_cva(ee_profile_zero, recovery_rate, annualized_pd)
    assert final_cva_zero == 0.0, "CVA calculation engine failed to return a zero state for a zero-exposure profile."
