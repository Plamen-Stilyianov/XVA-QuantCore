import numpy as np
from models.instrument_valuations import value_vanilla_swap_portfolio, value_fx_forward_portfolio, \
    value_credit_merton_portfolio


def test_swap_amortisation_roll_off():
    """Verifies Interest Rate Swap cash flow amortization decay constraints (Gregory Ch. 7)."""
    num_paths, num_steps = 100, 360
    rate_paths = np.full((num_paths, num_steps), 0.05)
    exposures = value_vanilla_swap_portfolio(rate_paths, notional=10000000.0, strike_rate=0.04)

    assert np.all(exposures[:, 0] == 0.0), "Initial condition broken: Swap exposure on Day 0 must be exactly zero."
    assert np.all(
        exposures[:, -1] == 0.0), "Quantitative validation failed: Swap exposure must roll back to zero at maturity."


def test_fx_forward_par_initialization():
    """Verifies FX Forward Day 0 initial par boundary constraints."""
    num_paths, num_steps = 100, 360
    fx_paths = np.full((num_paths, num_steps), 1.30)
    exposures = value_fx_forward_portfolio(fx_paths, notional=10000000.0, strike_fx=1.25)

    assert np.all(
        exposures[:, 0] == 0.0), "Initial condition broken: FX Forward exposure on Day 0 must be exactly zero."


def test_merton_credit_default_barrier():
    """Verifies binary boundary triggers inside your Merton structural default script."""
    num_paths, num_steps = 10, 100
    asset_paths = np.full((num_paths, num_steps), 65.0)  # Asset values initialized below debt barrier
    default_payout = 50000.0

    exposures = value_credit_merton_portfolio(asset_paths, debt_barrier=70.0, default_payout=default_payout)

    # 1. Enforce Day 0 Par Constraint (Column 0 must be exactly zero)
    assert np.all(exposures[:, 0] == 0.0), "Day 0 barrier trigger constraint violated."

    # 2. Enforce Default Trigger Payouts for the remaining steps (Columns 1 to end)
    assert np.all(
        exposures[:, 1:] == default_payout), "Merton model failed to trigger default payout at breached threshold node."
