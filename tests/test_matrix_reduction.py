import numpy as np
from risk_engine.matrix_reduction import reduce_matrix_to_expected_exposure, aggregate_lump_sum_cva


def test_cross_sectional_axis_0_reduction():
    """Verifies that the reduction module collapses a 2D matrix vertically down Axis 0."""
    num_paths, num_steps = 1000, 10
    mock_exposures = np.random.uniform(10000.0, 50000.0, (num_paths, num_steps))
    mock_exposures[:, 0] = 0.0

    ee_profile = reduce_matrix_to_expected_exposure(mock_exposures)

    assert len(ee_profile) == num_steps, "Reduction layer distorted timeline step shapes."
    assert ee_profile[0] == 0.0, "Cross-sectional array reduction failed to enforce the global Day 0 Par constraint."


def test_cva_lump_sum_aggregation():
    """Verifies CVA dot-product premium fee calculation execution logic (Gregory Ch. 14)."""
    ee_profile = np.full(360, 50000.0)
    final_cVA = aggregate_lump_sum_cva(ee_profile, recovery_rate=0.40, annualized_pd=0.02)

    assert final_cVA > 0.0, "CVA integration returned an invalid zero charge state."
