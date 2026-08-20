import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch, mock_open

# Force Python path configurations to find core repository roots cleanly
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))


def test_historically_grounded_macro_constants():
    """
    Verifies the hardcoded economic constants match real-world macro events
    (2008 Crash, 2011 Euro Debt Crisis, 2020 Pandemic Shock).
    """
    # Import the target constants safely
    from utils.generate_counterparty_data import years, historical_fred_spreads

    assert len(years) == 20
    assert years[0] == 2007
    assert years[-1] == 2026
    assert len(historical_fred_spreads) == 20

    # Check specific macro-indexed stress years
    assert historical_fred_spreads[1] == 5.62  # 2008 global financial crisis peak
    assert historical_fred_spreads[4] == 2.45  # 2011 Eurozone sovereign debt crisis
    assert historical_fred_spreads[13] == 2.78  # 2020 pandemic liquidity shock
    assert historical_fred_spreads[16] == 1.65  # 2023 SVB deposit flight crisis


def test_counterparty_matrix_shapes_and_features():
    """
    Verifies that all three counterparty dictionaries contain perfectly balanced arrays,
    and have correct target columns for XGBoost classifier ingestion.
    """
    from utils.generate_counterparty_data import barclays_grounded, abnamro_grounded, xyzbank_grounded

    target_columns = {
        "counterparty_id", "company_name", "year", "tier1_capital_ratio",
        "leverage_ratio", "liquidity_coverage_ratio", "fred_credit_spread", "default_label"
    }

    datasets = [barclays_grounded, abnamro_grounded, xyzbank_grounded]

    for ds in datasets:
        assert set(ds.keys()) == target_columns
        for feature, data_vector in ds.items():
            assert len(data_vector) == 20, f"Column '{feature}' failed row balance check."


def test_xyz_bank_default_labels_align_with_crises():
    """
    Verifies that the distressed bank model proxy (XYZBANK) triggers default labels (1)
    at the exact historical moments macro economic stress spikes.
    """
    from utils.generate_counterparty_data import xyzbank_grounded

    labels = xyzbank_grounded["default_label"]
    years = xyzbank_grounded["year"]

    year_to_label = dict(zip(years, labels))

    # Systemic macro crisis years must trigger high default classifications (1)
    assert year_to_label[2008] == 1, "2008 Lehman collapse must trigger default flag."
    assert year_to_label[2011] == 1, "2011 Euro debt crisis must trigger default flag."
    assert year_to_label[2020] == 1, "2020 Pandemic market crash must trigger default flag."
    assert year_to_label[2026] == 1, "2026 Live risk run row must trigger default flag."

    # Base calm periods should remain safe (0)
    assert year_to_label[2007] == 0
    assert year_to_label[2015] == 0


@patch("utils.generate_counterparty_data.os.makedirs")
def test_script_execution_flushes_to_disk_cleanly(mock_makedirs):
    """
    Ensures that running the main module successfully triggers 3 file open writes
    into the correct data subfolder path without modifying physical storage.
    """
    # Clear module from cache if it was already imported by previous tests
    if "utils.generate_counterparty_data" in sys.modules:
        del sys.modules["utils.generate_counterparty_data"]

    # FIXED: Intercepting the standard built-in file opener avoids unresolved string references
    builtins_open_path = "builtins.open"

    with patch(builtins_open_path, mock_open()) as mock_file:
        # Re-import dynamically inside the patch context to trap the file execution line triggers
        import utils.generate_counterparty_data

        # Verify that the folder system check executes smoothly
        mock_makedirs.assert_called()

        # Verify that exactly 3 distinct counterparty data sheets trigger file handles
        assert mock_file.call_count == 3
