import os
import sys
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Force python path configuration to find core project modules cleanly
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

from models.credit_model import calculate_ml_probability_of_default


@pytest.fixture
def mock_healthy_bank_df():
    """Generates a mock dataframe representing a highly stable bank with zero defaults."""
    years = list(range(2015, 2027))
    return pd.DataFrame({
        "counterparty_id": ["BARCGB22"] * len(years),
        "company_name": ["Barclays Bank PLC"] * len(years),
        "year": years,
        "tier1_capital_ratio": np.linspace(13.0, 14.0, len(years)),
        "leverage_ratio": np.linspace(4.5, 4.0, len(years)),
        "liquidity_coverage_ratio": np.linspace(140.0, 147.0, len(years)),
        "fred_credit_spread": np.linspace(1.5, 1.2, len(years)),
        "default_label": [0] * len(years)  # Perfect financial safety track record
    })


@pytest.fixture
def mock_distressed_bank_df():
    """Generates a mock dataframe representing a failing bank with explicit defaults."""
    years = list(range(2015, 2027))
    # Inject active bankruptcies/defaults (1) into the historical training path rows
    labels = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    return pd.DataFrame({
        "counterparty_id": ["XYZBANK"] * len(years),
        "company_name": ["XYZ Bank"] * len(years),
        "year": years,
        "tier1_capital_ratio": np.linspace(11.0, 8.0, len(years)),
        "leverage_ratio": np.linspace(3.5, 2.0, len(years)),
        "liquidity_coverage_ratio": np.linspace(110.0, 85.0, len(years)),
        "fred_credit_spread": np.linspace(2.5, 7.5, len(years)),
        "default_label": labels
    })


def test_unknown_counterparty_fallback_rule():
    """
    Ensures that passing a non-existent or unmapped counterparty ID
    safely skips execution and falls back to the standard 2% baseline rate.
    """
    fallback_pd = calculate_ml_probability_of_default("FAKE_BANK_999")
    assert fallback_pd == 0.02, "Unknown IDs must return standard 2% default baseline."


@patch("os.path.exists")
def test_missing_csv_file_fallback_rule(mock_exists):
    """
    Ensures that if a mapped CSV table file is physically missing from disk,
    the model catches the error and falls back to the safe 2% baseline rate.
    """
    mock_exists.return_value = False  # Simulate a missing file scenario
    fallback_pd = calculate_ml_probability_of_default("BARCGB22")
    assert fallback_pd == 0.02, "Missing data files must return standard 2% default baseline."


@patch("os.path.exists")
@patch("pandas.read_csv")
def test_healthy_bank_respects_minimum_floor(mock_read_csv, mock_exists, mock_healthy_bank_df):
    """
    Verifies that highly stable financial counterparties with low default risk
    have their probability rounded up to the standard 0.05% minimum floor constraint.
    """
    mock_exists.return_value = True
    mock_read_csv.return_value = mock_healthy_bank_df

    final_pd = calculate_ml_probability_of_default("BARCGB22")

    assert isinstance(final_pd, float), "Probability output must be a standard float."
    assert final_pd == 0.0005, f"Expected safe floor boundary of 0.0005 (0.05%), but got {final_pd}."


@patch("os.path.exists")
@patch("pandas.read_csv")
def test_distressed_bank_calculates_high_risk_premium(mock_read_csv, mock_exists, mock_distressed_bank_df):
    """
    Verifies that an institution showing deteriorating balance sheet metrics
    correctly generates a elevated default risk probability well above the minimum floor.
    """
    mock_exists.return_value = True
    mock_read_csv.return_value = mock_distressed_bank_df

    final_pd = calculate_ml_probability_of_default("XYZBANK")

    assert isinstance(final_pd, float)
    assert final_pd > 0.10, f"Distressed bank default risk should be high, but got {final_pd}."
    assert final_pd <= 1.0, "Probability values can never exceed the absolute 100% threshold."


@patch("os.path.exists")
@patch("pandas.read_csv")
def test_empty_historical_timeline_fallback_rule(mock_read_csv, mock_exists):
    """
    Ensures that if an empty or invalid spreadsheet dataset clears file filters,
    the script handles the exception and safely defaults to the standard 2% baseline.
    """
    mock_exists.return_value = True
    # Build an empty dataframe block matrix
    mock_read_csv.return_value = pd.DataFrame(columns=["year", "tier1_capital_ratio", "default_label"])

    fallback_pd = calculate_ml_probability_of_default("BARCGB22")
    assert fallback_pd == 0.02, "Empty data matrices must revert to standard 2% baseline."
