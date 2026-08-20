import numpy as np
from app import execution_pipeline_router


def test_execution_pipeline_router_fallback_resilience():
    """
    Verifies that the main data router handles missing file paths gracefully.
    It must return a valid 360-step NumPy array using fallback data records.
    """
    # Call the router with dummy parameters to trigger the fallback logic
    ee_profile = execution_pipeline_router(
        asset_class="rates",
        file_name="NON_EXISTENT_FILE.xml",
        ai_model_choice="LSTM"
    )

    assert isinstance(ee_profile, np.ndarray), "The pipeline router must always return a NumPy array."
    assert len(ee_profile) == 361, "The profile array length must match the 360-day step structure."
    assert ee_profile[0] == 0.0, "The fallback array must strictly enforce the Day 0 par constraint."
