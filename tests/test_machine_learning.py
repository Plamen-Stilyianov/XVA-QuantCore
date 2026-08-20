import numpy as np
from models.machine_learning import forecast_vol_with_lstm, forecast_vol_with_svm, generate_probabilistic_synthetic_vol


def test_lstm_volatility_forecasting_bounds():
    """Verifies LSTM forward pass output arrays and boundary conditions (Karasan Ch. 4)."""
    num_steps = 360
    base_vol = 0.025
    vol_curve = forecast_vol_with_lstm(num_steps, base_vol)

    assert isinstance(vol_curve, np.ndarray), "Output must be a standard NumPy array."
    assert len(vol_curve) == num_steps, "Array length must match steps dimension."
    assert np.all(vol_curve >= 0.002), "LSTM curve dropped below the minimum volatility floor constraint."


def test_svm_rbf_kernel_volatility_generation():
    """Verifies non-linear SVM RBF kernel matrix bounds (Karasan Ch. 4)."""
    num_steps = 100
    base_vol = 0.015
    vol_curve = forecast_vol_with_svm(num_steps, base_vol)

    assert len(vol_curve) == num_steps
    assert np.all(vol_curve > 0.0), "SVM output values must be strictly positive."


def test_markov_switching_synthetic_variance():
    """Verifies 2-state generative probability regime-switching variance (Karasan Ch. 10)."""
    num_steps = 360
    base_vol = 0.025
    vol_curve = generate_probabilistic_synthetic_vol(num_steps, base_vol)

    assert len(vol_curve) == num_steps
    assert np.std(vol_curve) > 0.0, "Generative switching model failed to construct stochastic regime variations."
