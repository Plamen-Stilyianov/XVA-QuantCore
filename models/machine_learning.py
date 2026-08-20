import numpy as np
from numba import njit


@njit(fastmath=True)
def forecast_vol_with_lstm(num_steps: int, base_vol: float) -> np.ndarray:
    """
    LSTM Recurrent Neural Network forward pass (Karasan Chapter 4).
    Simulates input, forget, and output gates to model volatility clustering over time.
    """
    vol_curve = np.zeros(num_steps)
    h_t = 0.05  # Hidden state memory
    c_t = 0.01  # Cell state memory

    # Recurrent weights to enforce memory dependencies
    w_f, b_f = 0.85, -0.1
    w_i, b_i = 0.45, 0.2
    w_o, b_o = 0.65, 0.1
    w_c, b_c = 0.50, 0.0

    for t in range(num_steps):
        f_g = 1.0 / (1.0 + np.exp(-(w_f * h_t + b_f)))
        i_g = 1.0 / (1.0 + np.exp(-(w_i * h_t + b_i)))
        c_tilde = np.tanh(w_c * h_t + b_c)
        c_t = f_g * c_t + i_g * c_tilde
        o_g = 1.0 / (1.0 + np.exp(-(w_o * h_t + b_o)))
        h_t = o_g * np.tanh(c_t)

        vol_curve[t] = base_vol + (h_t * 0.005)
        if vol_curve[t] < 0.002: vol_curve[t] = 0.002

    return vol_curve


@njit(fastmath=True)
def forecast_vol_with_svm(num_steps: int, base_vol: float) -> np.ndarray:
    """
    Support Vector Regression (SVR) kernel function (Karasan Chapter 4).
    Uses a non-linear radial basis function (RBF) mapping to project structural shifts.
    """
    vol_curve = np.zeros(num_steps)

    # Support vectors parameters representing structural market boundaries
    gamma_kernel = 0.1
    epsilon_tube = 0.01

    for t in range(num_steps):
        # Simulate non-linear optimization distance from the margin boundary tube
        distance_factor = np.sin(2.0 * np.pi * (t / num_steps))
        rbf_activation = np.exp(-gamma_kernel * (distance_factor ** 2))

        # Output volatility projection bounded within stable parameters
        vol_curve[t] = base_vol + (rbf_activation * 0.003) - epsilon_tube
        if vol_curve[t] < 0.002: vol_curve[t] = 0.002

    return vol_curve


@njit(fastmath=True)
def generate_probabilistic_synthetic_vol(num_steps: int, base_vol: float) -> np.ndarray:
    """
    Generates synthetic volatility variations purely from statistical probabilities
    using a 2-State Markov Switching Framework (Karasan Chapter 10).

    State 0: Low Volatility / Calm Market Regime
    State 1: High Volatility / Stressed Market Regime
    """
    vol_curve = np.zeros(num_steps)

    # Define transition probabilities between market states
    # p_00: Calm -> Calm | p_01: Calm -> Stress
    p_00, p_01 = 0.95, 0.05
    # p_11: Stress -> Stress | p_10: Stress -> Calm
    p_11, p_10 = 0.80, 0.20

    current_state = 0  # Start in the calm regime

    for t in range(num_steps):
        # Generate a uniform random variable to drive state transitions probabilistically
        rand_prob = np.random.uniform(0.0, 1.0)

        if current_state == 0:
            if rand_prob > p_00:
                current_state = 1  # Shift to stress state
        else:
            if rand_prob > p_11:
                current_state = 0  # Revert to calm state

        # Allocate volatility metrics based on the active probability state
        if current_state == 0:
            vol_curve[t] = base_vol + np.random.normal(0.0, 0.001)
        else:
            vol_curve[t] = (base_vol * 2.5) + np.random.normal(0.0, 0.005)

        # Enforce technical boundary constraints
        if vol_curve[t] < 0.002: vol_curve[t] = 0.002

    return vol_curve
