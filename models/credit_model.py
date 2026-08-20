import os
import pandas as pd
from xgboost import XGBClassifier


def calculate_ml_probability_of_default(counterparty_id: str) -> float:
    """
    Loads historical standalone bank data, trains an XGBoost model,
    and returns the predicted Probability of Default (PD) for 2026.
    """
    # 1. Establish file paths relative to the models directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # Go up one level to the main workspace root

    file_mapping = {
        "BARCGB22": "barclays_credit_scores.csv",
        "ABNAMRO": "abnamro_credit_scores.csv",
        "XYZBANK": "xyzbank_credit_scores.csv"
    }

    target_file = file_mapping.get(counterparty_id)
    if not target_file:
        print(f"   ⚠️ Credit Engine Warning: Unknown Counterparty ID '{counterparty_id}'. Using standard baseline.")
        return 0.02

    csv_path = os.path.join(root_dir, "data", "historical", "counterparty", target_file)

    if not os.path.exists(csv_path):
        print(f"   ⚠️ Credit Engine Error: Missing file '{csv_path}'. Using standard baseline.")
        return 0.02

    # 2. Ingest and sort historical dataset matrices
    df = pd.read_csv(csv_path).sort_values(by="year")
    feature_columns = ["tier1_capital_ratio", "leverage_ratio", "liquidity_coverage_ratio", "fred_credit_spread"]

    # 3. Train-Test Split (Isolate the current year from history)
    train_history = df[df["year"] < 2026]
    live_run_row = df[df["year"] == 2026]

    if train_history.empty or live_run_row.empty:
        return 0.02

    X_train = train_history[feature_columns]
    y_train = train_history["default_label"]
    X_live = live_run_row[feature_columns]

    # 4. Initialize and fit the Machine Learning Classifier
    ml_classifier = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    ml_classifier.fit(X_train, y_train)

    # 5. Predict the smooth mathematical Probability of Default (PD)
    raw_probabilities = ml_classifier.predict_proba(X_live)
    predicted_pd = float(raw_probabilities[0][1])

    return max(predicted_pd, 0.0005)


if __name__ == "__main__":
    print(f'{"BARCGB22"}: {calculate_ml_probability_of_default(counterparty_id="BARCGB22")}')
    print(f'{"ABNAMRO"}: {calculate_ml_probability_of_default(counterparty_id="ABNAMRO")}')
