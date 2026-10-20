from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from feature_pipeline import apply_categories, engineer_features

MODEL_EXPORT_PATH = Path("traffic_model_final.pkl")
FEATURE_DATA_PATH = Path("model_featured_traffic.parquet")

def run_diagnostics():
    artifacts = joblib.load(MODEL_EXPORT_PATH)
    model = artifacts["model"]
    features = artifacts["features"]

    df = pd.read_parquet(FEATURE_DATA_PATH)
    val_df = df[df["is_weekend"] == 1].sample(n=100000, random_state=42).copy()

    preds = model.predict(val_df[features])
    if artifacts.get("log_transformed"):
        preds = np.expm1(preds)
    val_df["pred_travel_time"] = preds
    val_df["abs_error"] = np.abs(val_df["travel_time"] - val_df["pred_travel_time"])

    print("\n--- Diagnostic 1: Error Across Weather States ---")
    weather_err = val_df.groupby("weather", observed=True)["abs_error"].agg(["mean", "count"]).rename(columns={"mean": "MAE_Minutes"})
    print(weather_err)

    print("\n--- Diagnostic 2: Peak vs Non-Peak Holdout Performance ---")
    peak_err = val_df.groupby("is_peak")["abs_error"].agg(["mean", "count"]).rename(columns={"mean": "MAE_Minutes"})
    print(peak_err)

    # Export Feature Importance
    importance_df = pd.DataFrame({
        "feature": features,
        "importance_gain": model.feature_importance(importance_type="gain")
    }).sort_values(by="importance_gain", ascending=False)

    print("\n--- Top 10 Most Critical Predictive Signals ---")
    print(importance_df.head(10).to_string(index=False))
    importance_df.to_csv("feature_importance_ranking.csv", index=False)

class TrafficInferenceService:
    """Production Inference Service for Member D's UI."""
    def __init__(self, artifact_path):
        bundle = joblib.load(artifact_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.raw_inputs = bundle["raw_inputs"]
        self.raw_dtypes = bundle["raw_dtypes"]
        self.categories = bundle["categories"]
        self.target_transform = bundle["target_transform"]
        if self.target_transform not in ("none", "log1p"):
            raise ValueError(f"Unsupported target_transform {self.target_transform!r}")

    def predict(self, record: dict) -> dict:
        missing = [c for c in self.raw_inputs if c not in record]
        unexpected = [c for c in record if c not in self.raw_inputs]
        if missing or unexpected:
            raise ValueError(f"Record inputs mismatch: missing {missing}, unexpected {unexpected}; expected {self.raw_inputs}")

        # Same feature path as training: raw dtypes -> feature_pipeline -> saved categories
        row = pd.DataFrame([{c: record[c] for c in self.raw_inputs}]).astype(self.raw_dtypes)
        row = apply_categories(engineer_features(row), self.categories)

        pred_time = float(self.model.predict(row[self.features])[0])
        if self.target_transform == "log1p":
            pred_time = float(np.expm1(pred_time))
        free_flow_est = float(row["distance"].values[0] / 40.0 * 60.0) # Assume 40 km/h baseline
        inferred_congestion_ratio = max(1.0, pred_time / max(free_flow_est, 1.0))

        return {
            "predicted_travel_time_minutes": pred_time,
            "derived_congestion_ratio": round(inferred_congestion_ratio, 2),
            "congestion_severity": "Severe" if inferred_congestion_ratio > 1.6 else ("Moderate" if inferred_congestion_ratio > 1.25 else "Low")
        }

if __name__ == "__main__":
    run_diagnostics()