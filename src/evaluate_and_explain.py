from pathlib import Path
import joblib
import numpy as np
import pandas as pd

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
    def __init__(self, artifact_path: str = "traffic_model_final.pkl"):
        bundle = joblib.load(artifact_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.log_transformed = bundle.get("log_transformed", False)

    def predict(self, record: dict) -> dict:
        row = pd.DataFrame([record])
        
        # Spatial derivations
        from feature_pipeline import haversine_np
        row["haversine_dist"] = haversine_np(row["lat_src"], row["lon_src"], row["lat_dest"], row["lon_dest"]).astype("float32")
        row["d_lat"] = (row["lat_dest"] - row["lat_src"]).astype("float32")
        row["d_lon"] = (row["lon_dest"] - row["lon_src"]).astype("float32")
        row["tortuosity"] = (row["distance"] / np.maximum(row["haversine_dist"], 0.05)).clip(0.8, 5.0).astype("float32")
        row["density_ratio"] = (row["vehicles"] / np.maximum(row["road_capacity"], 10.0)).astype("float32")
        row["signal_delay_ratio"] = (row["signal_time"] / np.maximum(row["distance"], 0.5)).astype("float32")
        row["sin_hour"] = np.sin(2 * np.pi * row["hour"] / 24.0).astype("float32")
        row["cos_hour"] = np.cos(2 * np.pi * row["hour"] / 24.0).astype("float32")
        row["day_of_week"] = row["day_of_week"].astype("category")
        row["weather"] = row["weather"].astype("category")

        pred_time = float(self.model.predict(row[self.features])[0])
        if self.log_transformed:
            pred_time = float(np.expm1(pred_time))
        free_flow_est = float(row["distance"].values[0] / 40.0 * 60.0) # Assume 40 km/h baseline
        inferred_congestion_ratio = max(1.0, pred_time / max(free_flow_est, 1.0))

        return {
            "predicted_travel_time_minutes": round(pred_time, 2),
            "derived_congestion_ratio": round(inferred_congestion_ratio, 2),
            "congestion_severity": "Severe" if inferred_congestion_ratio > 1.6 else ("Moderate" if inferred_congestion_ratio > 1.25 else "Low")
        }

if __name__ == "__main__":
    run_diagnostics()