from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_DATA_PATH = Path("model_featured_traffic.parquet")
MODEL_EXPORT_PATH = Path("traffic_model_final.pkl")

def train_production_model():
    print("Reading feature-engineered dataset...")
    df = pd.read_parquet(FEATURE_DATA_PATH)

    TARGET = "travel_time"
    FEATURES = [
        "lat_src", "lon_src", "lat_dest", "lon_dest",
        "distance", "haversine_dist", "d_lat", "d_lon", "tortuosity",
        "hour", "sin_hour", "cos_hour", "is_peak", "is_weekend",
        "road_capacity", "vehicles", "density_ratio", "signal_time", "signal_delay_ratio",
        "day_of_week", "weather"
    ]

    # Block Temporal Split: Train on Weekdays (0-4), Validate on Weekends (5-6)
    # This simulates out-of-sample forward deployment
    print("Constructing Temporal Block Holdout Split (Weekdays -> Weekend)...")
    train_mask = df["is_weekend"] == 0
    val_mask = df["is_weekend"] == 1

    X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET]
    X_val, y_val = df.loc[val_mask, FEATURES], df.loc[val_mask, TARGET]

    print(f"Training Rows: {len(X_train):,} | Validation Rows: {len(X_val):,}")

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=["day_of_week", "weather"])
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42
    }

    print("Beginning LightGBM training on 2.1 Million row pipeline...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print("\n================ FINAL VALIDATION METRICS ================")
    print(f"Validation RMSE: {rmse:.4f} minutes")
    print(f"Validation MAE:  {mae:.4f} minutes")
    print(f"Validation R2:   {r2:.4f}")
    print("==========================================================")

    # Save artifacts
    artifacts = {
        "model": model,
        "features": FEATURES,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2}
    }
    joblib.dump(artifacts, MODEL_EXPORT_PATH)
    print(f"Model and metadata saved to {MODEL_EXPORT_PATH}")

if __name__ == "__main__":
    train_production_model()