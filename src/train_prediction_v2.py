from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

FEATURE_DATA_PATH = Path("model_featured_traffic.parquet")
MODEL_EXPORT_PATH = Path("traffic_model_final.pkl")

def train_production_model_v2():
    print("Reading feature-engineered dataset...")
    df = pd.read_parquet(FEATURE_DATA_PATH)

    TARGET = "travel_time"
    FEATURES = [
        "lat_src", "lon_src", "lat_dest", "lon_dest",
        "distance", "haversine_dist", "d_lat", "d_lon", "tortuosity",
        "hour", "sin_hour", "cos_hour", "is_peak", "is_weekend",
        "road_capacity", "density_ratio", "signal_time", "signal_delay_ratio",
        "day_of_week", "weather"
    ]

    # 1. Temporal Holdout Split
    train_mask = df["is_weekend"] == 0
    val_mask = df["is_weekend"] == 1

    X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET]
    X_val, y_val = df.loc[val_mask, FEATURES], df.loc[val_mask, TARGET]

    # 2. Peak Error Mitigation: Apply sample weights to force learning on peak conditions
    weights_train = np.where(X_train["is_peak"] == 1, 2.5, 1.0)

    # 3. Target Stabilization: Transform with log1p to suppress extreme target outliers
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    train_data = lgb.Dataset(
        X_train, 
        label=y_train_log, 
        weight=weights_train,
        categorical_feature=["day_of_week", "weather"]
    )
    val_data = lgb.Dataset(
        X_val, 
        label=y_val_log, 
        reference=train_data
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.06,
        "num_leaves": 95,          # Expanded depth for peak interactions
        "max_depth": 9,
        "min_child_samples": 60,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42
    }

    print("Training Robust LightGBM on 2.1M rows with peak-hour reweighting...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1200,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Invert log transform back to original minute scale
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)

    rmse = root_mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print("\n================ V2 MODEL VALIDATION METRICS ================")
    print(f"Validation RMSE: {rmse:.4f} minutes")
    print(f"Validation MAE:  {mae:.4f} minutes")
    print(f"Validation R2:   {r2:.4f}")
    print("=============================================================")

    # Breakdown by Peak
    val_df = X_val.copy()
    val_df["abs_err"] = np.abs(y_val - preds)
    print("\nUpdated Peak Breakdown:")
    print(val_df.groupby("is_peak")["abs_err"].mean())

    artifacts = {
        "model": model,
        "features": FEATURES,
        "log_transformed": True,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2}
    }
    joblib.dump(artifacts, MODEL_EXPORT_PATH)
    print(f"Overwritten improved model in {MODEL_EXPORT_PATH}")

if __name__ == "__main__":
    train_production_model_v2()