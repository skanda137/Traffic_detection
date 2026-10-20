from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from feature_pipeline import TRAFFIC_INPUTS, apply_categories, engineer_features

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = ROOT / "model_cleaned_traffic.parquet"
MODEL_DIR = ROOT / "models"
NOWCAST_PATH = MODEL_DIR / "nowcast_model.pkl"
FORECAST_PATH = MODEL_DIR / "forecast_model.pkl"

TARGET = "travel_time"
CATEGORICAL = ["day_of_week", "weather"]
SEED = 42

# Stage 0 gate G2: a random 80/20 split scored within 0.5% of the weekday->weekend split,
# so the weekend rows are not a distinct regime and the split is random.
SPLIT_DESCRIPTION = ("random 70/15/15 train/val/test, numpy default_rng(42) permutation of "
                     "model_cleaned_traffic.parquet row order; no timestamp column, split is random not temporal")

# Hyperparameters and feature list verbatim from train_prediction.py (v1); no new tuning
V1_PARAMS = {
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
V1_FEATURES = [
    "lat_src", "lon_src", "lat_dest", "lon_dest",
    "distance", "haversine_dist", "d_lat", "d_lon", "tortuosity",
    "hour", "sin_hour", "cos_hour", "is_peak", "is_weekend",
    "road_capacity", "vehicles", "density_ratio", "signal_time", "signal_delay_ratio",
    "day_of_week", "weather"
]
# Forecast = feature set (B): nothing that needs live traffic counts, no is_weekend
FORECAST_FEATURES = [f for f in V1_FEATURES if f not in
                     {"vehicles", "density_ratio", "signal_delay_ratio", "road_capacity", "signal_time", "is_weekend"}]
BASE_RAW_INPUTS = ["lat_src", "lon_src", "lat_dest", "lon_dest", "distance", "day_of_week", "hour", "weather"]


def load_split(seed=SEED):
    """Cleaned rows -> feature_pipeline -> random 70/15/15. Returns raw rows too (same index) for inference checks."""
    raw = pd.read_parquet(CLEAN_DATA_PATH)
    df = engineer_features(raw)
    perm = np.random.default_rng(seed).permutation(len(df))
    n_train, n_val = int(0.70 * len(df)), int(0.85 * len(df))
    train, val, test = df.iloc[perm[:n_train]], df.iloc[perm[n_train:n_val]], df.iloc[perm[n_val:]]
    categories = {col: sorted(train[col].unique().tolist()) for col in CATEGORICAL}
    train, val, test = (apply_categories(part, categories) for part in (train, val, test))
    return raw, train, val, test, categories


def train_bundle(name, features, raw_inputs, raw, train, val, test, categories, path):
    train_data = lgb.Dataset(train[features], label=train[TARGET], categorical_feature=CATEGORICAL)
    val_data = lgb.Dataset(val[features], label=val[TARGET], reference=train_data)
    booster = lgb.train(V1_PARAMS, train_data, num_boost_round=1000, valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    # Keep only the trees up to the early-stopping point so every predict() uses the same model
    model = lgb.Booster(model_str=booster.model_to_string(num_iteration=booster.best_iteration))

    preds = model.predict(val[features])
    val_metrics = {"rmse": root_mean_squared_error(val[TARGET], preds), "mae": mean_absolute_error(val[TARGET], preds)}
    print(f"{name}: {len(features)} features | best_iteration {booster.best_iteration} | "
          f"val RMSE {val_metrics['rmse']:.4f} min | val MAE {val_metrics['mae']:.4f} min")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model": model,
        "model_name": name,
        "features": features,
        "raw_inputs": raw_inputs,
        "raw_dtypes": {col: str(raw[col].dtype) for col in raw_inputs},
        "categories": categories,
        "target_transform": "none",
        "split_description": SPLIT_DESCRIPTION,
        "row_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "best_iteration": booster.best_iteration,
        "val_metrics": val_metrics,
    }, path)
    print(f"Saved {name} bundle to {path.relative_to(ROOT)}")


def train_final_models():
    raw, train, val, test, categories = load_split()
    print(f"Split: {SPLIT_DESCRIPTION}")
    print(f"Rows: train {len(train):,} | val {len(val):,} | test {len(test):,} (test not evaluated here)")
    print(f"Categories from training rows: {categories}")

    nowcast_features = [f for f in V1_FEATURES if not (f == "is_weekend" and train["is_weekend"].nunique() == 1)]
    print(f"is_weekend distinct values in training rows: {train['is_weekend'].nunique()} "
          f"-> {'kept' if 'is_weekend' in nowcast_features else 'dropped'} in NOWCAST")

    train_bundle("NOWCAST", nowcast_features, BASE_RAW_INPUTS + TRAFFIC_INPUTS,
                 raw, train, val, test, categories, NOWCAST_PATH)
    train_bundle("FORECAST", FORECAST_FEATURES, BASE_RAW_INPUTS,
                 raw, train, val, test, categories, FORECAST_PATH)


if __name__ == "__main__":
    train_final_models()
