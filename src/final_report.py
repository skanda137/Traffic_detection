import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from feature_pipeline import raw_tortuosity
from train_final import FORECAST_PATH, NOWCAST_PATH, TARGET, load_split

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
METRICS_PATH = REPORT_DIR / "phase2_final_metrics.csv"
SLICES_PATH = REPORT_DIR / "phase2_slices.csv"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance_ranking.csv"
INPUT_RANGES_PATH = REPORT_DIR / "input_ranges.json"

LINEAR_BASELINE_FEATURES = ["distance", "hour", "is_peak"]


def metrics_row(name, y, preds):
    mae = mean_absolute_error(y, preds)
    return {"model": name, "rmse_min": root_mean_squared_error(y, preds), "mae_min": mae,
            "mae_over_mean_actual": mae / y.mean(), "mean_signed_residual_min": float(np.mean(preds - y)),
            "test_rows": len(y)}


def slice_rows(name, test, preds):
    frame = pd.DataFrame({"actual": test[TARGET].to_numpy(), "abs_error": np.abs(preds - test[TARGET].to_numpy()),
                          "distance_quintile": pd.qcut(test["distance"], 5).to_numpy(),
                          "hour": test["hour"].to_numpy(), "weather": test["weather"].astype(str).to_numpy(),
                          "is_peak": test["is_peak"].to_numpy()})
    rows = []
    for slice_name in ["distance_quintile", "hour", "weather", "is_peak"]:
        for value, g in frame.groupby(slice_name, sort=True, observed=True):
            rows.append({"model": name, "slice": slice_name, "value": str(value), "rows": len(g),
                         "mae_min": g["abs_error"].mean(), "mae_over_mean_actual": g["abs_error"].mean() / g["actual"].mean()})
    return rows


def run_final_report():
    raw, train, val, test, categories = load_split()
    bundles = {name: joblib.load(path) for name, path in [("NOWCAST", NOWCAST_PATH), ("FORECAST", FORECAST_PATH)]}
    for name, b in bundles.items():
        counts = {"train": len(train), "val": len(val), "test": len(test)}
        if b["row_counts"] != counts or b["categories"] != categories:
            raise ValueError(f"{name} bundle was trained on a different split: {b['row_counts']} vs {counts}")
        if b["target_transform"] != "none":
            raise ValueError(f"{name} bundle has target_transform {b['target_transform']!r}; expected 'none'")
    print(f"Split: {bundles['NOWCAST']['split_description']}")
    print(f"TEST SET ONLY: {len(test):,} rows (baselines fitted on {len(train):,} train rows)")

    y = test[TARGET].to_numpy()
    pace = (train[TARGET] / train["distance"]).groupby(train["hour"]).median()
    linear = LinearRegression().fit(train[LINEAR_BASELINE_FEATURES].astype(float), train[TARGET])
    predictions = {
        "baseline (a) per-hour median pace x distance": test["hour"].map(pace).to_numpy() * test["distance"].to_numpy(),
        "baseline (b) linear distance+hour+is_peak": linear.predict(test[LINEAR_BASELINE_FEATURES].astype(float)),
    }
    for name, b in bundles.items():
        predictions[name] = b["model"].predict(test[b["features"]])

    metrics = pd.DataFrame([metrics_row(name, y, p) for name, p in predictions.items()])
    slices = pd.DataFrame([r for name in bundles for r in slice_rows(name, test, predictions[name])])
    importance = pd.concat([
        pd.DataFrame({"model": name, "feature": b["features"],
                      "importance_gain": b["model"].feature_importance(importance_type="gain")})
        .sort_values("importance_gain", ascending=False)
        .assign(gain_share=lambda d: d["importance_gain"] / d["importance_gain"].sum())
        for name, b in bundles.items()])

    # Inference accepts only inputs inside the p1-p99 range of the training rows
    range_values = {"distance": train["distance"], "haversine_dist": train["haversine_dist"],
                    "raw_tortuosity": raw_tortuosity(train), "density_ratio": train["density_ratio"]}
    input_ranges = {field: {"p1": float(v.quantile(0.01)), "p99": float(v.quantile(0.99))}
                    for field, v in range_values.items()}

    REPORT_DIR.mkdir(exist_ok=True)
    with open(INPUT_RANGES_PATH, "w") as f:
        json.dump(input_ranges, f, indent=2)
    metrics.to_csv(METRICS_PATH, index=False)
    slices.to_csv(SLICES_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.max_columns", 20):
        print(f"\n===== {METRICS_PATH.relative_to(ROOT)} =====")
        print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n===== {SLICES_PATH.relative_to(ROOT)} =====")
        print(slices.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n===== {IMPORTANCE_PATH.relative_to(ROOT)} =====")
        print(importance.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(f"\n===== {INPUT_RANGES_PATH.relative_to(ROOT)} =====")
        print(json.dumps(input_ranges, indent=2))


if __name__ == "__main__":
    run_final_report()
