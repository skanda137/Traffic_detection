import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from feature_pipeline import input_range_violations, raw_tortuosity
from train_final import FORECAST_PATH, NOWCAST_PATH, TARGET, load_split

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
METRICS_PATH = REPORT_DIR / "phase2_final_metrics.csv"
SLICES_PATH = REPORT_DIR / "phase2_slices.csv"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance_ranking.csv"
INPUT_RANGES_PATH = REPORT_DIR / "input_ranges.json"
EXAMPLE_ROUTES_PATH = REPORT_DIR / "example_routes.csv"
ERROR_QUANTILES_PATH = REPORT_DIR / "phase2_error_quantiles.csv"

DISTANCE_BANDS = ["Very short", "Short", "Medium", "Long", "Very long"]
ROUTES_PER_BAND = [3, 3, 2, 2, 2]
ROUTE_COLUMNS = ["lat_src", "lon_src", "lat_dest", "lon_dest", "distance", "vehicles", "road_capacity", "signal_time"]

LINEAR_BASELINE_FEATURES = ["distance", "hour", "is_peak"]
SEED_ROUTES = 42


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

    # Empirical 80% interval multipliers: q10/q90 of actual / predicted over the test set (marginal, not per-input)
    for name in bundles:
        if (predictions[name] <= 0).any():
            raise ValueError(f"{name} has {(predictions[name] <= 0).sum()} non-positive test predictions; actual/predicted undefined")
    error_quantiles = pd.DataFrame([{
        "model": name,
        "q10_actual_over_predicted": float(np.quantile(y / predictions[name], 0.10)),
        "q90_actual_over_predicted": float(np.quantile(y / predictions[name], 0.90)),
        "mae_over_mean_actual": float(metrics.set_index("model").loc[name, "mae_over_mean_actual"]),
    } for name in bundles])

    # Inference accepts only inputs inside the p1-p99 range of the training rows
    range_values = {"distance": train["distance"], "haversine_dist": train["haversine_dist"],
                    "raw_tortuosity": raw_tortuosity(train), "density_ratio": train["density_ratio"]}
    input_ranges = {field: {"p1": float(v.quantile(0.01)), "p99": float(v.quantile(0.99))}
                    for field, v in range_values.items()}
    # Raw tortuosity is effectively constant (std ~1e-4), so p1/p99 bounds would be numeric noise: use training min/max
    tortuosity = range_values["raw_tortuosity"]
    input_ranges["raw_tortuosity"] = {"p1": float(tortuosity.min()), "p99": float(tortuosity.max())}
    # Dashboard defaults: place-name routes (distance = haversine x median tortuosity, coordinates inside the
    # training box) and live-traffic inputs at their training medians
    train_raw = raw.loc[train.index]
    input_ranges["raw_tortuosity_median"] = float(tortuosity.median())
    input_ranges["lat_min"] = float(min(train_raw["lat_src"].min(), train_raw["lat_dest"].min()))
    input_ranges["lat_max"] = float(max(train_raw["lat_src"].max(), train_raw["lat_dest"].max()))
    input_ranges["lon_min"] = float(min(train_raw["lon_src"].min(), train_raw["lon_dest"].min()))
    input_ranges["lon_max"] = float(max(train_raw["lon_src"].max(), train_raw["lon_dest"].max()))
    for field in ["vehicles", "road_capacity", "signal_time"]:
        input_ranges[f"{field}_median"] = float(train[field].median())

    # Dashboard example routes: test rows that pass every input check, stratified by distance quintile
    valid = test[~input_range_violations(test, input_ranges).any(axis=1)]
    band = pd.qcut(valid["distance"], 5, labels=DISTANCE_BANDS)
    routes = pd.concat([raw.loc[valid.index[band == name], ROUTE_COLUMNS].sample(k, random_state=SEED_ROUTES)
                        for name, k in zip(DISTANCE_BANDS, ROUTES_PER_BAND)]).sort_values("distance")
    routes.insert(0, "band", band.loc[routes.index].astype(str))
    routes.insert(0, "label", [f"{b} - {d:.2f} km" for b, d in zip(routes["band"], routes["distance"])])
    if routes["label"].duplicated().any():
        raise ValueError(f"Duplicate example route labels: {routes.loc[routes['label'].duplicated(), 'label'].tolist()}")
    print(f"Example routes drawn from {len(valid):,} of {len(test):,} test rows that pass the input checks")

    REPORT_DIR.mkdir(exist_ok=True)
    routes.to_csv(EXAMPLE_ROUTES_PATH, index=False)
    error_quantiles.to_csv(ERROR_QUANTILES_PATH, index=False)
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
        print(f"\n===== {ERROR_QUANTILES_PATH.relative_to(ROOT)} =====")
        print(error_quantiles.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n===== {EXAMPLE_ROUTES_PATH.relative_to(ROOT)} =====")
        print(routes.to_string(index=False))


if __name__ == "__main__":
    run_final_report()
