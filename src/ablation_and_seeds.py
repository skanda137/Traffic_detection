"""Reporting only: how much each feature set beats baseline (a) across split seeds. No model is selected here."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from train_final import CATEGORICAL, FORECAST_FEATURES, TARGET, V1_FEATURES, V1_PARAMS, load_split

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "reports" / "ablation_and_seeds.csv"
SPLIT_SEEDS = [42, 7, 13]
ABLATION_FEATURES = [f for f in V1_FEATURES if f not in {"vehicles", "density_ratio", "signal_delay_ratio"}]
METRIC_COLUMNS = ["rmse_min", "mae_min", "baseline_a_rmse_min", "baseline_a_mae_min",
                  "rmse_improvement_pct", "mae_improvement_pct", "best_iteration"]


def fit_and_score(features, train, val, test):
    train_data = lgb.Dataset(train[features], label=train[TARGET], categorical_feature=CATEGORICAL)
    val_data = lgb.Dataset(val[features], label=val[TARGET], reference=train_data)
    booster = lgb.train(V1_PARAMS, train_data, num_boost_round=1000, valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = booster.predict(test[features], num_iteration=booster.best_iteration)
    return preds, booster.best_iteration


def run():
    rows = []
    for seed in SPLIT_SEEDS:
        _, train, val, test, _ = load_split(seed=seed)
        y = test[TARGET].to_numpy()
        pace = (train[TARGET] / train["distance"]).groupby(train["hour"]).median()
        baseline = test["hour"].map(pace).to_numpy() * test["distance"].to_numpy()
        base_rmse, base_mae = root_mean_squared_error(y, baseline), mean_absolute_error(y, baseline)
        nowcast_features = [f for f in V1_FEATURES if not (f == "is_weekend" and train["is_weekend"].nunique() == 1)]
        for name, features in [("NOWCAST", nowcast_features), ("FORECAST", FORECAST_FEATURES),
                               ("ABLATION_18 (v1 minus vehicles, density_ratio, signal_delay_ratio)", ABLATION_FEATURES)]:
            preds, best_iteration = fit_and_score(features, train, val, test)
            rmse, mae = root_mean_squared_error(y, preds), mean_absolute_error(y, preds)
            rows.append({"seed": str(seed), "model": name, "n_features": len(features), "rmse_min": rmse, "mae_min": mae,
                         "baseline_a_rmse_min": base_rmse, "baseline_a_mae_min": base_mae,
                         "rmse_improvement_pct": 100 * (1 - rmse / base_rmse),
                         "mae_improvement_pct": 100 * (1 - mae / base_mae), "best_iteration": best_iteration})
            print(f"seed {seed} | {name}: test RMSE {rmse:.4f} MAE {mae:.4f} | baseline (a) RMSE {base_rmse:.4f} "
                  f"MAE {base_mae:.4f} | improvement RMSE {rows[-1]['rmse_improvement_pct']:.2f}% "
                  f"MAE {rows[-1]['mae_improvement_pct']:.2f}% | best_iteration {best_iteration}")

    per_seed = pd.DataFrame(rows)
    summary = []
    for stat in ("mean", "std"):
        agg = per_seed.groupby(["model", "n_features"], sort=False)[METRIC_COLUMNS].agg(stat).reset_index()
        summary.append(agg.assign(seed=stat))
    result = pd.concat([per_seed] + summary, ignore_index=True)[["seed", "model", "n_features"] + METRIC_COLUMNS]
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(f"\n===== {OUTPUT_PATH.relative_to(ROOT)} =====")
        print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    run()
