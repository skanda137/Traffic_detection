import json
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import joblib

from data_validator import validate_traffic_data

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "synthetic_traffic_dataset.csv"
MODEL_PATH = ROOT / "models" / "best_congestion_model.pkl"
REPORT_DIR = ROOT / "reports"

FEATURES = [
    "chokepoint_id", "time_slot", "day_of_week",
    "is_weekend", "is_festival", "school_term",
    "rainfall_mm", "visibility_km"
]
CATEGORICAL = ["chokepoint_id", "time_slot"]
TARGET = "congestion_ratio"

# Synopsis 4.5: train on weeks 1-5, hold out weeks 6-8.
# Tuning happens *inside* the training window (fit on weeks 1-4, tune on week 5)
# so the holdout weeks are touched exactly once, for the final report.
TRAIN_DAYS = 35
TUNE_DAYS = 28

# Commuter-facing severity bands (shared with inference_service)
SEVERITY_BINS = [-np.inf, 1.25, 1.60, np.inf]
SEVERITY_LABELS = ["Low", "Moderate", "Severe"]

SEED = 42


def load_data(path=DATA_PATH):
    return validate_traffic_data(pd.read_csv(path))


def split_by_date(df, split_days):
    split_date = df["date"].min() + pd.Timedelta(days=split_days)
    return df[df["date"] < split_date].copy(), df[df["date"] >= split_date].copy()


def apply_categories(X, categories):
    # Fixed category sets so LightGBM/XGBoost see identical codes at train, validation and inference time
    X = X.copy()
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def xy(train_df, val_df):
    categories = {col: sorted(train_df[col].unique()) for col in CATEGORICAL}
    return (apply_categories(train_df[FEATURES], categories), train_df[TARGET],
            apply_categories(val_df[FEATURES], categories), val_df[TARGET])


def run_temporal_split(df, split_days=TRAIN_DAYS):
    df = df.sort_values(by=["date", "time_slot"]).reset_index(drop=True)
    return xy(*split_by_date(df, split_days))


def to_severity(ratios):
    return pd.cut(np.asarray(ratios), bins=SEVERITY_BINS, labels=SEVERITY_LABELS)


def score(y_true, y_pred):
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "severity_accuracy": float(np.mean(to_severity(y_true) == to_severity(y_pred))),
    }


def data_volume_report(train_df):
    """Row counts behind the gradient-boosting-over-LSTM decision (synopsis 4.4)."""
    rows_per_cp = train_df.groupby("chokepoint_id", observed=True).size()
    series_len = train_df.groupby(["chokepoint_id", "time_slot"], observed=True).size()
    report = {
        "train_rows": int(len(train_df)),
        "chokepoints": int(train_df["chokepoint_id"].nunique()),
        "train_days": int(train_df["date"].nunique()),
        "rows_per_chokepoint": int(rows_per_cp.median()),
        "series_length_per_chokepoint_slot": int(series_len.median()),
        "distinct_feature_rows": int(train_df[FEATURES].drop_duplicates().shape[0]),
    }
    print("--- 0. Data Volume (model-choice justification) ---")
    print(f"Training rows:                          {report['train_rows']}")
    print(f"Chokepoints x days:                     {report['chokepoints']} x {report['train_days']}")
    print(f"Rows per chokepoint:                    {report['rows_per_chokepoint']}")
    print(f"Sequence length per chokepoint x slot:  {report['series_length_per_chokepoint_slot']} time steps")
    print("-> An LSTM would learn from ~35-step, 1-sample-per-day sequences with gaps between slots;")
    print("   that is far too short to learn temporal dynamics, while a tabular GBDT uses all rows jointly.")
    return report


def naive_baselines(train_df, val_df):
    """What a model has to beat: 'just use the historical average'."""
    global_mean = np.full(len(val_df), train_df[TARGET].mean())

    keys = ["chokepoint_id", "time_slot", "is_weekend"]
    hist = train_df.groupby(keys, observed=True)[TARGET].mean().rename("hist_mean").reset_index()
    fallback = train_df.groupby(["chokepoint_id", "time_slot"], observed=True)[TARGET].mean().rename("fallback").reset_index()
    merged = (val_df[keys].astype({"chokepoint_id": str, "time_slot": str})
              .merge(hist.astype({"chokepoint_id": str, "time_slot": str}), on=keys, how="left")
              .merge(fallback.astype({"chokepoint_id": str, "time_slot": str}), on=["chokepoint_id", "time_slot"], how="left"))
    historical = merged["hist_mean"].fillna(merged["fallback"]).fillna(train_df[TARGET].mean()).to_numpy()
    return {"Global mean": global_mean, "Historical mean (cp x slot x weekend)": historical}


def tune_lightgbm(X_fit, y_fit, X_tune, y_tune, n_trials=25):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": SEED,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        model = lgb.train(
            params,
            lgb.Dataset(X_fit, label=y_fit),
            num_boost_round=1000,
            valid_sets=[lgb.Dataset(X_tune, label=y_tune)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        trial.set_user_attr("best_iteration", model.best_iteration)
        return root_mean_squared_error(y_tune, model.predict(X_tune, num_iteration=model.best_iteration))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    params = {**study.best_params, "objective": "regression", "metric": "rmse",
              "verbosity": -1, "seed": SEED, "bagging_freq": 1}
    return params, study.best_trial.user_attrs["best_iteration"], study.best_value


XGB_PARAMS = dict(
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    tree_method="hist",
    random_state=SEED,
)


def execute_pipeline():
    df = load_data().sort_values(["date", "time_slot"]).reset_index(drop=True)
    train_df, holdout_df = split_by_date(df, TRAIN_DAYS)
    X_tr, y_tr, X_va, y_va = xy(train_df, holdout_df)
    categories = {col: list(X_tr[col].cat.categories) for col in CATEGORICAL}

    fit_mask = (train_df["date"] < df["date"].min() + pd.Timedelta(days=TUNE_DAYS)).to_numpy()
    X_fit, y_fit = X_tr[fit_mask], y_tr[fit_mask]
    X_tune, y_tune = X_tr[~fit_mask], y_tr[~fit_mask]

    print(f"Train: {train_df['date'].min().date()} -> {train_df['date'].max().date()} "
          f"(tuning fold from {train_df.loc[~fit_mask, 'date'].min().date()})")
    print(f"Holdout: {holdout_df['date'].min().date()} -> {holdout_df['date'].max().date()}\n")
    volume = data_volume_report(train_df)

    print("\n--- 1. Hyperparameter Tuning (Optuna, weeks 1-4 -> week 5) ---")
    lgb_params, lgb_rounds, lgb_tune_rmse = tune_lightgbm(X_fit, y_fit, X_tune, y_tune)
    print(f"LightGBM tuning-fold RMSE: {lgb_tune_rmse:.4f} ({lgb_rounds} rounds)")

    xgb_probe = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, **XGB_PARAMS)
    xgb_probe.fit(X_fit, y_fit, eval_set=[(X_tune, y_tune)], verbose=False)
    xgb_rounds = xgb_probe.best_iteration + 1
    xgb_tune_rmse = root_mean_squared_error(y_tune, xgb_probe.predict(X_tune))
    print(f"XGBoost  tuning-fold RMSE: {xgb_tune_rmse:.4f} ({xgb_rounds} rounds)")

    print("\n--- 2. Refit on all 5 training weeks ---")
    lgb_model = lgb.train(lgb_params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=lgb_rounds)
    xgb_model = xgb.XGBRegressor(n_estimators=xgb_rounds, **XGB_PARAMS).fit(X_tr, y_tr)

    # Model selection uses the tuning fold only; the holdout never influences which model ships
    chosen_name = "LightGBM" if lgb_tune_rmse <= xgb_tune_rmse else "XGBoost"
    chosen_model = lgb_model if chosen_name == "LightGBM" else xgb_model
    print(f"Selected on tuning fold: {chosen_name}")

    predictions = naive_baselines(train_df, holdout_df)
    predictions["LightGBM (tuned)"] = lgb_model.predict(X_va)
    predictions["XGBoost"] = xgb_model.predict(X_va)

    results = pd.DataFrame({name: score(y_va, p) for name, p in predictions.items()}).T
    print("\n================ HOLDOUT RESULTS (weeks 6-8, never seen in training/tuning) ================")
    print(results.to_string(float_format=lambda v: f"{v:.4f}"))
    print("=============================================================================================")

    hist_rmse = results.loc["Historical mean (cp x slot x weekend)", "rmse"]
    chosen_key = "LightGBM (tuned)" if chosen_name == "LightGBM" else "XGBoost"
    gain = 100 * (1 - results.loc[chosen_key, "rmse"] / hist_rmse)
    print(f"{chosen_name} RMSE improvement over historical-average baseline: {gain:.1f}%")

    if chosen_name == "LightGBM":
        importance = pd.Series(lgb_model.feature_importance("gain"), index=FEATURES)
    else:
        importance = pd.Series(xgb_model.get_booster().get_score(importance_type="gain")).reindex(FEATURES).fillna(0)
    importance = (importance / importance.sum()).sort_values(ascending=False)
    print("\n--- 3. Feature importance (share of total gain) ---")
    print(importance.to_string(float_format=lambda v: f"{v:.3f}"))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": chosen_model, "model_name": chosen_name,
                 "features": FEATURES, "categories": categories}, MODEL_PATH)
    results.to_csv(REPORT_DIR / "holdout_metrics.csv", float_format="%.4f")
    importance.rename("gain_share").to_csv(REPORT_DIR / "feature_importance.csv", float_format="%.4f")
    with open(REPORT_DIR / "model_selection.json", "w") as f:
        json.dump({"data_volume": volume, "selected_model": chosen_name,
                   "tuning_fold_rmse": {"LightGBM": lgb_tune_rmse, "XGBoost": xgb_tune_rmse},
                   "boosting_rounds": {"LightGBM": lgb_rounds, "XGBoost": xgb_rounds},
                   "lightgbm_params": lgb_params,
                   "improvement_over_historical_pct": gain}, f, indent=2, default=float)
    print(f"\nExported {chosen_name} model to {MODEL_PATH.relative_to(ROOT)}; reports in {REPORT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    execute_pipeline()
