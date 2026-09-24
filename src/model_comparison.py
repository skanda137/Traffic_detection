import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_temporal_split(df, split_days=35):
    df = df.sort_values(by=["date", "time_slot"]).reset_index(drop=True)
    split_date = df["date"].min() + pd.Timedelta(days=split_days)
    
    train_df = df[df["date"] < split_date].copy()
    val_df = df[df["date"] >= split_date].copy()
    
    features = [
        "chokepoint_id", "time_slot", "day_of_week",
        "is_weekend", "is_festival", "school_term",
        "rainfall_mm", "visibility_km"
    ]
    target = "congestion_ratio"
    
    # Categorical handling for GBDT
    for col in ["chokepoint_id", "time_slot"]:
        train_df[col] = train_df[col].astype("category")
        val_df[col] = val_df[col].astype("category")
        
    return train_df[features], train_df[target], val_df[features], val_df[target]

def objective(trial, X_tr, y_tr, X_va, y_va):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    preds = model.predict(X_va)
    return mean_squared_error(y_va, preds, squared=False)

def execute_pipeline():
    df = pd.read_csv("synthetic_traffic_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    X_tr, y_tr, X_va, y_va = run_temporal_split(df)
    
    print("--- 1. Hyperparameter Tuning with Optuna ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_va, y_va), n_trials=25)
    print(f"Best LGBM RMSE: {study.best_value:.4f}")
    
    # Train Best LightGBM
    best_lgb_params = study.best_params
    best_lgb_params.update({"objective": "regression", "metric": "rmse", "verbosity": -1})
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
    
    lgb_model = lgb.train(
        best_lgb_params,
        train_data,
        num_boost_round=600,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)]
    )
    lgb_preds = lgb_model.predict(X_va)
    
    print("\n--- 2. Training XGBoost Comparison ---")
    # Native category support via enable_categorical
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        early_stopping_rounds=40,
        random_state=42
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_preds = xgb_model.predict(X_va)
    
    print("\n================ BENCHMARK RESULTS ================")
    print(f"LightGBM -> RMSE: {mean_squared_error(y_va, lgb_preds, squared=False):.4f} | MAE: {mean_absolute_error(y_va, lgb_preds):.4f} | R2: {r2_score(y_va, lgb_preds):.4f}")
    print(f"XGBoost  -> RMSE: {mean_squared_error(y_va, xgb_preds, squared=False):.4f} | MAE: {mean_absolute_error(y_va, xgb_preds):.4f} | R2: {r2_score(y_va, xgb_preds):.4f}")
    print("===================================================")
    
    # Save best performing model (LightGBM)
    joblib.dump(lgb_model, "best_congestion_model.pkl")
    print("Exported best model to best_congestion_model.pkl")

if __name__ == "__main__":
    execute_pipeline()