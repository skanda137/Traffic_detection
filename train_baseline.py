import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Quick untuned LightGBM sanity check; the full tuned comparison lives in src/model_comparison.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model_comparison import load_data, run_temporal_split, CATEGORICAL

df = load_data()
X_train, y_train, X_val, y_val = run_temporal_split(df)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "verbose": -1,
}

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL)
model = lgb.train(params, train_data, num_boost_round=300)

preds = model.predict(X_val)
val_df = X_val.copy()
val_df['predicted_congestion_ratio'] = preds
val_df['error'] = np.abs(y_val.to_numpy() - preds)

print("\n--- Overall Holdout Performance (untuned baseline) ---")
print(f"Validation RMSE: {root_mean_squared_error(y_val, preds):.4f}")
print(f"Validation MAE:  {mean_absolute_error(y_val, preds):.4f}")
print(f"Validation R2:   {r2_score(y_val, preds):.4f}")

per_cp_mae = val_df.groupby('chokepoint_id', observed=True)['error'].mean().sort_values(ascending=False)
print("\nTop 5 Worst-Performing Chokepoints (by MAE):")
print(per_cp_mae.head(5))
