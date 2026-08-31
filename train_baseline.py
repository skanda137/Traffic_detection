import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("synthetic_traffic_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['date', 'chokepoint_id', 'time_slot']).reset_index(drop=True)

split_date = pd['date'].min() + pd.Timedelta(days=42)
train_df = df[df['date'] < split_date].copy()
val_df = df[df['date'] >= split_date].copy()

categorial_cols = ['chokepoint_id', 'time_slot']
categorical_features = ['chokepoint_id', 'time_slot', 'day_of_week', 
                        'is_weekend', 'is_festival', 'school_term', 'rainfall_mm', 'visibility_mm']
target = 'congestion_ratio'

for c in categorial_cols:
    train_df[c] = train_df[c].astype('category')
    val_df[c] = val_df[c].astype('category')

X_train = train_df[categorical_features]
y_train = train_df[target]
X_val = val_df[categorical_features]
y_val = val_df[target]

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "verbose": -1,
}

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorial_cols)
val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorial_cols, reference=train_data)

model = lgb.train(params, train_data, num_boost_round=500, valid_sets=[train_data, val_data], 
                  callbacks=[lgb.early_stopping (50), lgb.log_evaluation(50)])

preds = model.predict(X_val, num_iteration=model.best_iteration)
val_df['predicted_congestion_ratio'] = preds
val_df['error'] = np.abs(val_df['target'] - val_df['predicted_congestion_ratio'])

print("\n--- Overall Holdout Performance ---")
print(f"Validation RMSE: {mean_squared_error(y_val, preds, squared=False):.4f}")
print(f"Validation MAE:  {mean_absolute_error(y_val, preds):.4f}")
print(f"Validation R2:   {r2_score(y_val, preds):.4f}")

per_cp_mae = val_df.groupby('chokepoint_id')['error'].mean().sort_values(ascending=False)
print("\nTop 5 Worst-Performing Chokepoints (by MAE):")
print(per_cp_mae.head(5))