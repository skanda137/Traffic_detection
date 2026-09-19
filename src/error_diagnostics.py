import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
from model_comparison import run_temporal_split

def run_diagnostics():
    df = pd.read_csv("synthetic_traffic_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    X_tr, y_tr, X_va, y_va = run_temporal_split(df)
    
    model = joblib.load("best_congestion_model.pkl")
    preds = model.predict(X_va)
    
    eval_df = X_va.copy()
    eval_df["actual"] = y_va
    eval_df["predicted"] = preds
    eval_df["absolute_error"] = np.abs(eval_df["actual"] - eval_df["predicted"])
    eval_df["percentage_error"] = (eval_df["absolute_error"] / eval_df["actual"]) * 100
    
    print("\n--- Diagnostic 1: Top 5 Highest Error Chokepoints ---")
    chokepoint_err = eval_df.groupby("chokepoint_id").agg(
        mae=("absolute_error", "mean"),
        mape=("percentage_error", "mean"),
        sample_count=("actual", "count")
    ).sort_values(by="mae", ascending=False)
    print(chokepoint_err.head(5))

    print("\n--- Diagnostic 2: Contextual Performance Under Stressed Conditions ---")
    rain_slices = eval_df.groupby(eval_df["rainfall_mm"] > 0).agg(
        mae=("absolute_error", "mean"),
        count=("actual", "count")
    ).rename(index={False: "Dry Conditions", True: "Rain Conditions"})
    print(rain_slices)
    
    slot_slices = eval_df.groupby("time_slot")["absolute_error"].mean().sort_values(ascending=False)
    print("\n--- Diagnostic 3: Error by Time Slot ---")
    print(slot_slices)
    
    eval_df.to_csv("validation_diagnostics.csv", index=False)
    print("\nFull diagnostic data exported to validation_diagnostics.csv")

if __name__ == "__main__":
    run_diagnostics()