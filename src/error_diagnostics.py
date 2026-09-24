import pandas as pd
import numpy as np
import joblib

from model_comparison import load_data, run_temporal_split, to_severity, MODEL_PATH, REPORT_DIR


def slice_errors(eval_df, by):
    return eval_df.groupby(by, observed=True).agg(
        mae=("absolute_error", "mean"),
        bias=("residual", "mean"),
        count=("actual", "count")
    )


def run_diagnostics():
    df = load_data()
    X_tr, y_tr, X_va, y_va = run_temporal_split(df)

    bundle = joblib.load(MODEL_PATH)
    preds = bundle["model"].predict(X_va)
    print(f"Diagnosing {bundle['model_name']} on {len(X_va)} holdout rows")

    eval_df = X_va.copy()
    eval_df["actual"] = y_va.to_numpy()
    eval_df["predicted"] = preds
    eval_df["residual"] = eval_df["predicted"] - eval_df["actual"]  # >0 = over-predicts congestion
    eval_df["absolute_error"] = eval_df["residual"].abs()
    eval_df["percentage_error"] = (eval_df["absolute_error"] / eval_df["actual"]) * 100

    print("\n--- Diagnostic 1: Top 5 Highest Error Chokepoints ---")
    chokepoint_err = eval_df.groupby("chokepoint_id", observed=True).agg(
        mae=("absolute_error", "mean"),
        mape=("percentage_error", "mean"),
        sample_count=("actual", "count")
    ).sort_values(by="mae", ascending=False)
    print(chokepoint_err.head(5).to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"(best chokepoint MAE {chokepoint_err['mae'].min():.4f}, "
          f"median {chokepoint_err['mae'].median():.4f}, worst {chokepoint_err['mae'].max():.4f})")

    print("\n--- Diagnostic 2: Contextual Performance Under Stressed Conditions ---")
    rain_slices = slice_errors(eval_df, eval_df["rainfall_mm"] > 0).rename(
        index={False: "Dry Conditions", True: "Rain Conditions"})
    festival_slices = slice_errors(eval_df, eval_df["is_festival"] == 1).rename(
        index={False: "Normal Day", True: "Festival Day"})
    weekend_slices = slice_errors(eval_df, eval_df["is_weekend"] == 1).rename(
        index={False: "Weekday", True: "Weekend"})
    print(pd.concat([rain_slices, festival_slices, weekend_slices]).to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n--- Diagnostic 3: Error by Time Slot ---")
    slot_slices = slice_errors(eval_df, "time_slot").sort_values("mae", ascending=False)
    print(slot_slices.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n--- Diagnostic 4: Severity band confusion (rows = actual, cols = predicted) ---")
    confusion = pd.crosstab(to_severity(eval_df["actual"]), to_severity(eval_df["predicted"]),
                            rownames=["actual"], colnames=["predicted"])
    print(confusion)

    REPORT_DIR.mkdir(exist_ok=True)
    chokepoint_err.to_csv(REPORT_DIR / "error_by_chokepoint.csv", float_format="%.4f")
    slot_slices.to_csv(REPORT_DIR / "error_by_time_slot.csv", float_format="%.4f")
    eval_df.to_csv(REPORT_DIR / "validation_diagnostics.csv", index=False)
    print(f"\nFull diagnostic data exported to {REPORT_DIR.name}/")


if __name__ == "__main__":
    run_diagnostics()
