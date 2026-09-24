import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {
    "chokepoint_id": object,
    "date": object,
    "time_slot": object,
    "day_of_week": int,
    "is_weekend": int,
    "is_festival": int,
    "school_term": int,
    "rainfall_mm": float,
    "visibility_km": float,
    "congestion_ratio": float
}

VALID_SLOTS = {"morning_peak", "midday", "evening_peak", "night"}

def validate_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asserts schema fidelity, bounds-checks metrics, and prepares raw collector data.
    """
    df = df.copy()
    
    # 1. Check missing columns
    missing_cols = set(REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema violation: missing required columns {missing_cols}")

    # 2. Enforce types and non-null constraints
    for col, dtype in REQUIRED_COLUMNS.items():
        if col in ["rainfall_mm", "visibility_km", "congestion_ratio"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col in ["day_of_week", "is_weekend", "is_festival", "school_term"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if df[list(REQUIRED_COLUMNS.keys())].isnull().any().any():
        null_counts = df[list(REQUIRED_COLUMNS.keys())].isnull().sum()
        raise ValueError(f"Data contains unhandled NaNs:\n{null_counts[null_counts > 0]}")
    for col in ["day_of_week", "is_weekend", "is_festival", "school_term"]:
        df[col] = df[col].astype(int)

    # 3. Domain Logic Validations
    invalid_slots = set(df["time_slot"].unique()) - VALID_SLOTS
    if invalid_slots:
        raise ValueError(f"Invalid time slots found: {invalid_slots}")

    # Congestion ratio must be >= 1.0 (traffic duration >= free-flow duration)
    invalid_ratios = (df["congestion_ratio"] < 0.95).sum()
    if invalid_ratios > 0:
        raise ValueError(f"Found {invalid_ratios} records with congestion_ratio < 0.95. Target computation flawed.")

    # Convert dates
    df["date"] = pd.to_datetime(df["date"])
    return df

if __name__ == "__main__":
    test_df = pd.read_csv("synthetic_traffic_dataset.csv")
    cleaned_df = validate_traffic_data(test_df)
    print(f"Dataset successfully validated: {len(cleaned_df)} clean rows.")