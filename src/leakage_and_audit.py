from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("model_ready_traffic.csv")
CLEAN_DATA_PATH = Path("model_cleaned_traffic.parquet")

def run_audit_and_cleanup(data_path: Path = DATA_PATH) -> pd.DataFrame:
    print(f"Loading {data_path}...")
    # Load with optimized types to keep RAM under 1.5 GB
    dtypes = {
        "source_location": "category",
        "destination_location": "category",
        "lat_src": "float32",
        "lon_src": "float32",
        "lat_dest": "float32",
        "lon_dest": "float32",
        "distance": "float32",
        "day_of_week": "int8",
        "hour": "int8",
        "is_peak": "int8",
        "is_weekend": "int8",
        "weather": "category",
        "road_capacity": "float32",
        "vehicles": "float32",
        "speed": "float32",
        "signal_time": "float32",
        "travel_time": "float32",
    }
    
    df = pd.read_csv(data_path, dtype=dtypes)
    print(f"Dataset Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.")

    # 1. Leakage Elimination
    if "speed" in df.columns:
        print("ALERT: Dropping 'speed' to prevent causal target leakage (speed = dist/time).")
        df.drop(columns=["speed"], inplace=True)

    # 2. Bound filtering on travel_time and distance
    # Reject extreme physical outliers
    valid_mask = (df["travel_time"] > 0.5) & (df["travel_time"] < 600.0) & (df["distance"] > 0.1)
    dropped_count = (~valid_mask).sum()
    if dropped_count > 0:
        print(f"Filtered {dropped_count:,} out-of-bound records.")
        df = df[valid_mask].copy()

    # 3. Export to high-speed, compact Parquet format
    df.to_parquet(CLEAN_DATA_PATH, index=False)
    print(f"Cleaned dataset saved to {CLEAN_DATA_PATH} ({CLEAN_DATA_PATH.stat().st_size / (1024**2):.2f} MB)")
    return df

if __name__ == "__main__":
    run_audit_and_cleanup()