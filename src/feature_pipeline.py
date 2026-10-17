from pathlib import Path
import numpy as np
import pandas as pd

CLEAN_DATA_PATH = Path("model_cleaned_traffic.parquet")
FEATURE_DATA_PATH = Path("model_featured_traffic.parquet")

def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance calculation in kilometers."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367.0 * c

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features on 2.1M records...")
    df = df.copy()

    # 1. Spatial Vector Features
    df["haversine_dist"] = haversine_np(
        df["lat_src"].values, df["lon_src"].values,
        df["lat_dest"].values, df["lon_dest"].values
    ).astype("float32")
    
    df["d_lat"] = (df["lat_dest"] - df["lat_src"]).astype("float32")
    df["d_lon"] = (df["lon_dest"] - df["lon_src"]).astype("float32")
    
    # Route Tortuosity (Actual Route Distance vs Euclidean Haversine)
    df["tortuosity"] = (df["distance"] / np.maximum(df["haversine_dist"], 0.05)).clip(0.8, 5.0).astype("float32")

    # 2. Road Density and Congestion Stress Index
    df["density_ratio"] = (df["vehicles"] / np.maximum(df["road_capacity"], 10.0)).astype("float32")
    df["signal_delay_ratio"] = (df["signal_time"] / np.maximum(df["distance"], 0.5)).astype("float32")

    # 3. Cyclical Hour Encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype("float32")

    # 4. Contextual Categorical Encoding
    df["weather"] = df["weather"].astype("category")
    df["day_of_week"] = df["day_of_week"].astype("category")

    # Drop raw IDs that cause overfitting and memory failure
    drop_cols = ["source_location", "destination_location"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df

if __name__ == "__main__":
    raw_df = pd.read_parquet(CLEAN_DATA_PATH)
    featured_df = engineer_features(raw_df)
    featured_df.to_parquet(FEATURE_DATA_PATH, index=False)
    print(f"Features engineered successfully. Output shape: {featured_df.shape}")