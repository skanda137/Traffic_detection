import pandas as pd
import numpy as np
import joblib

class TrafficPredictor:
    def __init__(self, model_path="best_congestion_model.pkl"):
        self.model = joblib.load(model_path)
        self.feature_order = [
            "chokepoint_id", "time_slot", "day_of_week",
            "is_weekend", "is_festival", "school_term",
            "rainfall_mm", "visibility_km"
        ]

    def predict(self, 
                chokepoint_id: str, 
                date_str: str, 
                time_slot: str, 
                rainfall_mm: float = 0.0, 
                visibility_km: float = 10.0,
                is_festival: int = 0) -> dict:
        """
        Inference entry point for downstream dashboard (Member D).
        """
        date = pd.to_datetime(date_str)
        dow = date.weekday()
        is_weekend = int(dow >= 5)
        school_term = int(dow < 5 and not is_festival)
        
        row = pd.DataFrame([{
            "chokepoint_id": str(chokepoint_id),
            "time_slot": str(time_slot),
            "day_of_week": int(dow),
            "is_weekend": int(is_weekend),
            "is_festival": int(is_festival),
            "school_term": int(school_term),
            "rainfall_mm": float(rainfall_mm),
            "visibility_km": float(visibility_km)
        }])[self.feature_order]

        for col in ["chokepoint_id", "time_slot"]:
            row[col] = row[col].astype("category")

        predicted_ratio = float(self.model.predict(row)[0])
        
        # Categorize qualitative severity for commuter UI
        if predicted_ratio < 1.25:
            severity = "Low"
        elif predicted_ratio < 1.60:
            severity = "Moderate"
        else:
            severity = "Severe"
            
        return {
            "chokepoint_id": chokepoint_id,
            "predicted_congestion_ratio": round(predicted_ratio, 2),
            "congestion_level": severity,
            "interpretation": f"Travel duration is estimated to be {predicted_ratio:.2f}x free-flow time."
        }

if __name__ == "__main__":
    service = TrafficPredictor()
    sample_prediction = service.predict(
        chokepoint_id="CP_001",
        date_str="2026-07-20",
        time_slot="evening_peak",
        rainfall_mm=12.5,
        visibility_km=4.5,
        is_festival=0
    )
    print("Sample Dashboard API Response:")
    print(sample_prediction)