import pandas as pd
import joblib

from model_comparison import MODEL_PATH, apply_categories, to_severity


class TrafficPredictor:
    def __init__(self, model_path=MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_order = bundle["features"]
        self.categories = bundle["categories"]

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
        if chokepoint_id not in self.categories["chokepoint_id"]:
            raise ValueError(f"Unknown chokepoint {chokepoint_id!r}; model was trained on {len(self.categories['chokepoint_id'])} chokepoints")
        if time_slot not in self.categories["time_slot"]:
            raise ValueError(f"Unknown time slot {time_slot!r}; expected one of {self.categories['time_slot']}")

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
        row = apply_categories(row, self.categories)

        predicted_ratio = float(self.model.predict(row)[0])
        # Categorize qualitative severity for commuter UI
        severity = str(to_severity([predicted_ratio])[0])

        return {
            "chokepoint_id": chokepoint_id,
            "predicted_congestion_ratio": round(predicted_ratio, 2),
            "congestion_level": severity,
            "interpretation": f"Travel duration is estimated to be {predicted_ratio:.2f}x free-flow time."
        }


if __name__ == "__main__":
    service = TrafficPredictor()
    print("Sample Dashboard API Responses:")
    for slot, rain, vis in [("evening_peak", 12.5, 4.5), ("evening_peak", 0.0, 10.0), ("night", 0.0, 10.0)]:
        print(service.predict(
            chokepoint_id="CP_001",
            date_str="2026-07-20",
            time_slot=slot,
            rainfall_mm=rain,
            visibility_km=vis,
            is_festival=0
        ))
