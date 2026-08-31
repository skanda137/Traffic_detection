import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from datetime import datetime, timedelta

np.random.seed(42)
N_chokepoints = 75
DAYS = 56
TIME_SLOTS = ['morning', 'mid-day', 'evening', 'night']
BASE_DATE = datetime(2026, 1, 1)

records = []
chokepoints_ids = [f"CP_{i:03d}" for i in range(1, N_chokepoints+1)]
chokepoint_base = {cp: np.random.uniform(1.1,1.8) for cp in chokepoints_ids}

for day in range(DAYS):
    current_date = BASE_DATE + timedelta(days=day)
    dow = current_date.date.weekday()  # Monday=0, Sunday=6
    is_weekend = int(dow >= 5)
    is_festival = int(np.random.rand() < 0.05)
    school_term = int(dow < 5 and not is_festival)  

    daily_rain = 0.50 if current_date.month in [6, 7, 8] else 0.10
    has_rain = np.random().rand() < daily_rain

    for slot in TIME_SLOTS:
        rainfall_mm = np.random.exponential(scale=8.0) if has_rain else 0.0
        visibility_mm = max(1.0, 10.0 - (rainfall_mm / 4.0) + np.random.normal(0, 0.5)) 
        for cp in chokepoints_ids:
            base = chokepoint_base[cp]
            slot_mult = {
                "morning_peak": 1.55 if not is_weekend  else 1.10,
                "mid-day": 1.10,
                "evening_peak": 1.65 if not is_weekend  else 1.20,
                "night": 0.90
            }[slot]

            rain_mult = 1.0 + min((0.50, rainfall_mm/20.0)*0.35)
            festival_mult = 1.25 if is_festival and slot in ["mid-day", "evening_peak"] else 1.0
            mu = base * slot_mult * rain_mult * festival_mult
            noise = np.random.normal(0, 0.10)
            congestion_ratio = max(0.1, mu+noise)

            records.append({
                "timestamp": current_date.strftime("%Y-%m-%d %H:%M:%S") + timedelta(hours={"morning": 8, "mid-day": 13, "evening": 17, "night": 22}[slot]),
                "date": current_date.strftime("%Y-%m-%d"),
                "chokepoint_id": cp,
                "time_slot": slot,
                "day_of_week": dow,
                "is_weekend": is_weekend,
                "is_festival": is_festival,
                "school_term": school_term,
                "rainfall_mm": round(rainfall_mm, 2),
                "visibility_mm": round(visibility_mm, 2),
                "congestion_ratio": round(congestion_ratio, 4)
            })
df = pd.DataFrame(records)
df.to_csv("synthetic_traffic_data.csv", index=False)
print("Synthetic traffic data generated and saved to 'synthetic_traffic_data.csv'.")
print(f"Total records generated: {len(df)}")