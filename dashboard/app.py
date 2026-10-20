import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate_and_explain import TrafficInferenceService  # noqa: E402

MODES = {
    "Nowcast": (ROOT / "models" / "nowcast_model.pkl", "NOWCAST",
                "Nowcast: uses live traffic inputs (vehicles, road capacity, signal time)"),
    "Forecast": (ROOT / "models" / "forecast_model.pkl", "FORECAST",
                 "Forecast: uses only route, day, hour and weather (no live traffic inputs)"),
}
METRICS_PATH = ROOT / "reports" / "phase2_final_metrics.csv"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_resource
def load_service(path):
    return TrafficInferenceService(path)


@st.cache_data
def load_error_ratio(model_name):
    metrics = pd.read_csv(METRICS_PATH).set_index("model")
    return float(metrics.loc[model_name, "mae_over_mean_actual"])


st.title("Travel Time Predictor")
mode = st.radio("Mode", list(MODES), horizontal=True)
model_path, model_name, mode_label = MODES[mode]
service = load_service(str(model_path))
error_ratio = load_error_ratio(model_name)

# Defaults are the medians of the training rows
col1, col2 = st.columns(2)
lat_src = col1.number_input("Source latitude", value=12.977431, format="%.6f")
lon_src = col2.number_input("Source longitude", value=77.594780, format="%.6f")
lat_dest = col1.number_input("Destination latitude", value=12.977145, format="%.6f")
lon_dest = col2.number_input("Destination longitude", value=77.594833, format="%.6f")
distance = st.number_input("Route distance (km)", min_value=0.1, value=12.6, step=0.5)
day = st.selectbox("Day", service.categories["day_of_week"], format_func=lambda d: DAY_NAMES[d])
hour = st.slider("Hour", 0, 23, 8)
weather = st.selectbox("Weather", service.categories["weather"])

record = {"lat_src": lat_src, "lon_src": lon_src, "lat_dest": lat_dest, "lon_dest": lon_dest,
          "distance": distance, "day_of_week": day, "hour": hour, "weather": weather}
if mode == "Nowcast":
    record["vehicles"] = st.number_input("Vehicles on route", min_value=0, value=814, step=10)
    record["road_capacity"] = st.number_input("Road capacity", min_value=0, value=1500, step=50)
    record["signal_time"] = st.number_input("Signal time", min_value=0, value=50, step=10)

minutes = service.predict(record)["predicted_travel_time_minutes"]
band = minutes * error_ratio

st.caption(mode_label)
st.metric("Predicted travel time", f"{minutes:.1f} min")
st.write(f"Typical error: ±{band:.1f} min ({max(minutes - band, 0):.1f}–{minutes + band:.1f} min), "
         f"from test-set MAE / mean actual = {error_ratio:.1%}")

hourly = pd.DataFrame({
    "hour": range(24),
    "predicted_minutes": [service.predict({**record, "hour": h})["predicted_travel_time_minutes"] for h in range(24)],
}).set_index("hour")
st.subheader("Predicted travel time by hour")
st.line_chart(hourly)
