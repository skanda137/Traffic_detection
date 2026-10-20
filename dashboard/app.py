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
RANGES_PATH = ROOT / "reports" / "input_ranges.json"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_resource
def load_service(path):
    return TrafficInferenceService(path, RANGES_PATH)


@st.cache_data
def load_error_ratio(model_name):
    metrics = pd.read_csv(METRICS_PATH).set_index("model")
    return float(metrics.loc[model_name, "mae_over_mean_actual"])


def predict_or_stop(service, record):
    """Invalid input -> st.error and no prediction."""
    try:
        return service.predict(record)["predicted_travel_time_minutes"]
    except ValueError as e:
        st.error(f"Invalid input: {e}")
        st.stop()


st.title("Travel Time Predictor")
mode = st.radio("Mode", list(MODES), horizontal=True)
model_path, model_name, mode_label = MODES[mode]
service = load_service(str(model_path))
error_ratio = load_error_ratio(model_name)

col1, col2 = st.columns(2)
inputs = {
    "lat_src": col1.number_input("Source latitude", value=None, format="%.6f"),
    "lon_src": col2.number_input("Source longitude", value=None, format="%.6f"),
    "lat_dest": col1.number_input("Destination latitude", value=None, format="%.6f"),
    "lon_dest": col2.number_input("Destination longitude", value=None, format="%.6f"),
    "distance": st.number_input("Route distance (km)", value=None, step=0.5),
}
day = st.selectbox("Day", service.categories["day_of_week"], format_func=lambda d: DAY_NAMES[d])
hour = st.slider("Hour", 0, 23, 8)
weather = st.selectbox("Weather", service.categories["weather"])
if mode == "Nowcast":
    inputs["vehicles"] = st.number_input("Vehicles on route", min_value=0, value=None, step=10)
    inputs["road_capacity"] = st.number_input("Road capacity", min_value=0, value=None, step=50)
    inputs["signal_time"] = st.number_input("Signal time", min_value=0, value=None, step=10)

missing = [k for k, v in inputs.items() if v is None]
if missing:
    st.error(f"Missing input: {', '.join(missing)}")
    st.stop()
record = {**inputs, "day_of_week": day, "hour": hour, "weather": weather}

minutes = predict_or_stop(service, record)
band = minutes * error_ratio

st.caption(mode_label)
st.metric("Predicted travel time", f"{minutes:.1f} min")
st.write(f"Typical error: ±{band:.1f} min ({max(minutes - band, 0):.1f}–{minutes + band:.1f} min), "
         f"from test-set MAE / mean actual = {error_ratio:.1%}")

hourly = pd.DataFrame({
    "hour": range(24),
    "predicted_minutes": [predict_or_stop(service, {**record, "hour": h}) for h in range(24)],
}).set_index("hour")
st.subheader("Predicted travel time by hour")
st.line_chart(hourly)
