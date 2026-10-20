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
ROUTES_PATH = ROOT / "reports" / "example_routes.csv"
ROUTE_FIELDS = ["lat_src", "lon_src", "lat_dest", "lon_dest", "distance"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_resource
def load_service(path):
    return TrafficInferenceService(path, RANGES_PATH)


@st.cache_data
def load_error_ratio(model_name):
    metrics = pd.read_csv(METRICS_PATH).set_index("model")
    return float(metrics.loc[model_name, "mae_over_mean_actual"])


@st.cache_data
def load_routes():
    return pd.read_csv(ROUTES_PATH).set_index("label")


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

routes = load_routes()
route_label = st.selectbox("Route (example routes from the test set)", list(routes.index))
route = routes.loc[route_label]
with st.expander("Custom coordinates"):
    use_custom = st.checkbox("Use custom coordinates instead of the example route")
    col1, col2 = st.columns(2)
    custom = {
        "lat_src": col1.number_input("Source latitude", value=None, format="%.6f"),
        "lon_src": col2.number_input("Source longitude", value=None, format="%.6f"),
        "lat_dest": col1.number_input("Destination latitude", value=None, format="%.6f"),
        "lon_dest": col2.number_input("Destination longitude", value=None, format="%.6f"),
        "distance": st.number_input("Route distance (km)", value=None, step=0.5),
    }
inputs = custom if use_custom else {f: float(route[f]) for f in ROUTE_FIELDS}
day = st.selectbox("Day", service.categories["day_of_week"], format_func=lambda d: DAY_NAMES[d])
hour = st.slider("Hour", 0, 23, 8)
weather = st.selectbox("Weather", service.categories["weather"])
if mode == "Nowcast":
    # Defaults are the selected route's own row values; keyed per route so they reset when the route changes
    inputs["vehicles"] = st.number_input("Vehicles on route", min_value=0, value=int(route["vehicles"]),
                                         step=10, key=f"vehicles_{route_label}")
    inputs["road_capacity"] = st.number_input("Road capacity", min_value=0, value=int(route["road_capacity"]),
                                              step=50, key=f"road_capacity_{route_label}")
    inputs["signal_time"] = st.number_input("Signal time", min_value=0, value=int(route["signal_time"]),
                                            step=10, key=f"signal_time_{route_label}")

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
