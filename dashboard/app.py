import math
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate_and_explain import TrafficInferenceService  # noqa: E402

MODES = {
    "Forecast": (ROOT / "models" / "forecast_model.pkl", "FORECAST",
                 "Forecast: uses only route, day, hour and weather (no live traffic inputs)"),
    "Nowcast": (ROOT / "models" / "nowcast_model.pkl", "NOWCAST",
                "Needs live vehicle counts from a sensor feed."),
}
METRICS_PATH = ROOT / "reports" / "phase2_final_metrics.csv"
QUANTILES_PATH = ROOT / "reports" / "phase2_error_quantiles.csv"
RANGES_PATH = ROOT / "reports" / "input_ranges.json"
ROUTES_PATH = ROOT / "reports" / "example_routes.csv"
ROUTE_FIELDS = ["lat_src", "lon_src", "lat_dest", "lon_dest", "distance"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOUR_COLORS = {"Other hours": "#9aa5b1", "Chosen hour": "#1f77b4", "Best hour": "#2ca02c",
               "Chosen = best hour": "#9467bd"}
VEHICLE_POINTS = 40


@st.cache_resource
def load_service(path):
    return TrafficInferenceService(path, RANGES_PATH)


@st.cache_data
def load_error_quantiles(model_name):
    q = pd.read_csv(QUANTILES_PATH).set_index("model").loc[model_name]
    return float(q["q10_actual_over_predicted"]), float(q["q90_actual_over_predicted"])


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


def show_interval(minutes, quantiles):
    q10, q90 = quantiles
    st.write(f"80% of test-set trips fell within [{minutes * q10:.1f}, {minutes * q90:.1f}] min "
             f"(prediction × {q10:.3f} to × {q90:.3f}; empirical, marginal over the test set).")


def render_forecast(service, record, quantiles):
    hour = record["hour"]
    minutes = predict_or_stop(service, record)
    sweep = pd.DataFrame({"hour": range(24),
                          "minutes": [predict_or_stop(service, {**record, "hour": h}) for h in range(24)]})
    best = sweep.loc[sweep["minutes"].idxmin()]
    worst = sweep.loc[sweep["minutes"].idxmax()]
    best_hour, worst_hour = int(best["hour"]), int(worst["hour"])

    st.metric(f"Predicted travel time at {hour:02d}:00", f"{minutes:.1f} min")
    show_interval(minutes, quantiles)
    c1, c2, c3 = st.columns(3)
    c1.metric("Best hour", f"{best_hour:02d}:00", f"{best['minutes']:.1f} min", delta_color="off")
    c2.metric("Worst hour", f"{worst_hour:02d}:00", f"{worst['minutes']:.1f} min", delta_color="off")
    c3.metric(f"Saved by leaving at {best_hour:02d}:00", f"{minutes - best['minutes']:.1f} min")

    sweep["highlight"] = "Other hours"
    sweep.loc[sweep["hour"] == hour, "highlight"] = "Chosen hour"
    sweep.loc[sweep["hour"] == best_hour, "highlight"] = "Chosen = best hour" if best_hour == hour else "Best hour"
    st.subheader("Predicted travel time by departure hour")
    st.altair_chart(alt.Chart(sweep).mark_bar().encode(
        x=alt.X("hour:O", title="Hour"),
        y=alt.Y("minutes:Q", title="Predicted minutes"),
        color=alt.Color("highlight:N", title=None,
                        scale=alt.Scale(domain=list(HOUR_COLORS), range=list(HOUR_COLORS.values()))),
        tooltip=["hour", alt.Tooltip("minutes:Q", format=".2f")],
    ), width="stretch")

    st.subheader(f"Same route at {hour:02d}:00 by weather")
    st.table(pd.DataFrame({
        "weather": service.categories["weather"],
        "predicted minutes": [round(predict_or_stop(service, {**record, "weather": w}), 2)
                              for w in service.categories["weather"]],
    }).set_index("weather"))


def render_nowcast(service, record, quantiles):
    minutes = predict_or_stop(service, record)
    st.metric(f"Predicted travel time at {record['hour']:02d}:00", f"{minutes:.1f} min")
    show_interval(minutes, quantiles)

    # Vehicle counts spanning the training density_ratio p1-p99 at this road capacity (same formula as feature_pipeline)
    density = service.ranges["density_ratio"]
    capacity = max(record["road_capacity"], 10.0)
    low, high = math.ceil(density["p1"] * capacity), math.floor(density["p99"] * capacity)
    vehicles = sorted(set(np.linspace(low, high, VEHICLE_POINTS).round().astype(int).tolist()) | {record["vehicles"]})
    sweep = pd.DataFrame({"vehicles": vehicles,
                          "minutes": [predict_or_stop(service, {**record, "vehicles": v}) for v in vehicles]})
    sweep["point"] = np.where(sweep["vehicles"] == record["vehicles"], "Current input", "Sweep")

    st.subheader(f"Predicted travel time vs vehicles (road capacity {record['road_capacity']})")
    line = alt.Chart(sweep).mark_line(color="#1f77b4").encode(
        x=alt.X("vehicles:Q", title="Vehicles on route"),
        y=alt.Y("minutes:Q", title="Predicted minutes"),
        tooltip=["vehicles", alt.Tooltip("minutes:Q", format=".2f")])
    current = alt.Chart(sweep[sweep["point"] == "Current input"]).mark_point(
        color="#d62728", size=120, filled=True).encode(x="vehicles:Q", y="minutes:Q")
    st.altair_chart(line + current, width="stretch")


st.title("Travel Time Predictor")
mode = st.radio("Mode", list(MODES), horizontal=True)
model_path, model_name, mode_label = MODES[mode]
service = load_service(str(model_path))
quantiles = load_error_quantiles(model_name)
st.caption(mode_label)

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

with st.expander("Model card"):
    st.dataframe(pd.read_csv(METRICS_PATH).set_index("model").round(4))
    st.write("Split: random held-out split; dataset has no timestamp.")
    st.write("Predictions reflect the training dataset, not live traffic.")

if mode == "Forecast":
    render_forecast(service, record, quantiles)
else:
    render_nowcast(service, record, quantiles)
