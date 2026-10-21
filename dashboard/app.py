import math
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate_and_explain import TrafficInferenceService  # noqa: E402
from places import PLACES, RANGES, route_from_places  # noqa: E402

MODEL_PATHS = {"FORECAST": ROOT / "models" / "forecast_model.pkl", "NOWCAST": ROOT / "models" / "nowcast_model.pkl"}
METRICS_PATH = ROOT / "reports" / "phase2_final_metrics.csv"
QUANTILES_PATH = ROOT / "reports" / "phase2_error_quantiles.csv"
RANGES_PATH = ROOT / "reports" / "input_ranges.json"
ROUTES_PATH = ROOT / "reports" / "example_routes.csv"
AUDIT_PATH = ROOT / "reports" / "data_audit.md"
PROVENANCE_PATH = ROOT / "docs" / "provenance.txt"
SYNOPSIS_PATH = ROOT / "docs" / "synopsis_vs_built.md"
PROVENANCE_PLACEHOLDER = "<Member A to fill>"
BASELINE_A = "baseline (a) per-hour median pace x distance"
ROUTE_FIELDS = ["lat_src", "lon_src", "lat_dest", "lon_dest", "distance"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# Same meaning, same colour in every chart
BEST, CHOSEN, WORST, OTHER = "#2ca02c", "#1f77b4", "#ff7f0e", "#9aa5b1"
HOUR_COLORS = {"Best window": BEST, "Chosen hour": CHOSEN, "Worst window": WORST, "Other hours": OTHER}
WINDOW_TOLERANCE = 0.03  # hours within 3% of the minimum (maximum) form the best (worst) window
DAY_EFFECT_TOLERANCE = 0.03
VEHICLE_POINTS = 40
LIMITATIONS = [
    "Random train/val/test split: the dataset has no timestamp, so there is no out-of-time evidence.",
    "Routes are chosen by place name; distance is the straight-line distance between places.",
    "Live traffic (Nowcast) needs live vehicle counts; Plan ahead (Forecast) is the ahead-of-time estimate.",
    "The 80% range is empirical and marginal over the whole test set, not a per-input confidence interval.",
    "Phase 1 (congestion ratio) is synopsis design, implemented on synthetic data.",
]

st.set_page_config(page_title="Travel Time Predictor", layout="wide")


@st.cache_resource
def load_service(model_name):
    return TrafficInferenceService(MODEL_PATHS[model_name], RANGES_PATH)


@st.cache_data
def cached_predict_many(model_name, columns, rows):
    """columns/rows are tuples so the call is hashable. ValueError on invalid input is raised, never cached."""
    return load_service(model_name).predict_many(pd.DataFrame(list(rows), columns=list(columns))).tolist()


def predict_frame(model_name, frame):
    rows = tuple(tuple(r) for r in frame.itertuples(index=False))
    return np.asarray(cached_predict_many(model_name, tuple(frame.columns), rows))


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_data
def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def error_quantiles(model_name):
    q = read_csv(QUANTILES_PATH).set_index("model").loc[model_name]
    return float(q["q10_actual_over_predicted"]), float(q["q90_actual_over_predicted"])


def format_hour_window(hours):
    """Hours (0-23) -> consecutive runs as "HH:00-HH:00", merging across midnight; wrapping run listed first."""
    hours = set(hours)
    if len(hours) == 24:
        return "00:00-23:00"
    runs = []
    for start in sorted(h for h in hours if (h - 1) % 24 not in hours):
        end = start
        while (end + 1) % 24 in hours:
            end = (end + 1) % 24
        runs.append((start - 24 if end < start else start, start, end))
    return ", ".join(f"{s:02d}:00" if s == e else f"{s:02d}:00-{e:02d}:00" for _, s, e in sorted(runs))


def range_caption(minutes, model_name):
    q10, q90 = error_quantiles(model_name)
    return f"80% of test-set trips fell within [{minutes * q10:.1f}, {minutes * q90:.1f}] min"


def render_plan(record):
    service = load_service("FORECAST")
    hour, weather = record["hour"], record["weather"]
    minutes = service.predict(record)["predicted_travel_time_minutes"]  # raises ValueError on invalid input
    st.session_state["headline_minutes"] = minutes  # exact value, read by AppTest

    # One batch call: 24 hours x every weather; the chosen weather's 24 rows also feed the bar chart
    grid = pd.DataFrame([{**record, "hour": h, "weather": w} for w in service.categories["weather"] for h in range(24)])
    grid["minutes"] = predict_frame("FORECAST", grid[service.raw_inputs])
    sweep = grid.loc[grid["weather"] == weather, ["hour", "minutes"]].reset_index(drop=True)
    low, high = sweep["minutes"].min(), sweep["minutes"].max()
    best_hours = sweep.loc[sweep["minutes"] <= low * (1 + WINDOW_TOLERANCE), "hour"].tolist()
    worst_hours = sweep.loc[sweep["minutes"] >= high * (1 - WINDOW_TOLERANCE), "hour"].tolist()

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Predicted at {hour:02d}:00", f"{minutes:.1f} min")
    c1.caption(range_caption(minutes, "FORECAST"))
    c2.metric("Best window", f"{low:.1f} min")
    c2.caption(format_hour_window(best_hours))
    c3.metric("Minutes saved vs the chosen hour", f"{minutes - low:.1f} min")
    c3.caption("by leaving in the best window instead")

    st.markdown(f"**Best window:** {format_hour_window(best_hours)} (minimum {low:.1f} min)  \n"
                f"**Worst window:** {format_hour_window(worst_hours)} (maximum {high:.1f} min)")
    st.caption("Hours within 3% of the minimum are treated as equivalent.")

    sweep["highlight"] = "Other hours"
    sweep.loc[sweep["hour"].isin(worst_hours), "highlight"] = "Worst window"
    sweep.loc[sweep["hour"].isin(best_hours), "highlight"] = "Best window"
    sweep.loc[sweep["hour"] == hour, "highlight"] = "Chosen hour"
    st.subheader("Predicted minutes by departure hour")
    st.altair_chart(alt.Chart(sweep).mark_bar().encode(
        x=alt.X("hour:O", title="Hour"),
        y=alt.Y("minutes:Q", title="Predicted minutes"),
        color=alt.Color("highlight:N", title=None,
                        scale=alt.Scale(domain=list(HOUR_COLORS), range=list(HOUR_COLORS.values()))),
        tooltip=["hour", alt.Tooltip("minutes:Q", format=".2f")],
    ), width="stretch")

    st.subheader(f"Hour x weather on {DAY_NAMES[record['day_of_week']]}")
    cells = alt.Chart(grid[["hour", "weather", "minutes"]]).mark_rect().encode(
        x=alt.X("hour:O", title="Hour"), y=alt.Y("weather:N", title=None),
        color=alt.Color("minutes:Q", title="Predicted minutes", scale=alt.Scale(scheme="greys")),
        tooltip=["hour", "weather", alt.Tooltip("minutes:Q", format=".2f")])
    chosen = alt.Chart(pd.DataFrame({"hour": [hour], "weather": [weather]})).mark_rect(
        fill=None, stroke=CHOSEN, strokeWidth=3).encode(x="hour:O", y="weather:N")
    st.altair_chart(cells + chosen, width="stretch")

    days = pd.DataFrame([{**record, "day_of_week": d} for d in service.categories["day_of_week"]])
    by_day = predict_frame("FORECAST", days[service.raw_inputs])
    spread = (by_day.max() - by_day.min()) / by_day.min()
    if spread < DAY_EFFECT_TOLERANCE:
        st.caption("Day of week changes the prediction by under 3% on this dataset")
    else:
        st.caption(f"Day of week changes the prediction by {spread:.1%} at {hour:02d}:00 in {weather} weather "
                   f"({by_day.min():.1f}-{by_day.max():.1f} min)")
    return minutes


def render_live(record, route, route_label, forecast_minutes):
    service = load_service("NOWCAST")
    c1, c2, c3 = st.columns(3)
    # Defaults are the selected route's own row values; keyed per route so they reset when the route changes
    traffic = {
        "vehicles": c1.number_input("Vehicles on route", min_value=0, value=int(route["vehicles"]), step=10,
                                    key=f"vehicles_{route_label}"),
        "road_capacity": c2.number_input("Road capacity", min_value=0, value=int(route["road_capacity"]), step=50,
                                         key=f"road_capacity_{route_label}"),
        "signal_time": c3.number_input("Signal time", min_value=0, value=int(route["signal_time"]), step=10,
                                       key=f"signal_time_{route_label}"),
    }
    live = {**record, **traffic}
    try:
        minutes = service.predict(live)["predicted_travel_time_minutes"]
    except ValueError as e:
        st.error(f"Invalid input: {e}")
        return
    st.session_state["live_minutes"] = minutes  # exact value, read by AppTest

    st.metric(f"Live estimate at {record['hour']:02d}:00", f"{minutes:.1f} min")
    st.caption(range_caption(minutes, "NOWCAST"))
    st.write(f"Forecast for the same route/hour says {forecast_minutes:.1f} min, live says {minutes:.1f} min "
             f"(difference {minutes - forecast_minutes:+.1f} min)")

    density = service.ranges["density_ratio"]
    current_density = float(service.featurize(pd.DataFrame([live]))["density_ratio"].iloc[0])
    st.write(f"density_ratio (vehicles / road capacity) = {current_density:.3f}; "
             f"training p1-p99: [{density['p1']:.3f}, {density['p99']:.3f}]")

    # Vehicle counts spanning the training density p1-p99, using feature_pipeline's density for one vehicle
    per_vehicle = float(service.featurize(pd.DataFrame([{**live, "vehicles": 1}]))["density_ratio"].iloc[0])
    low, high = math.ceil(density["p1"] / per_vehicle), math.floor(density["p99"] / per_vehicle)
    vehicles = sorted(set(np.linspace(low, high, VEHICLE_POINTS).round().astype(int).tolist()) | {traffic["vehicles"]})
    grid = pd.DataFrame([{**live, "vehicles": v} for v in vehicles])
    sweep = pd.DataFrame({"vehicles": vehicles, "minutes": predict_frame("NOWCAST", grid[service.raw_inputs])})

    st.subheader(f"Predicted minutes vs vehicles (road capacity {traffic['road_capacity']})")
    line = alt.Chart(sweep).mark_line(color=OTHER).encode(
        x=alt.X("vehicles:Q", title="Vehicles on route"), y=alt.Y("minutes:Q", title="Predicted minutes"),
        tooltip=["vehicles", alt.Tooltip("minutes:Q", format=".2f")])
    current = alt.Chart(sweep[sweep["vehicles"] == traffic["vehicles"]]).mark_point(
        color=CHOSEN, size=140, filled=True).encode(x="vehicles:Q", y="minutes:Q")
    st.altair_chart(line + current, width="stretch")


def render_about():
    metrics = read_csv(METRICS_PATH).set_index("model")
    if BASELINE_A not in metrics.index:
        raise ValueError(f"{BASELINE_A!r} missing from {METRICS_PATH.name}")
    base = metrics.loc[BASELINE_A]
    table = metrics[["rmse_min", "mae_min", "mae_over_mean_actual"]].copy()
    table["rmse_improvement_vs_baseline_a_pct"] = 100 * (1 - metrics["rmse_min"] / base["rmse_min"])
    table["mae_improvement_vs_baseline_a_pct"] = 100 * (1 - metrics["mae_min"] / base["mae_min"])
    st.subheader("Test-set metrics")
    st.dataframe(table.round(4))
    st.write("Split: random held-out split; dataset has no timestamp.")
    st.caption(f"Details: {load_service('FORECAST').split_description}")

    st.subheader("Dataset provenance")
    provenance = read_text(PROVENANCE_PATH).strip()
    if PROVENANCE_PLACEHOLDER in provenance:
        st.error(provenance)
    else:
        st.info(provenance)

    st.subheader("Synopsis vs what was built")
    st.markdown(read_text(SYNOPSIS_PATH))
    with st.expander("Data audit (reports/data_audit.md)"):
        st.markdown(read_text(AUDIT_PATH))
    st.subheader("Limitations")
    st.markdown("\n".join(f"- {item}" for item in LIMITATIONS))


st.session_state.pop("headline_minutes", None)  # never carry a value over from a previous run
st.session_state.pop("live_minutes", None)
st.title("Travel Time Predictor")
st.write("Estimate trip minutes for a route by departure hour, ahead of time or from live traffic counts.")
st.caption("Predictions reflect the supplied dataset, not live traffic.")

forecast_service = load_service("FORECAST")
routes = read_csv(ROUTES_PATH).set_index("label")
with st.sidebar:
    st.header("Trip")
    route_mode = st.radio("Route", ["Between places", "Dataset example routes"])
    place_names = sorted(PLACES)
    route_error = None
    if route_mode == "Between places":
        src = st.selectbox("From", place_names, index=place_names.index("Silk Board"))
        dst = st.selectbox("To", place_names, index=place_names.index("Hebbal"))
        route_label = f"{src} -> {dst}"
        # Live-traffic defaults for place routes: training medians
        route = {f: RANGES[f"{f}_median"] for f in ["vehicles", "road_capacity", "signal_time"]}
        try:
            inputs = route_from_places(src, dst)
            st.caption("Distance is the straight-line distance between the two places (the dataset's definition), "
                       f"{inputs['distance']:.2f} km.")
        except ValueError as e:
            inputs, route_error = None, str(e)
    else:
        route_label = st.selectbox("Route (example routes from the test set)", list(routes.index))
        route = routes.loc[route_label]
        inputs = {f: float(route[f]) for f in ROUTE_FIELDS}
    day = st.selectbox("Day", forecast_service.categories["day_of_week"], format_func=lambda d: DAY_NAMES[d])
    hour = st.slider("Hour", 0, 23, 8)
    weather = st.selectbox("Weather", forecast_service.categories["weather"])

record = {**(inputs or {}), "day_of_week": day, "hour": hour, "weather": weather}

render_seconds = {}
plan_tab, live_tab, about_tab = st.tabs(["Plan ahead", "Live traffic", "About"])
with about_tab:
    start = time.perf_counter()
    render_about()
    render_seconds["About"] = time.perf_counter() - start

forecast_minutes, input_error = None, None
with plan_tab:
    start = time.perf_counter()
    if route_error:
        input_error = f"Invalid input: {route_error}"
    else:
        try:
            forecast_minutes = render_plan(record)
        except ValueError as e:
            input_error = f"Invalid input: {e}"
    if input_error:
        st.error(input_error)
    render_seconds["Plan ahead"] = time.perf_counter() - start
with live_tab:
    start = time.perf_counter()
    st.caption("Needs live vehicle counts from a sensor feed.")
    if input_error:
        st.error(input_error)
    else:
        render_live(record, route, route_label, forecast_minutes)
    render_seconds["Live traffic"] = time.perf_counter() - start
st.session_state["render_seconds"] = render_seconds  # read by AppTest
