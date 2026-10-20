import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feature_pipeline import haversine_np, raw_tortuosity

ROOT = Path(__file__).resolve().parent.parent
AUDIT_PATH = ROOT / "reports" / "data_audit.md"
SAMPLE_ROWS = 100_000
SAMPLE_SEED = 0
DATETIME_NAME_HINTS = ("date", "time", "stamp")


def md_table(df: pd.DataFrame, float_format="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(float_format.format(v) if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def datetime_columns(df: pd.DataFrame) -> list:
    """Columns whose values parse as datetimes (>90% of the first 1000 non-null values)."""
    found = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            found.append(col)
        elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            head = df[col].dropna().head(1000)
            if len(head) and pd.to_datetime(head, errors="coerce", format="mixed").notna().mean() > 0.9:
                found.append(col)
    return found


def run_audit(raw_csv_path: Path) -> str:
    if not raw_csv_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    n = len(df)
    out = [f"# Data audit\n", f"Source file: `{raw_csv_path.name}` (measured facts only; no claim about where the data came from)\n"]

    out.append("## Shape\n")
    out.append(f"- Rows: {n:,}\n- Columns: {df.shape[1]}\n- Column names: {', '.join(f'`{c}`' for c in df.columns)}\n")

    dt_cols = datetime_columns(df)
    named = [c for c in df.columns if any(h in c.lower() for h in DATETIME_NAME_HINTS)]
    out.append("## Date/time columns\n")
    out.append(f"- Columns whose values parse as datetimes: {', '.join(dt_cols) if dt_cols else 'none'}")
    out.append(f"- Columns whose name contains {'/'.join(DATETIME_NAME_HINTS)}: {', '.join(f'`{c}`' for c in named) if named else 'none'}\n")

    pairs = df.groupby(["source_location", "destination_location"]).ngroups
    out.append("## Routes\n")
    out.append(f"- Distinct (source_location, destination_location) pairs: {pairs:,}")
    out.append(f"- Distinct pairs / rows: {pairs / n:.4f}\n")

    geo = pd.DataFrame({"distance": df["distance"],
                        "haversine_dist": haversine_np(df["lat_src"].values, df["lon_src"].values,
                                                       df["lat_dest"].values, df["lon_dest"].values)})
    tort = raw_tortuosity(geo).replace([np.inf, -np.inf], np.nan)
    out.append("## Raw tortuosity (distance / straight-line distance, unclipped)\n")
    out.append(f"- Rows with a finite value: {tort.notna().sum():,} of {n:,} (float64 on the raw CSV)\n")
    out.append(md_table(pd.DataFrame({"percentile": ["p1", "p50", "p99"],
                                      "raw_tortuosity": [float(tort.quantile(q)) for q in (0.01, 0.50, 0.99)]}), "{:.5f}") + "\n")

    pos = df[df["distance"] > 0]
    pace = (pos["travel_time"] / pos["distance"]).groupby(pos["hour"]).median()
    out.append("## Median minutes per km by hour (rows with distance > 0)\n")
    out.append(md_table(pd.DataFrame({"hour": pace.index.astype(int), "median_min_per_km": pace.values.astype(float)})) + "\n")

    weather = df.groupby("weather")["travel_time"].mean()
    if "Clear" not in weather.index:
        raise ValueError(f"No 'Clear' weather rows to compare against; weather values: {list(weather.index)}")
    out.append("## Mean travel_time by weather\n")
    out.append(md_table(pd.DataFrame({"weather": weather.index, "mean_travel_time_min": weather.values.astype(float),
                                      "ratio_to_Clear": (weather / weather["Clear"]).values.astype(float)})) + "\n")

    sample = df.sample(SAMPLE_ROWS, random_state=SAMPLE_SEED)
    s = sample[sample["speed"] > 0]
    r2 = float(np.corrcoef(s["distance"] / s["speed"], s["travel_time"])[0, 1] ** 2)
    c = sample[sample["road_capacity"] > 0]
    corr = float(c["speed"].corr(c["vehicles"] / c["road_capacity"]))
    out.append(f"## Target structure ({SAMPLE_ROWS:,}-row sample, seed {SAMPLE_SEED})\n")
    out.append(f"- R² of travel_time vs distance/speed (squared Pearson correlation, {len(s):,} rows with speed > 0): {r2:.4f}")
    out.append(f"- Pearson corr(speed, vehicles/road_capacity) ({len(c):,} rows with road_capacity > 0): {corr:.4f}\n")
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write reports/data_audit.md with measured facts about the raw CSV.")
    parser.add_argument("raw_csv_path", type=Path, help="path to model_ready_traffic.csv (required)")
    args = parser.parse_args()
    report = run_audit(args.raw_csv_path)
    AUDIT_PATH.parent.mkdir(exist_ok=True)
    AUDIT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {AUDIT_PATH.relative_to(ROOT)}")
