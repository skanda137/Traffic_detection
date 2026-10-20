# Data audit

Source file: `model_ready_traffic.csv` (measured facts only; no claim about where the data came from)

## Shape

- Rows: 2,096,563
- Columns: 17
- Column names: `source_location`, `destination_location`, `lat_src`, `lon_src`, `lat_dest`, `lon_dest`, `distance`, `day_of_week`, `hour`, `is_peak`, `is_weekend`, `weather`, `road_capacity`, `vehicles`, `speed`, `signal_time`, `travel_time`

## Date/time columns

- Columns whose values parse as datetimes: none
- Columns whose name contains date/time/stamp: `signal_time`, `travel_time`

## Routes

- Distinct (source_location, destination_location) pairs: 2,082,676
- Distinct pairs / rows: 0.9934

## Raw tortuosity (distance / straight-line distance, unclipped)

- Rows with a finite value: 2,096,563 of 2,096,563 (float64 on the raw CSV)

| percentile | raw_tortuosity |
|---|---|
| p1 | 1.00045 |
| p50 | 1.00063 |
| p99 | 1.00080 |

## Median minutes per km by hour (rows with distance > 0)

| hour | median_min_per_km |
|---|---|
| 0 | 1.5237 |
| 1 | 1.5224 |
| 2 | 1.5213 |
| 3 | 1.5234 |
| 4 | 1.5209 |
| 5 | 1.5242 |
| 6 | 1.5226 |
| 7 | 1.5201 |
| 8 | 3.7235 |
| 9 | 3.7011 |
| 10 | 3.7209 |
| 11 | 3.7266 |
| 12 | 1.5259 |
| 13 | 1.5244 |
| 14 | 1.5238 |
| 15 | 1.5236 |
| 16 | 1.5264 |
| 17 | 3.7145 |
| 18 | 3.7266 |
| 19 | 3.7187 |
| 20 | 3.7318 |
| 21 | 3.7220 |
| 22 | 1.5247 |
| 23 | 1.5219 |

## Mean travel_time by weather

| weather | mean_travel_time_min | ratio_to_Clear |
|---|---|---|
| Clear | 29.9917 | 1.0000 |
| Cloudy | 29.9557 | 0.9988 |
| Foggy | 48.3345 | 1.6116 |
| Rainy | 39.4470 | 1.3153 |

## Target structure (100,000-row sample, seed 0)

- R² of travel_time vs distance/speed (squared Pearson correlation, 100,000 rows with speed > 0): 0.9905
- Pearson corr(speed, vehicles/road_capacity) (100,000 rows with road_capacity > 0): -0.8337
