# Traffic Detection

Travel-time prediction for city routes (Phase 2, the result) plus a congestion-ratio pipeline prototype (Phase 1: synopsis design, implemented on synthetic data; method only).

See docs/provenance.txt

What the synopsis promised versus what was built: [docs/synopsis_vs_built.md](docs/synopsis_vs_built.md). Every number for slides is in [reports/slide_numbers.md](reports/slide_numbers.md), generated from `reports/`; this README deliberately quotes no metric values.

## Run (Phase 2), from the repository root, in order

```
pip install -r requirements.txt
python src/leakage_and_audit.py                     # model_ready_traffic.csv (current dir) -> model_cleaned_traffic.parquet
python src/train_final.py                           # NOWCAST + FORECAST -> models/nowcast_model.pkl, models/forecast_model.pkl
python src/final_report.py                          # test-set metrics, slices, importance, error quantiles, input ranges, example routes -> reports/
python src/data_audit.py <path/to/model_ready_traffic.csv>   # measured dataset facts -> reports/data_audit.md
python src/ablation_and_seeds.py                    # ablation over split seeds -> reports/ablation_and_seeds.csv
python src/slide_numbers.py                         # reports/slide_numbers.md (needs the two steps above)
streamlit run dashboard/app.py                      # dashboard
```

`leakage_and_audit.py` uses paths relative to the current directory, so run it from the repository root.

`model_cleaned_traffic.parquet` must be in the repository root for `src/train_final.py`, `src/final_report.py` and `src/ablation_and_seeds.py` (all read it through `train_final.load_split()`, path anchored to the repository root). `src/feature_pipeline.py`, when run directly, reads it from the current directory. `src/leakage_and_audit.py` writes it.

## Dashboard

`streamlit run dashboard/app.py`. At runtime it reads only these files (no parquet, no raw data):

- `models/forecast_model.pkl`, `models/nowcast_model.pkl` (via `TrafficInferenceService`)
- `reports/input_ranges.json` (via `TrafficInferenceService`)
- `reports/example_routes.csv`, `reports/phase2_error_quantiles.csv`, `reports/phase2_final_metrics.csv`

All four `reports/` files are written by `src/final_report.py`.

- **Forecast** (default, listed first): route, day, hour, weather. Shows predicted minutes at the chosen hour, the best window and worst window (hours within 3% of the 24-hour minimum / maximum, shown as ranges that can wrap past midnight), minutes saved versus the minimum, a 24-hour bar chart with the best window and chosen hour highlighted, and the same route and hour under each weather.
- **Nowcast**: adds vehicles, road capacity and signal time (defaulting to the selected route's row values). Needs live vehicle counts from a sensor feed. Shows predicted minutes and a chart of predicted minutes vs vehicles across the training density_ratio p1-p99 range, with the current input marked.
- **Route selector**: 12 example routes from the test set, stratified by distance quintile; custom coordinates are in an expander.
- **Interval**: "80% of test-set trips fell within [pred × q10, pred × q90]", from the empirical q10/q90 of actual/predicted on the test set (marginal over the test set, not specific to the input).
- **Input validation**: the dashboard and `TrafficInferenceService` reject distance, straight-line distance, raw tortuosity and density_ratio outside the training p1-p99 range, and any route whose distance is shorter than the straight-line distance between its coordinates. The dashboard shows an error and no prediction.
- **Model card** expander: baselines vs NOWCAST vs FORECAST test metrics, the split ("random held-out split; dataset has no timestamp"), and "Predictions reflect the training dataset, not live traffic."

## Files

| File | What it does |
|---|---|
| `src/leakage_and_audit.py` | Loads `model_ready_traffic.csv`, maps day names to 0-6 (Monday=0), drops `speed` (target leakage), filters out-of-range rows, writes `model_cleaned_traffic.parquet`. |
| `src/feature_pipeline.py` | The single feature path for training and inference: spatial features, traffic ratios (only when vehicles/road_capacity/signal_time are given), `is_peak`/`is_weekend` derived from hour/day, hour encoding, `apply_categories` (raises on unknown categories) and the input range checks. Run directly, it writes `model_featured_traffic.parquet`. |
| `src/train_final.py` | Random 70/15/15 split (`load_split(seed=42)`), trains NOWCAST and FORECAST LightGBM models with the v1 hyperparameters, early stopping on val only, saves bundles to `models/`. |
| `src/final_report.py` | The only test-set evaluation of the shipped models: baselines, both models, slices, gain importance, error quantiles; also writes the input ranges (from training rows) and the dashboard's example routes. |
| `src/evaluate_and_explain.py` | `TrafficInferenceService`: `predict` (one record) and `predict_many` (a DataFrame), both via `feature_pipeline`, saved categories and the input range checks. Its `run_diagnostics()` still scores the older `traffic_model_final.pkl` (v2). |
| `src/data_audit.py` | Measured facts about the raw CSV (shape, date/time columns, route pairs, tortuosity, pace by hour, weather means, target structure) -> `reports/data_audit.md`. Makes no claim about the data source. |
| `src/ablation_and_seeds.py` | Reporting only: NOWCAST, FORECAST and an 18-feature ablation on several split seeds vs baseline (a) -> `reports/ablation_and_seeds.csv`. |
| `src/slide_numbers.py` | Builds `reports/slide_numbers.md` from `reports/*.csv`. |
| `dashboard/app.py` | Streamlit app; see [Dashboard](#dashboard). |
| `docs/provenance.txt` | Dataset provenance (to be filled in by Member A). |
| `docs/synopsis_vs_built.md` | Synopsis item -> what was built -> status. |
| `src/train_prediction.py`, `src/train_prediction_v2.py` | Earlier v1/v2 experiments (weekday train, weekend holdout); both write `traffic_model_final.pkl`. Superseded by `train_final.py`. |
| `synthetic_data.py` | Phase 1 (synopsis design, implemented on synthetic data): generates `synthetic_traffic_dataset.csv`. |
| `src/data_validator.py` | Phase 1: schema and bounds checks for the synthetic congestion data. |
| `src/model_comparison.py` | Phase 1: tuned LightGBM vs XGBoost vs historical-average baselines on synthetic data; writes `models/best_congestion_model.pkl` and `reports/holdout_metrics.csv`, `model_selection.json`, `feature_importance.csv`. |
| `src/error_diagnostics.py` | Phase 1: error slices for the Phase 1 model. |
| `src/inference_service.py` | Phase 1: `TrafficPredictor` for the congestion-ratio model. |
| `src/explainability.py` | Phase 1 SHAP script. Does not run as written (wrong model path, passes the bundle dict to SHAP). |
| `train_baseline.py` | Phase 1: untuned LightGBM sanity check. |

Phase 1 commands: `python synthetic_data.py`, `python src/model_comparison.py`, `python src/error_diagnostics.py`, `python src/inference_service.py` (needs xgboost and optuna, see `requirements.txt`).

## Models

- **NOWCAST**: all v1 features, including live traffic inputs (vehicles, road_capacity, signal_time and the ratios built from them). Use when current traffic counts are known.
- **FORECAST**: route geometry, hour, is_peak, day and weather only. No live traffic inputs, so it can be used ahead of time.

Both use a linear target (no log transform). The earlier v2 experiment (`src/train_prediction_v2.py`) used a log target, which biased its predictions downward on average; v1-style training was kept.

## Leakage finding

`speed` is dropped in cleaning because `travel_time` is almost a deterministic function of `distance / speed`, and the remaining traffic ratio `vehicles / road_capacity` is strongly correlated with `speed`; both measurements are in `reports/data_audit.md`. That is why NOWCAST (needs live counts) and FORECAST (does not) are reported separately.

## Limitations

- **Random split, no timestamp.** The data has only `day_of_week` and `hour`, no date or timestamp column (see `reports/data_audit.md`), so the 70/15/15 split is random, not temporal. There is no out-of-time evidence.
- **Distance equals straight-line distance.** In this dataset `distance` is effectively the straight-line distance between the coordinates (raw tortuosity percentiles in `reports/data_audit.md`), so tortuosity is effectively constant, `distance` and `haversine_dist` are the same signal, and custom coordinates only accept straight-line distances.
- **Nowcast vs forecast.** NOWCAST's accuracy depends on live traffic counts that are only available at trip time; FORECAST is the honest ahead-of-time number.
- The dashboard's 80% interval is empirical and marginal over the whole test set; it is not a per-input confidence interval.
- **Phase 1 is synthetic.** Phase 1 data comes from `synthetic_data.py`; its results demonstrate the pipeline and method, not real-world accuracy.
