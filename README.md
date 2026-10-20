# Traffic Detection

Travel-time prediction for city routes (Phase 2, the result) plus a congestion-ratio pipeline prototype on synthetic data (Phase 1, method only).

DATASET PROVENANCE: <Member A to fill>

## Run (Phase 2), from the repository root, in order

```
pip install -r requirements.txt
python src/leakage_and_audit.py      # model_ready_traffic.csv -> model_cleaned_traffic.parquet
python src/train_final.py            # trains NOWCAST + FORECAST -> models/nowcast_model.pkl, models/forecast_model.pkl
python src/final_report.py           # test-set metrics -> reports/phase2_final_metrics.csv, phase2_slices.csv, feature_importance_ranking.csv
streamlit run dashboard/app.py       # dashboard
```

`leakage_and_audit.py` uses paths relative to the current directory, so run it from the repository root.

## Files

| File | What it does |
|---|---|
| `src/leakage_and_audit.py` | Loads `model_ready_traffic.csv`, maps day names to 0-6 (Monday=0), drops `speed` (target leakage), filters out-of-range rows, writes `model_cleaned_traffic.parquet`. |
| `src/feature_pipeline.py` | The single feature path for training and inference: spatial features, traffic ratios (only when vehicles/road_capacity/signal_time are given), `is_peak`/`is_weekend` derived from hour/day, hour encoding, and `apply_categories` (raises on unknown categories). Run directly, it writes `model_featured_traffic.parquet`. |
| `src/train_final.py` | Random 70/15/15 split, trains NOWCAST and FORECAST LightGBM models with the v1 hyperparameters, early stopping on val only, saves bundles to `models/`. |
| `src/final_report.py` | The only test-set evaluation: baselines, both models, slices, gain importance; writes the three `reports/` CSVs. |
| `src/evaluate_and_explain.py` | `TrafficInferenceService` (bundle path -> feature_pipeline -> saved categories -> prediction), used by the dashboard. Its `run_diagnostics()` still scores the older `traffic_model_final.pkl` (v2). |
| `dashboard/app.py` | Streamlit app: Nowcast/Forecast, predicted minutes, error band from the test MAE/mean_actual, hour-by-hour chart. |
| `src/train_prediction.py`, `src/train_prediction_v2.py` | Earlier v1/v2 experiments (weekday train, weekend holdout); both write `traffic_model_final.pkl`. Superseded by `train_final.py`. |
| `synthetic_data.py` | Phase 1: generates `synthetic_traffic_dataset.csv`. |
| `src/data_validator.py` | Phase 1: schema and bounds checks for the synthetic congestion data. |
| `src/model_comparison.py` | Phase 1: tuned LightGBM vs XGBoost vs historical-average baselines on synthetic data; writes `models/best_congestion_model.pkl` and `reports/holdout_metrics.csv`, `model_selection.json`, `feature_importance.csv`. |
| `src/error_diagnostics.py` | Phase 1: error slices for the Phase 1 model. |
| `src/inference_service.py` | Phase 1: `TrafficPredictor` for the congestion-ratio model. |
| `src/explainability.py` | Phase 1 SHAP script. Does not run as written (wrong model path, passes the bundle dict to SHAP). |
| `train_baseline.py` | Phase 1: untuned LightGBM sanity check. |

Phase 1 commands: `python synthetic_data.py`, `python src/model_comparison.py`, `python src/error_diagnostics.py`, `python src/inference_service.py` (needs xgboost and optuna, see `requirements.txt`).

## Models

- **NOWCAST**: all 21 v1 features, including live traffic inputs (vehicles, road_capacity, signal_time and the ratios built from them). Use when current traffic counts are known.
- **FORECAST**: 15 features: route geometry, hour, is_peak, day, weather. No live traffic inputs, so it can be used ahead of time.

Both use a linear target (no log transform). v1-style training was kept over v2 because v2's log target biased predictions by -0.97 min on average (v1: -0.04 min).

## Leakage finding

`speed` is dropped in cleaning because `travel_time` is almost exactly `58 * distance / speed` (R² 0.99 on 100K rows). The remaining traffic features still carry much of that signal: `vehicles / road_capacity` correlates -0.83 with `speed`. That is why NOWCAST (needs live counts) and FORECAST (does not) are reported separately.

## Limitations

- **No timestamp column.** The data has only `day_of_week` and `hour`. The split is random 70/15/15 (seed 42), not temporal. A weekday-train/weekend-test split scored within 0.5% of a random split, so it gave no extra out-of-time evidence.
- **Nowcast vs forecast.** NOWCAST's accuracy depends on live traffic counts that are only available at trip time; FORECAST is the honest ahead-of-time number.
- The dashboard's error band is the test-set MAE/mean_actual ratio applied to the prediction, a typical error, not a confidence interval.
- Phase 1 data is synthetic (`synthetic_data.py`); its results demonstrate the pipeline and method, not real-world accuracy.
