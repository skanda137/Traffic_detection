# Synopsis vs what was built

| Synopsis item | What was built | Status |
|---|---|---|
| API collection pipeline (obj. 1 and 4.1) | Not built for this dataset. Phase 1 (synopsis design, implemented on synthetic data) implements the record schema, the validation logic and the daily time-slot design on SYNTHETIC data only; no collection scheduler or API client exists in the code. Phase 2 uses a supplied dataset whose source is recorded in `docs/provenance.txt`. | Not built for Phase 2 data |
| `congestion_ratio` target (4.2) | Phase 1 only (synthetic). The Phase 2 target is `travel_time` in minutes. | Changed |
| Calendar + weather features only (4.3) | FORECAST uses hour, day and weather plus route geometry. NOWCAST adds live vehicles, road capacity and signal time. | Extended |
| Temporal holdout (4.5) | Not possible: the dataset has no timestamp column. A random 70/15/15 train/val/test split is used instead. | Not possible |
| GBDT over sequence model (4.4) | GBDT (LightGBM). The dataset has no timestamps and near-unique routes, so no sequences exist to model. | Built (GBDT) |
| ToS / cost (4.6) | Applies only to API collection; not applicable to the supplied dataset. | Not applicable |
| Dashboard (obj. 4) | Built: Forecast ("plan ahead") and Nowcast ("live traffic"). | Built |
