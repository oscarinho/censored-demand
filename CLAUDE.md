# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this lab is

A CRISP-ML(Q) portfolio lab (`labs/censored-demand/`) by Oscar Ponce. The thesis: in retail, observed `sale_amount` ≠ demand. When a shelf goes empty (stockout), real demand is **right-censored** and invisible — "lost sales". Forecasting observed sales trains a model to *underestimate demand exactly when it matters most*. The lab (1) shows that trap, (2) recovers latent (uncensored) demand, (3) forecasts on the recovered demand, (4) quantifies the hidden lost sales.

The full spec lives in **`PROMPT-CLAUDE-CODE.md`** — read it before building anything; it defines methodology phase-by-phase, deliverables, and the Definition of Done. `HALLAZGOS.md` holds verified profiling findings; `README.md` is the public-facing summary.

Status: data + spec are in place. The notebooks (the main deliverable) **do not exist yet**.

## Environment & commands

Everything runs in conda env `ml-exp` (Python 3.11). Available: pandas, numpy, pyarrow, scikit-learn, lightgbm, xgboost, catboost, statsforecast, neuralforecast, pmdarima, prophet, statsmodels, shap, matplotlib, seaborn. Prefer **lightgbm + statsforecast** for speed. `requ.txt` is the full `conda list` dump of the env.

```bash
conda activate ml-exp
python scripts/get_and_profile.py   # downloads dataset, writes data/ sample + PROFILE.txt
```

`scripts/get_and_profile.py` pulls `Dingdong-Inc/FreshRetailNet-50K` via HuggingFace `datasets`, saves a 20k-row sample to `data/freshretailnet_train_sample.parquet`, and writes `data/PROFILE.txt`. Re-run only when internet is available.

**Iterate on the 20k sample; run final numbers on the full split (train 4.5M / eval 350k).** Ask before any long run over the full dataset.

## Dataset (FreshRetailNet-50K)

50k store×product series, 90 days, hourly resolution. 19 columns: IDs (`city_id`, `store_id`, `management_group_id`, `first/second/third_category_id`, `product_id`), `dt`, `sale_amount` (daily), `hours_sale` (array[24], hourly sales), `hours_stock_status` (array[24], **1 = STOCKOUT / empty shelf, 0 = in stock**), `stock_hour6_22_cnt`, `discount`, `holiday_flag`, `activity_flag`, weather (`precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level`).

Verified signal (verify against data, don't assume): ~25% of store-hours are stockouts; 44.6% of days have ≥1 OOS hour; hourly sale 0.004 (OOS) vs 0.054 (in-stock); corr(sale, discount) ≈ −0.34; holiday lift +24%.

## Methodology guardrails (hard rules from a retired prior lab)

These are the reason the lab exists — violating them invalidates the result:

- **The stock/status columns model the censorship; they are NEVER naive predictors of sales.** The prior lab (`forecasting-inventory`) was retired because it leaked the target via an inventory-level proxy of censorship. Do not feed any feature derived from the target.
- **You cannot evaluate against real demand — it isn't observable.** Evaluate only on *non-censored* observations (in-stock hours/days) or fully in-stock holdout series. Use the dataset's train/eval split. Show the censored-trained model underestimates vs the recovered-demand model on that clean holdout.
- **Temporal split only** — train ends strictly before eval; no leakage. Never random-split time series.
- **Imputed demand is an estimate, not ground truth — say so.** Declare assumptions and limits explicitly (the "Q" in CRISP-ML). Zero over-claiming; if recovery only helps stockout-prone series, say exactly that.
- **Nothing hardcoded** — every quantitative claim must be computed in the notebook.

## Deliverables & style

- `notebooks/Censored_Demand_CRISPML.ipynb` (EN) + Spanish twin `notebooks/Censored_Demand_CRISPML_ES.ipynb`.
- Style template (form only, NOT conclusions): `labs/forecasting-inventory/notebooks/Inventory_Forecasting_CRISPML.ipynb` — 6-phase CRISP-ML(Q) structure, bilingual EN/ES markdown, honest-caveat tone, "headline finding" up top. ⚠️ That lab was retired for data problems — copy its form, not its numbers.
- A 1–2 line headline finding in `README.md` once results exist.
