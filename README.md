# Studio North LTV holdout challenge: Chandra Gompa

**Model:** beta-discrete-Weibull (BdW) retention with recency-weighted cohorts, multiplied by an age- and calendar-trended spend per active member, calibrated to each cohort's recent spend.

This entry is for Path B (FAQ 8) of MBA 743's "Studio North: Choose Your Own Adventure". It forecasts expected retained members, net revenue and contribution for every acquisition cohort × channel × need cell, January to December 2026 (1,152 rows).

## Run

```text
uv sync --frozen
uv run --frozen python main.py --data-dir /path/to/released/data --prediction-grid prediction_grid.csv --output outputs/predictions.csv --seed 20261116
```

Python 3.12 with pandas, numpy and scipy, all locked in `uv.lock`. The full fit plus prediction takes about 2 seconds on one CPU core. It needs no network, no GPU and no precomputed files. The output is deterministic: two clean runs with seed 20261116 gave byte-identical files (SHA-256 `3efa62eb1429e534ceb499cd34e05082b17ab4c6ea7f505660356e8c1002f090`). It also passes `cohort_lab.challenge.validate_predictions` from the course repository.

## Model

The interface (`predict` in `main.py`) is unchanged. The model lives in `model.py`.

**Retention.** For each channel × need cell, survival to cohort age *t* is

S(t) = B(α, β + t^c) / B(α, β)

This is the beta-discrete-Weibull of Fader, Hardie, Liu, Davin and Steenburgh (2018). Each customer has a churn propensity θ ~ Beta(α, β). The shape parameter *c* allows duration dependence: when c = 1 the model reduces to the shifted-beta-geometric (sBG) used in the course's reference model. It is fit by maximum likelihood on individual customers. A customer who churned after paying through age L contributes S(L) − S(L+1) to the likelihood. A customer still active at the origin contributes S(L), as a right-censored observation. Older cohorts get exponentially lower weight (half-life of 6 cohort months), so a drift in acquisition quality shows up in the fit. The forecast for each cell starts from the observed survivors and conditions on survival: origin_retained × S(age) / S(origin age). In a mixture model this ratio already accounts for the fact that older survivors are more loyal.

**Spend per active member.** Monthly net revenue and contribution per retained member are fit by weighted least squares on the last 12 observed months:

log(value per active) = cell level + slope × log(1 + age) + γ × (calendar month − origin)

A Duan smearing factor corrects for the log transform. The age term captures upgrades and add-ons that come with tenure. The calendar term captures a steady price and mix trend: holding age fixed, the fitted trend at the December 2025 origin is about +0.8% a month for contribution and +0.4% for net revenue per active member. Each cohort's level is then scaled by its own observed/fitted ratio over the last 3 months. That ratio is shrunk toward 1 with k = 20 member-months. Contribution is capped at net revenue, and zero members always means zero receipts.

**Why this differs from the sBG + Gamma reference:** the reference keeps spend per active member constant. The backtest below shows that leaves a systematic 5 to 8% under-forecast of contribution. Most of the gain comes from the trended, cohort-calibrated spend and from recency weighting. The Weibull shape adds only a small edge over sBG once spend is trended (mean WAPE 0.045 vs 0.046). With flat spend, BdW was actually worse than sBG, so the shape parameter is not doing the heavy lifting.

**Extrapolation to ages 24 to 35** (needed only for the January 2024 cohort): BdW survival comes from a parametric curve, so it extrapolates smoothly instead of continuing the last observed hazard. Spend uses log(1 + age), so it grows more slowly at high ages. The calendar trend is extended for 12 months without damping. Damping it by half was worse in validation, but if the price trend stalls, this is the main risk of over-forecasting.

## Chronological validation

`backtest.py` rebuilds the eight observed tables as they stood at the end of each historical origin. It drops later transactions, and the next period's renewal shows as `pending`. It then forecasts forward and scores cohort × calendar-month contribution WAPE against released history (the same metric as the challenge). The private 2026 outcomes were never used.

| Model | Dec 2024 (12 mo) | Mar 2025 (9 mo) | Jun 2025 (6 mo) | Sep 2025 (3 mo) | Mean |
|---|---:|---:|---:|---:|---:|
| Constant hazard (scaffold) | 0.379 | 0.322 | 0.270 | 0.207 | 0.295 |
| sBG per cell, flat spend | 0.073 | 0.079 | 0.087 | 0.072 | 0.078 |
| BdW per cell, flat spend | 0.092 | 0.094 | 0.090 | 0.072 | 0.087 |
| sBG, recency + cohort calibration | 0.065 | 0.061 | 0.073 | 0.055 | 0.063 |
| sBG, recency + trend + calibration | 0.061 | 0.042 | 0.042 | 0.038 | 0.046 |
| **BdW, recency + trend + calibration (final)** | **0.054** | **0.045** | **0.042** | **0.037** | **0.045** |

Full results, including signed bias, are in `backtest_results.json`. Reproduce them with `uv run --frozen python backtest.py --data-dir /path/to/data`.

**Limitations.** At the 2025 origins the forecast still under-predicts retained members by about 3%, which suggests hazards were improving slightly over calendar time. The final model does not use engagement (`product_events`) or feedback signals. The two origins with the longest horizons have the least spend history for estimating the trend. Validation measures accuracy on history only; it does not tell us how the model will score on the private holdout.

## Files

- `main.py`: standard CLI, `predict()` entry point, input loading and output validation. The original scaffold model is kept as `example_predict` for comparison.
- `model.py`: BdW retention, spend curves and the final settings (`FINAL_SETTINGS`).
- `backtest.py`, `backtest_results.json`: chronological validation.
- `submission-contract.json`, `prediction_grid.csv`: unchanged from the course repository (dataset 5.1).
- `pyproject.toml`, `uv.lock`: locked dependencies.
