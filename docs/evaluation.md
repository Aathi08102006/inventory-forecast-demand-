# Demandly AI — Evaluation Report

> All results measured on 9,000 held-out test rows from the synthetic dataset (last 90 days).
> All data is synthetic — results reflect synthetic distribution, not real-world grocery demand.

## Experiment Design

**Research question:** Does showing uncertainty and scenario alternatives help planners make better decisions than a single-number forecast?

**Train/test split:** Last 90 days held out as test set (9,000 rows).

**Baseline:** Seasonal Naive — lag-7 (previous week same weekday).

## Point Forecast Results

| Metric | Baseline (Naive) | Demandly AI | Improvement |
|--------|-----------------|-------------|-------------|
| MAE    | 39.13 units     | 24.95 units | **36.2%**  |
| RMSE   | 55.66 units     | 34.76 units | **37.5%**  |
| MAPE   | 18.31%          | 11.98%      | **34.6%**  |

## Probabilistic Forecast Calibration

| Interval | Target Coverage | Actual Coverage | Calibration Error |
|----------|----------------|----------------|-------------------|
| 50%      | 50.0%          | 48.8%          | ±1.2%            |
| 80%      | 80.0%          | 78.5%          | ±1.5%            |

Interpretation: The 80% interval covers actual demand in ~78.5% of test cases (vs 80% target). The slight under-coverage suggests intervals could be marginally wider to achieve perfect calibration.

## Edge/Failure Case Results

| Case | Status | Notes |
|------|--------|-------|
| Festival Data Missing | PASS | Uncertainty increases, warning shown |
| Extreme Weather (45°C) | PASS | Risk=HIGH, interval expands |
| Delivery Delay 12h | PASS | Risk score increases, Conservative recommended |
| Demand Spike +50% | PARTIAL | P90 captures 78% of spikes (not all) |
| Missing Price/Weather | PASS | Warning displayed, no silent imputation |
| New Store Cold Start | PASS | Interval widens 35%, VERY HIGH uncertainty |

## Feature Importance (measured from P50 model)

| Feature | Importance |
|---------|------------|
| Store size | 69.1% |
| Recent demand (roll7_mean) | 18.4% |
| Month | 2.3% |
| Temperature | 2.3% |
| Rainfall | 1.1% |
| Day of week | 1.1% |
| Product category | 1.0% |
| Weekend | 1.0% |

Note: High store_size importance reflects the large variance in base demand across store sizes in the synthetic data generation process. In real-world data, this distribution would be calibrated against actual historical demand.

## Known Limitations

1. Synthetic data may not capture full complexity of real grocery demand
2. Festival effects use simplified city-level matching
3. Planner decision experiment not run with real planners (prototype only)
4. Demand spike detection (edge case 4) is partial — very sharp spikes escape P90
5. Feature importance dominated by store size due to synthetic data construction

## Recommendations for Production Deployment

1. Replace synthetic data with 2+ years of real store-level demand history
2. Add conformal prediction for distribution-free coverage guarantees
3. Implement time-series cross-validation (expanding window)
4. Add store-level model fine-tuning for high-volume stores
5. Integrate real-time weather and festival APIs
6. Conduct formal planner user study to measure decision quality improvement
