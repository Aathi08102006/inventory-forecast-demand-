# 📦 Demandly AI
### *Uncertainty-Aware Demand Forecasting Assistant*

> **Don't just forecast. Plan for what's possible.**

---

## ⚠️ Important Notice

All data in this project is **synthetic, generated purely for demonstration and evaluation purposes**. No real store, customer, or demand data was used.

---

## 1. Problem Statement

A grocery distributor supplies multiple stores where demand is driven by local festivals, weather, price changes, store characteristics, seasonal patterns, promotions, disruptions, and unexpected spikes.

Current forecasts give planners **one number** — e.g. "Tomorrow: 1,000 units."  
Planners may treat this as certain, leading to poor inventory, replenishment, and staffing decisions.

**Actual demand could be:**
```
750 ─────── 1000 ─────── 1300
      Possible demand range
```

---

## 2. Solution

Demandly AI answers four questions for every forecast:

| Question | Example Output |
|---|---|
| **HOW MUCH?** | Expected: 1,250 units |
| **HOW UNCERTAIN?** | 80% range: 1,050 – 1,500 |
| **WHY?** | Festival +18%, Temperature +12%, Weekend +8% |
| **WHAT IF?** | If temp rises to 38°C → 1,420 units |

---

## 3. Key Features

- **Uncertainty-aware forecasts** — P10/P25/P50/P75/P90 quantiles
- **Demand driver analysis** — Festival, weather, price, calendar effects
- **Scenario Lab** — Interactive what-if analysis with real-time forecast updates
- **Disruption Mode** — Tests forecasts under delivery delays, capacity loss, demand spikes
- **Normal vs Disruption comparison** — Side-by-side view
- **Planner Decision Support** — Conservative / Balanced / Aggressive options with trade-off explanation
- **Human override logging** — Planners can override AI recommendations with optional reasons
- **What-If AI chat** — Natural language scenario queries powered by Claude API
- **Calibration analysis** — Measured interval coverage vs stated coverage
- **Error analysis** — 6 edge/failure case categories
- **Responsible AI page** — Transparency, limitations, human control
- **34 automated tests** — All passing

---

## 4. Architecture

```
demandly-ai/
├── frontend/
│   └── index.html           ← Responsive, API-driven single-page dashboard
├── backend/
│   ├── main.py              ← HTTP server (stdlib only, no FastAPI needed)
│   ├── services/
│   │   └── forecast_service.py   ← Bundled quantile-model inference (optional ML extras)
│   └── forecasting/
│       └── train_model.py   ← Model training script
├── dataset/
│   ├── generate_data.py     ← Synthetic data generator
│   ├── demand_history.csv   ← 73,100 rows (generated)
│   ├── stores.csv
│   └── products.csv
├── models/
│   ├── quantile_models.pkl  ← Trained quantile GBM models (P10/P25/P50/P75/P90)
│   ├── enc_size.pkl
│   ├── enc_type.pkl
│   └── enc_cat.pkl
├── evaluation/
│   ├── evaluate.py          ← Evaluation script
│   └── results/
│       ├── model_results.json
│       └── evaluation_report.json
├── tests/
│   └── test_forecasting.py  ← 34 automated tests
├── docs/
│   ├── architecture.md
│   ├── responsible_ai.md
│   ├── evaluation.md
│   └── demo_script.md
├── README.md
├── .env.example
└── docker-compose.yml
```

---

## 5. Dataset

- **73,100 rows** of synthetic grocery demand data
- **10 stores** across Tamil Nadu (Madurai, Salem, Coimbatore, etc.)
- **10 products** across 5 categories (Beverages, Dairy, Snacks, Frozen, Staples)
- **2 years** of daily data (2023–2024)
- **12 festivals** with city-specific intensity curves
- **Realistic noise** — 12% coefficient of variation on demand
- **Disruption events** — delivery delays, warehouse capacity losses

Key fields: `date, store_id, product_id, actual_demand, temperature, rainfall, festival, festival_intensity, price, discount, disruption_flag, delivery_delay, warehouse_capacity`

---

## 6. Forecasting Model

**Method:** Quantile Gradient Boosting Regression (scikit-learn GradientBoostingRegressor)

**Quantiles trained:** P10, P25, P50, P75, P90

**Features used:**
- Lag features (lag1, lag7, rolling 7-day mean/std)
- Weather (temperature, rainfall, humidity, heatwave flag)
- Festival (intensity, days-to-festival)
- Price (discount %, promotion flag)
- Calendar (day-of-week, weekend, month)
- Store attributes (encoded size, type)
- Product category (encoded)
- Disruption flags

**Uncertainty method:** Quantile regression — each quantile model independently learns the conditional distribution boundary, providing calibrated prediction intervals without distributional assumptions.

---

## 7. Baseline

**Seasonal Naive:** Tomorrow's demand = demand from the same weekday last week (lag-7).

This is intentionally simple and commonly used in practice.

---

## 8. Evaluation Results (measured on synthetic test data)

| Metric | Baseline (Naive) | Demandly AI | Improvement |
|---|---|---|---|
| MAE (units) | 39.13 | 24.95 | **36.2%** |
| RMSE (units) | 55.66 | 34.76 | **37.5%** |
| MAPE (%) | 18.31% | 11.98% | **34.6%** |

| Calibration | Target | Actual | Error |
|---|---|---|---|
| 50% interval coverage | 50% | 48.8% | ±1.2% |
| 80% interval coverage | 80% | 78.5% | ±1.5% |

*All results measured on 9,000 held-out test rows (last 90 days of synthetic data).*

---

## 9. Installation & Running

### Prerequisites
- Python 3.10+
- pip (only required for optional training, model inference, and test tooling)

### Step 1 — Optional ML dependencies
```bash
pip install pandas numpy scikit-learn joblib pytest
```

### Step 2 — Generate dataset (already included)
```bash
python3 dataset/generate_data.py
```

### Step 3 — Train model (already included)
```bash
python3 backend/forecasting/train_model.py
```

### Step 4 — Run evaluation (already included)
```bash
python3 evaluation/evaluate.py
```

### Step 5 — Run tests
```bash
python3 tests/test_forecasting.py
```

### Step 6 — Start the working application
```bash
python backend/main.py
```

### Step 7 — Open frontend
```
Visit http://localhost:8000/ (the dashboard requires the local API server).
```

### Optional: Docker
```bash
docker-compose up
```

---

## 10. Scenario Feature

Scenario Lab is fully local and needs no API key. It calls the numerical forecast API with changed conditions such as heat, festival intensity, delivery delay, and warehouse capacity. When optional ML dependencies are installed, the server uses the supplied quantile models; otherwise it uses a deterministic demo fallback so the prototype remains runnable.

---

## 11. Demo Instructions (3 minutes)

See `docs/demo_script.md` for the full guided demo.

**Quick flow:**
1. Open Dashboard → see planning overview and forecast card
2. Click Forecasts → adjust sliders, see live forecast updates
3. Open Scenario Lab → change temperature to 42°C, see forecast shift
4. Open Disruptions → activate "Delivery Delay" → see risk change
5. Open Evaluation → see calibration charts and baseline comparison
6. Open Responsible AI → see transparency principles

---

## 12. Limitations

- Trained on synthetic data — real-world accuracy will differ
- Festival effects use simplified city-level matching
- No real-time data feeds in this prototype
- LLM integration requires API key
- Planner decisions stored in-memory (not persistent)
- Cold-start handling uses fallback lag values

---

## 13. Future Improvements

- Real FMCG demand data integration
- Conformal prediction for distribution-free coverage guarantees
- Time-series cross-validation
- Store-level model fine-tuning
- Real-time weather and festival API integration
- Persistent planner decision database
- Mobile-responsive layout improvements
- Multi-user planner collaboration features
