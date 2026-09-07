# Demandly AI — Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React)                      │
│  Dashboard │ Forecasts │ Scenarios │ Disruptions         │
│  Drivers   │ Decisions │ What-If   │ Evaluation          │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP / direct JS call
┌────────────────────────▼────────────────────────────────┐
│                  BACKEND (Python stdlib)                  │
│  GET /api/forecast    POST /api/scenario                  │
│  GET /api/stores      POST /api/planner-decision          │
│  GET /api/evaluation  GET  /api/calibration               │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              FORECAST SERVICE                            │
│  - Load quantile models (P10/P25/P50/P75/P90)            │
│  - Feature engineering (lags, encodings)                 │
│  - Driver computation                                    │
│  - Risk scoring                                          │
│  - Planning recommendations                              │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              ML MODELS (scikit-learn)                    │
│  GradientBoostingRegressor × 5 quantiles                 │
│  Trained on 64,100 rows of synthetic data                │
│  Test set: 9,000 rows (last 90 days)                     │
└─────────────────────────────────────────────────────────┘
```

## Frontend Architecture

The frontend is a **single self-contained HTML file** with:
- React 18 (via CDN)
- Recharts (via CDN)
- Babel standalone (JSX transpilation in-browser)
- No build step required — open directly in browser

All forecast computation is duplicated client-side in JavaScript for instant UI responsiveness.

## Model Architecture

**Quantile Gradient Boosting:**
- 5 separate GBM models, one per quantile (0.1, 0.25, 0.5, 0.75, 0.9)
- `loss="quantile"` with `alpha=q`
- 200 estimators, depth 5, learning rate 0.08, subsample 0.8
- 24 input features including lag features, weather, festival, price, calendar, store/product encodings

**Uncertainty method:** Quantile regression provides distribution-free prediction intervals. No parametric assumptions about residual distribution.

## Data Flow

```
Synthetic Generator → demand_history.csv
                             ↓
                      train_model.py
                      (feature engineering + training)
                             ↓
              quantile_models.pkl + encoders
                             ↓
                    forecast_service.py
                    (inference + driver calc)
                             ↓
                      backend/main.py
                      (HTTP API)
                             ↓
                    frontend/index.html
                    (React UI)
```

## AI Chat Integration

The What-If? page optionally calls the Anthropic Claude API:
1. User types a natural language question
2. Client-side JS extracts scenario parameters (temperature, delay, etc.)
3. Local JS model computes the forecast
4. Anthropic API receives: model output + user question → generates explanation
5. Numerical values ALWAYS come from the local model, never from LLM

This architecture ensures LLM is used only for natural language understanding and explanation, not for numerical prediction.
