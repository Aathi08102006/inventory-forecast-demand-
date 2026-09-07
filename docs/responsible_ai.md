# Demandly AI — Responsible Forecasting Principles

## Core Principle

> Forecasts should communicate uncertainty rather than hide it.

## Principles

### 1. No False Certainty
Every forecast shows prediction intervals. Point forecasts are never displayed as guaranteed outcomes. The UI explicitly labels intervals with their probabilistic meaning.

### 2. Transparency
Demand drivers are shown for every forecast. Planners can always ask "Why this forecast?" and receive a breakdown of contributing factors.

### 3. Human Control
The planner makes the final inventory decision. Demandly provides recommendations — not commands. All overrides are logged and respected.

### 4. Calibration Accountability
The system is evaluated against calibration standards. A stated 80% interval should cover actual demand ~80% of the time. Calibration is monitored and reported.

### 5. Scenario Awareness
Planners can explore alternative assumptions before committing. The Scenario Lab prevents over-reliance on any single forecast point.

### 6. Data Transparency
All data in this prototype is synthetic. This is prominently disclosed in the UI, documentation, and data files.

### 7. Model Limitations
The system acknowledges:
- Trained on synthetic data only
- May underperform on extreme events outside training distribution
- Cold-start limitations for new stores/products
- Festival effects simplified to city-level matching

### 8. Human + AI Learning Loop
Planner override decisions are logged for future analysis. This enables studying when human judgment outperforms the model — a valuable Responsible AI feedback mechanism.

## What We Never Do

- Display a point forecast as a guaranteed number
- Use the LLM for numerical forecasting
- Silently impute missing input data
- Claim formal validation without actual measurement
- Fabricate accuracy metrics
