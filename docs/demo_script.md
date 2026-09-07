# Demandly AI — 3-Minute Demo Script

## Setup
Open `frontend/index.html` in a browser (or `http://localhost:8000/` if running the backend).

---

## 0:00–0:30 — The Problem

> "Traditional demand forecasting gives planners a single number — like 1,250 units. But this hides all the uncertainty. Demand could actually be anywhere from 950 to 1,650. If planners treat the point forecast as certain, they make poor inventory decisions."

**Action:** Point to the Dashboard's planning risk bar showing 17 HIGH-risk forecasts.

---

## 0:30–1:00 — The Forecast Card

**Action:** Click **Forecasts** in the left nav.

> "Demandly shows the expected demand AND the full range of what's possible. The blue dot is P50 — the median forecast. The band shows the 80% prediction interval — demand falls inside this range 8 out of 10 times in comparable conditions."

Point out:
- The forecast range visualization (P10 → P50 → P90)
- The uncertainty badge (MEDIUM / HIGH)
- The 80% interval callout explaining what it means

---

## 1:00–1:30 — Why This Forecast?

**Action:** Scroll to the **Why this forecast?** section. Click "Show Detail".

> "The system explains what's driving the forecast. Here, the festival effect is adding 18%, temperature is adding 12%. The planner can understand the reasoning — it's not a black box."

Adjust the **Festival Intensity** slider to 0 — watch the drivers section update.

---

## 1:30–2:00 — Scenario Lab

**Action:** Click **Scenario Lab** in nav.

> "What if conditions change? The Scenario Lab lets planners explore alternatives before committing to a decision."

**Action:** Click the **🌡 Extreme Heat** demo button.

> "Temperature jumps to 42°C. Watch the forecast shift from 350 to 470 units, uncertainty rises to HIGH, and the system recommends pre-positioning stock."

**Action:** Show the Scenario Comparison Table — 5 scenarios side-by-side.

---

## 2:00–2:30 — Disruption Mode

**Action:** Click **Disruptions** in nav.

> "Supply chain disruptions are inevitable. Demandly has a dedicated Disruption Mode."

**Action:** Click **⏱ Delivery Delay**.

> "A 12-hour delivery delay doesn't change the demand forecast much — but it dramatically increases fulfillment risk. The system recommends expediting supply and increasing safety stock."

Point to the side-by-side Normal vs Disruption comparison.

**Action:** Also activate **🎉 Festival Surge** to show compounding effects.

---

## 2:30–3:00 — Evaluation & Decisions

**Action:** Click **Evaluation** in nav.

> "The model is evaluated against a calibration standard. Our stated 80% intervals actually cover 78.5% of real observations on the test set — close to target. And Demandly reduces MAE by 36% vs a simple seasonal naive baseline."

**Action:** Click **Planner Decisions**.

> "The system supports human-in-the-loop decisions. Planners choose Conservative, Balanced, or Aggressive stocking positions. If they override the AI recommendation, they can log the reason. This creates a feedback loop for model improvement."

---

## Key Takeaways

1. **Uncertainty is explicit** — planners see ranges, not just points
2. **Drivers are transparent** — no black-box forecasting
3. **Scenarios are testable** — before committing to decisions
4. **Disruptions are handled** — separate mode with dedicated recommendations
5. **Humans stay in control** — AI recommends, planner decides
