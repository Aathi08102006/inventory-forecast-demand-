"""
Demandly AI — Seasonal Naive Baseline
Computes lag-7 (same weekday last week) baseline and compares vs Demandly AI.
"""
import pandas as pd
import numpy as np
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_baseline():
    df = pd.read_csv("dataset/demand_history.csv", parse_dates=["date"])
    df = df.sort_values(["store_id","product_id","date"])
    df["lag7"] = df.groupby(["store_id","product_id"])["actual_demand"].shift(7)
    cutoff = df["date"].max() - pd.Timedelta(days=90)
    test = df[df["date"] > cutoff].dropna(subset=["lag7"])

    actual    = test["actual_demand"].values
    predicted = test["lag7"].values

    mae  = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100

    results = {"mae": round(mae,2), "rmse": round(rmse,2), "mape": round(mape,2), "n": len(test)}
    print("Seasonal Naive Baseline Results:")
    print(f"  MAE  = {results['mae']}")
    print(f"  RMSE = {results['rmse']}")
    print(f"  MAPE = {results['mape']}%")
    print(f"  N    = {results['n']:,} test rows")

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/baseline_results.json","w") as f:
        json.dump(results, f, indent=2)
    print("Saved to evaluation/results/baseline_results.json")
    return results

if __name__ == "__main__":
    run_baseline()
