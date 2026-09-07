"""
Demandly AI — Error Analysis
Classifies forecast errors into categories and produces a report.
"""
import pandas as pd
import numpy as np
import json, os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.forecast_service import MODELS, ENC_SIZE, ENC_TYPE, ENC_CAT, FEATURES, LOADED

def run_error_analysis():
    df = pd.read_csv("dataset/demand_history.csv", parse_dates=["date"])
    df = df.sort_values(["store_id","product_id","date"])
    df["lag7"] = df.groupby(["store_id","product_id"])["actual_demand"].shift(7).fillna(200)
    df["roll7"] = df.groupby(["store_id","product_id"])["actual_demand"].shift(1).transform(
        lambda x: x.rolling(7,min_periods=1).mean()).fillna(200)
    df["roll7_std"] = df.groupby(["store_id","product_id"])["actual_demand"].shift(1).transform(
        lambda x: x.rolling(7,min_periods=1).std()).fillna(60)
    df["lag1"] = df.groupby(["store_id","product_id"])["actual_demand"].shift(1).fillna(200)

    cutoff = df["date"].max() - pd.Timedelta(days=90)
    test = df[df["date"] > cutoff].copy()

    if LOADED:
        test["store_size_enc"] = ENC_SIZE.transform(test[["store_size"]].values)
        test["store_type_enc"] = ENC_TYPE.transform(test[["store_type"]].values)
        test["product_cat_enc"] = ENC_CAT.transform(test[["product_category"]].values)
        test.rename(columns={"lag7":"lag7","roll7":"roll7_mean","roll7_std":"roll7_std","lag1":"lag1"}, inplace=True)
        X = test[FEATURES].fillna(0)
        test["p10_pred"] = MODELS["0.1"].predict(X)
        test["p50_pred"] = MODELS["0.5"].predict(X)
        test["p90_pred"] = MODELS["0.9"].predict(X)
    else:
        test["p50_pred"] = test["lag7"]
        test["p10_pred"] = test["lag7"] * 0.8
        test["p90_pred"] = test["lag7"] * 1.2

    y = test["actual_demand"]
    total = len(test)

    categories = {
        "under_forecast":   int(((y > test["p90_pred"]) & (test["disruption_flag"]==0)).sum()),
        "over_forecast":    int(((y < test["p10_pred"]) & (test["disruption_flag"]==0)).sum()),
        "disrupt_failure":  int(((test["disruption_flag"]==1) & (y > test["p90_pred"])).sum()),
        "festival_miss":    int(((test["festival"]==1) & (y > test["p90_pred"])).sum()),
        "covered":          int(((y >= test["p10_pred"]) & (y <= test["p90_pred"])).sum()),
    }

    report = {
        "total_test_rows": total,
        "error_categories": [
            {"name":"Under-forecast","count":categories["under_forecast"],"pct":round(categories["under_forecast"]/total*100,1),"impact":"Stockout risk"},
            {"name":"Over-forecast","count":categories["over_forecast"],"pct":round(categories["over_forecast"]/total*100,1),"impact":"Excess inventory cost"},
            {"name":"Disruption failure","count":categories["disrupt_failure"],"pct":round(categories["disrupt_failure"]/total*100,1),"impact":"Fulfillment gap during disruption"},
            {"name":"Festival miss","count":categories["festival_miss"],"pct":round(categories["festival_miss"]/total*100,1),"impact":"Demand spike missed"},
            {"name":"Within interval","count":categories["covered"],"pct":round(categories["covered"]/total*100,1),"impact":"None — correctly bounded"},
        ],
        "coverage_80": round(float((y >= test["p10_pred"]).mean() and (y <= test["p90_pred"]).mean()), 3),
    }

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/error_analysis.json","w") as f:
        json.dump(report, f, indent=2)

    print("Error Analysis Complete:")
    for cat in report["error_categories"]:
        print(f"  {cat['name']:30s} {cat['count']:5d} rows ({cat['pct']}%)")
    return report

if __name__ == "__main__":
    run_error_analysis()
