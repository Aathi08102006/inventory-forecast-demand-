"""
Demandly AI — Forecasting Model Training
Uses quantile regression with GradientBoostingRegressor to produce
point forecasts and prediction intervals.
"""
import pandas as pd
import numpy as np
import joblib, os, json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
FEATURES = [
    "price","discount","promotion","temperature","rainfall","humidity",
    "heatwave","festival","festival_intensity","days_to_festival","holiday",
    "day_of_week","weekend","month","warehouse_capacity","delivery_delay",
    "disruption_flag","store_size_enc","store_type_enc","product_cat_enc",
    "lag1","lag7","roll7_mean","roll7_std",
]

def add_lag_features(df):
    df = df.sort_values(["store_id","product_id","date"]).copy()
    key = ["store_id","product_id"]
    df["lag1"] = df.groupby(key)["actual_demand"].shift(1).fillna(df["actual_demand"].mean())
    df["lag7"] = df.groupby(key)["actual_demand"].shift(7).fillna(df["actual_demand"].mean())
    df["roll7_mean"] = df.groupby(key)["actual_demand"].shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).mean()).fillna(df["actual_demand"].mean())
    df["roll7_std"] = df.groupby(key)["actual_demand"].shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).std()).fillna(50)
    return df

def train():
    print("Loading data...")
    df = pd.read_csv("dataset/demand_history.csv", parse_dates=["date"])
    df = add_lag_features(df)

    # Encode categoricals
    enc_size = OrdinalEncoder()
    enc_type = OrdinalEncoder()
    enc_cat  = OrdinalEncoder()
    df["store_size_enc"] = enc_size.fit_transform(df[["store_size"]])
    df["store_type_enc"] = enc_type.fit_transform(df[["store_type"]])
    df["product_cat_enc"] = enc_cat.fit_transform(df[["product_category"]])

    # Train/test split — last 90 days as test
    cutoff = df["date"].max() - pd.Timedelta(days=90)
    train_df = df[df["date"] <= cutoff]
    test_df  = df[df["date"] >  cutoff]

    X_train = train_df[FEATURES].fillna(0)
    y_train = train_df["actual_demand"]
    X_test  = test_df[FEATURES].fillna(0)
    y_test  = test_df["actual_demand"]

    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Baseline — seasonal naive (lag7)
    baseline_preds = test_df["lag7"].values
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    baseline_mape = np.mean(np.abs((y_test - baseline_preds) / (y_test + 1e-9))) * 100

    print(f"Baseline MAE={baseline_mae:.1f}  RMSE={baseline_rmse:.1f}  MAPE={baseline_mape:.1f}%")

    # Train quantile models
    models = {}
    for q in QUANTILES:
        print(f"  Training quantile {q}...")
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=200, max_depth=5,
            learning_rate=0.08, subsample=0.8,
            random_state=42
        )
        m.fit(X_train, y_train)
        models[str(q)] = m

    # Predictions
    preds = {str(q): models[str(q)].predict(X_test) for q in QUANTILES}
    point_pred = preds["0.5"]

    model_mae   = mean_absolute_error(y_test, point_pred)
    model_rmse  = np.sqrt(mean_squared_error(y_test, point_pred))
    model_mape  = np.mean(np.abs((y_test.values - point_pred) / (y_test.values + 1e-9))) * 100

    # Calibration — what fraction of actuals fall inside each interval?
    coverage_80 = np.mean((y_test.values >= preds["0.1"]) & (y_test.values <= preds["0.9"]))
    coverage_50 = np.mean((y_test.values >= preds["0.25"]) & (y_test.values <= preds["0.75"]))

    print(f"Model  MAE={model_mae:.1f}  RMSE={model_rmse:.1f}  MAPE={model_mape:.1f}%")
    print(f"80% coverage: {coverage_80:.3f} (target 0.80)")
    print(f"50% coverage: {coverage_50:.3f} (target 0.50)")

    # Feature importance from P50 model
    fi = pd.Series(models["0.5"].feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop features:")
    print(fi.head(8).to_string())

    os.makedirs("models", exist_ok=True)
    joblib.dump(models, "models/quantile_models.pkl")
    joblib.dump(enc_size, "models/enc_size.pkl")
    joblib.dump(enc_type, "models/enc_type.pkl")
    joblib.dump(enc_cat,  "models/enc_cat.pkl")

    results = {
        "baseline": {"mae": round(baseline_mae,2), "rmse": round(baseline_rmse,2), "mape": round(baseline_mape,2)},
        "demandly": {"mae": round(model_mae,2), "rmse": round(model_rmse,2), "mape": round(model_mape,2)},
        "coverage": {"pct80": round(coverage_80,3), "pct50": round(coverage_50,3)},
        "feature_importance": fi.head(10).round(4).to_dict(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/model_results.json","w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to models/ and evaluation/results/model_results.json")
    return results

if __name__ == "__main__":
    train()
