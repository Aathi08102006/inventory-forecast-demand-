"""
Demandly AI — Forecast Service
Loads trained models and produces forecasts with prediction intervals.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import date, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_artifacts():
    models = joblib.load(os.path.join(BASE, "models/quantile_models.pkl"))
    enc_size = joblib.load(os.path.join(BASE, "models/enc_size.pkl"))
    enc_type = joblib.load(os.path.join(BASE, "models/enc_type.pkl"))
    enc_cat  = joblib.load(os.path.join(BASE, "models/enc_cat.pkl"))
    df = pd.read_csv(os.path.join(BASE, "dataset/demand_history.csv"), parse_dates=["date"])
    return models, enc_size, enc_type, enc_cat, df

try:
    MODELS, ENC_SIZE, ENC_TYPE, ENC_CAT, HISTORY = load_artifacts()
    LOADED = True
except Exception as e:
    print(f"WARNING: Could not load models: {e}")
    LOADED = False
    HISTORY = pd.DataFrame()

FEATURES = [
    "price","discount","promotion","temperature","rainfall","humidity",
    "heatwave","festival","festival_intensity","days_to_festival","holiday",
    "day_of_week","weekend","month","warehouse_capacity","delivery_delay",
    "disruption_flag","store_size_enc","store_type_enc","product_cat_enc",
    "lag1","lag7","roll7_mean","roll7_std",
]

STORES_META = {
    "S011": {"store_name":"Madurai Central","city":"Madurai","store_type":"Large Supermarket","store_size":"large","customer_volume":2000,"location_type":"urban"},
    "S017": {"store_name":"Salem Hub","city":"Salem","store_type":"Wholesale","store_size":"large","customer_volume":3000,"location_type":"urban"},
    "S037": {"store_name":"Coimbatore Market","city":"Coimbatore","store_type":"Supermarket","store_size":"medium","customer_volume":1500,"location_type":"urban"},
    "S042": {"store_name":"Trichy Bazaar","city":"Trichy","store_type":"Neighborhood","store_size":"small","customer_volume":800,"location_type":"suburban"},
    "S051": {"store_name":"Chennai Express","city":"Chennai","store_type":"Large Supermarket","store_size":"large","customer_volume":2500,"location_type":"urban"},
    "S063": {"store_name":"Erode Store","city":"Erode","store_type":"Neighborhood","store_size":"small","customer_volume":600,"location_type":"rural"},
    "S071": {"store_name":"Vellore Mart","city":"Vellore","store_type":"Supermarket","store_size":"medium","customer_volume":1200,"location_type":"suburban"},
    "S082": {"store_name":"Dindigul Grocery","city":"Dindigul","store_type":"Neighborhood","store_size":"small","customer_volume":500,"location_type":"rural"},
    "S094": {"store_name":"Tirunelveli Center","city":"Tirunelveli","store_type":"Supermarket","store_size":"medium","customer_volume":1100,"location_type":"suburban"},
    "S105": {"store_name":"Puducherry Shop","city":"Puducherry","store_type":"Neighborhood","store_size":"small","customer_volume":700,"location_type":"urban"},
}

PRODUCTS_META = {
    "P001": {"product_name":"Cold Drinks 2L","category":"Beverages","base_price":50,"weather_sensitivity":0.9,"festival_sensitivity":0.6},
    "P002": {"product_name":"Mineral Water 1L","category":"Beverages","base_price":20,"weather_sensitivity":0.8,"festival_sensitivity":0.4},
    "P003": {"product_name":"Milk 1L","category":"Dairy","base_price":60,"weather_sensitivity":0.1,"festival_sensitivity":0.3},
    "P004": {"product_name":"Curd 500g","category":"Dairy","base_price":45,"weather_sensitivity":0.3,"festival_sensitivity":0.5},
    "P005": {"product_name":"Chips 200g","category":"Snacks","base_price":35,"weather_sensitivity":0.2,"festival_sensitivity":0.7},
    "P006": {"product_name":"Biscuits 400g","category":"Snacks","base_price":40,"weather_sensitivity":0.1,"festival_sensitivity":0.5},
    "P007": {"product_name":"Tomatoes 1kg","category":"Fresh Produce","base_price":30,"weather_sensitivity":0.4,"festival_sensitivity":0.3},
    "P008": {"product_name":"Onions 1kg","category":"Fresh Produce","base_price":25,"weather_sensitivity":0.2,"festival_sensitivity":0.4},
    "P009": {"product_name":"Ice Cream 500ml","category":"Frozen","base_price":80,"weather_sensitivity":0.95,"festival_sensitivity":0.5},
    "P010": {"product_name":"Rice 5kg","category":"Staples","base_price":250,"weather_sensitivity":0.0,"festival_sensitivity":0.6},
}

def _get_lags(store_id, product_id, ref_date):
    if HISTORY.empty:
        return 200, 200, 200, 60
    mask = (HISTORY["store_id"] == store_id) & (HISTORY["product_id"] == product_id)
    sub = HISTORY[mask].sort_values("date")
    recent = sub[sub["date"] < pd.Timestamp(ref_date)].tail(14)
    if len(recent) == 0:
        return 200, 200, 200, 60
    lag1 = recent.iloc[-1]["actual_demand"] if len(recent) >= 1 else 200
    lag7 = recent.iloc[-7]["actual_demand"] if len(recent) >= 7 else lag1
    roll7 = recent["actual_demand"].tail(7).mean()
    roll7_std = recent["actual_demand"].tail(7).std() or 60
    return lag1, lag7, roll7, roll7_std

def _encode(val, enc, col_name):
    try:
        arr = enc.transform([[val]])
        return float(arr[0][0])
    except:
        return 0.0

def compute_drivers(params, meta_store, meta_product):
    """Compute percentage contributions of each driver."""
    ws = meta_product["weather_sensitivity"]
    fs = meta_product["festival_sensitivity"]
    t = params.get("temperature", 30)
    rain = params.get("rainfall", 0)
    fint = params.get("festival_intensity", 0)
    discount = params.get("discount", 0)
    weekend = params.get("weekend", 0)
    month = params.get("month", 6)

    temp_eff = ws * max(0, (t - 30) / 15) * 100
    rain_eff = -ws * 0.2 * min(rain / 30, 1.0) * 100
    fest_eff = fs * fint * 0.8 * 100
    weekend_eff = 15 if weekend else 0
    price_eff = -0.3 * (discount / 100) * 100
    seasonal_eff = {1:-10,2:-5,3:0,4:5,5:10,6:5,7:0,8:-5,9:0,10:5,11:15,12:20}.get(month, 0)

    drivers = [
        {"name": "Festival effect", "pct": round(fest_eff, 1), "direction": "up" if fest_eff > 0 else "neutral"},
        {"name": "Temperature", "pct": round(temp_eff, 1), "direction": "up" if temp_eff > 0 else "neutral"},
        {"name": "Weekend effect", "pct": round(weekend_eff, 1), "direction": "up" if weekend_eff > 0 else "neutral"},
        {"name": "Seasonal trend", "pct": round(seasonal_eff, 1), "direction": "up" if seasonal_eff > 0 else ("down" if seasonal_eff < 0 else "neutral")},
        {"name": "Price/Discount", "pct": round(price_eff, 1), "direction": "down" if price_eff < 0 else "up"},
        {"name": "Rainfall effect", "pct": round(rain_eff, 1), "direction": "down" if rain_eff < 0 else "neutral"},
    ]
    return [d for d in drivers if abs(d["pct"]) > 0.5]

def forecast(store_id, product_id, params=None, ref_date=None):
    if ref_date is None:
        ref_date = date.today()
    if params is None:
        params = {}

    meta_store = STORES_META.get(store_id, STORES_META["S011"])
    meta_product = PRODUCTS_META.get(product_id, PRODUCTS_META["P001"])

    t = params.get("temperature", 32.0)
    rain = params.get("rainfall", 0.0)
    humidity = params.get("humidity", 55.0)
    fint = params.get("festival_intensity", 0.0)
    festival = int(fint > 0)
    discount = params.get("discount", 0)
    promotion = int(discount > 0)
    price = meta_product["base_price"] * (1 - discount / 100)
    warehouse_cap = params.get("warehouse_capacity", 100)
    del_delay = params.get("delivery_delay", 0)
    disruption = int(del_delay > 0 or warehouse_cap < 100)
    heatwave = int(t > 40)

    d = pd.Timestamp(ref_date)
    dow = d.dayofweek
    weekend = int(dow >= 5)
    month = d.month
    holiday = festival

    lag1, lag7, roll7, roll7_std = _get_lags(store_id, product_id, ref_date)

    row = {
        "price": price, "discount": discount, "promotion": promotion,
        "temperature": t, "rainfall": rain, "humidity": humidity, "heatwave": heatwave,
        "festival": festival, "festival_intensity": fint, "days_to_festival": params.get("days_to_festival", -1),
        "holiday": holiday, "day_of_week": dow, "weekend": weekend, "month": month,
        "warehouse_capacity": warehouse_cap, "delivery_delay": del_delay,
        "disruption_flag": disruption,
        "store_size_enc": _encode(meta_store["store_size"], ENC_SIZE, "store_size"),
        "store_type_enc": _encode(meta_store["store_type"], ENC_TYPE, "store_type"),
        "product_cat_enc": _encode(meta_product["category"], ENC_CAT, "product_category"),
        "lag1": lag1, "lag7": lag7, "roll7_mean": roll7, "roll7_std": roll7_std,
    }

    X = pd.DataFrame([row])[FEATURES]

    if LOADED:
        p10  = max(1, float(MODELS["0.1"].predict(X)[0]))
        p25  = max(1, float(MODELS["0.25"].predict(X)[0]))
        p50  = max(1, float(MODELS["0.5"].predict(X)[0]))
        p75  = max(1, float(MODELS["0.75"].predict(X)[0]))
        p90  = max(1, float(MODELS["0.9"].predict(X)[0]))
    else:
        # Fallback deterministic estimate
        base = 200 * {"large":2.0,"medium":1.3,"small":0.7}.get(meta_store["store_size"],1.0)
        p50 = base * (1 + meta_product["weather_sensitivity"] * max(0,(t-30)/15))
        p50 *= (1 + meta_product["festival_sensitivity"] * fint * 0.8)
        p10 = p50 * 0.75; p25 = p50 * 0.87; p75 = p50 * 1.15; p90 = p50 * 1.30

    width_80 = p90 - p10
    uncertainty = "LOW" if width_80 < 80 else ("MEDIUM" if width_80 < 180 else ("HIGH" if width_80 < 300 else "VERY HIGH"))

    # Risk score
    risk_factors = []
    risk_score = 0
    if fint > 0.5: risk_factors.append("High festival intensity"); risk_score += 30
    if t > 37: risk_factors.append("Extreme temperature"); risk_score += 20
    if del_delay > 0: risk_factors.append(f"Delivery delay {del_delay}h"); risk_score += 25
    if warehouse_cap < 80: risk_factors.append(f"Warehouse capacity {warehouse_cap}%"); risk_score += 20
    if disruption: risk_factors.append("Active disruption"); risk_score += 15
    risk_level = "LOW" if risk_score < 20 else ("MEDIUM" if risk_score < 45 else "HIGH")

    drivers = compute_drivers(
        {"temperature":t,"rainfall":rain,"festival_intensity":fint,
         "discount":discount,"weekend":weekend,"month":month},
        meta_store, meta_product
    )

    # Planning recommendations
    available_stock = params.get("available_stock", int(p50 * 1.1))
    rec_conservative = int(p90 * 1.05)
    rec_balanced     = int((p50 + p75) / 2)
    rec_aggressive   = int(p25)

    return {
        "store_id": store_id,
        "product_id": product_id,
        "store_name": meta_store["store_name"],
        "product_name": meta_product["product_name"],
        "forecast_date": str(ref_date),
        "point_forecast": round(p50),
        "p10": round(p10), "p25": round(p25), "p50": round(p50),
        "p75": round(p75), "p90": round(p90),
        "range_80_low": round(p10), "range_80_high": round(p90),
        "range_50_low": round(p25), "range_50_high": round(p75),
        "uncertainty": uncertainty,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "drivers": drivers,
        "params_used": row,
        "recommendations": {
            "conservative": rec_conservative,
            "balanced": rec_balanced,
            "aggressive": rec_aggressive,
        },
        "meta": {
            "store": meta_store,
            "product": meta_product,
        }
    }

def get_scenario_forecast(store_id, product_id, scenario_overrides, ref_date=None):
    """Run forecast under modified scenario assumptions."""
    return forecast(store_id, product_id, params=scenario_overrides, ref_date=ref_date)

def get_store_overview(ref_date=None):
    """Return overview stats for all stores."""
    if ref_date is None:
        ref_date = date.today()
    results = []
    for sid, smeta in list(STORES_META.items())[:5]:
        pid = "P001"
        pmeta = PRODUCTS_META[pid]
        f = forecast(sid, pid, ref_date=ref_date)
        results.append({
            "store_id": sid,
            "city": smeta["city"],
            "store_type": smeta["store_type"],
            "forecast_units": f["point_forecast"],
            "range_low": f["range_80_low"],
            "range_high": f["range_80_high"],
            "uncertainty": f["uncertainty"],
            "risk_level": f["risk_level"],
        })
    return results

if __name__ == "__main__":
    f = forecast("S011", "P001")
    print(json.dumps({k:v for k,v in f.items() if k != "params_used"}, indent=2))
