"""
Demandly AI — Synthetic Dataset Generator
All data is SYNTHETIC, created for demonstration and evaluation purposes only.
No real customer, store, or demand data is used.
"""
import pandas as pd
import numpy as np
import os, json

np.random.seed(42)

STORES = [
    ("S011","Madurai Central","Madurai","Large Supermarket","large",2000,"urban"),
    ("S017","Salem Hub","Salem","Wholesale","large",3000,"urban"),
    ("S037","Coimbatore Market","Coimbatore","Supermarket","medium",1500,"urban"),
    ("S042","Trichy Bazaar","Trichy","Neighborhood","small",800,"suburban"),
    ("S051","Chennai Express","Chennai","Large Supermarket","large",2500,"urban"),
    ("S063","Erode Store","Erode","Neighborhood","small",600,"rural"),
    ("S071","Vellore Mart","Vellore","Supermarket","medium",1200,"suburban"),
    ("S082","Dindigul Grocery","Dindigul","Neighborhood","small",500,"rural"),
    ("S094","Tirunelveli Center","Tirunelveli","Supermarket","medium",1100,"suburban"),
    ("S105","Puducherry Shop","Puducherry","Neighborhood","small",700,"urban"),
]

PRODUCTS = [
    ("P001","Cold Drinks 2L","Beverages",50,0.9,0.6),
    ("P002","Mineral Water 1L","Beverages",20,0.8,0.4),
    ("P003","Milk 1L","Dairy",60,0.1,0.3),
    ("P004","Curd 500g","Dairy",45,0.3,0.5),
    ("P005","Chips 200g","Snacks",35,0.2,0.7),
    ("P006","Biscuits 400g","Snacks",40,0.1,0.5),
    ("P007","Tomatoes 1kg","Fresh Produce",30,0.4,0.3),
    ("P008","Onions 1kg","Fresh Produce",25,0.2,0.4),
    ("P009","Ice Cream 500ml","Frozen",80,0.95,0.5),
    ("P010","Rice 5kg","Staples",250,0.0,0.6),
]

FESTIVALS = [
    ("Pongal","2023-01-14",0.9,["Madurai","Trichy","Coimbatore","Salem"]),
    ("Tamil New Year","2023-04-14",0.8,["Madurai","Salem","Trichy","Coimbatore","Chennai"]),
    ("Eid ul-Fitr","2023-04-21",0.7,["Madurai","Chennai","Vellore"]),
    ("Diwali","2023-11-12",0.95,["Madurai","Salem","Coimbatore","Trichy","Chennai","Erode","Vellore"]),
    ("Christmas","2023-12-25",0.6,["Chennai","Puducherry","Vellore"]),
    ("New Year","2023-12-31",0.5,["Chennai","Coimbatore","Madurai"]),
    ("Pongal","2024-01-15",0.9,["Madurai","Trichy","Coimbatore","Salem"]),
    ("Tamil New Year","2024-04-14",0.8,["Madurai","Salem","Trichy","Coimbatore","Chennai"]),
    ("Ganesh Chaturthi","2024-09-07",0.7,["Chennai","Coimbatore","Madurai"]),
    ("Navratri","2024-10-03",0.6,["Coimbatore","Salem","Madurai"]),
    ("Diwali","2024-11-01",0.95,["Madurai","Salem","Coimbatore","Trichy","Chennai","Erode","Vellore"]),
    ("Christmas","2024-12-25",0.6,["Chennai","Puducherry","Vellore"]),
]

def get_festival(date, city):
    best, best_int, best_dist = "None", 0.0, 999
    for fname, fdate_s, intensity, cities in FESTIVALS:
        fdate = pd.Timestamp(fdate_s)
        dist = abs((date - fdate).days)
        if dist <= 5 and city in cities:
            eff = intensity * max(0.3, 1 - dist * 0.15)
            if dist < best_dist or eff > best_int:
                best, best_int, best_dist = fname, eff, dist
    return best, round(best_int, 3), best_dist if best_dist < 999 else -1

def gen_weather(date, city):
    month = date.month
    base_t = {1:22,2:25,3:29,4:33,5:36,6:33,7:30,8:30,9:29,10:27,11:24,12:22}[month]
    if city in ["Madurai","Trichy","Dindigul"]: base_t += 2
    elif city in ["Coimbatore","Erode"]: base_t -= 2
    temp = base_t + np.random.normal(0, 2.5)
    monsoon = month in [6,7,8,9,10]
    rain = np.random.exponential(15) if np.random.random() < (0.6 if monsoon else 0.1) else 0
    humidity = np.random.uniform(60,90) if monsoon else np.random.uniform(35,65)
    heatwave = int(temp > 40)
    if rain > 0: cond = "rainy"
    elif temp > 38: cond = "heatwave"
    elif temp > 32: cond = "hot"
    else: cond = "normal"
    return round(temp,1), round(rain,1), round(humidity,1), cond, heatwave

def generate():
    dates = pd.date_range("2023-01-01", "2024-12-31")
    size_factor = {"large":2.0,"medium":1.3,"small":0.7}
    seasonal = {1:0.9,2:0.95,3:1.0,4:1.05,5:1.1,6:1.05,7:1.0,8:0.95,9:1.0,10:1.05,11:1.15,12:1.2}
    records = []
    for date in dates:
        dow = date.dayofweek
        is_weekend = int(dow >= 5)
        s_fact = seasonal[date.month]
        for (sid,sname,city,stype,ssize,cvol,loctype) in STORES:
            sf = size_factor[ssize]
            temp, rain, hum, cond, hw = gen_weather(date, city)
            fname, fint, fdist = get_festival(date, city)
            for (pid,pname,pcat,base_price,ws,fs) in PRODUCTS:
                base_demand = 120 * sf * (1 + 0.3*np.random.random())
                temp_eff = 1 + ws * max(0,(temp-30)/15)
                rain_eff = 1 - ws * 0.2 * min(rain/30, 1.0)
                fest_eff = 1 + fs * fint * 0.8
                cal_eff = (1.15 if is_weekend else 1.0) * s_fact
                discount = 0; promo = 0
                if np.random.random() < 0.05:
                    discount = np.random.choice([5,10,15,20])
                    promo = 1
                price = round(base_price * (1 - discount/100), 2)
                price_eff = 1 - 0.3*(discount/100)
                del_delay = 0; wh_cap = 100; disrupt = 0
                if np.random.random() < 0.03:
                    del_delay = int(np.random.choice([6,12,18,24]))
                    disrupt = 1
                if np.random.random() < 0.02:
                    wh_cap = int(np.random.randint(60,90))
                    disrupt = 1
                true_demand = base_demand * temp_eff * rain_eff * fest_eff * cal_eff * price_eff
                noise = np.random.normal(1, 0.12)
                actual = max(1, round(true_demand * noise))
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": sid, "store_name": sname, "store_type": stype,
                    "store_size": ssize, "city": city, "location_type": loctype,
                    "product_id": pid, "product_name": pname, "product_category": pcat,
                    "price": price, "base_price": base_price,
                    "discount": discount, "promotion": promo,
                    "temperature": temp, "rainfall": rain, "humidity": hum,
                    "weather_condition": cond, "heatwave": hw,
                    "festival": int(fname != "None"), "festival_name": fname,
                    "festival_intensity": fint, "days_to_festival": fdist,
                    "holiday": int(fname != "None"),
                    "day_of_week": dow, "weekend": is_weekend, "month": date.month,
                    "warehouse_capacity": wh_cap, "delivery_delay": del_delay,
                    "disruption_flag": disrupt, "actual_demand": actual,
                })
    df = pd.DataFrame(records)
    os.makedirs("dataset", exist_ok=True)
    df.to_csv("dataset/demand_history.csv", index=False)
    stores_df = pd.DataFrame([{"store_id":s[0],"store_name":s[1],"city":s[2],
        "store_type":s[3],"store_size":s[4],"customer_volume":s[5],"location_type":s[6]} for s in STORES])
    stores_df.to_csv("dataset/stores.csv", index=False)
    products_df = pd.DataFrame([{"product_id":p[0],"product_name":p[1],"category":p[2],
        "base_price":p[3],"weather_sensitivity":p[4],"festival_sensitivity":p[5]} for p in PRODUCTS])
    products_df.to_csv("dataset/products.csv", index=False)
    print(f"Generated {len(df):,} rows")
    return df

if __name__ == "__main__":
    df = generate()
    print(df.dtypes)
    print(df.head(2).to_string())
