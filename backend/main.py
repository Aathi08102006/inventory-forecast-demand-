"""Demandly AI demo server (standard-library, no external service required)."""
import json, os, urllib.parse
from datetime import date
from http.server import HTTPServer, BaseHTTPRequestHandler

# When the optional ML dependencies are installed, serve the bundled quantile
# models.  The deterministic fallback keeps the demo runnable in a clean Python
# installation, rather than failing before the UI can open.
try:
    from backend.services.forecast_service import forecast as learned_forecast
    MODEL_AVAILABLE = True
except Exception:
    learned_forecast = None
    MODEL_AVAILABLE = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")
RESULTS = os.path.join(ROOT, "evaluation", "results")
DECISIONS = os.path.join(ROOT, "data", "planner_decisions.json")
STORES = {"S011":{"store_name":"Madurai Central","city":"Madurai","store_type":"Large Supermarket","store_size":"large"},"S017":{"store_name":"Salem Hub","city":"Salem","store_type":"Wholesale","store_size":"large"},"S037":{"store_name":"Coimbatore Market","city":"Coimbatore","store_type":"Supermarket","store_size":"medium"},"S042":{"store_name":"Trichy Bazaar","city":"Trichy","store_type":"Neighborhood","store_size":"small"},"S051":{"store_name":"Chennai Express","city":"Chennai","store_type":"Large Supermarket","store_size":"large"},"S063":{"store_name":"Erode Store","city":"Erode","store_type":"Neighborhood","store_size":"small"}}
PRODUCTS = {"P001":{"product_name":"Cold Drinks 2L","category":"Beverages","weather":.90,"festival":.60},"P002":{"product_name":"Mineral Water 1L","category":"Beverages","weather":.80,"festival":.40},"P003":{"product_name":"Milk 1L","category":"Dairy","weather":.10,"festival":.30},"P005":{"product_name":"Chips 200g","category":"Snacks","weather":.20,"festival":.70},"P009":{"product_name":"Ice Cream 500ml","category":"Frozen","weather":.95,"festival":.50},"P010":{"product_name":"Rice 5kg","category":"Staples","weather":.00,"festival":.60}}
SIZE={"large":2,"medium":1.3,"small":.7}

def number(src,key,default):
    try: return float(src.get(key,default))
    except (ValueError,TypeError): return default

def fallback_forecast(store_id,product_id,raw=None):
    raw=raw or {}; store=STORES.get(store_id,STORES["S011"]); product=PRODUCTS.get(product_id,PRODUCTS["P001"])
    t=number(raw,"temperature",32); rain=number(raw,"rainfall",0); fest=max(0,min(1,number(raw,"festival_intensity",0))); discount=max(0,min(40,number(raw,"discount",0))); cap=max(0,min(100,number(raw,"warehouse_capacity",100))); delay=max(0,number(raw,"delivery_delay",0)); weekend=int(number(raw,"weekend",0)); month=int(number(raw,"month",date.today().month))
    seasonal={1:.9,2:.95,3:1,4:1.05,5:1.1,6:1.05,7:1,8:.95,9:1,10:1.05,11:1.15,12:1.2}.get(month,1); weather=product["weather"]; festival=product["festival"]
    point=max(5,round(180*SIZE[store["store_size"]]*seasonal*(1+weather*max(0,t-30)/15)*(1-weather*.2*min(rain/30,1))*(1+festival*fest*.8)*(1+.15*weekend)*(1+.3*discount/100)))
    uncertainty=.16+.10*fest+(.08 if t>38 else 0)+(.07 if delay else 0)+(.06 if cap<85 else 0); p10=max(1,round(point*(1-2*uncertainty))); p25=max(1,round(point*(1-uncertainty))); p75=round(point*(1+uncertainty)); p90=round(point*(1+2*uncertainty))
    factors=[]; score=0
    if fest>.5: factors.append("High festival intensity"); score+=30
    if t>37: factors.append("Temperature outside normal planning range"); score+=20
    if delay: factors.append(f"Delivery delay: {delay:g} hours"); score+=25
    if cap<80: factors.append(f"Warehouse capacity: {cap:g}%"); score+=25
    drivers=[]
    for name,pct,direction in [("Festival effect",festival*fest*.8*100,"up"),("Temperature",weather*max(0,t-30)/15*100,"up"),("Rainfall",weather*.2*min(rain/30,1)*100,"down"),("Weekend",15 if weekend else 0,"up"),("Price discount",discount*.3,"up"),("Seasonal pattern",(seasonal-1)*100,"up" if seasonal>=1 else "down")]:
        if abs(pct)>=.5: drivers.append({"name":name,"pct":round(pct,1),"direction":direction})
    width=p90-p10; unc="LOW" if width<120 else "MEDIUM" if width<260 else "HIGH" if width<420 else "VERY HIGH"; risk="LOW" if score<20 else "MEDIUM" if score<45 else "HIGH"
    return {"store_id":store_id,"product_id":product_id,"store_name":store["store_name"],"product_name":product["product_name"],"point_forecast":point,"p10":p10,"p25":p25,"p50":point,"p75":p75,"p90":p90,"range_80_low":p10,"range_80_high":p90,"uncertainty":unc,"risk_level":risk,"risk_factors":factors,"drivers":drivers,"params":{"temperature":t,"rainfall":rain,"festival_intensity":fest,"discount":discount,"warehouse_capacity":cap,"delivery_delay":delay,"weekend":weekend,"month":month},"recommendations":{"conservative":round(p90*1.05),"balanced":round((point+p75)/2),"aggressive":p25}}

def forecast(store_id, product_id, raw=None):
    if MODEL_AVAILABLE:
        return learned_forecast(store_id, product_id, raw or {})
    return fallback_forecast(store_id, product_id, raw)

def load(path,default):
    try:
        with open(path,encoding="utf-8") as f: return json.load(f)
    except (OSError,json.JSONDecodeError): return default

class App(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args): print("Demandly:",fmt%args)
    def send_json(self,payload,status=200):
        data=json.dumps(payload).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers(); self.wfile.write(data)
    def body(self):
        try: return json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))) or b"{}")
        except json.JSONDecodeError: return {}
    def do_GET(self):
        parsed=urllib.parse.urlparse(self.path); path=parsed.path; q=dict(urllib.parse.parse_qsl(parsed.query))
        if path in ("/","/index.html"):
            with open(os.path.join(FRONTEND,"index.html"),"rb") as f:
                data=f.read(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
        elif path=="/api/stores": self.send_json([{"store_id":k,**v} for k,v in STORES.items()])
        elif path=="/api/products": self.send_json([{"product_id":k,**v} for k,v in PRODUCTS.items()])
        elif path=="/api/forecast": self.send_json(forecast(q.get("store_id","S011"),q.get("product_id","P001"),q))
        elif path=="/api/evaluation": self.send_json(load(os.path.join(RESULTS,"model_results.json"),{}))
        elif path=="/api/errors": self.send_json(load(os.path.join(RESULTS,"error_analysis.json"),{}))
        elif path=="/api/planner-decisions": self.send_json(load(DECISIONS,[]))
        elif path=="/api/safety": self.send_json({"notice":"All records are synthetic demonstration data. Forecasts are estimates, not guarantees.","validation":"Prototype validation mode: no formal study with industry planners has been performed."})
        else: self.send_json({"error":"Not found"},404)
    def do_POST(self):
        path=urllib.parse.urlparse(self.path).path; body=self.body()
        if path in ("/api/forecast","/api/scenario"): self.send_json(forecast(body.get("store_id","S011"),body.get("product_id","P001"),body.get("params",body.get("overrides",{}))))
        elif path=="/api/planner-decision":
            records=load(DECISIONS,[]); body["recorded_on"]=str(date.today()); records.insert(0,body); os.makedirs(os.path.dirname(DECISIONS),exist_ok=True)
            with open(DECISIONS,"w",encoding="utf-8") as f: json.dump(records,f,indent=2)
            self.send_json({"status":"saved","decision":body})
        else: self.send_json({"error":"Not found"},404)

if __name__=="__main__":
    print("Demandly AI ready at http://localhost:8000")
    HTTPServer(("127.0.0.1",8000),App).serve_forever()
