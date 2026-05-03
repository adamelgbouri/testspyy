import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER — CFCAP registry + download functions
# ══════════════════════════════════════════════════════════════════════════════

TV_USERNAME = ""
TV_PASSWORD = ""
EIA_API_KEY = ""

MONTH_CODES = list("FGHJKMNQUVXZ")
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

_BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR       = _BASE_DIR / "data"
CURVES_DIR     = DATA_DIR / "curves"
DASHBOARDS_DIR = DATA_DIR / "dashboards"
LOGS_DIR       = DATA_DIR / "logs"
EIA_CACHE_DIR  = DATA_DIR / "eia_cache"
OPCAP_DIR      = _BASE_DIR / "data_opcap"

for _d in [CURVES_DIR, DASHBOARDS_DIR, LOGS_DIR, EIA_CACHE_DIR, OPCAP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

COMMODITY_REGISTRY = {
    "Energy": {
        "WTI Crude Oil": dict(name="WTI Crude Oil", unit="$/bbl",
            source="yahoo", yf_fmt="CL{M}{YY}.NYM",
            active_months="FGHJKMNQUVXZ", liquid_months=18,
            storage_cost=0.096, synthetic_spot=63.0,
            ns_bounds=([10,-150,-150,0.5],[300,150,150,60])),
        "Brent Crude Oil": dict(name="Brent Crude Oil", unit="$/bbl",
            source="yahoo", yf_fmt="BZ{M}{YY}.NYM",
            active_months="FGHJKMNQUVXZ", liquid_months=18,
            storage_cost=0.096, synthetic_spot=66.0,
            ns_bounds=([10,-150,-150,0.5],[300,150,150,60])),
        "Natural Gas (Henry Hub)": dict(name="Natural Gas (Henry Hub)", unit="$/MMBtu",
            source="yahoo", yf_fmt="NG{M}{YY}.NYM",
            active_months="FGHJKMNQUVXZ", liquid_months=12,
            storage_cost=0.120, synthetic_spot=3.2,
            ns_bounds=([0.1,-20,-20,0.5],[30,20,20,60])),
        "RBOB Gasoline": dict(name="RBOB Gasoline", unit="$/gallon",
            source="yahoo", yf_fmt="RB{M}{YY}.NYM",
            active_months="FGHJKMNQUVXZ", liquid_months=12,
            storage_cost=0.084, synthetic_spot=2.5,
            ns_bounds=([0.3,-5,-5,0.5],[15,5,5,60])),
        "Heating Oil (ULSD)": dict(name="Heating Oil (ULSD)", unit="$/gallon",
            source="yahoo", yf_fmt="HO{M}{YY}.NYM",
            active_months="FGHJKMNQUVXZ", liquid_months=12,
            storage_cost=0.084, synthetic_spot=2.7,
            ns_bounds=([0.3,-5,-5,0.5],[15,5,5,60])),
        "Gasoil ICE": dict(name="Gasoil ICE", unit="$/mt",
            source="yahoo", yf_fmt="LGO{M}{YY}.NYM",
            active_months="FGHJKMNQUVXZ", liquid_months=12,
            storage_cost=0.072, synthetic_spot=780.0,
            ns_bounds=([100,-800,-800,0.5],[3000,800,800,60])),
        "Jet Kerosene CIF NWE (Platts)": dict(name="Jet Kerosene CIF NWE (Platts)", unit="$/mt",
            source="tradingview", tv_prefix="AUJ", tv_exchange="NYMEX",
            active_months="FGHJKMNQUVXZ", liquid_months=12,
            storage_cost=0.072, synthetic_spot=850.0,
            ns_bounds=([100,-800,-800,0.5],[3000,800,800,60])),
    },
    "Metals": {
        "Gold": dict(name="Gold", unit="$/troy oz",
            source="yahoo", yf_fmt="GC{M}{YY}.CMX",
            active_months="GJMQVZ", liquid_months=8,
            storage_cost=0.024, synthetic_spot=3300.0,
            ns_bounds=([500,-2000,-2000,0.5],[8000,2000,2000,60])),
        "Silver": dict(name="Silver", unit="$/troy oz",
            source="yahoo", yf_fmt="SI{M}{YY}.CMX",
            active_months="HKNUZ", liquid_months=6,
            storage_cost=0.036, synthetic_spot=33.0,
            ns_bounds=([3,-100,-100,0.5],[500,100,100,60])),
        "Copper": dict(name="Copper", unit="$/lb",
            source="yahoo", yf_fmt="HG{M}{YY}.CMX",
            active_months="HKNUZ", liquid_months=8,
            storage_cost=0.048, synthetic_spot=4.6,
            ns_bounds=([0.5,-10,-10,0.5],[20,10,10,60])),
        "Platinum": dict(name="Platinum", unit="$/troy oz",
            source="yahoo", yf_fmt="PL{M}{YY}.NYM",
            active_months="FJNV", liquid_months=6,
            storage_cost=0.030, synthetic_spot=1000.0,
            ns_bounds=([100,-1500,-1500,0.5],[5000,1500,1500,60])),
        "Palladium": dict(name="Palladium", unit="$/troy oz",
            source="yahoo", yf_fmt="PA{M}{YY}.NYM",
            active_months="HMUZ", liquid_months=6,
            storage_cost=0.030, synthetic_spot=1100.0,
            ns_bounds=([100,-3000,-3000,0.5],[8000,3000,3000,60])),
    },
    "Agriculture": {
        "Corn": dict(name="Corn", unit="c/bushel",
            source="yahoo", yf_fmt="ZC{M}{YY}.CBT",
            active_months="HKNUZ", liquid_months=8,
            storage_cost=0.060, synthetic_spot=470.0,
            ns_bounds=([100,-600,-600,0.5],[2000,600,600,60])),
        "Wheat (CBOT)": dict(name="Wheat (CBOT)", unit="c/bushel",
            source="yahoo", yf_fmt="ZW{M}{YY}.CBT",
            active_months="HKNUZ", liquid_months=8,
            storage_cost=0.060, synthetic_spot=560.0,
            ns_bounds=([100,-600,-600,0.5],[3000,600,600,60])),
        "Soybeans": dict(name="Soybeans", unit="c/bushel",
            source="yahoo", yf_fmt="ZS{M}{YY}.CBT",
            active_months="FHKNQUX", liquid_months=8,
            storage_cost=0.060, synthetic_spot=1000.0,
            ns_bounds=([300,-800,-800,0.5],[3000,800,800,60])),
        "Sugar #11": dict(name="Sugar #11", unit="c/lb",
            source="yahoo", yf_fmt="SB{M}{YY}.NYB",
            active_months="HKNV", liquid_months=6,
            storage_cost=0.048, synthetic_spot=19.0,
            ns_bounds=([3,-30,-30,0.5],[100,30,30,60])),
        "Coffee (Arabica)": dict(name="Coffee (Arabica)", unit="c/lb",
            source="yahoo", yf_fmt="KC{M}{YY}.NYB",
            active_months="HKNUZ", liquid_months=6,
            storage_cost=0.048, synthetic_spot=350.0,
            ns_bounds=([30,-500,-500,0.5],[2000,500,500,60])),
        "Cocoa": dict(name="Cocoa", unit="$/mt",
            source="yahoo", yf_fmt="CC{M}{YY}.NYB",
            active_months="HKNUZ", liquid_months=6,
            storage_cost=0.048, synthetic_spot=8000.0,
            ns_bounds=([500,-8000,-8000,0.5],[20000,8000,8000,60])),
        "Live Cattle": dict(name="Live Cattle", unit="c/lb",
            source="yahoo", yf_fmt="LE{M}{YY}.CME",
            active_months="GJMQVZ", liquid_months=8,
            storage_cost=0.036, synthetic_spot=185.0,
            ns_bounds=([50,-100,-100,0.5],[400,100,100,60])),
        "Lean Hogs": dict(name="Lean Hogs", unit="c/lb",
            source="yahoo", yf_fmt="HE{M}{YY}.CME",
            active_months="GJKMNQVZ", liquid_months=6,
            storage_cost=0.036, synthetic_spot=92.0,
            ns_bounds=([20,-80,-80,0.5],[250,80,80,60])),
    },
    "Base Metals": {
        "LME Copper": dict(name="LME Copper", unit="$/mt",
            source="tradingview", tv_prefix="COPPER", tv_exchange="LME",
            active_months="FGHJKMNQUVXZ", liquid_months=15,
            storage_cost=0.048, synthetic_spot=9800.0,
            ns_bounds=([2000,-5000,-5000,0.5],[25000,5000,5000,60])),
        "LME Aluminum": dict(name="LME Aluminum", unit="$/mt",
            source="tradingview", tv_prefix="ALUMINUM", tv_exchange="LME",
            active_months="FGHJKMNQUVXZ", liquid_months=15,
            storage_cost=0.048, synthetic_spot=2400.0,
            ns_bounds=([500,-2000,-2000,0.5],[8000,2000,2000,60])),
        "LME Zinc": dict(name="LME Zinc", unit="$/mt",
            source="tradingview", tv_prefix="ZINC", tv_exchange="LME",
            active_months="FGHJKMNQUVXZ", liquid_months=15,
            storage_cost=0.048, synthetic_spot=2800.0,
            ns_bounds=([500,-2000,-2000,0.5],[8000,2000,2000,60])),
        "LME Nickel": dict(name="LME Nickel", unit="$/mt",
            source="tradingview", tv_prefix="NICKEL", tv_exchange="LME",
            active_months="FGHJKMNQUVXZ", liquid_months=15,
            storage_cost=0.048, synthetic_spot=16000.0,
            ns_bounds=([3000,-15000,-15000,0.5],[60000,15000,15000,60])),
        "LME Lead": dict(name="LME Lead", unit="$/mt",
            source="tradingview", tv_prefix="LEAD", tv_exchange="LME",
            active_months="FGHJKMNQUVXZ", liquid_months=15,
            storage_cost=0.048, synthetic_spot=2000.0,
            ns_bounds=([300,-1500,-1500,0.5],[6000,1500,1500,60])),
        "LME Tin": dict(name="LME Tin", unit="$/mt",
            source="tradingview", tv_prefix="TIN", tv_exchange="LME",
            active_months="FGHJKMNQUVXZ", liquid_months=12,
            storage_cost=0.048, synthetic_spot=32000.0,
            ns_bounds=([5000,-20000,-20000,0.5],[100000,20000,20000,60])),
    },
    "Energy (Additional)": {
        "Dutch TTF Natural Gas": dict(name="Dutch TTF Natural Gas", unit="EUR/MWh",
            source="tradingview", tv_prefix="TTFG", tv_exchange="ICEENDEX",
            active_months="FGHJKMNQUVXZ", liquid_months=24,
            storage_cost=0.120, synthetic_spot=35.0,
            ns_bounds=([2,-60,-60,0.5],[200,60,60,60])),
        "European Carbon (EUA)": dict(name="European Carbon (EUA)", unit="EUR/tCO2",
            source="tradingview", tv_prefix="EUAD", tv_exchange="ICE",
            active_months="HMNUZ", liquid_months=12,
            storage_cost=0.024, synthetic_spot=65.0,
            ns_bounds=([5,-80,-80,0.5],[200,80,80,60])),
        "Coal (API2 Rotterdam)": dict(name="Coal (API2 Rotterdam)", unit="$/mt",
            source="tradingview", tv_prefix="MTF", tv_exchange="ICE",
            active_months="FGHJKMNQUVXZ", liquid_months=12,
            storage_cost=0.048, synthetic_spot=110.0,
            ns_bounds=([20,-150,-150,0.5],[500,150,150,60])),
        "Uranium (UxC)": dict(name="Uranium (UxC)", unit="$/lb U3O8",
            source="tradingview", tv_prefix="UX1", tv_exchange="CME",
            active_months="HKNUZ", liquid_months=8,
            storage_cost=0.024, synthetic_spot=78.0,
            ns_bounds=([10,-80,-80,0.5],[300,80,80,60])),
    },
    "Freight": {
        "Capesize (BCI 5TC)": dict(name="Capesize (BCI 5TC)", unit="$/day",
            source="tradingview", tv_prefix="BCSA1", tv_exchange="CME",
            active_months="FGHJKMNQUVXZ", liquid_months=12,
            storage_cost=0.0, synthetic_spot=18000.0,
            ns_bounds=([1000,-20000,-20000,0.5],[80000,20000,20000,60])),
        "Panamax (BPI 4TC)": dict(name="Panamax (BPI 4TC)", unit="$/day",
            source="tradingview", tv_prefix="BPSA1", tv_exchange="CME",
            active_months="FGHJKMNQUVXZ", liquid_months=12,
            storage_cost=0.0, synthetic_spot=12000.0,
            ns_bounds=([500,-12000,-12000,0.5],[50000,12000,12000,60])),
    },
    "Carbon & Environmental": {
        "EU Carbon EUA": dict(name="EU Carbon EUA", unit="EUR/tCO2",
            source="tradingview", tv_prefix="EUAD", tv_exchange="ICE",
            active_months="HMNUZ", liquid_months=12,
            storage_cost=0.024, synthetic_spot=65.0,
            ns_bounds=([5,-80,-80,0.5],[200,80,80,60])),
        "UK Carbon UKA": dict(name="UK Carbon UKA", unit="GBP/tCO2",
            source="tradingview", tv_prefix="UKAD", tv_exchange="ICE",
            active_months="HMNUZ", liquid_months=12,
            storage_cost=0.024, synthetic_spot=40.0,
            ns_bounds=([3,-50,-50,0.5],[150,50,50,60])),
    },
}

# ── CODAP supplement: vol + convenience per commodity ─────────────────────────
OPCAP_PARAMS = {
    "WTI Crude Oil":              dict(vol=0.32, convenience=0.08),
    "Brent Crude Oil":            dict(vol=0.30, convenience=0.07),
    "Natural Gas (Henry Hub)":    dict(vol=0.55, convenience=0.10),
    "RBOB Gasoline":              dict(vol=0.36, convenience=0.07),
    "Heating Oil (ULSD)":         dict(vol=0.34, convenience=0.07),
    "Gasoil ICE":                 dict(vol=0.32, convenience=0.07),
    "Jet Kerosene CIF NWE (Platts)": dict(vol=0.35, convenience=0.08),
    "Gold":                       dict(vol=0.15, convenience=0.005),
    "Silver":                     dict(vol=0.28, convenience=0.010),
    "Copper":                     dict(vol=0.22, convenience=0.030),
    "Platinum":                   dict(vol=0.20, convenience=0.015),
    "Palladium":                  dict(vol=0.30, convenience=0.020),
    "Corn":                       dict(vol=0.25, convenience=0.04),
    "Wheat (CBOT)":               dict(vol=0.28, convenience=0.04),
    "Soybeans":                   dict(vol=0.23, convenience=0.05),
    "Sugar #11":                  dict(vol=0.30, convenience=0.04),
    "Coffee (Arabica)":           dict(vol=0.35, convenience=0.05),
    "Cocoa":                      dict(vol=0.32, convenience=0.04),
    "Live Cattle":                dict(vol=0.18, convenience=0.03),
    "Lean Hogs":                  dict(vol=0.25, convenience=0.03),
    "LME Copper":                 dict(vol=0.22, convenience=0.030),
    "LME Aluminum":               dict(vol=0.20, convenience=0.025),
    "LME Zinc":                   dict(vol=0.24, convenience=0.028),
    "LME Nickel":                 dict(vol=0.30, convenience=0.035),
    "LME Lead":                   dict(vol=0.22, convenience=0.025),
    "LME Tin":                    dict(vol=0.28, convenience=0.030),
    "Dutch TTF Natural Gas":      dict(vol=0.60, convenience=0.12),
    "European Carbon (EUA)":      dict(vol=0.35, convenience=0.02),
    "Coal (API2 Rotterdam)":      dict(vol=0.30, convenience=0.04),
    "Capesize (BCI 5TC)":         dict(vol=0.55, convenience=0.00),
    "Panamax (BPI 4TC)":          dict(vol=0.50, convenience=0.00),
    "EU Carbon EUA":              dict(vol=0.35, convenience=0.02),
}

def get_opcap_params(family: str, commodity: str) -> dict:
    cfg = COMMODITY_REGISTRY.get(family, {}).get(commodity, {})
    op  = OPCAP_PARAMS.get(commodity, {"vol": 0.30, "convenience": 0.05})
    return {
        "unit":          cfg.get("unit", "$/unit"),
        "spot":          cfg.get("synthetic_spot", 100.0),
        "storage":       cfg.get("storage_cost", 0.06),
        "vol":           op["vol"],
        "convenience":   op["convenience"],
        "liquid_months": cfg.get("liquid_months", 12),
        "source":        cfg.get("source", "yahoo"),
    }

CRACK_SPREADS = {
    "3-2-1 Crack (WTI->RBOB+HO)": dict(
        F1="WTI Crude Oil",      F1_unit="$/bbl",    F1_mult=3.0,
        F2="RBOB Gasoline",      F2_unit="$/gallon", F2_mult=2.0,
        F3="Heating Oil (ULSD)", F3_unit="$/gallon", F3_mult=1.0,
        description="3 bbl WTI -> 2 bbl RBOB + 1 bbl Heating Oil",
        typical=15.0, vol1=0.32, vol2=0.36, rho=0.85,
    ),
    "Simple Crack (WTI->RBOB)": dict(
        F1="WTI Crude Oil", F1_unit="$/bbl", F1_mult=1.0,
        F2="RBOB Gasoline", F2_unit="$/gallon", F2_mult=1.0,
        F3=None, F3_unit=None, F3_mult=0.0,
        description="1 bbl WTI -> 1 bbl RBOB",
        typical=12.0, vol1=0.32, vol2=0.36, rho=0.85,
    ),
    "Jet Crack (Brent->Jet Kero)": dict(
        F1="Brent Crude Oil",               F1_unit="$/bbl", F1_mult=1.0,
        F2="Jet Kerosene CIF NWE (Platts)", F2_unit="$/mt",  F2_mult=1.0,
        F3=None, F3_unit=None, F3_mult=0.0,
        description="1 bbl Brent -> 1 mt Jet Kerosene (airline hedging)",
        typical=80.0, vol1=0.30, vol2=0.35, rho=0.82,
    ),
    "Spark Spread (Gas->Power)": dict(
        F1="Dutch TTF Natural Gas", F1_unit="EUR/MWh", F1_mult=1.0,
        F2=None, F2_unit="EUR/MWh", F2_mult=1.0,
        F3=None, F3_unit=None, F3_mult=0.0,
        description="Gas -> Power (heat rate ~7 MMBtu/MWh)",
        typical=5.0, vol1=0.60, vol2=0.40, rho=0.50,
    ),
}

# ── Download functions (identical to CFCAP) ───────────────────────────────────

def build_tickers(cfg):
    """Build active (non-expired) contract list. Skips contracts past their expiry date."""
    now, n, active, contracts = datetime.now(), cfg["liquid_months"], cfg["active_months"], []
    month_offset = 0
    while len(contracts) < n and month_offset < n * 4:
        m    = (now.month - 1 + month_offset) % 12
        year = now.year + (now.month - 1 + month_offset) // 12
        month_offset += 1
        if MONTH_CODES[m] not in active:
            continue
        # Skip expired: contract expires ~20th of month prior to delivery
        exp_m, exp_y = (m-1, year) if m > 0 else (11, year-1)
        if now > datetime(exp_y, exp_m+1, 20):
            continue
        yr2 = str(year)[-2:]
        ticker = cfg["yf_fmt"].replace("{M}", MONTH_CODES[m]).replace("{YY}", yr2)                  if cfg["source"] == "yahoo" else f"{cfg['tv_prefix']}{MONTH_CODES[m]}{year}"
        contracts.append({"ticker": ticker, "label": f"{MONTH_NAMES[m]}-{year}",
                          "month_code": MONTH_CODES[m], "maturity": datetime(year, m+1, 20),
                          "months_to_mat": len(contracts)+1})
    return contracts

def get_forward_curve(cfg, rf, tv_user="", tv_pass=""):
    return _download_yahoo(cfg) if cfg["source"]=="yahoo" \
           else _download_tradingview(cfg, tv_user, tv_pass)

def _download_yahoo(cfg):
    try:
        import yfinance as yf
    except ImportError:
        return _synthetic_curve(cfg)
    contracts = build_tickers(cfg)
    tickers   = [c["ticker"] for c in contracts]
    time.sleep(2)
    raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
    if raw.empty:
        return _synthetic_curve(cfg)
    closes  = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]).iloc[-1]
    results = [{**c, "price": round(float(closes[c["ticker"]]),2)} for c in contracts
               if c["ticker"] in closes.index and pd.notna(closes[c["ticker"]])]
    return _to_df(results, cfg) if len(results) >= 2 else _synthetic_curve(cfg)

def _download_tradingview(cfg, tv_user, tv_pass):
    try:
        from tvdatafeed import TvDatafeed, Interval
    except ImportError:
        return _synthetic_curve(cfg)
    contracts = build_tickers(cfg)
    try:
        tv = TvDatafeed(tv_user, tv_pass) if tv_user else TvDatafeed()
    except Exception:
        tv = TvDatafeed()
    results, missing = [], []
    for i, c in enumerate(contracts, 1):
        if i > 1: time.sleep(1)
        try:
            hist = tv.get_hist(symbol=c["ticker"], exchange=cfg["tv_exchange"],
                               interval=Interval.in_daily, n_bars=5)
            if hist is not None and not hist.empty and "close" in hist.columns:
                results.append({**c, "price": round(float(hist["close"].dropna().iloc[-1]),2)})
            else:
                missing.append(c["ticker"])
        except Exception:
            missing.append(c["ticker"])
    return _to_df(results, cfg) if len(results) >= 2 else _synthetic_curve(cfg)

def _to_df(results, cfg):
    df = pd.DataFrame(results).sort_values("months_to_mat").reset_index(drop=True)
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    df["months_to_mat"] = range(1, len(df)+1)
    return df

def _synthetic_curve(cfg):
    np.random.seed(42)
    contracts = build_tickers(cfg)
    spot, records = cfg.get("synthetic_spot", 100.0), []
    for i, c in enumerate(contracts):
        T = (i+1)/12
        price = round(spot * np.exp((0.05 - cfg["storage_cost"]) * T)
                      + 0.02*spot*np.sin(2*np.pi*(i+2)/12)
                      + np.random.normal(0, spot*0.003), 4)
        records.append({**c, "price": price})
    return pd.DataFrame(records)

def load_cfcap_curve(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if not {"months_to_mat","price"}.issubset(df.columns):
            return None
        df = df[df["price"].notna()].reset_index(drop=True)
        df["month"]   = df["months_to_mat"].astype(int)
        df["T_yr"]    = df["month"] / 12
        df["forward"] = df["price"].astype(float)
        if "label" not in df.columns:
            df["label"] = [f"M{m}" for m in df["month"]]
        return df[["month","T_yr","forward","label"]]
    except Exception:
        return None

def forward_from_spot(spot, r, storage, convenience, T):
    return spot * np.exp((r + storage - convenience) * T)

def build_forward_curve(spot, r, storage, convenience, vol, n_months=12):
    rows = [{"month": m, "T_yr": round(m/12,4),
             "forward": round(forward_from_spot(spot,r,storage,convenience,m/12),4),
             "vol": round(vol,4)} for m in range(1, n_months+1)]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — BLACK-76 VANILLA OPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class Black76:
    """Black-76 model for European options on commodity futures."""

    def __init__(self, F, K, T, r, sigma, option_type="call"):
        self.F=float(F); self.K=float(K); self.T=float(T)
        self.r=float(r); self.sigma=float(sigma); self.opt=option_type.lower()
        if self.T<=0 or self.sigma<=0 or self.F<=0 or self.K<=0:
            raise ValueError("F,K,sigma,T must be positive.")

    def _d1_d2(self):
        d1 = (np.log(self.F/self.K) + 0.5*self.sigma**2*self.T) / (self.sigma*np.sqrt(self.T))
        return d1, d1 - self.sigma*np.sqrt(self.T)

    def price(self):
        d1,d2 = self._d1_d2(); disc=np.exp(-self.r*self.T)
        return disc*(self.F*norm.cdf(d1)-self.K*norm.cdf(d2)) if self.opt=="call" \
               else disc*(self.K*norm.cdf(-d2)-self.F*norm.cdf(-d1))

    def delta(self):
        d1,_=self._d1_d2(); disc=np.exp(-self.r*self.T)
        return disc*norm.cdf(d1) if self.opt=="call" else disc*(norm.cdf(d1)-1)

    def gamma(self):
        d1,_=self._d1_d2()
        return np.exp(-self.r*self.T)*norm.pdf(d1)/(self.F*self.sigma*np.sqrt(self.T))

    def vega(self):
        d1,_=self._d1_d2()
        return np.exp(-self.r*self.T)*self.F*norm.pdf(d1)*np.sqrt(self.T)/100

    def theta(self):
        d1,d2=self._d1_d2(); disc=np.exp(-self.r*self.T)
        t1 = -disc*self.F*norm.pdf(d1)*self.sigma/(2*np.sqrt(self.T))
        t2 = self.r*disc*(self.F*norm.cdf(d1)-self.K*norm.cdf(d2)) if self.opt=="call" \
             else self.r*disc*(self.K*norm.cdf(-d2)-self.F*norm.cdf(-d1))
        return (t1-t2)/365

    def rho(self):
        d1,d2=self._d1_d2(); disc=np.exp(-self.r*self.T)
        v = self.F*norm.cdf(d1)-self.K*norm.cdf(d2) if self.opt=="call" \
            else self.K*norm.cdf(-d2)-self.F*norm.cdf(-d1)
        return -self.T*disc*v/100

    def greeks(self):
        return {k: round(getattr(self,k)(),6) for k in ("price","delta","gamma","vega","theta","rho")}

    def implied_vol(self, mkt_price, lo=0.001, hi=10.0):
        try:
            return brentq(lambda s: Black76(self.F,self.K,self.T,self.r,s,self.opt).price()-mkt_price,
                          lo, hi, xtol=1e-8, maxiter=500)
        except Exception:
            return float("nan")

    def payoff_profile(self, spot_range):
        return np.maximum(spot_range-self.K,0) if self.opt=="call" \
               else np.maximum(self.K-spot_range,0)

    def pnl_profile(self, spot_range):
        return self.payoff_profile(spot_range) - self.price()


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — ASIAN OPTIONS (MONTE CARLO)
# ══════════════════════════════════════════════════════════════════════════════

class AsianOption:
    """Asian (Average Price) Option via Monte Carlo."""

    def __init__(self, F, K, T, r, sigma, n_obs=12,
                 option_type="call", average_type="arithmetic",
                 n_paths=50000, seed=42):
        self.F=float(F); self.K=float(K); self.T=float(T); self.r=float(r)
        self.sig=float(sigma); self.n=int(n_obs); self.opt=option_type.lower()
        self.avg=average_type.lower(); self.M=int(n_paths); self.seed=seed

    def _simulate_paths(self):
        np.random.seed(self.seed)
        dt  = self.T/self.n
        Z   = np.random.standard_normal((self.M,self.n))
        inc = (-0.5*self.sig**2*dt) + self.sig*np.sqrt(dt)*Z
        return self.F * np.exp(np.cumsum(inc, axis=1))

    def price(self):
        paths = self._simulate_paths()
        avgs  = paths.mean(axis=1) if self.avg=="arithmetic" \
                else np.exp(np.log(np.maximum(paths,1e-10)).mean(axis=1))
        disc  = np.exp(-self.r*self.T)
        pays  = np.maximum(avgs-self.K,0) if self.opt=="call" \
                else np.maximum(self.K-avgs,0)
        price   = disc*pays.mean()
        std_err = disc*pays.std()/np.sqrt(self.M)
        steps   = np.logspace(2, np.log10(self.M), 30).astype(int)
        conv    = [disc*np.maximum(
                      (paths[:n].mean(axis=1) if self.avg=="arithmetic"
                       else np.exp(np.log(np.maximum(paths[:n],1e-10)).mean(axis=1)))
                      - self.K, 0).mean() for n in steps]
        return {
            "price":       round(float(price),6),
            "std_error":   round(float(std_err),6),
            "ci_lo":       round(float(price-1.96*std_err),6),
            "ci_hi":       round(float(price+1.96*std_err),6),
            "n_paths":     self.M, "n_obs": self.n,
            "average_type": self.avg,
            "conv_paths":  [int(s) for s in steps],
            "conv_prices": [round(float(c),6) for c in conv],
            "sample_paths": paths[:100].tolist(),
            "sample_avgs":  avgs[:100].tolist(),
        }

    def geometric_analytical(self):
        n    = self.n
        sg   = self.sig * np.sqrt((2*n+1)/(6*(n+1)))
        bg   = 0.5*(-self.sig**2/6 + sg**2)
        Fadj = self.F * np.exp(bg*self.T)
        return Black76(Fadj, self.K, self.T, self.r, sg, self.opt).price()


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — CRACK SPREAD OPTIONS (KIRK 1995)
# ══════════════════════════════════════════════════════════════════════════════

class CrackSpreadOption:
    """Spread option on crack spread via Kirk (1995) approximation."""

    def __init__(self, F1, F2, K, T, r, sigma1, sigma2, rho=0.85,
                 option_type="call", multiplier1=1.0, multiplier2=1.0):
        self.F1=float(F1); self.F2=float(F2)*multiplier2/multiplier1
        self.K=float(K); self.T=float(T); self.r=float(r)
        self.s1=float(sigma1); self.s2=float(sigma2); self.rho=float(rho)
        self.opt=option_type.lower()
        self.raw_spread=self.F2-self.F1

    def _kirk_vol(self):
        adj = self.F2/(self.F2+self.K*np.exp(-self.r*self.T))
        return np.sqrt(self.s1**2 + (adj*self.s2)**2 - 2*self.rho*self.s1*self.s2*adj)

    def price(self):
        disc = np.exp(-self.r*self.T)
        F_sp = self.F2 - self.F1
        sk   = self._kirk_vol()
        if sk<=0 or self.T<=0:
            intr = max(F_sp-self.K,0) if self.opt=="call" else max(self.K-F_sp,0)
            return {"price":round(intr,4),"intrinsic":round(intr,4),"time_value":0.0,
                    "sigma_kirk":0.0,"current_spread":round(F_sp,4),
                    "delta_F1":0.0,"delta_F2":0.0,"vega":0.0}
        b76  = Black76(self.F2, self.F1+self.K*disc, self.T, self.r, sk, self.opt)
        p    = b76.price()
        intr = max(F_sp-self.K,0) if self.opt=="call" else max(self.K-F_sp,0)
        return {"price":round(float(p),4),"intrinsic":round(float(intr),4),
                "time_value":round(float(p-disc*intr),4),"sigma_kirk":round(float(sk),4),
                "current_spread":round(float(F_sp),4),
                "delta_F1":round(float(-b76.delta()),4),"delta_F2":round(float(b76.delta()),4),
                "vega":round(float(b76.vega()),6)}

    def payoff_grid(self, spread_range):
        return np.maximum(spread_range-self.K,0) if self.opt=="call" \
               else np.maximum(self.K-spread_range,0)

    def price_surface(self, K_range, T_range):
        surf = np.zeros((len(T_range),len(K_range)))
        for i,t in enumerate(T_range):
            for j,k in enumerate(K_range):
                surf[i,j] = CrackSpreadOption(self.F1,self.F2,k,t,self.r,
                                              self.s1,self.s2,self.rho,self.opt).price()["price"]
        return surf


# ══════════════════════════════════════════════════════════════════════════════
#  PART 4 — CALENDAR SPREAD OPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class CalendarSpreadOption:
    """Option on calendar spread (Kirk approximation on term structure)."""

    def __init__(self, F_near, F_far, K, T, r, sigma_near, sigma_far,
                 rho=0.95, option_type="call", near_label="M1", far_label="M6"):
        self.Fn=float(F_near); self.Ff=float(F_far)
        self.K=float(K); self.T=float(T); self.r=float(r)
        self.sn=float(sigma_near); self.sf=float(sigma_far); self.rho=float(rho)
        self.opt=option_type.lower(); self.n_lbl=near_label; self.f_lbl=far_label
        self.spread=self.Fn-self.Ff

    def price(self):
        disc = np.exp(-self.r*self.T)
        F_sp = self.Fn-self.Ff
        adj  = self.Fn/(self.Fn+self.K*disc) if (self.Fn+self.K*disc)!=0 else 1.0
        sk   = np.sqrt(self.sf**2+(adj*self.sn)**2-2*self.rho*self.sf*self.sn*adj)
        if sk<=0:
            intr=max(F_sp-self.K,0) if self.opt=="call" else max(self.K-F_sp,0)
            return {"price":round(intr,6),"intrinsic":round(intr,6),"time_value":0.0,
                    "sigma_kirk":0.0,"spread":round(F_sp,4),"near_label":self.n_lbl,
                    "far_label":self.f_lbl,"delta_near":0.0,"delta_far":0.0,"gamma":0.0,"vega":0.0}
        b76  = Black76(self.Fn, self.Ff+self.K*disc, self.T, self.r, sk, self.opt)
        p    = b76.price()
        intr = max(F_sp-self.K,0) if self.opt=="call" else max(self.K-F_sp,0)
        return {"price":round(float(p),6),"intrinsic":round(float(disc*intr),6),
                "time_value":round(float(p-disc*intr),6),"sigma_kirk":round(float(sk),6),
                "spread":round(float(F_sp),4),"near_label":self.n_lbl,"far_label":self.f_lbl,
                "delta_near":round(float(b76.delta()),6),"delta_far":round(float(-b76.delta()),6),
                "gamma":round(float(b76.gamma()),8),"vega":round(float(b76.vega()),6)}

    def payoff_grid(self, spread_range):
        return np.maximum(spread_range-self.K,0) if self.opt=="call" \
               else np.maximum(self.K-spread_range,0)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 5 — COMMODITY SWAPS
# ══════════════════════════════════════════════════════════════════════════════

class CommoditySwap:
    """Fixed-for-floating commodity swap pricer."""

    def __init__(self, forward_prices, fixed_rate, r, notional=1.0,
                 payment_freq="monthly", position="fixed_payer"):
        self.fwd=np.array(forward_prices,dtype=float); self.K=float(fixed_rate)
        self.r=float(r); self.notional=float(notional)
        self.freq=payment_freq; self.pos=position; self.N=len(self.fwd)
        dt = 1/12 if payment_freq=="monthly" else 1/4
        self.T_i    = np.array([(i+1)*dt for i in range(self.N)])
        self.disc_i = np.exp(-self.r*self.T_i)

    @property
    def fair_rate(self):
        return float(np.sum(self.disc_i*self.fwd)/np.sum(self.disc_i))

    def npv(self):
        pnl = np.sum(self.disc_i*(self.fwd-self.K)*self.notional)
        return float(pnl if self.pos=="fixed_payer" else -pnl)

    def dv01(self):
        return float(np.sum(self.disc_i*self.notional)*0.0001)

    def cashflows(self):
        net = (self.fwd-self.K)*self.notional * (1 if self.pos=="fixed_payer" else -1)
        rows = [{"Period":i+1,"Maturity (yr)":round(float(self.T_i[i]),4),
                 "Forward price":round(float(self.fwd[i]),4),"Fixed rate":round(self.K,4),
                 "Net cashflow":round(float(net[i]),4),"Discount":round(float(self.disc_i[i]),6),
                 "PV cashflow":round(float(self.disc_i[i]*net[i]),4)} for i in range(self.N)]
        df = pd.DataFrame(rows)
        df.loc[len(df)] = {"Period":"TOTAL","Maturity (yr)":"-",
            "Forward price":round(float(self.fwd.mean()),4),"Fixed rate":round(self.K,4),
            "Net cashflow":round(float(net.sum()),4),"Discount":"-","PV cashflow":round(self.npv(),4)}
        return df

    def sensitivity(self, shift_range):
        return pd.DataFrame([{"shift":round(float(s),2),
            "npv":round(CommoditySwap(self.fwd+s,self.K,self.r,self.notional,self.freq,self.pos).npv(),4)}
            for s in shift_range])


# ══════════════════════════════════════════════════════════════════════════════
#  PART 6 — STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

def _is_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

# Chart title helper — consistent across all charts
def _title(text, size=15):
    return dict(text=text, x=0.5, xanchor="center",
                font=dict(size=size, color="#E6EDF3", family="Inter"),
                pad=dict(t=20, b=10))

def _tab_title(st_mod, text, color="#F0A500", icon=""):
    prefix = icon + " " if icon else ""
    st_mod.markdown(
        '<div style="border-left:3px solid ' + color + ';padding-left:10px;'
        'margin:18px 0 8px 0">'
        '<span style="font-size:1.05rem;font-weight:500;color:#E6EDF3;'
        'letter-spacing:-0.01em">' + prefix + text + '</span>'
        '</div>',
        unsafe_allow_html=True)

def _tab_desc(st_mod, text):
    st_mod.markdown(
        '<div style="font-size:0.86rem;color:#8B949E;margin-bottom:16px;line-height:1.6;'
        'padding:8px 12px;background:#161B22;border-radius:6px;'
        'border-left:2px solid #30363D">' + text + '</div>',
        unsafe_allow_html=True)

def run_streamlit_app():
    import streamlit as st
    import plotly.graph_objects as go

    AMBER="#F0A500"; BLUE="#58A6FF"; GREEN="#3FB950"
    RED="#FF7B72"; GRAY="#8B949E"; PURPLE="#BC8CFF"
    PANEL="#161B22"; BG="#0D1117"; BORDER="#30363D"; TEXT="#E6EDF3"

    st.set_page_config(
        page_title="CODAP — Commodity Options & Derivatives Pricer",
        page_icon="📈", layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
    code,pre,[class*="monospace"]{font-family:'JetBrains Mono',monospace!important}
    .block-container{padding-top:1.0rem!important;max-width:1440px}
    [data-testid="stSidebar"]{
        background:linear-gradient(180deg,#0A0E17 0%,#0D1117 100%)!important;
        border-right:1px solid #1C2333!important}
    [data-testid="stSidebar"] label{font-size:0.70rem!important;font-weight:500!important;
        color:#6E7681!important;text-transform:uppercase!important;letter-spacing:.08em!important}
    [data-testid="stSidebar"] .stSelectbox>div>div{
        background:#161B22!important;border:0.5px solid #30363D!important;
        border-radius:7px!important;font-size:0.83rem!important}
    [data-testid="stSidebar"] .stNumberInput input{
        background:#161B22!important;border:0.5px solid #30363D!important;
        border-radius:7px!important;color:#E6EDF3!important;
        font-family:'JetBrains Mono',monospace!important}
    .stTabs [data-baseweb="tab-list"]{
        background:#0D1117!important;border-bottom:1px solid #1C2333!important;gap:2px!important}
    .stTabs [data-baseweb="tab"]{font-size:0.80rem!important;font-weight:500!important;
        color:#6E7681!important;padding:10px 18px!important;
        border-radius:6px 6px 0 0!important;transition:all .15s ease!important}
    .stTabs [aria-selected="true"]{color:#E6EDF3!important;background:#161B22!important;
        border-bottom:2px solid #F0A500!important;font-weight:500!important}
    .stTabs [data-baseweb="tab"]:hover{color:#E6EDF3!important;background:#161B2288!important}
    hr{border:none!important;border-top:1px solid #1C2333!important;margin:12px 0!important}
    div[data-testid="metric-container"]{background:#161B22!important;
        border:0.5px solid #30363D!important;border-radius:10px!important;padding:12px 16px!important}
    .stDataFrame{border:0.5px solid #1C2333!important;border-radius:8px!important}
    .stButton>button{font-family:'Inter',sans-serif!important;font-size:0.82rem!important;
        font-weight:700!important;border-radius:7px!important;letter-spacing:.02em!important;
        transition:all .15s ease!important}
    .stButton>button:hover{transform:translateY(-1px)!important;
        box-shadow:0 4px 12px rgba(240,165,0,.25)!important}
    </style>""", unsafe_allow_html=True)

    # ── Gradient header ────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-top:18px;margin-bottom:2px">'
        '<h1 style="font-family:Inter,sans-serif;font-size:1.9rem;font-weight:700;'
        'letter-spacing:-0.03em;margin:0;background:linear-gradient(90deg,#F0A500,#FF7B72);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent">CODAP</h1>'
        '<span style="font-size:0.95rem;color:#6E7681;font-weight:600">'
        'Commodity Derivatives Analytics Platform</span>'
        '</div>'
        '<div style="font-size:0.72rem;color:#444C56;margin-bottom:16px;letter-spacing:.02em">'
        ' '
        'Black-76 · Asian MC · Crack Spread · Calendar Spread · Swaps · Barrier · Vol Surface'
        '</div>',
        unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:12px 0 8px">'
            '<div style="font-size:1.4rem;font-weight:700;letter-spacing:-0.03em;'
            'background:linear-gradient(90deg,#F0A500,#FF7B72);'
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent">CODAP</div>'
            '<div style="font-size:0.9rem;color:#6E7681;margin-top:2px">'
            'Derivatives Pricer</div>'
            '</div>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<div style="font-size:1rem;font-weight:700;color:#58A6FF;'
                    'text-transform:uppercase;letter-spacing:.10em;margin-bottom:8px">'
                    '🏷 Commodity</div>', unsafe_allow_html=True)
        family    = st.selectbox("Asset class", list(COMMODITY_REGISTRY.keys()),
                                  label_visibility="collapsed")
        commodity = st.selectbox("Commodity", list(COMMODITY_REGISTRY[family].keys()),
                                  label_visibility="collapsed")
        cfg_reg      = COMMODITY_REGISTRY[family][commodity].copy()
        cp           = get_opcap_params(family, commodity)
        unit         = cp["unit"]
        spot_default = cp["spot"]

        src_lbl = "Yahoo Finance" if cfg_reg.get("source")=="yahoo"                   else f"TradingView ({cfg_reg.get('tv_exchange','')})"
        src_col = "#3FB950" if cfg_reg.get("source")=="yahoo" else "#F0A500"
        st.markdown(f'<span style="color:{src_col};font-size:0.78rem;font-weight:500">'
                    f'{src_lbl}</span> &nbsp;'
                    f'<span style="color:#8B949E;font-size:0.78rem">{unit}</span>',
                    unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#F0A500;'
                    'text-transform:uppercase;letter-spacing:.10em;margin-bottom:8px">'
                    '⚙ Option Parameters</div>', unsafe_allow_html=True)
        T_months = st.slider("Maturity (months)", 1, 24, 6)
        T        = T_months / 12
        K_pct    = st.number_input("Strike (% of spot)", 70.0, 150.0, 100.0, 1.0, format="%.1f")
        n_months = st.slider("Curve length (months)", 3, 36, int(cp["liquid_months"]))
        st.markdown("---")
        n_paths  = st.select_slider("Monte Carlo paths",
                    options=[10000, 25000, 50000, 100000], value=50000)

        st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#58A6FF;'
                    'text-transform:uppercase;letter-spacing:.10em;margin-bottom:8px">'
                    '📈 Market Parameters</div>', unsafe_allow_html=True)
        vol   = st.slider("Implied volatility (%)", 5, 120, int(cp["vol"]*100), 1) / 100
        if "rf_val_op" not in st.session_state: st.session_state["rf_val_op"] = 4.25
        rf_pct = st.number_input("Risk-free rate (%)", 0.0, 20.0,
                                  st.session_state["rf_val_op"], 0.01,
                                  format="%.2f", key="rf_op")
        st.session_state["rf_val_op"] = rf_pct
        r = rf_pct / 100
        st.markdown("---")

        st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#3FB950;'
                    'text-transform:uppercase;letter-spacing:.10em;margin-bottom:8px">'
                    '🔄 Cost of Carry</div>', unsafe_allow_html=True)
        storage     = st.number_input("Storage cost (%/yr)", 0.0, 20.0,
                                       cp["storage"]*100, 0.1, format="%.1f") / 100
        convenience = st.number_input("Convenience yield (%/yr)", 0.0, 30.0,
                                       cp["convenience"]*100, 0.1, format="%.1f") / 100
        st.markdown("---")

        tv_user   = st.text_input("TradingView username", value=TV_USERNAME)
        tv_pass_i = st.text_input("TradingView password", value=TV_PASSWORD, type="password")
        run_btn   = st.button("▶  Run Analysis", type="primary", use_container_width=True)

    # ── Sub-header badges ──────────────────────────────────────────────────────
    def _badge(text, color="#8B949E"):
        return (f'<span style="background:#1C2128;border:0.5px solid #30363D;'
                f'border-radius:4px;padding:2px 8px;font-size:0.72rem;'
                f'color:{color};font-family:JetBrains Mono,monospace">{text}</span>')
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px">'
        + _badge(commodity, "#E6EDF3")
        + _badge(family)
        + _badge(unit)
        + _badge(src_lbl, src_col)
        + _badge(f"RF {rf_pct:.2f}%")
        + _badge(datetime.now().strftime("%d %b %Y  %H:%M"))
        + '</div>',
        unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────────────
    for k in ["live_df","live_comm","live_ts"]:
        if k not in st.session_state: st.session_state[k] = None

    combo = (commodity, family, rf_pct, n_months)

    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch(cn, fn, rf_, u, p, _h):
        c = COMMODITY_REGISTRY[fn][cn].copy()
        return get_forward_curve(c, rf_, u, p)

    if run_btn or st.session_state.get("opcap_combo") != combo:
        with st.spinner(f"Loading {commodity}..."):
            try:
                live_df = _fetch(commodity, family, r, tv_user, tv_pass_i, str(cfg_reg))
                st.session_state["live_df"]     = live_df
                st.session_state["live_comm"]   = commodity
                st.session_state["live_ts"]     = datetime.now().strftime("%H:%M:%S")
                st.session_state["opcap_combo"] = combo
                spot_default = float(live_df["price"].iloc[0])
            except Exception as e:
                st.warning(f"Live download failed — synthetic prices used. ({e})")
                live_df = None

    live_df = st.session_state.get("live_df")
    if live_df is not None and st.session_state.get("live_comm") == commodity:
        spot_default = float(live_df["price"].iloc[0])

    # ── K from % of spot ───────────────────────────────────────────────────────
    spot = spot_default  # alias used throughout the app
    K    = round(spot_default * K_pct / 100, 4)

    # ── Build forward curve ────────────────────────────────────────────────────
    if live_df is not None and not live_df.empty:
        # Live data from Yahoo / TradingView — use real futures prices
        fwd_df = live_df[["months_to_mat","price","label"]].copy()
        fwd_df = fwd_df.rename(columns={"months_to_mat":"month","price":"forward"})
        fwd_df["T_yr"] = fwd_df["month"] / 12
        fwd_df["vol"]  = vol
        # Trim or extend to n_months
        if len(fwd_df) > n_months:
            fwd_df = fwd_df.iloc[:n_months].reset_index(drop=True)
        elif len(fwd_df) < n_months:
            # Extend with cost-of-carry if live data is shorter than requested
            last_fwd  = float(fwd_df["forward"].iloc[-1])
            last_mo   = int(fwd_df["month"].iloc[-1])
            extra = [{"month": last_mo+i+1,
                      "T_yr": (last_mo+i+1)/12,
                      "forward": round(last_fwd * np.exp((r+storage-convenience)*(i+1)/12), 4),
                      "label": f"M{last_mo+i+1}",
                      "vol": vol}
                     for i in range(n_months - len(fwd_df))]
            fwd_df = pd.concat([fwd_df, pd.DataFrame(extra)], ignore_index=True)
    else:
        # Synthetic cost-of-carry curve (fallback when download unavailable)
        fwd_df = build_forward_curve(spot, r, storage, convenience, vol, n_months)
        fwd_df["label"] = [f"M{m}" for m in fwd_df["month"]]

    F_m1  = float(fwd_df["forward"].iloc[0])
    F_m3  = float(fwd_df["forward"].iloc[min(2,len(fwd_df)-1)])
    F_m6  = float(fwd_df["forward"].iloc[min(5,len(fwd_df)-1)])
    F_m12 = float(fwd_df["forward"].iloc[min(11,len(fwd_df)-1)])
    t_idx = min(T_months-1, len(fwd_df)-1)
    F_T   = float(fwd_df["forward"].iloc[t_idx])

    # ── KPI bar ────────────────────────────────────────────────────────────────
    b76_call = Black76(F_T, K, T, r, vol, "call")
    b76_put  = Black76(F_T, K, T, r, vol, "put")
    gc       = b76_call.greeks(); gp = b76_put.greeks()
    fwd_str  = "BACKWARDATION" if F_m6<F_m1 else "CONTANGO"
    str_col  = GREEN if fwd_str=="BACKWARDATION" else RED

    # ── Synthetic data warning ────────────────────────────────────────────────
    if live_df is None:
        st.markdown(
            f'<div style="background:#1C1400;border:1px solid #F0A50066;'
            f'border-radius:8px;padding:8px 14px;margin-bottom:12px;'
            f'display:flex;align-items:center;gap:10px">'
            f'<span style="font-size:1.1rem">⚠</span>'
            f'<span style="color:#F0A500;font-size:.78rem;font-weight:600">'
            f'SYNTHETIC PRICES</span>'
            f'<span style="color:#8B949E;font-size:.75rem;margin-left:8px">'
            f' — click <b style="color:#E6EDF3">▶ Run Analysis</b> '
            f'to load live data from {src_lbl}</span>'
            f'</div>',
            unsafe_allow_html=True)

    def _kpi(label, value, sub, accent="#8B949E"):
        return (
            f'<div style="background:linear-gradient(135deg,#161B22 60%,{accent}18 100%);'
            f'border:0.5px solid {accent}44;border-left:3px solid {accent};'
            f'border-radius:10px;padding:13px 15px">'
            f'<div style="font-size:.82rem;font-weight:500;color:{accent};'
            f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px">{label}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:1rem;font-weight:500;'
            f'color:#E6EDF3;white-space:nowrap">{value}</div>'
            f'<div style="font-size:.8rem;color:#6E7681;margin-top:3px">{sub}</div>'
            f'</div>'
        )

    kpi_html = (
        f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:9px;margin-bottom:20px">'
        + _kpi("Spot M1",      f"{F_m1:.3f} {unit}",   "front-month price",    AMBER)
        + _kpi("Structure",    fwd_str,                  f"M1-M6 {F_m6-F_m1:+.2f}", str_col)
        + _kpi("ATM Call",     f"{gc['price']:.4f}",    "Black-76 premium",     AMBER)
        + _kpi("ATM Put",      f"{gp['price']:.4f}",    "Black-76 premium",     BLUE)
        + _kpi("Impl. Vol",    f"{vol*100:.1f}%",       "annual σ",             PURPLE)
        + _kpi("Delta (call)", f"{gc['delta']:.4f}",    "futures equivalent",   GREEN)
        + _kpi(f"F(T) M{T_months}", f"{F_T:.3f}",       unit,                  GRAY)
        + '</div>'
    )
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown("---")

    t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs([
        "📉 Vanilla Options","🎲 Asian Options",
        "🏭 Crack Spread","📅 Calendar Spread",
        "🔄 Commodity Swaps","🚧 Barrier Options",
        "📊 Vol Surface","📈 Forward Curve"])

    # ── TAB 1 — VANILLA OPTIONS ───────────────────────────────────────────────
    with t1:
        _tab_title(st, "Black-76 European Option Pricer", "#F0A500", "📉")
        _tab_desc(st, "Black-76 is the industry standard for commodity options. "
            "It prices European options on <b>futures prices</b> — directly observable and tradeable. "
            "All Greeks are computed analytically. Put-call parity is verified.")

        _hdr1, _hdr2, _hdr3 = st.columns([2,2,2])
        with _hdr1:
            opt_type = st.radio("Option type", ["Call ▲","Put ▼"], horizontal=True, key="ot1")
            opt_type = "Call" if "Call" in opt_type else "Put"
        with _hdr2:
            st.markdown(
                f'<div style="background:#161B22;border:0.5px solid #30363D;border-radius:8px;'
                f'padding:8px 14px;font-family:JetBrains Mono,monospace;font-size:.80rem;color:#E6EDF3">'
                f'F(M{T_months}) = <b style="color:#F0A500">{F_T:.4f}</b> {unit} &nbsp;·&nbsp; '
                f'K = <b style="color:#3FB950">{K:.4f}</b> &nbsp;·&nbsp; '
                f'T = <b style="color:#58A6FF">{T:.3f}yr</b></div>',
                unsafe_allow_html=True)
        with _hdr3:
            st.markdown(
                f'<div style="background:#161B22;border:0.5px solid #30363D;border-radius:8px;'
                f'padding:8px 14px;font-family:JetBrains Mono,monospace;font-size:.80rem;color:#E6EDF3">'
                f'σ = <b style="color:#BC8CFF">{vol*100:.1f}%</b> &nbsp;·&nbsp; '
                f'r = <b style="color:#39D0D8">{r*100:.2f}%</b> &nbsp;·&nbsp; '
                f'Structure = <b style="color:{"#3FB950" if fwd_str=="BACKWARDATION" else "#FF7B72"}">{fwd_str[:4]}</b></div>',
                unsafe_allow_html=True)
        st.markdown("")

        col_g, col_p = st.columns(2)

        with col_g:
            st.markdown('<div style="font-size:0.78rem;font-weight:500;color:#E6EDF3;'
                'margin-bottom:8px">Greeks — Call vs Put</div>', unsafe_allow_html=True)
            call=Black76(F_T,K,T,r,vol,"call"); put=Black76(F_T,K,T,r,vol,"put")
            gc2=call.greeks(); gp2=put.greeks()
            greek_meta = {
                "price": ("Price",  "Premium to buy the option"),
                "delta": ("Delta",  "Futures to hold for delta-hedge"),
                "gamma": ("Gamma",  "Rate of change of delta"),
                "vega":  ("Vega",   "P&L per 1% vol move"),
                "theta": ("Theta",  "Value lost per calendar day"),
                "rho":   ("Rho",    "Sensitivity to risk-free rate (per 1%)"),
            }
            g_html = '<div style="display:grid;gap:5px">'
            for key,(lbl,tip) in greek_meta.items():
                cv=gc2[key]; pv=gp2[key]
                cc=AMBER if key=="price" else (GREEN if cv>0 else RED)
                pc=BLUE  if key=="price" else (GREEN if pv>0 else RED)
                g_html += (
                    f'<div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:8px;'
                    f'padding:9px 13px;display:flex;justify-content:space-between;align-items:center">'
                    f'<div><div style="font-size:.74rem;font-weight:500;color:{TEXT}">{lbl}</div>'
                    f'<div style="font-size:.64rem;color:{GRAY}">{tip}</div></div>'
                    f'<div style="display:flex;gap:10px;font-family:JetBrains Mono,monospace">'
                    f'<div style="text-align:right"><div style="font-size:.58rem;color:{GRAY}">CALL</div>'
                    f'<div style="font-size:.80rem;color:{cc}">{cv:.6f}</div></div>'
                    f'<div style="text-align:right"><div style="font-size:.58rem;color:{GRAY}">PUT</div>'
                    f'<div style="font-size:.80rem;color:{pc}">{pv:.6f}</div></div>'
                    f'</div></div>')
            g_html += '</div>'
            st.markdown(g_html, unsafe_allow_html=True)

            # Put-call parity
            parity_lhs = gc2["price"]-gp2["price"]
            parity_rhs = np.exp(-r*T)*(F_T-K)
            _perr = abs(parity_lhs - parity_rhs)
            _ptag = (f'<span style="color:{GREEN}">✓ verified</span>'
                     if _perr < 1e-4 else
                     f'<span style="color:{RED}">Error = {_perr:.2e}</span>')
            st.markdown(
                f'<div style="margin-top:8px;padding:7px 11px;background:{PANEL};'
                f'border:0.5px solid {BORDER};border-radius:7px;font-family:JetBrains Mono,monospace;'
                f'font-size:.70rem;color:{GRAY}">Put-Call Parity: '
                f'C−P = {parity_lhs:.4f} &nbsp;|&nbsp; e^(−rT)(F−K) = {parity_rhs:.4f} '
                f'&nbsp;|&nbsp; {_ptag}</div>',
                unsafe_allow_html=True)

        with col_p:
            # Payoff diagram
            spot_range  = np.linspace(K*0.5, K*1.5, 300)
            call_pnl    = call.pnl_profile(spot_range)
            put_pnl     = put.pnl_profile(spot_range)
            call_payoff = call.payoff_profile(spot_range)

            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(x=spot_range,y=call_pnl,name="Call P&L",
                line=dict(color=AMBER,width=2)))
            fig_pay.add_trace(go.Scatter(x=spot_range,y=put_pnl,name="Put P&L",
                line=dict(color=BLUE,width=2)))
            fig_pay.add_trace(go.Scatter(x=spot_range,y=call_payoff,
                name="Call payoff (intrinsic)",line=dict(color=AMBER,width=1,dash="dot"),opacity=0.45))
            fig_pay.add_hline(y=0, line=dict(color=GRAY,width=0.8))
            fig_pay.add_vline(x=K, line=dict(color=GREEN,dash="dash",width=1),
                annotation_text=f"Strike K = {K:.2f}",
                annotation_position="top left",
                annotation=dict(yref="paper",y=0.96,
                    font=dict(color=GREEN,size=11,family="JetBrains Mono"),
                    bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1))
            fig_pay.add_vline(x=F_T, line=dict(color=RED,dash="dot",width=1),
                annotation_text=f"Forward F = {F_T:.2f}",
                annotation_position="top right",
                annotation=dict(yref="paper",y=0.80,
                    font=dict(color=RED,size=11,family="JetBrains Mono"),
                    bgcolor="#2D0A09",bordercolor=RED,borderwidth=1))
            fig_pay.update_layout(template="plotly_dark",height=330,
                title=dict(text="P&L at Expiry",y=0.97,x=0.5,xanchor="center",
                           font=dict(size=13,color="#8B949E")),
                xaxis=dict(title=f"Price at expiry ({unit})"),
                yaxis=dict(title=f"P&L ({unit})"),
                legend=dict(orientation="h",yanchor="bottom",y=1.02),
                margin=dict(l=60,r=20,t=75,b=60))
            st.plotly_chart(fig_pay, use_container_width=True)

        # Sensitivity surfaces
        _tab_title(st, "Price & Delta Sensitivity", "#58A6FF", "🌡")
        K_range = np.linspace(K*0.7, K*1.3, 15)
        T_rng   = np.array([1/12,3/12,6/12,9/12,12/12,18/12,24/12])
        T_lbls  = ["1M","3M","6M","9M","1Y","18M","2Y"]
        p_surf  = np.zeros((len(T_rng),len(K_range)))
        d_surf  = np.zeros_like(p_surf)
        for i,t in enumerate(T_rng):
            for j,k in enumerate(K_range):
                b=Black76(F_T,k,t,r,vol,opt_type.lower())
                p_surf[i,j]=b.price(); d_surf[i,j]=b.delta()

        cs1, cs2 = st.columns(2)
        with cs1:
            fig_ps = go.Figure(go.Heatmap(z=p_surf,x=[f"{k:.1f}" for k in K_range],y=T_lbls,
                colorscale="YlOrRd",
                hovertemplate="Strike: %{x}<br>Maturity: %{y}<br>Price: %{z:.4f}<extra></extra>"))
            fig_ps.update_layout(template="plotly_dark",height=290,
                title=_title(f"{opt_type} Price Surface  (σ={vol*100:.0f}%)"),
                xaxis=dict(title=f"Strike ({unit})"),yaxis=dict(title="Maturity"),
                margin=dict(l=60,r=20,t=75,b=50))
            st.plotly_chart(fig_ps, use_container_width=True)
        with cs2:
            fig_ds = go.Figure(go.Heatmap(z=d_surf,x=[f"{k:.1f}" for k in K_range],y=T_lbls,
                colorscale="RdBu",zmid=0,
                hovertemplate="Strike: %{x}<br>Maturity: %{y}<br>Delta: %{z:.4f}<extra></extra>"))
            fig_ds.update_layout(template="plotly_dark",height=290,
                title=_title(f"{opt_type} Delta Surface"),
                xaxis=dict(title=f"Strike ({unit})"),yaxis=dict(title="Maturity"),
                margin=dict(l=60,r=20,t=75,b=50))
            st.plotly_chart(fig_ds, use_container_width=True)

    # ── TAB 2 — ASIAN OPTIONS ─────────────────────────────────────────────────
    with t2:
        _tab_title(st, "Asian (Average Price) Option Pricer", "#BC8CFF", "🎲")
        _tab_desc(st, "Asian options pay off on the <b>average price</b> over their life. "
            "Used by airlines, refiners and consumers whose procurement is priced on monthly averages. "
            "No closed-form for arithmetic — Monte Carlo simulation with Kemna-Vorst benchmark.")

        ca1,ca2,ca3 = st.columns([2,2,2])
        with ca1:
            opt_type_a = st.radio("Option type", ["Call ▲","Put ▼"], horizontal=True, key="ot2")
            opt_type_a = "call" if "Call" in opt_type_a else "put"
            avg_type   = st.radio("Average type",["arithmetic","geometric"],horizontal=True)
        with ca2:
            n_obs_v = st.select_slider("Observations",options=[4,12,22,52,252],value=12,
                format_func=lambda x:{4:"Quarterly",12:"Monthly",22:"Weekly×22",52:"Weekly×52",252:"Daily"}[x])
        with ca3:
            st.markdown(
                '<div style="background:#1A0F2E;border:0.5px solid #6E40C9;border-radius:8px;'
                'padding:10px 14px;font-size:0.74rem;color:#BC8CFF;line-height:1.6">'
                '<b>Arithmetic</b>: standard in commodities. MC only.<br>'
                '<b>Geometric</b>: Kemna-Vorst analytical. Lower bound.<br>'
                '<b>Asian &lt; Vanilla</b>: averaging reduces vol.</div>',
                unsafe_allow_html=True)

        with st.spinner(f"Running {n_paths:,} Monte Carlo paths..."):
            asian = AsianOption(F_T,K,T,r,vol,n_obs_v,opt_type_a,avg_type,n_paths)
            res   = asian.price()
            geo_p = asian.geometric_analytical()

        vanilla_p = b76_call.price() if opt_type_a=="call" else b76_put.price()
        res_html = f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px">
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:12px 14px">
            <div style="font-size:.64rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Asian (MC)</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.95rem;color:{AMBER}">{res['price']:.6f}</div>
            <div style="font-size:.68rem;color:{GRAY};margin-top:3px">{unit}</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:12px 14px">
            <div style="font-size:.64rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">95% CI</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:{TEXT}">[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]</div>
            <div style="font-size:.68rem;color:{GRAY};margin-top:3px">width {res['ci_hi']-res['ci_lo']:.4f}</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:12px 14px">
            <div style="font-size:.64rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Geometric</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.95rem;color:{BLUE}">{geo_p:.6f}</div>
            <div style="font-size:.68rem;color:{GRAY};margin-top:3px">Kemna-Vorst</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:12px 14px">
            <div style="font-size:.64rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Vanilla Black-76</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.95rem;color:{GREEN}">{vanilla_p:.6f}</div>
            <div style="font-size:.68rem;color:{GRAY};margin-top:3px">European benchmark</div>
          </div>
        </div>"""
        st.markdown(res_html, unsafe_allow_html=True)

        cm1,cm2 = st.columns(2)
        with cm1:
            sample_paths = np.array(res["sample_paths"])
            times = np.linspace(0,T,n_obs_v)
            fig_mc = go.Figure()
            for i in range(min(25,len(sample_paths))):
                fig_mc.add_trace(go.Scatter(x=times,y=sample_paths[i],mode="lines",
                    line=dict(width=0.5,color=AMBER),opacity=0.25,showlegend=False))
            fig_mc.add_trace(go.Scatter(x=times,y=sample_paths.mean(axis=0),mode="lines",
                name="Mean path",line=dict(color=RED,width=2)))
            fig_mc.add_hline(y=K,line=dict(color=GREEN,dash="dash"),
                annotation_text=f"K = {K:.2f}",
                annotation_position="right",
                annotation=dict(font=dict(color=GREEN,size=11,family="JetBrains Mono"),
                    bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1))
            fig_mc.update_layout(template="plotly_dark",height=300,
                title=_title("Monte Carlo Price Paths (25 shown)"),
                xaxis=dict(title="Time (yr)"),yaxis=dict(title=f"Price ({unit})"),
                margin=dict(l=60,r=20,t=75,b=50))
            st.plotly_chart(fig_mc, use_container_width=True)
        with cm2:
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Scatter(x=res["conv_paths"],y=res["conv_prices"],
                mode="lines+markers",name="MC estimate",
                line=dict(color=AMBER,width=2),marker=dict(size=4)))
            fig_cv.add_hline(y=res["price"],line=dict(color=GREEN,dash="dash"))
            fig_cv.add_annotation(
                x=0.98, xref="paper", y=res["price"], yref="y",
                text=f"Final = {res['price']:.4f}",
                showarrow=False, xanchor="right", yanchor="bottom",
                font=dict(color=GREEN,size=11,family="JetBrains Mono"),
                bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1,borderpad=4)
            if geo_p>0:
                fig_cv.add_hline(y=geo_p,line=dict(color=BLUE,dash="dot"))
                fig_cv.add_annotation(
                    x=0.02, xref="paper", y=geo_p, yref="y",
                    text=f"Geometric = {geo_p:.4f}",
                    showarrow=False, xanchor="left", yanchor="bottom",
                    font=dict(color=BLUE,size=11,family="JetBrains Mono"),
                    bgcolor="#051D4D",bordercolor=BLUE,borderwidth=1,borderpad=4)
            fig_cv.update_layout(template="plotly_dark",height=300,
                title=_title("Monte Carlo Convergence"),
                xaxis=dict(title="Paths",type="log"),
                yaxis=dict(title=f"Price ({unit})"),
                margin=dict(l=60,r=20,t=75,b=50))
            st.plotly_chart(fig_cv, use_container_width=True)

        avgs = np.array(res["sample_avgs"])
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=avgs,nbinsx=50,marker_color=AMBER,
            opacity=0.7,histnorm="probability density",name="Avg prices"))
        fig_dist.add_vline(x=K,line=dict(color=GREEN,dash="dash",width=1.5),
            annotation_text=f"K = {K:.2f}",
            annotation_position="top left",
            annotation=dict(font=dict(color=GREEN,size=12,family="JetBrains Mono"),
                            bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1))
        fig_dist.add_vline(x=F_T,line=dict(color=BLUE,dash="dot",width=1.5),
            annotation_text=f"F = {F_T:.2f}",
            annotation_position="top right",
            annotation=dict(font=dict(color=BLUE,size=12,family="JetBrains Mono"),
                            bgcolor="#051D4D",bordercolor=BLUE,borderwidth=1))
        fig_dist.update_layout(template="plotly_dark",height=260,
            title=_title("Distribution of Average Prices at Expiry (100 paths sample)"),
            xaxis=dict(title=f"Avg price ({unit})"),yaxis=dict(title="Density"),
            margin=dict(l=60,r=20,t=75,b=50))
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── TAB 3 — CRACK SPREAD OPTIONS ─────────────────────────────────────────
    with t3:
        _tab_title(st, "Crack Spread Option Pricer — Kirk (1995)", "#FF7B72", "🏭")
        _tab_desc(st, "Options on the <b>gross refining margin</b> (crack spread). "
            "A refiner long crude and short refined products hedges margin compression with a put. "
            "Kirk (1995) converts the two-asset spread into an effective single-asset Black-76 problem.")

        _t3a, _t3b = st.columns([3,1])
        with _t3a:
            crack_name = st.selectbox("Crack spread type", list(CRACK_SPREADS.keys()))
        with _t3b:
            opt_type_c = st.radio("Option type", ["Call ▲","Put ▼"], horizontal=True, key="ot3")
            opt_type_c = "call" if "Call" in opt_type_c else "put"

        cs_def     = CRACK_SPREADS[crack_name]

        cc1,cc2,cc3 = st.columns(3)
        with cc1:
            st.markdown(f"**Leg 1 — {cs_def['F1']}**")
            _f1_fam  = next((fam for fam,comms in COMMODITY_REGISTRY.items()
                              if cs_def.get("F1") and cs_def["F1"] in comms), "Energy")
            _f1_spot = get_opcap_params(_f1_fam, cs_def["F1"])["spot"] if cs_def.get("F1") else 70.0
            F1_val   = st.number_input("F1 price", value=float(_f1_spot), step=0.1,
                                        key="f1_val", format="%.3f")
            vol1     = st.slider("F1 vol (%)",5,100,int(cs_def["vol1"]*100),1,key="v1")/100
        with cc2:
            st.markdown(f"**Leg 2 — {cs_def['F2'] or 'Power/Other'}**")
            _f2_key    = cs_def.get("F2")
            _f2_fam    = next((fam for fam,comms in COMMODITY_REGISTRY.items()
                               if _f2_key and _f2_key in comms), "Energy")
            F2_default = get_opcap_params(_f2_fam, _f2_key)["spot"]                          if _f2_key and _f2_key in COMMODITY_REGISTRY.get(_f2_fam,{})                          else float(CRACK_SPREADS[crack_name].get("typical", 60.0)) * 1.2
            F2_val   = st.number_input("F2 price", value=float(F2_default), step=0.1,
                                        key="f2_val", format="%.3f")
            vol2     = st.slider("F2 vol (%)",5,100,int(cs_def["vol2"]*100),1,key="v2")/100
        with cc3:
            st.markdown("**Spread Parameters**")
            rho_cs   = st.slider("Correlation F1/F2",-100,100,int(cs_def["rho"]*100),1,key="rcs")/100
            K_crack  = st.number_input("Strike on spread",value=float(cs_def["typical"]),
                                        step=0.5,key="k_crack",format="%.3f")
            # opt_type_c defined above

        mult1=cs_def["F1_mult"]; mult2=cs_def["F2_mult"]
        if "3-2-1" in crack_name:
            F1_conv=F1_val; F2_conv=F2_val*42*mult2/mult1
        else:
            F1_conv=F1_val*mult1; F2_conv=F2_val*mult2

        crack_opt = CrackSpreadOption(F1_conv,F2_conv,K_crack,T,r,vol1,vol2,rho_cs,opt_type_c)
        cr = crack_opt.price()

        st.markdown(f'<div style="font-size:.78rem;color:{GRAY};margin:6px 0 10px">'
                    f'{cs_def["description"]} &nbsp;|&nbsp; '
                    f'Current spread: {cr["current_spread"]:.3f} $/bbl &nbsp;|&nbsp; '
                    f'Kirk σ: {cr["sigma_kirk"]*100:.1f}%</div>', unsafe_allow_html=True)

        cr_html = f"""
        <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:14px">
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:11px 13px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Price</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.92rem;color:{AMBER}">{cr['price']:.4f}</div>
            <div style="font-size:.66rem;color:{GRAY};margin-top:2px">$/bbl</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:11px 13px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Intrinsic</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.92rem;color:{TEXT}">{cr['intrinsic']:.4f}</div>
            <div style="font-size:.66rem;color:{GRAY};margin-top:2px">$/bbl</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:11px 13px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Time Value</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.92rem;color:{BLUE}">{cr['time_value']:.4f}</div>
            <div style="font-size:.66rem;color:{GRAY};margin-top:2px">$/bbl</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:11px 13px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Δ crude</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.92rem;color:{RED}">{cr['delta_F1']:.4f}</div>
            <div style="font-size:.66rem;color:{GRAY};margin-top:2px">sensitivity F1</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:11px 13px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Δ product</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.92rem;color:{GREEN}">{cr['delta_F2']:.4f}</div>
            <div style="font-size:.66rem;color:{GRAY};margin-top:2px">sensitivity F2</div>
          </div>
        </div>"""
        st.markdown(cr_html, unsafe_allow_html=True)

        cr1,cr2 = st.columns(2)
        with cr1:
            sr = np.linspace(cr["current_spread"]-30, cr["current_spread"]+30, 300)
            pv = crack_opt.payoff_grid(sr); pnl_cr = pv-cr["price"]
            fig_cr = go.Figure()
            fig_cr.add_trace(go.Scatter(x=sr,y=pnl_cr,name="P&L at expiry",
                line=dict(color=AMBER,width=2.5)))
            fig_cr.add_trace(go.Scatter(x=sr,y=pv,name="Payoff (intrinsic)",
                line=dict(color=AMBER,width=1,dash="dot"),opacity=0.4))
            fig_cr.add_hline(y=0,line=dict(color=GRAY,width=0.8))
            fig_cr.add_vline(x=K_crack,line=dict(color=GREEN,dash="dash"),
                annotation_text=f"K = {K_crack:.1f}",
                annotation_position="top left",
                annotation=dict(yref="paper",y=0.96,
                    font=dict(color=GREEN,size=11,family="JetBrains Mono"),
                    bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1))
            fig_cr.add_vline(x=cr["current_spread"],line=dict(color=RED,dash="dot"),
                annotation_text=f"Spread = {cr['current_spread']:.1f}",
                annotation_position="top right",
                annotation=dict(yref="paper",y=0.80,
                    font=dict(color=RED,size=11,family="JetBrains Mono"),
                    bgcolor="#2D0A09",bordercolor=RED,borderwidth=1))
            fig_cr.update_layout(template="plotly_dark",height=330,
                title=dict(text="Crack Spread — P&L at Expiry",y=0.97,x=0.5,
                           xanchor="center",font=dict(size=12,color="#8B949E")),
                xaxis=dict(title="Crack spread at expiry ($/bbl)"),
                yaxis=dict(title="P&L ($/bbl)"),
                legend=dict(orientation="h",yanchor="bottom",y=1.02),
                margin=dict(l=60,r=20,t=75,b=55))
            st.plotly_chart(fig_cr, use_container_width=True)
        with cr2:
            K_rng  = np.linspace(max(0,K_crack-15),K_crack+15,12)
            rho_rng= np.linspace(0.5,0.99,8)
            surf   = np.zeros((len(rho_rng),len(K_rng)))
            for i,rv in enumerate(rho_rng):
                for j,kv in enumerate(K_rng):
                    surf[i,j]=CrackSpreadOption(F1_conv,F2_conv,kv,T,r,vol1,vol2,rv,
                                                opt_type_c).price()["price"]
            fig_cs = go.Figure(go.Heatmap(z=surf,
                x=[f"{k:.1f}" for k in K_rng],y=[f"{rv:.2f}" for rv in rho_rng],
                colorscale="YlOrRd",
                hovertemplate="K: %{x}<br>ρ: %{y}<br>Price: %{z:.4f}<extra></extra>"))
            fig_cs.update_layout(template="plotly_dark",height=330,
                title=_title("Price Surface: Strike × Correlation"),
                xaxis=dict(title="Strike ($/bbl)"),yaxis=dict(title="Correlation"),
                margin=dict(l=60,r=20,t=75,b=50))
            st.plotly_chart(fig_cs, use_container_width=True)

    # ── TAB 4 — CALENDAR SPREAD OPTIONS ──────────────────────────────────────
    with t4:
        _tab_title(st, "Calendar Spread Option Pricer — Kirk Approximation", "#F0A500", "📅")
        _tab_desc(st, "Options on the <b>slope of the forward curve</b> (M_near − M_far). "
            "A call pays off when backwardation widens; a put when contango deepens. "
            "Used to hedge roll yield risk or to trade curve steepening/flattening.")

        _t4hdr1, _t4hdr2 = st.columns([1,3])
        with _t4hdr1:
            opt_type_cs = st.radio("Option type", ["Call ▲","Put ▼"], horizontal=True, key="ot4")
            opt_type_cs = "call" if "Call" in opt_type_cs else "put"

        csp1,csp2 = st.columns(2)
        with csp1:
            near_m = st.selectbox("Near contract",[f"M{m}" for m in range(1,13)],index=0)
            far_m  = st.selectbox("Far contract", [f"M{m}" for m in range(1,25)],index=5)
            ni=min(int(near_m[1:])-1,len(fwd_df)-1); fi=min(int(far_m[1:])-1,len(fwd_df)-1)
            F_near=float(fwd_df["forward"].iloc[ni]); F_far=float(fwd_df["forward"].iloc[fi])
            cs_spr=F_near-F_far
        with csp2:
            K_cs    = st.number_input("Strike on spread",value=round(cs_spr*0.8,3),
                                       step=0.01,key="k_cs",format="%.4f")
            vol_n   = st.slider("Near vol (%)",5,100,int(vol*100*1.05),1,key="vn")/100
            vol_f   = st.slider("Far vol (%)",5,100,int(vol*100*0.90),1,key="vf")/100
            rho_cs2 = st.slider("Correlation",70,100,95,1,key="rho2")/100
            T_cs    = max(float(fwd_df["T_yr"].iloc[ni]),0.01)
            # opt_type_cs defined above

        cso = CalendarSpreadOption(F_near,F_far,K_cs,min(T,T_cs),r,
                                    vol_n,vol_f,rho_cs2,opt_type_cs,near_m,far_m)
        csr = cso.price()

        st.markdown(
            f'<div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:8px;'
            f'padding:9px 14px;font-family:JetBrains Mono,monospace;font-size:.73rem;'
            f'color:{GRAY};margin-bottom:10px">'
            f'{near_m}: {F_near:.4f} {unit} &nbsp;|&nbsp; {far_m}: {F_far:.4f} {unit} &nbsp;|&nbsp; '
            f'Spread: {cs_spr:+.4f} &nbsp;|&nbsp; Kirk σ: {csr["sigma_kirk"]*100:.2f}% &nbsp;|&nbsp; '
            f'{"BACKWARDATION" if cs_spr>0 else "CONTANGO"}</div>',
            unsafe_allow_html=True)

        csr_html = f"""
        <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:7px;margin-bottom:14px">
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:10px 12px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:3px">Price</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.88rem;color:{AMBER}">{csr['price']:.6f}</div>
            <div style="font-size:.66rem;color:{GRAY};margin-top:2px">{unit}</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:10px 12px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:3px">Intrinsic</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.88rem;color:{TEXT}">{csr['intrinsic']:.6f}</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:10px 12px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:3px">Time Value</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.88rem;color:{BLUE}">{csr['time_value']:.6f}</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:10px 12px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:3px">Δ near</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.88rem;color:{GREEN}">{csr['delta_near']:.5f}</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:10px 12px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:3px">Δ far</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.88rem;color:{RED}">{csr['delta_far']:.5f}</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:10px 12px">
            <div style="font-size:.62rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:3px">Vega</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.88rem;color:{PURPLE}">{csr['vega']:.6f}</div>
          </div>
        </div>"""
        st.markdown(csr_html, unsafe_allow_html=True)

        csp_a,csp_b = st.columns(2)
        with csp_a:
            sv_rng = np.linspace(cs_spr-20,cs_spr+20,300)
            py_cs  = cso.payoff_grid(sv_rng); pnl_cs=py_cs-csr["price"]
            fig_csp = go.Figure()
            fig_csp.add_trace(go.Scatter(x=sv_rng,y=pnl_cs,name="P&L at expiry",
                line=dict(color=AMBER,width=2.5)))
            fig_csp.add_hline(y=0,line=dict(color=GRAY,width=0.8))
            fig_csp.add_vline(x=K_cs,line=dict(color=GREEN,dash="dash"),
                annotation_text=f"K = {K_cs:.2f}",
                annotation_position="top left",
                annotation=dict(yref="paper",y=0.96,
                    font=dict(color=GREEN,size=11,family="JetBrains Mono"),
                    bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1))
            fig_csp.add_vline(x=cs_spr,line=dict(color=RED,dash="dot"),
                annotation_text=f"Spread = {cs_spr:.2f}",
                annotation_position="top right",
                annotation=dict(yref="paper",y=0.80,
                    font=dict(color=RED,size=11,family="JetBrains Mono"),
                    bgcolor="#2D0A09",bordercolor=RED,borderwidth=1))
            fig_csp.update_layout(template="plotly_dark",height=330,
                title=dict(text=f"Calendar Spread {near_m}−{far_m} — P&L at Expiry",
                           y=0.97,x=0.5,xanchor="center",font=dict(size=12,color="#8B949E")),
                xaxis=dict(title=f"{near_m}−{far_m} spread ({unit})"),
                yaxis=dict(title=f"P&L ({unit})"),
                margin=dict(l=60,r=20,t=75,b=55))
            st.plotly_chart(fig_csp, use_container_width=True)
        with csp_b:
            if len(fwd_df)>=4:
                pairs=[]
                seen=set()
                for i in range(len(fwd_df)-1):
                    for j in [i+1,min(i+2,len(fwd_df)-1),min(i+5,len(fwd_df)-1)]:
                        if j<len(fwd_df) and j!=i and (i,j) not in seen:
                            seen.add((i,j))
                            Fn_=float(fwd_df["forward"].iloc[i]); Ff_=float(fwd_df["forward"].iloc[j])
                            T_=max(float(fwd_df["T_yr"].iloc[i]),0.01)
                            p_=CalendarSpreadOption(Fn_,Ff_,0,T_,r,vol*1.05,vol*0.95,
                                                    rho_cs2,"call",f"M{i+1}",f"M{j+1}").price()["price"]
                            pairs.append({"pair":f"M{i+1}-M{j+1}","spread":round(Fn_-Ff_,4),"price":round(p_,4)})
                spdf=pd.DataFrame(pairs[:20])
                fig_sp=go.Figure()
                fig_sp.add_trace(go.Bar(x=spdf["pair"],y=spdf["price"],name="ATM option price",
                    marker_color=[GREEN if s<0 else RED for s in spdf["spread"]],opacity=0.8))
                fig_sp.add_trace(go.Scatter(x=spdf["pair"],y=spdf["spread"],name="Spread",
                    mode="lines+markers",line=dict(color=AMBER,width=1.5),yaxis="y2"))
                fig_sp.update_layout(template="plotly_dark",height=330,
                    title=_title("ATM Calendar Spread Prices across Curve"),
                    xaxis=dict(title="Pair"),yaxis=dict(title=f"Option price ({unit})"),
                    yaxis2=dict(title="Spread",overlaying="y",side="right",showgrid=False),
                    legend=dict(orientation="h",yanchor="bottom",y=1.02),
                    margin=dict(l=60,r=80,t=75,b=55))
            st.plotly_chart(fig_sp, use_container_width=True)

    # ── TAB 5 — COMMODITY SWAPS ───────────────────────────────────────────────
    with t5:
        _tab_title(st, "Commodity Fixed-for-Floating Swap Pricer", "#3FB950", "🔄")
        _tab_desc(st, "One party pays a <b>fixed price</b> per period; the other pays the "
            "floating market average. Airlines pay fixed on jet fuel to hedge procurement. "
            "Refiners receive fixed on crude to lock in revenue. "
            "NPV = Σ disc_i × (F_i − K) × notional.")

        sw1,sw2,sw3 = st.columns(3)
        with sw1:
            swap_n    = st.slider("Number of periods",3,24,12)
            swap_freq = st.radio("Payment frequency",["monthly","quarterly"],horizontal=True)
        with sw2:
            swap_not  = st.number_input("Notional (units/period)",value=1000.0,step=100.0,format="%.0f")
            swap_pos  = st.radio("Position",
                ["Fixed payer (pay fixed, receive float)",
                 "Fixed receiver (receive fixed, pay float)"],horizontal=False)
        with sw3:
            fwd_swap = fwd_df["forward"].values[:swap_n].tolist()
            if len(fwd_swap)<swap_n:
                fwd_swap += [fwd_swap[-1]]*(swap_n-len(fwd_swap)) if fwd_swap else [spot]*swap_n
            sw_tmp   = CommoditySwap(fwd_swap,spot,r,swap_not,swap_freq,"fixed_payer")
            fair_def = round(sw_tmp.fair_rate,4)
            swap_fix = st.number_input("Fixed rate",value=fair_def,step=0.01,
                                        format="%.4f",
                                        help=f"Fair (break-even) rate = {fair_def:.4f} {unit}")

        pos_str = "fixed_payer" if "payer" in swap_pos else "fixed_receiver"
        sw = CommoditySwap(fwd_swap,swap_fix,r,swap_not,swap_freq,pos_str)
        npv=sw.npv(); fair_r=sw.fair_rate; dv01=sw.dv01(); cf_df2=sw.cashflows()

        npv_c = GREEN if npv>0 else RED
        sw_html = f"""
        <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:14px">
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:12px 14px">
            <div style="font-size:.64rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">NPV</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.95rem;color:{npv_c}">{npv:+.2f}</div>
            <div style="font-size:.68rem;color:{GRAY};margin-top:3px">{unit} · notional</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:12px 14px">
            <div style="font-size:.64rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Fair Rate</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.95rem;color:{AMBER}">{fair_r:.4f}</div>
            <div style="font-size:.68rem;color:{GRAY};margin-top:3px">{unit} break-even</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:12px 14px">
            <div style="font-size:.64rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Fixed Rate</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.95rem;color:{TEXT}">{swap_fix:.4f}</div>
            <div style="font-size:.68rem;color:{GRAY};margin-top:3px">{unit} agreed</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:12px 14px">
            <div style="font-size:.64rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">DV01</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.95rem;color:{BLUE}">{dv01:.4f}</div>
            <div style="font-size:.68rem;color:{GRAY};margin-top:3px">per 1bp shift</div>
          </div>
          <div style="background:{PANEL};border:0.5px solid {BORDER};border-radius:10px;padding:12px 14px">
            <div style="font-size:.64rem;font-weight:500;color:{GRAY};text-transform:uppercase;margin-bottom:4px">Position</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.78rem;color:{TEXT}">{pos_str.upper()}</div>
            <div style="font-size:.68rem;color:{GRAY};margin-top:3px">{swap_n} periods</div>
          </div>
        </div>"""
        st.markdown(sw_html, unsafe_allow_html=True)

        swa,swb = st.columns(2)
        with swa:
            cf_data2 = cf_df2[cf_df2["Period"]!="TOTAL"].copy()
            cf_data2["Period"]=cf_data2["Period"].astype(int)
            pv_v=cf_data2["PV cashflow"].astype(float).values
            nt_v=cf_data2["Net cashflow"].astype(float).values
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Bar(x=cf_data2["Period"],y=nt_v,name="Net cashflow",
                marker_color=[GREEN if v>0 else RED for v in nt_v],opacity=0.7))
            fig_cf.add_trace(go.Bar(x=cf_data2["Period"],y=pv_v,name="PV cashflow",
                marker_color=[AMBER if v>0 else BLUE for v in pv_v],opacity=0.85))
            fig_cf.add_hline(y=0,line=dict(color=GRAY,width=0.8))
            fig_cf.update_layout(template="plotly_dark",height=330,barmode="group",
                title=_title("Cashflows per Period"),
                xaxis=dict(title="Period"),
                yaxis=dict(title=f"Cashflow ({unit}·notional)"),
                legend=dict(orientation="h",yanchor="bottom",y=1.02),
                margin=dict(l=60,r=20,t=75,b=50))
            st.plotly_chart(fig_cf, use_container_width=True)
        with swb:
            _shift_scale = max(spot_default * 0.10, 0.5)  # ±10% of spot
            sens = sw.sensitivity(np.linspace(-_shift_scale, _shift_scale, 41))
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=sens["shift"],y=sens["npv"],mode="lines",
                name="NPV",line=dict(color=AMBER,width=2.5)))
            fig_s.add_hline(y=0,line=dict(color=GRAY,width=0.8))
            fig_s.add_vline(x=0,line=dict(color=GREEN,dash="dash",width=1),
                annotation_text="Current",
                annotation_position="top right",
                annotation=dict(font=dict(color=GREEN,size=11,family="JetBrains Mono"),
                    bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1))
            fig_s.update_layout(template="plotly_dark",height=330,
                title=_title("NPV Sensitivity to Forward Curve Shift"),
                xaxis=dict(title=f"Parallel shift ({unit})"),
                yaxis=dict(title=f"NPV ({unit}·notional)"),
                margin=dict(l=60,r=20,t=75,b=50))
            st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("**Cashflow Schedule**")
        st.dataframe(cf_df2, use_container_width=True, hide_index=True, height=260)

    # ── TAB 6 — FORWARD CURVE ─────────────────────────────────────────────────
    with t8:
        _tab_title(st, "Forward Curve", "#58A6FF", "📊")
        st.markdown(
            f'<div style="font-size:0.78rem;color:#8B949E;margin-bottom:12px">'
            f'Source: <b>{src_lbl}</b> &nbsp;·&nbsp; {commodity} &nbsp;·&nbsp; {unit}</div>',
            unsafe_allow_html=True)

        fc1,fc2 = st.columns([2,1])
        with fc1:
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=fwd_df["month"],y=fwd_df["forward"],
                mode="lines+markers",name=commodity,
                line=dict(color=AMBER,width=2.5),marker=dict(size=7)))
            if live_df is None:
                theo = [forward_from_spot(spot,r,storage,convenience,m/12) for m in fwd_df["month"]]
                fig_fc.add_trace(go.Scatter(x=fwd_df["month"],y=theo,mode="lines",
                    name="Cost-of-carry",line=dict(color=BLUE,width=1.5,dash="dot")))
            fig_fc.add_hline(y=F_m1,line=dict(color=GREEN,width=1,dash="dashdot"),
                annotation_text=f"M1 = {F_m1:.3f}",
                annotation_position="right",
                annotation=dict(font=dict(color=GREEN,size=11,family="JetBrains Mono"),
                    bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1))
            lbs = fwd_df["label"].tolist() if "label" in fwd_df else [f"M{m}" for m in fwd_df["month"]]
            fig_fc.update_layout(template="plotly_dark",height=370,
                xaxis=dict(title="Maturity (months)",tickvals=fwd_df["month"].tolist()[::2],ticktext=lbs[::2]),
                yaxis=dict(title=unit),
                legend=dict(orientation="h",yanchor="bottom",y=1.02),
                margin=dict(l=60,r=80,t=75,b=55),hovermode="x unified")
            st.plotly_chart(fig_fc, use_container_width=True)
        with fc2:
            st.markdown("**Curve statistics**")
            stats = {"M1":f"{F_m1:.4f}","M3":f"{F_m3:.4f}","M6":f"{F_m6:.4f}","M12":f"{F_m12:.4f}",
                     "M1-M6 spread":f"{F_m6-F_m1:+.4f}","M1-M12 spread":f"{F_m12-F_m1:+.4f}",
                     "Structure":"BACKWARDATION" if F_m6<F_m1 else "CONTANGO",
                     "Source":src_lbl,"Vol":f"{vol*100:.1f}%",
                     "RF":f"{r*100:.2f}%","Storage":f"{storage*100:.1f}%/yr",
                     "Conv. yield":f"{convenience*100:.1f}%/yr"}
            for k,v in stats.items():
                col = GREEN if k=="Structure" and "BACK" in v \
                      else RED if k=="Structure" else TEXT
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:3px 0;border-bottom:0.5px solid {BORDER}">'
                    f'<span style="color:{GRAY};font-size:11px">{k}</span>'
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:{col}">{v}</span></div>',
                    unsafe_allow_html=True)

        display = fwd_df.copy()
        if "label" not in display: display["label"]=[f"M{m}" for m in display["month"]]
        if "vol" not in display:   display["vol"]=vol
        display["theoretical"]=[round(forward_from_spot(spot,r,storage,convenience,float(t)),4)
                                 for t in display["T_yr"]]
        display["basis"]=(display["forward"]-display["theoretical"]).round(4)
        st.markdown("**Forward Curve Data**")
        st.dataframe(display[["label","month","T_yr","forward","theoretical","basis","vol"]],
                     use_container_width=True,hide_index=True,height=240)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — BARRIER OPTIONS
    # ══════════════════════════════════════════════════════════════════════════
    with t7:
        _tab_title(st, "Barrier Options — Knock-In / Knock-Out", "#F0A500", "🚧")
        _tab_desc(st, "Barrier options activate (knock-in) or expire worthless (knock-out) "
            "if the underlying hits a barrier level during the option's life. "
            "Cheaper than vanilla — used by airlines (KO calls) and producers (KI puts). "
            "Priced via daily Monte Carlo simulation.")

        _b1, _b2, _b3 = st.columns(3)
        with _b1:
            barrier_type = st.selectbox("Barrier type", [
                "Down-and-Out (KO)", "Down-and-In (KI)",
                "Up-and-Out (KO)",   "Up-and-In (KI)"])
            opt_type_b = st.radio("Option type", ["Call ▲", "Put ▼"],
                                   horizontal=True, key="ot7")
            opt_type_b = "call" if "Call" in opt_type_b else "put"
        with _b2:
            _b_default = 85.0 if "Down" in barrier_type else 115.0
            barrier_pct = st.number_input("Barrier level (% of spot)",
                30.0, 200.0, _b_default, 1.0, format="%.1f")
            B = round(F_T * barrier_pct / 100, 4)
            rebate = st.number_input("Rebate (if knocked)", 0.0,
                                      F_T * 0.1, 0.0, 0.01, format="%.3f")
        with _b3:
            st.markdown(
                f'<div style="background:#161B22;border:0.5px solid #30363D;' +
                f'border-radius:8px;padding:10px 14px;font-family:JetBrains Mono,' +
                f'monospace;font-size:.78rem;color:#8B949E;line-height:1.8">' +
                f'F(T) = <b style="color:#E6EDF3">{F_T:.3f}</b> {unit}<br>' +
                f'K = <b style="color:#3FB950">{K:.3f}</b> {unit}<br>' +
                f'Barrier B = <b style="color:#FF7B72">{B:.3f}</b> {unit}<br>' +
                f'B / F = <b style="color:#E6EDF3">{B/F_T*100:.1f}%</b><br>' +
                f'σ = {vol*100:.1f}%  T = {T:.3f} yr</div>',
                unsafe_allow_html=True)

        with st.spinner(f"Running {n_paths:,} daily paths..."):
            np.random.seed(42)
            n_steps = max(int(T * 252), 1)
            Z_b     = np.random.standard_normal((n_paths, n_steps))
            dt_b    = T / n_steps
            log_inc = (-0.5*vol**2*dt_b) + vol*np.sqrt(dt_b)*Z_b
            paths_b = F_T * np.exp(np.cumsum(log_inc, axis=1))
            full_b  = np.hstack([np.full((n_paths,1), F_T), paths_b])
            S_T_b   = full_b[:, -1]
            p_min   = full_b.min(axis=1)
            p_max   = full_b.max(axis=1)
            disc_b  = np.exp(-r * T)
            vpay    = np.maximum(S_T_b-K,0) if opt_type_b=="call" else np.maximum(K-S_T_b,0)

            if barrier_type == "Down-and-Out (KO)":
                alive = p_min > B
                bpay  = np.where(alive, vpay, rebate)
                kp    = (~alive).mean()*100
            elif barrier_type == "Down-and-In (KI)":
                ki    = p_min <= B
                bpay  = np.where(ki, vpay, rebate)
                kp    = ki.mean()*100
            elif barrier_type == "Up-and-Out (KO)":
                alive = p_max < B
                bpay  = np.where(alive, vpay, rebate)
                kp    = (~alive).mean()*100
            else:
                ki    = p_max >= B
                bpay  = np.where(ki, vpay, rebate)
                kp    = ki.mean()*100

            bprice  = disc_b * bpay.mean()
            bse     = disc_b * bpay.std() / np.sqrt(n_paths)
            vprice  = disc_b * vpay.mean()
            disc_p  = (vprice-bprice)/vprice*100 if vprice>0 else 0

        kpi_bar = (
            f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px">' +
            _kpi("Barrier Price",  f"{bprice:.4f}", f"± {bse:.4f}", AMBER) +
            _kpi("Vanilla Price",  f"{vprice:.4f}", "reference",    BLUE) +
            _kpi("Discount",       f"{disc_p:.1f}%", "vs vanilla",
                 GREEN if disc_p>0 else RED) +
            _kpi("Barrier B",      f"{B:.3f} {unit}", f"{barrier_pct:.1f}% of spot", RED) +
            _kpi("Knock prob.",    f"{kp:.1f}%", "knock event") +
            '</div>'
        )
        st.markdown(kpi_bar, unsafe_allow_html=True)

        _bc1, _bc2 = st.columns(2)
        with _bc1:
            fig_bar = go.Figure()
            for i in range(min(40, n_paths)):
                alive_i = (p_min[i]>B if "Down" in barrier_type else p_max[i]<B)                           if "Out" in barrier_type                           else (p_min[i]<=B if "Down" in barrier_type else p_max[i]>=B)
                c = AMBER if alive_i else GRAY
                fig_bar.add_trace(go.Scatter(
                    x=np.linspace(0,T,n_steps+1), y=full_b[i],
                    mode="lines", line=dict(width=0.5, color=c),
                    opacity=0.35, showlegend=False))
            fig_bar.add_hline(y=B, line=dict(color=RED, width=2, dash="dash"))
            fig_bar.add_annotation(
                x=0.72, xref="paper", y=B, yref="y",
                text=f"B = {B:.2f}", showarrow=False,
                xanchor="right", yanchor="bottom",
                font=dict(color=RED,size=11,family="JetBrains Mono"),
                bgcolor="#2D0A09",bordercolor=RED,borderwidth=1,borderpad=4)
            fig_bar.add_hline(y=K, line=dict(color=GREEN, width=1.5, dash="dot"))
            fig_bar.add_annotation(
                x=0.02, xref="paper", y=K, yref="y",
                text=f"K = {K:.2f}", showarrow=False,
                xanchor="left", yanchor="bottom",
                font=dict(color=GREEN,size=11,family="JetBrains Mono"),
                bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1,borderpad=4)
            fig_bar.update_layout(template="plotly_dark", height=320,
                title=_title("MC Paths — Alive (amber) vs Knocked (gray)"),
                xaxis=dict(title="Time (yr)"),
                yaxis=dict(title=f"Price ({unit})"),
                margin=dict(l=60,r=20,t=75,b=50))
            st.plotly_chart(fig_bar, use_container_width=True)

        with _bc2:
            b_rng = np.linspace(F_T*0.5, F_T*1.5, 40)
            b_pxs = []
            for bv in b_rng:
                if barrier_type=="Down-and-Out (KO)":
                    px=disc_b*np.where(p_min>bv,vpay,rebate).mean()
                elif barrier_type=="Down-and-In (KI)":
                    px=disc_b*np.where(p_min<=bv,vpay,rebate).mean()
                elif barrier_type=="Up-and-Out (KO)":
                    px=disc_b*np.where(p_max<bv,vpay,rebate).mean()
                else:
                    px=disc_b*np.where(p_max>=bv,vpay,rebate).mean()
                b_pxs.append(px)
            fig_bv = go.Figure()
            fig_bv.add_trace(go.Scatter(x=b_rng, y=b_pxs, mode="lines",
                name="Barrier price", line=dict(color=AMBER,width=2.5)))
            fig_bv.add_hline(y=vprice, line=dict(color=BLUE,dash="dot"))
            fig_bv.add_annotation(
                x=0.60, xref="paper", y=vprice, yref="y",
                text=f"Vanilla = {vprice:.4f}", showarrow=False,
                xanchor="right", yanchor="bottom",
                font=dict(color=BLUE,size=11,family="JetBrains Mono"),
                bgcolor="#051D4D",bordercolor=BLUE,borderwidth=1,borderpad=4)
            fig_bv.add_vline(x=F_T, line=dict(color=GREEN,dash="dashdot"),
                annotation_text=f"F = {F_T:.2f}",
                annotation_position="top right",
                annotation=dict(yref="paper", y=0.96,
                    font=dict(color=GREEN,size=11,family="JetBrains Mono"),
                    bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1,borderpad=4))
            fig_bv.update_layout(template="plotly_dark", height=320,
                title=_title("Price vs Barrier Level"),
                xaxis=dict(title=f"Barrier ({unit})"),
                yaxis=dict(title=f"Option price ({unit})"),
                margin=dict(l=60,r=20,t=75,b=50))
            st.plotly_chart(fig_bv, use_container_width=True)

        # P&L at expiry
        sr     = np.linspace(K*0.4, K*1.6, 400)
        vpnl   = (np.maximum(sr-K,0) if opt_type_b=="call" else np.maximum(K-sr,0)) - vprice
        bpnl   = vpnl.copy()
        mask   = sr<=B if "Down" in barrier_type else sr>=B
        if "Out" in barrier_type: bpnl[mask]  = rebate - bprice
        else:                     bpnl[~mask] = rebate - bprice
        fig_bp = go.Figure()
        fig_bp.add_trace(go.Scatter(x=sr,y=vpnl,name="Vanilla",
            line=dict(color=GRAY,width=1.5,dash="dot"),opacity=0.6))
        fig_bp.add_trace(go.Scatter(x=sr,y=bpnl,name="Barrier",
            line=dict(color=AMBER,width=2.5)))
        fig_bp.add_hline(y=0,line=dict(color=GRAY,width=0.8))
        fig_bp.add_vline(x=B, line=dict(color=RED,dash="dash",width=1.5),
            annotation_text=f"B = {B:.2f}",annotation_position="top left",
            annotation=dict(font=dict(color=RED,size=11,family="JetBrains Mono"),
                bgcolor="#2D0A09",bordercolor=RED,borderwidth=1,borderpad=4))
        fig_bp.add_vline(x=K, line=dict(color=GREEN,dash="dot",width=1.5),
            annotation_text=f"K = {K:.2f}",annotation_position="top right",
            annotation=dict(font=dict(color=GREEN,size=11,family="JetBrains Mono"),
                bgcolor="#0D2819",bordercolor=GREEN,borderwidth=1,borderpad=4))
        fig_bp.update_layout(template="plotly_dark",height=280,
            title=_title("P&L at Expiry — Barrier vs Vanilla"),
            xaxis=dict(title=f"Price at expiry ({unit})"),
            yaxis=dict(title=f"P&L ({unit})"),
            legend=dict(orientation="h",yanchor="bottom",y=1.02),
            margin=dict(l=60,r=20,t=75,b=50))
        st.plotly_chart(fig_bp, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 — VOLATILITY SURFACE  (now tab label says 📊 Vol Surface)
    # ══════════════════════════════════════════════════════════════════════════
    with t6:
        _tab_title(st, "Implied Volatility Surface", "#58A6FF", "📊")
        _tab_desc(st, "How implied volatility varies across strikes (smile/skew) "
            "and maturities (term structure). Built with Black-76 + parametric skew. "
            "Skew < 0 = puts expensive (normal commodity market). "
            "Hover the 3D surface to read individual vol levels.")

        _vs1,_vs2,_vs3 = st.columns(3)
        with _vs1:
            atm_v  = st.slider("ATM vol (%)",5,100,int(vol*100),1,key="vs_atm")/100
            skew_v = st.slider("Skew (put wing premium)",-20,20,-3,1,key="vs_sk")/100
        with _vs2:
            curv_v = st.slider("Smile curvature",0,20,5,1,key="vs_cur")/100
            vov_v  = st.slider("Term structure steepness",0,30,8,1,key="vs_vov")/100
        with _vs3:
            st.markdown(
                f'<div style="background:#161B22;border:0.5px solid #30363D;' +
                f'border-radius:8px;padding:10px 14px;font-size:.76rem;' +
                f'color:#8B949E;line-height:1.7">' +
                f'<b style="color:#E6EDF3">σ(K,T)</b> = σ_ATM·(1+vov·√T)<br>' +
                f'&nbsp;&nbsp;&nbsp;+ skew·ln(K/F)<br>' +
                f'&nbsp;&nbsp;&nbsp;+ curvature·ln(K/F)²<br><br>' +
                f'<span style="color:#F0A500">Skew &lt; 0</span>: put wing rich (commodity norm)<br>' +
                f'<span style="color:#58A6FF">Curvature &gt; 0</span>: symmetric smile</div>',
                unsafe_allow_html=True)

        K_rng = np.linspace(F_T*0.60, F_T*1.40, 25)
        T_rng = np.array([1/12,2/12,3/12,6/12,9/12,1,1.5,2])
        T_lbl = ["1M","2M","3M","6M","9M","1Y","18M","2Y"]

        def _vsig(K_arr, t_val):
            at = atm_v*(1+vov_v*np.sqrt(t_val))
            lm = np.log(K_arr/F_T)
            return np.clip(at + skew_v*lm + curv_v*lm**2, 0.01, 5.0)

        vol_s   = np.array([_vsig(K_rng,t)*100 for t in T_rng])
        price_s = np.zeros_like(vol_s)
        for i,t_val in enumerate(T_rng):
            for j,(kv,sv) in enumerate(zip(K_rng,_vsig(K_rng,t_val))):
                price_s[i,j] = Black76(F_T,kv,t_val,r,sv,"call").price()

        fig_3d = go.Figure(go.Surface(
            z=vol_s, x=K_rng, y=T_lbl,
            colorscale="RdYlGn_r", opacity=0.92,
            contours=dict(z=dict(show=True,usecolormap=True,project_z=True)),
            hovertemplate="K: %{x:.2f}<br>T: %{y}<br>Vol: %{z:.2f}%<extra></extra>"))
        fig_3d.update_layout(template="plotly_dark",height=440,
            title=_title("Implied Volatility Surface (%)"),
            scene=dict(
                xaxis=dict(title=f"Strike ({unit})",gridcolor="#30363D",backgroundcolor="#0D1117"),
                yaxis=dict(title="Maturity",gridcolor="#30363D",backgroundcolor="#0D1117"),
                zaxis=dict(title="Vol (%)",gridcolor="#30363D",backgroundcolor="#0D1117"),
                bgcolor="#0D1117",
                camera=dict(eye=dict(x=1.8,y=-1.8,z=0.8))),
            margin=dict(l=10,r=10,t=75,b=10))
        st.plotly_chart(fig_3d, use_container_width=True)

        _vsc1,_vsc2 = st.columns(2)
        with _vsc1:
            fig_sm = go.Figure()
            cols_sm = [AMBER,BLUE,GREEN,RED,PURPLE,"#39D0D8","#E3B341",GRAY]
            for i,(t_val,t_lbl) in enumerate(zip(T_rng,T_lbl)):
                fig_sm.add_trace(go.Scatter(
                    x=K_rng/F_T*100, y=_vsig(K_rng,t_val)*100,
                    mode="lines",name=t_lbl,
                    line=dict(color=cols_sm[i%len(cols_sm)],width=1.8)))
            fig_sm.add_vline(x=100,line=dict(color=GRAY,dash="dash"),
                annotation_text="ATM",annotation_position="top right",
                annotation=dict(font=dict(color=GRAY,size=11,family="JetBrains Mono"),
                    bgcolor="#1C2128",bordercolor=GRAY,borderwidth=1,borderpad=3))
            fig_sm.update_layout(template="plotly_dark",height=320,
                title=dict(text="Vol Smile by Maturity", x=0.5,
                           xanchor="center", y=0.97,
                           font=dict(size=15,color="#E6EDF3",family="Inter")),
                xaxis=dict(title="Strike (% of spot)"),
                yaxis=dict(title="Impl. Vol (%)"),
                legend=dict(orientation="h",yanchor="bottom",y=1.02,font=dict(size=10)),
                margin=dict(l=60,r=20,t=115,b=50))
            st.plotly_chart(fig_sm, use_container_width=True)
        with _vsc2:
            atm_ts = [atm_v*(1+vov_v*np.sqrt(t))*100 for t in T_rng]
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=[t*12 for t in T_rng],y=atm_ts,
                mode="lines+markers",name="ATM vol",
                line=dict(color=AMBER,width=2.5),
                marker=dict(size=8,color=AMBER,line=dict(color="#0D1117",width=1.5))))
            fig_ts.add_hline(y=atm_v*100, line=dict(color=GRAY,dash="dot"))
            fig_ts.add_annotation(
                x=0.55, xref="paper", y=atm_v*100, yref="y",
                text=f"Flat = {atm_v*100:.1f}%", showarrow=False,
                xanchor="right", yanchor="bottom",
                font=dict(color=GRAY,size=11,family="JetBrains Mono"),
                bgcolor="#1C2128",bordercolor=GRAY,borderwidth=1,borderpad=3)
            fig_ts.update_layout(template="plotly_dark",height=300,
                title=_title("ATM Vol Term Structure"),
                xaxis=dict(title="Maturity (months)"),
                yaxis=dict(title="Vol (%)"),
                margin=dict(l=60,r=20,t=75,b=50))
            st.plotly_chart(fig_ts, use_container_width=True)

        fig_hp = go.Figure(go.Heatmap(
            z=price_s, x=[f"{k:.1f}" for k in K_rng[::2]], y=T_lbl,
            colorscale="YlOrRd",
            hovertemplate="K: %{x}<br>T: %{y}<br>Price: %{z:.4f}<extra></extra>"))
        fig_hp.update_layout(template="plotly_dark",height=250,
            title=_title("Call Price Heatmap (Strike × Maturity)"),
            xaxis=dict(title=f"Strike ({unit})"),yaxis=dict(title="Maturity"),
            margin=dict(l=60,r=20,t=75,b=50))
        st.plotly_chart(fig_hp, use_container_width=True)


    st.markdown("---")
    st.caption(f"CODAP · Commodity Options & Derivatives Analytics Platform · "
           f"{datetime.now().strftime('%d %b %Y  %H:%M')} · ™ by AEG")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — set_page_config called only inside run_streamlit_app()
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__" or _is_streamlit():
    if _is_streamlit():
        run_streamlit_app()
    else:
        print("OCDAP — Options & Commodity Derivatives Analytics Platform")
        print("Run with: streamlit run opcap.py")
