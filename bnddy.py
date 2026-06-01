"""
Portfolio Optimizer -- Streamlit Edition  (c) AEG
Markowitz Modern Portfolio Theory -- Premium UI
Scenario Analysis · Multi-Currency · Weight Constraints · Ticker Validation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
import yfinance as yf
from datetime import date

# ─── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root {
    --bg:#05080D; --bg2:#0A1220; --border:rgba(212,175,55,0.18);
    --gold:#D4AF37; --gold-dim:#B8860B; --gold-glow:rgba(212,175,55,0.25);
    --text:#EEF2F7; --muted:#64748B; --green:#10B981; --red:#EF4444; --blue:#3B82F6;
}
*,*::before,*::after{font-family:'Inter',-apple-system,sans-serif!important;box-sizing:border-box;}
[data-testid="stAppViewContainer"]{background:var(--bg)!important;}
[data-testid="stHeader"]{background:rgba(5,8,13,0.95)!important;backdrop-filter:blur(12px);border-bottom:1px solid var(--border)!important;}
[data-testid="stMain"]{padding-top:0!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#060B14 0%,#0B1828 100%)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,[data-testid="stSidebar"] div{color:var(--text)!important;}
[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] small{color:var(--muted)!important;font-size:0.72rem!important;}
[data-testid="metric-container"]{background:linear-gradient(135deg,#0D1828 0%,#121F33 100%)!important;border:1px solid var(--border)!important;border-radius:14px!important;padding:18px 20px!important;box-shadow:0 4px 24px rgba(0,0,0,0.5)!important;transition:transform 0.2s!important;}
[data-testid="metric-container"]:hover{transform:translateY(-2px)!important;}
div[data-testid="stMetricValue"]{color:var(--gold)!important;font-size:1.45rem!important;font-weight:700!important;}
div[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:0.68rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.1em!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;gap:0!important;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-weight:500!important;font-size:0.82rem!important;padding:12px 18px!important;border:none!important;background:transparent!important;border-bottom:2px solid transparent!important;}
.stTabs [aria-selected="true"]{color:var(--gold)!important;border-bottom:2px solid var(--gold)!important;background:rgba(212,175,55,0.05)!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:24px!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--gold-dim),var(--gold),var(--gold-dim))!important;color:#05080D!important;border:none!important;font-weight:700!important;font-size:0.82rem!important;letter-spacing:0.14em!important;text-transform:uppercase!important;padding:0.75rem 1rem!important;border-radius:8px!important;box-shadow:0 0 28px rgba(212,175,55,0.35)!important;transition:all 0.3s!important;}
.stButton>button[kind="primary"]:hover{box-shadow:0 0 42px rgba(212,175,55,0.65)!important;transform:translateY(-1px)!important;}
.stButton>button:not([kind="primary"]){background:rgba(13,18,28,0.8)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:7px!important;font-size:0.78rem!important;transition:all 0.2s!important;}
.stButton>button:not([kind="primary"]):hover{border-color:var(--gold)!important;color:var(--gold)!important;}
.stTextInput>div>div>input,.stNumberInput>div>div>input{background:var(--bg2)!important;border:1px solid rgba(212,175,55,0.2)!important;border-radius:8px!important;color:var(--text)!important;caret-color:var(--gold)!important;}
.stTextInput>div>div>input:focus,.stNumberInput>div>div>input:focus{border-color:var(--gold)!important;box-shadow:0 0 0 3px rgba(212,175,55,0.12)!important;}
.stSelectbox>div>div{background:var(--bg2)!important;border:1px solid rgba(212,175,55,0.2)!important;border-radius:8px!important;color:var(--text)!important;}
.stSlider>div>div>div>div[role="slider"]{background:var(--gold)!important;border:2px solid var(--bg)!important;}
.stDataFrame,[data-testid="stDataFrameResizable"]{border:1px solid var(--border)!important;border-radius:12px!important;overflow:hidden!important;}
[data-testid="stDataFrame"] th{background:linear-gradient(135deg,#0D1828,#121F33)!important;color:var(--muted)!important;font-size:0.7rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.08em!important;border-bottom:1px solid var(--border)!important;padding:10px 14px!important;}
[data-testid="stDataFrame"] td{color:var(--text)!important;font-size:0.82rem!important;padding:9px 14px!important;}
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,var(--gold),transparent)!important;opacity:0.25!important;margin:1.2rem 0!important;}
[data-testid="stDownloadButton"]>button{background:transparent!important;color:var(--gold)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
.stSpinner>div>div{border-top-color:var(--gold)!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:rgba(212,175,55,0.22);border-radius:3px;}
@keyframes pulse-gold{0%,100%{opacity:1;box-shadow:0 0 6px rgba(212,175,55,0.6);}50%{opacity:0.5;}}
@keyframes fade-in{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}
.fade-in{animation:fade-in 0.4s ease forwards;}
[data-testid="stMarkdownContainer"] p{color:var(--text)!important;}
[data-testid="stMarkdownContainer"] code{background:rgba(212,175,55,0.1)!important;color:var(--gold)!important;border:1px solid rgba(212,175,55,0.2)!important;border-radius:4px!important;padding:1px 5px!important;}
</style>
""", unsafe_allow_html=True)

# ─── Theme constants ──────────────────────────────────────────────
BG     = "#05080D"
PANEL  = "#0A1220"
GRID   = "rgba(255,255,255,0.05)"
TEXT   = "#EEF2F7"
MUTED  = "#64748B"
GOLD   = "#D4AF37"
GREEN  = "#10B981"
RED    = "#EF4444"
BLUE   = "#3B82F6"
ORANGE = "#F59E0B"
PURPLE = "#8B5CF6"

PALETTE = [
    "#D4AF37", "#3B82F6", "#10B981", "#EF4444",
    "#8B5CF6", "#F59E0B", "#06B6D4", "#EC4899",
    "#84CC16", "#F97316",
]

N_MC = 8_000
SEED = 42

# ─── Scenario definitions ─────────────────────────────────────────
SCENARIOS = {
    "2008 – Global Financial Crisis": {
        "start": "2008-09-01", "end": "2009-03-31",
        "description": "Lehman Brothers collapse. S&P 500 -57% peak-to-trough.",
        "color": RED,
    },
    "2020 – COVID Crash": {
        "start": "2020-02-19", "end": "2020-03-23",
        "description": "Fastest bear market in history. S&P 500 -34% in 33 days.",
        "color": ORANGE,
    },
    "2022 – Rate Hike Shock": {
        "start": "2022-01-01", "end": "2022-12-31",
        "description": "Fed raised rates 425 bps. Bonds and equities fell together.",
        "color": PURPLE,
    },
    "2000–2002 – Dot-com Bust": {
        "start": "2000-03-10", "end": "2002-10-09",
        "description": "NASDAQ lost 78%. Tech-heavy portfolios devastated.",
        "color": "#EC4899",
    },
    "2011 – European Debt Crisis": {
        "start": "2011-07-01", "end": "2011-10-03",
        "description": "Sovereign debt fears in Greece, Italy, Spain.",
        "color": "#06B6D4",
    },
    "📐 Custom Scenario": {
        "start": None, "end": None,
        "description": "Define your own stress period.",
        "color": GOLD,
    },
}

# ─── Currency config ──────────────────────────────────────────────
CURRENCIES = {
    "EUR 🇪🇺": {"symbol": "€",  "ticker": "EURUSD=X",  "fallback": 1.08},
    "USD 🇺🇸": {"symbol": "$",  "ticker": None,          "fallback": 1.00},
    "GBP 🇬🇧": {"symbol": "£",  "ticker": "GBPUSD=X",  "fallback": 1.27},
    "CHF 🇨🇭": {"symbol": "₣",  "ticker": "CHFUSD=X",  "fallback": 1.12},
    "JPY 🇯🇵": {"symbol": "¥",  "ticker": "JPYUSD=X",  "fallback": 0.0067},
    "CAD 🇨🇦": {"symbol": "C$", "ticker": "CADUSD=X",  "fallback": 0.74},
    "AUD 🇦🇺": {"symbol": "A$", "ticker": "AUDUSD=X",  "fallback": 0.65},
}


# ═══════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════

def base_layout(title="", xtitle="", ytitle="", **kw):
    return dict(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        title=dict(text=title, font=dict(size=14, color=TEXT, family="Inter")),
        xaxis=dict(
            title=xtitle, gridcolor=GRID, zerolinecolor=GRID,
            color=MUTED, title_font=dict(color=MUTED, size=11),
            tickfont=dict(color=MUTED, size=10), linecolor=GRID,
        ),
        yaxis=dict(
            title=ytitle, gridcolor=GRID, zerolinecolor=GRID,
            color=MUTED, title_font=dict(color=MUTED, size=11),
            tickfont=dict(color=MUTED, size=10), linecolor=GRID,
        ),
        legend=dict(
            bgcolor="rgba(10,18,32,0.85)", bordercolor=GRID,
            borderwidth=1, font=dict(size=11, color=TEXT),
        ),
        margin=dict(l=55, r=25, t=56, b=50),
        **kw,
    )


def page_header():
    st.markdown("""
<div class="fade-in" style="display:flex;align-items:center;gap:16px;
    padding:28px 0 22px 0;border-bottom:1px solid rgba(212,175,55,0.15);margin-bottom:28px;">
  <div style="width:46px;height:46px;flex-shrink:0;
      background:linear-gradient(135deg,#B8860B,#D4AF37);border-radius:12px;
      display:flex;align-items:center;justify-content:center;font-size:22px;
      box-shadow:0 0 28px rgba(212,175,55,0.4);">📈</div>
  <div>
    <div style="font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:700;
        line-height:1.15;background:linear-gradient(135deg,#C9A440,#F0D060,#C9A440);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
      Portfolio Optimizer</div>
    <div style="color:#64748B;font-size:0.72rem;font-weight:500;
        letter-spacing:0.14em;text-transform:uppercase;margin-top:3px;">
      Modern Portfolio Theory · Markowitz Efficient Frontier · by AEG
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def section_title(text, sub=""):
    sub_html = (f'<div style="color:#64748B;font-size:0.73rem;margin-top:3px;">{sub}</div>'
                if sub else "")
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:28px 0 16px 0;">
  <div style="width:3px;height:26px;flex-shrink:0;border-radius:2px;
      background:linear-gradient(180deg,#D4AF37,rgba(212,175,55,0));"></div>
  <div>
    <div style="font-size:0.95rem;font-weight:600;color:#EEF2F7;">{text}</div>
    {sub_html}
  </div>
</div>""", unsafe_allow_html=True)


def live_badge():
    return (
        '<span style="display:inline-flex;align-items:center;gap:5px;'
        'background:rgba(212,175,55,0.1);border:1px solid rgba(212,175,55,0.3);'
        'border-radius:20px;padding:2px 10px;font-size:0.68rem;font-weight:600;'
        'color:#D4AF37;letter-spacing:0.08em;text-transform:uppercase;'
        'vertical-align:middle;margin-left:8px;">'
        '<span style="width:6px;height:6px;border-radius:50%;background:#D4AF37;'
        'animation:pulse-gold 1.4s ease-in-out infinite;"></span>LIVE</span>'
    )


def kpi_card(label, value, icon=""):
    return (
        f'<div style="background:linear-gradient(135deg,#0D1828 0%,#121F33 100%);'
        f'border:1px solid rgba(212,175,55,0.18);border-radius:14px;'
        f'padding:18px 20px;height:100%;box-shadow:0 4px 24px rgba(0,0,0,0.5);">'
        f'<div style="color:#64748B;font-size:0.66rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.1em;">{icon}&nbsp;{label}</div>'
        f'<div style="color:#D4AF37;font-size:1.45rem;font-weight:700;'
        f'letter-spacing:-0.02em;margin-top:7px;line-height:1.1;">{value}</div></div>'
    )


def sidebar_section(title):
    st.sidebar.markdown(
        f'<div style="font-size:0.68rem;font-weight:700;color:#64748B;'
        f'text-transform:uppercase;letter-spacing:0.12em;'
        f'margin:18px 0 8px 0;padding-bottom:6px;'
        f'border-bottom:1px solid rgba(212,175,55,0.1);">{title}</div>',
        unsafe_allow_html=True,
    )


def delta_color(val):
    if isinstance(val, str) and val.startswith("+"):
        return f"color:{GREEN}"
    if isinstance(val, str) and val.startswith("-"):
        return f"color:{RED}"
    return f"color:{MUTED}"


# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO MATH
# ═══════════════════════════════════════════════════════════════════

def pmetrics(w, mu, cov, rf):
    r = float(w @ mu)
    v = float(np.sqrt(max(float(w @ cov @ w), 0)))
    s = (r - rf) / v if v > 1e-9 else 0.0
    return r, v, s


def port_cum(prices, w):
    ret = prices.pct_change().dropna()
    return (1 + (ret * w).sum(axis=1)).cumprod() * 100


def port_max_dd(prices, w):
    cum = port_cum(prices, w) / 100
    return float((cum / cum.cummax() - 1).min())


def port_cagr(prices, w):
    cum     = port_cum(prices, w) / 100
    n_years = len(cum) / 252
    return float(cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0.1 else 0.0


def asset_max_dd(s):
    return float((s / s.cummax() - 1).min())


def asset_cagr(s):
    n_years = len(s) / 252
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / n_years) - 1) if n_years > 0.1 else 0.0


def risk_contributions(w, cov):
    port_vol = np.sqrt(w @ cov @ w)
    if port_vol < 1e-9:
        return np.zeros(len(w))
    mcr   = (cov @ w) / port_vol
    rc    = w * mcr
    total = rc.sum()
    return rc / total * 100 if total > 1e-9 else rc * 0


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    try:
        raw = yf.download(list(tickers), start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            lvl = raw.columns.get_level_values(0)
            key = "Close" if "Close" in lvl else lvl[0]
            raw = raw[key]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])
        raw = raw.ffill(limit=5)
        raw = raw.loc[:, raw.notna().sum() >= 20]
        raw = raw.dropna(how="all")
        return raw
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_scenario_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """
    Robust loader for scenario stress windows.
    Key differences from load_prices:
      - auto_adjust=True  (more reliable for dates before 2010)
      - ffill + bfill     (handles missing rows mid-series)
      - dropna(how='all') (never drops a row just because one ticker is missing)
      - keeps any column with >= 5 valid rows
    """
    try:
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return pd.DataFrame()

        # extract the Close price level
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0)
            key  = "Close" if "Close" in lvl0 else lvl0[0]
            raw  = raw[key]

        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])

        # fill short gaps (weekends, holidays, data lags) then backfill series start
        raw = raw.ffill(limit=5).bfill(limit=2)

        # drop columns with fewer than 5 valid data points
        raw = raw.loc[:, raw.notna().sum() >= 5]

        # drop rows where every column is NaN (i.e. non-trading days outside all markets)
        raw = raw.dropna(how="all")

        return raw

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def get_live_prices(tickers: tuple) -> dict:
    try:
        raw = yf.download(list(tickers), period="5d", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return {}
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].ffill().iloc[-1]
        else:
            close = raw["Close"].ffill()
            if isinstance(close, pd.Series):
                return {tickers[0]: float(close.iloc[-1])}
        return {
            str(t): float(close[t])
            for t in tickers
            if t in close.index and not pd.isna(close[t])
        }
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_fx_rates() -> dict:
    rates = {}
    for name, cfg in CURRENCIES.items():
        if cfg["ticker"] is None:
            rates[name] = cfg["fallback"]
            continue
        try:
            price = yf.Ticker(cfg["ticker"]).fast_info.last_price
            rates[name] = float(price) if price and price > 0 else cfg["fallback"]
        except Exception:
            rates[name] = cfg["fallback"]
    return rates


# ═══════════════════════════════════════════════════════════════════
# TICKER VALIDATION
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def validate_ticker(ticker: str) -> dict:
    try:
        t     = yf.Ticker(ticker)
        price = getattr(t.fast_info, "last_price", None)
        if price is None or price == 0:
            return {"valid": False, "name": ticker, "error": "No price data found"}
        try:
            info     = t.info
            name     = info.get("longName") or info.get("shortName") or ticker
            exchange = info.get("exchange", "")
            currency = info.get("currency", "USD")
            atype    = info.get("quoteType", "EQUITY")
        except Exception:
            name, exchange, currency, atype = ticker, "", "USD", "EQUITY"
        return {
            "valid": True, "name": name, "exchange": exchange,
            "currency": currency, "asset_type": atype, "price": price,
        }
    except Exception as e:
        return {"valid": False, "name": ticker, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# ASSET DATABASE
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_asset_db() -> list:
    try:
        import financedatabase as fd
    except ImportError:
        return []
    ALLOWED = {"NMS","NAS","NYQ","NYS","PCX","ARCX","PAR","LSE","SHH","JPX","FRA","XETRA"}
    loaders = {
        "Equity":   lambda: fd.Equities().select(),
        "ETF":      lambda: fd.ETFs().select(),
        "Crypto":   lambda: fd.Cryptos().select(),
        "Index":    lambda: fd.Indices().select(),
        "Fund":     lambda: fd.Funds().select(),
        "Currency": lambda: fd.Currencies().select(),
    }
    out = []
    for atype, loader in loaders.items():
        try:
            df = loader()
            if df is None or df.empty:
                continue
            if "exchange" in df.columns:
                df = df[df["exchange"].isin(ALLOWED) | df["exchange"].isna()]
            if atype == "Equity" and "market_cap" in df.columns:
                df = df[df["market_cap"].isin(["Large Cap", "Mega Cap", "Mid Cap"])]
            for sym, row in df.iterrows():
                if not sym or str(sym).strip() in ("", "nan"):
                    continue
                name  = str(row.get("name", "") or "").strip()
                exch  = str(row.get("exchange", "") or "").strip()
                label = f"{name}  ({sym})  —  {atype}" + (f"  {exch}" if exch else "")
                out.append({"label": label, "symbol": str(sym).upper()})
        except Exception:
            continue
    return sorted(out, key=lambda x: x["label"].lower())


def search_assets(db, query, limit=120):
    if not query:
        return db[:60]
    q = query.lower()
    return [item for item in db if q in item["label"].lower()][:limit]


# ═══════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def optimize(mu_t: tuple, cov_t: tuple, n: int, rf: float,
             min_w_t: tuple = None, max_w_t: tuple = None):
    mu  = np.array(mu_t)
    cov = np.array(cov_t).reshape(n, n)

    min_w = np.array(min_w_t) if min_w_t else np.zeros(n)
    max_w = np.array(max_w_t) if max_w_t else np.ones(n)
    bds   = [(float(min_w[i]), float(max_w[i])) for i in range(n)]
    eq    = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    w0    = np.ones(n) / n
    rng   = np.random.default_rng(SEED)

    # Min Variance
    sol   = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bds, constraints=eq)
    w_mvp = sol.x if sol.success else w0.copy()

    # Tangent – multi-start
    best, w_tan = -np.inf, w0.copy()
    for _ in range(50):
        r = minimize(
            lambda w: -(float(w @ mu) - rf) / max(np.sqrt(float(w @ cov @ w)), 1e-9),
            rng.dirichlet(np.ones(n)),
            method="SLSQP", bounds=bds, constraints=eq,
        )
        if r.success and -r.fun > best:
            best, w_tan = -r.fun, r.x

    # Risk Parity
    def rp_obj(w):
        pv  = np.sqrt(w @ cov @ w)
        mcr = (cov @ w) / pv
        rc  = w * mcr
        return np.sum((rc - rc.mean()) ** 2)

    rp_bds   = [(max(float(min_w[i]), 1e-4), float(max_w[i])) for i in range(n)]
    best_f, w_rp = np.inf, w0.copy()
    for _ in range(20):
        r = minimize(
            rp_obj, rng.dirichlet(np.ones(n)), method="SLSQP",
            bounds=rp_bds, constraints=eq,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if r.success and r.fun < best_f:
            best_f, w_rp = r.fun, r.x
    w_rp = w_rp / w_rp.sum()

    # Efficient frontier
    lo, hi = float(w_mvp @ mu), float(mu.max()) * 0.98
    fv, fr  = [], []
    for t in np.linspace(lo, hi, 100):
        r = minimize(
            lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bds,
            constraints=eq + [{"type": "eq",
                                "fun": lambda w, t=t: float(w @ mu) - t}],
        )
        if r.success:
            fv.append(float(np.sqrt(r.x @ cov @ r.x)))
            fr.append(t)

    # Monte Carlo
    rng2 = np.random.default_rng(SEED)
    mc_r, mc_v, mc_s = [], [], []
    for _ in range(N_MC):
        x       = rng2.exponential(1, n)
        ww      = x / x.sum()
        r2,v2,s2 = pmetrics(ww, mu, cov, rf)
        mc_r.append(r2); mc_v.append(v2); mc_s.append(s2)

    return (
        w_tan, w_mvp, w_rp,
        np.array(fv), np.array(fr),
        np.array(mc_r), np.array(mc_v), np.array(mc_s),
    )


# ═══════════════════════════════════════════════════════════════════
# SCENARIO ENGINE
# ═══════════════════════════════════════════════════════════════════

def run_scenario(prices_sc: pd.DataFrame, w_tan, w_mvp, w_rp,
                 assets: list, name: str, meta: dict) -> dict:
    """
    Compute scenario stats.  Handles partial data gracefully:
      - uses only assets present AND with >= 5 rows in this window
      - fills remaining per-column gaps
      - re-normalises weights to sum to 1 for the available subset
    """
    if prices_sc is None or prices_sc.empty:
        return {}

    available = [
        a for a in assets
        if a in prices_sc.columns and prices_sc[a].notna().sum() >= 5
    ]
    if not available:
        return {}

    p = prices_sc[available].copy()
    p = p.ffill().bfill()
    p = p.dropna(how="all")

    if len(p) < 3:
        return {}

    # index of available assets in the full asset list
    idx = [assets.index(a) for a in available]

    results = {}
    port_defs = {
        "Tangent":      np.array(w_tan, dtype=float)[idx],
        "Min Variance": np.array(w_mvp, dtype=float)[idx],
        "Risk Parity":  np.array(w_rp,  dtype=float)[idx],
    }

    for pname, w in port_defs.items():
        s = w.sum()
        if s < 1e-9:
            continue
        w = w / s  # re-normalise

        cum       = port_cum(p, w) / 100
        mdd       = float((cum / cum.cummax() - 1).min())
        tot_ret   = float((p / p.iloc[0]).iloc[-1].values @ w) - 1
        daily_ret = p.pct_change().dropna()
        vol       = float((daily_ret * w).sum(axis=1).std() * np.sqrt(252))

        results[pname] = {
            "total_return": tot_ret,
            "max_drawdown": mdd,
            "volatility":   vol,
            "cum_series":   port_cum(p, w),
        }

    asset_stats = {}
    for a in available:
        s = p[a].dropna()
        if len(s) < 2:
            continue
        asset_stats[a] = {
            "total_return": float(s.iloc[-1] / s.iloc[0] - 1),
            "max_drawdown": asset_max_dd(s),
        }

    return {
        "portfolios": results,
        "assets":     asset_stats,
        "prices":     p,
        "available":  available,
        "missing":    [a for a in assets if a not in available],
        "name":       name,
        "meta":       meta,
    }


# ═══════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════

def chart_frontier(res) -> go.Figure:
    mu, cov, rf           = res["mu"], res["cov"], res["rf"]
    assets                = res["assets"]
    w_tan, w_mvp, w_rp    = res["w_tan"], res["w_mvp"], res["w_rp"]
    vols                  = np.sqrt(np.diag(cov))
    tan_r, tan_v, tan_sh  = pmetrics(w_tan, mu, cov, rf)
    mvp_r, mvp_v, _       = pmetrics(w_mvp, mu, cov, rf)
    rp_r,  rp_v,  rp_sh   = pmetrics(w_rp,  mu, cov, rf)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=res["mc_vols"]*100, y=res["mc_rets"]*100, mode="markers",
        marker=dict(size=3, color=res["mc_sh"], colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Sharpe", x=0.975, thickness=12,
                                  tickfont=dict(color=MUTED, size=9),
                                  title_font=dict(color=MUTED, size=10))),
        opacity=0.35, name="Simulated portfolios",
        hovertemplate="Vol %{x:.2f}%  ·  Ret %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=res["front_vols"]*100, y=res["front_rets"]*100,
        mode="lines", line=dict(color=GOLD, width=2.5), name="Efficient Frontier",
    ))
    xr = np.linspace(0, res["mc_vols"].max()*100*1.25, 120)
    fig.add_trace(go.Scatter(
        x=xr, y=rf*100 + tan_sh*xr, mode="lines",
        line=dict(color="rgba(212,175,55,0.35)", width=1.5, dash="dash"),
        name="Capital Market Line",
    ))
    for x, y, sym, col, sz, nm in [
        (tan_v*100, tan_r*100, "star",    "#FF6B6B", 24, f"Tangent  Sharpe={tan_sh:.2f}"),
        (mvp_v*100, mvp_r*100, "diamond", BLUE,      18, "Min Variance"),
        (rp_v*100,  rp_r*100,  "pentagon",PURPLE,    18, f"Risk Parity  Sharpe={rp_sh:.2f}"),
        (0,         rf*100,    "circle",  ORANGE,    13, f"Risk-free {rf*100:.1f}%"),
    ]:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(size=sz, symbol=sym, color=col, line=dict(color=BG, width=2)),
            name=nm,
        ))
    for i, a in enumerate(assets):
        ar, av = float(mu[i]), float(vols[i])
        fig.add_trace(go.Scatter(
            x=[av*100], y=[ar*100], mode="markers+text",
            marker=dict(size=10, symbol="circle-open",
                        color=PALETTE[i % len(PALETTE)], line=dict(width=2.2)),
            text=[a], textposition="top right",
            textfont=dict(color=TEXT, size=10), name=a,
            hovertemplate=(f"<b>{a}</b><br>Vol {av*100:.2f}%<br>"
                           f"Ret {ar*100:.2f}%<br>Sharpe {(ar-rf)/av:.3f}<extra></extra>"),
        ))
    fig.update_layout(**base_layout(
        "Markowitz Efficient Frontier",
        "Annual Volatility (%)", "Annual Return (%)", height=560,
    ))
    return fig


def chart_prices(prices) -> go.Figure:
    norm = prices / prices.iloc[0] * 100
    fig  = go.Figure()
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col], mode="lines", name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            hovertemplate=f"{col}: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**base_layout("Normalized Prices (base 100)", "Date", "Price Index"))
    return fig


def chart_drawdown(prices) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(prices.columns):
        dd = (prices[col] - prices[col].cummax()) / prices[col].cummax() * 100
        fig.add_trace(go.Scatter(
            x=prices.index, y=dd, mode="lines", name=col,
            fill="tozeroy", line=dict(color=PALETTE[i % len(PALETTE)], width=1.4),
            opacity=0.6, hovertemplate=f"{col}: %{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(**base_layout("Drawdown (%)", "Date", "Drawdown (%)"))
    return fig


def chart_pie(w, assets, title, ret, vol, height=520) -> go.Figure:
    mask   = w > 0.01
    labels = [assets[i] for i in range(len(assets)) if mask[i]]
    vals   = w[mask]
    if not len(vals):
        fig = go.Figure()
        fig.add_annotation(text="No significant weights", xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(color=TEXT, size=14), showarrow=False)
        fig.update_layout(**base_layout(title, height=height))
        return fig
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.44,
        marker=dict(colors=PALETTE[:len(labels)], line=dict(color=BG, width=2.5)),
        textinfo="label+percent", textfont=dict(size=13),
        insidetextorientation="radial",
        hovertemplate="%{label}: %{percent} · %{value:.3f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        f"{title}  ·  Ret {ret*100:+.1f}%  ·  Vol {vol*100:.1f}%", height=height,
    ))
    return fig


def chart_cumulative(prices, w_tan, w_mvp, w_rp) -> go.Figure:
    fig = go.Figure()
    for w, nm, color in [
        (w_tan, "Tangent Portfolio", GOLD),
        (w_mvp, "Min Variance",      BLUE),
        (w_rp,  "Risk Parity",       PURPLE),
    ]:
        cum = port_cum(prices, w)
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines",
            name=nm, line=dict(color=color, width=2.2),
            hovertemplate=f"{nm}: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**base_layout(
        "Cumulative Portfolio Performance (base 100)", "Date", "Value",
    ))
    return fig


def chart_sharpe_bars(mu, vols, rf, assets, tan_sh) -> go.Figure:
    shs = [(mu[i] - rf) / vols[i] for i in range(len(assets))]
    fig = go.Figure(go.Bar(
        x=shs, y=assets, orientation="h",
        marker=dict(color=[GREEN if s >= 0 else RED for s in shs],
                    line=dict(color=BG, width=1), opacity=0.85),
        text=[f"{s:.2f}" for s in shs], textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.add_vline(x=tan_sh,
                  line=dict(color=GOLD, width=1.8, dash="dash"),
                  annotation_text=f"Tangent ({tan_sh:.2f})",
                  annotation_font_color=GOLD, annotation_font_size=11)
    fig.update_layout(**base_layout(
        "Individual Sharpe Ratios", "Sharpe Ratio", "",
        height=max(300, len(assets) * 46 + 120),
    ))
    return fig


def chart_correlation(returns) -> go.Figure:
    corr = returns.corr()
    z    = np.round(corr.values, 2)
    fig  = go.Figure(go.Heatmap(
        z=z, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0.0, "#EF4444"], [0.5, "#1A2540"], [1.0, "#D4AF37"]],
        zmid=0, zmin=-1, zmax=1, text=z, texttemplate="%{text:.2f}",
        textfont=dict(size=11),
        colorbar=dict(title="ρ", tickfont=dict(color=MUTED),
                      title_font=dict(color=MUTED)),
        hovertemplate="<b>%{x} × %{y}</b><br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        "Return Correlation Matrix",
        height=max(380, len(returns.columns) * 65 + 80),
    ))
    return fig


def chart_rolling_sharpe(returns, rf, window) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(returns.columns):
        rs = ((returns[col].rolling(window).mean() * 252 - rf) /
              (returns[col].rolling(window).std() * np.sqrt(252)))
        fig.add_trace(go.Scatter(
            x=returns.index, y=rs, mode="lines", name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.6),
            hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_layout(**base_layout(f"Rolling {window}d Sharpe Ratio", "Date", "Sharpe"))
    return fig


def chart_risk_contrib(assets, w_rp, w_tan, w_mvp, cov) -> go.Figure:
    fig = go.Figure()
    for w, nm, color in [
        (w_rp,  "Risk Parity",  PURPLE),
        (w_tan, "Tangent",      GOLD),
        (w_mvp, "Min Variance", BLUE),
    ]:
        rc = risk_contributions(w, cov)
        fig.add_trace(go.Bar(
            name=nm, x=assets, y=rc,
            marker=dict(color=color, opacity=0.82, line=dict(color=BG, width=1)),
        ))
    fig.update_layout(**base_layout(
        "Risk Contribution per Asset (%)", "Asset", "Risk Contribution (%)",
        barmode="group", height=380,
    ))
    return fig


def chart_scenario_cum(sc_result) -> go.Figure:
    portfolios = sc_result.get("portfolios", {})
    prices_p   = sc_result.get("prices", pd.DataFrame())
    if prices_p.empty:
        return go.Figure()
    port_colors = {"Tangent": GOLD, "Min Variance": BLUE, "Risk Parity": PURPLE}
    fig = go.Figure()
    for nm, stats in portfolios.items():
        cum = stats["cum_series"]
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines",
            name=nm, line=dict(color=port_colors.get(nm, GOLD), width=2.5),
            hovertemplate=f"{nm}: %{{y:.1f}}<extra></extra>",
        ))
    norm = prices_p / prices_p.iloc[0] * 100
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col], mode="lines", name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1, dash="dot"),
            opacity=0.4, hovertemplate=f"{col}: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**base_layout(
        f"Scenario: {sc_result['name']}  (base 100)", "Date", "Value", height=440,
    ))
    return fig


def chart_scenario_bars(sc_result) -> go.Figure:
    portfolios = sc_result.get("portfolios", {})
    if not portfolios:
        return go.Figure()
    names = list(portfolios.keys())
    rets  = [portfolios[n]["total_return"] * 100 for n in names]
    mdds  = [portfolios[n]["max_drawdown"]  * 100 for n in names]
    fig   = go.Figure()
    fig.add_trace(go.Bar(
        name="Total Return (%)", x=names, y=rets,
        marker=dict(color=[GREEN if r >= 0 else RED for r in rets],
                    opacity=0.85, line=dict(color=BG, width=1)),
        text=[f"{r:+.1f}%" for r in rets], textposition="outside",
        textfont=dict(color=TEXT, size=11),
    ))
    fig.add_trace(go.Bar(
        name="Max Drawdown (%)", x=names, y=mdds,
        marker=dict(color=RED, opacity=0.6, line=dict(color=BG, width=1)),
        text=[f"{d:.1f}%" for d in mdds], textposition="outside",
        textfont=dict(color=TEXT, size=11),
    ))
    fig.update_layout(**base_layout(
        "Performance During Scenario", "", "%",
        barmode="group", height=380,
    ))
    return fig


def chart_budget_bars(assets, weights, budget, label, sym) -> go.Figure:
    amounts = [budget * w for w in weights]
    mask    = np.array(weights) > 0.005
    a_show  = [assets[i] for i in range(len(assets)) if mask[i]]
    v_show  = [amounts[i] for i in range(len(assets)) if mask[i]]
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(assets)) if mask[i]]
    fig = go.Figure(go.Bar(
        x=v_show, y=a_show, orientation="h",
        marker=dict(color=colors, line=dict(color=BG, width=1), opacity=0.88),
        text=[f"{sym}{v:,.0f}" for v in v_show], textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate=f"%{{y}}  ·  {sym}%{{x:,.0f}}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        f"Budget Allocation — {label}  (Total {sym}{budget:,})",
        f"Amount ({sym})", "",
        height=max(300, len(a_show) * 46 + 120),
    ))
    return fig


def chart_live_vs_backtest(assets, hist_prices, live_prices) -> go.Figure:
    pct_changes, colors, labels = [], [], []
    for a in assets:
        hist = hist_prices.get(a)
        live = live_prices.get(a)
        if hist and live and hist > 0:
            chg = (live / hist - 1) * 100
            pct_changes.append(chg)
            colors.append(GREEN if chg >= 0 else RED)
            labels.append(f"{chg:+.1f}%")
        else:
            pct_changes.append(0)
            colors.append(MUTED)
            labels.append("N/A")
    fig = go.Figure(go.Bar(
        x=pct_changes, y=assets, orientation="h",
        marker=dict(color=colors, line=dict(color=BG, width=1), opacity=0.85),
        text=labels, textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(**base_layout(
        "Performance Since Backtest End",
        "Change (%)", "",
        height=max(300, len(assets) * 46 + 120),
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════
# ALLOCATION TABLE
# ═══════════════════════════════════════════════════════════════════

def build_alloc_table(assets, weights, budget, live_prices,
                      hist_end, fx_rate, sym) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(assets):
        w         = float(weights[i])
        alloc_loc = budget * w
        alloc_usd = alloc_loc * fx_rate
        live      = live_prices.get(a)
        hist      = hist_end.get(a)
        shares    = alloc_usd / live if live and live > 0 else None
        chg_pct   = (live / hist - 1) * 100 if live and hist and hist > 0 else None
        live_val  = (shares * live / fx_rate) if shares and live else None
        rows.append({
            "Asset":              a,
            "Weight":             f"{w*100:.1f}%",
            f"Allocated ({sym})": f"{sym}{alloc_loc:,.0f}",
            "Allocated ($)":      f"${alloc_usd:,.0f}",
            "Live Price":         f"${live:,.2f}" if live else "—",
            "Shares":             f"{shares:.4f}" if shares else "—",
            f"Value ({sym})":     f"{sym}{live_val:,.0f}" if live_val else "—",
            "Δ Backtest End":     (f"+{chg_pct:.1f}%" if chg_pct and chg_pct >= 0
                                   else f"{chg_pct:.1f}%" if chg_pct else "—"),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.markdown("""
<div style="padding:20px 0 14px 0;border-bottom:1px solid rgba(212,175,55,0.14);margin-bottom:4px;">
  <div style="font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;color:#D4AF37;">
    📈 Portfolio Optimizer</div>
  <div style="color:#64748B;font-size:0.66rem;font-weight:600;text-transform:uppercase;
      letter-spacing:0.14em;margin-top:5px;">Markowitz · by AEG</div>
</div>""", unsafe_allow_html=True)

    sidebar_section("Parameters")
    c1, c2 = st.sidebar.columns(2)
    start  = c1.date_input("Start", date(2019, 1, 1))
    end    = c2.date_input("End",   date(2026, 1, 1))
    rf_pct = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.25, format="%.2f%%")

    sidebar_section("Currency")
    currency_key = st.sidebar.selectbox(
        "Base currency", list(CURRENCIES.keys()),
        index=0, label_visibility="collapsed",
        format_func=lambda x: f"💱  {x}",
    )

    sidebar_section("Budget")
    budget = st.sidebar.number_input(
        "Amount", min_value=100, max_value=10_000_000,
        value=4000, step=100, format="%d", label_visibility="collapsed",
    )
    sym = CURRENCIES[currency_key]["symbol"]
    st.sidebar.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(212,175,55,0.08),rgba(212,175,55,0.04));'
        f'border:1px solid rgba(212,175,55,0.18);border-radius:8px;'
        f'padding:8px 12px;margin-top:4px;display:flex;align-items:center;gap:8px;">'
        f'<span>💰</span>'
        f'<span style="color:#D4AF37;font-weight:700;font-size:0.95rem;">{sym} {budget:,}</span>'
        f'<span style="color:#64748B;font-size:0.72rem;">total budget</span></div>',
        unsafe_allow_html=True,
    )

    sidebar_section("Asset Database")
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = ["AAPL", "MSFT"]

    with st.sidebar:
        with st.spinner("Loading…"):
            db = load_asset_db()

    if db:
        st.sidebar.caption(f"{len(db):,} assets available")
        ASSET_TYPES = ["All", "Equity", "ETF", "Crypto", "Index", "Fund", "Currency"]
        asset_type  = st.sidebar.selectbox(
            "Asset type filter", ASSET_TYPES, index=0,
            key="asset_type_filter", label_visibility="collapsed",
            format_func=lambda x: f"🔎 Filter: {x}",
        )
        filtered_db = (
            [item for item in db if f"—  {asset_type}" in item["label"]]
            if asset_type != "All" else db
        )
        st.sidebar.caption(f"{len(filtered_db):,} matching")
        query   = st.sidebar.text_input("🔍", placeholder="Name or ticker…",
                                         key="search_query", label_visibility="collapsed")
        results = search_assets(filtered_db, query)
        if results:
            options = [r["label"] for r in results]
            chosen  = st.sidebar.selectbox("select_result", options,
                                            label_visibility="collapsed", key="search_select")
            if st.sidebar.button("＋  Add to portfolio", use_container_width=True):
                symbol = next((r["symbol"] for r in results if r["label"] == chosen), None)
                if symbol:
                    if symbol not in st.session_state.selected_tickers:
                        st.session_state.selected_tickers.append(symbol)
                        st.rerun()
                    else:
                        st.sidebar.warning(f"{symbol} already selected.")
        else:
            st.sidebar.caption("No results.")
    else:
        st.sidebar.caption("Manual entry.")
        col_in, col_btn = st.sidebar.columns([3, 1])
        manual = col_in.text_input("Ticker", placeholder="AAPL",
                                    label_visibility="collapsed", key="manual_ticker")
        if col_btn.button("＋", key="manual_add"):
            t = manual.strip().upper()
            if t and t not in st.session_state.selected_tickers:
                st.session_state.selected_tickers.append(t)
                st.rerun()

    sidebar_section(f"Portfolio  ·  {len(st.session_state.selected_tickers)} asset(s)")
    for i, ticker in enumerate(list(st.session_state.selected_tickers)):
        c1, c2 = st.sidebar.columns([5, 1])
        c1.markdown(
            f'<span style="color:#64748B;font-size:0.72rem;">#{i+1}</span>'
            f'&nbsp;&nbsp;<code style="color:#D4AF37;background:rgba(212,175,55,0.1);'
            f'border:1px solid rgba(212,175,55,0.2);border-radius:4px;'
            f'padding:2px 6px;font-size:0.82rem;">{ticker}</code>',
            unsafe_allow_html=True,
        )
        if c2.button("✕", key=f"rm_{i}", help=f"Remove {ticker}"):
            if len(st.session_state.selected_tickers) > 2:
                st.session_state.selected_tickers.pop(i)
                st.rerun()
            else:
                st.sidebar.warning("Minimum 2 assets required.")

    sidebar_section("Weight Constraints")
    use_constraints = st.sidebar.toggle("Enable custom constraints", value=False)
    constraints = {}
    if use_constraints:
        st.sidebar.caption("Set min / max weight % per asset")
        for ticker in st.session_state.selected_tickers:
            with st.sidebar.expander(f"⚖️ {ticker}", expanded=False):
                mn = st.slider(f"Min % ({ticker})", 0, 50, 0, 1, key=f"min_{ticker}")
                mx = st.slider(f"Max % ({ticker})", 1, 100, 100, 1, key=f"max_{ticker}")
                if mn >= mx:
                    st.warning("Min must be < Max")
                constraints[ticker] = (mn / 100.0, mx / 100.0)

    st.sidebar.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    run = st.sidebar.button("▶  RUN ANALYSIS", type="primary", use_container_width=True)

    return (
        start, end, rf_pct / 100.0,
        list(st.session_state.selected_tickers),
        budget, currency_key,
        constraints if use_constraints else {},
        run,
    )


# ═══════════════════════════════════════════════════════════════════
# VALIDATION PANEL
# ═══════════════════════════════════════════════════════════════════

def render_validation_panel(tickers, btn_key="revalidate"):
    section_title("Ticker Validation", "Live verification against Yahoo Finance")
    if st.button("🔄 Re-validate all", key=btn_key):
        validate_ticker.clear()
        st.rerun()
    rows      = []
    all_valid = True
    for t in tickers:
        r = validate_ticker(t)
        if r["valid"]:
            rows.append({
                "Ticker":     t,
                "Status":     "✅ Valid",
                "Name":       r.get("name", "—"),
                "Exchange":   r.get("exchange", "—"),
                "Currency":   r.get("currency", "—"),
                "Type":       r.get("asset_type", "—"),
                "Live Price": f"${r.get('price', 0):,.2f}",
            })
        else:
            all_valid = False
            rows.append({
                "Ticker":     t,
                "Status":     "❌ Invalid",
                "Name":       r.get("name", t),
                "Exchange":   "—", "Currency": "—", "Type": "—",
                "Live Price": r.get("error", "Not found"),
            })

    def color_status(val):
        if "✅" in str(val): return f"color:{GREEN};font-weight:600"
        if "❌" in str(val): return f"color:{RED};font-weight:600"
        return ""

    st.dataframe(
        pd.DataFrame(rows).style.map(color_status, subset=["Status"]),
        use_container_width=True, hide_index=True,
    )
    if not all_valid:
        invalid = [r["Ticker"] for r in rows if "❌" in r["Status"]]
        st.warning(f"⚠️ Invalid tickers will be skipped: **{', '.join(invalid)}**")
    return {r["Ticker"] for r in rows if "✅" in r["Status"]}


# ═══════════════════════════════════════════════════════════════════
# SCENARIO TAB
# ═══════════════════════════════════════════════════════════════════

def render_scenario_tab(res):
    assets             = res["assets"]
    w_tan, w_mvp, w_rp = res["w_tan"], res["w_mvp"], res["w_rp"]

    section_title(
        "Stress Test Scenarios",
        "Portfolio behaviour during historical crises and custom periods",
    )

    # ── Scenario selector ─────────────────────────────────────────
    selected     = []
    preset_names = [k for k in SCENARIOS if k != "📐 Custom Scenario"]
    cols         = st.columns(3)

    for i, sc_name in enumerate(preset_names):
        meta = SCENARIOS[sc_name]
        with cols[i % 3]:
            if st.checkbox(sc_name, value=(i == 0), key=f"sc_{i}"):
                selected.append(sc_name)
            st.markdown(
                f'<div style="font-size:0.68rem;color:#64748B;margin-top:2px;'
                f'margin-bottom:10px;padding:6px 8px;background:rgba(255,255,255,0.02);'
                f'border-left:2px solid {meta["color"]}55;border-radius:0 4px 4px 0;">'
                f'{meta["description"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    if st.checkbox("📐 Add custom scenario", key="sc_custom"):
        cc1, cc2, cc3 = st.columns(3)
        c_start = cc1.date_input("Custom start", date(2015, 1, 1),  key="c_start")
        c_end   = cc2.date_input("Custom end",   date(2015, 12, 31), key="c_end")
        c_name  = cc3.text_input("Label", value="My Scenario",      key="c_name")
        SCENARIOS["📐 Custom Scenario"]["start"]       = str(c_start)
        SCENARIOS["📐 Custom Scenario"]["end"]         = str(c_end)
        SCENARIOS["📐 Custom Scenario"]["description"] = c_name
        selected.append("📐 Custom Scenario")

    if not selected:
        st.info("Select at least one scenario above to run the stress test.")
        return

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Run each scenario ─────────────────────────────────────────
    for sc_name in selected:
        meta = SCENARIOS[sc_name]
        if not meta["start"] or not meta["end"]:
            st.warning(f"**{sc_name}**: date range not set.")
            continue

        with st.spinner(f"Downloading data for {sc_name}…"):
            sc_prices = load_scenario_prices(
                tuple(assets), meta["start"], meta["end"],
            )

        sc_result = run_scenario(
            sc_prices, w_tan, w_mvp, w_rp, assets, sc_name, meta,
        )

        # ── No usable data ────────────────────────────────────────
        if not sc_result or not sc_result.get("portfolios"):
            st.warning(
                f"**{sc_name}** — no usable data found for any of your assets "
                f"between {meta['start']} and {meta['end']}.\n\n"
                f"**Likely reasons:**\n"
                f"- Your assets did not exist yet (e.g. crypto before 2017, recent IPOs)\n"
                f"- Yahoo Finance returned an empty response for this window\n\n"
                f"**Tip:** Add a long-standing asset like **SPY** or **AAPL** "
                f"to your portfolio to confirm that data is available for this period."
            )
            continue

        # ── Partial data notice ───────────────────────────────────
        missing = sc_result.get("missing", [])
        avail   = sc_result.get("available", [])
        if missing:
            st.info(
                f"ℹ️  **{sc_name}**: no historical data for "
                f"**{', '.join(missing)}** during this window. "
                f"Results computed on **{', '.join(avail)}** only "
                f"(weights re-normalised to 100%)."
            )

        portfolios = sc_result["portfolios"]
        n_days     = len(sc_result["prices"])

        # ── Scenario header ───────────────────────────────────────
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;'
            f'background:linear-gradient(135deg,#0D1828,#121F33);'
            f'border:1px solid {meta["color"]}33;border-radius:12px;'
            f'padding:16px 20px;margin:16px 0 12px 0;">'
            f'<div style="width:4px;height:40px;border-radius:2px;'
            f'background:{meta["color"]};flex-shrink:0;"></div>'
            f'<div>'
            f'<div style="font-size:1rem;font-weight:600;color:#EEF2F7;">{sc_name}</div>'
            f'<div style="font-size:0.72rem;color:#64748B;margin-top:2px;">'
            f'{meta["start"]} → {meta["end"]}  ·  {n_days} trading days  ·  '
            f'{meta["description"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # ── KPI strip ─────────────────────────────────────────────
        port_colors = {"Tangent": GOLD, "Min Variance": BLUE, "Risk Parity": PURPLE}
        kpi_cols    = st.columns(len(portfolios) * 2)
        idx = 0
        for pname, stats in portfolios.items():
            col   = port_colors.get(pname, GOLD)
            ret_s = f"{stats['total_return']*100:+.1f}%"
            mdd_s = f"{stats['max_drawdown']*100:.1f}%"
            ret_c = GREEN if stats["total_return"] >= 0 else RED

            kpi_cols[idx].markdown(
                f'<div style="background:linear-gradient(135deg,#0D1828,#121F33);'
                f'border:1px solid {col}33;border-radius:10px;'
                f'padding:14px;text-align:center;">'
                f'<div style="color:#64748B;font-size:0.62rem;text-transform:uppercase;'
                f'letter-spacing:0.1em;">{pname}</div>'
                f'<div style="color:{ret_c};font-size:1.2rem;font-weight:700;'
                f'margin-top:4px;">{ret_s}</div>'
                f'<div style="color:#64748B;font-size:0.68rem;">return</div></div>',
                unsafe_allow_html=True,
            )
            kpi_cols[idx + 1].markdown(
                f'<div style="background:linear-gradient(135deg,#0D1828,#121F33);'
                f'border:1px solid {col}33;border-radius:10px;'
                f'padding:14px;text-align:center;">'
                f'<div style="color:#64748B;font-size:0.62rem;text-transform:uppercase;'
                f'letter-spacing:0.1em;">{pname} MDD</div>'
                f'<div style="color:{RED};font-size:1.2rem;font-weight:700;'
                f'margin-top:4px;">{mdd_s}</div>'
                f'<div style="color:#64748B;font-size:0.68rem;">max drawdown</div></div>',
                unsafe_allow_html=True,
            )
            idx += 2

        # ── Charts ────────────────────────────────────────────────
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(chart_scenario_cum(sc_result),  use_container_width=True)
        with c2:
            st.plotly_chart(chart_scenario_bars(sc_result), use_container_width=True)

        # ── Per-asset table ───────────────────────────────────────
        asset_stats = sc_result.get("assets", {})
        if asset_stats:
            with st.expander("Individual asset performance", expanded=False):
                rows = [
                    {
                        "Asset":        a,
                        "Total Return": f"{s['total_return']*100:+.1f}%",
                        "Max Drawdown": f"{s['max_drawdown']*100:.1f}%",
                    }
                    for a, s in asset_stats.items()
                ]
                adf = pd.DataFrame(rows)
                st.dataframe(
                    adf.style.map(delta_color, subset=["Total Return", "Max Drawdown"]),
                    use_container_width=True, hide_index=True,
                )

        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    page_header()

    start, end, rf, tickers, budget, currency_key, constraints, run = render_sidebar()

    if run:
        if len(tickers) < 2:
            st.error("Please select at least 2 assets.")
            st.stop()
        if start >= end:
            st.error("Start date must be earlier than end date.")
            st.stop()

        # Ticker validation
        with st.spinner("Validating tickers…"):
            valid_set = render_validation_panel(tickers, btn_key="revalidate_run")

        tickers_ok = [t for t in tickers if t in valid_set]
        if len(tickers_ok) < 2:
            st.error("Fewer than 2 valid tickers. Please fix your selection.")
            st.stop()

        with st.spinner("Downloading historical data…"):
            prices = load_prices(tuple(tickers_ok), str(start), str(end))

        if prices.empty:
            st.error("Download failed — please check the tickers and date range.")
            st.stop()

        available = [t for t in tickers_ok if t in prices.columns]
        prices    = prices[available]
        prices    = prices.loc[:, prices.notna().sum() >= 20]
        available = list(prices.columns)

        if len(available) < 2:
            st.error("Fewer than 2 assets with sufficient historical data.")
            st.stop()

        prices  = prices.ffill().bfill()
        log_ret = np.log(prices / prices.shift(1)).dropna()
        mu      = log_ret.mean().values * 252
        cov     = log_ret.cov().values  * 252

        min_w = np.array([constraints.get(t, (0, 1))[0] for t in available])
        max_w = np.array([constraints.get(t, (0, 1))[1] for t in available])

        with st.spinner("Running Markowitz optimisation…"):
            w_tan, w_mvp, w_rp, fv, fr, mc_r, mc_v, mc_s = optimize(
                tuple(float(x) for x in mu),
                tuple(float(x) for x in cov.flatten()),
                len(available), rf,
                tuple(float(x) for x in min_w),
                tuple(float(x) for x in max_w),
            )

        st.session_state.res = dict(
            prices=prices, returns=log_ret,
            mu=mu, cov=cov,
            w_tan=w_tan, w_mvp=w_mvp, w_rp=w_rp,
            front_vols=fv, front_rets=fr,
            mc_rets=mc_r, mc_vols=mc_v, mc_sh=mc_s,
            assets=available, rf=rf,
            budget=budget, currency_key=currency_key,
        )

    # ── Landing page ──────────────────────────────────────────────
    if "res" not in st.session_state:
        st.markdown("""
<div style="background:linear-gradient(135deg,#0D1828,#121F33);
    border:1px solid rgba(212,175,55,0.18);border-radius:16px;
    padding:40px;text-align:center;margin-top:20px;">
  <div style="font-size:3rem;margin-bottom:16px;">📊</div>
  <div style="font-size:1.1rem;font-weight:600;color:#EEF2F7;margin-bottom:8px;">
    Ready to optimise your portfolio</div>
  <div style="color:#64748B;font-size:0.85rem;max-width:440px;margin:0 auto;">
    Search for assets in the sidebar, configure currency and constraints,
    then click <strong style="color:#D4AF37;">▶ RUN ANALYSIS</strong>.
  </div>
</div>""", unsafe_allow_html=True)
        return

    # ── Unpack session state ──────────────────────────────────────
    res                    = st.session_state.res
    assets                 = res["assets"]
    prices                 = res["prices"]
    returns                = res["returns"]
    mu, cov, rf            = res["mu"], res["cov"], res["rf"]
    w_tan, w_mvp, w_rp     = res["w_tan"], res["w_mvp"], res["w_rp"]
    vols                   = np.sqrt(np.diag(cov))
    n                      = len(assets)
    bud                    = res["budget"]
    curr_key               = res["currency_key"]
    sym                    = CURRENCIES[curr_key]["symbol"]

    tan_r, tan_v, tan_sh  = pmetrics(w_tan, mu, cov, rf)
    mvp_r, mvp_v, mvp_sh  = pmetrics(w_mvp, mu, cov, rf)
    rp_r,  rp_v,  rp_sh   = pmetrics(w_rp,  mu, cov, rf)
    tan_mdd  = port_max_dd(prices, w_tan)
    tan_cagr = port_cagr(prices, w_tan)
    mvp_mdd  = port_max_dd(prices, w_mvp)
    mvp_cagr = port_cagr(prices, w_mvp)
    rp_mdd   = port_max_dd(prices, w_rp)
    rp_cagr  = port_cagr(prices, w_rp)

    # ── KPI strip ─────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.68rem;font-weight:700;color:#64748B;'
        'text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px;">'
        'Tangent Portfolio &nbsp;·&nbsp; Key Metrics</div>',
        unsafe_allow_html=True,
    )
    kpis = [
        ("Sharpe Ratio",  f"{tan_sh:.3f}",          "◆"),
        ("Annual Return", f"{tan_r*100:+.2f}%",      "↗"),
        ("Volatility",    f"{tan_v*100:.2f}%",        "〜"),
        ("CAGR",          f"{tan_cagr*100:+.2f}%",   "∑"),
        ("Max Drawdown",  f"{tan_mdd*100:.1f}%",     "↘"),
        ("Assets",        str(n),                     "#"),
        ("Budget",        f"{sym} {bud:,}",           "💶"),
    ]
    cols = st.columns(7)
    for col, (label, value, icon) in zip(cols, kpis):
        col.markdown(kpi_card(label, value, icon), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "  📊  Frontier  ",
        "  📈  Prices & Drawdown  ",
        "  🥧  Allocations  ",
        "  🔬  Risk  ",
        "  🌪️  Scenarios  ",
        "  💰  Budget & Live  ",
        "  📋  Summary  ",
        "  ✅  Validation  ",
    ])

    with t1:
        st.plotly_chart(chart_frontier(res), use_container_width=True)

    with t2:
        st.plotly_chart(chart_prices(prices),   use_container_width=True)
        st.plotly_chart(chart_drawdown(prices), use_container_width=True)

    with t3:
        # Row 1 — Tangent + Min Variance side by side
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(chart_pie(w_tan, assets, "Tangent Portfolio", tan_r, tan_v, height=520), use_container_width=True)
        with c2: st.plotly_chart(chart_pie(w_mvp, assets, "Min Variance",      mvp_r, mvp_v, height=520), use_container_width=True)
        # Row 2 — Risk Parity centred (half width)
        _, c_mid, _ = st.columns([1, 2, 1])
        with c_mid: st.plotly_chart(chart_pie(w_rp, assets, "Risk Parity", rp_r, rp_v, height=520), use_container_width=True)
        st.plotly_chart(chart_cumulative(prices, w_tan, w_mvp, w_rp), use_container_width=True)

    with t4:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(chart_sharpe_bars(mu, vols, rf, assets, tan_sh), use_container_width=True)
        with c2: st.plotly_chart(chart_correlation(returns),                       use_container_width=True)
        st.plotly_chart(chart_risk_contrib(assets, w_rp, w_tan, w_mvp, cov), use_container_width=True)
        win = st.slider("Rolling window (days)", 60, 504, 252, 21)
        st.plotly_chart(chart_rolling_sharpe(returns, rf, win), use_container_width=True)

    with t5:
        render_scenario_tab(res)

    with t6:
        hdr, btn = st.columns([6, 1])
        hdr.markdown(
            f"<div style='font-size:1rem;font-weight:600;color:#EEF2F7;padding:4px 0;'>"
            f"Budget Allocation {live_badge()}</div>",
            unsafe_allow_html=True,
        )
        if btn.button("🔄", help="Refresh live prices"):
            get_live_prices.clear()
            get_fx_rates.clear()
            st.rerun()

        with st.spinner("Fetching live prices & FX rates…"):
            live_prices = get_live_prices(tuple(assets))
            fx_rates    = get_fx_rates()
            fx_rate     = fx_rates.get(curr_key, 1.0)

        hist_end = {a: float(prices[a].iloc[-1]) for a in assets}

        st.markdown(
            f'<div style="display:flex;gap:20px;align-items:center;'
            f'background:rgba(13,24,40,0.7);border:1px solid rgba(212,175,55,0.12);'
            f'border-radius:10px;padding:10px 16px;margin-bottom:20px;'
            f'font-size:0.75rem;color:#64748B;">'
            f'<span>💱 {curr_key}: <strong style="color:#D4AF37;">1 {sym} = ${fx_rate:.4f}</strong></span>'
            f'<span>·</span>'
            f'<span>💰 Budget: <strong style="color:#D4AF37;">{sym} {bud:,}</strong></span>'
            f'<span>·</span><span>🔄 Cached 60s</span></div>',
            unsafe_allow_html=True,
        )

        for port_name, w_arr in [
            ("Tangent Portfolio (Max Sharpe)", w_tan),
            ("Min Variance Portfolio",          w_mvp),
            ("Risk Parity Portfolio",            w_rp),
        ]:
            section_title(port_name)
            df_t = build_alloc_table(
                assets, w_arr, bud, live_prices, hist_end, fx_rate, sym,
            )
            st.dataframe(
                df_t.style.map(delta_color, subset=["Δ Backtest End"]),
                use_container_width=True, hide_index=True,
            )
            st.plotly_chart(
                chart_budget_bars(assets, w_arr, bud, port_name, sym),
                use_container_width=True,
            )
            st.divider()

        section_title("Live Price vs Backtest End")
        st.plotly_chart(
            chart_live_vs_backtest(assets, hist_end, live_prices),
            use_container_width=True,
        )

        section_title("Live Price Details")
        live_rows = []
        for a in assets:
            live = live_prices.get(a)
            hist = hist_end.get(a)
            chg  = (live / hist - 1) * 100 if live and hist and hist > 0 else None
            live_rows.append({
                "Ticker":               a,
                "Backtest End Price":   f"${hist:,.2f}" if hist else "—",
                "Live Price":           f"${live:,.2f}" if live else "—",
                "Change":               (f"+{chg:.2f}%" if chg and chg >= 0
                                         else f"{chg:.2f}%" if chg else "—"),
                f"Value ({sym}/share)": f"{sym}{live/fx_rate:,.2f}" if live else "—",
            })
        ldf = pd.DataFrame(live_rows)
        st.dataframe(
            ldf.style.map(delta_color, subset=["Change"]),
            use_container_width=True, hide_index=True,
        )

    with t7:
        section_title("Portfolio Weights")
        st.dataframe(pd.DataFrame({
            "Asset":            assets,
            "Tangent (%)":      [f"{w_tan[i]*100:.1f}%" for i in range(n)],
            "Min Variance (%)": [f"{w_mvp[i]*100:.1f}%" for i in range(n)],
            "Risk Parity (%)":  [f"{w_rp[i]*100:.1f}%"  for i in range(n)],
        }), use_container_width=True, hide_index=True)

        section_title("Portfolio Comparison")
        st.dataframe(pd.DataFrame({
            "Metric":       ["Annual Return", "Volatility", "Sharpe", "CAGR", "Max Drawdown"],
            "Tangent":      [f"{tan_r*100:+.2f}%", f"{tan_v*100:.2f}%",
                             f"{tan_sh:.3f}", f"{tan_cagr*100:+.2f}%", f"{tan_mdd*100:.2f}%"],
            "Min Variance": [f"{mvp_r*100:+.2f}%", f"{mvp_v*100:.2f}%",
                             f"{mvp_sh:.3f}", f"{mvp_cagr*100:+.2f}%", f"{mvp_mdd*100:.2f}%"],
            "Risk Parity":  [f"{rp_r*100:+.2f}%",  f"{rp_v*100:.2f}%",
                             f"{rp_sh:.3f}",  f"{rp_cagr*100:+.2f}%",  f"{rp_mdd*100:.2f}%"],
        }), use_container_width=True, hide_index=True)

        section_title("Individual Asset Statistics")
        st.dataframe(pd.DataFrame({
            "Asset":         assets,
            "Annual Return": [f"{mu[i]*100:+.2f}%" for i in range(n)],
            "Volatility":    [f"{vols[i]*100:.2f}%" for i in range(n)],
            "Sharpe":        [f"{(mu[i]-rf)/vols[i]:.3f}" for i in range(n)],
            "CAGR":          [f"{asset_cagr(prices[a])*100:+.2f}%" for a in assets],
            "Max DD":        [f"{asset_max_dd(prices[a])*100:.2f}%" for a in assets],
        }), use_container_width=True, hide_index=True)

        section_title("Export")
        export_df = pd.DataFrame({
            "asset":             assets,
            "tangent_weight":    w_tan,
            "minvar_weight":     w_mvp,
            "riskparity_weight": w_rp,
            "tangent_alloc":     [bud * w for w in w_tan],
            "minvar_alloc":      [bud * w for w in w_mvp],
            "riskparity_alloc":  [bud * w for w in w_rp],
            "currency":          curr_key,
            "annual_return":     mu,
            "annual_volatility": vols,
            "sharpe":            [(mu[i] - rf) / vols[i] for i in range(n)],
        })
        st.download_button(
            "⬇  Download Full Report (CSV)",
            data=export_df.to_csv(index=False),
            file_name="portfolio_optimizer_results.csv",
            mime="text/csv",
        )
        st.markdown("""
<div style="text-align:center;margin-top:40px;padding:20px;
    border-top:1px solid rgba(212,175,55,0.1);
    color:#64748B;font-size:0.72rem;letter-spacing:0.06em;">
  PORTFOLIO OPTIMIZER · MARKOWITZ · (c) AEG<br>
  <span style="color:rgba(212,175,55,0.4);font-size:0.65rem;">
    Not financial advice — Educational purposes only</span>
</div>""", unsafe_allow_html=True)

    with t8:
        render_validation_panel(assets, btn_key="revalidate_tab")
        section_title("Weight Constraints Applied")
        if constraints:
            rows = [
                {
                    "Asset":      t,
                    "Min Weight": f"{constraints.get(t,(0,1))[0]*100:.0f}%",
                    "Max Weight": f"{constraints.get(t,(0,1))[1]*100:.0f}%",
                    "Status":     ("⚠️ Tight"
                                   if (constraints.get(t,(0,1))[1] - constraints.get(t,(0,1))[0]) < 0.1
                                   else "✅ OK"),
                }
                for t in assets
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No custom constraints applied — weights free between 0% and 100%.")


if __name__ == "__main__":
    main()
