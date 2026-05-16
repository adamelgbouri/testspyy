"""
Portfolio Optimizer — Streamlit Edition  ™ by AEG
Markowitz Modern Portfolio Theory — Premium UI
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

# ─── Premium CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --bg:        #05080D;
    --bg2:       #0A1220;
    --bg3:       #0F1A2E;
    --card:      #111927;
    --border:    rgba(212,175,55,0.18);
    --gold:      #D4AF37;
    --gold-dim:  #B8860B;
    --gold-glow: rgba(212,175,55,0.25);
    --text:      #EEF2F7;
    --muted:     #64748B;
    --green:     #10B981;
    --red:       #EF4444;
    --blue:      #3B82F6;
}

/* ── Global ─────────────────────────────────────── */
*, *::before, *::after {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
}
[data-testid="stHeader"] {
    background: rgba(5,8,13,0.95) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stMain"] {
    padding-top: 0 !important;
}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060B14 0%, #0B1828 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: var(--text) !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
}

/* ── Metric cards ────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0D1828 0%, #121F33 100%) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5),
                inset 0 1px 0 rgba(212,175,55,0.07) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    cursor: default;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6),
                0 0 0 1px rgba(212,175,55,0.3) !important;
}
div[data-testid="stMetricValue"] {
    color: var(--gold) !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* ── Tabs ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 2px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: rgba(212,175,55,0.05) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text) !important;
    background: rgba(255,255,255,0.03) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 24px !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--gold-dim) 0%, var(--gold) 50%, var(--gold-dim) 100%) !important;
    color: #05080D !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1rem !important;
    border-radius: 8px !important;
    box-shadow: 0 0 28px rgba(212,175,55,0.35),
                0 2px 8px rgba(0,0,0,0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 42px rgba(212,175,55,0.65),
                0 4px 16px rgba(0,0,0,0.5) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]) {
    background: rgba(13,18,28,0.8) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--gold) !important;
    color: var(--gold) !important;
    background: rgba(212,175,55,0.06) !important;
}

/* ── Inputs ──────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg2) !important;
    border: 1px solid rgba(212,175,55,0.2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.875rem !important;
    caret-color: var(--gold) !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(212,175,55,0.12) !important;
    outline: none !important;
}
.stSelectbox > div > div {
    background: var(--bg2) !important;
    border: 1px solid rgba(212,175,55,0.2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(212,175,55,0.12) !important;
}

/* ── Slider ──────────────────────────────────────── */
.stSlider > div > div > div > div[role="slider"] {
    background: var(--gold) !important;
    border: 2px solid var(--bg) !important;
    box-shadow: 0 0 10px var(--gold-glow) !important;
}
[data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
}

/* ── DataFrames ──────────────────────────────────── */
.stDataFrame, [data-testid="stDataFrameResizable"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}
[data-testid="stDataFrame"] table {
    background: var(--bg2) !important;
}
[data-testid="stDataFrame"] th {
    background: linear-gradient(135deg, #0D1828, #121F33) !important;
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 10px 14px !important;
}
[data-testid="stDataFrame"] td {
    color: var(--text) !important;
    font-size: 0.82rem !important;
    padding: 9px 14px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background: rgba(212,175,55,0.04) !important;
}

/* ── Dividers ────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent 0%, var(--gold) 50%, transparent 100%) !important;
    opacity: 0.25 !important;
    margin: 1.2rem 0 !important;
}

/* ── Alerts ──────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: rgba(13,18,28,0.8) !important;
}
[data-testid="stAlert"][data-baseweb="notification"] {
    border-left: 3px solid var(--gold) !important;
}

/* ── Download button ─────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(212,175,55,0.08) !important;
    border-color: var(--gold) !important;
}

/* ── Number input stepper ────────────────────────── */
.stNumberInput button {
    color: var(--gold) !important;
    background: var(--bg2) !important;
    border-color: rgba(212,175,55,0.15) !important;
}

/* ── Spinner ─────────────────────────────────────── */
.stSpinner > div > div {
    border-top-color: var(--gold) !important;
}

/* ── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar          { width: 5px; height: 5px; }
::-webkit-scrollbar-track    { background: var(--bg); }
::-webkit-scrollbar-thumb    { background: rgba(212,175,55,0.22); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(212,175,55,0.5); }

/* ── Animations ──────────────────────────────────── */
@keyframes pulse-gold {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px rgba(212,175,55,0.6); }
    50%       { opacity: 0.5; box-shadow: 0 0 2px rgba(212,175,55,0.2); }
}
@keyframes fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fade-in 0.4s ease forwards; }

/* ── Caption / small text ────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
}

/* ── Markdown text ───────────────────────────────── */
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
[data-testid="stMarkdownContainer"] code {
    background: rgba(212,175,55,0.1) !important;
    color: var(--gold) !important;
    border: 1px solid rgba(212,175,55,0.2) !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 0.82em !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Theme constants ──────────────────────────────────────────────
BG      = "#05080D"
PANEL   = "#0A1220"
GRID    = "rgba(255,255,255,0.05)"
TEXT    = "#EEF2F7"
MUTED   = "#64748B"
GOLD    = "#D4AF37"
GREEN   = "#10B981"
RED     = "#EF4444"
BLUE    = "#3B82F6"
ORANGE  = "#F59E0B"

PALETTE = [
    "#D4AF37", "#3B82F6", "#10B981", "#EF4444",
    "#8B5CF6", "#F59E0B", "#06B6D4", "#EC4899",
    "#84CC16", "#F97316",
]

N_MC = 8_000
SEED = 42


def base_layout(title="", xtitle="", ytitle="", **kw) -> dict:
    return dict(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        title=dict(text=title, font=dict(size=14, color=TEXT, family="Inter")),
        xaxis=dict(title=xtitle, gridcolor=GRID, zerolinecolor=GRID,
                   color=MUTED, title_font=dict(color=MUTED, size=11),
                   tickfont=dict(color=MUTED, size=10),
                   linecolor=GRID),
        yaxis=dict(title=ytitle, gridcolor=GRID, zerolinecolor=GRID,
                   color=MUTED, title_font=dict(color=MUTED, size=11),
                   tickfont=dict(color=MUTED, size=10),
                   linecolor=GRID),
        legend=dict(bgcolor="rgba(10,18,32,0.85)", bordercolor=GRID,
                    borderwidth=1, font=dict(size=11, color=TEXT)),
        margin=dict(l=55, r=25, t=56, b=50),
        **kw,
    )


# ─── UI helpers ───────────────────────────────────────────────────

def page_header():
    st.markdown("""
<div class="fade-in" style="
    display:flex; align-items:center; gap:16px;
    padding:28px 0 22px 0;
    border-bottom:1px solid rgba(212,175,55,0.15);
    margin-bottom:28px;
">
    <div style="
        width:46px; height:46px; flex-shrink:0;
        background:linear-gradient(135deg,#B8860B,#D4AF37);
        border-radius:12px;
        display:flex; align-items:center; justify-content:center;
        font-size:22px;
        box-shadow:0 0 28px rgba(212,175,55,0.4);
    ">📈</div>
    <div>
        <div style="
            font-family:'Playfair Display',serif;
            font-size:1.9rem; font-weight:700; line-height:1.15;
            background:linear-gradient(135deg,#C9A440,#F0D060,#C9A440);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text;
        ">Portfolio Optimizer</div>
        <div style="
            color:#64748B; font-size:0.72rem; font-weight:500;
            letter-spacing:0.14em; text-transform:uppercase; margin-top:3px;
        ">Modern Portfolio Theory &nbsp;·&nbsp; Markowitz Efficient Frontier &nbsp;·&nbsp; by AEG</div>
    </div>
</div>
""", unsafe_allow_html=True)


def section_title(text: str, sub: str = ""):
    sub_html = (f'<div style="color:#64748B;font-size:0.73rem;margin-top:3px;">{sub}</div>'
                if sub else "")
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:28px 0 16px 0;">
    <div style="
        width:3px;height:26px;flex-shrink:0;border-radius:2px;
        background:linear-gradient(180deg,#D4AF37,rgba(212,175,55,0));
    "></div>
    <div>
        <div style="font-size:0.95rem;font-weight:600;color:#EEF2F7;letter-spacing:0.01em;">
            {text}
        </div>{sub_html}
    </div>
</div>
""", unsafe_allow_html=True)


def live_badge():
    return """<span style="
        display:inline-flex;align-items:center;gap:5px;
        background:rgba(212,175,55,0.1);border:1px solid rgba(212,175,55,0.3);
        border-radius:20px;padding:2px 10px;font-size:0.68rem;font-weight:600;
        color:#D4AF37;letter-spacing:0.08em;text-transform:uppercase;
        vertical-align:middle;margin-left:8px;
    ">
        <span style="
            width:6px;height:6px;border-radius:50%;background:#D4AF37;
            animation:pulse-gold 1.4s ease-in-out infinite;
            box-shadow:0 0 6px rgba(212,175,55,0.8);
        "></span>LIVE
    </span>"""


def kpi_card(label: str, value: str, icon: str = "") -> str:
    return f"""
<div style="
    background:linear-gradient(135deg,#0D1828 0%,#121F33 100%);
    border:1px solid rgba(212,175,55,0.18);border-radius:14px;
    padding:18px 20px;height:100%;
    box-shadow:0 4px 24px rgba(0,0,0,0.5),inset 0 1px 0 rgba(212,175,55,0.06);
    transition:transform 0.2s,box-shadow 0.2s;
">
    <div style="color:#64748B;font-size:0.66rem;font-weight:600;
                text-transform:uppercase;letter-spacing:0.1em;">
        {icon}&nbsp; {label}
    </div>
    <div style="color:#D4AF37;font-size:1.45rem;font-weight:700;
                letter-spacing:-0.02em;margin-top:7px;line-height:1.1;">
        {value}
    </div>
</div>"""


def sidebar_section(title: str):
    st.sidebar.markdown(f"""
<div style="
    font-size:0.68rem;font-weight:700;color:#64748B;
    text-transform:uppercase;letter-spacing:0.12em;
    margin:18px 0 8px 0;padding-bottom:6px;
    border-bottom:1px solid rgba(212,175,55,0.1);
">
    {title}
</div>""", unsafe_allow_html=True)


# ─── Portfolio math ───────────────────────────────────────────────

def pmetrics(w, mu, cov, rf):
    r = float(w @ mu)
    v = float(np.sqrt(max(float(w @ cov @ w), 0)))
    s = (r - rf) / v if v > 1e-9 else 0.0
    return r, v, s


def port_cum(prices: pd.DataFrame, w: np.ndarray) -> pd.Series:
    ret = prices.pct_change().dropna()
    return (1 + (ret * w).sum(axis=1)).cumprod() * 100


def port_max_dd(prices: pd.DataFrame, w: np.ndarray) -> float:
    cum = port_cum(prices, w) / 100
    return float((cum / cum.cummax() - 1).min())


def port_cagr(prices: pd.DataFrame, w: np.ndarray) -> float:
    cum = port_cum(prices, w) / 100
    n_years = len(cum) / 252
    return float(cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0.1 else 0.0


def asset_max_dd(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min())


def asset_cagr(s: pd.Series) -> float:
    n_years = len(s) / 252
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / n_years) - 1) if n_years > 0.1 else 0.0


# ─── Asset database ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_asset_db() -> list:
    try:
        import financedatabase as fd
        ALLOWED = {"NMS", "NAS", "NYQ", "NYS", "PCX", "ARCX",
                   "PAR", "LSE", "SHH", "JPX", "FRA", "XETRA"}
        datasets = {
            "Equity":   fd.Equities().select(market_cap=["Large Cap", "Mega Cap", "Mid Cap"]),
            "ETF":      fd.ETFs().select(),
            "Crypto":   fd.Cryptos().select(),
            "Index":    fd.Indices().select(),
            "Fund":     fd.Funds().select(),
            "Currency": fd.Currencies().select(),
        }
        out = []
        for atype, df in datasets.items():
            if "exchange" in df.columns:
                df = df[df["exchange"].isin(ALLOWED) | df["exchange"].isna()]
            for sym, row in df.iterrows():
                name  = str(row.get("name", "") or "")
                exch  = str(row.get("exchange", "") or "")
                label = f"{name}  ({sym})  —  {atype}  {exch}".strip()
                out.append({"label": label, "symbol": str(sym).upper()})
        return sorted(out, key=lambda x: x["label"].lower())
    except Exception:
        return []


def search_assets(db: list, query: str, limit: int = 120) -> list:
    if not query:
        return db[:60]
    q = query.lower()
    return [item for item in db if q in item["label"].lower()][:limit]


# ─── Live prices ─────────────────────────────────────────────────

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
        return {str(t): float(close[t])
                for t in tickers
                if t in close.index and not pd.isna(close[t])}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_eur_usd() -> float:
    try:
        rate = yf.Ticker("EURUSD=X").fast_info.last_price
        return float(rate) if rate and rate > 0 else 1.08
    except Exception:
        return 1.08


# ─── Data & optimization ─────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        key = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        raw = raw[key]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    return raw.dropna()


@st.cache_data(show_spinner=False)
def optimize(mu_t: tuple, cov_t: tuple, n: int, rf: float):
    mu  = np.array(mu_t)
    cov = np.array(cov_t).reshape(n, n)
    eq  = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bds = [(0, 1)] * n
    w0  = np.ones(n) / n
    rng = np.random.default_rng(SEED)

    sol   = minimize(lambda w: w @ cov @ w, w0, method="SLSQP",
                     bounds=bds, constraints=eq)
    w_mvp = sol.x

    best, w_tan = -np.inf, w0.copy()
    for _ in range(50):
        r = minimize(
            lambda w: -(float(w @ mu) - rf) / max(np.sqrt(float(w @ cov @ w)), 1e-9),
            rng.dirichlet(np.ones(n)),
            method="SLSQP", bounds=bds, constraints=eq,
        )
        if r.success and -r.fun > best:
            best, w_tan = -r.fun, r.x

    lo, hi = float(w_mvp @ mu), float(mu.max()) * 0.98
    fv, fr = [], []
    for t in np.linspace(lo, hi, 100):
        r = minimize(
            lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bds,
            constraints=eq + [{"type": "eq",
                                "fun": lambda w, t=t: float(w @ mu) - t}],
        )
        if r.success:
            fv.append(float(np.sqrt(r.x @ cov @ r.x)))
            fr.append(t)

    rng2 = np.random.default_rng(SEED)
    mc_r, mc_v, mc_s = [], [], []
    for _ in range(N_MC):
        x  = rng2.exponential(1, n)
        ww = x / x.sum()
        r, v, s = pmetrics(ww, mu, cov, rf)
        mc_r.append(r); mc_v.append(v); mc_s.append(s)

    return (w_tan, w_mvp,
            np.array(fv), np.array(fr),
            np.array(mc_r), np.array(mc_v), np.array(mc_s))


# ─── Charts ───────────────────────────────────────────────────────

def chart_frontier(res: dict) -> go.Figure:
    mu, cov, rf  = res["mu"], res["cov"], res["rf"]
    assets       = res["assets"]
    w_tan, w_mvp = res["w_tan"], res["w_mvp"]
    vols         = np.sqrt(np.diag(cov))
    tan_r, tan_v, tan_sh = pmetrics(w_tan, mu, cov, rf)
    mvp_r, mvp_v, _      = pmetrics(w_mvp, mu, cov, rf)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=res["mc_vols"]*100, y=res["mc_rets"]*100, mode="markers",
        marker=dict(size=3, color=res["mc_sh"], colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Sharpe", x=1.01, thickness=12,
                                  tickfont=dict(color=MUTED, size=9),
                                  title_font=dict(color=MUTED, size=10))),
        opacity=0.35, name="Simulated portfolios",
        hovertemplate="Vol %{x:.2f}%  ·  Ret %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=res["front_vols"]*100, y=res["front_rets"]*100,
        mode="lines", line=dict(color=GOLD, width=2.5),
        name="Efficient Frontier",
    ))
    xr = np.linspace(0, res["mc_vols"].max()*100*1.25, 120)
    fig.add_trace(go.Scatter(
        x=xr, y=rf*100 + tan_sh*xr, mode="lines",
        line=dict(color="rgba(212,175,55,0.35)", width=1.5, dash="dash"),
        name="Capital Market Line",
    ))
    for x, y, sym, col, sz, nm in [
        (tan_v*100, tan_r*100, "star",    "#FF6B6B", 24,
         f"Tangent  ·  Sharpe={tan_sh:.2f}"),
        (mvp_v*100, mvp_r*100, "diamond", BLUE,      18, "Min Variance"),
        (0,         rf*100,    "circle",  ORANGE,    13, f"Risk-free {rf*100:.1f}%"),
    ]:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(size=sz, symbol=sym, color=col,
                        line=dict(color=BG, width=2)),
            name=nm,
        ))
    for i, a in enumerate(assets):
        ar, av = float(mu[i]), float(vols[i])
        fig.add_trace(go.Scatter(
            x=[av*100], y=[ar*100], mode="markers+text",
            marker=dict(size=10, symbol="circle-open",
                        color=PALETTE[i % len(PALETTE)],
                        line=dict(width=2.2)),
            text=[a], textposition="top right",
            textfont=dict(color=TEXT, size=10, family="Inter"), name=a,
            hovertemplate=(f"<b>{a}</b><br>Vol {av*100:.2f}%<br>"
                           f"Ret {ar*100:.2f}%<br>"
                           f"Sharpe {(ar-rf)/av:.3f}<extra></extra>"),
        ))
    fig.update_layout(**base_layout(
        "Markowitz Efficient Frontier",
        "Annual Volatility (%)", "Annual Return (%)", height=560,
    ))
    return fig


def chart_prices(prices: pd.DataFrame) -> go.Figure:
    norm = prices / prices.iloc[0] * 100
    fig  = go.Figure()
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col], mode="lines", name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            hovertemplate=f"{col}: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**base_layout(
        "Normalized Prices (base 100)", "Date", "Price Index"))
    return fig


def chart_drawdown(prices: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(prices.columns):
        roll_max = prices[col].cummax()
        dd = (prices[col] - roll_max) / roll_max * 100
        fig.add_trace(go.Scatter(
            x=prices.index, y=dd, mode="lines", name=col,
            fill="tozeroy",
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.4),
            opacity=0.6,
            hovertemplate=f"{col}: %{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(**base_layout("Drawdown (%)", "Date", "Drawdown (%)"))
    return fig


def chart_pie(w: np.ndarray, assets: list,
              title: str, ret: float, vol: float) -> go.Figure:
    mask   = w > 0.01
    labels = [assets[i] for i in range(len(assets)) if mask[i]]
    vals   = w[mask]
    if len(vals) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No significant weights",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color=TEXT, size=14), showarrow=False)
        fig.update_layout(**base_layout(title))
        return fig
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.42,
        marker=dict(colors=PALETTE[:len(labels)],
                    line=dict(color=BG, width=2.5)),
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="%{label}: %{percent} · %{value:.3f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        f"{title}  ·  Ret {ret*100:+.1f}%  ·  Vol {vol*100:.1f}%",
        height=420,
    ))
    return fig


def chart_cumulative(prices: pd.DataFrame,
                     w_tan: np.ndarray, w_mvp: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for w, name, color in [
        (w_tan, "Tangent Portfolio", GOLD),
        (w_mvp, "Min Variance",      BLUE),
    ]:
        cum = port_cum(prices, w)
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines",
            name=name, line=dict(color=color, width=2.2),
            hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**base_layout(
        "Cumulative Portfolio Performance (base 100)", "Date", "Value"))
    return fig


def chart_sharpe_bars(mu, vols, rf, assets, tan_sh) -> go.Figure:
    shs = [(mu[i] - rf) / vols[i] for i in range(len(assets))]
    fig = go.Figure(go.Bar(
        x=shs, y=assets, orientation="h",
        marker=dict(
            color=[GREEN if s >= 0 else RED for s in shs],
            line=dict(color=BG, width=1),
            opacity=0.85,
        ),
        text=[f"{s:.2f}" for s in shs], textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.add_vline(x=tan_sh,
                  line=dict(color=GOLD, width=1.8, dash="dash"),
                  annotation_text=f"Tangent ({tan_sh:.2f})",
                  annotation_font_color=GOLD,
                  annotation_font_size=11)
    fig.update_layout(**base_layout(
        "Individual Sharpe Ratios", "Sharpe Ratio", "",
        height=max(300, len(assets) * 46 + 120),
    ))
    return fig


def chart_correlation(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    z    = np.round(corr.values, 2)
    fig  = go.Figure(go.Heatmap(
        z=z, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[
            [0.0,  "#EF4444"], [0.5, "#1A2540"], [1.0, "#D4AF37"]
        ],
        zmid=0, zmin=-1, zmax=1,
        text=z, texttemplate="%{text:.2f}",
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


def chart_rolling_sharpe(returns: pd.DataFrame, rf: float, window: int) -> go.Figure:
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
    fig.update_layout(**base_layout(
        f"Rolling {window}d Sharpe Ratio", "Date", "Sharpe"))
    return fig


def chart_budget_bars(assets, weights, budget, label="Portfolio") -> go.Figure:
    amounts = [budget * w for w in weights]
    mask    = np.array(weights) > 0.005
    a_show  = [assets[i] for i in range(len(assets)) if mask[i]]
    v_show  = [amounts[i] for i in range(len(assets)) if mask[i]]
    colors  = [PALETTE[i % len(PALETTE)]
               for i in range(len(assets)) if mask[i]]
    fig = go.Figure(go.Bar(
        x=v_show, y=a_show, orientation="h",
        marker=dict(color=colors, line=dict(color=BG, width=1), opacity=0.88),
        text=[f"€{v:,.0f}" for v in v_show], textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="%{y}  ·  €%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        f"Budget Allocation — {label}  (Total €{budget:,})",
        "Montant alloué (€)", "",
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
        "Performance depuis la fin du backtest (prix live vs dernier prix historique)",
        "Variation (%)", "",
        height=max(300, len(assets) * 46 + 120),
    ))
    return fig


# ─── Budget allocation table ──────────────────────────────────────

def build_alloc_table(assets, weights, budget, live_prices,
                      hist_end, eur_usd) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(assets):
        w         = float(weights[i])
        alloc_eur = budget * w
        alloc_usd = alloc_eur * eur_usd
        live      = live_prices.get(a)
        hist      = hist_end.get(a)
        shares    = alloc_usd / live if live and live > 0 else None
        chg_pct   = (live / hist - 1) * 100 if live and hist and hist > 0 else None
        live_val  = (shares * live / eur_usd) if shares and live else None

        rows.append({
            "Asset":               a,
            "Poids":               f"{w*100:.1f}%",
            "Alloué (€)":          f"€ {alloc_eur:,.0f}",
            "Alloué ($)":          f"$ {alloc_usd:,.0f}",
            "Prix live":           f"$ {live:,.2f}" if live else "—",
            "Nb actions":          f"{shares:.4f}" if shares else "—",
            "Valeur actuelle (€)": f"€ {live_val:,.0f}" if live_val else "—",
            "Δ fin backtest":      (f"+{chg_pct:.1f}%" if chg_pct and chg_pct >= 0
                                    else f"{chg_pct:.1f}%" if chg_pct else "—"),
        })
    return pd.DataFrame(rows)


def style_delta(df: pd.DataFrame):
    def _color(val):
        if isinstance(val, str) and val.startswith("+"):
            return f"color:{GREEN}"
        if isinstance(val, str) and val.startswith("-"):
            return f"color:{RED}"
        return f"color:{MUTED}"
    return df.style.map(_color, subset=["Δ fin backtest"])


# ─── Sidebar ──────────────────────────────────────────────────────

def render_sidebar() -> tuple:
    st.sidebar.markdown(f"""
<div style="
    padding:20px 0 14px 0;
    border-bottom:1px solid rgba(212,175,55,0.14);
    margin-bottom:4px;
">
    <div style="
        font-family:'Playfair Display',serif;
        font-size:1.15rem;font-weight:700;
        color:#D4AF37;letter-spacing:0.02em;
    ">📈 Portfolio Optimizer</div>
    <div style="
        color:#64748B;font-size:0.66rem;font-weight:600;
        text-transform:uppercase;letter-spacing:0.14em;margin-top:5px;
    ">Markowitz MPT  ·  by AEG</div>
</div>
""", unsafe_allow_html=True)

    sidebar_section("Paramètres")
    c1, c2 = st.sidebar.columns(2)
    start  = c1.date_input("Début", date(2019, 1, 1))
    end    = c2.date_input("Fin",   date(2026, 1, 1))
    rf_pct = st.sidebar.slider(
        "Taux sans risque (%)", 0.0, 10.0, 4.0, 0.25, format="%.2f%%"
    )

    sidebar_section("Budget")
    budget = st.sidebar.number_input(
        "Montant (€)", min_value=100, max_value=10_000_000,
        value=4000, step=100, format="%d",
        label_visibility="collapsed",
    )
    st.sidebar.markdown(f"""
<div style="
    background:linear-gradient(135deg,rgba(212,175,55,0.08),rgba(212,175,55,0.04));
    border:1px solid rgba(212,175,55,0.18);border-radius:8px;
    padding:8px 12px;margin-top:4px;display:flex;align-items:center;gap:8px;
">
    <span style="font-size:1rem;">💶</span>
    <span style="color:#D4AF37;font-weight:700;font-size:0.95rem;">
        € {budget:,}
    </span>
    <span style="color:#64748B;font-size:0.72rem;">budget total</span>
</div>""", unsafe_allow_html=True)

    sidebar_section("Base d'actifs")
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = ["AAPL", "MSFT"]

    with st.sidebar:
        with st.spinner("Chargement…"):
            db = load_asset_db()

    if db:
        st.sidebar.caption(f"{len(db):,} actifs disponibles")
        query   = st.sidebar.text_input(
            "🔍", placeholder="Nom ou ticker…", key="search_query",
            label_visibility="collapsed",
        )
        results = search_assets(db, query)
        if results:
            options = [r["label"] for r in results]
            chosen  = st.sidebar.selectbox(
                "select_result", options,
                label_visibility="collapsed", key="search_select",
            )
            if st.sidebar.button("＋  Ajouter au portfolio",
                                  use_container_width=True):
                symbol = next(
                    (r["symbol"] for r in results if r["label"] == chosen), None
                )
                if symbol:
                    if symbol not in st.session_state.selected_tickers:
                        st.session_state.selected_tickers.append(symbol)
                        st.rerun()
                    else:
                        st.sidebar.warning(f"{symbol} déjà sélectionné.")
        else:
            st.sidebar.caption("Aucun résultat.")
    else:
        st.sidebar.caption("Saisie manuelle.")
        col_in, col_btn = st.sidebar.columns([3, 1])
        manual = col_in.text_input("Ticker", placeholder="AAPL",
                                    label_visibility="collapsed",
                                    key="manual_ticker")
        if col_btn.button("＋", key="manual_add"):
            t = manual.strip().upper()
            if t and t not in st.session_state.selected_tickers:
                st.session_state.selected_tickers.append(t)
                st.rerun()

    sidebar_section(f"Portfolio  ·  {len(st.session_state.selected_tickers)} actif(s)")
    for i, ticker in enumerate(list(st.session_state.selected_tickers)):
        c1, c2 = st.sidebar.columns([5, 1])
        c1.markdown(f"""<span style="color:#64748B;font-size:0.72rem;">#{i+1}</span>
&nbsp;&nbsp;<code style="color:#D4AF37;background:rgba(212,175,55,0.1);
border:1px solid rgba(212,175,55,0.2);border-radius:4px;
padding:2px 6px;font-size:0.82rem;">{ticker}</code>""",
                    unsafe_allow_html=True)
        if c2.button("✕", key=f"rm_{i}", help=f"Retirer {ticker}"):
            if len(st.session_state.selected_tickers) > 2:
                st.session_state.selected_tickers.pop(i)
                st.rerun()
            else:
                st.sidebar.warning("Minimum 2 actifs.")

    st.sidebar.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    run = st.sidebar.button("▶  RUN ANALYSIS", type="primary",
                             use_container_width=True)
    return (start, end, rf_pct / 100.0,
            list(st.session_state.selected_tickers), budget, run)


# ─── Main ─────────────────────────────────────────────────────────

def main():
    page_header()

    start, end, rf, tickers, budget, run = render_sidebar()

    if run:
        if len(tickers) < 2:
            st.error("Sélectionner au moins 2 actifs.")
            st.stop()
        if start >= end:
            st.error("La date de début doit être antérieure à la date de fin.")
            st.stop()

        with st.spinner("Téléchargement des données Yahoo Finance…"):
            prices = load_prices(tuple(tickers), str(start), str(end))

        if prices.empty:
            st.error("Échec du téléchargement — vérifier les tickers.")
            st.stop()

        available = [t for t in tickers if t in prices.columns]
        missing   = [t for t in tickers if t not in prices.columns]
        if missing:
            st.warning(f"Tickers introuvables (ignorés) : **{', '.join(missing)}**")
        if len(available) < 2:
            st.error("Moins de 2 actifs disponibles.")
            st.stop()

        prices  = prices[available].dropna()
        log_ret = np.log(prices / prices.shift(1)).dropna()
        mu      = log_ret.mean().values * 252
        cov     = log_ret.cov().values  * 252

        with st.spinner("Optimisation Markowitz en cours…"):
            w_tan, w_mvp, fv, fr, mc_r, mc_v, mc_s = optimize(
                tuple(float(x) for x in mu),
                tuple(float(x) for x in cov.flatten()),
                len(available), rf,
            )

        st.session_state.res = dict(
            prices=prices, returns=log_ret,
            mu=mu, cov=cov, w_tan=w_tan, w_mvp=w_mvp,
            front_vols=fv, front_rets=fr,
            mc_rets=mc_r, mc_vols=mc_v, mc_sh=mc_s,
            assets=available, rf=rf, budget=budget,
        )

    if "res" not in st.session_state:
        st.markdown("""
<div style="
    background:linear-gradient(135deg,#0D1828,#121F33);
    border:1px solid rgba(212,175,55,0.18);border-radius:16px;
    padding:40px;text-align:center;margin-top:20px;
">
    <div style="font-size:3rem;margin-bottom:16px;">📊</div>
    <div style="font-size:1.1rem;font-weight:600;color:#EEF2F7;margin-bottom:8px;">
        Prêt à optimiser votre portfolio
    </div>
    <div style="color:#64748B;font-size:0.85rem;max-width:400px;margin:0 auto;">
        Recherchez vos actifs dans la sidebar, définissez votre budget,
        puis cliquez sur <strong style="color:#D4AF37;">▶ RUN ANALYSIS</strong>.
    </div>
</div>
""", unsafe_allow_html=True)
        return

    res          = st.session_state.res
    assets       = res["assets"]
    prices       = res["prices"]
    returns      = res["returns"]
    mu, cov, rf  = res["mu"], res["cov"], res["rf"]
    w_tan, w_mvp = res["w_tan"], res["w_mvp"]
    vols         = np.sqrt(np.diag(cov))
    n            = len(assets)
    bud          = res.get("budget", budget)

    tan_r, tan_v, tan_sh = pmetrics(w_tan, mu, cov, rf)
    mvp_r, mvp_v, mvp_sh = pmetrics(w_mvp, mu, cov, rf)
    tan_mdd  = port_max_dd(prices, w_tan)
    tan_cagr = port_cagr(prices, w_tan)
    mvp_mdd  = port_max_dd(prices, w_mvp)
    mvp_cagr = port_cagr(prices, w_mvp)

    # ── KPI strip ───────────────────────────────────────────────
    st.markdown("""
<div style="font-size:0.68rem;font-weight:700;color:#64748B;
            text-transform:uppercase;letter-spacing:0.12em;
            margin-bottom:12px;">
    Tangent Portfolio &nbsp;·&nbsp; Métriques clés
</div>""", unsafe_allow_html=True)

    kpis = [
        ("Sharpe Ratio",     f"{tan_sh:.3f}",           "◆"),
        ("Rendement annuel", f"{tan_r*100:+.2f}%",      "↗"),
        ("Volatilité",       f"{tan_v*100:.2f}%",       "〜"),
        ("CAGR",             f"{tan_cagr*100:+.2f}%",   "∑"),
        ("Max Drawdown",     f"{tan_mdd*100:.1f}%",     "↘"),
        ("Actifs",           str(n),                    "#"),
        ("Budget",           f"€ {bud:,}",              "💶"),
    ]
    cols = st.columns(7)
    for col, (label, value, icon) in zip(cols, kpis):
        col.markdown(kpi_card(label, value, icon), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.divider()

    # ── Tabs ────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "  📊  Frontier  ",
        "  📈  Prix & Drawdown  ",
        "  🥧  Allocations  ",
        "  🔬  Risque  ",
        "  💰  Budget & Live  ",
        "  📋  Résumé  ",
    ])

    with t1:
        st.plotly_chart(chart_frontier(res), use_container_width=True)

    with t2:
        st.plotly_chart(chart_prices(prices), use_container_width=True)
        st.plotly_chart(chart_drawdown(prices), use_container_width=True)

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                chart_pie(w_tan, assets, "Tangent Portfolio", tan_r, tan_v),
                use_container_width=True)
        with c2:
            st.plotly_chart(
                chart_pie(w_mvp, assets, "Min Variance", mvp_r, mvp_v),
                use_container_width=True)
        st.plotly_chart(chart_cumulative(prices, w_tan, w_mvp),
                        use_container_width=True)

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_sharpe_bars(mu, vols, rf, assets, tan_sh),
                            use_container_width=True)
        with c2:
            st.plotly_chart(chart_correlation(returns), use_container_width=True)
        win = st.slider("Fenêtre glissante (jours)", 60, 504, 252, 21)
        st.plotly_chart(chart_rolling_sharpe(returns, rf, win),
                        use_container_width=True)

    with t5:
        hdr, btn = st.columns([6, 1])
        hdr.markdown(
            f"<div style='font-size:1rem;font-weight:600;color:#EEF2F7;"
            f"padding:4px 0;'>Allocation budgétaire {live_badge()}</div>",
            unsafe_allow_html=True)
        if btn.button("🔄", help="Rafraîchir les prix live"):
            get_live_prices.clear()
            get_eur_usd.clear()
            st.rerun()

        with st.spinner("Récupération des prix live…"):
            live_prices = get_live_prices(tuple(assets))
            eur_usd     = get_eur_usd()

        hist_end = {a: float(prices[a].iloc[-1]) for a in assets}

        st.markdown(f"""
<div style="
    display:flex;gap:20px;align-items:center;
    background:rgba(13,24,40,0.7);border:1px solid rgba(212,175,55,0.12);
    border-radius:10px;padding:10px 16px;margin-bottom:20px;
    font-size:0.75rem;color:#64748B;
">
    <span>💱 EUR/USD : <strong style="color:#D4AF37;">{eur_usd:.4f}</strong></span>
    <span>·</span>
    <span>💶 Budget : <strong style="color:#D4AF37;">€ {bud:,}</strong></span>
    <span>·</span>
    <span>🔄 Mise à jour automatique toutes les 60 s</span>
</div>""", unsafe_allow_html=True)

        section_title("Tangent Portfolio (Max Sharpe)")
        df_tan = build_alloc_table(assets, w_tan, bud, live_prices, hist_end, eur_usd)
        st.dataframe(style_delta(df_tan), use_container_width=True, hide_index=True)
        st.plotly_chart(chart_budget_bars(assets, w_tan, bud, "Tangent Portfolio"),
                        use_container_width=True)

        st.divider()

        section_title("Min Variance Portfolio")
        df_mvp = build_alloc_table(assets, w_mvp, bud, live_prices, hist_end, eur_usd)
        st.dataframe(style_delta(df_mvp), use_container_width=True, hide_index=True)
        st.plotly_chart(chart_budget_bars(assets, w_mvp, bud, "Min Variance"),
                        use_container_width=True)

        st.divider()

        section_title("Prix live vs fin du backtest")
        st.plotly_chart(
            chart_live_vs_backtest(assets, hist_end, live_prices),
            use_container_width=True)

        section_title("Prix live détaillés")
        live_rows = []
        for a in assets:
            live = live_prices.get(a)
            hist = hist_end.get(a)
            chg  = (live / hist - 1) * 100 if live and hist and hist > 0 else None
            live_rows.append({
                "Ticker":             a,
                "Prix fin backtest":  f"$ {hist:,.2f}" if hist else "—",
                "Prix live":          f"$ {live:,.2f}" if live else "—",
                "Variation":          (f"+{chg:.2f}%" if chg and chg >= 0
                                       else f"{chg:.2f}%" if chg else "—"),
                "Valeur (€ / action)": f"€ {live/eur_usd:,.2f}" if live else "—",
            })
        live_df = pd.DataFrame(live_rows)
        st.dataframe(
            live_df.style.applymap(
                lambda v: (f"color:{GREEN}" if isinstance(v, str) and v.startswith("+")
                           else f"color:{RED}" if isinstance(v, str) and v.startswith("-")
                           else ""),
                subset=["Variation"],
            ),
            use_container_width=True, hide_index=True,
        )

    with t6:
        section_title("Poids du portfolio")
        st.dataframe(pd.DataFrame({
            "Asset":            assets,
            "Tangent (%)":      [f"{w_tan[i]*100:.1f}%" for i in range(n)],
            "Min Variance (%)": [f"{w_mvp[i]*100:.1f}%" for i in range(n)],
        }), use_container_width=True, hide_index=True)

        section_title("Comparaison des portfolios")
        st.dataframe(pd.DataFrame({
            "Métrique": ["Rendement annuel", "Volatilité", "Sharpe",
                         "CAGR", "Max Drawdown"],
            "Tangent Portfolio": [
                f"{tan_r*100:+.2f}%", f"{tan_v*100:.2f}%",
                f"{tan_sh:.3f}", f"{tan_cagr*100:+.2f}%",
                f"{tan_mdd*100:.2f}%"],
            "Min Variance": [
                f"{mvp_r*100:+.2f}%", f"{mvp_v*100:.2f}%",
                f"{mvp_sh:.3f}", f"{mvp_cagr*100:+.2f}%",
                f"{mvp_mdd*100:.2f}%"],
        }), use_container_width=True, hide_index=True)

        section_title("Statistiques individuelles")
        st.dataframe(pd.DataFrame({
            "Asset":        assets,
            "Rend. annuel": [f"{mu[i]*100:+.2f}%" for i in range(n)],
            "Volatilité":   [f"{vols[i]*100:.2f}%" for i in range(n)],
            "Sharpe":       [f"{(mu[i]-rf)/vols[i]:.3f}" for i in range(n)],
            "CAGR":         [f"{asset_cagr(prices[a])*100:+.2f}%" for a in assets],
            "Max DD":       [f"{asset_max_dd(prices[a])*100:.2f}%" for a in assets],
        }), use_container_width=True, hide_index=True)

        section_title("Export")
        export_df = pd.DataFrame({
            "asset":             assets,
            "tangent_weight":    w_tan,
            "minvar_weight":     w_mvp,
            "tangent_alloc_eur": [bud * w for w in w_tan],
            "minvar_alloc_eur":  [bud * w for w in w_mvp],
            "annual_return":     mu,
            "annual_volatility": vols,
            "sharpe":            [(mu[i]-rf)/vols[i] for i in range(n)],
        })
        st.download_button(
            "⬇  Télécharger le rapport complet (CSV)",
            data=export_df.to_csv(index=False),
            file_name="portfolio_optimizer_results.csv",
            mime="text/csv",
        )

        st.markdown("""
<div style="
    text-align:center;margin-top:40px;padding:20px;
    border-top:1px solid rgba(212,175,55,0.1);
    color:#64748B;font-size:0.72rem;letter-spacing:0.06em;
">
    PORTFOLIO OPTIMIZER  ·  MARKOWITZ MPT  ·  ™ by AEG<br>
    <span style="color:rgba(212,175,55,0.4);font-size:0.65rem;">
        Not financial advice — Educational purposes only
    </span>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
