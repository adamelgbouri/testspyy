"""
Commodity Trading Desk — standalone single-file Streamlit app
by Adam EL GBOURI

LIVE MARK-TO-MARKET BUILD — revision 2.

Every contract carries BOTH a live front-month settle AND a live dated forward strip.
No hardcoded marks. No cost-of-carry curves standing in for a market. No cross-exchange
proxies. If the feed dies, the screen says so rather than showing a fabricated price.

── What changed in revision 2 ─────────────────────────────────────────────────
DATA
  • One grouped download per data family: all front months (1 call), all dated strips
    across the whole board (1 call), one 15y history panel sliced everywhere (1 call).
    The signal scanner went from ~250 requests to 3.
  • Marks carry their settle DATE. A holiday-stale mark is shown dated, not hidden
    and not silently treated as today's. Freshness is part of honesty.
  • Every fetch failure is logged to an in-app diagnostics ring (sidebar) instead of
    being swallowed by a bare `except`.
CORRECTNESS
  • Strip tenor T is now CALENDAR time to delivery, not ordinal position/12. Roll
    yields and option forward anchoring were wrong for non-monthly cycles (GC, ZC…).
  • Per-contract expiry rules (CBOT grains expire mid delivery month, COMEX metals at
    end of delivery month, ICE Brent at end of M-2…). The old "20th of the preceding
    month" proxy dropped live fronts weeks early outside energy.
  • Options age: trade date is stored, remaining tenor is computed at every mark, an
    expired option marks at intrinsic. Theta now actually shows up in the P&L.
  • page_risk no longer mutates the blotter's vol in place (session-state bug).
RISK
  • Options enter BOTH VaR methods as Black-76 delta-cash — previously the parametric
    VaR treated them as full futures and the historical VaR ignored them entirely.
  • Missing pair correlations no longer get zero-filled (that silently assumed
    independence). The book falls back to the conservative sum and says why.
  • Historical VaR at h>1 days uses overlapping h-day windows instead of √h scaling —
    √h reintroduced the Gaussian assumption the method exists to avoid.
  • Stress episodes and parallel shocks fully revalue options (Black-76 at the
    shocked forward) instead of ignoring them.
MODELS
  • Monte Carlo now reverts to the LIVE FORWARD CURVE, not flat ln(spot): paths are
    centred so that E[S_t] = F(t) read off the strip. Seasonality in the curve (NG
    winters, RB driving season) propagates into the fan. Exact OU discretisation.
BOOK
  • Blotter is per-book, keyed by a `?book=` id in the URL — on Streamlit Cloud the
    old single blotter.json was shared by every visitor and wiped on redeploy.
    Export/import JSON remains the durable mechanism.
  • Dated-contract booking: a future can be booked on a specific strip month and is
    marked to that dated ticker, so P&L is clean through rolls.
ENGINEERING
  • Strict registry validation (unknown/missing keys and bad types fail at import —
    a typo'd `price_divisor` used to silently cost a ×100 notional error).
  • API keys can come from st.secrets (EIA_KEY / FRED_KEY) as well as the sidebar.
  • Pure analytics are covered by test_desk.py (pytest, no network needed).
  • Calendar: weekly prints (EIA Wed/Thu, rigs Fri) are computed; everything else is
    labelled approximate instead of being invented from today+N.

Run:
    pip install -r requirements.txt
    streamlit run desk.py
Test:
    pytest test_desk.py -q

EIA fundamentals (optional): set an API key in the sidebar or st.secrets["EIA_KEY"].
Free at eia.gov/opendata. FRED likewise: fred.stlouisfed.org → st.secrets["FRED_KEY"].
"""
from __future__ import annotations

import json
import logging
import math
import os
import uuid
from calendar import monthrange
from collections import deque
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING — every feed failure lands here, visible in the sidebar diagnostics
# ══════════════════════════════════════════════════════════════════════════════
FEED_LOG: deque = deque(maxlen=80)   # module-level: survives Streamlit reruns


class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            FEED_LOG.append(f"{datetime.now():%H:%M:%S}  {record.levelname:<7} {record.getMessage()}")
        except Exception:
            pass


LOG = logging.getLogger("desk")
if not any(isinstance(h, _RingHandler) for h in LOG.handlers):
    LOG.addHandler(_RingHandler())
    LOG.setLevel(logging.INFO)

# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE
# ══════════════════════════════════════════════════════════════════════════════
AMBER  = "#F0A500"
BLUE   = "#58A6FF"
GREEN  = "#3FB950"
RED    = "#FF7B72"
GRAY   = "#8B949E"
PURPLE = "#BC8CFF"
TEAL   = "#39D0D8"
PANEL  = "#161B22"
BG     = "#0D1117"
BORDER = "#30363D"
TEXT   = "#E6EDF3"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
.stApp {{ background-color:{BG}; color:{TEXT}; font-family:'Inter',system-ui; }}
section[data-testid="stSidebar"] {{ background-color:{PANEL}; border-right:1px solid {BORDER}; }}
h1,h2,h3,h4 {{ color:{TEXT}; font-family:'Inter',system-ui; letter-spacing:-0.02em; }}
[data-testid="stMetricLabel"] {{ color:{GRAY}; font-size:11px; font-family:'JetBrains Mono',monospace;
    letter-spacing:0.12em; text-transform:uppercase; }}
[data-testid="stMetricValue"] {{ color:{TEXT}; font-family:'JetBrains Mono',monospace; font-size:1.4rem; }}
[data-testid="stMetricDelta"] {{ font-family:'JetBrains Mono',monospace; font-size:0.75rem; }}
.stMetric {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:14px 16px; }}
div[data-testid="stHorizontalBlock"] {{ gap:10px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; background:{PANEL}; border-radius:8px; padding:4px; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; color:{GRAY}; border-radius:6px; padding:6px 16px;
    font-family:'JetBrains Mono',monospace; font-size:0.78rem; letter-spacing:0.05em; }}
.stTabs [aria-selected="true"] {{ background:{AMBER} !important; color:{BG} !important; font-weight:700; }}
.badge {{ display:inline-block; padding:2px 8px; border:1px solid {BORDER}; border-radius:5px;
    font-size:10px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase;
    color:{TEXT}; background:{PANEL}; font-family:'JetBrains Mono',monospace; margin-right:4px; }}
.badge-amber {{ border-color:rgba(240,165,0,0.5); color:{AMBER}; }}
.badge-green {{ border-color:rgba(63,185,80,0.5); color:{GREEN}; }}
.badge-red   {{ border-color:rgba(255,123,114,0.5); color:{RED}; }}
.kpi-card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:14px 16px;
    border-left:3px solid {AMBER}; }}
.kpi-label {{ font-size:10px; color:{GRAY}; font-family:'JetBrains Mono',monospace;
    text-transform:uppercase; letter-spacing:0.12em; margin-bottom:4px; }}
.kpi-value {{ font-size:1.3rem; color:{TEXT}; font-family:'JetBrains Mono',monospace; font-weight:600; }}
.kpi-sub   {{ font-size:0.68rem; color:{GRAY}; margin-top:3px; }}
hr {{ border-color:{BORDER}; }}
/* ── sectioned sidebar navigation ── */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{ gap:0.15rem; }}
.nav-sec {{ font-size:9px; color:{GRAY}; font-family:'JetBrains Mono',monospace;
    letter-spacing:0.2em; text-transform:uppercase; margin:14px 0 3px 4px;
    border-bottom:1px solid {BORDER}; padding-bottom:3px; }}
.nav-active {{ background:rgba(240,165,0,0.13); border-left:2px solid {AMBER};
    color:{AMBER}; padding:5px 10px; border-radius:0 6px 6px 0;
    font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:600; }}
section[data-testid="stSidebar"] button[kind="tertiary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {{
    justify-content:flex-start; text-align:left; color:{GRAY};
    font-family:'JetBrains Mono',monospace; font-size:0.78rem;
    padding:4px 10px; min-height:0; border-radius:6px; }}
section[data-testid="stSidebar"] button[kind="tertiary"]:hover,
section[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover {{
    color:{AMBER}; background:rgba(240,165,0,0.07); }}
</style>
"""


def _setup_page() -> None:
    """Page config + CSS. Called from main() so the module stays import-safe for tests."""
    st.set_page_config(page_title="S&D — Commodity Trading Desk", page_icon="🌐",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONTRACT REGISTRY — LIVE MARK-TO-MARKET ONLY
# ══════════════════════════════════════════════════════════════════════════════
#  Inclusion rule (enforced by the validator below):
#    yf_ticker — continuous front month, drives the mark
#    yf_fmt    — dated contract template, drives the forward strip
#  Missing either => the contract cannot be marked honestly => it is not on this desk.
#
#  contract_size / price_divisor convert a price move into cash.
#  Grains, softs and livestock quote in CENTS: price_divisor=100.
#  bbl_conv: gallons->barrels factor for the crack stack (42 gal = 1 bbl).
#
#  expiry_rule — approximate last trading day, per contract:
#    prec_25    ~25th of the month PRECEDING delivery      (CL: 25th − 3 bd)
#    prec_eom   end of the month preceding delivery        (NG/RB/HO/SB)
#    prec2_eom  end of the SECOND month before delivery    (ICE Brent)
#    del_15     ~15th of the DELIVERY month                (CBOT grains, CC, HE)
#    del_20     ~20th of the delivery month                (KC)
#    del_eom    end of the delivery month                  (COMEX metals, LE)
#  Estimates are deliberately GENEROUS (kept a few days past true expiry): a dead
#  ticker simply returns nothing and drops out of the strip, whereas dropping a live
#  front early — the old single-rule proxy did this for every non-energy contract —
#  silently shifts the whole curve.
#
#  mr_halflife: mean-reversion half-life in YEARS, used by the simulator.
#    Storables with tight inventory linkage revert fast (gas). Precious metals barely
#    revert at all — they behave closer to a financial asset. None => GBM.
#
#  EXCLUDED and why (all need a paid feed to return):
#    LME Copper       — was proxied off HG=F (COMEX, $/lb) but labelled $/mt. ~2200x error.
#    ICE Gasoil       — live front (LGO=F) but no dated strip. Curve was a model.
#    EUA / Coal API2  — no feed. Marks were constants.
#    LME Alu / Nickel — no feed. Marks were constants.
#    Capesize/Panamax — no feed, and cost-of-carry on a non-storable TC route is meaningless.
# ══════════════════════════════════════════════════════════════════════════════
COMMODITIES: Dict[str, dict] = {
    # ── Energy ────────────────────────────────────────────────────────────────
    "WTI Crude (CL)": dict(
        sector="Energy", exchange="NYMEX", unit="$/bbl",
        yf_ticker="CL=F", yf_fmt="CL{M}{YY}.NYM", expiry_rule="prec_25",
        active_months="FGHJKMNQUVXZ", liquid_months=18,
        contract_size=1_000, size_unit="bbl", bbl_conv=1.0,
        vol=0.32, mr_halflife=2.0, ticker="CL",
        reg_unit="mb/d", reg_label="Million barrels per day",
    ),
    "Brent Crude (BZ)": dict(
        sector="Energy", exchange="ICE", unit="$/bbl",
        yf_ticker="BZ=F", yf_fmt="BZ{M}{YY}.NYM", expiry_rule="prec2_eom",
        active_months="FGHJKMNQUVXZ", liquid_months=18,
        contract_size=1_000, size_unit="bbl", bbl_conv=1.0,
        vol=0.30, mr_halflife=2.0, ticker="BZ",
        reg_unit="mb/d", reg_label="Million barrels per day",
    ),
    "Henry Hub Nat Gas (NG)": dict(
        sector="Energy", exchange="NYMEX", unit="$/MMBtu",
        yf_ticker="NG=F", yf_fmt="NG{M}{YY}.NYM", expiry_rule="prec_eom",
        active_months="FGHJKMNQUVXZ", liquid_months=12,
        contract_size=10_000, size_unit="MMBtu",
        vol=0.55, mr_halflife=0.75, ticker="NG", seasonal=True,
        reg_unit="bcf/d", reg_label="Billion cubic feet per day",
    ),
    "RBOB Gasoline (RB)": dict(
        sector="Energy", exchange="NYMEX", unit="$/gal",
        yf_ticker="RB=F", yf_fmt="RB{M}{YY}.NYM", expiry_rule="prec_eom",
        active_months="FGHJKMNQUVXZ", liquid_months=12,
        contract_size=42_000, size_unit="gal", bbl_conv=42.0,
        vol=0.36, mr_halflife=1.5, ticker="RB", seasonal=True,
        reg_unit="mb/d", reg_label="Million barrels per day",
    ),
    "ULSD Heating Oil (HO)": dict(
        sector="Energy", exchange="NYMEX", unit="$/gal",
        yf_ticker="HO=F", yf_fmt="HO{M}{YY}.NYM", expiry_rule="prec_eom",
        active_months="FGHJKMNQUVXZ", liquid_months=12,
        contract_size=42_000, size_unit="gal", bbl_conv=42.0,
        vol=0.34, mr_halflife=1.5, ticker="HO", seasonal=True,
        reg_unit="mb/d", reg_label="Million barrels per day",
    ),
    # ── Metals ────────────────────────────────────────────────────────────────
    "Gold (GC)": dict(
        sector="Metals", exchange="COMEX", unit="$/troy oz",
        yf_ticker="GC=F", yf_fmt="GC{M}{YY}.CMX", expiry_rule="del_eom",
        active_months="GJMQVZ", liquid_months=8,
        contract_size=100, size_unit="troy oz",
        vol=0.15, mr_halflife=None, ticker="GC",
        reg_unit="t/y", reg_label="Tonnes per year",
    ),
    "Silver (SI)": dict(
        sector="Metals", exchange="COMEX", unit="$/troy oz",
        yf_ticker="SI=F", yf_fmt="SI{M}{YY}.CMX", expiry_rule="del_eom",
        active_months="HKNUZ", liquid_months=6,
        contract_size=5_000, size_unit="troy oz",
        vol=0.28, mr_halflife=None, ticker="SI",
        reg_unit="Moz/y", reg_label="Million troy oz per year",
    ),
    "Copper (HG)": dict(
        sector="Metals", exchange="COMEX", unit="$/lb",
        yf_ticker="HG=F", yf_fmt="HG{M}{YY}.CMX", expiry_rule="del_eom",
        active_months="HKNUZ", liquid_months=8,
        contract_size=25_000, size_unit="lb",
        vol=0.22, mr_halflife=3.0, ticker="HG",
        reg_unit="kt/y", reg_label="Thousand tonnes per year",
    ),
    "Platinum (PL)": dict(
        sector="Metals", exchange="NYMEX", unit="$/troy oz",
        yf_ticker="PL=F", yf_fmt="PL{M}{YY}.NYM", expiry_rule="del_eom",
        active_months="FJNV", liquid_months=6,
        contract_size=50, size_unit="troy oz",
        vol=0.20, mr_halflife=None, ticker="PL",
        reg_unit="Moz/y", reg_label="Million troy oz per year",
    ),
    "Palladium (PA)": dict(
        sector="Metals", exchange="NYMEX", unit="$/troy oz",
        yf_ticker="PA=F", yf_fmt="PA{M}{YY}.NYM", expiry_rule="del_eom",
        active_months="HMUZ", liquid_months=6,
        contract_size=100, size_unit="troy oz",
        vol=0.30, mr_halflife=None, ticker="PA",
        reg_unit="Moz/y", reg_label="Million troy oz per year",
    ),
    # ── Grains & Oilseeds ─────────────────────────────────────────────────────
    "Corn (ZC)": dict(
        sector="Grains", exchange="CBOT", unit="c/bu",
        yf_ticker="ZC=F", yf_fmt="ZC{M}{YY}.CBT", expiry_rule="del_15",
        active_months="HKNUZ", liquid_months=8,
        contract_size=5_000, size_unit="bu", price_divisor=100.0,
        vol=0.25, mr_halflife=1.5, ticker="ZC", seasonal=True,
        reg_unit="Mbu/y", reg_label="Million bushels per year",
    ),
    "Wheat CBOT SRW (ZW)": dict(
        sector="Grains", exchange="CBOT", unit="c/bu",
        yf_ticker="ZW=F", yf_fmt="ZW{M}{YY}.CBT", expiry_rule="del_15",
        active_months="HKNUZ", liquid_months=8,
        contract_size=5_000, size_unit="bu", price_divisor=100.0,
        vol=0.28, mr_halflife=1.5, ticker="ZW", seasonal=True,
        reg_unit="Mbu/y", reg_label="Million bushels per year",
    ),
    "Soybeans (ZS)": dict(
        sector="Grains", exchange="CBOT", unit="c/bu",
        yf_ticker="ZS=F", yf_fmt="ZS{M}{YY}.CBT", expiry_rule="del_15",
        active_months="FHKNQUX", liquid_months=8,
        contract_size=5_000, size_unit="bu", price_divisor=100.0,
        vol=0.23, mr_halflife=1.5, ticker="ZS", seasonal=True,
        reg_unit="Mbu/y", reg_label="Million bushels per year",
    ),
    "Soybean Meal (ZM)": dict(
        sector="Grains", exchange="CBOT", unit="$/short ton",
        yf_ticker="ZM=F", yf_fmt="ZM{M}{YY}.CBT", expiry_rule="del_15",
        active_months="FHKNQUVZ", liquid_months=8,
        contract_size=100, size_unit="short ton",
        vol=0.26, mr_halflife=1.5, ticker="ZM", seasonal=True,
        reg_unit="kt/y", reg_label="Thousand tonnes per year",
    ),
    "Soybean Oil (ZL)": dict(
        sector="Grains", exchange="CBOT", unit="c/lb",
        yf_ticker="ZL=F", yf_fmt="ZL{M}{YY}.CBT", expiry_rule="del_15",
        active_months="FHKNQUVZ", liquid_months=8,
        contract_size=60_000, size_unit="lb", price_divisor=100.0,
        vol=0.30, mr_halflife=1.5, ticker="ZL", seasonal=True,
        reg_unit="kt/y", reg_label="Thousand tonnes per year",
    ),
    # ── Softs ─────────────────────────────────────────────────────────────────
    "Sugar #11 (SB)": dict(
        sector="Softs", exchange="ICE US", unit="c/lb",
        yf_ticker="SB=F", yf_fmt="SB{M}{YY}.NYB", expiry_rule="prec_eom",
        active_months="HKNV", liquid_months=6,
        contract_size=112_000, size_unit="lb", price_divisor=100.0,
        vol=0.30, mr_halflife=2.0, ticker="SB",
        reg_unit="Mt/y", reg_label="Million tonnes per year",
    ),
    "Arabica Coffee (KC)": dict(
        sector="Softs", exchange="ICE US", unit="c/lb",
        yf_ticker="KC=F", yf_fmt="KC{M}{YY}.NYB", expiry_rule="del_20",
        active_months="HKNUZ", liquid_months=6,
        contract_size=37_500, size_unit="lb", price_divisor=100.0,
        vol=0.35, mr_halflife=2.0, ticker="KC",
        reg_unit="M bags/y", reg_label="Million 60-kg bags per year",
    ),
    "Cocoa (CC)": dict(
        sector="Softs", exchange="ICE US", unit="$/mt",
        yf_ticker="CC=F", yf_fmt="CC{M}{YY}.NYB", expiry_rule="del_15",
        active_months="HKNUZ", liquid_months=6,
        contract_size=10, size_unit="mt",
        vol=0.32, mr_halflife=2.0, ticker="CC",
        reg_unit="kt/y", reg_label="Thousand tonnes per year",
    ),
    # ── Livestock ─────────────────────────────────────────────────────────────
    "Live Cattle (LE)": dict(
        sector="Livestock", exchange="CME", unit="c/lb",
        yf_ticker="LE=F", yf_fmt="LE{M}{YY}.CME", expiry_rule="del_eom",
        active_months="GJMQVZ", liquid_months=8,
        contract_size=40_000, size_unit="lb", price_divisor=100.0,
        vol=0.18, mr_halflife=1.0, ticker="LE", seasonal=True,
        reg_unit="Mlb/y", reg_label="Million pounds per year",
    ),
    "Lean Hogs (HE)": dict(
        sector="Livestock", exchange="CME", unit="c/lb",
        yf_ticker="HE=F", yf_fmt="HE{M}{YY}.CME", expiry_rule="del_15",
        active_months="GJKMNQVZ", liquid_months=6,
        contract_size=40_000, size_unit="lb", price_divisor=100.0,
        vol=0.25, mr_halflife=0.75, ticker="HE", seasonal=True,
        reg_unit="Mlb/y", reg_label="Million pounds per year",
    ),
}

# ── Registry validation — fail loudly at import rather than ship a bad mark ──
_EXPIRY_RULES = {"prec_25", "prec_eom", "prec2_eom", "del_15", "del_20", "del_eom"}
_REQUIRED_KEYS = {"sector", "exchange", "unit", "yf_ticker", "yf_fmt", "expiry_rule",
                  "active_months", "liquid_months", "contract_size", "size_unit",
                  "vol", "ticker", "reg_unit", "reg_label"}
_OPTIONAL_KEYS = {"price_divisor", "bbl_conv", "mr_halflife", "seasonal"}
MONTH_CODES = list("FGHJKMNQUVXZ")
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _validate_registry(reg: Dict[str, dict]) -> None:
    """A typo'd optional key (price_divsor…) used to vanish into .get(default) and
    mis-scale notionals by 100x. Unknown keys, missing keys and bad types now stop
    the app at import with a message naming the contract."""
    for name, c in reg.items():
        keys = set(c)
        unknown = keys - _REQUIRED_KEYS - _OPTIONAL_KEYS
        assert not unknown, f"{name}: unknown registry key(s) {sorted(unknown)} — typo?"
        missing = _REQUIRED_KEYS - keys
        assert not missing, f"{name}: missing required key(s) {sorted(missing)}"
        assert c["expiry_rule"] in _EXPIRY_RULES, f"{name}: bad expiry_rule {c['expiry_rule']!r}"
        assert c["yf_ticker"] and c["yf_fmt"], f"{name}: no live feed — cannot be on this desk"
        assert set(c["active_months"]) <= set(MONTH_CODES), f"{name}: bad month code"
        assert isinstance(c["liquid_months"], int) and c["liquid_months"] > 0, f"{name}: liquid_months"
        assert c["contract_size"] > 0, f"{name}: contract_size"
        assert 0.01 < c["vol"] < 3.0, f"{name}: vol {c['vol']} out of sane range"
        if "price_divisor" in c:
            assert c["price_divisor"] in (100.0,), f"{name}: unexpected price_divisor"
        if c.get("mr_halflife") is not None:
            assert 0.05 < c["mr_halflife"] < 20, f"{name}: mr_halflife out of range"


_validate_registry(COMMODITIES)

ALL_SECTORS = sorted({v["sector"] for v in COMMODITIES.values()})
YF_TICKERS  = {n: c["yf_ticker"] for n, c in COMMODITIES.items()}

# ── Structural spread definitions ────────────────────────────────────────────
# Legs are (contract, ratio). Ratio sign = long(+) / short(-) in the structure.
STRUCTURES = {
    "3-2-1 Crack Spread": dict(
        kind="crack",
        legs=[("RBOB Gasoline (RB)", 2), ("ULSD Heating Oil (HO)", 1), ("WTI Crude (CL)", -3)],
        divisor=3, unit="$/bbl",
        desc=("Refining margin. Three barrels of crude in, two of gasoline and one of "
              "distillate out. RB and HO quote in $/gal and are converted at 42 gal/bbl. "
              "Long the crack = long the refiner."),
        typical=(5, 35),
    ),
    "Gasoline Crack (RB-CL)": dict(
        kind="crack",
        legs=[("RBOB Gasoline (RB)", 1), ("WTI Crude (CL)", -1)],
        divisor=1, unit="$/bbl",
        desc="Single-product gasoline crack. Peaks into US driving season, collapses in Q4.",
        typical=(2, 40),
    ),
    "Distillate Crack (HO-CL)": dict(
        kind="crack",
        legs=[("ULSD Heating Oil (HO)", 1), ("WTI Crude (CL)", -1)],
        divisor=1, unit="$/bbl",
        desc="Diesel/heating oil crack. Winter-weighted; driven by distillate stocks.",
        typical=(5, 50),
    ),
    "Board Crush (ZS/ZM/ZL)": dict(
        kind="crush",
        legs=[("Soybean Meal (ZM)", 1), ("Soybean Oil (ZL)", 1), ("Soybeans (ZS)", -1)],
        divisor=1, unit="$/bu",
        desc=("Processor margin. One bushel of beans yields ~44 lb meal and ~11 lb oil. "
              "Long the crush = long the crusher. The live print pairs matched delivery "
              "months across the three strips when available."),
        typical=(0.4, 2.5),
    ),
    "WTI-Brent Arb": dict(
        kind="simple",
        legs=[("WTI Crude (CL)", 1), ("Brent Crude (BZ)", -1)],
        divisor=1, unit="$/bbl",
        desc="Transatlantic arb. Drives US export economics. Historically -1 to -8.",
        typical=(-8, 0),
    ),
    "Gold-Silver Ratio": dict(
        kind="ratio",
        legs=[("Gold (GC)", 1), ("Silver (SI)", 1)],
        divisor=1, unit="ratio",
        desc="Classic precious relative-value gauge. Ratio, not a spread.",
        typical=(60, 95),
    ),
}

# Soybean crush yields: 1 bu (60 lb) -> ~44 lb meal + ~11 lb oil
CRUSH_MEAL_LB = 44.0
CRUSH_OIL_LB  = 11.0
LB_PER_SHORT_TON = 2000.0


def notional_per_lot(commodity: str, price: float) -> float:
    """Cash value of one lot. Handles cent-quoted contracts."""
    c = COMMODITIES[commodity]
    return price / c.get("price_divisor", 1.0) * c["contract_size"]


def price_multiplier(commodity: str) -> float:
    """Cash P&L per 1.0 move in the quoted price, per lot."""
    c = COMMODITIES[commodity]
    return c["contract_size"] / c.get("price_divisor", 1.0)


def to_bbl(commodity: str, price: float) -> float:
    """Normalise a refined-product price to $/bbl for the crack stack."""
    return price * COMMODITIES[commodity].get("bbl_conv", 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  CONTRACT CALENDAR — expiry estimates and CALENDAR tenor
# ══════════════════════════════════════════════════════════════════════════════
def _eom(y: int, m: int) -> date:
    return date(y, m, monthrange(y, m)[1])


def _shift_month(y: int, m: int, k: int) -> Tuple[int, int]:
    idx = (y * 12 + (m - 1)) + k
    return idx // 12, idx % 12 + 1


def estimate_expiry(rule: str, dy: int, dm: int) -> date:
    """Approximate LAST TRADING DAY for a contract delivering in (dy, dm).
    Deliberately generous — see the registry note on cost asymmetry."""
    if rule == "prec_25":
        y, m = _shift_month(dy, dm, -1)
        return date(y, m, 25)
    if rule == "prec_eom":
        y, m = _shift_month(dy, dm, -1)
        return _eom(y, m)
    if rule == "prec2_eom":
        y, m = _shift_month(dy, dm, -2)
        return _eom(y, m)
    if rule == "del_15":
        return date(dy, dm, 15)
    if rule == "del_20":
        return date(dy, dm, 20)
    if rule == "del_eom":
        return _eom(dy, dm)
    raise ValueError(f"unknown expiry rule {rule!r}")


def strip_contract_specs(commodity: str, today: Optional[date] = None) -> List[dict]:
    """Pure builder for the dated strip: exchange codes, delivery months and CALENDAR
    tenor. No network. T is the year-fraction to mid-delivery (15th), which is what
    annualised roll yields and option anchoring actually need — the old T = seq/12
    was wrong for every non-monthly cycle (GC's 5th contract is ~10 months out, not 5/12)."""
    today = today or date.today()
    c = COMMODITIES[commodity]
    specs: List[dict] = []
    offset = 0
    # Cap generous enough for sparse cycles (PL trades 4 delivery months a year).
    while len(specs) < c["liquid_months"] and offset < 60:
        m0 = (today.month - 1 + offset) % 12
        dy = today.year + (today.month - 1 + offset) // 12
        offset += 1
        dm = m0 + 1
        if MONTH_CODES[m0] not in c["active_months"]:
            continue
        if today > estimate_expiry(c["expiry_rule"], dy, dm):
            continue
        T = max((date(dy, dm, 15) - today).days, 7) / 365.25
        specs.append(dict(
            label=f"{MONTH_NAMES[m0]}-{dy}",
            month=len(specs) + 1,
            T=round(T, 4),
            delivery=f"{dy}-{dm:02d}",
            ticker=c["yf_fmt"].replace("{M}", MONTH_CODES[m0]).replace("{YY}", str(dy)[-2:]),
        ))
    return specs


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE DATA LAYER — no fallbacks, no fabrication, one grouped call per family
# ══════════════════════════════════════════════════════════════════════════════
class MarkBoard:
    """Front-month marks with their settle DATES. Dict-like on prices so the rest
    of the code reads naturally; .asof(name) exposes freshness. A mark from a prior
    session/holiday is shown dated rather than hidden — a dated settle is honest,
    a hidden stale one is not."""

    STALE_DAYS = 4  # weekend + a Monday holiday

    def __init__(self, prices: Dict[str, Optional[float]],
                 asof: Dict[str, Optional[date]]):
        self.prices = prices
        self._asof = asof

    def get(self, name: str, default=None):
        return self.prices.get(name, default)

    def __getitem__(self, name: str):
        return self.prices[name]

    def items(self):
        return self.prices.items()

    def values(self):
        return self.prices.values()

    def asof(self, name: str) -> Optional[date]:
        return self._asof.get(name)

    def is_stale(self, name: str) -> bool:
        d = self._asof.get(name)
        return d is not None and (date.today() - d).days > self.STALE_DAYS

    def stale_names(self) -> List[str]:
        return [n for n in self.prices if self.prices[n] is not None and self.is_stale(n)]


def _yf_closes(tickers: Sequence[str], period: str) -> pd.DataFrame:
    """Low-level grouped download -> Close matrix (columns = tickers). Every failure
    is logged to the diagnostics ring instead of being silently swallowed."""
    if not YF_AVAILABLE or not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(list(tickers), period=period, auto_adjust=True,
                          progress=False, threads=True)
        if raw is None or raw.empty:
            LOG.warning("yf.download returned empty for %d ticker(s) [%s]", len(tickers), period)
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]]
            closes.columns = [list(tickers)[0]]
        closes.index = pd.to_datetime(closes.index)
        return closes
    except Exception as e:  # yfinance raises a zoo of types; log, never fabricate
        LOG.warning("yf.download failed (%d tickers, %s): %s", len(tickers), period, e)
        return pd.DataFrame()


def _last_valid(closes: pd.DataFrame, col: str) -> Tuple[Optional[float], Optional[date]]:
    if col not in closes.columns:
        return None, None
    s = closes[col].dropna()
    if s.empty:
        return None, None
    return float(s.iloc[-1]), s.index[-1].date()


@st.cache_data(ttl=300)
def _fetch_live_marks_raw() -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[str]]]:
    closes = _yf_closes(list(YF_TICKERS.values()), period="10d")
    prices: Dict[str, Optional[float]] = {}
    asof: Dict[str, Optional[str]] = {}
    for n, t in YF_TICKERS.items():
        p, d = _last_valid(closes, t)
        prices[n] = p
        asof[n] = d.isoformat() if d else None
    return prices, asof


def fetch_live_marks() -> MarkBoard:
    prices, asof_iso = _fetch_live_marks_raw()
    asof = {n: (date.fromisoformat(d) if d else None) for n, d in asof_iso.items()}
    return MarkBoard(prices, asof)


# ── History panel: ONE 15y grouped download, sliced everywhere ───────────────
PANEL_YEARS_MAX = 15


@st.cache_data(ttl=3600, persist="disk")
def fetch_panel_max() -> pd.DataFrame:
    """Aligned close panel for the whole board, max depth, downloaded ONCE.
    Correlation, structures, seasonality, momentum and dashboards all slice this."""
    closes = _yf_closes(list(YF_TICKERS.values()), period=f"{PANEL_YEARS_MAX}y")
    if closes.empty:
        return pd.DataFrame()
    inv = {v: k for k, v in YF_TICKERS.items()}
    closes = closes.rename(columns=inv)
    keep = [c for c in closes.columns if c in COMMODITIES]
    return closes[keep].dropna(how="all")


def panel_years(years: float) -> pd.DataFrame:
    panel = fetch_panel_max()
    if panel.empty:
        return panel
    cutoff = pd.Timestamp(datetime.now()) - pd.DateOffset(days=int(years * 365.25))
    return panel[panel.index >= cutoff]


def realised_vol(commodity: str, window: int = 60) -> Optional[float]:
    """Annualised close-to-close realised vol, off the shared panel (no extra call)."""
    panel = panel_years(1.2)
    if panel.empty or commodity not in panel.columns:
        return None
    s = panel[commodity].dropna()
    if len(s) < window + 1:
        return None
    lr = np.log(s / s.shift(1)).dropna()
    return float(lr.tail(window).std() * math.sqrt(252))


@st.cache_data(ttl=3600)
def correlation_matrix(years: float = 2, window: int = 252) -> pd.DataFrame:
    """Correlation of daily log returns across the board. This is what turns the VaR
    from an undiversified sum into a real portfolio number."""
    panel = panel_years(years)
    if panel.empty or len(panel) < 30:
        return pd.DataFrame()
    lr = np.log(panel / panel.shift(1)).dropna(how="all")
    lr = lr.tail(window)
    return lr.corr(min_periods=30)


# ── Forward strips: ONE grouped download for every dated contract on the board ─
@st.cache_data(ttl=1800, persist="disk")
def fetch_all_strips() -> Dict[str, pd.DataFrame]:
    """Live dated forward strips for the whole board in a single grouped download
    (~150-200 tickers). Empty frame per contract => no curve is drawn — this desk
    will not fit a model and call it a market. Each row carries the settle's asof
    date: thin deferred months often last traded a day or two ago, and that date is
    part of the mark."""
    specs = {name: strip_contract_specs(name) for name in COMMODITIES}
    all_tickers = [k["ticker"] for rows in specs.values() for k in rows]
    closes = _yf_closes(all_tickers, period="10d")
    out: Dict[str, pd.DataFrame] = {}
    for name, rows in specs.items():
        recs = []
        for k in rows:
            p, d = _last_valid(closes, k["ticker"])
            if p is not None:
                recs.append(dict(label=k["label"], month=len(recs) + 1, T=k["T"],
                                 delivery=k["delivery"], price=round(p, 4),
                                 ticker=k["ticker"], asof=d.isoformat()))
        out[name] = pd.DataFrame(recs)
        if not recs:
            LOG.warning("strip empty: %s", name)
    return out


def fetch_forward_strip(commodity: str) -> pd.DataFrame:
    return fetch_all_strips().get(commodity, pd.DataFrame())


def dated_mark(ticker: str) -> Optional[dict]:
    """Mark for a specific dated contract already on a strip (used by dated blotter
    lines). None if it has rolled off — the line then errors rather than proxying."""
    for name, df in fetch_all_strips().items():
        if df.empty:
            continue
        hit = df[df["ticker"] == ticker]
        if not hit.empty:
            r = hit.iloc[0]
            return dict(commodity=name, price=float(r["price"]), label=str(r["label"]),
                        T=float(r["T"]), asof=str(r["asof"]))
    return None


@st.cache_data(ttl=3600)
def fetch_pair_history(t1: str, t2: str, period: str = "2y") -> pd.DataFrame:
    """History of two dated tickers, one grouped call."""
    closes = _yf_closes([t1, t2], period=period)
    if closes.empty or t1 not in closes.columns or t2 not in closes.columns:
        return pd.DataFrame()
    df = pd.DataFrame({"near": closes[t1], "far": closes[t2]}).dropna()
    return df


def fetch_spread_history(commodity: str, m1_offset: int = 0, m2_offset: int = 1,
                         period: str = "2y") -> pd.DataFrame:
    """Calendar-spread history by tracking two specific dated contracts through time.
    A single M1-M2 print is a point; its 2y percentile is what says whether it is cheap."""
    strip = fetch_forward_strip(commodity)
    if strip.empty or len(strip) <= max(m1_offset, m2_offset):
        return pd.DataFrame()
    t1 = strip["ticker"].iloc[m1_offset]
    t2 = strip["ticker"].iloc[m2_offset]
    df = fetch_pair_history(t1, t2, period)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["spread"] = df["near"] - df["far"]
    df.attrs["near_label"] = strip["label"].iloc[m1_offset]
    df.attrs["far_label"]  = strip["label"].iloc[m2_offset]
    return df


@st.cache_data(ttl=3600)
def fetch_structure_history(structure: str, years: float = 3) -> pd.DataFrame:
    """History of a crack / crush / arb, computed off the continuous front months.
    Everything is normalised to a common unit before the legs are combined.
    NOTE — the continuous series are NOT roll-adjusted (Yahoo limitation): each roll
    injects a level jump, so treat month-boundary wiggles with suspicion."""
    spec  = STRUCTURES[structure]
    panel = panel_years(years)
    if panel.empty:
        return pd.DataFrame()

    legs = [n for n, _ in spec["legs"]]
    if any(l not in panel.columns for l in legs):
        return pd.DataFrame()
    df = panel[legs].dropna()
    if df.empty:
        return pd.DataFrame()

    kind = spec["kind"]
    if kind == "crack":
        val = 0.0
        for name, ratio in spec["legs"]:
            val = val + ratio * df[name] * COMMODITIES[name].get("bbl_conv", 1.0)
        out = val / spec["divisor"]
    elif kind == "crush":
        meal = df["Soybean Meal (ZM)"] * CRUSH_MEAL_LB / LB_PER_SHORT_TON
        oil  = df["Soybean Oil (ZL)"] / 100.0 * CRUSH_OIL_LB
        bean = df["Soybeans (ZS)"] / 100.0
        out  = meal + oil - bean
    elif kind == "ratio":
        a, b = spec["legs"][0][0], spec["legs"][1][0]
        out = df[a] / df[b]
    else:  # simple
        val = 0.0
        for name, ratio in spec["legs"]:
            val = val + ratio * df[name]
        out = val / spec["divisor"]

    return pd.DataFrame({"value": out}).dropna()


def matched_month_crush(strips: Dict[str, pd.DataFrame]) -> Optional[dict]:
    """The real board crush pairs the SAME delivery month across ZS/ZM/ZL — the
    continuous fronts do not always coincide (bean and product cycles differ).
    Returns the nearest common delivery month, or None to fall back to fronts."""
    need = ["Soybeans (ZS)", "Soybean Meal (ZM)", "Soybean Oil (ZL)"]
    dfs = {n: strips.get(n, pd.DataFrame()) for n in need}
    if any(d.empty for d in dfs.values()):
        return None
    common = set(dfs[need[0]]["delivery"])
    for n in need[1:]:
        common &= set(dfs[n]["delivery"])
    if not common:
        return None
    month = sorted(common)[0]
    px = {}
    for n in need:
        row = dfs[n][dfs[n]["delivery"] == month].iloc[0]
        px[n] = float(row["price"])
    label = dfs[need[0]][dfs[need[0]]["delivery"] == month].iloc[0]["label"]
    meal = px["Soybean Meal (ZM)"] * CRUSH_MEAL_LB / LB_PER_SHORT_TON
    oil  = px["Soybean Oil (ZL)"] / 100.0 * CRUSH_OIL_LB
    bean = px["Soybeans (ZS)"] / 100.0
    return dict(value=meal + oil - bean, label=str(label), legs=px)


@st.cache_data(ttl=3600)
def fetch_board_closes(d_a: date, d_b: date) -> pd.DataFrame:
    """Settles for the whole board at two dates — ONE grouped download instead of the
    old two-requests-per-contract (~40 calls)."""
    start = (datetime.combine(min(d_a, d_b), datetime.min.time()) - timedelta(days=12))
    end   = (datetime.combine(max(d_a, d_b), datetime.min.time()) + timedelta(days=1))
    if not YF_AVAILABLE:
        return pd.DataFrame()
    try:
        raw = yf.download(list(YF_TICKERS.values()),
                          start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False, threads=True)
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    except Exception as e:
        LOG.warning("board closes download failed: %s", e)
        return pd.DataFrame()
    rows = []
    for n, t in YF_TICKERS.items():
        if t not in closes.columns:
            continue
        s = closes[t].dropna()
        sa = s[s.index.date <= d_a]
        sb = s[s.index.date <= d_b]
        if sa.empty or sb.empty:
            continue
        pa, pb = float(sa.iloc[-1]), float(sb.iloc[-1])
        if pa > 0:
            rows.append(dict(name=n, sector=COMMODITIES[n]["sector"], px=round(pb, 2),
                             chg=(pb - pa) / pa * 100, chg_str=f"{(pb - pa) / pa * 100:+.2f}%"))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  API KEYS — sidebar input OR st.secrets (EIA_KEY / FRED_KEY)
# ══════════════════════════════════════════════════════════════════════════════
def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


def eia_key() -> str:
    return st.session_state.get("eia_key", "") or _secret("EIA_KEY")


def fred_key() -> str:
    return st.session_state.get("fred_key", "") or _secret("FRED_KEY")


# ══════════════════════════════════════════════════════════════════════════════
#  EIA FUNDAMENTALS  (free API key at eia.gov/opendata)
# ══════════════════════════════════════════════════════════════════════════════
EIA_SERIES = {
    "US Crude Stocks (ex-SPR)":  dict(sid="PET.WCESTUS1.W",  unit="kbbl",   sector="Energy"),
    "Cushing Crude Stocks":      dict(sid="PET.W_EPC0_SAX_YCUOK_MBBL.W", unit="kbbl", sector="Energy"),
    "US Gasoline Stocks":        dict(sid="PET.WGTSTUS1.W",  unit="kbbl",   sector="Energy"),
    "US Distillate Stocks":      dict(sid="PET.WDISTUS1.W",  unit="kbbl",   sector="Energy"),
    "US Crude Production":       dict(sid="PET.WCRFPUS2.W",  unit="kb/d",   sector="Energy"),
    "US Nat Gas Storage (L48)":  dict(sid="NG.NW2_EPG0_SWO_R48_BCF.W", unit="bcf", sector="Energy"),
}

EIA_MAP = {
    "WTI Crude (CL)":         ["US Crude Stocks (ex-SPR)", "Cushing Crude Stocks", "US Crude Production"],
    "Brent Crude (BZ)":       ["US Crude Stocks (ex-SPR)", "US Crude Production"],
    "RBOB Gasoline (RB)":     ["US Gasoline Stocks"],
    "ULSD Heating Oil (HO)":  ["US Distillate Stocks"],
    "Henry Hub Nat Gas (NG)": ["US Nat Gas Storage (L48)"],
}


@st.cache_data(ttl=3600)
def fetch_eia(series_name: str, api_key: str, n: int = 260) -> pd.DataFrame:
    """One weekly EIA series. Empty frame on failure — the page then says the feed is
    unavailable rather than inventing a stock level."""
    if not REQUESTS_AVAILABLE or not api_key:
        return pd.DataFrame()
    sid = EIA_SERIES[series_name]["sid"]
    url = "https://api.eia.gov/v2/seriesid/" + sid
    try:
        r = requests.get(url, params={"api_key": api_key, "length": n}, timeout=15)
        if r.status_code != 200:
            LOG.warning("EIA %s -> HTTP %s", sid, r.status_code)
            return pd.DataFrame()
        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            LOG.warning("EIA %s -> empty payload", sid)
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"]  = pd.to_datetime(df["period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]].dropna().sort_values("date").set_index("date")
    except requests.RequestException as e:
        LOG.warning("EIA %s request failed: %s", sid, e)
        return pd.DataFrame()
    except Exception as e:
        LOG.warning("EIA %s parse failed: %s", sid, e)
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  FRED SERIES MAP  (free key at fred.stlouisfed.org)
# ══════════════════════════════════════════════════════════════════════════════
FRED_SERIES = {
    "USA":       dict(cpi_yoy="CPIAUCSL", policy_rate="DFF",     gdp="GDPC1"),
    "Euro Area": dict(cpi_yoy="CP0000EZ19M086NEST", policy_rate="ECBDFR", gdp="CLVMNACSCAB1GQEA19"),
    "Germany":   dict(cpi_yoy="DEUCPIALLMINMEI", policy_rate="ECBDFR", gdp="CLVMNACSCAB1GQDE"),
    "France":    dict(cpi_yoy="FRACPIALLMINMEI", policy_rate="ECBDFR", gdp="CLVMNACSCAB1GQFR"),
    "UK":        dict(cpi_yoy="GBRCPIALLMINMEI", policy_rate="IUDSOIA", gdp="NGDPRSAXDCGBQ"),
    "Japan":     dict(cpi_yoy="JPNCPIALLMINMEI", policy_rate="IRSTCI01JPM156N", gdp="JPNRGDPEXP"),
    "China":     dict(cpi_yoy="CHNCPIALLMINMEI", policy_rate=None, gdp="NGDPRSAXDCCNQ"),
    "Brazil":    dict(cpi_yoy="BRACPIALLMINMEI", policy_rate="INTDSRBRM193N", gdp="NGDPRSAXDCBRQ"),
    "India":     dict(cpi_yoy="INDCPIALLMINMEI", policy_rate="INTDSRINM193N", gdp="NGDPRSAXDCINQ"),
}

MACRO_METRICS = {
    "cpi_yoy":     dict(label="CPI (index)",      note="Index level. YoY % is derived below."),
    "policy_rate": dict(label="Policy rate (%)",  note="Central bank target / overnight rate."),
    "gdp":         dict(label="Real GDP",         note="Real GDP, local units. Quarterly."),
}

FRED_COMMODITY_CONTEXT = {
    "US Dollar Index (DXY proxy)": "DTWEXBGS",
    "US 10Y Treasury Yield":       "DGS10",
    "US 10Y Breakeven Inflation":  "T10YIE",
    "US Industrial Production":    "INDPRO",
}


@st.cache_data(ttl=3600)
def fetch_fred(series_id: str, api_key: str, start: str = "2015-01-01") -> pd.DataFrame:
    """One FRED series. Real data or nothing — same contract as the price feed."""
    if not REQUESTS_AVAILABLE or not api_key or not series_id:
        return pd.DataFrame()
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": api_key, "file_type": "json",
                    "observation_start": start},
            timeout=15)
        if r.status_code != 200:
            LOG.warning("FRED %s -> HTTP %s", series_id, r.status_code)
            return pd.DataFrame()
        obs = r.json().get("observations", [])
        if not obs:
            return pd.DataFrame()
        df = pd.DataFrame(obs)
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]].dropna().set_index("date")
    except requests.RequestException as e:
        LOG.warning("FRED %s request failed: %s", series_id, e)
        return pd.DataFrame()
    except Exception as e:
        LOG.warning("FRED %s parse failed: %s", series_id, e)
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  REGIONAL BALANCES  (static IEA/USDA-style estimates — not a live feed)
# ══════════════════════════════════════════════════════════════════════════════
REGIONAL_DATA = {
    "WTI Crude (CL)": [
        dict(region="North America", supply=15.2, demand=10.8, lat=45,  lon=-100),
        dict(region="Middle East",   supply=28.1, demand=8.2,  lat=25,  lon=50),
        dict(region="Russia/FSU",    supply=13.5, demand=5.1,  lat=60,  lon=70),
        dict(region="Europe",        supply=3.4,  demand=12.6, lat=52,  lon=10),
        dict(region="Asia Pacific",  supply=8.1,  demand=34.5, lat=30,  lon=115),
        dict(region="Africa",        supply=7.9,  demand=4.3,  lat=5,   lon=20),
        dict(region="Latin America", supply=5.8,  demand=6.2,  lat=-15, lon=-60),
    ],
    "Gold (GC)": [
        dict(region="China",     supply=370, demand=950, lat=35,  lon=105),
        dict(region="Australia", supply=330, demand=30,  lat=-25, lon=133),
        dict(region="Russia",    supply=295, demand=90,  lat=60,  lon=70),
        dict(region="Canada",    supply=190, demand=50,  lat=60,  lon=-95),
        dict(region="USA",       supply=170, demand=230, lat=38,  lon=-97),
        dict(region="S. Africa", supply=120, demand=45,  lat=-30, lon=25),
        dict(region="India",     supply=35,  demand=800, lat=20,  lon=80),
        dict(region="Europe",    supply=30,  demand=280, lat=50,  lon=15),
    ],
    "Corn (ZC)": [
        dict(region="USA",          supply=387, demand=295, lat=38,  lon=-97),
        dict(region="China",        supply=277, demand=305, lat=35,  lon=105),
        dict(region="Brazil",       supply=137, demand=78,  lat=-15, lon=-55),
        dict(region="EU",           supply=62,  demand=71,  lat=50,  lon=15),
        dict(region="Argentina",    supply=55,  demand=16,  lat=-35, lon=-65),
        dict(region="Ukraine",      supply=27,  demand=12,  lat=49,  lon=32),
        dict(region="Mexico",       supply=28,  demand=45,  lat=24,  lon=-102),
        dict(region="South Africa", supply=16,  demand=14,  lat=-29, lon=25),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT CALENDAR — computed cadences, honestly-labelled approximations
# ══════════════════════════════════════════════════════════════════════════════
def _next_weekdays(anchor: date, weekday: int, n: int) -> List[date]:
    """Next n dates falling on `weekday` (Mon=0), strictly >= anchor."""
    d = anchor + timedelta(days=(weekday - anchor.weekday()) % 7)
    return [d + timedelta(weeks=i) for i in range(n)]


def build_calendar_events(today: Optional[date] = None) -> List[dict]:
    """The old table invented every date as today+N — a fabricated calendar on a desk
    whose whole premise is not fabricating data. Weekly prints have a real cadence and
    are COMPUTED (holiday weeks can shift them by a day). Monthly/irregular releases
    are anchored to their usual slot and labelled APPROXIMATE — verify against the
    official calendars before carrying risk into one."""
    today = today or date.today()
    ev: List[dict] = []
    for d in _next_weekdays(today, 2, 5):   # Wednesday
        ev.append(dict(date=d, event="EIA Weekly Petroleum Status Report",
                       tags=["Energy", "Crude"], basis="computed (weekly, Wed)"))
    for d in _next_weekdays(today, 3, 5):   # Thursday
        ev.append(dict(date=d, event="EIA Natural Gas Storage Report",
                       tags=["Energy", "Gas"], basis="computed (weekly, Thu)"))
    for d in _next_weekdays(today, 4, 5):   # Friday
        ev.append(dict(date=d, event="Baker Hughes Rig Count",
                       tags=["Energy"], basis="computed (weekly, Fri)"))
    for d in _next_weekdays(today, 0, 5):   # Monday, Apr-Nov only
        if 4 <= d.month <= 11:
            ev.append(dict(date=d, event="USDA Crop Progress",
                           tags=["Grains"], basis="computed (weekly in season, Mon)"))
    # Monthly, usual slot — approximate.
    wasde = date(today.year, today.month, 12)
    if wasde < today:
        y, m = _shift_month(today.year, today.month, 1)
        wasde = date(y, m, 12)
    ev.append(dict(date=wasde, event="USDA WASDE", tags=["Grains", "Softs"],
                   basis="approximate (~12th) — verify usda.gov"))
    momr = date(today.year, today.month, 13)
    if momr < today:
        y, m = _shift_month(today.year, today.month, 1)
        momr = date(y, m, 13)
    ev.append(dict(date=momr, event="OPEC MOMR", tags=["Energy", "OPEC"],
                   basis="approximate (mid-month) — verify opec.org"))
    omr = date(today.year, today.month, 15)
    if omr < today:
        y, m = _shift_month(today.year, today.month, 1)
        omr = date(y, m, 15)
    ev.append(dict(date=omr, event="IEA Oil Market Report", tags=["Energy"],
                   basis="approximate (mid-month) — verify iea.org"))
    ev.append(dict(date=None, event="FOMC / ECB decisions, USDA Grain Stocks, Cattle on Feed",
                   tags=["Macro", "Grains", "Livestock"],
                   basis="irregular — check the official calendars; not invented here"))
    return sorted(ev, key=lambda e: (e["date"] is None, e["date"] or date.max))


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def black76(F, K, T, r, sigma, option_type="call"):
    """European option on a futures contract. Greeks per unit of underlying.
    T<=0 collapses to discounted intrinsic (an expired option is worth its payoff,
    not zero — the old return-all-zeros silently erased expired positions)."""
    if F <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        intr = max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
        delta = (1.0 if option_type == "call" else -1.0) if intr > 0 else 0.0
        return dict(price=intr, delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    d1   = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2   = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    if option_type == "call":
        price, delta = disc * (F * norm.cdf(d1) - K * norm.cdf(d2)), disc * norm.cdf(d1)
    else:
        price, delta = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1)), -disc * norm.cdf(-d1)
    gamma = disc * norm.pdf(d1) / (F * sigma * math.sqrt(T))
    vega  = disc * F * norm.pdf(d1) * math.sqrt(T) / 100        # per vol point
    theta = (-(disc * F * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * price) / 365
    rho   = -T * price / 100
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def vol_surface_fn(F, atm_vol, skew=-0.05, curv=0.02, vov=0.15):
    """Parametric surface in log-moneyness. NOT calibrated to listed quotes."""
    mats  = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
    Kgrid = F * np.exp(np.linspace(-0.40, 0.40, 25))
    Z = np.zeros((len(mats), len(Kgrid)))
    for i, T in enumerate(mats):
        for j, K in enumerate(Kgrid):
            x = math.log(K / F)
            Z[i, j] = max(atm_vol * (1 + vov * math.sqrt(T)) + skew * x + curv * x**2, 0.01)
    return mats, Kgrid, Z


def implied_carry(strip: pd.DataFrame) -> pd.DataFrame:
    """Carry read straight off the strip. No storage cost or convenience yield assumed —
    the market has already priced them, and this just reads what it priced. Roll yield
    is annualised on CALENDAR tenor (T is real year-fraction to delivery now)."""
    if strip.empty or len(strip) < 2:
        return pd.DataFrame()
    f1  = float(strip["price"].iloc[0])
    t1  = float(strip["T"].iloc[0])
    out = strip.copy()
    out["spread_vs_M1"] = out["price"] - f1
    out["spread_pct"]   = (out["price"] / f1 - 1) * 100
    dT = (out["T"] - t1).clip(lower=1e-6)
    out["roll_yield"] = np.where(out["month"] > 1,
                                 (f1 - out["price"]) / out["price"] / dT * 100, 0.0)
    return out


# ── Monte Carlo — centred on the LIVE forward curve ──────────────────────────
def simulate(spot: float, vol: float, n_paths: int = 1000, horizon: int = 18,
             halflife: Optional[float] = None,
             forward: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
             seed: int = 0) -> dict:
    """
    Price simulator, revision 2.

    The mean level is no longer flat ln(spot): paths are centred on the LIVE forward
    strip, so E[S_t] = F(t) by construction. That matters twice over —
      1. The market has already priced carry and seasonality (NG winter premia, RB
         driving season); a flat mean throws that information away.
      2. Decomposing x_t = g(t) + y_t with g(t) = ln F(t) − Var_y(t)/2 and y a
         zero-mean OU (or Brownian) process makes the centring EXACT, and lets the OU
         step use its exact discretisation (φ = e^{−κΔ}) instead of Euler.

    halflife=None -> y is Brownian (GBM shape). Appropriate for gold/silver, which
                     behave like financial assets rather than consumables.
    halflife=h    -> y is OU with κ = ln2/h (Schwartz 1-factor shape). This matters:
                     an unreverted walk at nat-gas vol over 3y puts P95 near 3× spot
                     and P5 near zero — physically meaningless for a storable with an
                     inventory-driven price. Reversion fixes the tails.

    Note the mean/median distinction: E[S_t] = F(t) exactly; the MEDIAN sits at
    F(t)·e^{−Var(t)/2}, slightly below. That is a choice (risk-neutral-style
    centring), stated here rather than left implicit.

    forward: (T_years, prices) from the strip. None => flat forward at spot.
    """
    rng = np.random.default_rng(seed)
    dt  = 1 / 12
    tgrid = np.arange(horizon + 1) * dt

    # Log-forward on the monthly grid, anchored at ln(spot) at t=0, flat beyond strip.
    if forward is not None and len(forward[0]) >= 1:
        fT = np.concatenate(([0.0], np.asarray(forward[0], dtype=float)))
        fP = np.concatenate(([spot], np.asarray(forward[1], dtype=float)))
        order = np.argsort(fT)
        lnF = np.interp(tgrid, fT[order], np.log(fP[order]))
        fwd_note = f"centred on live forward strip ({len(forward[0])} pts)"
    else:
        lnF = np.full(horizon + 1, math.log(spot))
        fwd_note = "flat forward (strip unavailable)"

    y = np.zeros((n_paths, horizon + 1))
    if halflife is None:
        var_t = vol**2 * tgrid
        for t in range(1, horizon + 1):
            y[:, t] = y[:, t - 1] + vol * math.sqrt(dt) * rng.standard_normal(n_paths)
        model = f"GBM shape (no reversion) — {fwd_note}"
    else:
        kappa = math.log(2) / halflife
        phi   = math.exp(-kappa * dt)
        sd    = vol * math.sqrt((1 - phi**2) / (2 * kappa))    # exact OU step
        var_t = vol**2 * (1 - np.exp(-2 * kappa * tgrid)) / (2 * kappa)
        for t in range(1, horizon + 1):
            y[:, t] = y[:, t - 1] * phi + sd * rng.standard_normal(n_paths)
        model = (f"Schwartz 1-factor (half-life {halflife:.2f}y, κ={kappa:.2f}) — {fwd_note}")

    paths = np.exp(lnF[None, :] - var_t[None, :] / 2 + y)
    paths[:, 0] = spot

    fan_dates = [date.today() + timedelta(days=30 * i) for i in range(horizon + 1)]
    pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fan  = pd.DataFrame(dict(date=fan_dates, p5=pcts[0], p25=pcts[1],
                             p50=pcts[2], p75=pcts[3], p95=pcts[4]))
    hb = np.histogram(paths[:, -1], bins=40)
    return dict(fan=fan, model=model,
                mean=float(paths[:, -1].mean()),
                median=float(np.median(paths[:, -1])),
                p5=float(np.percentile(paths[:, -1], 5)),
                p95=float(np.percentile(paths[:, -1], 95)),
                hist_x=hb[1][:-1].tolist(), hist_y=hb[0].tolist())


# ══════════════════════════════════════════════════════════════════════════════
#  POSITION VALUATION — one set of helpers used by blotter, VaR and stress
# ══════════════════════════════════════════════════════════════════════════════
def option_time_remaining(p: dict, today: Optional[date] = None) -> float:
    """Remaining tenor in years. Options AGE now: the trade date is stored at booking
    and elapsed time is subtracted — the old frozen tenor meant a 6-month option
    stayed a 6-month option forever and no theta ever showed up in P&L."""
    today = today or date.today()
    td = p.get("trade_date")
    if not td:
        return float(p["tenor"])
    try:
        elapsed = (today - date.fromisoformat(td)).days / 365.25
    except Exception:
        elapsed = 0.0
    return max(float(p["tenor"]) - elapsed, 0.0)


def position_base_price(p: dict, marks: MarkBoard) -> Optional[float]:
    """The price this position is marked against: dated strip price for a dated
    future, front-month mark otherwise. None => the line errors, never proxies."""
    if p.get("kind", "future") == "future" and p.get("strip_ticker"):
        dm = dated_mark(p["strip_ticker"])
        return dm["price"] if dm else None
    return marks.get(p["commodity"])


def position_pnl_at(p: dict, base: float, ret: float, r: float = 0.05) -> float:
    """Cash P&L of one position if its underlying moves by `ret` (fractional).
    Futures are linear; OPTIONS ARE FULLY REVALUED (Black-76 at the shocked forward)
    — the old stress path ignored them entirely, and linear delta is exactly wrong
    in the large moves stress testing exists for."""
    mult = price_multiplier(p["commodity"])
    sign = 1 if p["side"] == "Long" else -1
    if p.get("kind", "future") == "option":
        T = option_time_remaining(p)
        v0 = black76(base, p["strike"], T, r, p["vol"], p["opt_type"])["price"]
        v1 = black76(base * (1 + ret), p["strike"], T, r, p["vol"], p["opt_type"])["price"]
        return sign * (v1 - v0) * mult * p["lots"]
    return sign * base * ret * mult * p["lots"]


def delta_cash(p: dict, marks: MarkBoard, r: float = 0.05) -> Optional[float]:
    """Signed cash sensitivity to a 100% move of the underlying — the common currency
    of both VaR methods. Future: ±notional. Option: Black-76 delta × F × mult × lots.
    This is the fix for the old asymmetry where parametric VaR charged options full
    futures notional while historical VaR ignored them."""
    base = position_base_price(p, marks)
    if base is None:
        return None
    mult = price_multiplier(p["commodity"])
    sign = 1 if p["side"] == "Long" else -1
    if p.get("kind", "future") == "option":
        d = black76(base, p["strike"], option_time_remaining(p), r,
                    p["vol"], p["opt_type"])["delta"]
        return sign * d * base * mult * p["lots"]
    return sign * base * mult * p["lots"]


def _position_label(p: dict) -> str:
    n = p["commodity"]
    if p.get("kind", "future") == "option":
        rem = option_time_remaining(p) * 12
        return f"{n} {p['opt_type'][:1].upper()}{p['strike']:g} ({rem:.1f}m left)"
    if p.get("strip_ticker"):
        return f"{n} {p.get('strip_label', p['strip_ticker'])}"
    return f"{n} fut"


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO RISK
# ══════════════════════════════════════════════════════════════════════════════
def portfolio_var(positions: List[dict], marks, corr: pd.DataFrame,
                  conf: float = 0.95, horizon: int = 1,
                  diversified: bool = True) -> dict:
    """
    Parametric VaR / ES on the delta-equivalent book.

    Options enter at Black-76 delta-cash, not full notional. Each position's risk
    vol is the UNDERLYING's vol (the delta-equivalent futures position carries the
    underlying's distribution, whatever σ the option was booked at).

    diversified=True  -> σₚ = √(wᵀΣw) with the live correlation matrix. A long WTI /
                         short Brent book nets down to near-nothing — the economic
                         truth only the matrix can see.
    Missing history   -> if ANY needed pair correlation is NaN the book falls back to
                         the conservative sum and says so. The old zero-fill silently
                         assumed independence, which UNDERSTATES risk for same-sign
                         correlated legs — the opposite of a safe default.
    """
    z = norm.ppf(conf)
    rows, names = [], []
    gross = 0.0

    for p in positions:
        w = delta_cash(p, marks)
        if w is None:
            continue
        name = p["commodity"]
        vol  = p.get("risk_vol", COMMODITIES[name]["vol"])
        dvol = vol / math.sqrt(252)
        sd_var = abs(w) * dvol * z * math.sqrt(horizon)
        sd_es  = abs(w) * dvol * norm.pdf(z) / (1 - conf) * math.sqrt(horizon)
        gross += abs(w)
        names.append(name)
        rows.append(dict(Position=_position_label(p), Contract=name, Side=p["side"],
                         Lots=p["lots"], DeltaCash=w, Vol=vol * 100,
                         StandaloneVaR=sd_var, StandaloneES=sd_es, _dvol=dvol))

    if not rows:
        return dict(rows=[], var=0.0, es=0.0, undiversified=0.0,
                    gross=0.0, benefit=0.0, corr_used=False, reason="no marked positions")

    undiversified = sum(r["StandaloneVaR"] for r in rows)
    sigma_vec = np.array([abs(r["DeltaCash"]) * r["_dvol"] for r in rows])
    sgn       = np.array([1 if r["DeltaCash"] >= 0 else -1 for r in rows])
    w_vec     = sigma_vec * sgn

    corr_used, reason = False, ""
    if diversified and not corr.empty and all(n in corr.index for n in names):
        R = corr.loc[names, names].values.astype(float)
        if np.isnan(R).any():
            reason = ("missing pair history — a NaN correlation would have to be "
                      "invented; falling back to the conservative sum instead")
        else:
            np.fill_diagonal(R, 1.0)
            port_sigma = math.sqrt(max(float(w_vec @ R @ w_vec), 0.0))
            corr_used = True
    elif diversified:
        reason = "correlation matrix unavailable for one or more legs"

    if not corr_used:
        port_sigma = float(np.abs(w_vec).sum())

    var = port_sigma * z * math.sqrt(horizon)
    es  = port_sigma * norm.pdf(z) / (1 - conf) * math.sqrt(horizon)

    if corr_used and port_sigma > 0:
        mcv = (R @ w_vec) / port_sigma
        for i, rrow in enumerate(rows):
            comp = w_vec[i] * mcv[i] / port_sigma * var
            rrow["ComponentVaR"] = float(comp)
            rrow["PctOfVaR"] = float(comp / var * 100) if var else 0.0
    else:
        for rrow in rows:
            rrow["ComponentVaR"] = rrow["StandaloneVaR"]
            rrow["PctOfVaR"] = (rrow["StandaloneVaR"] / undiversified * 100
                                if undiversified else 0.0)

    for rrow in rows:
        rrow.pop("_dvol", None)

    return dict(rows=rows, var=var, es=es, undiversified=undiversified, gross=gross,
                benefit=(undiversified - var) / undiversified * 100 if undiversified else 0.0,
                corr_used=corr_used, reason=reason)


def historical_var(positions: List[dict], marks, conf: float = 0.95,
                   horizon: int = 1, lookback: int = 500) -> dict:
    """
    Historical-simulation VaR: replays actual daily return vectors on today's book.
    No normality, no correlation matrix — the joint behaviour, fat tails included,
    is already in the data.

    Revision 2:
      • Options included via delta-cash (the old version dropped them — a book that
        was all options showed zero historical risk).
      • horizon>1 uses OVERLAPPING h-day windows of the replayed daily P&L. The old
        √h scaling re-imported the Gaussian assumption this method exists to avoid.
      • Dated futures map onto their underlying's front-month return series (a
        one-factor approximation, stated on screen).
    """
    panel = panel_years(3)
    if panel.empty:
        return dict(available=False)

    w_map: Dict[str, float] = {}
    for p in positions:
        w = delta_cash(p, marks)
        if w is None:
            continue
        n = p["commodity"]
        if n not in panel.columns:
            continue
        w_map[n] = w_map.get(n, 0.0) + w

    if not w_map:
        return dict(available=False)

    names = list(w_map)
    rets = panel[names].pct_change().dropna().tail(lookback + horizon)
    if len(rets) < 100:
        return dict(available=False)

    pnl_daily = pd.Series(rets.values @ np.array([w_map[n] for n in names]),
                          index=rets.index)
    if horizon > 1:
        pnl = pnl_daily.rolling(horizon).sum().dropna()
        note = (f"{horizon}-day P&L from overlapping windows "
                f"({len(pnl)} obs; overlap shrinks the effective sample)")
    else:
        pnl = pnl_daily
        note = f"{len(pnl)} daily observations"

    var = float(-np.percentile(pnl.values, (1 - conf) * 100))
    tail = pnl[pnl <= -var]
    es = float(-tail.mean()) if len(tail) else var
    worst_i = int(np.argmin(pnl.values))
    return dict(available=True, var=var, es=es, pnl=pnl.values, n_days=len(pnl),
                note=note, worst_date=pnl.index[worst_i].date(),
                worst_pnl=float(pnl.values[worst_i]))


# ── Historical stress episodes ───────────────────────────────────────────────
STRESS_EPISODES = {
    "COVID crash (Feb–Mar 2020)":        ("2020-02-19", "2020-03-23"),
    "WTI negative print (Apr 2020)":     ("2020-04-01", "2020-04-30"),
    "Ukraine invasion (Feb–Mar 2022)":   ("2022-02-21", "2022-03-09"),
    "2022 energy peak → bust (Jun–Sep)": ("2022-06-08", "2022-09-26"),
    "Banking wobble (Mar 2023)":         ("2023-03-08", "2023-03-24"),
}


def stress_replay(positions: List[dict], marks, start: str, end: str) -> dict:
    """Apply the actual per-contract move of a dated episode to the current book.
    Options are FULLY REVALUED at the shocked forward; dated futures take their
    underlying's front-month move (factor approximation, stated on screen)."""
    panel = panel_years(6)
    if panel.empty:
        return dict(available=False)
    try:
        window = panel.loc[start:end]
    except Exception:
        return dict(available=False)
    if window.empty or len(window) < 2:
        return dict(available=False)

    rows, total = [], 0.0
    for p in positions:
        n = p["commodity"]
        base = position_base_price(p, marks)
        if base is None or n not in window.columns:
            continue
        s = window[n].dropna()
        if len(s) < 2:
            continue
        move = float(s.iloc[-1] / s.iloc[0] - 1)
        pnl = position_pnl_at(p, base, move)
        total += pnl
        rows.append(dict(Position=_position_label(p), Side=p["side"], Lots=p["lots"],
                         Move=move * 100, PnL=pnl))
    if not rows:
        return dict(available=False)
    return dict(available=True, rows=rows, total=total,
                start=str(window.index[0].date()), end=str(window.index[-1].date()))


# ══════════════════════════════════════════════════════════════════════════════
#  SEASONALITY & SIGNALS — all off the shared panel / shared strips
# ══════════════════════════════════════════════════════════════════════════════
def seasonality(commodity: str, years: int = 10) -> pd.DataFrame:
    """Monthly seasonal distribution of returns off the shared panel (no extra call).
    CAVEAT — the continuous series is not roll-adjusted: persistent contango puts a
    systematic negative bias on roll months, material for NG over 10 years."""
    panel = panel_years(years + 0.2)
    if panel.empty or commodity not in panel.columns:
        return pd.DataFrame()
    s = panel[commodity].dropna()
    if len(s) < 250:
        return pd.DataFrame()
    m = s.resample("ME").last()
    r = (m / m.shift(1) - 1).dropna() * 100
    out = pd.DataFrame({"ret": r})
    out["month"] = out.index.month
    out["year"]  = out.index.year
    return out


@st.cache_data(ttl=900)
def build_signals() -> pd.DataFrame:
    """One row per contract, one column per signal. Pure aggregation of what the other
    pages already compute — nothing here is modelled, and (revision 2) nothing here
    triggers a download: the whole scan runs off the shared panel and the single
    grouped strip download (~250 requests -> 3)."""
    strips = fetch_all_strips()
    panel  = panel_years(1.2)
    try:
        cot = fetch_cot_all()          # one batched CFTC request, cached 6h
    except Exception as e:
        LOG.warning("COT unavailable for scanner: %s", e)
        cot = {}
    rows = []
    this_month = date.today().month

    for name, c in COMMODITIES.items():
        row = dict(Contract=name, Sector=c["sector"])

        strip = strips.get(name, pd.DataFrame())
        if not strip.empty and len(strip) >= 2:
            f1 = float(strip["price"].iloc[0])
            f2 = float(strip["price"].iloc[1])
            fn = float(strip["price"].iloc[-1])
            row["Mark"]      = f1
            row["Carry%"]    = (fn - f1) / f1 * 100
            row["Structure"] = ("CONTANGO" if row["Carry%"] > 0.5
                                else "BACKWARD" if row["Carry%"] < -0.5 else "FLAT")
            row["M1M2"] = f1 - f2
        else:
            row.update(Mark=np.nan, **{"Carry%": np.nan}, Structure="n/a", M1M2=np.nan)

        if not panel.empty and name in panel.columns:
            px_s = panel[name].dropna()
        else:
            px_s = pd.Series(dtype=float)

        if len(px_s) > 130:
            lr = np.log(px_s / px_s.shift(1)).dropna()
            rv60  = float(lr.tail(60).std() * math.sqrt(252))
            rv252 = float(lr.tail(252).std() * math.sqrt(252))
            row["RV60"]      = rv60 * 100
            row["VolRegime"] = rv60 / rv252 if rv252 > 0 else np.nan
        else:
            row["RV60"], row["VolRegime"] = np.nan, np.nan

        if len(px_s) > 63:
            row["Chg1M"] = float(px_s.iloc[-1] / px_s.iloc[-21] - 1) * 100
            row["Chg3M"] = float(px_s.iloc[-1] / px_s.iloc[-63] - 1) * 100
            row["Px%ile1y"] = float((px_s < px_s.iloc[-1]).mean() * 100)
        else:
            row["Chg1M"] = row["Chg3M"] = row["Px%ile1y"] = np.nan

        if c.get("seasonal"):
            s = seasonality(name, 10)
            if not s.empty:
                d = s[s["month"] == this_month]["ret"]
                row["SeasonMed"] = float(d.median()) if len(d) else np.nan
                row["SeasonHit"] = float((d > 0).mean() * 100) if len(d) else np.nan
            else:
                row["SeasonMed"] = row["SeasonHit"] = np.nan
        else:
            row["SeasonMed"] = row["SeasonHit"] = np.nan

        cdf = cot.get(name)
        if cdf is not None and not cdf.empty:
            row["MMnet%OI"] = float(cdf["mm_net_pct_oi"].iloc[-1])
            h3 = cdf.tail(156)
            row["COT%ile"] = float((h3["mm_net"] < cdf["mm_net"].iloc[-1]).mean() * 100)
        else:
            row["MMnet%OI"] = row["COT%ile"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  BOOK ANALYTICS — Greeks, roll P&L
# ══════════════════════════════════════════════════════════════════════════════
def book_greeks(positions: List[dict], marks, r: float = 0.05) -> dict:
    """Net Greeks across the whole book, in cash. Futures carry delta 1.0 per unit
    and nothing else. Options use their REMAINING tenor (they age now)."""
    tot = dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    rows = []
    for p in positions:
        base = position_base_price(p, marks)
        if base is None:
            continue
        n = p["commodity"]
        mult = price_multiplier(n)
        sign = 1 if p["side"] == "Long" else -1
        lots = p["lots"]
        if p.get("kind", "future") == "option":
            g = black76(base, p["strike"], option_time_remaining(p), r,
                        p["vol"], p["opt_type"])
            d, gm = g["delta"] * mult * lots * sign, g["gamma"] * mult * lots * sign
            v, th = g["vega"] * mult * lots * sign, g["theta"] * mult * lots * sign
        else:
            d, gm, v, th = 1.0 * mult * lots * sign, 0.0, 0.0, 0.0
        tot["delta"] += d
        tot["gamma"] += gm
        tot["vega"]  += v
        tot["theta"] += th
        rows.append(dict(Position=_position_label(p), Side=p["side"], Lots=lots,
                         Delta=d, Gamma=gm, Vega=v, Theta=th))
    return dict(total=tot, rows=rows)


def roll_pnl(positions: List[dict], marks) -> List[dict]:
    """
    Split P&L into price and carry for FRONT-MONTH futures.

    Price P&L = (mark − entry) × multiplier × lots
    Roll P&L  = (M1 − M2) × multiplier × lots, sign-adjusted
    Annualised on CALENDAR spacing between M1 and M2 (was hardcoded ×12 — wrong for
    every non-monthly cycle).

    Dated lines are excluded on purpose: a dated contract has no roll bleed — that is
    precisely why you book one.
    """
    strips = fetch_all_strips()
    out = []
    for p in positions:
        if p.get("kind", "future") == "option" or p.get("strip_ticker"):
            continue
        n = p["commodity"]
        mark = marks.get(n)
        if mark is None:
            continue
        mult = price_multiplier(n)
        sign = 1 if p["side"] == "Long" else -1
        price_p = sign * (mark - p["entry"]) * mult * p["lots"]

        strip = strips.get(n, pd.DataFrame())
        if strip.empty or len(strip) < 2:
            roll_p, m1m2, ann = 0.0, None, None
        else:
            m1, m2 = float(strip["price"].iloc[0]), float(strip["price"].iloc[1])
            dT = max(float(strip["T"].iloc[1]) - float(strip["T"].iloc[0]), 1e-6)
            m1m2 = m1 - m2
            roll_p = sign * m1m2 * mult * p["lots"]
            ann = (m1m2 / m2 / dT * 100) if m2 else None

        out.append(dict(Contract=n, Side=p["side"], Lots=p["lots"],
                        PricePnL=price_p, MonthlyRoll=roll_p,
                        M1M2=m1m2, RollAnnPct=ann))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  BLOTTER PERSISTENCE — per-book files keyed by a ?book= id in the URL
# ══════════════════════════════════════════════════════════════════════════════
BLOTTER_DIR = ".blotters"
LEGACY_BLOTTER = "blotter.json"


def _book_id() -> str:
    """Book id carried in the URL (?book=…). On Streamlit Cloud the old single
    blotter.json was shared by EVERY visitor and wiped on redeploy — a privacy bug
    and a durability lie. A capability-URL book id fixes the sharing; export/import
    JSON remains the honest durability mechanism (stated in the UI)."""
    try:
        qp = st.query_params
        if "book" not in qp or not qp["book"]:
            qp["book"] = uuid.uuid4().hex[:12]
        return "".join(ch for ch in qp["book"] if ch.isalnum())[:32] or "default"
    except Exception:
        return "default"


def _blotter_path(book: str) -> str:
    return os.path.join(BLOTTER_DIR, f"{book}.json")


def blotter_serialise(positions: List[dict]) -> str:
    return json.dumps(positions, indent=2, default=str)


def blotter_deserialise(payload) -> List[dict]:
    """Validate an imported book. Unknown contracts are dropped (registry may have
    changed). Legacy options without a trade_date get today — their tenor restarts,
    which is stated to the user on import rather than silently assumed."""
    raw = json.loads(payload) if isinstance(payload, str) else payload
    out = []
    for p in raw:
        if p.get("commodity") not in COMMODITIES:
            continue
        q = dict(commodity=p["commodity"],
                 kind=p.get("kind", "future"),
                 side=p.get("side", "Long"),
                 lots=int(p.get("lots", 1)),
                 entry=float(p.get("entry", 0.0)))
        if q["kind"] == "option":
            q.update(opt_type=p.get("opt_type", "call"),
                     strike=float(p.get("strike", 0.0)),
                     tenor=float(p.get("tenor", 0.25)),
                     vol=float(p.get("vol", 0.3)),
                     premium=float(p.get("premium", 0.0)),
                     trade_date=p.get("trade_date") or date.today().isoformat())
        else:
            if p.get("strip_ticker"):
                q.update(strip_ticker=str(p["strip_ticker"]),
                         strip_label=str(p.get("strip_label", p["strip_ticker"])))
            if p.get("trade_date"):
                q["trade_date"] = str(p["trade_date"])
        out.append(q)
    return out


def blotter_save(positions: List[dict]) -> None:
    try:
        os.makedirs(BLOTTER_DIR, exist_ok=True)
        with open(_blotter_path(_book_id()), "w") as f:
            f.write(blotter_serialise(positions))
    except Exception as e:
        LOG.warning("blotter save failed: %s", e)


def blotter_load() -> List[dict]:
    path = _blotter_path(_book_id())
    try:
        if os.path.exists(path):
            with open(path) as f:
                return blotter_deserialise(f.read())
        # One-time migration from the legacy shared file.
        if os.path.exists(LEGACY_BLOTTER):
            with open(LEGACY_BLOTTER) as f:
                pos = blotter_deserialise(f.read())
            if pos:
                LOG.info("migrated %d position(s) from legacy blotter.json", len(pos))
            return pos
    except Exception as e:
        LOG.warning("blotter load failed: %s", e)
    return []


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _styled(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="JetBrains Mono, monospace", size=11, color=TEXT),
        height=height, margin=dict(l=50, r=25, t=45, b=40),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER,
                        font=dict(family="JetBrains Mono, monospace", size=11)),
    )
    return fig


def kpi(col, label: str, value: str, sub: str = "", accent: str = AMBER) -> None:
    col.markdown(
        f"""<div class="kpi-card" style="border-left-color:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div></div>""",
        unsafe_allow_html=True)


def pctile_badge(pct: Optional[float]) -> str:
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return '<span class="badge">n/a</span>'
    if pct >= 80:
        return f'<span class="badge badge-red">{pct:.0f}th %ile — RICH</span>'
    if pct <= 20:
        return f'<span class="badge badge-green">{pct:.0f}th %ile — CHEAP</span>'
    return f'<span class="badge badge-amber">{pct:.0f}th %ile</span>'


def require_mark(marks: MarkBoard, commodity: str) -> Optional[float]:
    """Gate every page on a real mark. Stale-but-dated marks pass WITH their date
    shown — a dated settle is honest; only a missing mark blocks the page."""
    m = marks.get(commodity)
    if m is None:
        st.error(f"**NO LIVE MARK** for {commodity} — feed unavailable. "
                 "This desk does not fabricate prices; analytics are disabled "
                 "until the feed returns. Try Refresh in the sidebar.")
        return None
    if marks.is_stale(commodity):
        st.caption(f"⚠️ Mark is a dated settle from **{marks.asof(commodity)}** "
                   "(no fresher print on the feed).")
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR & HEADER
# ══════════════════════════════════════════════════════════════════════════════
# Navigation grouped the way a desk thinks: read the market, check the physical
# story, price the optionality, run the book, then the reference shelf.
NAV_SECTIONS = {
    "Markets": [
        "📊  Dashboard", "📈  Forward Curves", "🔀  Calendar Spreads",
        "🏭  Cracks, Crush & Arbs", "📡  Signal Scanner",
    ],
    "Fundamentals & Flows": [
        "📦  Storage & Carry", "🧭  COT Positioning", "🛢️  EIA Fundamentals",
        "🗺️  Regional Balances", "📅  Event Calendar",
    ],
    "Analytics": [
        "🔗  Correlation", "🌡️  Seasonality", "🎲  Monte Carlo",
    ],
    "Volatility & Options": [
        "🎯  Options Pricer", "🌊  Vol Surface",
    ],
    "Book & Risk": [
        "📒  Trade Blotter", "⚠️  Portfolio Risk",
    ],
    "Reference": [
        "🌍  Macro Rates", "ℹ️  About",
    ],
}
ALL_PAGES = [p for ps in NAV_SECTIONS.values() for p in ps]
DEFAULT_PAGE = ALL_PAGES[0]


def _nav_to(page_name: str) -> None:
    st.session_state.nav_page = page_name


def render_sidebar(marks: MarkBoard) -> str:
    if st.session_state.get("nav_page") not in ALL_PAGES:
        st.session_state.nav_page = DEFAULT_PAGE
    current = st.session_state.nav_page

    with st.sidebar:
        st.markdown(f"""
        <div style="padding:6px 0 10px 0">
          <span style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;
          font-weight:700;color:{AMBER}">S&D DESK</span><br>
          <span style="font-size:0.68rem;color:{GRAY}">COMMODITY TRADING &nbsp;·&nbsp; LIVE MTM</span>
        </div>""", unsafe_allow_html=True)

        for section, pages in NAV_SECTIONS.items():
            st.markdown(f'<div class="nav-sec">{section}</div>', unsafe_allow_html=True)
            for p in pages:
                if p == current:
                    st.markdown(f'<div class="nav-active">{p}</div>', unsafe_allow_html=True)
                else:
                    st.button(p, key=f"nav_{p}", type="tertiary",
                              use_container_width=True, on_click=_nav_to, args=(p,))
        page = current
        st.markdown("---")

        n_ok  = sum(1 for v in marks.values() if v is not None)
        n_all = len(COMMODITIES)
        stale = marks.stale_names()
        colr  = GREEN if n_ok == n_all else (AMBER if n_ok else RED)
        st.markdown(
            f'<span class="badge" style="border-color:{colr};color:{colr}">'
            f'FEED {n_ok}/{n_all} MARKED</span>'
            + (f' <span class="badge badge-amber">{len(stale)} STALE</span>' if stale else ""),
            unsafe_allow_html=True)

        if st.button("🔄 Refresh marks", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        with st.expander("🔑 Data keys (optional)"):
            st.text_input("EIA API key", type="password", key="eia_key",
                          help="Free at eia.gov/opendata — or set st.secrets['EIA_KEY']")
            st.text_input("FRED API key", type="password", key="fred_key",
                          help="Free at fred.stlouisfed.org — or set st.secrets['FRED_KEY']")

        with st.expander("🔧 Feed diagnostics"):
            if FEED_LOG:
                st.code("\n".join(list(FEED_LOG)[-25:]), language=None)
            else:
                st.caption("No feed warnings this session.")
            st.caption("Every fetch failure lands here — nothing is silently swallowed.")

        st.markdown("---")
        st.markdown(
            f"""<div style="font-size:0.62rem;color:{GRAY};line-height:1.5">
            Marks &amp; strips: Yahoo Finance (delayed). Fundamentals: EIA v2.
            Macro: FRED. No fabricated data — unavailable feeds show as unavailable.<br>
            by <b>Adam EL GBOURI</b></div>""",
            unsafe_allow_html=True)
    return page


def render_header(marks: MarkBoard, title: str, subtitle: str) -> None:
    n_ok = sum(1 for v in marks.values() if v is not None)
    live = n_ok > 0
    asofs = [marks.asof(n) for n in COMMODITIES if marks.asof(n)]
    latest = max(asofs).isoformat() if asofs else "—"
    badge = (f'<span class="badge badge-green">LIVE · {latest}</span>' if live
             else '<span class="badge badge-red">FEED DOWN</span>')
    st.markdown(
        f"""<div style="display:flex;justify-content:space-between;align-items:baseline">
        <div><h2 style="margin-bottom:0">{title}</h2>
        <span style="color:{GRAY};font-size:0.8rem">{subtitle}</span></div>
        <div>{badge}</div></div>""",
        unsafe_allow_html=True)
    st.markdown("")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard(marks: MarkBoard) -> None:
    render_header(marks, "Trading Dashboard", "Live board · front-month marks with settle dates")

    panel = panel_years(0.3)
    c1, c2, c3, c4 = st.columns(4)
    n_ok = sum(1 for v in marks.values() if v is not None)
    kpi(c1, "Contracts marked", f"{n_ok}/{len(COMMODITIES)}",
        "live Yahoo front months", GREEN if n_ok == len(COMMODITIES) else AMBER)
    stale = marks.stale_names()
    kpi(c2, "Stale marks", f"{len(stale)}",
        "settles older than 4 days" if stale else "all fresh", RED if stale else GREEN)

    movers = []
    if not panel.empty:
        for n in COMMODITIES:
            if n in panel.columns:
                s = panel[n].dropna()
                if len(s) >= 2 and s.iloc[-2]:
                    movers.append((n, (s.iloc[-1] / s.iloc[-2] - 1) * 100))
    if movers:
        up = max(movers, key=lambda x: x[1])
        dn = min(movers, key=lambda x: x[1])
        kpi(c3, "Top mover", f"{up[1]:+.2f}%", COMMODITIES[up[0]]["ticker"], GREEN)
        kpi(c4, "Worst mover", f"{dn[1]:+.2f}%", COMMODITIES[dn[0]]["ticker"], RED)
    else:
        kpi(c3, "Top mover", "—", "panel unavailable", GRAY)
        kpi(c4, "Worst mover", "—", "panel unavailable", GRAY)

    st.markdown("### Board")
    rows = []
    for n, c in COMMODITIES.items():
        p = marks.get(n)
        chg = None
        if not panel.empty and n in panel.columns:
            s = panel[n].dropna()
            if len(s) >= 2 and s.iloc[-2]:
                chg = (s.iloc[-1] / s.iloc[-2] - 1) * 100
        rows.append(dict(Contract=n, Sector=c["sector"], Unit=c["unit"],
                         Mark=(f"{p:,.2f}" if p is not None else "NO MARK"),
                         AsOf=(marks.asof(n).isoformat() if marks.asof(n) else "—"),
                         Chg=(f"{chg:+.2f}%" if chg is not None else "—")))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 height=420)

    st.markdown("### Performance treemap")
    cA, cB = st.columns(2)
    d_a = cA.date_input("From", date.today() - timedelta(days=30), max_value=date.today())
    d_b = cB.date_input("To", date.today(), max_value=date.today())
    tm = fetch_board_closes(d_a, d_b)
    if tm.empty:
        st.info("Treemap unavailable — feed did not return settles for that window.")
    else:
        fig = px.treemap(tm, path=["sector", "name"], values=[1] * len(tm),
                         color="chg", color_continuous_scale=[RED, PANEL, GREEN],
                         color_continuous_midpoint=0, custom_data=["chg_str", "px"])
        fig.update_traces(hovertemplate="<b>%{label}</b><br>%{customdata[0]}<br>last %{customdata[1]}")
        st.plotly_chart(_styled(fig, 460), use_container_width=True)

    st.markdown("### Chart")
    sel = st.selectbox("Contract", list(COMMODITIES), label_visibility="collapsed")
    hist = panel_years(2)
    if hist.empty or sel not in hist.columns or hist[sel].dropna().empty:
        st.info("History unavailable.")
        return
    s = hist[sel].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name=sel, line=dict(color=AMBER, width=1.6)))
    ma50, ma200 = s.rolling(50).mean(), s.rolling(200).mean()
    fig.add_trace(go.Scatter(x=s.index, y=ma50, name="MA50", line=dict(color=BLUE, width=1)))
    fig.add_trace(go.Scatter(x=s.index, y=ma200, name="MA200", line=dict(color=GRAY, width=1)))
    fig.update_layout(title=f"{sel} — 2y continuous front month ({COMMODITIES[sel]['unit']})")
    st.plotly_chart(_styled(fig, 420), use_container_width=True)


def page_curve(marks: MarkBoard) -> None:
    render_header(marks, "Forward Curves", "Live dated strips — every point is a traded settle")
    sel = st.selectbox("Contract", list(COMMODITIES))
    if require_mark(marks, sel) is None:
        return
    strip = fetch_forward_strip(sel)
    if strip.empty:
        st.error("**NO LIVE STRIP** — dated contracts returned nothing. "
                 "No curve is fitted in its place.")
        return

    c = COMMODITIES[sel]
    carry = implied_carry(strip)
    f1, fn = float(strip["price"].iloc[0]), float(strip["price"].iloc[-1])
    slope = (fn / f1 - 1) * 100 if f1 else 0.0
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Front", f"{f1:,.2f}", f"{strip['label'].iloc[0]} · {c['unit']}")
    kpi(k2, "Back", f"{fn:,.2f}", strip["label"].iloc[-1])
    kpi(k3, "Curve shape", "CONTANGO" if slope > 0.5 else "BACKWARDATION" if slope < -0.5 else "FLAT",
        f"{slope:+.2f}% front→back", RED if slope > 0.5 else GREEN if slope < -0.5 else GRAY)
    ann1 = float(carry["roll_yield"].iloc[1]) if len(carry) > 1 else 0.0
    kpi(k4, "M1→M2 roll (ann.)", f"{ann1:+.1f}%",
        "positive = backwardation pays longs", GREEN if ann1 > 0 else RED)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strip["label"], y=strip["price"], mode="lines+markers",
                             line=dict(color=AMBER, width=2), marker=dict(size=7),
                             name="Strip"))
    fig.update_layout(title=f"{sel} forward strip ({c['unit']}) — calendar T on hover")
    fig.update_traces(customdata=np.stack([strip["T"], strip["asof"]], axis=-1),
                      hovertemplate="%{x}<br>%{y:,.2f}<br>T=%{customdata[0]:.2f}y · settle %{customdata[1]}")
    st.plotly_chart(_styled(fig, 400), use_container_width=True)

    st.markdown("### Strip detail")
    show = carry[["label", "delivery", "T", "price", "asof", "spread_vs_M1",
                  "spread_pct", "roll_yield"]].rename(columns={
        "label": "Contract", "delivery": "Delivery", "T": "T (yrs)", "price": "Price",
        "asof": "Settle date", "spread_vs_M1": "vs M1", "spread_pct": "vs M1 %",
        "roll_yield": "Roll yield ann.%"})
    st.dataframe(show.style.format({"T (yrs)": "{:.2f}", "Price": "{:,.2f}",
                                    "vs M1": "{:+,.2f}", "vs M1 %": "{:+.2f}",
                                    "Roll yield ann.%": "{:+.1f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("T is calendar year-fraction to mid-delivery. Deferred months trade thin — "
               "each settle date is shown because a two-day-old settle is still a settle, "
               "not today's price.")


def page_spreads(marks: MarkBoard) -> None:
    render_header(marks, "Calendar Spreads", "Dated M1−Mn with a 2y percentile off the same contracts")
    sel = st.selectbox("Contract", list(COMMODITIES))
    strip = fetch_forward_strip(sel)
    if strip.empty or len(strip) < 2:
        st.error("**NO LIVE STRIP** — cannot build a calendar spread without dated contracts.")
        return
    c1, c2 = st.columns(2)
    max_leg = len(strip)
    near_i = c1.selectbox("Near leg", range(max_leg), format_func=lambda i: strip["label"].iloc[i])
    far_opts = [i for i in range(max_leg) if i != near_i]
    far_i = c2.selectbox("Far leg", far_opts, format_func=lambda i: strip["label"].iloc[i])

    live = float(strip["price"].iloc[near_i] - strip["price"].iloc[far_i])
    unit = COMMODITIES[sel]["unit"]

    hist = fetch_spread_history(sel, near_i, far_i)
    pct = None
    if not hist.empty and len(hist) > 60:
        pct = float((hist["spread"] < live).mean() * 100)

    k1, k2, k3 = st.columns(3)
    kpi(k1, "Live spread", f"{live:+,.3f}", f"{strip['label'].iloc[near_i]} − {strip['label'].iloc[far_i]} · {unit}")
    kpi(k2, "2y percentile", f"{pct:.0f}%" if pct is not None else "n/a",
        "of this dated pair's own history",
        RED if (pct or 50) >= 80 else GREEN if (pct or 50) <= 20 else AMBER)
    kpi(k3, "History depth", f"{len(hist)}" if not hist.empty else "0", "daily observations")
    st.markdown(pctile_badge(pct), unsafe_allow_html=True)

    if hist.empty:
        st.info("Pair history unavailable for these dated contracts.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["spread"], line=dict(color=TEAL, width=1.4),
                             name="Spread"))
    fig.add_hline(y=live, line=dict(color=AMBER, dash="dash"),
                  annotation_text=f"live {live:+.3f}")
    fig.update_layout(title=f"{sel} {hist.attrs.get('near_label','M1')}−{hist.attrs.get('far_label','M2')} — 2y of the SAME dated pair")
    st.plotly_chart(_styled(fig, 420), use_container_width=True)
    st.caption("History tracks the exact dated pair, not a continuous proxy — "
               "the percentile is of the thing itself.")


def page_structures(marks: MarkBoard) -> None:
    render_header(marks, "Cracks, Crush & Arbs", "Physical-margin structures off live legs")
    sel = st.selectbox("Structure", list(STRUCTURES))
    spec = STRUCTURES[sel]
    st.caption(spec["desc"])

    legs_px = {}
    missing = []
    for n, _ in spec["legs"]:
        v = marks.get(n)
        if v is None:
            missing.append(n)
        legs_px[n] = v
    if missing:
        st.error(f"**NO LIVE MARK** for: {', '.join(missing)} — structure disabled.")
        return

    crush_note = ""
    if spec["kind"] == "crack":
        val = sum(r * to_bbl(n, legs_px[n]) for n, r in spec["legs"]) / spec["divisor"]
    elif spec["kind"] == "crush":
        mm = matched_month_crush(fetch_all_strips())
        if mm:
            val = mm["value"]
            crush_note = f"matched delivery month **{mm['label']}** across ZS/ZM/ZL"
        else:
            meal = legs_px["Soybean Meal (ZM)"] * CRUSH_MEAL_LB / LB_PER_SHORT_TON
            oil  = legs_px["Soybean Oil (ZL)"] / 100.0 * CRUSH_OIL_LB
            bean = legs_px["Soybeans (ZS)"] / 100.0
            val = meal + oil - bean
            crush_note = "front months (no common delivery month on the strips right now)"
    elif spec["kind"] == "ratio":
        a, b = spec["legs"][0][0], spec["legs"][1][0]
        val = legs_px[a] / legs_px[b]
    else:
        val = sum(r * legs_px[n] for n, r in spec["legs"]) / spec["divisor"]

    hist = fetch_structure_history(sel)
    pct = float((hist["value"] < val).mean() * 100) if not hist.empty and len(hist) > 60 else None
    lo, hi = spec["typical"]

    k1, k2, k3 = st.columns(3)
    kpi(k1, "Live value", f"{val:,.2f}", spec["unit"])
    kpi(k2, "3y percentile", f"{pct:.0f}%" if pct is not None else "n/a", "of front-month history",
        RED if (pct or 50) >= 80 else GREEN if (pct or 50) <= 20 else AMBER)
    kpi(k3, "Typical range", f"{lo} – {hi}", "rough historical band", GRAY)
    st.markdown(pctile_badge(pct), unsafe_allow_html=True)
    if crush_note:
        st.caption(f"Live crush uses {crush_note}.")

    if hist.empty:
        st.info("Structure history unavailable.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["value"], line=dict(color=PURPLE, width=1.4),
                             name=sel))
    fig.add_hline(y=val, line=dict(color=AMBER, dash="dash"), annotation_text=f"live {val:,.2f}")
    fig.update_layout(title=f"{sel} — 3y ({spec['unit']})")
    st.plotly_chart(_styled(fig, 420), use_container_width=True)
    st.caption("History is computed on continuous front months, which Yahoo does NOT "
               "roll-adjust — expect a level jump at each roll. The live print above is "
               "the honest one; treat the history as shape, not level gospel.")


def page_correlation(marks: MarkBoard) -> None:
    render_header(marks, "Correlation", "Daily log-return correlation off the shared 15y panel")
    c1, c2 = st.columns(2)
    years  = c1.slider("History (years)", 1, 5, 2)
    window = c2.slider("Window (trading days)", 60, 750, 252, step=21)
    corr = correlation_matrix(years, window)
    if corr.empty:
        st.error("**NO PANEL** — correlation unavailable.")
        return
    short = {n: COMMODITIES[n]["ticker"] for n in corr.columns}
    cs = corr.rename(index=short, columns=short)
    fig = go.Figure(go.Heatmap(z=cs.values, x=cs.columns, y=cs.index,
                               colorscale=[[0, RED], [0.5, PANEL], [1, GREEN]],
                               zmin=-1, zmax=1, text=np.round(cs.values, 2),
                               texttemplate="%{text}", textfont=dict(size=8)))
    fig.update_layout(title=f"Correlation — last {window} sessions within {years}y")
    st.plotly_chart(_styled(fig, 640), use_container_width=True)
    st.caption("This matrix is what the parametric VaR uses. NaNs (short history) are "
               "never zero-filled downstream — the VaR falls back to a conservative sum "
               "and says so.")


def page_seasonality(marks: MarkBoard) -> None:
    render_header(marks, "Seasonality", "Monthly return distributions — physical cycles, honestly caveated")
    seasonal = [n for n, c in COMMODITIES.items() if c.get("seasonal")]
    sel = st.selectbox("Contract (seasonal set)", seasonal)
    years = st.slider("Lookback (years)", 5, 15, 10)
    s = seasonality(sel, years)
    if s.empty:
        st.error("**NO HISTORY** — seasonality unavailable.")
        return
    this_month = date.today().month
    d = s[s["month"] == this_month]["ret"]
    k1, k2, k3 = st.columns(3)
    kpi(k1, f"{MONTH_NAMES[this_month-1]} median", f"{d.median():+.2f}%" if len(d) else "n/a",
        f"{len(d)} obs", GREEN if len(d) and d.median() > 0 else RED)
    kpi(k2, "Hit rate", f"{(d > 0).mean()*100:.0f}%" if len(d) else "n/a", "share of positive years")
    best = s.groupby("month")["ret"].median()
    kpi(k3, "Best month (median)", MONTH_NAMES[int(best.idxmax())-1], f"{best.max():+.2f}%")

    fig = go.Figure()
    for m in range(1, 13):
        dm = s[s["month"] == m]["ret"]
        fig.add_trace(go.Box(y=dm, name=MONTH_NAMES[m-1],
                             marker_color=AMBER if m == this_month else BLUE))
    fig.update_layout(title=f"{sel} — monthly return distribution, {years}y", showlegend=False)
    st.plotly_chart(_styled(fig, 460), use_container_width=True)
    st.caption("Computed on the continuous front month, which is NOT roll-adjusted: in "
               "persistent contango (NG most years) roll months carry a systematic "
               "negative bias that is a property of the SERIES, not the month. Read the "
               "shape with that in mind.")


def page_eia(marks: MarkBoard) -> None:
    render_header(marks, "EIA Fundamentals", "Weekly US inventories & production — the real S&D data")
    key = eia_key()
    if not key:
        st.warning("Enter an EIA API key in the sidebar (or set st.secrets['EIA_KEY']). "
                   "Free at eia.gov/opendata. No key — no data; nothing is simulated here.")
        return
    covered = [n for n in COMMODITIES if n in EIA_MAP]
    sel = st.selectbox("Contract", covered)
    for series_name in EIA_MAP[sel]:
        df = fetch_eia(series_name, key)
        meta = EIA_SERIES[series_name]
        st.markdown(f"#### {series_name} ({meta['unit']})")
        if df.empty:
            st.info("Feed unavailable for this series (see sidebar diagnostics).")
            continue
        last, prev = float(df["value"].iloc[-1]), float(df["value"].iloc[-2])
        yr_ago = df[df.index <= df.index[-1] - pd.DateOffset(years=1)]
        yoy = float(yr_ago["value"].iloc[-1]) if not yr_ago.empty else None
        c1, c2, c3 = st.columns(3)
        kpi(c1, "Latest", f"{last:,.0f}", str(df.index[-1].date()))
        kpi(c2, "WoW", f"{last-prev:+,.0f}", f"{(last/prev-1)*100:+.2f}%",
            GREEN if last < prev else RED)
        kpi(c3, "vs 1y ago", f"{last-yoy:+,.0f}" if yoy else "n/a",
            f"{(last/yoy-1)*100:+.2f}%" if yoy else "")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["value"], line=dict(color=TEAL, width=1.4)))
        st.plotly_chart(_styled(fig, 300), use_container_width=True)


def page_regional(marks: MarkBoard) -> None:
    render_header(marks, "Regional Balances", "Static IEA/USDA-style S&D estimates — NOT a live feed")
    covered = list(REGIONAL_DATA)
    sel = st.selectbox("Contract", covered)
    df = pd.DataFrame(REGIONAL_DATA[sel])
    df["balance"] = df["supply"] - df["demand"]
    unit = COMMODITIES[sel]["reg_unit"]
    st.caption(f"Units: {COMMODITIES[sel]['reg_label']}. These are static annual estimates "
               "for orientation — the one table on this desk that is not live, and it says so.")
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=df["lat"], lon=df["lon"], text=df["region"], customdata=df[["balance"]],
        marker=dict(size=np.abs(df["balance"]) * (30 / max(np.abs(df["balance"]).max(), 1)) + 8,
                    color=[GREEN if b > 0 else RED for b in df["balance"]],
                    line=dict(color=BORDER, width=1)),
        hovertemplate="%{text}<br>net balance %{customdata[0]:+,.0f} " + unit +
                      "<extra></extra>"))
    fig.update_geos(bgcolor=BG, landcolor=PANEL, coastlinecolor=BORDER, showcountries=True,
                    countrycolor=BORDER)
    fig.update_layout(title=f"{sel} — net exporters (green) vs importers (red), {unit}")
    st.plotly_chart(_styled(fig, 430), use_container_width=True)
    bar = df.sort_values("balance")
    fig2 = go.Figure(go.Bar(x=bar["balance"], y=bar["region"], orientation="h",
                            marker_color=[GREEN if b > 0 else RED for b in bar["balance"]]))
    fig2.update_layout(title=f"Net balance by region ({unit})")
    st.plotly_chart(_styled(fig2, 360), use_container_width=True)


def page_options(marks: MarkBoard) -> None:
    render_header(marks, "Options Pricer", "Black-76 on the live dated forward")
    sel = st.selectbox("Underlying", list(COMMODITIES))
    mark = require_mark(marks, sel)
    if mark is None:
        return
    c = COMMODITIES[sel]
    strip = fetch_forward_strip(sel)

    c1, c2, c3 = st.columns(3)
    if not strip.empty:
        leg = c1.selectbox("Expiry (strip month)", range(len(strip)),
                           format_func=lambda i: f"{strip['label'].iloc[i]}  (T={strip['T'].iloc[i]:.2f}y)")
        F = float(strip["price"].iloc[leg])
        T_def = float(strip["T"].iloc[leg])
        c1.caption(f"Forward = live {strip['label'].iloc[leg]} settle {F:,.2f} "
                   f"(dated {strip['asof'].iloc[leg]})")
    else:
        F, T_def = mark, 0.25
        c1.caption("Strip unavailable — anchored on the front-month mark.")
    K = c2.number_input("Strike", value=float(round(F, 2)), step=max(F * 0.005, 0.01))
    T_in = c2.number_input("Tenor (years, calendar)", value=float(round(T_def, 3)),
                           min_value=0.0, step=0.05)
    rv = realised_vol(sel)
    sigma = c3.slider("Vol (ann.)", 0.05, 1.50, float(rv or c["vol"]), 0.01,
                      help="Seeded with 60d realised vol off the panel when available.")
    r = c3.slider("Rate", 0.0, 0.10, 0.05, 0.005)
    otype = c3.radio("Type", ["call", "put"], horizontal=True)

    g = black76(F, K, T_in, r, sigma, otype)
    mult = price_multiplier(sel)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Premium", f"{g['price']:,.4f}", f"{c['unit']} · ${g['price']*mult:,.0f}/lot")
    kpi(k2, "Delta", f"{g['delta']:+.3f}", f"${g['delta']*F*mult:,.0f} delta-cash/lot")
    kpi(k3, "Gamma", f"{g['gamma']:.5f}", "per unit²")
    kpi(k4, "Vega", f"{g['vega']:,.4f}", "per vol pt · per unit")
    kpi(k5, "Theta/day", f"{g['theta']:,.5f}", f"${g['theta']*mult:,.0f}/lot/day",
        RED if g["theta"] < 0 else GREEN)

    Ks = np.linspace(F * 0.6, F * 1.4, 60)
    prem = [black76(F, k, T_in, r, sigma, otype)["price"] for k in Ks]
    intr = [max(F - k, 0) if otype == "call" else max(k - F, 0) for k in Ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ks, y=prem, name="Premium", line=dict(color=AMBER, width=2)))
    fig.add_trace(go.Scatter(x=Ks, y=intr, name="Intrinsic", line=dict(color=GRAY, dash="dot")))
    fig.add_vline(x=K, line=dict(color=BLUE, dash="dash"), annotation_text=f"K={K:,.2f}")
    fig.update_layout(title=f"{sel} {otype} — premium vs strike (F={F:,.2f}, T={T_in:.2f}y)")
    st.plotly_chart(_styled(fig, 400), use_container_width=True)
    st.caption("Anchored on the live dated forward with CALENDAR tenor. σ is a realised-vol "
               "seed, not an implied quote — this desk has no listed-options feed, and "
               "won't pretend otherwise.")


def page_vol_surface(marks: MarkBoard) -> None:
    render_header(marks, "Vol Surface", "Parametric sticky-moneyness surface — illustrative shape")
    sel = st.selectbox("Underlying", list(COMMODITIES))
    mark = require_mark(marks, sel)
    if mark is None:
        return
    rv = realised_vol(sel) or COMMODITIES[sel]["vol"]
    c1, c2, c3 = st.columns(3)
    atm  = c1.slider("ATM vol", 0.05, 1.2, float(rv), 0.01)
    skew = c2.slider("Skew", -0.5, 0.5, -0.05, 0.01)
    curv = c3.slider("Smile curvature", 0.0, 0.5, 0.05, 0.01)
    mats, Ks, Z = vol_surface_fn(mark, atm, skew, curv)
    fig = go.Figure(go.Surface(x=Ks, y=mats, z=Z * 100, colorscale="Viridis"))
    fig.update_layout(title=f"{sel} parametric surface (ATM {atm*100:.0f}%)",
                      scene=dict(xaxis_title="Strike", yaxis_title="T (yrs)",
                                 zaxis_title="Vol %",
                                 bgcolor=BG),
                      height=560, paper_bgcolor=BG,
                      font=dict(family="JetBrains Mono, monospace", size=10, color=TEXT))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A stated PARAMETRISATION seeded off realised vol — not calibrated to listed "
               "quotes (no options feed). Use it to reason about shape, not to price skew risk.")


def page_blotter(marks: MarkBoard) -> None:
    render_header(marks, "Trade Blotter", "Futures (front or dated), options — marked live, per-book storage")
    if "positions" not in st.session_state:
        st.session_state.positions = blotter_load()
    positions = st.session_state.positions

    tab_f, tab_o, tab_io = st.tabs(["BOOK FUTURE", "BOOK OPTION", "BOOK MANAGEMENT"])

    with tab_f:
        c1, c2, c3, c4, c5 = st.columns([2.2, 1.6, 1, 1, 1.2])
        n = c1.selectbox("Contract", list(COMMODITIES), key="bf_n")
        strip = fetch_forward_strip(n)
        month_opts = ["Front (continuous)"] + (
            [f"{r.label}  ({r.ticker})" for r in strip.itertuples()] if not strip.empty else [])
        msel = c2.selectbox("Contract month", month_opts, key="bf_m",
                            help="Dated lines mark to their own ticker — clean P&L through rolls, no roll bleed.")
        side = c3.radio("Side", ["Long", "Short"], key="bf_s")
        lots = c4.number_input("Lots", 1, 500, 1, key="bf_l")
        if msel == "Front (continuous)":
            default_px = marks.get(n) or 0.0
        else:
            default_px = float(strip["price"].iloc[month_opts.index(msel) - 1])
        entry = c5.number_input("Entry", value=float(round(default_px, 4)), key="bf_e",
                                format="%.4f")
        if st.button("Book future", type="primary"):
            p = dict(commodity=n, kind="future", side=side, lots=int(lots),
                     entry=float(entry), trade_date=date.today().isoformat())
            if msel != "Front (continuous)":
                i = month_opts.index(msel) - 1
                p["strip_ticker"] = str(strip["ticker"].iloc[i])
                p["strip_label"]  = str(strip["label"].iloc[i])
            positions.append(p)
            blotter_save(positions)
            st.rerun()

    with tab_o:
        c1, c2, c3 = st.columns(3)
        n = c1.selectbox("Underlying", list(COMMODITIES), key="bo_n")
        otype = c1.radio("Type", ["call", "put"], horizontal=True, key="bo_t")
        side = c1.radio("Side", ["Long", "Short"], horizontal=True, key="bo_s")
        base = marks.get(n) or 100.0
        strike = c2.number_input("Strike", value=float(round(base, 2)), key="bo_k")
        tenor = c2.number_input("Tenor (yrs)", 0.05, 3.0, 0.25, 0.05, key="bo_ten",
                                help="Tenor AT BOOKING. The book ages it daily from today.")
        lots = c2.number_input("Lots", 1, 500, 1, key="bo_l")
        rv = realised_vol(n)
        vol = c3.slider("Booked vol", 0.05, 1.5, float(rv or COMMODITIES[n]["vol"]), 0.01, key="bo_v")
        theo = black76(base, strike, tenor, 0.05, vol, otype)["price"]
        prem = c3.number_input("Premium paid/received (per unit)", value=float(round(theo, 4)),
                               key="bo_p", format="%.4f")
        c3.caption(f"Black-76 theo at booking: {theo:,.4f}")
        if st.button("Book option", type="primary"):
            positions.append(dict(commodity=n, kind="option", side=side, lots=int(lots),
                                  opt_type=otype, strike=float(strike), tenor=float(tenor),
                                  vol=float(vol), premium=float(prem), entry=float(prem),
                                  trade_date=date.today().isoformat()))
            blotter_save(positions)
            st.rerun()

    with tab_io:
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Export book (JSON)", blotter_serialise(positions),
                           file_name=f"blotter_{_book_id()}.json", use_container_width=True)
        up = c2.file_uploader("Import book", type="json", label_visibility="collapsed")
        if up is not None and c2.button("Load imported book", use_container_width=True):
            try:
                st.session_state.positions = blotter_deserialise(up.read().decode())
                blotter_save(st.session_state.positions)
                st.success("Book loaded. Legacy options without a trade date restart "
                           "their tenor from today (stated, not silent).")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")
        if c3.button("🗑️ Flatten book", use_container_width=True):
            st.session_state.positions = []
            blotter_save([])
            st.rerun()
        st.caption(f"Book id **{_book_id()}** — carried in this page's URL (?book=…): "
                   "bookmark it to keep the book across sessions. Server storage on "
                   "Streamlit Cloud is ephemeral and wiped on redeploy; the JSON export "
                   "is the durable copy. Books are per-URL, not shared between visitors.")

    if not positions:
        st.info("Book is flat. Book a future or an option above.")
        return

    st.markdown("### Marked positions")
    rows, tot_pnl, tot_notional, unmarked = [], 0.0, 0.0, 0
    for i, p in enumerate(positions):
        n = p["commodity"]
        mult = price_multiplier(n)
        sign = 1 if p["side"] == "Long" else -1
        if p.get("kind", "future") == "option":
            base = marks.get(n)
            if base is None:
                unmarked += 1
                rows.append(dict(idx=i, Position=_position_label(p), Side=p["side"],
                                 Lots=p["lots"], Entry=p["entry"], Mark="NO MARK",
                                 PnL=None, Note="feed down"))
                continue
            T = option_time_remaining(p)
            g = black76(base, p["strike"], T, 0.05, p["vol"], p["opt_type"])
            pnl = sign * (g["price"] - p["premium"]) * mult * p["lots"]
            note = "EXPIRED — intrinsic" if T <= 0 else f"T={T:.2f}y left"
            tot_pnl += pnl
            tot_notional += abs(g["price"]) * mult * p["lots"]
            rows.append(dict(idx=i, Position=_position_label(p), Side=p["side"],
                             Lots=p["lots"], Entry=p["premium"],
                             Mark=round(g["price"], 4), PnL=pnl, Note=note))
        else:
            base = position_base_price(p, marks)
            if base is None:
                unmarked += 1
                note = ("dated contract rolled off / feed down" if p.get("strip_ticker")
                        else "feed down")
                rows.append(dict(idx=i, Position=_position_label(p), Side=p["side"],
                                 Lots=p["lots"], Entry=p["entry"], Mark="NO MARK",
                                 PnL=None, Note=note))
                continue
            pnl = sign * (base - p["entry"]) * mult * p["lots"]
            tot_pnl += pnl
            tot_notional += notional_per_lot(n, base) * p["lots"]
            note = ""
            if p.get("strip_ticker"):
                dm = dated_mark(p["strip_ticker"])
                note = f"dated · settle {dm['asof']}" if dm else ""
            elif marks.is_stale(n):
                note = f"stale settle {marks.asof(n)}"
            rows.append(dict(idx=i, Position=_position_label(p), Side=p["side"],
                             Lots=p["lots"], Entry=p["entry"], Mark=round(base, 4),
                             PnL=pnl, Note=note))

    k1, k2, k3 = st.columns(3)
    kpi(k1, "Open P&L", f"${tot_pnl:+,.0f}", f"{len(positions)} position(s)",
        GREEN if tot_pnl >= 0 else RED)
    kpi(k2, "Gross notional (marked)", f"${tot_notional:,.0f}", "delta-cash for options")
    kpi(k3, "Unmarked lines", f"{unmarked}", "excluded from totals — never proxied",
        RED if unmarked else GREEN)

    df = pd.DataFrame(rows)
    st.dataframe(df.drop(columns=["idx"]).style.format({"Entry": "{:,.4f}", "PnL": "{:+,.0f}"},
                                                       na_rep="—"),
                 use_container_width=True, hide_index=True)

    kill = st.selectbox("Close a line", ["—"] + [f"{r['idx']}: {r['Position']} {r['Side']} x{r['Lots']}"
                                                 for r in rows])
    if kill != "—" and st.button("Close selected"):
        positions.pop(int(kill.split(":")[0]))
        blotter_save(positions)
        st.rerun()

    st.markdown("### P&L attribution — price vs roll")
    attr = roll_pnl(positions, marks)
    if attr:
        adf = pd.DataFrame(attr)
        st.dataframe(adf.style.format({"PricePnL": "{:+,.0f}", "MonthlyRoll": "{:+,.0f}",
                                       "M1M2": "{:+,.3f}", "RollAnnPct": "{:+.1f}"},
                                      na_rep="—"),
                     use_container_width=True, hide_index=True)
        st.caption("Front-month futures only. Roll = M1−M2 carry bleed, annualised on the "
                   "CALENDAR gap between the two contracts. Dated lines have no roll bleed "
                   "— that is why you book them — and options attribute through Greeks below.")
    else:
        st.caption("No front-month futures to attribute.")

    st.markdown("### Net book Greeks")
    gk = book_greeks(positions, marks)
    t = gk["total"]
    g1, g2, g3, g4 = st.columns(4)
    kpi(g1, "Delta ($/1.0 move)", f"{t['delta']:+,.0f}", "cash per unit price move")
    kpi(g2, "Gamma", f"{t['gamma']:+,.2f}", "delta change per unit")
    kpi(g3, "Vega ($/vol pt)", f"{t['vega']:+,.0f}", "options only",
        PURPLE if t["vega"] else GRAY)
    kpi(g4, "Theta ($/day)", f"{t['theta']:+,.0f}", "options age daily now",
        RED if t["theta"] < 0 else GREEN)


def page_risk(marks: MarkBoard) -> None:
    render_header(marks, "Portfolio Risk", "Delta-equivalent VaR/ES, historical replay, dated stress")
    positions = st.session_state.get("positions") or blotter_load()
    if not positions:
        st.info("Book is flat — book positions in the Trade Blotter first.")
        return

    # Work on COPIES: the old page mutated the blotter's vols in session_state.
    pos = [dict(p) for p in positions]

    c1, c2, c3, c4 = st.columns(4)
    conf = c1.select_slider("Confidence", [0.90, 0.95, 0.99], 0.95)
    horizon = c2.select_slider("Horizon (days)", [1, 5, 10], 1)
    vol_src = c3.radio("Risk vol source", ["Realised 60d (live)", "Registry"],
                       help="Applied to each position's UNDERLYING for risk. Booked option σ still prices the delta.")
    diversified = c4.toggle("Use live correlation", value=True)

    for p in pos:
        if vol_src.startswith("Realised"):
            rv = realised_vol(p["commodity"])
            p["risk_vol"] = rv if rv else COMMODITIES[p["commodity"]]["vol"]
        else:
            p["risk_vol"] = COMMODITIES[p["commodity"]]["vol"]

    corr = correlation_matrix(2, 252)
    res = portfolio_var(pos, marks, corr, conf, horizon, diversified)
    if not res["rows"]:
        st.error("No position could be marked — VaR unavailable (no proxying).")
        return

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, f"VaR {conf:.0%} / {horizon}d", f"${res['var']:,.0f}",
        "diversified" if res["corr_used"] else "conservative sum", RED)
    kpi(k2, "Expected Shortfall", f"${res['es']:,.0f}", "mean loss beyond VaR", RED)
    kpi(k3, "Undiversified VaR", f"${res['undiversified']:,.0f}", "sum of standalones")
    kpi(k4, "Diversification benefit", f"{res['benefit']:.1f}%",
        "netting + correlation", GREEN if res["benefit"] > 0 else GRAY)

    if res["corr_used"]:
        st.caption("Correlation matrix in use — live 252d off the shared panel.")
    elif res["reason"]:
        st.warning(f"Diversified VaR not applied: {res['reason']}.")

    df = pd.DataFrame(res["rows"])
    st.dataframe(df.style.format({"DeltaCash": "{:+,.0f}", "Vol": "{:.1f}%",
                                  "StandaloneVaR": "{:,.0f}", "StandaloneES": "{:,.0f}",
                                  "ComponentVaR": "{:+,.0f}", "PctOfVaR": "{:+.1f}%"}),
                 use_container_width=True, hide_index=True)
    st.caption("Options enter at Black-76 delta-cash and carry their UNDERLYING's vol — "
               "gamma is not charged at this horizon (stated limitation, not an oversight).")

    st.markdown("### Historical-simulation VaR")
    hv = historical_var(pos, marks, conf, horizon)
    if hv.get("available"):
        h1, h2, h3 = st.columns(3)
        kpi(h1, f"Hist VaR {conf:.0%}", f"${hv['var']:,.0f}", hv["note"], RED)
        kpi(h2, "Hist ES", f"${hv['es']:,.0f}", "empirical tail mean", RED)
        kpi(h3, "Worst window", f"${hv['worst_pnl']:+,.0f}", f"ending {hv['worst_date']}")
        fig = go.Figure(go.Histogram(x=hv["pnl"], nbinsx=60, marker_color=BLUE))
        fig.add_vline(x=-hv["var"], line=dict(color=RED, dash="dash"),
                      annotation_text=f"VaR {conf:.0%}")
        fig.update_layout(title="Replayed P&L on today's book (options via delta-cash)")
        st.plotly_chart(_styled(fig, 340), use_container_width=True)
        st.caption("Actual joint return vectors replayed on the current book — fat tails "
                   "included, no normality. Options are linearised at today's delta; dated "
                   "legs ride their underlying's front-month factor.")
    else:
        st.info("Historical VaR unavailable (panel or marks missing).")

    st.markdown("### Stress — dated historical episodes")
    epi = st.selectbox("Episode", list(STRESS_EPISODES))
    sres = stress_replay(pos, marks, *STRESS_EPISODES[epi])
    if sres.get("available"):
        kpi(st.columns(3)[0], "Episode P&L", f"${sres['total']:+,.0f}",
            f"{sres['start']} → {sres['end']}", RED if sres["total"] < 0 else GREEN)
        sdf = pd.DataFrame(sres["rows"])
        st.dataframe(sdf.style.format({"Move": "{:+.1f}%", "PnL": "{:+,.0f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("Each contract takes ITS OWN move from the dated window; options are "
                   "FULLY REVALUED at the shocked forward (same σ and T — a conservative "
                   "simplification, stated).")
    else:
        st.info("Episode replay unavailable — panel history missing for these legs.")

    st.markdown("### Parallel shock")
    shock = st.slider("Move all underlyings by", -30, 30, -10, 1, format="%d%%")
    tot, srows = 0.0, []
    for p in pos:
        base = position_base_price(p, marks)
        if base is None:
            continue
        pnl = position_pnl_at(p, base, shock / 100)
        tot += pnl
        srows.append(dict(Position=_position_label(p), PnL=pnl))
    if srows:
        kpi(st.columns(3)[0], f"P&L at {shock:+d}%", f"${tot:+,.0f}",
            "options revalued, not linearised", RED if tot < 0 else GREEN)
        st.dataframe(pd.DataFrame(srows).style.format({"PnL": "{:+,.0f}"}),
                     use_container_width=True, hide_index=True)


def page_mc(marks: MarkBoard) -> None:
    render_header(marks, "Monte Carlo", "Paths centred on the LIVE forward curve — exact OU discretisation")
    sel = st.selectbox("Contract", list(COMMODITIES))
    mark = require_mark(marks, sel)
    if mark is None:
        return
    c = COMMODITIES[sel]
    strip = fetch_forward_strip(sel)
    fwd = ((strip["T"].tolist(), strip["price"].tolist())
           if not strip.empty and len(strip) >= 2 else None)

    c1, c2, c3, c4 = st.columns(4)
    horizon = c1.slider("Horizon (months)", 3, 36, 18)
    n_paths = c2.select_slider("Paths", [1000, 5000, 10000, 20000], 5000)
    rv = realised_vol(sel)
    vol = c3.slider("Vol (ann.)", 0.05, 1.2, float(rv or c["vol"]), 0.01)
    default_mr = c.get("mr_halflife") is not None
    use_mr = c4.toggle("Mean reversion (Schwartz)", value=default_mr)
    hl = c4.slider("Half-life (yrs)", 0.1, 5.0, float(c.get("mr_halflife") or 1.5), 0.05,
                   disabled=not use_mr)

    res = simulate(mark, vol, n_paths, horizon, hl if use_mr else None, fwd, seed=42)
    badge = ("badge-green" if fwd else "badge-amber")
    st.markdown(f'<span class="badge {badge}">{res["model"]}</span>', unsafe_allow_html=True)

    fan = res["fan"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["p95"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["p5"], fill="tonexty",
                             fillcolor="rgba(240,165,0,0.10)", line=dict(width=0),
                             name="5–95%"))
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["p75"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["p25"], fill="tonexty",
                             fillcolor="rgba(240,165,0,0.22)", line=dict(width=0),
                             name="25–75%"))
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["p50"], name="Median",
                             line=dict(color=AMBER, width=2)))
    if fwd:
        fig.add_trace(go.Scatter(
            x=[date.today() + timedelta(days=int(t * 365.25)) for t in fwd[0]],
            y=fwd[1], mode="markers", marker=dict(color=BLUE, size=6, symbol="diamond"),
            name="Live strip"))
    fig.update_layout(title=f"{sel} — {n_paths:,} paths, {horizon}m ({c['unit']})")
    st.plotly_chart(_styled(fig, 440), use_container_width=True)

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Mean (terminal)", f"{res['mean']:,.2f}", "= forward by construction")
    kpi(k2, "Median (terminal)", f"{res['median']:,.2f}", "F·e^(−Var/2) — stated choice")
    kpi(k3, "P5", f"{res['p5']:,.2f}", "", RED)
    kpi(k4, "P95", f"{res['p95']:,.2f}", "", GREEN)

    fig2 = go.Figure(go.Bar(x=res["hist_x"], y=res["hist_y"], marker_color=BLUE))
    fig2.add_vline(x=mark, line=dict(color=AMBER, dash="dash"), annotation_text="spot")
    fig2.update_layout(title=f"Terminal distribution at {horizon}m")
    st.plotly_chart(_styled(fig2, 320), use_container_width=True)

    with st.expander("Why the fan is centred on the forward, and GBM vs Schwartz"):
        st.markdown(f"""
- **Centring** — paths satisfy **E[Sₜ] = F(t)** read off the live strip: the market has
  already priced carry and seasonality (NG winter premium, RB driving season), and a
  flat-at-spot fan throws that away. The **median** sits at F·e^(−Var/2), slightly below
  the mean — a deliberate risk-neutral-style choice, stated rather than implicit.
- **GBM** (no reversion): variance grows without bound — right shape for gold, which
  trades like a financial asset.
- **Schwartz 1-factor** (OU on log price, exact step φ=e^(−κΔ)): variance saturates at
  σ²/2κ. At nat-gas vol over 3y, an unreverted walk puts P95 near 3× spot and P5 near
  zero — physically meaningless for a storable commodity with inventory-driven prices.
- Half-life here: **{c.get('mr_halflife') or '— (none: GBM default)'}** yrs for {sel}.
        """)


def page_macro(marks: MarkBoard) -> None:
    render_header(marks, "Macro Rates", "FRED — CPI, policy rates, GDP and dollar context")
    key = fred_key()
    if not key:
        st.warning("Enter a FRED API key in the sidebar (or set st.secrets['FRED_KEY']). "
                   "Free at fred.stlouisfed.org. No key — no data.")
        return
    tab1, tab2 = st.tabs(["COUNTRY DASHBOARD", "COMMODITY CONTEXT"])
    with tab1:
        ctry = st.selectbox("Country / area", list(FRED_SERIES))
        cfg = FRED_SERIES[ctry]
        cols = st.columns(3)
        for col, (metric, meta) in zip(cols, MACRO_METRICS.items()):
            sid = cfg.get(metric)
            with col:
                st.markdown(f"#### {meta['label']}")
                if not sid:
                    st.info("No series mapped.")
                    continue
                df = fetch_fred(sid, key)
                if df.empty:
                    st.info("Feed unavailable.")
                    continue
                if metric == "cpi_yoy":
                    yoy = (df["value"] / df["value"].shift(12) - 1) * 100
                    plot, last_lab = yoy.dropna(), "YoY %"
                else:
                    plot, last_lab = df["value"], meta["label"]
                st.metric(last_lab, f"{plot.iloc[-1]:,.2f}",
                          f"{plot.iloc[-1]-plot.iloc[-2]:+,.2f}")
                fig = go.Figure(go.Scatter(x=plot.index, y=plot, line=dict(color=TEAL, width=1.3)))
                st.plotly_chart(_styled(fig, 240), use_container_width=True)
                st.caption(meta["note"])
    with tab2:
        for label, sid in FRED_COMMODITY_CONTEXT.items():
            df = fetch_fred(sid, key)
            st.markdown(f"#### {label}")
            if df.empty:
                st.info("Feed unavailable.")
                continue
            fig = go.Figure(go.Scatter(x=df.index, y=df["value"], line=dict(color=PURPLE, width=1.3)))
            st.plotly_chart(_styled(fig, 240), use_container_width=True)
        st.caption("Dollar up + real yields up is the classic headwind for gold; industrial "
                   "production proxies the demand side for energy and base metals.")


def page_signals(marks: MarkBoard) -> None:
    render_header(marks, "Signal Scanner", "One row per contract — carry, momentum, vol regime, seasonality")
    df = build_signals()
    if df.empty:
        st.error("Scanner unavailable — no feed.")
        return
    sectors = st.multiselect("Sectors", ALL_SECTORS, default=ALL_SECTORS)
    df = df[df["Sector"].isin(sectors)]
    fmt = {"Mark": "{:,.2f}", "Carry%": "{:+.2f}", "M1M2": "{:+,.3f}", "RV60": "{:.1f}",
           "VolRegime": "{:.2f}", "Chg1M": "{:+.1f}", "Chg3M": "{:+.1f}",
           "Px%ile1y": "{:.0f}", "SeasonMed": "{:+.2f}", "SeasonHit": "{:.0f}",
           "MMnet%OI": "{:+.1f}", "COT%ile": "{:.0f}"}
    st.dataframe(df.style.format(fmt, na_rep="—"), use_container_width=True,
                 hide_index=True, height=680)
    st.markdown("""
**Reading the columns** — `Carry%`: strip slope front→back (negative = backwardation,
longs get paid to roll). `M1M2`: front spread in price units. `RV60` vs `VolRegime`:
realised vol and its ratio to the 1y norm (>1.3 = stressed regime). `Chg1M/3M`:
momentum. `Px%ile1y`: where the mark sits in its own 1y range. `SeasonMed/Hit`: this
calendar month's median return and hit-rate over 10y (seasonal contracts only).
`MMnet%OI` / `COT%ile`: Managed Money net as % of open interest and its 3y percentile
(CFTC weekly — crowded longs at high percentiles are washout fuel; BZ blank on purpose,
ICE Brent is not CFTC-reported). Everything is computed off the shared live panel, the
single strip download and one batched CFTC request — nothing here is modelled, and the
whole scan costs four requests, not 250.
""")


def page_events(marks: MarkBoard) -> None:
    render_header(marks, "Event Calendar", "Computed weekly cadences · labelled approximations — nothing invented")
    ev = build_calendar_events()
    rows = []
    for e in ev:
        rows.append(dict(Date=(e["date"].isoformat() if e["date"] else "—"),
                         Event=e["event"], Tags=", ".join(e["tags"]), Basis=e["basis"]))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=560)
    st.caption("Weekly prints (EIA Wed/Thu, Baker Hughes Fri, Crop Progress Mon in season) "
               "follow a real cadence and are computed — holiday weeks can shift them a day. "
               "Monthly reports are anchored to their usual slot and marked approximate. "
               "Irregular events (FOMC, Grain Stocks…) are listed without a date rather "
               "than being invented: verify against the official calendars before carrying "
               "risk into a print.")


def page_about(marks: MarkBoard) -> None:
    render_header(marks, "About", "Design contract, exclusions, revision 2 changelog")
    st.markdown(f"""
### The contract this desk keeps

**Live mark-to-market only.** Every number that looks like a price IS a price — a Yahoo
Finance settle for a specific contract, carrying its settle date. When a feed dies the
screen says NO MARK and the analytics stand down. Nothing is interpolated into the gap.

**What is honestly NOT live** (each labelled on its page): regional balances (static
IEA/USDA-style estimates), the vol surface (a stated parametrisation — no options feed),
monthly event dates (approximate anchors), and the stress episodes (historical, dated).

### Revision 3 — from reading the market to pricing the trade

**📦 Storage & Cash-and-Carry** — the curve minus financing (live SOFR off FRED, or
manual) gives the MARKET-IMPLIED storage rate with zero assumptions; your editable
all-in storage cost turns it into an arb verdict (net margin per lot, annualised ROI,
% of full carry). Backwardation reads as a convenience yield; livestock is flagged
non-storable rather than pretending a carry arb exists on live animals.
**🧭 COT Positioning** — CFTC Disaggregated in ONE batched Socrata request: Managed
Money net vs price, net %OI with 3y crowdedness bands, commercials net, plus
`MMnet%OI` / `COT%ile` columns in the Signal Scanner. BZ is deliberately absent (ICE
Brent reports to ICE Europe, not the CFTC — no proxying).
**Navigation** — pages grouped the way a desk thinks: Markets · Fundamentals & Flows ·
Analytics · Volatility & Options · Book & Risk · Reference.

### Revision 2 — what changed and why

| Area | Before | Now |
|---|---|---|
| Strip tenor | ordinal `seq/12` | **calendar** year-fraction to delivery — roll yields and option anchors were wrong for GC/ZC cycles |
| Expiry | "20th of prior month" for everything | per-contract rules (grains mid-delivery-month, metals end of delivery, Brent M-2) |
| Options in VaR | full notional (parametric) / ignored (historical) | **Black-76 delta-cash in both**, underlying's vol |
| Options in stress | ignored | fully revalued at the shocked forward |
| Option ageing | tenor frozen forever | trade date stored; expired = intrinsic; theta is real |
| Missing correlations | zero-filled (silent independence) | conservative-sum fallback, with the reason on screen |
| Hist. VaR horizon | ×√h (Gaussian by the back door) | overlapping h-day windows |
| Monte Carlo | flat mean at ln(spot), Euler | **centred on the live forward curve**, exact OU step, E[Sₜ]=F(t) |
| Requests | ~250 for one signal scan | 3 grouped downloads for the whole app |
| Marks | undated; stale hidden | every settle carries its date; stale is shown, not hidden |
| Blotter | one shared file (Cloud privacy bug) | per-book `?book=` URL id + JSON export as the durable copy |
| Failures | silent `except` | logged to the sidebar diagnostics ring |
| Calendar | fabricated `today+N` dates | computed weekly cadences; the rest labelled approximate |
| Config | typos silently defaulted | strict registry validation at import |
| Tests | none | `pytest test_desk.py` covers the pure analytics, no network needed |

### Known limits, stated plainly

Yahoo's continuous front months are **not roll-adjusted**: structure and seasonality
histories carry a jump at each roll (captioned on those pages). The vol slider is
realised-vol seeded, not implied. Historical VaR linearises options at today's delta.
Regional balances are static. Delayed quotes, not exchange real-time.

### Author

Built by **Adam EL GBOURI** — quantitative finance.
Live app: [aeg-snd.streamlit.app](https://aeg-snd.streamlit.app) ·
GitHub: [github.com/adamelgbouri](https://github.com/adamelgbouri) ·
Related: [CFCAP](https://cfcap.streamlit.app) · [CODAP](https://codap.streamlit.app) ·
[Markowitz](https://aeg-markowitz.streamlit.app)
""")


# ══════════════════════════════════════════════════════════════════════════════
#  STORAGE & CASH-AND-CARRY — the trade behind the curve
# ══════════════════════════════════════════════════════════════════════════════
#  The contango/backwardation label says WHAT the curve looks like; this page says
#  WHETHER IT PAYS. Split cleanly in two, per the desk's honesty contract:
#    • MARKET-IMPLIED numbers (implied net storage, implied carry) come straight off
#      the live strip minus financing — no assumptions at all.
#    • The ARB VERDICT needs a physical storage cost, which is YOURS: defaults below
#      are indicative and editable, and the page says so.
#  Livestock is flagged non-storable: its curve is expectations, not carry — knowing
#  where the framework stops applying is part of the framework.
STORAGE_ASSUMPTIONS: Dict[str, dict] = {
    # mode: per_unit -> quote-units per month; pct_year -> % of mark per year (vaulting)
    "WTI Crude (CL)":         dict(mode="per_unit", value=0.40,
                                   note="Cushing tank lease, all-in ~$0.25–0.60/bbl/mo depending on cycle"),
    "Brent Crude (BZ)":       dict(mode="per_unit", value=0.45,
                                   note="onshore NWE tankage; floating storage costs more"),
    "Henry Hub Nat Gas (NG)": dict(mode="per_unit", value=0.06,
                                   note="salt-dome cycling ~$0.04–0.10/MMBtu/mo incl. fuel; seasonal capacity"),
    "RBOB Gasoline (RB)":     dict(mode="per_unit", value=0.011,
                                   note="~$0.45/bbl/mo ÷ 42 gal; RVP spec changes limit season-crossing storage"),
    "ULSD Heating Oil (HO)":  dict(mode="per_unit", value=0.011,
                                   note="~$0.45/bbl/mo ÷ 42 gal, clean tankage"),
    "Gold (GC)":              dict(mode="pct_year", value=0.0015,
                                   note="allocated vaulting + insurance ~10–20 bp/yr — financing dominates"),
    "Silver (SI)":            dict(mode="pct_year", value=0.0030,
                                   note="bulkier per $ than gold: ~25–40 bp/yr vaulting"),
    "Copper (HG)":            dict(mode="per_unit", value=0.004,
                                   note="exchange warehouse rent ~$0.003–0.006/lb/mo"),
    "Platinum (PL)":          dict(mode="pct_year", value=0.0020, note="vaulting ~15–25 bp/yr"),
    "Palladium (PA)":         dict(mode="pct_year", value=0.0020, note="vaulting ~15–25 bp/yr"),
    "Corn (ZC)":              dict(mode="per_unit", value=5.5,
                                   note="commercial elevator ~5–6 c/bu/mo; on-farm cheaper"),
    "Wheat CBOT SRW (ZW)":    dict(mode="per_unit", value=5.5,
                                   note="elevator tariff ~5–6 c/bu/mo (VSR can move it)"),
    "Soybeans (ZS)":          dict(mode="per_unit", value=6.0, note="elevator ~5–7 c/bu/mo"),
    "Soybean Meal (ZM)":      dict(mode="per_unit", value=2.0,
                                   note="~$2/short-ton/mo; meal cakes — real shelf-life limits"),
    "Soybean Oil (ZL)":       dict(mode="per_unit", value=0.15, note="~0.15 c/lb/mo bulk liquid"),
    "Sugar #11 (SB)":         dict(mode="per_unit", value=0.12, note="~0.12 c/lb/mo warehouse, raw bulk"),
    "Arabica Coffee (KC)":    dict(mode="per_unit", value=0.50,
                                   note="certified warehouse ~0.4–0.7 c/lb/mo incl. handling"),
    "Cocoa (CC)":             dict(mode="per_unit", value=8.0, note="~$6–10/mt/mo certified warehouse"),
    "Live Cattle (LE)":       dict(mode="none", value=0.0,
                                   note="live animals: feeding is not storage — curve is expectations"),
    "Lean Hogs (HE)":         dict(mode="none", value=0.0,
                                   note="live animals: no carry arb exists — curve is expectations"),
}


def default_storage_pm(commodity: str, mark: float) -> Tuple[Optional[float], str]:
    """Default all-in storage cost per QUOTE UNIT per month, or (None, note) if the
    commodity is not storable. Indicative and user-editable — stated on screen."""
    a = STORAGE_ASSUMPTIONS.get(commodity)
    if a is None or a["mode"] == "none":
        return None, (a["note"] if a else "no storage assumption on file")
    if a["mode"] == "pct_year":
        return mark * a["value"] / 12.0, a["note"]
    return float(a["value"]), a["note"]


def live_sofr() -> Optional[Tuple[float, date]]:
    """Latest SOFR print off FRED (decimal, with its date). None without a key —
    the page then takes a manual rate, clearly labelled manual."""
    key = fred_key()
    if not key:
        return None
    df = fetch_fred("SOFR", key, start=(date.today() - timedelta(days=45)).isoformat())
    if df.empty:
        return None
    return float(df["value"].iloc[-1]) / 100.0, df.index[-1].date()


def carry_economics(f1: float, fn: float, t1: float, tn: float,
                    r: float, storage_pm: float) -> dict:
    """
    Pure cash-and-carry arithmetic between two strip points (quote units per unit
    of commodity). Conventions, stated: simple interest on F1 over calendar dT;
    storage charged per month on the assumed all-in rate; returns on full price
    notional (futures margin would gear the cash ROI up — stated, not modelled).

      gross            = Fn − F1                      (what the curve pays)
      financing        = F1 · r · dT
      implied_stor_pm  = (gross − financing) / months (market-implied, no assumptions)
      implied_carry_ann= (gross − financing)/F1/dT    (negative = convenience yield)
      net              = gross − financing − storage  (YOUR economics)
      full_carry_pct   = gross / (financing + storage)
    """
    dT = tn - t1
    months = dT * 12.0
    gross = fn - f1
    fin = f1 * r * dT
    stor_total = storage_pm * months
    net = gross - fin - stor_total
    denom = fin + stor_total
    return dict(
        dT=dT, months=months, gross=gross, financing=fin,
        storage_total=stor_total, net=net,
        implied_storage_pm=(gross - fin) / months if months > 0 else float("nan"),
        implied_carry_ann_pct=(gross - fin) / f1 / dT * 100 if dT > 0 else float("nan"),
        full_carry_pct=gross / denom * 100 if denom > 1e-12 else float("nan"),
        ann_roi_pct=net / f1 / dT * 100 if dT > 0 else float("nan"),
    )


def page_storage(marks: MarkBoard) -> None:
    render_header(marks, "Storage & Cash-and-Carry",
                  "What the curve pays for storage — market-implied first, your assumptions second")
    sel = st.selectbox("Contract", list(COMMODITIES))
    mark = require_mark(marks, sel)
    if mark is None:
        return
    strip = fetch_forward_strip(sel)
    if strip.empty or len(strip) < 2:
        st.error("**NO LIVE STRIP** — carry economics need at least two dated contracts.")
        return

    c = COMMODITIES[sel]
    unit = c["unit"]
    stor_default, stor_note = default_storage_pm(sel, mark)
    storable = stor_default is not None

    f1, t1 = float(strip["price"].iloc[0]), float(strip["T"].iloc[0])

    # ── Inputs: financing + horizon + storage assumption ─────────────────────
    i1, i2, i3 = st.columns(3)
    sofr = live_sofr()
    r_pct = i1.number_input("Financing rate (ann. %, simple)", 0.0, 20.0,
                            float(round((sofr[0] * 100) if sofr else 4.00, 2)), 0.05)
    if sofr:
        i1.caption(f"Seeded with **live SOFR {sofr[0]*100:.2f}%** (FRED, {sofr[1]}). Editable.")
    else:
        i1.caption("Manual rate — set a FRED key in the sidebar for live SOFR.")
    r = r_pct / 100.0

    far_i = i2.selectbox("Carry M1 →", range(1, len(strip)), index=len(strip) - 2,
                         format_func=lambda i: f"{strip['label'].iloc[i]}  (T={strip['T'].iloc[i]:.2f}y)")
    fn, tn = float(strip["price"].iloc[far_i]), float(strip["T"].iloc[far_i])

    if storable:
        stor = i3.number_input(f"All-in storage ({unit} per month)",
                               value=float(round(stor_default, 4)), min_value=0.0,
                               step=max(stor_default * 0.05, 1e-4), format="%.4f")
        i3.caption(f"Default: {stor_note}. **Indicative — edit to your economics.** "
                   "Everything above the verdict line is pure market.")
    else:
        stor = 0.0
        i3.markdown('<span class="badge badge-red">NON-STORABLE</span>', unsafe_allow_html=True)
        i3.caption(stor_note)

    eco = carry_economics(f1, fn, t1, tn, r, stor)
    mult = price_multiplier(sel)

    # ── Market-implied block (no assumptions) ────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Front", f"{f1:,.2f}", f"{strip['label'].iloc[0]} · {unit}")
    kpi(k2, "Gross carry to " + str(strip['label'].iloc[far_i]),
        f"{eco['gross']:+,.2f}", f"{unit} over {eco['months']:.1f} months",
        RED if eco["gross"] > 0 else GREEN)
    isp = eco["implied_storage_pm"]
    kpi(k3, "Implied net storage", f"{isp:+,.4f}",
        f"{unit}/month the market pays after financing",
        AMBER if isp > 0 else GREEN)
    ica = eco["implied_carry_ann_pct"]
    kpi(k4, "Implied carry (ann.)" if ica >= 0 else "Convenience yield (ann.)",
        f"{abs(ica):.2f}%", "market pays storage" if ica >= 0 else "market pays for scarcity",
        AMBER if ica >= 0 else GREEN)

    # ── Verdict block (your assumptions) ─────────────────────────────────────
    if storable:
        st.markdown("### The arb, on your numbers")
        v1, v2, v3, v4 = st.columns(4)
        net_lot = eco["net"] * mult
        pays = eco["net"] > 0
        kpi(v1, "Net C&C margin", f"{eco['net']:+,.4f}",
            f"{unit} per unit · buy {strip['label'].iloc[0]}, store, deliver {strip['label'].iloc[far_i]}",
            GREEN if pays else RED)
        kpi(v2, "Per lot", f"${net_lot:+,.0f}",
            f"financing ${eco['financing']*mult:,.0f} · storage ${eco['storage_total']*mult:,.0f}",
            GREEN if pays else RED)
        kpi(v3, "Ann. return on notional", f"{eco['ann_roi_pct']:+.2f}%",
            "unlevered — futures margin gears cash ROI up", GREEN if pays else RED)
        fc = eco["full_carry_pct"]
        kpi(v4, "% of full carry", f"{fc:,.0f}%" if not math.isnan(fc) else "n/a",
            "100% = curve exactly covers financing + storage",
            RED if (not math.isnan(fc) and fc >= 95) else AMBER)
        badge = ('<span class="badge badge-green">CASH-AND-CARRY PAYS</span>' if pays
                 else '<span class="badge badge-red">CARRY DOES NOT COVER COSTS</span>')
        st.markdown(badge, unsafe_allow_html=True)
        if eco["gross"] < 0:
            st.caption("Curve is **backwardated**: there is no cash-and-carry here. The market "
                       "is paying a convenience premium to whoever holds inventory NOW — the "
                       "trade, if you own physical, is the reverse: sell spot, buy the deferred.")
    else:
        st.info("**No storage arb exists for live animals.** The deferred price is an "
                "expectation of future cash cattle/hogs (feed costs, weights, slaughter "
                "capacity), not spot plus carry. The implied-carry numbers above describe "
                "the curve's shape; they are not an arbitrage.")

    # ── Charts ───────────────────────────────────────────────────────────────
    fin_curve = [f1 * (1 + r * (float(t) - t1)) for t in strip["T"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strip["label"], y=strip["price"], mode="lines+markers",
                             name="Live strip", line=dict(color=AMBER, width=2),
                             marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=strip["label"], y=fin_curve, mode="lines",
                             name=f"Financing only ({r_pct:.2f}%)",
                             line=dict(color=BLUE, width=1.4, dash="dash")))
    fig.update_layout(title=f"{sel} — strip vs financing-only forward ({unit}). "
                            "Gap above dashed = market-paid storage; below = convenience.")
    st.plotly_chart(_styled(fig, 400), use_container_width=True)

    rows = []
    for i in range(1, len(strip)):
        e = carry_economics(f1, float(strip["price"].iloc[i]), t1,
                            float(strip["T"].iloc[i]), r, 0.0)
        rows.append(dict(Month=strip["label"].iloc[i], T=float(strip["T"].iloc[i]),
                         Price=float(strip["price"].iloc[i]), Gross=e["gross"],
                         Financing=e["financing"], ImplStorPM=e["implied_storage_pm"],
                         ImplAnnPct=e["implied_carry_ann_pct"]))
    idf = pd.DataFrame(rows)
    fig2 = go.Figure(go.Bar(x=idf["Month"], y=idf["ImplStorPM"],
                            marker_color=[AMBER if v >= 0 else GREEN for v in idf["ImplStorPM"]]))
    fig2.add_hline(y=0, line=dict(color=BORDER))
    if storable:
        fig2.add_hline(y=stor, line=dict(color=RED, dash="dot"),
                       annotation_text=f"your storage cost {stor:,.4f}")
    fig2.update_layout(title=f"Market-implied net storage, M1 → each month ({unit}/month). "
                             "Bars above the red dotted line = the arb window.")
    st.plotly_chart(_styled(fig2, 340), use_container_width=True)

    st.markdown("### Carry table (market-implied, no assumptions)")
    st.dataframe(idf.style.format({"T": "{:.2f}", "Price": "{:,.2f}", "Gross": "{:+,.3f}",
                                   "Financing": "{:+,.3f}", "ImplStorPM": "{:+,.4f}",
                                   "ImplAnnPct": "{:+.2f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("Conventions, stated: simple interest on M1 over calendar dT (T is real "
               "year-fraction to delivery); storage charged monthly; returns on full price "
               "notional. Not modelled — and said so: in/out pump fees, quality/location "
               "basis between the paper and your tank, margin financing, insurance beyond "
               "the vault rate. The implied columns are pure market; only the verdict "
               "uses your storage number.")


# ══════════════════════════════════════════════════════════════════════════════
#  COT POSITIONING — CFTC Disaggregated report, one batched request
# ══════════════════════════════════════════════════════════════════════════════
#  Third pillar next to fundamentals and price: WHO is positioned, and how crowded.
#  Managed Money = the speculative flow; Producer/Merchant = the physical hedgers.
#  Weekly, released Friday ~15:30 ET, data as of the prior Tuesday. Source is the
#  CFTC's public Socrata API — keyless, and fetched for the WHOLE board in a single
#  request (rev-2 batching discipline). ICE Brent is reported under ICE Europe's own
#  COT, not the CFTC — so BZ is honestly not wired rather than proxied off WTI.
COT_MARKET_CODES: Dict[str, Optional[str]] = {
    "WTI Crude (CL)": "067651",  "Brent Crude (BZ)": None,
    "Henry Hub Nat Gas (NG)": "023651", "RBOB Gasoline (RB)": "111659",
    "ULSD Heating Oil (HO)": "022651",
    "Gold (GC)": "088691", "Silver (SI)": "084691", "Copper (HG)": "085692",
    "Platinum (PL)": "076651", "Palladium (PA)": "075651",
    "Corn (ZC)": "002602", "Wheat CBOT SRW (ZW)": "001602", "Soybeans (ZS)": "005602",
    "Soybean Meal (ZM)": "026603", "Soybean Oil (ZL)": "007601",
    "Sugar #11 (SB)": "080732", "Arabica Coffee (KC)": "083731", "Cocoa (CC)": "073732",
    "Live Cattle (LE)": "057642", "Lean Hogs (HE)": "054642",
}
# Disaggregated futures-only first; futures+options combined as fallback.
COT_DATASETS = ["72hh-3qpy", "kh3c-gbw2"]
_COT_SELECT = ("report_date_as_yyyy_mm_dd,cftc_contract_market_code,"
               "open_interest_all,m_money_positions_long_all,m_money_positions_short_all,"
               "prod_merc_positions_long_all,prod_merc_positions_short_all")


def _first_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    """Socrata field names have historical quirks (double underscores…). Pick the
    first variant that exists instead of hard-failing on one spelling."""
    return next((n for n in names if n in df.columns), None)


@st.cache_data(ttl=6 * 3600, persist="disk")
def fetch_cot_all(years: int = 5) -> Dict[str, pd.DataFrame]:
    """Disaggregated COT for every wired contract — ONE request. Tries the dataset
    candidates in order, with and without a $select (a schema drift then degrades to
    a bigger payload, not a dead page). Empty dict on failure, logged, page says so."""
    if not REQUESTS_AVAILABLE:
        return {}
    codes = {c: n for n, c in COT_MARKET_CODES.items() if c}
    since = (date.today() - timedelta(days=int(years * 365.25))).isoformat()
    in_list = ",".join(f"'{c}'" for c in codes)
    where = (f"report_date_as_yyyy_mm_dd >= '{since}' "
             f"AND cftc_contract_market_code in({in_list})")
    rows = None
    for ds in COT_DATASETS:
        url = f"https://publicreporting.cftc.gov/resource/{ds}.json"
        for params in ({"$where": where, "$select": _COT_SELECT,
                        "$order": "report_date_as_yyyy_mm_dd", "$limit": 9000},
                       {"$where": where,
                        "$order": "report_date_as_yyyy_mm_dd", "$limit": 9000}):
            try:
                resp = requests.get(url, params=params, timeout=25)
                if resp.status_code != 200:
                    LOG.warning("COT %s -> HTTP %s", ds, resp.status_code)
                    continue
                payload = resp.json()
                if payload:
                    rows = payload
                    LOG.info("COT dataset %s: %d rows", ds, len(rows))
                    break
                LOG.warning("COT %s -> empty payload", ds)
            except requests.RequestException as e:
                LOG.warning("COT %s request failed: %s", ds, e)
            except Exception as e:
                LOG.warning("COT %s parse failed: %s", ds, e)
        if rows:
            break
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    c_date = _first_col(df, "report_date_as_yyyy_mm_dd")
    c_code = _first_col(df, "cftc_contract_market_code")
    c_oi   = _first_col(df, "open_interest_all")
    c_mml  = _first_col(df, "m_money_positions_long_all")
    c_mms  = _first_col(df, "m_money_positions_short_all")
    c_pml  = _first_col(df, "prod_merc_positions_long_all")
    c_pms  = _first_col(df, "prod_merc_positions_short_all")
    if not all([c_date, c_code, c_oi, c_mml, c_mms]):
        LOG.warning("COT schema unexpected — columns: %s", list(df.columns)[:12])
        return {}

    df["date"] = pd.to_datetime(df[c_date])
    for col in [c_oi, c_mml, c_mms, c_pml, c_pms]:
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out: Dict[str, pd.DataFrame] = {}
    for code, name in codes.items():
        sub = df[df[c_code] == code].sort_values("date")
        if sub.empty:
            LOG.warning("COT: no rows for %s (%s)", name, code)
            continue
        g = pd.DataFrame({
            "oi": sub[c_oi].values,
            "mm_long": sub[c_mml].values, "mm_short": sub[c_mms].values,
            "pm_long": (sub[c_pml].values if c_pml else np.nan),
            "pm_short": (sub[c_pms].values if c_pms else np.nan),
        }, index=pd.DatetimeIndex(sub["date"]))
        g = g[~g.index.duplicated(keep="last")]
        g["mm_net"] = g["mm_long"] - g["mm_short"]
        g["pm_net"] = g["pm_long"] - g["pm_short"]
        g["mm_net_pct_oi"] = np.where(g["oi"] > 0, g["mm_net"] / g["oi"] * 100, np.nan)
        out[name] = g
    return out


def page_cot(marks: MarkBoard) -> None:
    render_header(marks, "COT Positioning",
                  "CFTC Disaggregated — who holds the risk, and how crowded it is")
    covered = [n for n, c in COT_MARKET_CODES.items() if c]
    sel = st.selectbox("Contract", covered)
    if COT_MARKET_CODES.get("Brent Crude (BZ)") is None:
        st.caption("BZ is absent on purpose: ICE Brent reports under ICE Europe's COT, "
                   "not the CFTC — proxying it off WTI would be a fabricated position.")

    data = fetch_cot_all()
    cot = data.get(sel, pd.DataFrame())
    if cot.empty:
        st.error("**COT feed unavailable** (CFTC Socrata API returned nothing — see "
                 "sidebar diagnostics). Nothing is shown in its place.")
        return

    last, prev = cot.iloc[-1], cot.iloc[-2] if len(cot) > 1 else cot.iloc[-1]
    hist3y = cot.tail(156)
    pctile = float((hist3y["mm_net"] < last["mm_net"]).mean() * 100)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Managed Money net", f"{last['mm_net']:+,.0f}",
        f"contracts · report {cot.index[-1].date()}",
        GREEN if last["mm_net"] > 0 else RED)
    kpi(k2, "WoW change", f"{last['mm_net'] - prev['mm_net']:+,.0f}",
        "specs adding" if last["mm_net"] > prev["mm_net"] else "specs cutting",
        GREEN if last["mm_net"] > prev["mm_net"] else RED)
    kpi(k3, "MM net / OI", f"{last['mm_net_pct_oi']:+.1f}%",
        f"open interest {last['oi']:,.0f}")
    kpi(k4, "3y crowdedness", f"{pctile:.0f}th %ile",
        "of MM net positioning",
        RED if pctile >= 85 else GREEN if pctile <= 15 else AMBER)
    kpi(k5, "Commercials net", f"{last['pm_net']:+,.0f}" if not math.isnan(last["pm_net"]) else "n/a",
        "Producer/Merchant — the physical side",
        BLUE)
    st.markdown(pctile_badge(pctile), unsafe_allow_html=True)

    # MM net vs price
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cot.index, y=cot["mm_net"], name="MM net (contracts)",
                         marker_color=[GREEN if v >= 0 else RED for v in cot["mm_net"]],
                         opacity=0.55))
    panel = panel_years(5.2)
    if not panel.empty and sel in panel.columns:
        px_w = panel[sel].dropna().reindex(cot.index, method="ffill")
        fig.add_trace(go.Scatter(x=cot.index, y=px_w, name="Price (front)",
                                 yaxis="y2", line=dict(color=AMBER, width=1.6)))
    fig.update_layout(
        title=f"{sel} — Managed Money net vs price, 5y",
        yaxis=dict(title="MM net, contracts"),
        yaxis2=dict(title="Price", overlaying="y", side="right", showgrid=False))
    st.plotly_chart(_styled(fig, 430), use_container_width=True)

    # Crowdedness bands
    p20, p80 = hist3y["mm_net_pct_oi"].quantile([0.20, 0.80])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cot.index, y=cot["mm_net_pct_oi"], name="MM net %OI",
                              line=dict(color=TEAL, width=1.5)))
    fig2.add_hline(y=float(p80), line=dict(color=RED, dash="dash"),
                   annotation_text="3y p80 — crowded long")
    fig2.add_hline(y=float(p20), line=dict(color=GREEN, dash="dash"),
                   annotation_text="3y p20 — crowded short / washed out")
    fig2.update_layout(title="Speculative intensity — MM net as % of open interest")
    st.plotly_chart(_styled(fig2, 340), use_container_width=True)

    st.markdown("### Last 8 reports")
    tail = cot.tail(8).iloc[::-1].reset_index().rename(columns={"index": "Report"})
    tail["Report"] = tail["Report"].dt.date
    show = tail[["Report", "oi", "mm_long", "mm_short", "mm_net", "mm_net_pct_oi", "pm_net"]]
    show.columns = ["Report", "Open Int", "MM Long", "MM Short", "MM Net", "MM %OI", "PM Net"]
    st.dataframe(show.style.format({"Open Int": "{:,.0f}", "MM Long": "{:,.0f}",
                                    "MM Short": "{:,.0f}", "MM Net": "{:+,.0f}",
                                    "MM %OI": "{:+.1f}", "PM Net": "{:+,.0f}"}, na_rep="—"),
                 use_container_width=True, hide_index=True)
    st.caption("CFTC Disaggregated report (futures-only), weekly: released Friday "
               "~15:30 ET with positions as of Tuesday — the print is always 3 days "
               "stale, which matters in fast tape. Crowded longs at high percentiles are "
               "fuel for washouts; commercials leaning the other way is the classic tell. "
               "Reading, not gospel: COT is positioning, not a signal by itself.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
ROUTES = {
    "📊  Dashboard": page_dashboard, "📈  Forward Curves": page_curve,
    "🔀  Calendar Spreads": page_spreads, "🏭  Cracks, Crush & Arbs": page_structures,
    "📦  Storage & Carry": page_storage, "🧭  COT Positioning": page_cot,
    "🔗  Correlation": page_correlation, "🌡️  Seasonality": page_seasonality,
    "🛢️  EIA Fundamentals": page_eia, "🗺️  Regional Balances": page_regional,
    "🎯  Options Pricer": page_options, "🌊  Vol Surface": page_vol_surface,
    "📒  Trade Blotter": page_blotter, "⚠️  Portfolio Risk": page_risk,
    "🎲  Monte Carlo": page_mc, "🌍  Macro Rates": page_macro,
    "📡  Signal Scanner": page_signals, "📅  Event Calendar": page_events,
    "ℹ️  About": page_about,
}


def main() -> None:
    _setup_page()
    if not YF_AVAILABLE:
        st.error("yfinance is not installed — this desk has no data source without it. "
                 "`pip install yfinance`")
        return
    marks = fetch_live_marks()
    page = render_sidebar(marks)
    stale = marks.stale_names()
    if stale:
        st.warning("Dated (stale) settles in use for: " + ", ".join(
            COMMODITIES[n]["ticker"] for n in stale) +
            " — shown with their dates, never passed off as today's prints.")
    ROUTES[page](marks)


if __name__ == "__main__":
    main()
