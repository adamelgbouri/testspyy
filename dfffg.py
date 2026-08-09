"""
S&D DESK — Commodity Trading Desk
=================================
Live mark-to-market analytics across 20 futures contracts (energy, metals, grains,
softs, livestock): forward curves and how they move, calendar spreads, refinery and
processing margins, cash-and-carry economics, CFTC positioning, options, portfolio
VaR, and a physical cargo book with basis attribution.

DESIGN CONTRACT — every number that looks like a price IS a price. When a feed dies
the screen says NO MARK and the analytics stand down: nothing is interpolated,
proxied or modelled into the gap, and a stale settle is shown WITH ITS DATE rather
than passed off as today's. The few elements that are not live — regional balances,
the vol surface, monthly event dates, storage-cost defaults — say so on their page.

LAYOUT — this file reads top-down in dependency order. Nothing below is needed by
anything above it:

    1. CONTRACT REGISTRY & CALENDAR   static truth, validated at import
    2. LIVE DATA LAYER                every fetch and cache, grouped
    3. ANALYTICS                      pure maths, no I/O
    4. BOOK                           positions, cargoes, persistence
    5. PORTFOLIO RISK                 VaR, historical simulation, stress
    6. UI                             chrome shared by every page
    7. PAGES                          one section per screen (20)
    8. MAIN                           router, with a per-page guard

Run:    streamlit run desk.py
Tests:  pytest test_desk.py -q          (65 tests, no network, no Streamlit runtime)
Keys:   EIA_KEY / FRED_KEY — sidebar, or .streamlit/secrets.toml. Optional.

by Adam EL GBOURI
"""
from __future__ import annotations

import json
import logging
import math
import os
import traceback
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


# ════════════════════════════════════════════════════════════════════════════
#  CONTRACT REGISTRY & CALENDAR
#  Static truth: palette, the twenty contracts, expiry rules, calendar tenor and
#  unit tables. Validated at import — a typo in the registry fails loudly here
#  rather than silently mispricing a lot somewhere downstream.
# ════════════════════════════════════════════════════════════════════════════


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
/* ── sidebar navigation: native radios, typography only (layout stays Streamlit's) ── */
section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {{
    font-family:'JetBrains Mono',monospace; font-size:9.5px; color:{GRAY};
    letter-spacing:0.2em; text-transform:uppercase;
    border-bottom:1px solid {BORDER}; padding-bottom:4px; width:100%; }}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] p {{
    font-family:'JetBrains Mono',monospace; font-size:0.78rem; }}
/* ── per-page intro box rendered by render_header ── */
.page-help {{ background:{PANEL}; border:1px solid {BORDER}; border-left:3px solid {BLUE};
    border-radius:8px; padding:10px 14px; margin:6px 0 4px 0;
    font-size:0.78rem; line-height:1.6; color:#B9C4CF; }}
.page-help b {{ color:{TEXT}; }}
</style>
"""


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


# ── Historical stress episodes ───────────────────────────────────────────────
STRESS_EPISODES = {
    "COVID crash (Feb–Mar 2020)":        ("2020-02-19", "2020-03-23"),
    "WTI negative print (Apr 2020)":     ("2020-04-01", "2020-04-30"),
    "Ukraine invasion (Feb–Mar 2022)":   ("2022-02-21", "2022-03-09"),
    "2022 energy peak → bust (Jun–Sep)": ("2022-06-08", "2022-09-26"),
    "Banking wobble (Mar 2023)":         ("2023-03-08", "2023-03-24"),
}


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL CARGO — the merchant layer
# ══════════════════════════════════════════════════════════════════════════════
#  A physical cargo is automatically long flat price. This module books the cargo,
#  DERIVES its futures hedge, and splits the result into the components a merchant
#  is actually paid for: basis, freight, carry and the residual flat price.
#
#  Three design rules, and they are the whole point:
#
#  1. ONE MODEL, NO SPECIAL CASES. A cargo is a buy leg, an optional sell leg and a
#     hedge. An outright ("fixed price") purchase is not a separate mode: the desk
#     computes the differential it implies against the pricing month and runs the
#     same arithmetic. Flexibility comes from empty fields, not from branches.
#
#  2. THE COMPONENTS MUST RECONCILE. Attribution is computed twice — once by
#     component, once directly from the legs — and the difference is displayed as
#     UNEXPLAINED RESIDUAL. It is zero by algebra, so any non-zero value is a real
#     implementation defect made visible rather than rounded away.
#
#  3. ASSESSMENTS ARE NOT MARKET DATA. Differentials and freight are user inputs
#     (Platts/Argus/Baltic are licensed and this desk has no feed). Every number
#     derived from them is tagged, and an unrealised basis P&L is labelled a MARK
#     against the user's own assessment — never presented as a fact.

# Native units per one trade unit. Density and test weight are grade-dependent, so
# these are INDICATIVE defaults: the booking form shows the factor and lets the user
# overwrite it, and the factor used is stored with the cargo.
CARGO_UNIT_FACTORS: Dict[str, Dict[str, float]] = {
    "WTI Crude (CL)":         {"bbl": 1.0, "mt": 7.33, "m3": 6.2898},
    "Brent Crude (BZ)":       {"bbl": 1.0, "mt": 7.45, "m3": 6.2898},
    "RBOB Gasoline (RB)":     {"gal": 1.0, "bbl": 42.0, "mt": 333.4},
    "ULSD Heating Oil (HO)":  {"gal": 1.0, "bbl": 42.0, "mt": 309.0},
    "Henry Hub Nat Gas (NG)": {"MMBtu": 1.0, "therm": 0.1, "MWh": 3.412},
    "Gold (GC)":              {"troy oz": 1.0, "kg": 32.1507},
    "Silver (SI)":            {"troy oz": 1.0, "kg": 32.1507, "mt": 32150.7},
    "Copper (HG)":            {"lb": 1.0, "mt": 2204.62, "short ton": 2000.0},
    "Platinum (PL)":          {"troy oz": 1.0, "kg": 32.1507},
    "Palladium (PA)":         {"troy oz": 1.0, "kg": 32.1507},
    "Corn (ZC)":              {"bu": 1.0, "mt": 39.3680, "short ton": 35.714},
    "Wheat CBOT SRW (ZW)":    {"bu": 1.0, "mt": 36.7437, "short ton": 33.333},
    "Soybeans (ZS)":          {"bu": 1.0, "mt": 36.7437, "short ton": 33.333},
    "Soybean Meal (ZM)":      {"short ton": 1.0, "mt": 1.10231},
    "Soybean Oil (ZL)":       {"lb": 1.0, "mt": 2204.62, "short ton": 2000.0},
    "Sugar #11 (SB)":         {"lb": 1.0, "mt": 2204.62},
    "Arabica Coffee (KC)":    {"lb": 1.0, "mt": 2204.62, "bag (60kg)": 132.277},
    "Cocoa (CC)":             {"mt": 1.0, "short ton": 0.907185},
    "Live Cattle (LE)":       {"lb": 1.0, "head (1,350lb)": 1350.0},
    "Lean Hogs (HE)":         {"lb": 1.0, "head (285lb)": 285.0},
}


# The native unit must map to itself, or the booking form would offer a conversion
# that silently rescales the cargo. Checked at import, like the contract registry.
for _n, _u in CARGO_UNIT_FACTORS.items():
    assert _n in COMMODITIES, f"cargo units: unknown contract {_n}"
    assert _u.get(COMMODITIES[_n]["size_unit"]) == 1.0, (
        f"cargo units: {_n} is missing an identity factor for its native unit "
        f"{COMMODITIES[_n]['size_unit']!r}")


assert set(CARGO_UNIT_FACTORS) == set(COMMODITIES), "cargo units: registry mismatch"


CARGO_STAGES = ["Booked", "In transit", "Priced", "Sold", "Settled"]


INCOTERMS = ["FOB", "CFR", "CIF", "DAP", "EXW"]


def cargo_trade_units(commodity: str) -> Dict[str, float]:
    """Trade units offered for a commodity, native unit first."""
    return CARGO_UNIT_FACTORS.get(
        commodity, {COMMODITIES[commodity]["size_unit"]: 1.0})


def cargo_money(commodity: str, price_qu: float, vol_native: float) -> float:
    """Cash value of `price_qu` (in QUOTE units) applied to `vol_native` native
    units. The cents divisor is applied here and nowhere else: corn at 18 c/bu over
    100,000 bu is $18,000, not $1.8m."""
    div = COMMODITIES[commodity].get("price_divisor", 1.0)
    return price_qu * vol_native / div


def cargo_volume_native(cg: dict) -> float:
    return float(cg["volume"]) * float(cg.get("unit_factor", 1.0))


def cargo_hedge_lots(commodity: str, vol_native: float, ratio: float = 1.0) -> int:
    """Whole lots closest to the requested hedge ratio. The leftover is NEVER
    rounded away silently — cargo_attribution reports it as residual flat price."""
    cs = COMMODITIES[commodity]["contract_size"]
    return int(round(vol_native * ratio / cs))


def cargo_carries_freight(side: str, incoterm: str) -> bool:
    """Who pays the freight, by Incoterm. A buyer pays it on FOB/EXW terms; on
    CFR/CIF/DAP it is inside the seller's price. Seeded, and overridable — real
    contracts are messier than the three-letter code suggests."""
    if side == "Buy":
        return incoterm in ("FOB", "EXW")
    return incoterm in ("CFR", "CIF", "DAP")


def default_basis_vol(commodity: str, price: float) -> float:
    """Indicative annual volatility of the DIFFERENTIAL, in quote units. Differential
    histories are licensed data this desk does not have, so this is a stated default
    (a small fraction of flat-price vol) and the page invites the user to replace it
    with their own number."""
    c = COMMODITIES[commodity]
    return abs(price) * c["vol"] * 0.05


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


# ════════════════════════════════════════════════════════════════════════════
#  LIVE DATA LAYER
#  Every fetch and cache. Grouped calls only: the whole board costs a handful of
#  requests, never one per cell. No fallbacks and no fabrication — a dead feed
#  shows as NO MARK, a stale settle shows with its date.
# ════════════════════════════════════════════════════════════════════════════


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
#  CURVE EVOLUTION — how the curve MOVED, not just where it is
# ══════════════════════════════════════════════════════════════════════════════
#  A single strip is a photograph. The desk trades the MOVEMENT: three markets can
#  share the same front price and tell three opposite stories — a parallel shift
#  (macro), a front steepening (prompt physical tightness), or the back repricing
#  (a structural change in production economics).
#
#  Every historical point is the settle of the SAME dated contract on that past
#  date — not a continuous stitched series. No roll artefacts, no proxying: the
#  Dec-26 contract's price last Tuesday is exactly that, or the point is absent.
@st.cache_data(ttl=1800, persist="disk")
def fetch_strip_history(commodity: str, period: str = "6mo") -> pd.DataFrame:
    """Close history for every dated contract of one commodity — ONE grouped call.
    Columns = tickers, index = dates. Empty frame if the feed returns nothing."""
    specs = strip_contract_specs(commodity)
    tickers = [s["ticker"] for s in specs]
    closes = _yf_closes(tickers, period=period)
    if closes.empty:
        LOG.warning("strip history empty: %s", commodity)
        return pd.DataFrame()
    keep = [t for t in tickers if t in closes.columns]
    return closes[keep].dropna(how="all")


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
        }, index=pd.DatetimeIndex(sub["date"].values))   # .values: index stays UNNAMED
        g = g[~g.index.duplicated(keep="last")]
        g["mm_net"] = g["mm_long"] - g["mm_short"]
        g["pm_net"] = g["pm_long"] - g["pm_short"]
        g["mm_net_pct_oi"] = np.where(g["oi"] > 0, g["mm_net"] / g["oi"] * 100, np.nan)
        out[name] = g
    return out


# ════════════════════════════════════════════════════════════════════════════
#  ANALYTICS
#  Pure maths over the data layer: Black-76, carry, simulation, curve moves.
#  No widgets and no I/O, which is what makes all of it testable offline.
# ════════════════════════════════════════════════════════════════════════════


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


def curve_on_date(strip: pd.DataFrame, hist: pd.DataFrame,
                  asof: date) -> Optional[pd.DataFrame]:
    """Rebuild the same strip as it settled on (or just before) `asof`.
    Returns None if that date predates the available history. Each row keeps its
    own settle date, so a thin deferred month that last traded earlier is visible
    rather than silently carried."""
    if strip.empty or hist.empty:
        return None
    idx = hist.index[hist.index.date <= asof]
    if len(idx) == 0:
        return None
    cutoff = idx[-1]
    rows = []
    for r in strip.itertuples():
        if r.ticker not in hist.columns:
            continue
        s = hist[r.ticker].loc[:cutoff].dropna()
        if s.empty:
            continue
        rows.append(dict(label=r.label, delivery=r.delivery, T=r.T,
                         ticker=r.ticker, price=float(s.iloc[-1]),
                         asof=s.index[-1].date().isoformat()))
    if not rows:
        return None
    return pd.DataFrame(rows)


def curve_move(now: pd.DataFrame, then: pd.DataFrame) -> pd.DataFrame:
    """Per-delivery-month change between two curves, matched on delivery month
    (never on position: the front rolls, so position i is not the same contract)."""
    if now is None or then is None or now.empty or then.empty:
        return pd.DataFrame()
    m = now.merge(then[["delivery", "price"]], on="delivery",
                  suffixes=("", "_then"), how="inner")
    if m.empty:
        return pd.DataFrame()
    m["change"] = m["price"] - m["price_then"]
    m["change_pct"] = np.where(m["price_then"] != 0,
                               m["change"] / m["price_then"] * 100, np.nan)
    return m


def decompose_move(move: pd.DataFrame) -> dict:
    """Split a curve move into level and slope — the two things a trader reads.

      shift = average change across the curve      (parallel: macro, currency, rates)
      twist = front change − back change           (>0: front outperformed = prompt
                                                    tightness; <0: the back repriced)

    A front-led move is usually temporary and physical; a back-led move is a
    structural reassessment and matters more. The label below says which happened,
    and it deliberately calls a move 'parallel' only when the twist is small
    relative to the shift."""
    if move.empty:
        return dict(available=False)
    ch = move["change"].astype(float)
    shift = float(ch.mean())
    front, back = float(ch.iloc[0]), float(ch.iloc[-1])
    twist = front - back
    if abs(twist) < max(abs(shift) * 0.35, 1e-9):
        shape = "PARALLEL SHIFT"
        read = "whole curve moved together — macro, currency or rates, not a physical story"
    elif twist > 0:
        shape = "FRONT-LED (steepening)"
        read = ("prompt outperformed the deferred — physical tightness now; the back "
                "does not believe it lasts")
    else:
        shape = "BACK-LED (flattening)"
        read = ("the deferred repriced more than the prompt — a structural reassessment "
                "of long-run supply or cost")
    return dict(available=True, shift=shift, twist=twist, front=front, back=back,
                shape=shape, read=read,
                shift_pct=float(move["change_pct"].mean()))


# ── Curve-shape stress: the twist that actually kills physical books ──────────
def shape_shock_factors(Ts: Sequence[float], front_pct: float, back_pct: float,
                        pivot_years: float = 1.0) -> np.ndarray:
    """Per-contract shock, interpolated on CALENDAR tenor between a front shock and
    a back shock. Linear in T up to `pivot_years`, flat beyond — deferred months
    move together once you are far enough out the curve.

    A parallel shock is the special case front == back. Books that are flat outright
    but long the front spread survive parallel shocks and die on twists, which is
    exactly why this exists next to the parallel slider."""
    Ts = np.asarray(Ts, dtype=float)
    w = np.clip(Ts / max(pivot_years, 1e-6), 0.0, 1.0)
    return (front_pct + (back_pct - front_pct) * w) / 100.0


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


# ════════════════════════════════════════════════════════════════════════════
#  BOOK — positions, cargoes, valuation, persistence
#  The single source of truth for what the desk owns, and the only place that
#  writes it to disk. Cargo hedges are DERIVED here, never stored separately.
# ════════════════════════════════════════════════════════════════════════════


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
    if p.get("cargo_id"):
        tag = "physical" if p.get("physical") else "hedge"
        return f"{n} {p.get('strip_label') or 'fut'} · {tag} {p['cargo_id']}"
    if p.get("kind", "future") == "option":
        rem = option_time_remaining(p) * 12
        return f"{n} {p['opt_type'][:1].upper()}{p['strike']:g} ({rem:.1f}m left)"
    if p.get("strip_ticker"):
        return f"{n} {p.get('strip_label', p['strip_ticker'])}"
    return f"{n} fut"


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


# ── Roll calendar: an operational task, not an analytic ──────────────────────
def roll_calendar(positions: List[dict], marks,
                  today: Optional[date] = None) -> List[dict]:
    """When does each position have to roll, and what does the roll cost today?

    Front-month futures must be rolled before the front contract's last trading day
    or they go to delivery — an operational deadline, not a market view. The cost is
    today's M1−M2 spread applied to the position's size and direction.

    Dated futures are shown with their own expiry and no roll cost: not rolling is
    the reason you booked them. Options show days to expiry instead.
    """
    today = today or date.today()
    strips = fetch_all_strips()
    out = []
    for p in positions:
        n = p["commodity"]
        c = COMMODITIES[n]
        mult = price_multiplier(n)
        sign = 1 if p["side"] == "Long" else -1
        kind = p.get("kind", "future")

        if kind == "option":
            days = int(round(option_time_remaining(p, today) * 365.25))
            out.append(dict(Position=_position_label(p), Kind="Option", Lots=p["lots"],
                            Contract=n, Expiry=None, Days=days, RollCost=None,
                            Action="expires — exercise or close before then"))
            continue

        if p.get("strip_ticker"):
            exp = None
            sdf = strips.get(n, pd.DataFrame())
            if not sdf.empty:
                hit = sdf[sdf["ticker"] == p["strip_ticker"]]
                if not hit.empty:
                    try:
                        y, m = (int(x) for x in str(hit["delivery"].iloc[0]).split("-"))
                        exp = estimate_expiry(c["expiry_rule"], y, m)
                    except (ValueError, TypeError) as e:
                        LOG.warning("roll calendar: bad delivery for %s: %s",
                                    p["strip_ticker"], e)
            out.append(dict(Position=_position_label(p), Kind="Dated future",
                            Lots=p["lots"], Contract=n, Expiry=exp,
                            Days=((exp - today).days if exp else None), RollCost=None,
                            Action=("no roll — dated line held to its own expiry" if exp
                                    else "contract no longer listed — verify manually")))
            continue

        strip = strips.get(n, pd.DataFrame())
        if strip.empty:
            continue
        dy, dm_ = (int(x) for x in str(strip["delivery"].iloc[0]).split("-"))
        exp = estimate_expiry(c["expiry_rule"], dy, dm_)
        days = (exp - today).days
        cost = None
        if len(strip) > 1:
            m1, m2 = float(strip["price"].iloc[0]), float(strip["price"].iloc[1])
            cost = sign * (m1 - m2) * mult * p["lots"]
        action = ("ROLL NOW — front expires within a week" if days <= 7 else
                  "roll this week" if days <= 14 else "monitor")
        out.append(dict(Position=_position_label(p), Kind="Front future",
                        Lots=p["lots"], Contract=n, Expiry=exp, Days=days,
                        RollCost=cost, Action=action))
    return sorted(out, key=lambda r: (r["Days"] is None, r["Days"]))


def cargo_pricing_price(cg: dict, marks) -> Optional[float]:
    """Live price of the cargo's PRICING contract. None => the cargo cannot be
    marked and its attribution stands down (no proxying)."""
    if cg.get("pricing_ticker"):
        dm = dated_mark(cg["pricing_ticker"])
        return dm["price"] if dm else None
    return marks.get(cg["commodity"])


def cargo_hedge_price(cg: dict, marks) -> Optional[float]:
    if not cg.get("hedge_lots"):
        return None
    if cg.get("hedge_ticker"):
        dm = dated_mark(cg["hedge_ticker"])
        return dm["price"] if dm else None
    return marks.get(cg["commodity"])


def cargo_attribution(cg: dict, marks) -> dict:
    """
    Exact P&L decomposition of one cargo plus its hedge.

    Notation, per trade unit and in quote units:
        B0, B1  pricing-benchmark price at purchase / now (or at sale, if realised)
        d0, d1  purchase differential / sale differential (realised) or user mark
        H0, H1  hedge contract price at entry / now
        V       cargo volume in native units,  Vh = hedged volume (lots × size)

    Physical P&L = [(B1+d1) − (B0+d0)] · V · sgn
    Hedge P&L    = −(H1−H0) · Vh · sgn                       (short hedge for a buy)

    Which splits EXACTLY into three readable pieces:
        flat_residual = (B1−B0)·(V−Vh)·sgn        unhedged volume only
        carry_timing  = −[(H1−H0) − (B1−B0)]·Vh·sgn   hedge month ≠ pricing month
        basis         = (d1−d0)·V·sgn             the trade the merchant is paid for
    (expand and the B- and H-terms cancel: the sum is Physical + Hedge, identically).

    Costs — freight, storage, financing, other — are then subtracted. `residual` is
    NET computed directly minus the sum of the components: zero by algebra, shown
    anyway so that a missing leg is visible instead of silent.
    """
    name = cg["commodity"]
    sgn = 1.0 if cg["side"] == "Buy" else -1.0
    V = cargo_volume_native(cg)
    cs = COMMODITIES[name]["contract_size"]
    Vh = float(cg.get("hedge_lots", 0)) * cs

    realised = (cg.get("stage") in ("Sold", "Settled")
                and cg.get("diff_sell") is not None
                and cg.get("bench_sell") is not None)

    B0 = float(cg["bench_buy"])
    d0 = float(cg["diff_buy"])

    if realised:
        B1 = float(cg["bench_sell"])
        d1 = float(cg["diff_sell"])
    else:
        live = cargo_pricing_price(cg, marks)
        if live is None:
            return dict(available=False,
                        reason="pricing contract has no mark — cargo cannot be valued")
        B1 = float(live)
        d1 = float(cg.get("diff_mark", d0))

    H0 = float(cg.get("hedge_entry", B0))
    if Vh == 0:
        H1 = H0                       # no hedge: the hedge legs vanish, not error
    elif cg.get("hedge_exit") is not None:
        H1 = float(cg["hedge_exit"])
    elif realised and (cg.get("hedge_ticker") == cg.get("pricing_ticker")):
        # Same contract: the hedge must be marked at the same price as the pricing
        # leg, or a phantom carry appears out of nothing.
        H1 = B1
    else:
        hp = cargo_hedge_price(cg, marks)
        if hp is None:
            return dict(available=False,
                        reason="hedge contract has no mark — cargo cannot be valued")
        H1 = float(hp)

    # ── Components ───────────────────────────────────────────────────────────
    flat_residual = cargo_money(name, B1 - B0, V - Vh) * sgn
    carry_timing = -cargo_money(name, (H1 - H0) - (B1 - B0), Vh) * sgn
    basis = cargo_money(name, d1 - d0, V) * sgn

    freight_used = cg.get("freight_actual")
    freight_is_actual = freight_used is not None
    if not freight_is_actual:
        freight_used = cg.get("freight_budget", 0.0) or 0.0
    freight_cost = (cargo_money(name, float(freight_used), V)
                    if cg.get("carries_freight") else 0.0)

    months = float(cg.get("storage_days", 0) or 0) / 30.4375
    storage_cost = cargo_money(name, float(cg.get("storage_rate", 0.0) or 0.0) * months, V)
    fin_days = float(cg.get("finance_days", 0) or 0)
    fin_rate = float(cg.get("finance_rate", 0.0) or 0.0)
    finance_cost = cargo_money(name, (B0 + d0) * fin_rate * fin_days / 365.0, V)
    other_cost = float(cg.get("other_cost", 0.0) or 0.0)

    # ── Direct computation, for reconciliation ───────────────────────────────
    physical = cargo_money(name, (B1 + d1) - (B0 + d0), V) * sgn
    hedge = -cargo_money(name, H1 - H0, Vh) * sgn
    costs = freight_cost + storage_cost + finance_cost + other_cost
    net_direct = physical + hedge - costs

    comps = [
        dict(label="Flat price — residual (unhedged volume)", value=flat_residual,
             source="live" if not realised else "realised", kind="market"),
        dict(label="Carry / timing (hedge vs pricing month)", value=carry_timing,
             source="live" if not realised else "realised", kind="market"),
        dict(label="Basis (differential)", value=basis,
             source="realised" if realised else "USER MARK", kind="basis"),
        dict(label=f"Freight ({'actual' if freight_is_actual else 'budget'})",
             value=-freight_cost, source="USER INPUT", kind="cost"),
        dict(label="Storage", value=-storage_cost, source="USER INPUT", kind="cost"),
        dict(label="Financing", value=-finance_cost, source="SOFR + user terms", kind="cost"),
        dict(label="Demurrage / other", value=-other_cost, source="USER INPUT", kind="cost"),
    ]
    comp_sum = sum(c["value"] for c in comps)
    residual = net_direct - comp_sum

    gross_abs = sum(abs(c["value"]) for c in comps) or 1.0
    for c in comps:
        c["share"] = abs(c["value"]) / gross_abs * 100

    landed = (B0 + d0
              + (float(freight_used) if cg.get("carries_freight") else 0.0)
              + float(cg.get("storage_rate", 0.0) or 0.0) * months
              + (B0 + d0) * fin_rate * fin_days / 365.0)

    return dict(available=True, realised=realised, components=comps,
                net=net_direct, residual=residual, physical=physical, hedge=hedge,
                costs=costs, flat_net=physical + hedge,
                B0=B0, B1=B1, d0=d0, d1=d1, H0=H0, H1=H1,
                V=V, Vh=Vh, residual_volume=V - Vh, landed_cost=landed,
                basis_value=basis, hedge_ratio=(Vh / V * 100 if V else 0.0))


def cargo_hedge_positions(cargos: List[dict]) -> List[dict]:
    """Futures legs DERIVED from the cargo book — never stored separately.

    Storing the hedge as its own blotter row would let it drift out of sync with the
    cargo it hedges (edit the volume, forget the leg). Deriving it makes that class
    of bug impossible: the hedge is a view of the cargo, tagged and read-only in the
    Blotter, and it flows into the risk engine like any other position."""
    out = []
    for cg in cargos:
        lots = int(cg.get("hedge_lots", 0) or 0)
        if lots <= 0:
            continue
        out.append(dict(
            commodity=cg["commodity"], kind="future",
            side="Short" if cg["side"] == "Buy" else "Long",
            lots=lots, entry=float(cg.get("hedge_entry", 0.0)),
            strip_ticker=cg.get("hedge_ticker"),
            strip_label=cg.get("hedge_label", ""),
            trade_date=cg.get("booked_date"),
            cargo_id=cg["id"], derived=True))
    return out


def cargo_risk_legs(cargos: List[dict], marks) -> Tuple[List[dict], List[dict]]:
    """Risk representation of the PHYSICAL side.

    Returns (flat_legs, basis_legs):
      • flat_legs  — synthetic futures positions carrying the cargo's flat-price
        exposure at its pricing month. Netted against the derived hedge legs by the
        VaR engine, they leave exactly the unhedged residual — which is the honest
        answer, not zero.
      • basis_legs — the differential's own risk, in cash per 1.0 move, with an
        annual vol in quote units. Treated as INDEPENDENT of the futures book because
        this desk has no differential history to estimate a correlation from; the
        assumption is stated on the page rather than buried.
    """
    flat, basis = [], []
    for cg in cargos:
        name = cg["commodity"]
        V = cargo_volume_native(cg)
        if V <= 0:
            continue
        cs = COMMODITIES[name]["contract_size"]
        px = cargo_pricing_price(cg, marks)
        if px is None:
            continue
        flat.append(dict(commodity=name, kind="future",
                         side="Long" if cg["side"] == "Buy" else "Short",
                         lots=V / cs, entry=float(cg["bench_buy"]),
                         strip_ticker=cg.get("pricing_ticker"),
                         strip_label=cg.get("pricing_label", ""),
                         cargo_id=cg["id"], physical=True, derived=True))
        bvol = cg.get("basis_vol")
        if bvol is None:
            bvol = default_basis_vol(name, px)
        basis.append(dict(label=f"{cg.get('grade') or name} basis ({cg['id']})",
                          exposure=cargo_money(name, 1.0, V), vol=float(bvol)))
    return flat, basis


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


def book_serialise(positions: List[dict], cargos: List[dict]) -> str:
    """Whole-book payload. The legacy format was a bare list of positions; this one
    is a dict so cargoes travel with the book, and the loader still accepts a list."""
    return json.dumps(dict(version=2, positions=positions, cargos=cargos),
                      indent=2, default=str)


def cargo_deserialise(raw) -> List[dict]:
    """Validate imported cargoes. Unknown contracts are dropped (the registry may
    have changed); every numeric field is coerced with a safe default so a truncated
    file degrades into a valid cargo rather than crashing a page."""
    out = []
    for c in (raw or []):
        if c.get("commodity") not in COMMODITIES:
            continue
        def f(k, d=0.0):
            try:
                v = c.get(k)
                return d if v is None else float(v)
            except (TypeError, ValueError):
                return d
        cg = dict(
            id=str(c.get("id") or f"c{uuid.uuid4().hex[:6]}"),
            commodity=c["commodity"],
            side="Buy" if c.get("side", "Buy") == "Buy" else "Sell",
            grade=str(c.get("grade") or ""),
            volume=f("volume"), trade_unit=str(c.get("trade_unit") or
                                               COMMODITIES[c["commodity"]]["size_unit"]),
            unit_factor=f("unit_factor", 1.0) or 1.0,
            pricing_ticker=c.get("pricing_ticker"),
            pricing_label=str(c.get("pricing_label") or ""),
            pricing_basis=str(c.get("pricing_basis") or "Single settle"),
            pricing_window=c.get("pricing_window"),
            bench_buy=f("bench_buy"), diff_buy=f("diff_buy"),
            diff_mark=(f("diff_mark") if c.get("diff_mark") is not None else f("diff_buy")),
            diff_mark_date=str(c.get("diff_mark_date") or date.today().isoformat()),
            diff_sell=(float(c["diff_sell"]) if c.get("diff_sell") is not None else None),
            bench_sell=(float(c["bench_sell"]) if c.get("bench_sell") is not None else None),
            hedge_lots=int(f("hedge_lots")), hedge_ticker=c.get("hedge_ticker"),
            hedge_label=str(c.get("hedge_label") or ""),
            hedge_entry=f("hedge_entry"),
            hedge_exit=(float(c["hedge_exit"]) if c.get("hedge_exit") is not None else None),
            incoterm=str(c.get("incoterm") or "FOB"),
            carries_freight=bool(c.get("carries_freight", False)),
            freight_budget=f("freight_budget"),
            freight_actual=(float(c["freight_actual"])
                            if c.get("freight_actual") is not None else None),
            finance_rate=f("finance_rate"), finance_days=int(f("finance_days")),
            storage_rate=f("storage_rate"), storage_days=int(f("storage_days")),
            other_cost=f("other_cost"),
            basis_vol=(float(c["basis_vol"]) if c.get("basis_vol") is not None else None),
            stage=(c.get("stage") if c.get("stage") in CARGO_STAGES else "Booked"),
            booked_date=str(c.get("booked_date") or date.today().isoformat()),
            notes=str(c.get("notes") or ""))
        out.append(cg)
    return out


def book_deserialise(payload) -> Tuple[List[dict], List[dict]]:
    raw = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(raw, list):                       # legacy: positions only
        return blotter_deserialise(raw), []
    return (blotter_deserialise(raw.get("positions", [])),
            cargo_deserialise(raw.get("cargos", [])))


def book_save(positions: List[dict], cargos: List[dict]) -> None:
    try:
        os.makedirs(BLOTTER_DIR, exist_ok=True)
        with open(_blotter_path(_book_id()), "w") as f:
            f.write(book_serialise(positions, cargos))
    except Exception as e:
        LOG.warning("book save failed: %s", e)


def book_load() -> Tuple[List[dict], List[dict]]:
    path = _blotter_path(_book_id())
    try:
        if os.path.exists(path):
            with open(path) as f:
                return book_deserialise(f.read())
        if os.path.exists(LEGACY_BLOTTER):
            with open(LEGACY_BLOTTER) as f:
                return book_deserialise(f.read())
    except Exception as e:
        LOG.warning("book load failed: %s", e)
    return [], []


def ensure_book() -> Tuple[List[dict], List[dict]]:
    """THE single entry point to the book. Loads positions AND cargoes together,
    exactly once per session, and returns live references to both.

    This exists because three pages used to load the book three different ways. The
    Physical Cargo page loaded only cargoes, so a visitor landing there first left
    `positions` unset — and the next save wrote an EMPTY position list over their
    blotter. Two halves of one file must be loaded by one function, or they will
    eventually be saved by only one of them."""
    if "positions" not in st.session_state or "cargos" not in st.session_state:
        pos, cgs = book_load()
        st.session_state.setdefault("positions", pos)
        st.session_state.setdefault("cargos", cgs)
    return st.session_state.positions, st.session_state.cargos


def save_book() -> None:
    """Persist whatever is in session state. Never takes arguments: a save that can
    be handed a partial book is a save that will eventually be handed one."""
    book_save(st.session_state.get("positions", []), st.session_state.get("cargos", []))


def blotter_load() -> List[dict]:
    """Positions only — for call sites that genuinely do not touch cargoes."""
    return book_load()[0]


# ════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO RISK
#  Parametric VaR/ES on the correlated book, historical simulation, dated stress
#  episodes and curve-shape twists. Options enter at delta-cash and are fully
#  revalued in stress.
# ════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO RISK
# ══════════════════════════════════════════════════════════════════════════════
def portfolio_var(positions: List[dict], marks, corr: pd.DataFrame,
                  conf: float = 0.95, horizon: int = 1,
                  diversified: bool = True,
                  basis_legs: Optional[List[dict]] = None) -> dict:
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
    basis_legs        -> differential risk carried by physical cargoes: cash exposure
                         per 1.0 move of the diff, with an annual vol in quote units.
                         Added as an INDEPENDENT variance block (σ² = σ_futures² +
                         Σσ_basis²) because this desk has no differential history to
                         estimate a correlation from. Stated, not buried: a hedged
                         cargo whose flat price nets to zero is NOT riskless, and the
                         basis is exactly the risk a merchant is left holding.
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

    basis_rows, basis_var_sq = [], 0.0
    for bl in (basis_legs or []):
        sd = abs(float(bl["exposure"])) * float(bl["vol"]) / math.sqrt(252)
        basis_var_sq += sd ** 2
        basis_rows.append(dict(Factor=bl["label"], DailySigma=sd,
                               StandaloneVaR=sd * z * math.sqrt(horizon), _sd=sd))

    if not rows and not basis_rows:
        return dict(rows=[], basis_rows=[], var=0.0, es=0.0, undiversified=0.0,
                    gross=0.0, benefit=0.0, corr_used=False, basis_var=0.0,
                    reason="no marked positions")

    if not rows:
        sigma = math.sqrt(basis_var_sq)
        var = sigma * z * math.sqrt(horizon)
        return dict(rows=[], basis_rows=basis_rows, var=var,
                    es=sigma * norm.pdf(z) / (1 - conf) * math.sqrt(horizon),
                    undiversified=sum(b["StandaloneVaR"] for b in basis_rows),
                    gross=0.0, benefit=0.0, corr_used=False, basis_var=var,
                    reason="basis risk only — no futures position marked")

    undiversified = (sum(r["StandaloneVaR"] for r in rows)
                     + sum(b["StandaloneVaR"] for b in basis_rows))
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
        port_sigma_f = float(np.abs(w_vec).sum())
    else:
        port_sigma_f = port_sigma

    # Independent variance block: total σ² = σ_futures² + Σ σ_basis².
    port_sigma = (math.sqrt(port_sigma_f ** 2 + basis_var_sq) if corr_used
                  else port_sigma_f + math.sqrt(basis_var_sq))

    var = port_sigma * z * math.sqrt(horizon)
    es  = port_sigma * norm.pdf(z) / (1 - conf) * math.sqrt(horizon)

    if corr_used and port_sigma > 0:
        # Euler decomposition on the TOTAL sigma, so futures and basis components
        # still sum exactly to VaR.
        mcv = (R @ w_vec) / port_sigma
        for i, rrow in enumerate(rows):
            comp = w_vec[i] * mcv[i] / port_sigma * var
            rrow["ComponentVaR"] = float(comp)
            rrow["PctOfVaR"] = float(comp / var * 100) if var else 0.0
        for brow in basis_rows:
            comp = brow["_sd"] ** 2 / port_sigma ** 2 * var
            brow["ComponentVaR"] = float(comp)
            brow["PctOfVaR"] = float(comp / var * 100) if var else 0.0
    else:
        for rrow in rows:
            rrow["ComponentVaR"] = rrow["StandaloneVaR"]
            rrow["PctOfVaR"] = (rrow["StandaloneVaR"] / undiversified * 100
                                if undiversified else 0.0)
        for brow in basis_rows:
            brow["ComponentVaR"] = brow["StandaloneVaR"]
            brow["PctOfVaR"] = (brow["StandaloneVaR"] / undiversified * 100
                                if undiversified else 0.0)
    for brow in basis_rows:
        brow.pop("_sd", None)

    for rrow in rows:
        rrow.pop("_dvol", None)

    return dict(rows=rows, basis_rows=basis_rows, var=var, es=es,
                undiversified=undiversified, gross=gross,
                basis_var=math.sqrt(basis_var_sq) * z * math.sqrt(horizon),
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


def stress_curve_shape(positions: List[dict], marks, front_pct: float,
                       back_pct: float, pivot_years: float = 1.0) -> dict:
    """Revalue the book under a curve TWIST.

    Each position is shocked by the factor at ITS OWN tenor: a dated future uses its
    strip T, an option its remaining tenor, a front-month future the front shock.
    Options are fully revalued at the shocked forward (Black-76), never linearised —
    same discipline as the historical episodes."""
    rows, total = [], 0.0
    for p in positions:
        base = position_base_price(p, marks)
        if base is None:
            continue
        if p.get("kind", "future") == "option":
            T = option_time_remaining(p)
        elif p.get("strip_ticker"):
            dm = dated_mark(p["strip_ticker"])
            T = dm["T"] if dm else 0.0
        else:
            T = 0.0
        f = float(shape_shock_factors([T], front_pct, back_pct, pivot_years)[0])
        pnl = position_pnl_at(p, base, f)
        total += pnl
        rows.append(dict(Position=_position_label(p), Tenor=T, Shock=f * 100, PnL=pnl))
    if not rows:
        return dict(available=False)
    return dict(available=True, rows=rows, total=total)


# ════════════════════════════════════════════════════════════════════════════
#  UI — chrome shared by every page
#  Page setup, KPI cards, badges, the header, the sectioned sidebar and the
#  per-page explanations.
# ════════════════════════════════════════════════════════════════════════════


def _setup_page() -> None:
    """Page config + CSS. Called from main() so the module stays import-safe for tests."""
    st.set_page_config(page_title="S&D — Commodity Trading Desk", page_icon="🌐",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


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
        "📒  Trade Blotter", "🚢  Physical Cargo", "⚠️  Portfolio Risk",
    ],
    "Reference": [
        "🌍  Macro Rates", "ℹ️  About",
    ],
}


ALL_PAGES = [p for ps in NAV_SECTIONS.values() for p in ps]


DEFAULT_PAGE = ALL_PAGES[0]


def _on_nav(section: str) -> None:
    """Radio-sync: selecting a page in one section clears the other sections'
    radios and records the destination. Native widgets only — the previous
    HTML-in-flow nav collided with Streamlit's internal markdown margins."""
    val = st.session_state.get(f"navsec_{section}")
    if val is None:
        return
    st.session_state.nav_page = val
    for s in NAV_SECTIONS:
        if s != section:
            st.session_state[f"navsec_{s}"] = None


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
            k = f"navsec_{section}"
            if k not in st.session_state:
                st.session_state[k] = current if current in pages else None
            st.radio(section, pages, key=k, on_change=_on_nav, args=(section,))
        page = st.session_state.nav_page
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


# ── Per-page explanations, rendered by render_header under the title ─────────
#  Plain English (B2/C1): short sentences, every abbreviation defined where used.
PAGE_HELP: Dict[str, str] = {
    "Trading Dashboard":
        "The overview page. It shows the last traded price (the <b>mark</b>) for every "
        "contract, the date of that settle, and the daily change. <b>MTM</b> = "
        "mark-to-market: everything is valued with real market prices, never estimates. "
        "<b>MA50 / MA200</b> = the average price of the last 50 / 200 trading days.",
    "Forward Curves":
        "A <b>forward curve</b> (the 'strip') is the price of the same commodity for "
        "different delivery months. <b>M1</b> = the first month (the 'front'). "
        "<b>Contango</b> = later months cost more. <b>Backwardation</b> = later months "
        "cost less. <b>Roll yield</b> = the yearly gain or loss from holding a position "
        "as it moves from one month to the next. <b>T</b> = time to delivery, in years. "
        "The <b>Evolution</b> tab compares today's curve with earlier dates: a "
        "<b>parallel shift</b> means the whole curve moved together (a macro story), a "
        "<b>twist</b> means the front and the back moved differently (a physical one).",
    "Calendar Spreads":
        "A <b>calendar spread</b> is the price difference between two delivery months "
        "of the same commodity (for example M1 − M2). The <b>percentile</b> shows where "
        "today's spread sits inside its own 2-year history: high = historically "
        "expensive, low = historically cheap.",
    "Cracks, Crush & Arbs":
        "Margin trades between related products. A <b>crack</b> = the refinery margin "
        "(fuel prices minus crude, in $ per barrel, <b>bbl</b>). The <b>crush</b> = the "
        "soybean processing margin (meal + oil − beans). An <b>arb</b> (arbitrage) = the "
        "price gap between two markets for the same thing. Being long a structure means "
        "being long that processing margin.",
    "Correlation":
        "How contracts move together, from daily <b>log returns</b> (percentage "
        "changes). <b>+1</b> = they move the same way, <b>−1</b> = opposite ways, "
        "<b>0</b> = no link. The Portfolio Risk page uses this matrix to net long and "
        "short positions against each other.",
    "Seasonality":
        "Some commodities repeat a pattern every year: heating gas in winter, driving "
        "gasoline in summer. Each box shows the spread of returns for that calendar "
        "month over the lookback. <b>Hit rate</b> = the share of years where the month "
        "was positive. Read the roll-bias warning at the bottom before trading this.",
    "EIA Fundamentals":
        "<b>EIA</b> = the US Energy Information Administration, the official source of "
        "US energy data. Weekly <b>inventories</b> (stocks in storage) and production. "
        "<b>WoW</b> = week over week change; <b>vs 1y ago</b> = against the same week "
        "last year. Falling stocks usually support prices; building stocks weigh on them.",
    "Regional Balances":
        "A simple map of who produces and who consumes. <b>S&D</b> = supply and demand. "
        "Green = net exporter (supply above demand); red = net importer. These are "
        "static yearly estimates for orientation — the one page here that is not live, "
        "and it says so.",
    "Options Pricer":
        "Prices a European option on a future with the <b>Black-76</b> model. "
        "<b>F</b> = forward price, <b>K</b> = strike, <b>T</b> = time to expiry in "
        "years, <b>σ</b> (sigma) = volatility. The Greeks: <b>delta</b> = how much the "
        "option moves when the future moves; <b>gamma</b> = how fast delta itself "
        "changes; <b>vega</b> = sensitivity to volatility; <b>theta</b> = value lost "
        "per day. <b>RV</b> = realised volatility (measured from history), used as the "
        "starting value for σ.",
    "Vol Surface":
        "A 3D picture of volatility across strikes and expiries. <b>ATM</b> = "
        "at-the-money, a strike close to the forward. <b>Skew</b> = the tilt of the "
        "curve (downside protection priced differently from upside). The <b>smile</b> = "
        "its curvature. This surface is a stated formula seeded from realised vol, not "
        "market quotes — use it to reason about shape.",
    "Trade Blotter":
        "Your book of positions. Book a <b>future</b> (the continuous front month, or a "
        "specific dated month) or an <b>option</b>. <b>Lots</b> = number of contracts; "
        "<b>entry</b> = your trade price; <b>P&L</b> = profit and loss against today's "
        "mark. Options lose time value every day and expire at intrinsic value. The book "
        "is stored under the <b>?book=</b> id in the URL — export the JSON as your "
        "durable backup. The <b>roll calendar</b> at the bottom shows when each front "
        "contract must be rolled and what rolling costs today.",
    "Portfolio Risk":
        "How much the book can lose. <b>VaR</b> (Value at Risk) = the loss you should "
        "not exceed on most days, at the chosen confidence level. <b>ES</b> (Expected "
        "Shortfall) = the average loss on the bad days beyond VaR. Options are counted "
        "at <b>delta-cash</b>: their futures-equivalent size. The stress section replays "
        "real historical episodes (dated) on today's book, and the <b>curve-shape</b> "
        "(twist) test moves the front and the back by different amounts — the shock a "
        "spread book actually fears.",
    "Monte Carlo":
        "Simulates thousands of possible price paths. <b>GBM</b> = a random walk with "
        "no anchor (fits gold). <b>Schwartz / OU</b> = prices pulled back toward a "
        "level over time (<b>mean reversion</b>); the <b>half-life</b> = the time to "
        "close half of any gap. Paths are centred on the live forward curve, so the "
        "average path equals the market's own forward. <b>P5 / P95</b> = the 5% and "
        "95% levels of the fan.",
    "Macro Rates":
        "The macro backdrop from <b>FRED</b> (the Federal Reserve's economic database). "
        "<b>CPI</b> = consumer price inflation; <b>GDP</b> = economic output; the "
        "policy rate = the central bank's rate. The commodity tab adds the dollar index "
        "(<b>DXY</b> proxy), the 10-year yield and <b>breakeven inflation</b> — a "
        "stronger dollar with higher real yields is the classic headwind for gold.",
    "Signal Scanner":
        "One row per contract, one column per signal: carry, momentum, volatility "
        "regime, seasonality and positioning. Nothing here is a model — every number "
        "comes from live market data. The legend under the table explains each column "
        "and its abbreviations.",
    "Event Calendar":
        "The scheduled reports that move these markets. Weekly prints (EIA, rig count) "
        "follow a fixed weekday and are computed. Monthly reports are anchored to their "
        "usual date and marked approximate: <b>WASDE</b> = the USDA's world "
        "supply-and-demand report; <b>MOMR</b> = OPEC's monthly oil market report. "
        "Always verify against the official calendar before carrying risk into a print.",
    "Storage & Cash-and-Carry":
        "The trade behind the curve. <b>Cash-and-carry (C&C)</b> = buy the commodity "
        "now, store it, and sell it forward — it pays when the forward premium covers "
        "financing plus storage. First the page shows what the market implies with zero "
        "assumptions (the strip minus financing at <b>SOFR</b>, the US overnight rate); "
        "then your verdict with an editable storage cost. <b>Full carry</b> = the curve "
        "exactly covers all costs. <b>Convenience yield</b> = the premium the market "
        "pays to whoever holds physical now (seen in backwardation).",
    "COT Positioning":
        "<b>COT</b> = Commitments of Traders, the weekly <b>CFTC</b> (US regulator) "
        "report on who holds futures. <b>MM</b> = Managed Money — funds, the "
        "speculators. <b>PM</b> / commercials = Producers and Merchants — the physical "
        "players. <b>OI</b> = open interest, the total contracts outstanding. "
        "<b>Net</b> = long minus short. A very high percentile = a crowded trade that "
        "can unwind fast. Positions are as of Tuesday, published Friday.",
    "Physical Cargo":
        "A physical cargo is automatically <b>long flat price</b> — you own the barrels "
        "or the bushels. This page books the cargo, derives the futures hedge that "
        "cancels that exposure, and splits the result into what a merchant is actually "
        "paid for. <b>Basis</b> (or the differential) = the price of a specific grade at "
        "a specific place, quoted against the benchmark, e.g. WTI minus $0.90. "
        "<b>Incoterm</b> (FOB, CFR, CIF) = who pays freight and insurance. <b>Laycan</b> "
        "= the loading window. <b>Landed cost</b> = purchase price plus every cost you "
        "carry. <b>Demurrage</b> = the penalty for holding a ship too long. Differentials "
        "and freight are your own inputs — this desk has no assessment feed, so every "
        "number derived from them is tagged, and an unsold cargo is marked, not realised.",
    "About":
        "The design contract of this desk: what is live, what is honestly not, what was "
        "excluded and why, the full revision changelog, and the known limits — stated "
        "plainly.",
}


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
    help_txt = PAGE_HELP.get(title)
    if help_txt:
        st.markdown(f'<div class="page-help">{help_txt}</div>', unsafe_allow_html=True)
    st.markdown("")


# ════════════════════════════════════════════════════════════════════════════
#  PAGES — one section per screen
#  Each page is self-contained: it reads marks, renders, and returns. A page
#  cannot break another one — main() runs each inside a guard.
# ════════════════════════════════════════════════════════════════════════════


# ────────────────────────────────────────────────────────────────────────────
#  Trading Dashboard — the board at a glance: marks, settle dates, movers,
#  the performance treemap and the two-year chart.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Forward Curves — the live dated strip, and the Evolution tab that compares it
#  with earlier dates and splits the move into shift and twist.
# ────────────────────────────────────────────────────────────────────────────


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
    tab_now, tab_evo = st.tabs(["CURRENT STRIP", "EVOLUTION"])

    with tab_now:
        _curve_snapshot(sel, strip, c)
    with tab_evo:
        _curve_evolution(sel, strip, c)


def _curve_snapshot(sel: str, strip: pd.DataFrame, c: dict) -> None:
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


def _curve_evolution(sel: str, strip: pd.DataFrame, c: dict) -> None:
    """Compare today's strip against the SAME dated contracts on earlier dates."""
    st.markdown("A strip is a photograph; the desk trades the movement. Each past curve "
                "below is the settle of the **same dated contracts** on that date — no "
                "continuous series, no roll artefacts.")

    hist = fetch_strip_history(sel, "1y")
    if hist.empty:
        st.error("**NO STRIP HISTORY** — dated contracts returned no history. "
                 "Nothing is drawn in its place.")
        return

    today = date.today()
    hmin = hist.index[0].date()
    c1, c2 = st.columns([2, 1])
    presets = {"1 week ago": 7, "2 weeks ago": 14, "1 month ago": 30,
               "3 months ago": 91, "Custom…": None}
    choice = c1.radio("Compare with", list(presets), horizontal=True, index=0)
    if presets[choice] is None:
        cmp_dates = c1.multiselect(
            "Pick dates", options=[d.date() for d in hist.index[::-1]],
            default=[hist.index[-1].date()],
            format_func=lambda d: d.isoformat(), max_selections=3)
    else:
        cmp_dates = [max(today - timedelta(days=presets[choice]), hmin)]
    show_pct = c2.toggle("Show change in %", value=False)
    c2.caption(f"History available from **{hmin}**.")

    curves = []
    for d0 in cmp_dates:
        past = curve_on_date(strip, hist, d0)
        if past is None:
            st.warning(f"No settles on or before {d0} — that date predates the history.")
            continue
        curves.append((d0, past))
    if not curves:
        return

    # ── Overlay ─────────────────────────────────────────────────────────────
    fig = go.Figure()
    shades = [BLUE, PURPLE, TEAL]
    for i, (d0, past) in enumerate(curves):
        fig.add_trace(go.Scatter(x=past["label"], y=past["price"], mode="lines+markers",
                                 name=f"{d0}", line=dict(color=shades[i % 3], width=1.5,
                                                         dash="dot"),
                                 marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=strip["label"], y=strip["price"], mode="lines+markers",
                             name="Today", line=dict(color=AMBER, width=2.4),
                             marker=dict(size=7)))
    fig.update_layout(title=f"{sel} forward strip — today vs earlier ({c['unit']})")
    st.plotly_chart(_styled(fig, 420), use_container_width=True)

    # ── Decomposition against the FIRST comparison date ──────────────────────
    d0, past = curves[0]
    move = curve_move(strip, past)
    if move.empty:
        st.info("No delivery months in common with that date — nothing to decompose.")
        return
    dec = decompose_move(move)

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Parallel shift", f"{dec['shift']:+,.3f}",
        f"average across the curve · {dec['shift_pct']:+.2f}%",
        GREEN if dec["shift"] > 0 else RED)
    kpi(k2, "Twist (front − back)", f"{dec['twist']:+,.3f}",
        "positive = front outperformed", AMBER)
    kpi(k3, "Front move", f"{dec['front']:+,.3f}", str(move["label"].iloc[0]))
    kpi(k4, "Back move", f"{dec['back']:+,.3f}", str(move["label"].iloc[-1]))
    st.markdown(f'<span class="badge badge-amber">{dec["shape"]}</span> '
                f'<span style="color:{GRAY};font-size:0.8rem">{dec["read"]}</span>',
                unsafe_allow_html=True)

    col = "change_pct" if show_pct else "change"
    unit_lbl = "%" if show_pct else c["unit"]
    fig2 = go.Figure(go.Bar(x=move["label"], y=move[col],
                            marker_color=[GREEN if v >= 0 else RED for v in move[col]]))
    fig2.add_hline(y=0, line=dict(color=BORDER))
    fig2.update_layout(title=f"Change by delivery month vs {d0} ({unit_lbl}) — "
                             "the shape of the move")
    st.plotly_chart(_styled(fig2, 340), use_container_width=True)

    tbl = move[["label", "delivery", "T", "price_then", "price", "change", "change_pct"]].copy()
    tbl.columns = ["Contract", "Delivery", "T (yrs)", f"Price {d0}", "Price today",
                   "Change", "Change %"]
    st.dataframe(tbl.style.format({"T (yrs)": "{:.2f}", f"Price {d0}": "{:,.2f}",
                                   "Price today": "{:,.2f}", "Change": "{:+,.3f}",
                                   "Change %": "{:+.2f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("Months are matched on DELIVERY, never on position — the front rolls, so "
               "'M1 today' and 'M1 last month' are different contracts. Missing months "
               "(expired, or not yet listed on the earlier date) are simply absent.")


# ────────────────────────────────────────────────────────────────────────────
#  Calendar Spreads — the price difference between two delivery months of the
#  same commodity, with the history of that exact dated pair.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Cracks, Crush & Arbs — refinery and processing margins, the transatlantic
#  arb and the gold/silver ratio.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Signal Scanner — the whole board in one table: carry, momentum, vol regime,
#  seasonality and positioning, off four grouped requests.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Storage & Cash-and-Carry — what the curve implies for storage with no
#  assumptions, then the arb verdict with yours.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  COT Positioning — the CFTC weekly report: who holds the risk and how crowded
#  the trade has become.
# ────────────────────────────────────────────────────────────────────────────


def cot_tail_table(cot: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Display table of the last n reports, newest first. The date column is taken
    by POSITION after reset_index — the production KeyError came from renaming a
    column literally called 'index' while the index was actually named 'date'
    (DatetimeIndex built from a named Series keeps that name)."""
    tail = cot.tail(n).iloc[::-1].reset_index()
    tail = tail.rename(columns={tail.columns[0]: "Report"})
    tail["Report"] = pd.to_datetime(tail["Report"]).dt.date
    show = tail[["Report", "oi", "mm_long", "mm_short", "mm_net", "mm_net_pct_oi", "pm_net"]].copy()
    show.columns = ["Report", "Open Int", "MM Long", "MM Short", "MM Net", "MM %OI", "PM Net"]
    return show


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
    show = cot_tail_table(cot, 8)
    st.dataframe(show.style.format({"Open Int": "{:,.0f}", "MM Long": "{:,.0f}",
                                    "MM Short": "{:,.0f}", "MM Net": "{:+,.0f}",
                                    "MM %OI": "{:+.1f}", "PM Net": "{:+,.0f}"}, na_rep="—"),
                 use_container_width=True, hide_index=True)
    st.caption("CFTC Disaggregated report (futures-only), weekly: released Friday "
               "~15:30 ET with positions as of Tuesday — the print is always 3 days "
               "stale, which matters in fast tape. Crowded longs at high percentiles are "
               "fuel for washouts; commercials leaning the other way is the classic tell. "
               "Reading, not gospel: COT is positioning, not a signal by itself.")


# ────────────────────────────────────────────────────────────────────────────
#  EIA Fundamentals — the weekly US energy data: inventories, Cushing, production.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Regional Balances — a static map of who produces and who consumes. The one
#  screen here that is not live, and it says so.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Event Calendar — the scheduled prints, with the basis of each date stated:
#  computed, approximate, or genuinely irregular.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Correlation — how the board moves together, and the matrix the portfolio VaR
#  runs on.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Seasonality — monthly return distributions, with the roll-bias caveat that
#  matters more than the boxes.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Monte Carlo — price paths centred on the live forward curve, GBM or a
#  mean-reverting Schwartz process.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Options Pricer — Black-76 on a live dated forward, with every Greek shown in
#  cash terms.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Vol Surface — a stated parametrisation seeded from realised vol, for reasoning
#  about shape. Not a marking tool: there is no options feed here.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  Trade Blotter — the book: futures (front or dated), options, cargo hedge legs,
#  P&L attribution, net Greeks and the roll calendar.
# ────────────────────────────────────────────────────────────────────────────


def page_blotter(marks: MarkBoard) -> None:
    render_header(marks, "Trade Blotter", "Futures (front or dated), options — marked live, per-book storage")
    positions, cargos = ensure_book()
    hedges = cargo_hedge_positions(cargos)
    if hedges:
        st.caption(f"➕ {len(hedges)} cargo hedge leg(s) are included below and in risk. "
                   "They are **derived** from the Physical Cargo page — edit the cargo, "
                   "not the leg, and the two can never drift apart.")

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
            save_book()
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
            save_book()
            st.rerun()

    with tab_io:
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Export book (JSON)", book_serialise(positions, cargos),
                           file_name=f"book_{_book_id()}.json", use_container_width=True)
        up = c2.file_uploader("Import book", type="json", label_visibility="collapsed")
        if up is not None and c2.button("Load imported book", use_container_width=True):
            try:
                pos_in, cgs_in = book_deserialise(up.read().decode())
                st.session_state.positions = pos_in
                st.session_state.cargos = cgs_in
                save_book()
                st.success(f"Book loaded — {len(pos_in)} position(s), {len(cgs_in)} "
                           "cargo(es). Legacy options without a trade date restart "
                           "their tenor from today (stated, not silent).")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")
        if c3.button("🗑️ Flatten book", use_container_width=True):
            st.session_state.positions = []
            st.session_state.cargos = []
            save_book()
            st.rerun()
        st.caption("Export and import carry **positions and cargoes together** — half a "
                   "book is not a backup.  \n"
                   f"Book id **{_book_id()}** — carried in this page's URL (?book=…): "
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
        save_book()
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

    book = positions + hedges
    st.markdown("### Net book Greeks")
    gk = book_greeks(book, marks)
    t = gk["total"]
    g1, g2, g3, g4 = st.columns(4)
    kpi(g1, "Delta ($/1.0 move)", f"{t['delta']:+,.0f}", "cash per unit price move")
    kpi(g2, "Gamma", f"{t['gamma']:+,.2f}", "delta change per unit")
    kpi(g3, "Vega ($/vol pt)", f"{t['vega']:+,.0f}", "options only",
        PURPLE if t["vega"] else GRAY)
    kpi(g4, "Theta ($/day)", f"{t['theta']:+,.0f}", "options age daily now",
        RED if t["theta"] < 0 else GREEN)

    st.markdown("### Roll calendar")
    rc = roll_calendar(book, marks)
    if not rc:
        st.caption("Nothing to roll.")
        return
    urgent = [r for r in rc if r["Days"] is not None and r["Days"] <= 7
              and r["Kind"] == "Front future"]
    tot_roll = sum(r["RollCost"] for r in rc if r["RollCost"] is not None)
    r1, r2, r3 = st.columns(3)
    kpi(r1, "Rolls due within 7d", f"{len(urgent)}",
        "front contracts approaching expiry", RED if urgent else GREEN)
    kpi(r2, "Cost of rolling today", f"${tot_roll:+,.0f}",
        "M1−M2 carry across front lines", GREEN if tot_roll >= 0 else RED)
    nxt = next((r for r in rc if r["Days"] is not None), None)
    kpi(r3, "Next deadline", f"{nxt['Days']}d" if nxt else "—",
        (f"{nxt['Contract']} · {nxt['Expiry']}" if nxt and nxt["Expiry"] else ""))
    if urgent:
        st.warning("**Roll now:** " + ", ".join(r["Position"] for r in urgent) +
                   " — a front future not rolled before its last trading day goes to "
                   "delivery. This is an operational deadline, not a view.")
    rdf = pd.DataFrame(rc)
    rdf["Expiry"] = rdf["Expiry"].apply(lambda d: d.isoformat() if d else "—")
    st.dataframe(rdf.style.format({"RollCost": "{:+,.0f}", "Days": "{:.0f}"}, na_rep="—"),
                 use_container_width=True, hide_index=True)
    st.caption("Expiry dates are the desk's per-contract estimates (generous by design — "
               "see About). Roll cost is today's M1−M2 spread applied to your size and "
               "direction: negative means the roll costs you, positive means backwardation "
               "pays you to roll. Dated lines show no roll cost — not rolling is why you "
               "booked them. Options show days to expiry instead.")


# ────────────────────────────────────────────────────────────────────────────
#  Physical Cargo — the merchant layer: book a cargo, derive its hedge, and split
#  the result into basis, freight, carry and residual flat price.
# ────────────────────────────────────────────────────────────────────────────


def page_cargo(marks: MarkBoard) -> None:
    render_header(marks, "Physical Cargo",
                  "Cargo economics with the paper hedge — flat price out, basis and freight in")
    _, cargos = ensure_book()
    t_book, t_list, t_attr, t_life = st.tabs(
        ["BOOK CARGO", f"CARGO BOOK ({len(cargos)})", "ATTRIBUTION", "LIFECYCLE"])

    with t_book:
        _cargo_book_form(marks)
    with t_list:
        _cargo_book_list(marks, cargos)
    with t_attr:
        _cargo_attribution_tab(marks, cargos)
    with t_life:
        _cargo_lifecycle_tab(marks, cargos)


def _cargo_book_form(marks: MarkBoard) -> None:
    st.markdown("Every field below the volume is optional — leave freight, storage or "
                "financing at zero and the model collapses cleanly to a simple hedged "
                "cargo.")
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("Benchmark contract", list(COMMODITIES), key="cg_bench")
    side = c2.radio("Side", ["Buy", "Sell forward"], horizontal=True, key="cg_side")
    side_key = "Buy" if side == "Buy" else "Sell"
    grade = c3.text_input("Grade / location", value="", placeholder="e.g. Midland, FOB Houston")

    mark = require_mark(marks, name)
    if mark is None:
        return
    c = COMMODITIES[name]
    strip = fetch_forward_strip(name)
    if strip.empty:
        st.error("**NO LIVE STRIP** — a cargo prices against a dated month. "
                 "No curve is fitted in its place.")
        return

    # ── Pricing ──────────────────────────────────────────────────────────────
    st.markdown("##### Pricing")
    p1, p2, p3 = st.columns(3)
    pidx = p1.selectbox("Pricing month", range(len(strip)),
                        format_func=lambda i: f"{strip['label'].iloc[i]} (T={strip['T'].iloc[i]:.2f}y)")
    B0_live = float(strip["price"].iloc[pidx])
    price_mode = p2.radio("Price as", ["Differential to benchmark", "Outright (fixed)"],
                          key="cg_pmode",
                          help="An outright price is not a separate model: the desk "
                               "computes the differential it implies against the "
                               "pricing month and runs the same arithmetic.")
    bench_buy = p3.number_input("Benchmark at pricing", value=float(round(B0_live, 4)),
                                step=0.01, format="%.4f",
                                help="Defaults to the live settle of the pricing month. "
                                     "Overwrite with the price you actually priced against.")
    q1, q2 = st.columns(2)
    if price_mode.startswith("Differential"):
        diff_buy = q1.number_input(f"Differential ({c['unit']})", value=0.0,
                                   step=0.01, format="%.4f")
        q2.metric("Implied outright", f"{bench_buy + diff_buy:,.4f}")
    else:
        outright = q1.number_input(f"Outright price ({c['unit']})",
                                   value=float(round(B0_live, 4)), step=0.01, format="%.4f")
        diff_buy = outright - bench_buy
        q2.metric("Implied differential", f"{diff_buy:+,.4f}",
                  help="This is what the outright price is worth relative to the "
                       "pricing month — the number the desk actually trades.")

    pb1, pb2 = st.columns(2)
    pricing_basis = pb1.radio("Pricing basis", ["Single settle", "Average over window", "Fixed"],
                              horizontal=True)
    if pricing_basis.startswith("Average"):
        win = pb2.date_input("Pricing window", value=(date.today(), date.today() + timedelta(days=30)))
        st.caption("⚠️ A hedge against an **average** unwinds progressively; this desk "
                   "models a single hedge price, so the timing risk inside the window is "
                   "**not** captured. Stated, not hidden.")
    else:
        win = None

    # ── Volume ───────────────────────────────────────────────────────────────
    st.markdown("##### Volume")
    v1, v2, v3 = st.columns(3)
    units = cargo_trade_units(name)
    tu = v1.selectbox("Trade unit", list(units))
    factor = v2.number_input(f"{c['size_unit']} per {tu}", value=float(units[tu]),
                             step=0.0001, format="%.4f",
                             help="Density and test weight are grade-dependent — this "
                                  "default is indicative. Overwrite it and the value used "
                                  "is stored with the cargo.")
    volume = v3.number_input(f"Volume ({tu})", value=100_000.0, step=1_000.0, min_value=0.0)
    V = volume * factor
    st.caption(f"= **{V:,.0f} {c['size_unit']}** · one lot = {c['contract_size']:,} "
               f"{c['size_unit']} → {V / c['contract_size']:,.2f} lots")

    # ── Hedge ────────────────────────────────────────────────────────────────
    st.markdown("##### Hedge")
    h1, h2, h3 = st.columns(3)
    ratio = h1.slider("Hedge ratio", 0.0, 1.2, 1.0, 0.05,
                      help="Deliberate under-hedging is a merchant decision, not an "
                           "error — the open slice is reported as residual flat price.")
    proposed = cargo_hedge_lots(name, V, ratio)
    lots = h2.number_input("Hedge lots", value=int(proposed), step=1, min_value=0)
    hidx = h3.selectbox("Hedge month", range(len(strip)), index=pidx,
                        format_func=lambda i: strip["label"].iloc[i])
    hedge_entry = st.number_input("Hedge entry price",
                                  value=float(round(strip["price"].iloc[hidx], 4)),
                                  step=0.01, format="%.4f")
    Vh = lots * c["contract_size"]
    resid = V - Vh
    r1, r2, r3 = st.columns(3)
    r1.metric("Hedged volume", f"{Vh:,.0f} {c['size_unit']}")
    r2.metric("Residual (unhedged)", f"{resid:+,.0f} {c['size_unit']}",
              delta=f"{resid / V * 100:+.2f}%" if V else None, delta_color="off")
    r3.metric("Hedge month vs pricing",
              "same" if hidx == pidx else f"{strip['label'].iloc[hidx]} vs {strip['label'].iloc[pidx]}")
    if hidx != pidx:
        st.caption("Hedging a different month than you price against creates a **carry / "
                   "timing** leg — it will appear as its own line in the attribution.")

    # ── Costs ────────────────────────────────────────────────────────────────
    st.markdown("##### Costs")
    inc1, inc2, inc3 = st.columns(3)
    incoterm = inc1.selectbox("Incoterm", INCOTERMS)
    carries = inc2.toggle("I carry the freight",
                          value=cargo_carries_freight(side_key, incoterm),
                          help="Seeded from the Incoterm and your side. Real contracts "
                               "are messier than the three-letter code — override freely.")
    freight_budget = inc3.number_input(f"Freight budget ({c['unit']})", value=0.0,
                                       step=0.01, format="%.4f")
    s1, s2, s3, s4 = st.columns(4)
    sofr = live_sofr()
    fin_rate = s1.number_input("Financing rate (%)",
                               value=float(round((sofr[0] * 100) if sofr else 4.50, 2)),
                               step=0.05) / 100.0
    fin_days = s2.number_input("Days financed", value=0, step=1, min_value=0)
    stor_def, _ = default_storage_pm(name, mark)
    stor_rate = s3.number_input(f"Storage ({c['unit']}/month)",
                                value=float(round(stor_def or 0.0, 4)),
                                step=0.001, format="%.4f")
    stor_days = s4.number_input("Days stored", value=0, step=1, min_value=0)
    o1, o2 = st.columns(2)
    other = o1.number_input("Demurrage / other ($ cash)", value=0.0, step=100.0)
    bvol = o2.number_input(f"Basis vol (annual, {c['unit']})",
                           value=float(round(default_basis_vol(name, mark), 4)),
                           step=0.01, format="%.4f",
                           help="Volatility of the DIFFERENTIAL, used by Portfolio Risk. "
                                "Differential histories are licensed data this desk does "
                                "not have — this default is indicative.")
    notes = st.text_input("Notes", value="", placeholder="laycan, counterparty, terms…")

    if st.button("📦 Book cargo + hedge", type="primary", use_container_width=True):
        cg = dict(
            id=f"c{uuid.uuid4().hex[:6]}", commodity=name, side=side_key,
            grade=grade.strip(), volume=float(volume), trade_unit=tu,
            unit_factor=float(factor),
            pricing_ticker=str(strip["ticker"].iloc[pidx]),
            pricing_label=str(strip["label"].iloc[pidx]),
            pricing_basis=pricing_basis,
            pricing_window=[d.isoformat() for d in win] if win else None,
            bench_buy=float(bench_buy), diff_buy=float(diff_buy),
            diff_mark=float(diff_buy), diff_mark_date=date.today().isoformat(),
            diff_sell=None, bench_sell=None,
            hedge_lots=int(lots), hedge_ticker=str(strip["ticker"].iloc[hidx]),
            hedge_label=str(strip["label"].iloc[hidx]),
            hedge_entry=float(hedge_entry), hedge_exit=None,
            incoterm=incoterm, carries_freight=bool(carries),
            freight_budget=float(freight_budget), freight_actual=None,
            finance_rate=float(fin_rate), finance_days=int(fin_days),
            storage_rate=float(stor_rate), storage_days=int(stor_days),
            other_cost=float(other), basis_vol=float(bvol),
            stage="Booked", booked_date=date.today().isoformat(), notes=notes.strip())
        st.session_state.cargos = list(cargos) + [cg]
        save_book()
        st.success(f"Cargo **{cg['id']}** booked — hedge {lots} lots "
                   f"{'short' if side_key == 'Buy' else 'long'} "
                   f"{strip['label'].iloc[hidx]} is now live in the Trade Blotter.")
        st.rerun()


def _cargo_book_list(marks: MarkBoard, cargos: List[dict]) -> None:
    if not cargos:
        st.info("No cargo booked yet.")
        return
    rows = []
    for cg in cargos:
        a = cargo_attribution(cg, marks)
        rows.append(dict(
            ID=cg["id"], Stage=cg["stage"], Side=cg["side"],
            Contract=cg["commodity"], Grade=cg.get("grade") or "—",
            Volume=f"{cg['volume']:,.0f} {cg['trade_unit']}",
            Pricing=cg.get("pricing_label", "—"),
            Diff=cg["diff_buy"], Hedge=f"{cg.get('hedge_lots', 0)} lots",
            Landed=(a["landed_cost"] if a.get("available") else np.nan),
            Result=(a["net"] if a.get("available") else np.nan)))
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"Diff": "{:+,.4f}", "Landed": "{:,.4f}",
                                  "Result": "{:+,.0f}"}, na_rep="NO MARK"),
                 use_container_width=True, hide_index=True)
    st.caption("Landed cost is purchase price plus every cost you carry, per unit. "
               "A cargo whose pricing contract has no mark shows **NO MARK** and is "
               "excluded from totals rather than proxied.")

    d1, d2 = st.columns([3, 1])
    victim = d1.selectbox("Remove cargo", [c["id"] for c in cargos],
                          format_func=lambda i: f"{i} — "
                          f"{next(c['commodity'] for c in cargos if c['id'] == i)}")
    if d2.button("Delete", use_container_width=True):
        st.session_state.cargos = [c for c in cargos if c["id"] != victim]
        save_book()
        st.rerun()
    st.caption("Deleting a cargo removes its hedge leg too — the hedge is derived from "
               "the cargo, never stored separately, so the two can never drift apart.")


def _cargo_attribution_tab(marks: MarkBoard, cargos: List[dict]) -> None:
    if not cargos:
        st.info("No cargo booked yet.")
        return
    cid = st.selectbox("Cargo", [c["id"] for c in cargos],
                       format_func=lambda i: (
                           f"{i} — {next(c['commodity'] for c in cargos if c['id'] == i)} "
                           f"{next(c.get('grade') or '' for c in cargos if c['id'] == i)}"))
    cg = next(c for c in cargos if c["id"] == cid)
    a = cargo_attribution(cg, marks)
    if not a.get("available"):
        st.error(f"**NO MARK** — {a.get('reason')}. Nothing is shown in its place.")
        return

    name = cg["commodity"]
    c = COMMODITIES[name]
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Landed cost", f"{a['landed_cost']:,.4f}", f"{c['unit']} all-in")
    kpi(k2, "Hedge ratio", f"{a['hedge_ratio']:.1f}%",
        f"residual {a['residual_volume']:+,.0f} {c['size_unit']}",
        GREEN if abs(a["hedge_ratio"] - 100) < 2 else AMBER)
    kpi(k3, "Flat price, net", f"${a['flat_net']:+,.0f}",
        "physical + hedge — should be ~0 when fully hedged",
        GREEN if abs(a["flat_net"]) < abs(a["net"]) else AMBER)
    kpi(k4, "Net result", f"${a['net']:+,.0f}",
        "realised" if a["realised"] else "marked (unrealised)",
        GREEN if a["net"] >= 0 else RED)

    st.markdown("### Attribution")
    rows = [dict(Component=x["label"], Amount=x["value"], Share=x["share"],
                 Source=x["source"]) for x in a["components"]]
    rows.append(dict(Component="── NET RESULT ──", Amount=a["net"],
                     Share=100.0, Source="sum of components"))
    rows.append(dict(Component="UNEXPLAINED RESIDUAL", Amount=a["residual"],
                     Share=np.nan, Source="must be 0.00"))
    adf = pd.DataFrame(rows)
    st.dataframe(adf.style.format({"Amount": "{:+,.0f}", "Share": "{:.0f}%"}, na_rep="—"),
                 use_container_width=True, hide_index=True)

    if abs(a["residual"]) > 0.01:
        st.error(f"**RECONCILIATION FAILED** — components miss the direct result by "
                 f"${a['residual']:,.2f}. A leg is missing from the model; treat the "
                 f"attribution as unreliable until it is fixed.")
    else:
        st.markdown('<span class="badge badge-green">RECONCILED · residual 0.00</span>',
                    unsafe_allow_html=True)

    # Which bet actually paid?
    market_share = sum(x["share"] for x in a["components"] if x["kind"] == "market")
    basis_share = sum(x["share"] for x in a["components"] if x["kind"] == "basis")
    cost_share = sum(x["share"] for x in a["components"] if x["kind"] == "cost")
    st.markdown(f"""**What this behaved like** — of the gross variation,
**{basis_share:.0f}%** came from the differential, **{market_share:.0f}%** from flat
price and carry, and **{cost_share:.0f}%** from costs. Shares are computed on
*absolute* contributions, so one large offsetting leg cannot flatter the read.""")

    if not a["realised"]:
        st.warning(f"**Unrealised.** The basis leg is marked against your own assessment "
                   f"of **{a['d1']:+,.4f}** (entered {cg.get('diff_mark_date', '—')}), not "
                   f"against a traded differential. Until the cargo is sold this is a "
                   f"mark, not a fact — update it in the Lifecycle tab.")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(a["components"]) + ["total"],
        x=[x["label"] for x in a["components"]] + ["NET"],
        y=[x["value"] for x in a["components"]] + [0],
        connector=dict(line=dict(color=BORDER)),
        increasing=dict(marker=dict(color=GREEN)),
        decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=AMBER))))
    fig.update_layout(title=f"{cid} — how the result was made ($)")
    st.plotly_chart(_styled(fig, 400), use_container_width=True)

    with st.expander("Marks used"):
        st.dataframe(pd.DataFrame([
            dict(Leg="Pricing benchmark", At_entry=a["B0"], Now=a["B1"],
                 Move=a["B1"] - a["B0"], Contract=cg.get("pricing_label")),
            dict(Leg="Differential", At_entry=a["d0"], Now=a["d1"],
                 Move=a["d1"] - a["d0"], Contract="USER MARK"),
            dict(Leg="Hedge", At_entry=a["H0"], Now=a["H1"],
                 Move=a["H1"] - a["H0"], Contract=cg.get("hedge_label")),
        ]).style.format({"At_entry": "{:,.4f}", "Now": "{:,.4f}", "Move": "{:+,.4f}"}),
            use_container_width=True, hide_index=True)


def _cargo_lifecycle_tab(marks: MarkBoard, cargos: List[dict]) -> None:
    if not cargos:
        st.info("No cargo booked yet.")
        return
    cid = st.selectbox("Cargo", [c["id"] for c in cargos], key="cg_life")
    cg = next(c for c in cargos if c["id"] == cid)
    idx = next(i for i, c in enumerate(cargos) if c["id"] == cid)

    st.markdown(f"**{cid}** · {cg['commodity']} · {cg.get('grade') or '—'} · "
                f"booked {cg.get('booked_date')}")
    l1, l2 = st.columns([1, 2])
    stage = l1.selectbox("Stage", CARGO_STAGES, index=CARGO_STAGES.index(cg["stage"]))
    l2.caption("A cargo is not one event. Attribution stays **marked** until the sale is "
               "entered below, then becomes **realised** — the same numbers, but facts "
               "instead of assessments.")

    st.markdown("##### Update the differential mark")
    m1, m2 = st.columns(2)
    new_mark = m1.number_input(f"Current differential ({COMMODITIES[cg['commodity']]['unit']})",
                              value=float(cg.get("diff_mark", cg["diff_buy"])),
                              step=0.01, format="%.4f")
    m2.caption(f"Last updated **{cg.get('diff_mark_date', '—')}**. This is your own "
               "assessment — the attribution tags every number derived from it.")

    st.markdown("##### Realise the sale")
    s1, s2, s3 = st.columns(3)
    sell_diff = s1.number_input("Sale differential", value=float(cg.get("diff_sell") or 0.0),
                                step=0.01, format="%.4f")
    live_px = cargo_pricing_price(cg, marks)
    sell_bench = s2.number_input("Benchmark at sale",
                                 value=float(cg.get("bench_sell") or (live_px or cg["bench_buy"])),
                                 step=0.01, format="%.4f")
    hedge_exit = s3.number_input("Hedge exit price (0 = mark live)",
                                 value=float(cg.get("hedge_exit") or 0.0),
                                 step=0.01, format="%.4f")
    f1, f2 = st.columns(2)
    freight_actual = f1.number_input("Freight actual (0 = use budget)",
                                     value=float(cg.get("freight_actual") or 0.0),
                                     step=0.01, format="%.4f")
    apply_sale = f2.toggle("Mark this cargo as sold", value=cg["stage"] in ("Sold", "Settled"))

    if st.button("Save changes", type="primary", use_container_width=True):
        cg = dict(cg)
        cg["stage"] = stage
        if new_mark != cg.get("diff_mark"):
            cg["diff_mark"] = float(new_mark)
            cg["diff_mark_date"] = date.today().isoformat()
        cg["freight_actual"] = float(freight_actual) if freight_actual else None
        cg["hedge_exit"] = float(hedge_exit) if hedge_exit else None
        if apply_sale:
            cg["diff_sell"] = float(sell_diff)
            cg["bench_sell"] = float(sell_bench)
            if cg["stage"] not in ("Sold", "Settled"):
                cg["stage"] = "Sold"
        else:
            cg["diff_sell"] = cg["bench_sell"] = None
        st.session_state.cargos = [*cargos[:idx], cg, *cargos[idx + 1:]]
        save_book()
        st.success("Cargo updated.")
        st.rerun()

    if cg.get("hedge_lots"):
        rc = roll_calendar(cargo_hedge_positions([cg]), marks)
        if rc and rc[0].get("Days") is not None:
            st.info(f"**Hedge roll deadline:** {rc[0]['Days']} days "
                    f"({rc[0]['Contract']}, expiry {rc[0]['Expiry']}). A hedge whose "
                    f"contract expires before the cargo prices out must be rolled — "
                    f"see the Roll calendar in the Blotter.")


# ────────────────────────────────────────────────────────────────────────────
#  Portfolio Risk — parametric VaR/ES on the correlated book, historical
#  simulation, dated stress episodes and curve-shape twists.
# ────────────────────────────────────────────────────────────────────────────


def page_risk(marks: MarkBoard) -> None:
    render_header(marks, "Portfolio Risk", "Delta-equivalent VaR/ES, historical replay, dated stress")
    positions, cargos = ensure_book()
    flat_legs, basis_legs = cargo_risk_legs(cargos, marks)
    book = positions + cargo_hedge_positions(cargos) + flat_legs
    if not book:
        st.info("Book is flat — book positions in the Trade Blotter first.")
        return

    view = st.radio("View", ["All", "Paper only", "Physical only"], horizontal=True,
                    help="One book, three lenses. 'Physical only' isolates the cargoes "
                         "and their hedges — the merchant's own P&L.")
    if view == "Paper only":
        book = [p for p in book if not p.get("cargo_id")]
        basis_legs = []
    elif view == "Physical only":
        book = [p for p in book if p.get("cargo_id")]
    if not book and not basis_legs:
        st.info("Nothing in this view.")
        return

    # Work on COPIES: the old page mutated the blotter's vols in session_state.
    pos = [dict(p) for p in book]

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
    res = portfolio_var(pos, marks, corr, conf, horizon, diversified, basis_legs)
    if not res["rows"] and not res.get("basis_rows"):
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

    if res["rows"]:
        df = pd.DataFrame(res["rows"])
        st.dataframe(df.style.format({"DeltaCash": "{:+,.0f}", "Vol": "{:.1f}%",
                                      "StandaloneVaR": "{:,.0f}", "StandaloneES": "{:,.0f}",
                                      "ComponentVaR": "{:+,.0f}", "PctOfVaR": "{:+.1f}%"}),
                     use_container_width=True, hide_index=True)
        st.caption("Options enter at Black-76 delta-cash and carry their UNDERLYING's vol — "
                   "gamma is not charged at this horizon (stated limitation, not an oversight).")

    if res.get("basis_rows"):
        st.markdown("### Basis risk (physical cargoes)")
        bk1, bk2 = st.columns(2)
        kpi(bk1, "Basis VaR contribution", f"${res['basis_var']:,.0f}",
            "differential risk left after the flat-price hedge", AMBER)
        kpi(bk2, "Share of total VaR",
            f"{res['basis_var'] / res['var'] * 100:.0f}%" if res["var"] else "—",
            "a hedged cargo is not a riskless cargo")
        st.dataframe(pd.DataFrame(res["basis_rows"]).style.format(
            {"DailySigma": "{:,.0f}", "StandaloneVaR": "{:,.0f}",
             "ComponentVaR": "{:+,.0f}", "PctOfVaR": "{:+.1f}%"}),
            use_container_width=True, hide_index=True)
        st.caption("Differentials are treated as **independent** risk factors: this desk "
                   "has no assessment history to estimate their correlation with flat "
                   "price, so it adds their variance rather than inventing a number. "
                   "Vols come from the cargo booking and are yours to set.")

    st.markdown("### Historical-simulation VaR")
    if basis_legs:
        st.caption("Basis risk is **not** in the historical figures below — replaying "
                   "differential history would need an assessment feed this desk does "
                   "not have. These numbers cover the flat-price book only.")
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

    st.markdown("### Curve-shape stress (twist)")
    st.caption("A parallel shock leaves a spread book untouched — and spread books are "
               "exactly what dies when the curve TWISTS. Each position is shocked at its "
               "own calendar tenor: front-month legs take the front shock, dated legs and "
               "options take the interpolated one.")
    t1, t2, t3 = st.columns(3)
    front_pct = t1.slider("Front shock", -40, 40, 10, 1, format="%d%%")
    back_pct  = t2.slider("Back shock", -40, 40, -5, 1, format="%d%%")
    pivot     = t3.slider("Pivot tenor (yrs)", 0.25, 3.0, 1.0, 0.25,
                          help="Shock interpolates linearly from front to back up to this "
                               "tenor, then stays flat further out the curve.")
    sh = stress_curve_shape(pos, marks, front_pct, back_pct, pivot)
    if sh.get("available"):
        label = ("steepening" if front_pct > back_pct else
                 "flattening" if front_pct < back_pct else "parallel")
        kpi(st.columns(3)[0], f"P&L on {label}", f"${sh['total']:+,.0f}",
            f"front {front_pct:+d}% → back {back_pct:+d}%",
            RED if sh["total"] < 0 else GREEN)
        st.dataframe(pd.DataFrame(sh["rows"]).style.format(
            {"Tenor": "{:.2f}", "Shock": "{:+.1f}%", "PnL": "{:+,.0f}"}),
            use_container_width=True, hide_index=True)
        curve_note = ("Front-led moves are usually physical and temporary; back-led moves "
                      "reprice long-run economics. Compare this number with the parallel "
                      "shock above — a large gap means your risk is in the SHAPE, not the level.")
        st.caption(curve_note)
    else:
        st.info("Curve-shape stress unavailable — no position could be marked.")


# ────────────────────────────────────────────────────────────────────────────
#  Macro Rates — the FRED backdrop: inflation, policy rates, the dollar and real
#  yields, read for what they do to commodities.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
#  About — the design contract, the exclusions and why they were excluded, the
#  changelog and the known limits.
# ────────────────────────────────────────────────────────────────────────────


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

### Revision 5 — the merchant layer

**🚢 Physical Cargo** — book a cargo (grade, location, volume in any trade unit, a
differential OR an outright price, Incoterm, freight, storage, financing) and the desk
**derives** the futures hedge. One model, no special cases: an outright purchase is
stored as the differential it implies against the pricing month.
**Attribution that reconciles** — the result splits into residual flat price, carry /
timing (hedge month ≠ pricing month), basis, and costs. Components are computed twice —
by part and directly from the legs — and the difference is displayed as **UNEXPLAINED
RESIDUAL**. It is zero by algebra, so any non-zero value is a real defect made visible.
Unrealised basis is tagged **USER MARK**, never presented as a fact.
**Derived hedges** — cargo hedge legs are generated from the cargo, never stored
separately, so they cannot drift out of sync with it. They appear tagged in the Blotter
and flow into Greeks, the roll calendar and risk.
**Basis risk in VaR** — a fully hedged cargo nets flat price to zero, and that is not
riskless. Differentials enter as independent variance factors (no assessment history
exists here to correlate them), with the assumption stated on the page. Portfolio Risk
gains an All / Paper only / Physical only lens.
**Not modelled, and said so** — no Platts/Argus/Baltic feed, no freight curve, no
quality optimisation, and a hedge against an *average* pricing window carries timing
risk the desk does not capture.

### Revision 4 — the curve through time

**Forward Curves → Evolution tab** — today's strip against any earlier date (one week
back by default, or your own dates), rebuilt from the SAME dated contracts, so there
are no roll artefacts. The move is decomposed into a **parallel shift** and a
**twist** (front minus back), and labelled: front-led moves are physical and usually
temporary, back-led moves reprice long-run economics. Months are matched on delivery,
never on position.
**Curve-shape stress** (Portfolio Risk) — front and back shocked by different amounts,
interpolated on calendar tenor, options fully revalued. A spread book survives every
parallel shock and dies on the twist; now that shows up in the numbers.
**Roll calendar** (Trade Blotter) — days to expiry for every front line, what rolling
costs at today's M1−M2, and an alert inside seven days. Dated lines show no roll cost;
options show days to expiry. An operational deadline, not a view.

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


# ════════════════════════════════════════════════════════════════════════════
#  MAIN — router
#  One guard around every page: an exception costs you that page, not the desk.
# ════════════════════════════════════════════════════════════════════════════


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
    "📒  Trade Blotter": page_blotter, "🚢  Physical Cargo": page_cargo, "⚠️  Portfolio Risk": page_risk,
    "🎲  Monte Carlo": page_mc, "🌍  Macro Rates": page_macro,
    "📡  Signal Scanner": page_signals, "📅  Event Calendar": page_events,
    "ℹ️  About": page_about,
}


def render_page(page: str, marks: MarkBoard) -> None:
    """Run one page inside a guard.

    Without this, any exception anywhere takes the whole app down: the sidebar
    disappears and every other page becomes unreachable behind a red traceback. A
    page is not the application — one broken page should cost you that page, not
    the desk. The failure is logged into Feed diagnostics so it is never silent."""
    try:
        ROUTES[page](marks)
    except Exception as e:                                    # noqa: BLE001
        # One entry, not two: the ring handler behind LOG already feeds the sidebar
        # diagnostics panel, so the message carries the detail itself.
        LOG.exception("PAGE ERROR %s: %s: %s", page, type(e).__name__, e)
        st.error(f"**This page hit an error and stopped.** The rest of the desk still "
                 f"works — pick another page in the sidebar.\n\n"
                 f"`{type(e).__name__}: {e}`")
        with st.expander("Details for debugging"):
            st.code(traceback.format_exc(), language="text")
        st.caption("If this persists, clear the caches from **🔧 Feed diagnostics** in "
                   "the sidebar — a stale cached frame from an earlier version is the "
                   "most common cause after a redeploy.")


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
    render_page(page, marks)


if __name__ == "__main__":
    main()
