"""
CODAP — Commodity Options Desk
==============================
A working desk for paper options traders: an option chain you can deal off, a
trade ticket, a book with cash Greeks and margin, and the volatility work that
decides whether any of it is worth doing.

VOLATILITY IS AUTOMATIC. You never type a volatility. An implied vol is an option
price turned inside out, so the desk goes and gets option prices, inverts them,
and maps the result onto every futures month. Sources are tried in order and the
winner is shown on every screen:

    1. IMPLIED     Listed option chains on the exchange-traded fund that tracks the
                   commodity (USO for WTI, GLD for gold, UNG for gas, ...). Each
                   expiry is inverted strike by strike from the bid/ask mid, a
                   SABR smile is fitted per expiry, and the ATM term structure is
                   interpolated in total variance. The fund's volatility is then
                   mapped onto each futures month through the Samuelson curve: the
                   ratio of that month's realised vol to the fund's, measured on
                   the same sessions and fitted across the strip.
    2. FORECAST    No listed options anywhere (coffee, cocoa, livestock)? A
                   GARCH(1,1) forecast on five years of front-month history gives
                   a vol term structure that mean-reverts to its long-run level,
                   mapped to each month by the same Samuelson curve. A forecast of
                   realised vol, labelled as one, priced with a flat smile.
    3. REGISTRY    No data at all: a constant in this file, flagged in red.

DESIGN CONTRACT — market data is fetched, dated and never invented; a dead feed
says NO MARKET DATA and the screen stands down. Every model input is either
measured or labelled. Independent checks run on demand and show their residuals.

TENOR IS REAL. Each listed month carries its futures last trading day AND its
option expiry, from the exchange rule for that family (gold options expire four
business days before the end of the month BEFORE delivery, grain options on a
Friday in the prior month, coffee and cocoa on the second Friday of the prior
month...). Weekends are handled, exchange holidays are not: dates can be a day
early. Pricing a December gold option to the December future's expiry would
overstate its life by a month.

SCREENS
    Chain        the strike ladder on the contract's increment grid, and the ticket
    Trade        one option or a structure, strikes by price / % / delta
    Book         saved to disk; every line repriced off ITS OWN contract and month
    Volatility   the automatic surface · realised-vol cones · gamma scalping
    Structures   cracks and crush · calendar spreads · Asians · barriers
    Market       the forward curve behind every price
    Checks       model validation, stated limits, and what was excluded

Run:    streamlit run codap.py
Tests:  pytest test_codap.py -q
Book:   ~/.codap/book.json   (override with the CODAP_BOOK environment variable)

by Adam El Gbouri
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import traceback
from calendar import monthrange
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq, least_squares, minimize
from scipy.signal import lfilter
from scipy.stats import norm

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:                                                 # pragma: no cover
    yf = None
    YF_AVAILABLE = False


def _st_version() -> Tuple[int, int]:
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])   # type: ignore
    except Exception:                                              # noqa: BLE001
        return (0, 0)


# Streamlit deprecated use_container_width in favour of width="stretch". Use the
# new form where it exists so the app does not spray deprecation warnings.
FULL = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}


# ══════════════════════════════════════════════════════════════════════════════
#  1. REGISTRY & CALENDAR
#  Static truth: palette, the contracts, their delivery cycles, expiry rules and
#  where their volatility comes from. Validated at import: a typo here would
#  misprice every option downstream, so it fails loudly at start-up.
# ══════════════════════════════════════════════════════════════════════════════

BG, PANEL, BORDER = "#0D1117", "#161B22", "#30363D"
TEXT, GRAY = "#E6EDF3", "#8B949E"
AMBER, BLUE, GREEN, RED, PURPLE, TEAL = (
    "#F0A500", "#58A6FF", "#3FB950", "#FF7B72", "#BC8CFF", "#39D0D8")
INK = "inherit"          # neutral text on the page itself: follows the light/dark theme

MONTH_CODES = list("FGHJKMNQUVXZ")
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Futures last trading day, per family. Weekdays only, exchange holidays ignored.
FUT_RULES = {
    "cl":       "3 business days before the 25th of the month before delivery",
    "bz":       "on the last business day of the second month before delivery",
    "ng":       "3 business days before the 1st of the delivery month",
    "eom_prev": "on the last business day of the month before delivery",
    "metals":   "on the third-last business day of the delivery month",
    "grains":   "on the business day before the 15th of the delivery month",
    "kc":       "8 business days before the last business day of the delivery month",
    "cc":       "11 business days before the last business day of the delivery month",
    "eom":      "on the last business day of the delivery month",
    "he":       "on the 10th business day of the delivery month",
}

# Option expiry, per family. This is the date that sets an option's life — for
# metals, grains and softs it is weeks before the future stops trading.
OPT_RULES = {
    "lag3":      "3 business days before the future expires",
    "lag1":      "1 business day before the future expires",
    "metals":    "on the 4th-last business day of the month before delivery (a day earlier if that is a Friday)",
    "grains":    "on the last Friday at least 2 business days before the end of the month before delivery",
    "sb":        "on the 15th of the month before delivery (next business day if not one)",
    "fri2_prev": "on the second Friday of the month before delivery",
    "fri1":      "on the first Friday of the delivery month",
    "same":      "on the same day as the future",
}

# Where the implied volatility comes from. Each entry is (ETF ticker, kind):
#   direct — the fund holds this commodity's futures, so its options price this
#            commodity's volatility;
#   cross  — a related market, used only when the contract has no listed fund of
#            its own; the level is rescaled by the two markets' realised ratio.
# An empty list means no listed options anywhere: the desk forecasts instead.
CONTRACTS: Dict[str, dict] = {
    # ── Energy ────────────────────────────────────────────────────────────────
    "WTI Crude (CL)": dict(
        sector="Energy", unit="$/bbl", yf="CL{M}{YY}.NYM", size=1_000,
        size_unit="bbl", divisor=1.0, months="FGHJKMNQUVXZ", n_liquid=18,
        fut_rule="cl", opt_rule="lag3", proxies=[("USO", "direct")],
        vol=0.32, storage=0.096, convenience=0.08, strike_inc=0.5),
    "Brent Crude (BZ)": dict(
        sector="Energy", unit="$/bbl", yf="BZ{M}{YY}.NYM", size=1_000,
        size_unit="bbl", divisor=1.0, months="FGHJKMNQUVXZ", n_liquid=18,
        fut_rule="bz", opt_rule="lag3", proxies=[("BNO", "direct"), ("USO", "cross")],
        vol=0.30, storage=0.096, convenience=0.07, strike_inc=0.5),
    "Henry Hub Gas (NG)": dict(
        sector="Energy", unit="$/MMBtu", yf="NG{M}{YY}.NYM", size=10_000,
        size_unit="MMBtu", divisor=1.0, months="FGHJKMNQUVXZ", n_liquid=18,
        fut_rule="ng", opt_rule="lag1", proxies=[("UNG", "direct")],
        vol=0.55, storage=0.120, convenience=0.10, strike_inc=0.05),
    "RBOB Gasoline (RB)": dict(
        sector="Energy", unit="$/gal", yf="RB{M}{YY}.NYM", size=42_000,
        size_unit="gal", divisor=1.0, months="FGHJKMNQUVXZ", n_liquid=12,
        fut_rule="eom_prev", opt_rule="lag3", proxies=[("UGA", "direct"), ("USO", "cross")],
        vol=0.36, storage=0.084, convenience=0.07, strike_inc=0.01),
    "ULSD Heating Oil (HO)": dict(
        sector="Energy", unit="$/gal", yf="HO{M}{YY}.NYM", size=42_000,
        size_unit="gal", divisor=1.0, months="FGHJKMNQUVXZ", n_liquid=12,
        fut_rule="eom_prev", opt_rule="lag3", proxies=[("USO", "cross")],
        vol=0.34, storage=0.084, convenience=0.07, strike_inc=0.01),
    # ── Metals ────────────────────────────────────────────────────────────────
    "Gold (GC)": dict(
        sector="Metals", unit="$/troy oz", yf="GC{M}{YY}.CMX", size=100,
        size_unit="troy oz", divisor=1.0, months="GJMQVZ", n_liquid=8,
        fut_rule="metals", opt_rule="metals", proxies=[("GLD", "direct")],
        vol=0.15, storage=0.024, convenience=0.005, strike_inc=5.0),
    "Silver (SI)": dict(
        sector="Metals", unit="$/troy oz", yf="SI{M}{YY}.CMX", size=5_000,
        size_unit="troy oz", divisor=1.0, months="HKNUZ", n_liquid=6,
        fut_rule="metals", opt_rule="metals", proxies=[("SLV", "direct")],
        vol=0.28, storage=0.036, convenience=0.010, strike_inc=0.25),
    "Copper (HG)": dict(
        sector="Metals", unit="$/lb", yf="HG{M}{YY}.CMX", size=25_000,
        size_unit="lb", divisor=1.0, months="HKNUZ", n_liquid=8,
        fut_rule="metals", opt_rule="metals", proxies=[("CPER", "direct")],
        vol=0.22, storage=0.048, convenience=0.030, strike_inc=0.05),
    "Platinum (PL)": dict(
        sector="Metals", unit="$/troy oz", yf="PL{M}{YY}.NYM", size=50,
        size_unit="troy oz", divisor=1.0, months="FJNV", n_liquid=6,
        fut_rule="metals", opt_rule="metals", proxies=[("PPLT", "direct")],
        vol=0.20, storage=0.030, convenience=0.015, strike_inc=5.0),
    "Palladium (PA)": dict(
        sector="Metals", unit="$/troy oz", yf="PA{M}{YY}.NYM", size=100,
        size_unit="troy oz", divisor=1.0, months="HMUZ", n_liquid=6,
        fut_rule="metals", opt_rule="metals", proxies=[("PALL", "direct")],
        vol=0.30, storage=0.030, convenience=0.020, strike_inc=5.0),
    # ── Grains & oilseeds (quoted in CENTS — divisor 100) ─────────────────────
    "Corn (ZC)": dict(
        sector="Grains", unit="c/bu", yf="ZC{M}{YY}.CBT", size=5_000,
        size_unit="bu", divisor=100.0, months="HKNUZ", n_liquid=8,
        fut_rule="grains", opt_rule="grains", proxies=[("CORN", "direct")],
        vol=0.25, storage=0.060, convenience=0.04, strike_inc=10.0),
    "Wheat CBOT (ZW)": dict(
        sector="Grains", unit="c/bu", yf="ZW{M}{YY}.CBT", size=5_000,
        size_unit="bu", divisor=100.0, months="HKNUZ", n_liquid=8,
        fut_rule="grains", opt_rule="grains", proxies=[("WEAT", "direct")],
        vol=0.28, storage=0.060, convenience=0.04, strike_inc=10.0),
    "Soybeans (ZS)": dict(
        sector="Grains", unit="c/bu", yf="ZS{M}{YY}.CBT", size=5_000,
        size_unit="bu", divisor=100.0, months="FHKNQUX", n_liquid=8,
        fut_rule="grains", opt_rule="grains", proxies=[("SOYB", "direct")],
        vol=0.23, storage=0.060, convenience=0.05, strike_inc=20.0),
    "Soybean Meal (ZM)": dict(
        sector="Grains", unit="$/short ton", yf="ZM{M}{YY}.CBT", size=100,
        size_unit="short ton", divisor=1.0, months="FHKNQUVZ", n_liquid=8,
        fut_rule="grains", opt_rule="grains", proxies=[("SOYB", "cross")],
        vol=0.24, storage=0.060, convenience=0.05, strike_inc=5.0),
    "Soybean Oil (ZL)": dict(
        sector="Grains", unit="c/lb", yf="ZL{M}{YY}.CBT", size=60_000,
        size_unit="lb", divisor=100.0, months="FHKNQUVZ", n_liquid=8,
        fut_rule="grains", opt_rule="grains", proxies=[("SOYB", "cross")],
        vol=0.26, storage=0.060, convenience=0.05, strike_inc=0.5),
    # ── Softs & livestock ────────────────────────────────────────────────────
    "Sugar #11 (SB)": dict(
        sector="Softs", unit="c/lb", yf="SB{M}{YY}.NYB", size=112_000,
        size_unit="lb", divisor=100.0, months="HKNV", n_liquid=6,
        fut_rule="eom_prev", opt_rule="sb", proxies=[("CANE", "direct")],
        vol=0.30, storage=0.048, convenience=0.04, strike_inc=0.25),
    "Arabica Coffee (KC)": dict(
        sector="Softs", unit="c/lb", yf="KC{M}{YY}.NYB", size=37_500,
        size_unit="lb", divisor=100.0, months="HKNUZ", n_liquid=6,
        fut_rule="kc", opt_rule="fri2_prev", proxies=[],
        vol=0.35, storage=0.048, convenience=0.05, strike_inc=2.5),
    "Cocoa (CC)": dict(
        sector="Softs", unit="$/mt", yf="CC{M}{YY}.NYB", size=10,
        size_unit="mt", divisor=1.0, months="HKNUZ", n_liquid=6,
        fut_rule="cc", opt_rule="fri2_prev", proxies=[],
        vol=0.32, storage=0.048, convenience=0.04, strike_inc=50.0),
    "Live Cattle (LE)": dict(
        sector="Livestock", unit="c/lb", yf="LE{M}{YY}.CME", size=40_000,
        size_unit="lb", divisor=100.0, months="GJMQVZ", n_liquid=8,
        fut_rule="eom", opt_rule="fri1", proxies=[],
        vol=0.18, storage=0.036, convenience=0.03, strike_inc=1.0),
    "Lean Hogs (HE)": dict(
        sector="Livestock", unit="c/lb", yf="HE{M}{YY}.CME", size=40_000,
        size_unit="lb", divisor=100.0, months="GJKMNQVZ", n_liquid=6,
        fut_rule="he", opt_rule="same", proxies=[],
        vol=0.25, storage=0.036, convenience=0.03, strike_inc=1.0),
}

# What was deliberately dropped, and why — stated rather than left as broken
# entries that would silently fall back to invented prices.
EXCLUSIONS = {
    "LME Copper / Aluminium / Zinc / Nickel / Lead / Tin":
        "LME contracts do not price off dated month codes the way these tickers "
        "assumed, and no free feed returns them.",
    "Capesize / Panamax freight (BCI, BPI)":
        "Baltic Exchange assessments are licensed data. No feed, no curve.",
    "EU / UK Carbon (EUA, UKA)":
        "No free ticker resolves for ICE carbon futures.",
    "Coal API2, Uranium UxC, Gasoil ICE, Jet Kero CIF NWE":
        "Platts and Argus assessments are licensed; there is no free feed.",
    "Fixed-for-floating swaps":
        "A commodity swap is corporate treasury, not a paper options trader's "
        "instrument; the option chain is the screen this audience works on.",
    "Hand-typed volatility":
        "Removed. Every volatility is now implied from listed option prices or, "
        "where none exist, forecast from price history — and labelled with which.",
    "Dutch TTF Natural Gas":
        "No free dated-contract feed; a spread that needs it cannot be priced here.",
}

_REQUIRED_KEYS = {"sector", "unit", "yf", "size", "size_unit", "divisor",
                  "months", "n_liquid", "fut_rule", "opt_rule", "proxies", "vol",
                  "storage", "convenience", "strike_inc"}


def _validate_registry(reg: Dict[str, dict]) -> None:
    """Fail at import on a malformed contract."""
    assert reg, "contract registry is empty"
    for name, c in reg.items():
        missing = _REQUIRED_KEYS - set(c)
        assert not missing, f"{name}: missing keys {sorted(missing)}"
        unknown = set(c) - _REQUIRED_KEYS
        assert not unknown, f"{name}: unknown keys {sorted(unknown)}"
        assert c["fut_rule"] in FUT_RULES, f"{name}: bad futures rule {c['fut_rule']!r}"
        assert c["opt_rule"] in OPT_RULES, f"{name}: bad option rule {c['opt_rule']!r}"
        assert set(c["months"]) <= set(MONTH_CODES), f"{name}: bad month codes"
        assert c["months"], f"{name}: no delivery months"
        assert c["divisor"] in (1.0, 100.0), f"{name}: divisor must be 1 or 100"
        assert c["size"] > 0 and c["n_liquid"] > 0, f"{name}: bad size/n_liquid"
        assert 0 < c["vol"] < 3, f"{name}: implausible fallback vol"
        assert 0 <= c["storage"] < 1 and 0 <= c["convenience"] < 1, f"{name}: bad carry"
        assert c["strike_inc"] > 0, f"{name}: bad strike increment"
        for p in c["proxies"]:
            assert isinstance(p, tuple) and len(p) == 2 and p[1] in ("direct", "cross"), \
                f"{name}: proxies must be (ticker, 'direct'|'cross')"


_validate_registry(CONTRACTS)

SECTORS = sorted({c["sector"] for c in CONTRACTS.values()})


def contracts_in(sector: str) -> List[str]:
    return [n for n, c in CONTRACTS.items() if c["sector"] == sector]


def price_multiplier(name: str) -> float:
    """Cash P&L per 1.0 move in the QUOTED price, per lot. The cents divisor is
    applied here and nowhere else, so a c/bu price can never inflate a dollar
    figure a hundredfold."""
    c = CONTRACTS[name]
    return c["size"] / c["divisor"]


def snap_strike(name: str, K: float) -> float:
    """Round a strike onto the contract's listed grid. The increments are
    INDICATIVE registry defaults: exchanges vary them by expiry and moneyness."""
    inc = CONTRACTS[name]["strike_inc"]
    return round(round(float(K) / inc) * inc, 10)


def notional_per_lot(name: str, price: float) -> float:
    c = CONTRACTS[name]
    return price / c["divisor"] * c["size"]


# ── Business-day calendar (weekdays; exchange holidays are not modelled) ─────
def _eom(y: int, m: int) -> date:
    return date(y, m, monthrange(y, m)[1])


def _shift_month(y: int, m: int, k: int) -> Tuple[int, int]:
    idx = (y * 12 + (m - 1)) + k
    return idx // 12, idx % 12 + 1


def _is_bday(d: date) -> bool:
    return d.weekday() < 5


def _prev_bday(d: date) -> date:
    """d itself if a business day, else the business day before it."""
    while not _is_bday(d):
        d -= timedelta(days=1)
    return d


def _next_bday(d: date) -> date:
    while not _is_bday(d):
        d += timedelta(days=1)
    return d


def _bdays_before(d: date, n: int) -> date:
    """The business day n business days strictly before d."""
    for _ in range(n):
        d -= timedelta(days=1)
        while not _is_bday(d):
            d -= timedelta(days=1)
    return d


def _last_bday(y: int, m: int) -> date:
    return _prev_bday(_eom(y, m))


def _nth_bday(y: int, m: int, n: int) -> date:
    d = _next_bday(date(y, m, 1))
    for _ in range(n - 1):
        d = _next_bday(d + timedelta(days=1))
    return d


def _nth_weekday(y: int, m: int, weekday: int, n: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def futures_expiry(rule: str, dy: int, dm: int) -> date:
    """Last trading day of the future for delivery month (dy, dm)."""
    if rule == "cl":
        y, m = _shift_month(dy, dm, -1)
        return _bdays_before(_prev_bday(date(y, m, 25)), 3)
    if rule == "bz":
        y, m = _shift_month(dy, dm, -2)
        return _last_bday(y, m)
    if rule == "ng":
        return _bdays_before(date(dy, dm, 1), 3)
    if rule == "eom_prev":
        y, m = _shift_month(dy, dm, -1)
        return _last_bday(y, m)
    if rule == "metals":
        return _bdays_before(_last_bday(dy, dm), 2)
    if rule == "grains":
        return _bdays_before(date(dy, dm, 15), 1)
    if rule == "kc":
        return _bdays_before(_last_bday(dy, dm), 8)
    if rule == "cc":
        return _bdays_before(_last_bday(dy, dm), 11)
    if rule == "he":
        return _nth_bday(dy, dm, 10)
    return _last_bday(dy, dm)                                    # "eom"


def option_expiry(opt_rule: str, fut_rule: str, dy: int, dm: int) -> date:
    """Expiry of the standard option on the future for delivery month (dy, dm).
    Never later than the future itself."""
    fut = futures_expiry(fut_rule, dy, dm)
    py, pm = _shift_month(dy, dm, -1)
    if opt_rule == "lag3":
        d = _bdays_before(fut, 3)
    elif opt_rule == "lag1":
        d = _bdays_before(fut, 1)
    elif opt_rule == "metals":
        d = _bdays_before(_last_bday(py, pm), 3)                  # 4th-last
        if d.weekday() == 4:
            d = _bdays_before(d, 1)
    elif opt_rule == "grains":
        d = _bdays_before(_last_bday(py, pm), 2)
        while d.weekday() != 4:
            d -= timedelta(days=1)
    elif opt_rule == "sb":
        d = _next_bday(date(py, pm, 15))
    elif opt_rule == "fri2_prev":
        d = _nth_weekday(py, pm, 4, 2)
    elif opt_rule == "fri1":
        d = _nth_weekday(dy, dm, 4, 1)
    else:                                                          # "same"
        d = fut
    return min(d, fut)


def year_frac(d: date, today: date) -> float:
    return (d - today).days / 365.25


def strip_specs(name: str, today: Optional[date] = None) -> List[dict]:
    """The live strip: every listed delivery month with its ticker, the future's
    last trading day, the option's expiry, and both CALENDAR tenors.

    `T` is the future's year-fraction to expiry; `T_opt` the option's. A month
    whose option has already expired is still a live future (it belongs on the
    curve) but is flagged `option_live=False` so nothing tries to price one.
    """
    today = today or date.today()
    c = CONTRACTS[name]
    out: List[dict] = []
    k = 0
    while len(out) < c["n_liquid"] and k < c["n_liquid"] * 6 + 24:
        y, m = _shift_month(today.year, today.month, k)
        k += 1
        code = MONTH_CODES[m - 1]
        if code not in c["months"]:
            continue
        fexp = futures_expiry(c["fut_rule"], y, m)
        if fexp <= today:
            continue
        oexp = option_expiry(c["opt_rule"], c["fut_rule"], y, m)
        out.append(dict(
            ticker=c["yf"].replace("{M}", code).replace("{YY}", f"{y % 100:02d}"),
            label=f"{MONTH_NAMES[m - 1]}-{y}",
            delivery=f"{y}-{m:02d}",
            expiry=fexp,
            opt_expiry=oexp,
            T=year_frac(fexp, today),
            T_opt=year_frac(oexp, today),
            option_live=oexp > today,
            seq=len(out) + 1,
        ))
    return out


# ── Multi-leg structures. Every leg carries its own conversion to the common
#    quote unit, so a 3-2-1 crack is the whole 3-2-1 crack.
STRUCTURES: Dict[str, dict] = {
    "3-2-1 Crack (WTI / RBOB + HO)": dict(
        unit="$/bbl", basis="per barrel of crude input",
        short=[("WTI Crude (CL)", 3.0, 1.0)],
        long=[("RBOB Gasoline (RB)", 2.0, 42.0), ("ULSD Heating Oil (HO)", 1.0, 42.0)],
        divide_by=3.0, rho=0.85,
        note="3 bbl crude in, 2 bbl gasoline + 1 bbl distillate out — the US "
             "refiner's benchmark gross margin."),
    "2-1-1 Crack (WTI / RBOB + HO)": dict(
        unit="$/bbl", basis="per barrel of crude input",
        short=[("WTI Crude (CL)", 2.0, 1.0)],
        long=[("RBOB Gasoline (RB)", 1.0, 42.0), ("ULSD Heating Oil (HO)", 1.0, 42.0)],
        divide_by=2.0, rho=0.85,
        note="An even gasoline/distillate yield — closer to a European or winter "
             "configuration than the 3-2-1."),
    "Gasoline Crack (WTI / RBOB)": dict(
        unit="$/bbl", basis="per barrel",
        short=[("WTI Crude (CL)", 1.0, 1.0)],
        long=[("RBOB Gasoline (RB)", 1.0, 42.0)],
        divide_by=1.0, rho=0.88,
        note="The single-product gasoline margin — the cleanest crack to trade."),
    "Heat Crack (WTI / ULSD)": dict(
        unit="$/bbl", basis="per barrel",
        short=[("WTI Crude (CL)", 1.0, 1.0)],
        long=[("ULSD Heating Oil (HO)", 1.0, 42.0)],
        divide_by=1.0, rho=0.87,
        note="The distillate margin: diesel, heating oil and the jet complex."),
    "Brent Gasoil-style Crack (BZ / ULSD)": dict(
        unit="$/bbl", basis="per barrel",
        short=[("Brent Crude (BZ)", 1.0, 1.0)],
        long=[("ULSD Heating Oil (HO)", 1.0, 42.0)],
        divide_by=1.0, rho=0.86,
        note="ULSD against Brent — a proxy for the European distillate margin. "
             "A proxy, stated as one: ICE gasoil itself has no free feed here."),
    "Board Crush (Beans / Meal + Oil)": dict(
        unit="$/bu", basis="per bushel of beans",
        short=[("Soybeans (ZS)", 1.0, 0.01)],
        long=[("Soybean Meal (ZM)", 1.0, 44.0 / 2000.0),
              ("Soybean Oil (ZL)", 1.0, 11.0 * 0.01)],
        divide_by=1.0, rho=0.80,
        note="One 60 lb bushel yields ~44 lb of meal and ~11 lb of oil. The "
             "processor's margin, and self-stabilising: at a thin crush, plants "
             "slow down and the margin recovers."),
}


# ══════════════════════════════════════════════════════════════════════════════
#  2. MARKET DATA
#  One grouped download per commodity, dated marks, and no fabrication anywhere.
#  A dead feed produces an empty curve and a message; a model curve exists only
#  on explicit request and is labelled on every screen it touches.
# ══════════════════════════════════════════════════════════════════════════════

LOG = logging.getLogger("codap")
LOG.setLevel(logging.INFO)
FEED_LOG: deque = deque(maxlen=120)


class _RingHandler(logging.Handler):
    def emit(self, record):
        FEED_LOG.append(f"{datetime.now():%H:%M:%S}  {record.levelname:<7} "
                        f"{record.getMessage()}")


if not any(isinstance(h, _RingHandler) for h in LOG.handlers):
    LOG.addHandler(_RingHandler())

STALE_DAYS = 4            # header turns amber when the newest settle is older
DEAD_MARK_DAYS = 10       # a month whose last print lags the strip by more is dropped
RATE_FALLBACK = 0.04      # used only if the T-bill feed is down — and said so


def _closes_frame(raw, tickers: Sequence[str]) -> pd.DataFrame:
    """Normalise whatever yfinance returned into one Close column per ticker,
    with a tz-naive daily index. yfinance has changed this layout several times
    (flat vs MultiIndex, field-first vs ticker-first), so every shape is handled
    here and nowhere else."""
    if raw is None or getattr(raw, "empty", True):
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(raw.columns.get_level_values(0))
        lv1 = set(raw.columns.get_level_values(1))
        if "Close" in lv0:
            closes = raw["Close"]
        elif "Close" in lv1:
            closes = raw.xs("Close", axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        closes = raw[["Close"]].copy()
        closes.columns = list(tickers[:1])
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    closes = closes.copy()
    idx = pd.to_datetime(closes.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    closes.index = idx.normalize()
    closes = closes.apply(pd.to_numeric, errors="coerce")
    return closes[~closes.index.duplicated(keep="last")].sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_strip_closes(name: str) -> pd.DataFrame:
    """Six months of daily settles for every listed month — ONE grouped request.

    The last row gives the curve; the history gives each month's own realised
    volatility, which is what maps a fund's implied vol onto a dated future."""
    specs = strip_specs(name)
    if not YF_AVAILABLE or not specs:
        LOG.warning("no yfinance / no listed contracts for %s", name)
        return pd.DataFrame()
    tickers = [s["ticker"] for s in specs]
    try:
        raw = yf.download(tickers, period="6mo", auto_adjust=True, progress=False,
                          group_by="column", threads=True)
    except Exception as e:                                        # noqa: BLE001
        LOG.warning("download failed for %s: %s", name, e)
        return pd.DataFrame()
    closes = _closes_frame(raw, tickers)
    if closes.empty:
        LOG.warning("empty payload for %s", name)
    else:
        LOG.info("%s: %d/%d listed months returned settles", name,
                 int(closes.notna().any().sum()), len(tickers))
    return closes


def curve_from_closes(name: str, closes: pd.DataFrame,
                      today: Optional[date] = None) -> pd.DataFrame:
    """Build the dated forward curve from the strip's closes. Pure — testable.

    Every row keeps the settle date it actually printed. A month whose last print
    is more than DEAD_MARK_DAYS older than the rest of the strip is dropped: a
    stale deferred settle presented as today's is worse than a gap."""
    if closes is None or closes.empty:
        return pd.DataFrame()
    rows = []
    for s in strip_specs(name, today):
        if s["ticker"] not in closes.columns:
            continue
        ser = closes[s["ticker"]].dropna()
        ser = ser[ser > 0]
        if ser.empty:
            continue
        rows.append({**s, "price": float(ser.iloc[-1]), "asof": ser.index[-1].date()})
    if not rows:
        return pd.DataFrame()
    newest = max(r["asof"] for r in rows)
    kept = [r for r in rows if (newest - r["asof"]).days <= DEAD_MARK_DAYS]
    for r in rows:
        if r not in kept:
            LOG.info("%s %s dropped: last print %s lags the strip", name, r["label"], r["asof"])
    if len(kept) < 2:
        LOG.warning("%s: only %d dated contract(s) returned a usable settle", name, len(kept))
        return pd.DataFrame()
    df = pd.DataFrame(kept).sort_values("T").reset_index(drop=True)
    df["seq"] = range(1, len(df) + 1)
    return df


def fetch_curve(name: str) -> pd.DataFrame:
    return curve_from_closes(name, fetch_strip_closes(name))


def curve_is_stale(df: pd.DataFrame) -> bool:
    if df.empty or "asof" not in df:
        return False
    marks = [d for d in df["asof"] if d is not None]
    return bool(marks) and (date.today() - max(marks)).days > STALE_DAYS


def model_curve(name: str, spot: float, r: float, storage: float,
                convenience: float, n: int = 12) -> pd.DataFrame:
    """A cost-of-carry curve from a spot price YOU supply:
    F(T) = S · exp((r + storage − convenience) · T). Labelled MODEL CURVE on every
    screen. Tenors are the real listed ones, so the shape still respects the
    contract's delivery cycle and option calendar."""
    specs = strip_specs(name)[:n]
    if not specs or spot <= 0:
        return pd.DataFrame()
    rows = [{**s, "price": float(spot * math.exp((r + storage - convenience) * s["T"])),
             "asof": None} for s in specs]
    return pd.DataFrame(rows)


def curve_stats(df: pd.DataFrame) -> dict:
    """Front, back, slope and structure label for a curve of any length."""
    if df.empty:
        return dict(available=False)
    f1, fn = float(df["price"].iloc[0]), float(df["price"].iloc[-1])
    t1, tn = float(df["T"].iloc[0]), float(df["T"].iloc[-1])
    dT = max(tn - t1, 1e-9)
    slope_ann = (fn - f1) / f1 / dT * 100
    if abs(fn - f1) / f1 < 0.005:
        structure = "FLAT"
    elif fn > f1:
        structure = "CONTANGO"
    else:
        structure = "BACKWARDATION"
    return dict(available=True, f1=f1, fn=fn, t1=t1, tn=tn, slope_ann=slope_ann,
                structure=structure, front_label=str(df["label"].iloc[0]),
                back_label=str(df["label"].iloc[-1]))


def forward_at(df: pd.DataFrame, T: float) -> Optional[float]:
    """Forward at an arbitrary futures tenor, log-linear on calendar T, flat
    beyond the ends of the strip."""
    if df is None or df.empty:
        return None
    Ts = df["T"].to_numpy(dtype=float)
    Ps = df["price"].to_numpy(dtype=float)
    if np.any(Ps <= 0):
        return float(np.interp(T, Ts, Ps))
    return float(np.exp(np.interp(T, Ts, np.log(Ps))))


def implied_carry(df: pd.DataFrame) -> pd.DataFrame:
    """Annualised carry from the front to each listed month, on calendar tenor.
    Positive = contango (a long pays to roll); negative = backwardation."""
    if df.empty or len(df) < 2:
        return pd.DataFrame()
    f1, t1 = float(df["price"].iloc[0]), float(df["T"].iloc[0])
    rows = []
    for r_ in df.itertuples():
        dT = float(r_.T) - t1
        rows.append(dict(
            label=r_.label, T=float(r_.T), price=float(r_.price),
            spread=float(r_.price) - f1,
            carry_ann=((float(r_.price) - f1) / f1 / dT * 100) if dT > 1e-9 else np.nan))
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rate() -> dict:
    """Discount rate from the 13-week T-bill (^IRX), converted from its bank-
    discount quote to a continuously compounded rate. Falls back to a stated
    constant if the feed is down."""
    fallback = dict(rate=RATE_FALLBACK, source="fallback", asof=None,
                    detail=f"T-bill feed unavailable — {RATE_FALLBACK:.2%} default")
    if not YF_AVAILABLE:
        return fallback
    try:
        raw = yf.download("^IRX", period="1mo", auto_adjust=False, progress=False)
    except Exception as e:                                         # noqa: BLE001
        LOG.warning("rate download failed: %s", e)
        return fallback
    closes = _closes_frame(raw, ["^IRX"])
    if closes.empty:
        return fallback
    ser = closes.iloc[:, 0].dropna()
    if ser.empty:
        return fallback
    d = float(ser.iloc[-1]) / 100.0
    if not (0.0 <= d < 0.25):
        return fallback
    r = -math.log(1.0 - d * 91.0 / 360.0) / (91.0 / 365.0)
    return dict(rate=r, source="T-bill", asof=ser.index[-1].date(),
                detail=f"13-week T-bill {d:.2%} discount → {r:.2%} continuous")


CONTINUOUS_TICKER = {
    "WTI Crude (CL)": "CL=F", "Brent Crude (BZ)": "BZ=F", "Henry Hub Gas (NG)": "NG=F",
    "RBOB Gasoline (RB)": "RB=F", "ULSD Heating Oil (HO)": "HO=F",
    "Gold (GC)": "GC=F", "Silver (SI)": "SI=F", "Copper (HG)": "HG=F",
    "Platinum (PL)": "PL=F", "Palladium (PA)": "PA=F",
    "Corn (ZC)": "ZC=F", "Wheat CBOT (ZW)": "ZW=F", "Soybeans (ZS)": "ZS=F",
    "Soybean Meal (ZM)": "ZM=F", "Soybean Oil (ZL)": "ZL=F",
    "Sugar #11 (SB)": "SB=F", "Arabica Coffee (KC)": "KC=F", "Cocoa (CC)": "CC=F",
    "Live Cattle (LE)": "LE=F", "Lean Hogs (HE)": "HE=F",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(name: str, period: str = "5y") -> pd.Series:
    """Continuous front-month closes, for volatility work only. NOT roll-adjusted:
    each roll injects one artificial jump, which inflates realised vol slightly.
    The GARCH fit clips gross outliers; the cone states the caveat."""
    tk = CONTINUOUS_TICKER.get(name)
    if not YF_AVAILABLE or not tk:
        return pd.Series(dtype=float)
    try:
        raw = yf.download(tk, period=period, auto_adjust=True, progress=False)
    except Exception as e:                                         # noqa: BLE001
        LOG.warning("history failed for %s: %s", name, e)
        return pd.Series(dtype=float)
    closes = _closes_frame(raw, [tk])
    if closes.empty:
        LOG.warning("no history for %s (%s)", name, tk)
        return pd.Series(dtype=float)
    ser = closes.iloc[:, 0].dropna()
    return ser[ser > 0]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_etf_history(etf: str, period: str = "1y") -> pd.Series:
    """Daily closes of an options proxy fund — the reference series against which
    each futures month's realised volatility is measured."""
    if not YF_AVAILABLE:
        return pd.Series(dtype=float)
    try:
        raw = yf.download(etf, period=period, auto_adjust=True, progress=False)
    except Exception as e:                                         # noqa: BLE001
        LOG.warning("history failed for %s: %s", etf, e)
        return pd.Series(dtype=float)
    closes = _closes_frame(raw, [etf])
    if closes.empty:
        return pd.Series(dtype=float)
    ser = closes.iloc[:, 0].dropna()
    return ser[ser > 0]


def last_known_spot(name: str) -> Optional[float]:
    """A starting point for the model curve's spot: the live front month if it
    exists, else the last continuous close. None if neither is reachable."""
    cur = fetch_curve(name)
    if not cur.empty:
        return float(cur["price"].iloc[0])
    hist = fetch_history(name, "1mo")
    if not hist.empty:
        return float(hist.iloc[-1])
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  3. PRICING MODELS
#  Pure maths on forwards. No Streamlit, no I/O, no globals — every function here
#  is exercised offline by the test suite, and every approximation states what it
#  approximates.
# ══════════════════════════════════════════════════════════════════════════════

SQRT_2PI = math.sqrt(2 * math.pi)


class Black76:
    """European option on a futures/forward price.

    C = e^(-rT)[F·N(d1) - K·N(d2)],  d1 = (ln(F/K) + σ²T/2)/(σ√T),  d2 = d1 - σ√T

    Conventions, stated because they are where pricers disagree: vega is per ONE
    volatility POINT (a move from 30% to 31%), theta is per CALENDAR day, rho is
    per one percentage point of rate. T ≤ 0 or σ ≤ 0 returns intrinsic value with
    a binary delta rather than raising — an expired option is worth its intrinsic,
    not an exception, and the blotter of any real desk depends on that.
    """

    def __init__(self, F, K, T, r, sigma, option_type="call"):
        self.F = float(F)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.opt = str(option_type).lower()
        if self.opt not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.F <= 0 or self.K <= 0:
            raise ValueError("F and K must be positive")

    # ── internals ────────────────────────────────────────────────────────────
    @property
    def _degenerate(self) -> bool:
        return self.T <= 0 or self.sigma <= 0

    def _d1_d2(self) -> Tuple[float, float]:
        v = self.sigma * math.sqrt(self.T)
        d1 = (math.log(self.F / self.K) + 0.5 * self.sigma ** 2 * self.T) / v
        return d1, d1 - v

    def _intrinsic(self) -> float:
        return max(self.F - self.K, 0.0) if self.opt == "call" else max(self.K - self.F, 0.0)

    # ── value and Greeks ─────────────────────────────────────────────────────
    def price(self) -> float:
        if self._degenerate:
            return math.exp(-self.r * max(self.T, 0.0)) * self._intrinsic()
        d1, d2 = self._d1_d2()
        disc = math.exp(-self.r * self.T)
        if self.opt == "call":
            return disc * (self.F * norm.cdf(d1) - self.K * norm.cdf(d2))
        return disc * (self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1))

    def delta(self) -> float:
        """∂price/∂F. Discounted, because the option settles at T."""
        if self._degenerate:
            itm = (self.F > self.K) if self.opt == "call" else (self.F < self.K)
            sign = 1.0 if self.opt == "call" else -1.0
            return sign * (1.0 if itm else 0.0)
        d1, _ = self._d1_d2()
        disc = math.exp(-self.r * self.T)
        return disc * (norm.cdf(d1) if self.opt == "call" else norm.cdf(d1) - 1.0)

    def gamma(self) -> float:
        if self._degenerate:
            return 0.0
        d1, _ = self._d1_d2()
        return (math.exp(-self.r * self.T) * norm.pdf(d1)
                / (self.F * self.sigma * math.sqrt(self.T)))

    def vega(self) -> float:
        """Per ONE volatility point (0.30 → 0.31)."""
        if self._degenerate:
            return 0.0
        d1, _ = self._d1_d2()
        return (math.exp(-self.r * self.T) * self.F * norm.pdf(d1)
                * math.sqrt(self.T)) / 100.0

    def theta(self) -> float:
        """Value lost per CALENDAR day, everything else held fixed.

        θ = -∂C/∂T = r·e^(-rT)[F·N(d1) - K·N(d2)] - e^(-rT)·F·φ(d1)·σ/(2√T)

        Note the PLUS sign on the discounting term. Subtracting it — the previous
        behaviour — overstates the daily decay: at F=K=80, T=0.5, σ=30%, r=5% it
        reports -0.0189/day against a true -0.0171, an 11% exaggeration that grows
        with the rate and the tenor. Verified here against a finite difference of
        the price itself, which is what the test suite checks.
        """
        if self._degenerate:
            return 0.0
        d1, d2 = self._d1_d2()
        disc = math.exp(-self.r * self.T)
        decay = -disc * self.F * norm.pdf(d1) * self.sigma / (2 * math.sqrt(self.T))
        undisc = (self.F * norm.cdf(d1) - self.K * norm.cdf(d2)) if self.opt == "call" \
            else (self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1))
        return (decay + self.r * disc * undisc) / 365.0

    def rho(self) -> float:
        """Per one percentage point of rate. On a FUTURES option the forward does
        not move with r, so rho is pure discounting: ρ = -T · price."""
        return -self.T * self.price() / 100.0

    def greeks(self) -> dict:
        return {k: float(getattr(self, k)()) for k in
                ("price", "delta", "gamma", "vega", "theta", "rho")}

    # ── calibration ──────────────────────────────────────────────────────────
    def implied_vol(self, market_price: float,
                    lo: float = 1e-4, hi: float = 5.0) -> float:
        """Volatility that reproduces a market price, or NaN when none can.

        Checked against the no-arbitrage bounds first: below intrinsic or above
        the discounted forward there is no σ that works, and returning NaN is the
        honest answer. The old version let brentq fail and swallowed it, which
        looked the same on screen whether the input was unattainable or the
        solver had merely wandered."""
        if self.T <= 0:
            return float("nan")
        disc = math.exp(-self.r * self.T)
        lower = disc * self._intrinsic()
        upper = disc * (self.F if self.opt == "call" else self.K)
        if not (lower - 1e-12 <= market_price <= upper + 1e-12):
            return float("nan")
        f = lambda s: Black76(self.F, self.K, self.T, self.r, s, self.opt).price() - market_price
        try:
            if f(lo) * f(hi) > 0:
                return float("nan")
            return float(brentq(f, lo, hi, xtol=1e-10, maxiter=500))
        except Exception:                                          # noqa: BLE001
            return float("nan")

    # ── profiles ─────────────────────────────────────────────────────────────
    def payoff(self, spots: np.ndarray) -> np.ndarray:
        spots = np.asarray(spots, dtype=float)
        return (np.maximum(spots - self.K, 0.0) if self.opt == "call"
                else np.maximum(self.K - spots, 0.0))

    def pnl(self, spots: np.ndarray) -> np.ndarray:
        return self.payoff(spots) - self.price()


def put_call_parity_error(F, K, T, r, sigma) -> float:
    """|C - P - e^(-rT)(F - K)|. Zero to machine precision for any correct
    Black-76 implementation; displayed on screen as a live self-check."""
    c = Black76(F, K, T, r, sigma, "call").price()
    p = Black76(F, K, T, r, sigma, "put").price()
    return abs((c - p) - math.exp(-r * T) * (F - K))


# ── Asian options ────────────────────────────────────────────────────────────
def kemna_vorst(F, K, T, r, sigma, n_obs, option_type="call") -> dict:
    """Closed form for a GEOMETRIC-average Asian option (Kemna-Vorst 1990).

    With observations at t_i = iT/n, i = 1..n, the geometric average is lognormal:

        σ_G² = σ²(n+1)(2n+1)/(6n²)          → σ/√3 as n → ∞
        E[ln G] = ln F − σ²T(n+1)/(4n)
        F_G     = F·exp(½T(σ_G² − σ²(n+1)/(2n)))

    and the option is Black-76 on (F_G, σ_G). The observation grid is stated
    because the constants differ by convention — this one matches the grid the
    Monte Carlo below actually simulates, which is what lets the two be compared
    as a validation rather than merely quoted side by side.
    """
    n = int(n_obs)
    if n < 1 or T <= 0 or sigma <= 0:
        return dict(price=Black76(F, K, max(T, 0), r, max(sigma, 0), option_type).price(),
                    sigma_g=0.0, F_g=float(F))
    sig_g = sigma * math.sqrt((n + 1) * (2 * n + 1) / (6.0 * n * n))
    F_g = F * math.exp(0.5 * T * (sig_g ** 2 - sigma ** 2 * (n + 1) / (2.0 * n)))
    return dict(price=Black76(F_g, K, T, r, sig_g, option_type).price(),
                sigma_g=sig_g, F_g=F_g)


def asian_mc(F, K, T, r, sigma, n_obs=12, option_type="call",
             average="arithmetic", n_paths=50_000, seed=42,
             antithetic=True, control_variate=True) -> dict:
    """Monte Carlo Asian option with two variance reductions.

    ANTITHETIC pairs each path with its mirror image, which cancels the
    first-order sampling error in the drift.

    CONTROL VARIATE uses the geometric average — whose exact price is known from
    Kemna-Vorst above — to correct the arithmetic estimate:

        price = mean(payoff_arith) − β·(mean(payoff_geo) − exact_geo)

    with β the regression coefficient of one payoff on the other. The two are
    correlated above 0.99, so the standard error typically falls by an order of
    magnitude for the same number of paths. This is the standard technique for
    arithmetic Asians and it costs nothing: the geometric payoff is already on
    the same simulated paths.
    """
    n = max(int(n_obs), 1)
    M = max(int(n_paths), 100)
    if antithetic:
        M = 2 * (M // 2)
    rng = np.random.default_rng(seed)
    dt = T / n

    half = M // 2 if antithetic else M
    Z = rng.standard_normal((half, n))
    if antithetic:
        Z = np.vstack([Z, -Z])
    log_inc = (-0.5 * sigma ** 2 * dt) + sigma * math.sqrt(dt) * Z
    paths = F * np.exp(np.cumsum(log_inc, axis=1))

    disc = math.exp(-r * T)
    arith = paths.mean(axis=1)
    geo = np.exp(np.log(np.maximum(paths, 1e-300)).mean(axis=1))
    target = arith if average == "arithmetic" else geo

    def _pay(x):
        return np.maximum(x - K, 0.0) if option_type == "call" else np.maximum(K - x, 0.0)

    pay = _pay(target)
    beta, cv_used = 0.0, False
    if control_variate and average == "arithmetic":
        pay_geo = _pay(geo)
        var_g = pay_geo.var()
        if var_g > 1e-14:
            beta = float(np.cov(pay, pay_geo, bias=True)[0, 1] / var_g)
            exact_geo = kemna_vorst(F, K, T, r, sigma, n, option_type)["price"] / disc
            pay = pay - beta * (pay_geo - exact_geo)
            cv_used = True

    price = disc * float(pay.mean())
    se = disc * float(pay.std(ddof=1)) / math.sqrt(M)

    # Convergence path — uses the SAME payoff vector, so a put converges to the
    # put price. The previous version always accumulated the call payoff and
    # drew a convergence curve for a price it was not computing.
    steps = np.unique(np.logspace(2, math.log10(M), 30).astype(int))
    conv = [disc * float(pay[:s].mean()) for s in steps]

    return dict(price=price, std_error=se,
                ci_lo=price - 1.96 * se, ci_hi=price + 1.96 * se,
                n_paths=M, n_obs=n, average=average,
                control_variate=cv_used, beta=beta,
                conv_paths=[int(s) for s in steps], conv_prices=conv,
                sample_paths=paths[:60], sample_avgs=target[:2000])


# ── Spread options (Kirk 1995) ───────────────────────────────────────────────
def kirk_sigma(F_long, F_short, K, sigma_long, sigma_short, rho) -> float:
    """Effective volatility for an option on (F_long − F_short) with strike K.

        a = F_short / (F_short + K)
        σ_K = √(σ_long² + (a·σ_short)² − 2ρ·σ_long·σ_short·a)

    The adjustment factor belongs to the SHORT leg — the one absorbed into the
    effective strike (F_short + K). Applying it to the long leg instead, and
    building it from the long forward, is a subtle inversion that survives every
    sanity check a human eye applies: the number is positive, of plausible size,
    and moves the right way with ρ. Against a two-factor Monte Carlo it prices a
    typical crack option ~3% too cheap; with the factor on the correct leg the
    error falls to ~0.1%, which is Kirk's own documented accuracy.
    """
    denom = F_short + K
    if denom <= 0:
        return float(max(sigma_long, 0.0))
    a = F_short / denom
    var = sigma_long ** 2 + (a * sigma_short) ** 2 - 2 * rho * sigma_long * sigma_short * a
    return math.sqrt(max(var, 0.0))


class SpreadOption:
    """Option on the spread (F_long − F_short) via Kirk's approximation.

    Kirk reduces the two-asset problem to one Black-76 on F_long with strike
    (F_short + K) and the effective volatility above. It is an approximation, and
    the app never asks you to take that on trust: the validation panel prices the
    same option by two-factor Monte Carlo and shows the difference in cash.

    Greeks are with respect to the two forwards. Delta on the short leg is
    −e^(−rT)·N(d2), not −delta: the short forward enters through the strike, not
    the underlying, so it carries the N(d2) sensitivity.
    """

    def __init__(self, F_long, F_short, K, T, r, sigma_long, sigma_short,
                 rho=0.85, option_type="call"):
        self.FL = float(F_long)
        self.FS = float(F_short)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sL = float(sigma_long)
        self.sS = float(sigma_short)
        self.rho = float(rho)
        self.opt = str(option_type).lower()

    @property
    def spread(self) -> float:
        return self.FL - self.FS

    def sigma_kirk(self) -> float:
        return kirk_sigma(self.FL, self.FS, self.K, self.sL, self.sS, self.rho)

    def price(self) -> dict:
        disc = math.exp(-self.r * max(self.T, 0.0))
        intrinsic = (max(self.spread - self.K, 0.0) if self.opt == "call"
                     else max(self.K - self.spread, 0.0))
        strike_eff = self.FS + self.K
        if self.T <= 0 or strike_eff <= 0 or self.FL <= 0:
            return dict(price=disc * intrinsic, intrinsic=disc * intrinsic,
                        time_value=0.0, sigma_kirk=0.0, spread=self.spread,
                        delta_long=0.0, delta_short=0.0, gamma=0.0, vega=0.0,
                        strike_eff=strike_eff)
        sk = self.sigma_kirk()
        b = Black76(self.FL, strike_eff, self.T, self.r, sk, self.opt)
        px = b.price()
        if sk > 0:
            d1, d2 = b._d1_d2()
            nd2 = norm.cdf(d2) if self.opt == "call" else -norm.cdf(-d2)
        else:
            nd2 = 0.0
        return dict(price=px, intrinsic=disc * intrinsic,
                    time_value=px - disc * intrinsic, sigma_kirk=sk,
                    spread=self.spread, strike_eff=strike_eff,
                    delta_long=b.delta(), delta_short=-disc * nd2,
                    gamma=b.gamma(), vega=b.vega())

    def mc_price(self, n_paths=400_000, seed=11) -> dict:
        """Independent two-factor Monte Carlo — the benchmark Kirk is measured
        against. Both forwards are martingales under the pricing measure, so each
        is simulated driftless with correlated shocks."""
        if self.T <= 0:
            return dict(price=self.price()["price"], std_error=0.0)
        rng = np.random.default_rng(seed)
        n = 2 * (int(n_paths) // 2)
        z1 = rng.standard_normal(n // 2)
        z2 = rng.standard_normal(n // 2)
        z1 = np.concatenate([z1, -z1])
        z2 = np.concatenate([z2, -z2])
        w = self.rho * z1 + math.sqrt(max(1 - self.rho ** 2, 0.0)) * z2
        sq = math.sqrt(self.T)
        L = self.FL * np.exp(-0.5 * self.sL ** 2 * self.T + self.sL * sq * z1)
        S = self.FS * np.exp(-0.5 * self.sS ** 2 * self.T + self.sS * sq * w)
        sp = L - S
        pay = np.maximum(sp - self.K, 0.0) if self.opt == "call" else np.maximum(self.K - sp, 0.0)
        disc = math.exp(-self.r * self.T)
        return dict(price=disc * float(pay.mean()),
                    std_error=disc * float(pay.std(ddof=1)) / math.sqrt(n))

    def payoff(self, spreads: np.ndarray) -> np.ndarray:
        spreads = np.asarray(spreads, dtype=float)
        return (np.maximum(spreads - self.K, 0.0) if self.opt == "call"
                else np.maximum(self.K - spreads, 0.0))


def structure_legs(struct: str, prices: Dict[str, float],
                   vols: Optional[Dict[str, float]] = None,
                   corr: Optional[pd.DataFrame] = None) -> dict:
    """Collapse a multi-leg structure into the two synthetic forwards Kirk needs.

    Each leg is converted to the structure's common unit by its own factor, so a
    3-2-1 crack carries BOTH products: (2·RB·42 + 1·HO·42 − 3·CL)/3.

    With a leg correlation matrix, each side's volatility is the volatility of
    its value-weighted basket, σ² = ΣΣ wᵢwⱼσᵢσⱼρᵢⱼ, and the correlation between
    the two sides follows from the same matrix. Without one, a side's vol is the
    value-weighted average (perfect correlation inside the side, an upper bound)
    and the cross-correlation is the registry default — and `rho_source` says so.
    """
    cfg = STRUCTURES[struct]
    vols = vols or {}
    div = float(cfg["divide_by"])

    def side(legs):
        total, missing, detail = 0.0, [], []
        for name, qty, conv in legs:
            px = prices.get(name)
            if px is None or not np.isfinite(px):
                missing.append(name)
                continue
            val = px * qty * conv / div
            total += val
            detail.append(dict(leg=name, qty=qty, conv=conv, price=px, value=val,
                               vol=float(vols.get(name, CONTRACTS[name]["vol"]))))
        return total, missing, detail

    def rho_of(a, b):
        if a == b:
            return 1.0
        if corr is None or a not in corr.index or b not in corr.columns:
            return None
        v = float(corr.loc[a, b])
        return v if np.isfinite(v) else None

    def weights(det, total):
        return [d["value"] / total if abs(total) > 1e-12 else 0.0 for d in det]

    FL, miss_l, det_l = side(cfg["long"])
    FS, miss_s, det_s = side(cfg["short"])
    missing = miss_l + miss_s
    wl, ws = weights(det_l, FL), weights(det_s, FS)
    measured = all(rho_of(a["leg"], b["leg"]) is not None
                   for a in det_l + det_s for b in det_l + det_s)

    def basket_cov(da, wa, db, wb):
        return sum(wa[i] * wb[j] * da[i]["vol"] * db[j]["vol"] * (rho_of(da[i]["leg"], db[j]["leg"]) or 1.0)
                   for i in range(len(da)) for j in range(len(db)))

    if measured and det_l and det_s:
        sL = math.sqrt(max(basket_cov(det_l, wl, det_l, wl), 0.0))
        sS = math.sqrt(max(basket_cov(det_s, ws, det_s, ws), 0.0))
        rho = basket_cov(det_l, wl, det_s, ws) / (sL * sS) if sL > 0 and sS > 0 else cfg["rho"]
        rho = float(np.clip(rho, -0.999, 0.999))
        rho_source = "measured from daily returns"
    else:
        sL = sum(abs(w) * d["vol"] for w, d in zip(wl, det_l))
        sS = sum(abs(w) * d["vol"] for w, d in zip(ws, det_s))
        rho, rho_source = cfg["rho"], "registry default"
    return dict(available=not missing, missing=missing,
                F_long=FL, F_short=FS, sigma_long=sL, sigma_short=sS,
                spread=FL - FS, unit=cfg["unit"], rho=rho, rho_source=rho_source,
                note=cfg["note"], basis=cfg["basis"], legs=det_l + det_s)


# ── Barrier options ──────────────────────────────────────────────────────────
BARRIER_TYPES = ["Down-and-Out", "Down-and-In", "Up-and-Out", "Up-and-In"]


def barrier_analytic(F, K, H, T, r, sigma, option_type="call",
                     barrier="Down-and-Out", rebate=0.0) -> float:
    """Reiner-Rubinstein closed form for a CONTINUOUSLY monitored barrier, in the
    futures framework (cost of carry b = 0, the forward is driftless).

    Continuous monitoring is an idealisation: a real contract checks the barrier
    at discrete fixings, which knock less often than a continuous watch. Use
    `barrier_discrete_correction` below to move between the two rather than
    quietly conflating them.
    """
    F, K, H, T, r, sigma = map(float, (F, K, H, T, r, sigma))
    phi = 1.0 if option_type == "call" else -1.0
    eta = 1.0 if barrier.lower().startswith("down") else -1.0
    knock_in = barrier.lower().endswith("in")

    if T <= 0 or sigma <= 0:
        intr = max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
        breached = (F <= H) if eta > 0 else (F >= H)
        alive = breached if knock_in else not breached
        return math.exp(-r * T) * (intr if alive else rebate)

    v = sigma * math.sqrt(T)
    mu = -0.5                                     # (b - σ²/2)/σ² with b = 0
    lam = math.sqrt(mu ** 2 + 2 * r / sigma ** 2)
    disc = math.exp(-r * T)

    x1 = math.log(F / K) / v + (1 + mu) * v
    x2 = math.log(F / H) / v + (1 + mu) * v
    y1 = math.log(H * H / (F * K)) / v + (1 + mu) * v
    y2 = math.log(H / F) / v + (1 + mu) * v
    z = math.log(H / F) / v + lam * v
    HS = H / F

    A = phi * F * disc * norm.cdf(phi * x1) - phi * K * disc * norm.cdf(phi * (x1 - v))
    B = phi * F * disc * norm.cdf(phi * x2) - phi * K * disc * norm.cdf(phi * (x2 - v))
    C = (phi * F * disc * HS ** (2 * (mu + 1)) * norm.cdf(eta * y1)
         - phi * K * disc * HS ** (2 * mu) * norm.cdf(eta * (y1 - v)))
    D = (phi * F * disc * HS ** (2 * (mu + 1)) * norm.cdf(eta * y2)
         - phi * K * disc * HS ** (2 * mu) * norm.cdf(eta * (y2 - v)))
    E = rebate * disc * (norm.cdf(eta * (x2 - v)) - HS ** (2 * mu) * norm.cdf(eta * (y2 - v)))
    Fr = rebate * (HS ** (mu + lam) * norm.cdf(eta * z)
                   + HS ** (mu - lam) * norm.cdf(eta * (z - 2 * lam * v)))

    up = eta < 0
    K_above_H = K > H
    if knock_in:
        if not up and option_type == "call":
            val = (C + E) if K_above_H else (A - B + D + E)
        elif up and option_type == "call":
            val = (A + E) if K_above_H else (B - C + D + E)
        elif not up:
            val = (B - C + D + E) if K_above_H else (A + E)
        else:
            val = (A - B + D + E) if K_above_H else (C + E)
    else:
        if not up and option_type == "call":
            val = (A - C + Fr) if K_above_H else (B - D + Fr)
        elif up and option_type == "call":
            val = Fr if K_above_H else (A - B + C - D + Fr)
        elif not up:
            val = (A - B + C - D + Fr) if K_above_H else Fr
        else:
            val = (B - D + Fr) if K_above_H else (A - C + Fr)
    return float(max(val, 0.0))


def barrier_discrete_correction(H, sigma, T, n_fixings, barrier="Down-and-Out") -> float:
    """Broadie-Glasserman-Kou barrier shift for DISCRETE monitoring.

        H_adj = H · exp(±β·σ·√(T/m)),   β = 0.5826,  + for up, − for down

    A discretely monitored barrier is harder to breach than a continuous one:
    the path can dip below and recover between fixings. Shifting the barrier
    slightly away from the spot and applying the continuous formula reproduces
    the discrete price closely — and gives the app a way to reconcile its Monte
    Carlo with its closed form instead of leaving a gap it cannot explain.
    """
    beta = 0.5826
    m = max(int(n_fixings), 1)
    sign = 1.0 if barrier.lower().startswith("up") else -1.0
    return float(H * math.exp(sign * beta * sigma * math.sqrt(T / m)))


def barrier_mc(F, K, H, T, r, sigma, option_type="call", barrier="Down-and-Out",
               rebate=0.0, n_paths=50_000, n_fixings=None, seed=42) -> dict:
    """Monte Carlo barrier with the monitoring frequency made explicit.

    Returns the knock probability alongside the price, and the vanilla price on
    the same paths so the discount is measured rather than asserted.
    """
    n_fix = int(n_fixings) if n_fixings else max(int(round(T * 252)), 1)
    M = 2 * (max(int(n_paths), 100) // 2)
    rng = np.random.default_rng(seed)
    dt = T / n_fix
    Z = rng.standard_normal((M // 2, n_fix))
    Z = np.vstack([Z, -Z])
    inc = (-0.5 * sigma ** 2 * dt) + sigma * math.sqrt(dt) * Z
    paths = F * np.exp(np.cumsum(inc, axis=1))

    S_T = paths[:, -1]
    lo = paths.min(axis=1)
    hi = paths.max(axis=1)
    disc = math.exp(-r * T)
    van = np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)

    down = barrier.lower().startswith("down")
    breached = (lo <= H) if down else (hi >= H)
    alive = breached if barrier.lower().endswith("in") else ~breached
    pay = np.where(alive, van, rebate)

    return dict(price=disc * float(pay.mean()),
                std_error=disc * float(pay.std(ddof=1)) / math.sqrt(M),
                vanilla=disc * float(van.mean()),
                knock_prob=float(breached.mean()),
                n_fixings=n_fix, n_paths=M,
                paths=paths[:60], breached=breached[:60])


# ── American exercise ────────────────────────────────────────────────────────
#  Most listed commodity options are AMERICAN: LO on WTI, the grain options, the
#  metals. Pricing them with Black-76 ignores the right to exercise early, which
#  on a futures option is worth 0.4% at a 2% rate and 2.6% at 10% — small, real,
#  and the first thing an options trader checks.
OPTION_EXPIRY_LAG_DAYS = 5      # options stop trading before their future does


def option_tenor(future_T: float, lag_days: int = OPTION_EXPIRY_LAG_DAYS) -> float:
    """An option on a future expires BEFORE the future itself — typically a few
    business days. Pricing to the future's expiry overstates the tenor."""
    return max(future_T - lag_days / 365.25, 1e-6)


def crr_price(F, K, T, r, sigma, option_type="call", n_steps=400,
              american=True) -> float:
    """Cox-Ross-Rubinstein on a futures underlying.

    The forward is a martingale under the pricing measure, so the up-probability
    is (1 − d)/(u − d) with NO growth term — the single most common mistake when
    adapting an equity tree to futures. Each step discounts at r; an American
    node takes max(continuation, intrinsic).
    """
    F, K, T, r, sigma = map(float, (F, K, T, r, sigma))
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
    n = max(int(n_steps), 8)
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (1.0 - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(n + 1)
    S = F * u ** (n - j) * d ** j
    V = np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)
    for i in range(n - 1, -1, -1):
        jj = np.arange(i + 1)
        S = F * u ** (i - jj) * d ** jj
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            ex = np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)
            V = np.maximum(V, ex)
    return float(V[0])


def baw_price(F, K, T, r, sigma, option_type="call") -> float:
    """Barone-Adesi-Whaley quadratic approximation, b = 0.

    A closed form, so it is fast enough to reprice a whole book — but it IS an
    approximation, drifting above the tree as rates rise. The Model Checks tab
    prints the gap rather than letting you assume it is zero.
    """
    F, K, T, r, sigma = map(float, (F, K, T, r, sigma))
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
    eur = Black76(F, K, T, r, sigma, option_type).price()
    M = 2 * r / sigma ** 2
    Kf = 1 - math.exp(-r * T)
    if Kf <= 1e-12:
        return eur

    def _d1(S):
        return (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))

    if option_type == "call":
        q2 = (1 + math.sqrt(1 + 4 * M / Kf)) / 2
        lo, hi = K, K * 20
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            f = (mid - K - Black76(mid, K, T, r, sigma, "call").price()
                 - (1 - math.exp(-r * T) * norm.cdf(_d1(mid))) * mid / q2)
            if f > 0:
                hi = mid
            else:
                lo = mid
        Sx = 0.5 * (lo + hi)
        if F >= Sx:
            return F - K
        A2 = (Sx / q2) * (1 - math.exp(-r * T) * norm.cdf(_d1(Sx)))
        return eur + A2 * (F / Sx) ** q2

    q1 = (1 - math.sqrt(1 + 4 * M / Kf)) / 2
    lo, hi = 1e-8, K
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        f = (K - mid - Black76(mid, K, T, r, sigma, "put").price()
             + (1 - math.exp(-r * T) * norm.cdf(-_d1(mid))) * mid / q1)
        if f > 0:
            lo = mid
        else:
            hi = mid
    Sx = 0.5 * (lo + hi)
    if F <= Sx:
        return K - F
    A1 = -(Sx / q1) * (1 - math.exp(-r * T) * norm.cdf(-_d1(Sx)))
    return eur + A1 * (F / Sx) ** q1


def american_greeks(F, K, T, r, sigma, option_type="call", n_steps=300) -> dict:
    """Greeks on the tree by finite difference. Slower than a formula and exact
    to the model, which is the right trade when the model has no formula."""
    h = max(F * 1e-3, 1e-6)
    p0 = crr_price(F, K, T, r, sigma, option_type, n_steps)
    pu = crr_price(F + h, K, T, r, sigma, option_type, n_steps)
    pd_ = crr_price(F - h, K, T, r, sigma, option_type, n_steps)
    dv = 0.005
    pv = crr_price(F, K, T, r, sigma + dv, option_type, n_steps)
    pv2 = crr_price(F, K, T, r, max(sigma - dv, 1e-6), option_type, n_steps)
    dt = 1 / 365
    pt = crr_price(F, K, max(T - dt, 1e-6), r, sigma, option_type, n_steps)
    return dict(price=p0, delta=(pu - pd_) / (2 * h), gamma=(pu - 2 * p0 + pd_) / h ** 2,
                vega=(pv - pv2) / (2 * dv) / 100, theta=(pt - p0),
                early_premium=p0 - Black76(F, K, T, r, sigma, option_type).price())


# ── The delta language ───────────────────────────────────────────────────────
def strike_from_delta(F, T, r, sigma, target_delta, option_type="call") -> float:
    """Strike with a given delta — how traders actually name options.

    Nobody quotes "the 74.50 put"; they quote "the 25-delta put". This inverts
    the Black-76 delta so the whole platform can speak that language: strategies
    can be built at 25d, and the risk reversal and butterfly below are defined
    on it.
    """
    d = abs(float(target_delta))
    if not (1e-6 < d < 1 - 1e-6) or T <= 0 or sigma <= 0:
        return float(F)
    disc = math.exp(-r * T)
    d_undisc = min(max(d / disc, 1e-9), 1 - 1e-9)
    z = norm.ppf(d_undisc if option_type == "call" else 1 - d_undisc)
    return float(F * math.exp(-z * sigma * math.sqrt(T) + 0.5 * sigma ** 2 * T))


def risk_reversal_butterfly(F, T, r, sabr_params, delta=0.25) -> dict:
    """The two numbers a vol desk quotes for skew and smile.

        RR(Δ) = σ(call Δ) − σ(put Δ)       negative = puts bid, the usual shape
        BF(Δ) = ½(σ(call Δ) + σ(put Δ)) − σ_ATM     the wings against the body

    Quoted in VOL POINTS, not prices — "the 25-delta risk reversal is at minus
    three vols" is a complete description of a skew, and it is the sentence this
    function exists to produce.
    """
    a, b, rho, nu = (sabr_params[k] for k in ("alpha", "beta", "rho", "nu"))
    atm = sabr_vol(F, F, T, a, b, rho, nu)
    kc = strike_from_delta(F, T, r, atm, delta, "call")
    kp = strike_from_delta(F, T, r, atm, delta, "put")
    for _ in range(6):                       # iterate: strike depends on its own vol
        vc = sabr_vol(F, kc, T, a, b, rho, nu)
        vp = sabr_vol(F, kp, T, a, b, rho, nu)
        kc = strike_from_delta(F, T, r, vc, delta, "call")
        kp = strike_from_delta(F, T, r, vp, delta, "put")
    vc = sabr_vol(F, kc, T, a, b, rho, nu)
    vp = sabr_vol(F, kp, T, a, b, rho, nu)
    return dict(delta=delta, K_call=kc, K_put=kp, vol_call=vc, vol_put=vp,
                atm=atm, rr=vc - vp, bf=0.5 * (vc + vp) - atm)


# ── SABR (Hagan et al. 2002) ─────────────────────────────────────────────────
def sabr_vol(F, K, T, alpha, beta, rho, nu) -> float:
    """Lognormal implied volatility under SABR — the market standard for
    commodity option surfaces.

    Four parameters with distinct jobs: α sets the level, β the backbone (how
    the smile moves when the forward moves; 0.5 is the usual commodity choice,
    1.0 is lognormal), ρ the skew, ν the smile's curvature. Unlike a polynomial
    fit, each one means something a trader can argue about.
    """
    F, K, T = float(F), float(K), float(T)
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return float("nan")
    one_b = 1.0 - beta
    corr = (one_b ** 2 / 24.0 * alpha ** 2 / (F * K) ** one_b
            + 0.25 * rho * beta * nu * alpha / ((F * K) ** (one_b / 2))
            + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0)
    if abs(F - K) < 1e-9 * max(F, 1.0):
        Fb = F ** one_b
        atm_corr = (one_b ** 2 / 24.0 * alpha ** 2 / Fb ** 2
                    + 0.25 * rho * beta * nu * alpha / Fb
                    + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0)
        return float(alpha / Fb * (1.0 + atm_corr * T))
    logFK = math.log(F / K)
    FK = (F * K) ** (one_b / 2.0)
    z = (nu / alpha) * FK * logFK
    num = math.sqrt(max(1.0 - 2.0 * rho * z + z * z, 1e-16)) + z - rho
    den = 1.0 - rho
    if abs(den) < 1e-12 or num <= 0:
        return float("nan")
    zx = 1.0 if abs(z) < 1e-9 else z / math.log(num / den)
    denom = FK * (1.0 + one_b ** 2 / 24.0 * logFK ** 2 + one_b ** 4 / 1920.0 * logFK ** 4)
    return float(alpha / denom * zx * (1.0 + corr * T))


def sabr_alpha_from_atm(F, T, atm_vol, beta, rho, nu) -> float:
    """Solve α so the surface reproduces the observed ATM volatility EXACTLY.

    Market convention, and it matters: ATM is the one quote everybody agrees on,
    so it is an input, not something a fit is allowed to miss. That leaves ρ and
    ν to carry the skew and the smile."""
    one_b = 1.0 - beta
    Fb = F ** one_b
    c3 = one_b ** 2 / 24.0 * T / Fb ** 3
    c2 = 0.25 * rho * beta * nu * T / Fb ** 2
    c1 = (1.0 + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0 * T) / Fb
    c0 = -atm_vol
    if abs(c3) > 1e-14:
        roots = np.roots([c3, c2, c1, c0])
    elif abs(c2) > 1e-14:
        roots = np.roots([c2, c1, c0])
    else:
        roots = np.array([-c0 / c1])
    real = [float(x.real) for x in np.atleast_1d(roots)
            if abs(np.imag(x)) < 1e-8 and x.real > 0]
    return min(real) if real else float(atm_vol * F ** one_b)


def sabr_calibrate(F, T, strikes, vols, beta=0.5, atm: Optional[float] = None) -> dict:
    """Least-squares fit of ρ and ν to a strip of quotes, α pinned to ATM.

    Started from several initial guesses because the objective has local minima —
    a single start converges to a plausible-looking wrong answer often enough to
    matter."""
    strikes = np.asarray(strikes, dtype=float)
    vols = np.asarray(vols, dtype=float)
    order = np.argsort(strikes)
    strikes, vols = strikes[order], vols[order]
    atm = float(np.interp(F, strikes, vols)) if atm is None else float(atm)

    def resid(p):
        rho, nu = p
        a = sabr_alpha_from_atm(F, T, atm, beta, rho, nu)
        model = np.array([sabr_vol(F, k, T, a, beta, rho, nu) for k in strikes])
        return np.nan_to_num(model, nan=1e3) - vols

    best = None
    for start in ((-0.3, 0.4), (0.0, 0.6), (-0.6, 0.9), (0.3, 0.3)):
        try:
            s = least_squares(resid, list(start), bounds=([-0.999, 1e-4], [0.999, 5.0]),
                              xtol=1e-12, ftol=1e-12)
            if best is None or s.cost < best.cost:
                best = s
        except Exception:                                          # noqa: BLE001
            continue
    if best is None:
        return dict(ok=False)
    rho, nu = float(best.x[0]), float(best.x[1])
    alpha = sabr_alpha_from_atm(F, T, atm, beta, rho, nu)
    fit = np.array([sabr_vol(F, k, T, alpha, beta, rho, nu) for k in strikes])
    return dict(ok=True, alpha=alpha, beta=beta, rho=rho, nu=nu, atm=atm,
                strikes=strikes, market=vols, fit=fit,
                rmse=float(np.sqrt(np.mean((fit - vols) ** 2))),
                max_err=float(np.max(np.abs(fit - vols))))



# ══════════════════════════════════════════════════════════════════════════════
#  4. VOLATILITY ANALYTICS
#  What a vol trader actually looks at: is the quoted volatility rich against
#  what this market has really delivered, and does hedging it pay?
# ══════════════════════════════════════════════════════════════════════════════


def realised_vol(prices: pd.Series, window: int) -> pd.Series:
    lr = np.log(prices / prices.shift(1))
    return lr.rolling(window).std() * math.sqrt(252)


def vol_cone(prices: pd.Series,
             windows: Sequence[int] = (10, 20, 30, 60, 90, 120, 252),
             pctiles: Sequence[int] = (5, 25, 50, 75, 95)) -> pd.DataFrame:
    """Realised volatility at several horizons, each with its own historical
    distribution.

    This is how a vol trader decides whether an implied quote is rich: not
    against one number but against the range that horizon has actually produced.
    A 35% implied is cheap in gas and extravagant in gold, and the cone is what
    makes that comparison without needing an opinion.
    """
    rows = []
    for w in windows:
        s = realised_vol(prices, w).dropna()
        if len(s) < max(40, w // 2):
            continue
        row = dict(window=w, current=float(s.iloc[-1]), n=int(len(s)))
        for p in pctiles:
            row[f"p{p}"] = float(np.percentile(s, p))
        row["rank"] = float((s < s.iloc[-1]).mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def forward_vol(sigma_1, T1, sigma_2, T2) -> float:
    """Volatility implied between two expiries — what a calendar option trades.

        σ_fwd² · (T2 − T1) = σ2²·T2 − σ1²·T1

    Returns NaN when the term structure is steep enough downward to imply a
    negative forward variance: that is an arbitrage, not a number, and printing
    it as one would be worse than printing nothing."""
    if T2 <= T1:
        return float("nan")
    var = (sigma_2 ** 2 * T2 - sigma_1 ** 2 * T1) / (T2 - T1)
    return math.sqrt(var) if var > 0 else float("nan")


def gamma_scalp(F, K, T, r, sigma_implied, sigma_realised, n_rebal=63,
                n_paths=4000, seed=42, structure="straddle",
                position="long") -> dict:
    """Delta-hedged option P&L when realised volatility differs from implied.

    The classic result is that a continuously hedged long option earns
    ½∫Γ·F²(σ_real² − σ_impl²)dt — money when the market moves more than you paid
    for. Simulating it adds the half of the story the formula hides: DISPERSION.
    Hedging at discrete intervals leaves path risk, so a correct volatility view
    still loses on a meaningful fraction of paths, and the width of that
    distribution is the real risk of a gamma book.
    """
    legs = ["call", "put"] if structure == "straddle" else [structure]
    sgn = 1.0 if position == "long" else -1.0
    rng = np.random.default_rng(seed)
    dt = T / max(int(n_rebal), 1)
    M = 2 * (max(int(n_paths), 100) // 2)
    Z = rng.standard_normal((M // 2, n_rebal))
    Z = np.vstack([Z, -Z])
    inc = (-0.5 * sigma_realised ** 2 * dt) + sigma_realised * math.sqrt(dt) * Z
    paths = np.hstack([np.full((M, 1), float(F)), F * np.exp(np.cumsum(inc, axis=1))])

    def value_delta(Fv: np.ndarray, Tr: float):
        v = np.zeros_like(Fv)
        d = np.zeros_like(Fv)
        for leg in legs:
            if Tr <= 1e-9:
                intr = (np.maximum(Fv - K, 0.0) if leg == "call"
                        else np.maximum(K - Fv, 0.0))
                dd = ((Fv > K).astype(float) if leg == "call"
                      else -(Fv < K).astype(float))
                v += intr
                d += dd
                continue
            vv = sigma_implied * math.sqrt(Tr)
            d1 = (np.log(Fv / K) + 0.5 * sigma_implied ** 2 * Tr) / vv
            d2 = d1 - vv
            disc = math.exp(-r * Tr)
            if leg == "call":
                v += disc * (Fv * norm.cdf(d1) - K * norm.cdf(d2))
                d += disc * norm.cdf(d1)
            else:
                v += disc * (K * norm.cdf(-d2) - Fv * norm.cdf(-d1))
                d += disc * (norm.cdf(d1) - 1.0)
        return v, d

    v_prev, d_prev = value_delta(paths[:, 0], T)
    premium = float(v_prev[0])
    opt_pnl = np.zeros(M)
    hedge_pnl = np.zeros(M)
    for i in range(1, n_rebal + 1):
        Tr = max(T - i * dt, 0.0)
        v, d = value_delta(paths[:, i], Tr)
        dF = paths[:, i] - paths[:, i - 1]
        opt_pnl += sgn * (v - v_prev)
        hedge_pnl += -sgn * d_prev * dF
        v_prev, d_prev = v, d
    total = opt_pnl + hedge_pnl

    d1 = (math.log(F / K) + 0.5 * sigma_implied ** 2 * T) / (sigma_implied * math.sqrt(T))
    gam = math.exp(-r * T) * norm.pdf(d1) / (F * sigma_implied * math.sqrt(T))
    theory = sgn * 0.5 * gam * len(legs) * F ** 2 * (sigma_realised ** 2 - sigma_implied ** 2) * T

    return dict(pnl=total, mean=float(total.mean()), std=float(total.std(ddof=1)),
                p5=float(np.percentile(total, 5)), p95=float(np.percentile(total, 95)),
                win_rate=float((total > 0).mean() * 100), premium=premium * sgn,
                theory=float(theory), paths=paths[:40],
                n_rebal=int(n_rebal), n_paths=M)



# ══════════════════════════════════════════════════════════════════════════════
#  4b. AUTOMATIC VOLATILITY
#  No volatility on this desk is typed in. It is implied from listed option
#  prices where any exist, forecast from price history where none do, and the
#  screen always says which. The pieces, bottom-up:
#
#    chain_implied_vols   one expiry of a fund's option chain → (F, k, σ) points
#    fit_expiry_smile     those points → ATM vol + SABR (ρ, ν), outliers dropped
#    garch_fit            front-month returns → GARCH(1,1) vol term structure
#    samuelson_fit        each month's realised / reference realised → g(τ)
#    build_vol_surface    the hierarchy: implied > forecast > registry
#    surface_vol_fn       σ(K, T) for one futures month, SABR off its forward
# ══════════════════════════════════════════════════════════════════════════════

SABR_BETA = 0.5            # backbone used for every automatic fit (stated, not fitted)
VOL_WINDOW = 60            # sessions over which realised vols are compared
MIN_SMILE_POINTS = 5       # fewer quotes than this: ATM only, no smile
VOL_FLOOR, VOL_CAP = 0.02, 3.0
_TARGET_DAYS = (30, 61, 91, 182, 273, 365, 547, 730)


def pick_expiries(expiries: Sequence[str], today: date, n: int = 8) -> List[str]:
    """Choose up to n listed expiries spread across the term structure — one near
    each of 1M, 2M, 3M, 6M, 9M, 1Y, 18M, 2Y — so a handful of requests covers the
    whole curve. Expiries inside five days are skipped: their vols are noise."""
    parsed = []
    for e in expiries:
        try:
            d = datetime.strptime(str(e)[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        days = (d - today).days
        if 5 <= days <= 800:
            parsed.append((days, str(e)[:10]))
    if not parsed:
        return []
    chosen: List[Tuple[int, str]] = []
    for tgt in _TARGET_DAYS[:max(n, 1)]:
        best = min(parsed, key=lambda x: abs(x[0] - tgt))
        if best not in chosen and abs(best[0] - tgt) <= max(25, tgt * 0.35):
            chosen.append(best)
    if not chosen:
        chosen = [parsed[0]]
    return [e for _, e in sorted(chosen)]


class FeedUnavailable(Exception):
    """Raised inside a cached fetch so a failure is NOT cached: Streamlit caches
    return values, never exceptions. Carries the fallback result in args[0]."""


NEG_TTL = 120              # seconds before a failed options request is tried again


@st.cache_resource(show_spinner=False)
def _failed_fetches() -> dict:
    """Process-wide memory of recent failures, so a Yahoo outage costs one slow
    request every two minutes rather than one per click."""
    return {}


def fetch_option_chains(etf: str, max_expiries: int = 8) -> dict:
    """Listed option chains on a fund, cached for 30 minutes when they arrive.
    A failure (rate limit, timeout, empty answer) is remembered for NEG_TTL
    seconds and then retried — never pinned for half an hour."""
    neg = _failed_fetches()
    hit = neg.get(etf)
    if hit and time.time() - hit[0] < NEG_TTL:
        return hit[1]
    try:
        out = _option_chains_cached(etf, max_expiries)
        neg.pop(etf, None)
        return out
    except FeedUnavailable as e:
        neg[etf] = (time.time(), e.args[0])
        return e.args[0]


@st.cache_data(ttl=1800, show_spinner=False)
def _option_chains_cached(etf: str, max_expiries: int = 8) -> dict:
    out = _download_option_chains(etf, max_expiries)
    if not out["chains"]:
        raise FeedUnavailable(out)
    return out


def _download_option_chains(etf: str, max_expiries: int = 8) -> dict:
    """Listed option chains on a fund — a few expiries across the curve.

    Only the columns the inversion needs are kept. The fund's spot is read from
    the chain itself where Yahoo supplies it, else from the last close."""
    out = dict(etf=etf, spot=None, chains=[], error=None, fetched=str(date.today()))
    if not YF_AVAILABLE:
        out["error"] = "yfinance not installed"
        return out
    try:
        tk = yf.Ticker(etf)
        exps = list(tk.options or [])
    except Exception as e:                                         # noqa: BLE001
        out["error"] = f"no option list: {e}"
        LOG.warning("%s options list failed: %s", etf, e)
        return out
    if not exps:
        out["error"] = ("Yahoo returned no option list — usually a temporary rate limit; "
                        f"retried automatically every {NEG_TTL // 60} minutes")
        LOG.warning("%s: empty option list", etf)
        return out
    chosen = pick_expiries(exps, date.today(), max_expiries)
    if not chosen:
        out["error"] = "no listed expiries between 5 days and 2 years"
        return out
    keep = ["strike", "bid", "ask", "lastPrice", "volume", "openInterest", "lastTradeDate"]
    for e in chosen:
        try:
            ch = tk.option_chain(e)
        except Exception as ex:                                    # noqa: BLE001
            LOG.warning("%s chain %s failed: %s", etf, e, ex)
            continue
        calls, puts = getattr(ch, "calls", None), getattr(ch, "puts", None)
        if calls is None or puts is None or len(calls) + len(puts) == 0:
            continue
        und = getattr(ch, "underlying", None)
        if out["spot"] is None and isinstance(und, dict):
            for fld in ("regularMarketPrice", "postMarketPrice", "regularMarketPreviousClose"):
                v = und.get(fld)
                if v and float(v) > 0:
                    out["spot"] = float(v)
                    break

        def slim(df):
            df = df.reindex(columns=keep).copy()
            ltd = pd.to_datetime(df["lastTradeDate"], errors="coerce", utc=True)
            df["lastTradeDate"] = ltd.dt.strftime("%Y-%m-%d")
            for c in keep[:-1]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.reset_index(drop=True)

        out["chains"].append(dict(expiry=e, calls=slim(calls), puts=slim(puts)))
    if out["spot"] is None:
        try:
            h = tk.history(period="5d")
            out["spot"] = float(h["Close"].dropna().iloc[-1])
        except Exception:                                          # noqa: BLE001
            pass
    if not out["chains"]:
        out["error"] = "every chain request came back empty"
    LOG.info("%s: %d option expiries fetched", etf, len(out["chains"]))
    return out


def _quote_prices(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """Usable premium per strike: the bid/ask mid where a two-sided market exists
    and is not absurdly wide; otherwise the last trade, only if it is recent and
    actually traded. Each row says which it used."""
    df = df.copy()
    for c in ("bid", "ask", "lastPrice", "volume", "openInterest"):
        if c not in df:
            df[c] = np.nan
    bid, ask = df["bid"].fillna(0.0), df["ask"].fillna(0.0)
    mid = 0.5 * (bid + ask)
    two_sided = (bid > 0) & (ask > bid) & ((ask - bid) / mid.where(mid > 0, np.nan) < 0.6)
    df["px"] = np.where(two_sided, mid, np.nan)
    df["src"] = np.where(two_sided, "mid", "")
    if two_sided.sum() < 3:                    # after-hours zeros: fall back to trades
        ltd = (pd.to_datetime(df["lastTradeDate"], errors="coerce") if "lastTradeDate" in df
               else pd.Series(pd.NaT, index=df.index))
        if getattr(ltd.dt, "tz", None) is not None:
            ltd = ltd.dt.tz_localize(None)
        recent = ltd.notna() & ((pd.Timestamp(today) - ltd).dt.days <= 5)
        traded = (df["lastPrice"].fillna(0) > 0) & (df["volume"].fillna(0) > 0) & recent
        use = ~two_sided & traded
        df.loc[use, "px"] = df.loc[use, "lastPrice"]
        df.loc[use, "src"] = "last"
    return df


def chain_implied_vols(calls: pd.DataFrame, puts: pd.DataFrame, T: float, r: float,
                       spot: Optional[float], today: Optional[date] = None
                       ) -> Tuple[Optional[float], pd.DataFrame]:
    """One expiry → the fund's forward and an implied vol per out-of-the-money strike.

    FORWARD from put-call parity, F = K + e^{rT}(C − P), median over the strikes
    nearest the money. That absorbs dividends and borrow automatically; if it
    disagrees with S·e^{rT} by more than 5% something is wrong with the quotes and
    the carry forward is used instead.

    Only out-of-the-money options are inverted (puts below F, calls above): they
    carry no early-exercise value worth the name — these fund options are
    American — and they are the liquid side of every strike.
    """
    today = today or date.today()
    if T <= 0:
        return None, pd.DataFrame()
    c = _quote_prices(calls, today).set_index("strike")
    p = _quote_prices(puts, today).set_index("strike")
    c = c[~c.index.duplicated()]
    p = p[~p.index.duplicated()]
    disc = math.exp(-r * T)
    F_carry = spot * math.exp(r * T) if spot else None

    both = c[["px"]].join(p[["px"]], lsuffix="_c", rsuffix="_p", how="inner").dropna()
    F = None
    if len(both):
        centre = spot if spot else float(np.median(both.index))
        near = both.iloc[np.argsort(np.abs(both.index.to_numpy(dtype=float) - centre))[:5]]
        est = near.index.to_numpy(dtype=float) + (near["px_c"] - near["px_p"]).to_numpy() / disc
        F = float(np.median(est))
    if F is None or F <= 0 or (F_carry and abs(F / F_carry - 1) > 0.05):
        F = F_carry
    if not F or F <= 0:
        return None, pd.DataFrame()

    floor = max(0.03, 0.001 * F)
    rows = []
    for typ, book_ in (("put", p), ("call", c)):
        for K, row in book_.iterrows():
            K = float(K)
            px = row["px"]
            if not np.isfinite(px) or px < floor or not (0.5 * F <= K <= 1.6 * F):
                continue
            if (typ == "put" and K > F * 1.005) or (typ == "call" and K < F * 0.995):
                continue
            iv = Black76(F, K, T, r, 0.3, typ).implied_vol(float(px))
            if not np.isfinite(iv) or not (0.03 <= iv <= VOL_CAP):
                continue
            rows.append(dict(strike=K, k=math.log(K / F), type=typ, price=float(px),
                             iv=iv, src=row["src"]))
    # Near the money both legs of a strike are kept as separate points: if one is
    # a stale print, it is an outlier on its own instead of half of an average.
    return F, pd.DataFrame(rows)


def robust_atm(k: Sequence[float], iv: Sequence[float]) -> Optional[float]:
    """ATM vol from the quotes around k = 0.

    A weighted local quadratic through every strike within ±12% of the forward,
    after quotes more than four robust deviations off a Cauchy-loss fit (which no
    single print can drag) are discarded.
    Interpolating between the two nearest strikes — the obvious method — lets a
    single stale print at the money move the whole expiry by several vols, and on
    a thin fund chain that happens daily. None if nothing is near the money."""
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    if len(k) == 0 or np.min(np.abs(k)) > 0.06:
        return None
    near = np.abs(k) <= 0.12
    if near.sum() >= 5 and k[near].min() < 0 < k[near].max():
        kk, vv = k[near], iv[near]
        w = np.sqrt(np.exp(-(kk / 0.08) ** 2))
        deg = 2 if len(kk) >= 6 else 1
        # Robust start: a Cauchy loss caps any one quote's pull on the fit, so a
        # bad print at the money cannot drag the curve to itself and hide there.
        x0 = np.zeros(deg + 1)
        x0[-1] = float(np.median(vv))
        c0 = least_squares(lambda c: w * (np.polyval(c, kk) - vv), x0,
                           loss="cauchy", f_scale=0.005).x
        res = vv - np.polyval(c0, kk)
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        keep = np.abs(res) <= max(4 * mad, 0.01)
        if keep.sum() >= 4 and kk[keep].min() < 0 < kk[keep].max():
            kk, vv, w = kk[keep], vv[keep], w[keep]
        coef = np.polyfit(kk, vv, 2 if len(kk) >= 6 else 1, w=w)
        return float(np.polyval(coef, 0.0))
    o = np.argsort(k)
    k, iv = k[o], iv[o]
    if k[0] <= 0 <= k[-1] and len(k) >= 2:
        return float(np.interp(0.0, k, iv))
    return float(iv[int(np.argmin(np.abs(k)))])


def fit_expiry_smile(F: float, T: float, pts: pd.DataFrame,
                     beta: float = SABR_BETA) -> Optional[dict]:
    """ATM vol and a SABR smile for one expiry.

    ATM is read off the quotes themselves (interpolated at k = 0), never fitted:
    it is the number everybody agrees on. With enough strikes on both sides, ρ and
    ν are fitted with α pinned to that ATM; points the first pass misses by more
    than max(3·RMSE, 3 vols) are treated as bad prints and dropped once. A fit
    still worse than 3 vols RMSE is not trusted and the expiry keeps its ATM only.
    """
    if pts is None or pts.empty:
        return None
    pts = pts.sort_values("k")
    k, iv = pts["k"].to_numpy(float), pts["iv"].to_numpy(float)
    atm = robust_atm(k, iv)
    if atm is None:
        return None
    out = dict(T=float(T), F=float(F), atm=atm, rho=0.0, nu=0.0, beta=beta,
               has_smile=False, rmse=float("nan"), n=int(len(k)),
               k=k.tolist(), iv=iv.tolist(), dropped=0, quality="ATM only")
    wings = (k < -0.02).sum() >= 2 and (k > 0.02).sum() >= 2
    if len(k) < MIN_SMILE_POINTS or not wings:
        return out
    strikes = F * np.exp(k)
    use = np.ones(len(k), dtype=bool)
    for _ in range(3):
        cal = sabr_calibrate(F, T, strikes[use], iv[use], beta, atm=atm)
        if not cal.get("ok"):
            return out
        a_ = sabr_alpha_from_atm(F, T, atm, beta, cal["rho"], cal["nu"])
        model = np.array([sabr_vol(F, K, T, a_, beta, cal["rho"], cal["nu"]) for K in strikes])
        resid = np.abs(np.nan_to_num(model, nan=9.0) - iv)
        bad = resid > max(3 * cal["rmse"], 0.03)
        new = use & ~bad
        if new.sum() == use.sum() or new.sum() < MIN_SMILE_POINTS:
            break
        use = new
    rmse = float(np.sqrt(np.mean((model[use] - iv[use]) ** 2)))
    out["dropped"] = int((~use).sum())
    if rmse > 0.03:
        out.update(quality="noisy", rmse=rmse)
        return out
    if abs(cal["rho"]) >= 0.98:            # ρ pinned to its bound: a degenerate fit
        out.update(quality="ATM only — degenerate smile", rmse=rmse)
        return out
    out.update(rho=float(cal["rho"]), nu=float(cal["nu"]), has_smile=True, rmse=rmse,
               quality="smile")
    return out


def interp_total_variance(Ts: Sequence[float], vols: Sequence[float], T: float) -> float:
    """ATM vol at any tenor from a set of expiries, linear in TOTAL VARIANCE σ²T.
    Total variance is forced non-decreasing first (a dip would be a calendar
    arbitrage); flat vol beyond either end."""
    Ts = np.asarray(Ts, float)
    vols = np.asarray(vols, float)
    o = np.argsort(Ts)
    Ts, vols = Ts[o], vols[o]
    if len(Ts) == 1 or T <= Ts[0]:
        return float(vols[0])
    if T >= Ts[-1]:
        return float(vols[-1])
    w = np.maximum.accumulate(vols ** 2 * Ts)
    return float(math.sqrt(max(np.interp(T, Ts, w), 1e-12) / T))


# ── GARCH(1,1): the forecast when nothing is implied ─────────────────────────
def garch_fit(returns: Sequence[float]) -> Optional[dict]:
    """Gaussian GARCH(1,1) by maximum likelihood, with variance targeting.

        h_t = ω + α·r²_{t−1} + β·h_{t−1},   ω = σ²_LR·(1 − α − β)

    Targeting the long-run variance to the sample leaves two parameters, which
    makes the fit stable on five years of daily data. Returns beyond eight robust
    standard deviations are clipped first: on a continuous futures series those
    are roll jumps and bad prints, not information about volatility.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 250:
        return None
    r = r - r.mean()
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    if mad > 0:
        r = np.clip(r, -8 * mad, 8 * mad)
    var_lr = float(r.var())
    if var_lr <= 0:
        return None
    r2 = r ** 2

    def filt(a, b):
        x = var_lr * (1 - a - b) + a * r2[:-1]
        h = lfilter([1.0], [1.0, -b], x, zi=[b * var_lr])[0]
        return np.concatenate([[var_lr], h])

    def nll(p):
        a, b = p
        if a <= 0 or b < 0 or a + b >= 0.999:
            return 1e12
        h = filt(a, b)
        if np.any(h <= 0):
            return 1e12
        return 0.5 * float(np.sum(np.log(h) + r2 / h))

    best = None
    for x0 in ((0.06, 0.90), (0.10, 0.85), (0.03, 0.95)):
        try:
            res = minimize(nll, x0, method="Nelder-Mead",
                           options=dict(xatol=1e-6, fatol=1e-6, maxiter=800))
            if best is None or res.fun < best.fun:
                best = res
        except Exception:                                          # noqa: BLE001
            continue
    if best is None or best.fun >= 1e11:
        return None
    a, b = (float(x) for x in best.x)
    h = filt(a, b)
    h_next = var_lr * (1 - a - b) + a * r2[-1] + b * h[-1]
    phi = a + b
    return dict(alpha=a, beta=b, persistence=phi, var_lr=var_lr, h_next=float(h_next),
                vol_now=math.sqrt(252 * h_next), vol_lr=math.sqrt(252 * var_lr),
                half_life=(math.log(0.5) / math.log(phi)) if 0 < phi < 1 else float("inf"),
                n=int(len(r)))


def garch_term_vol(g: dict, T: float) -> float:
    """Expected average volatility over the next T years:
        σ̄²(N) = σ²_LR + (h_next − σ²_LR)·(1 − φᴺ)/(N(1 − φ)),  N = 252·T sessions."""
    N = max(252.0 * float(T), 1.0)
    phi = g["persistence"]
    if phi >= 1 - 1e-9:
        avg = g["h_next"]
    else:
        avg = g["var_lr"] + (g["h_next"] - g["var_lr"]) * (1 - phi ** N) / (N * (1 - phi))
    return math.sqrt(252.0 * max(avg, 1e-12))


# ── Samuelson mapping: a fund's (or the front's) vol → each futures month ────
def log_returns(s: pd.Series) -> pd.Series:
    s = s[s > 0]
    return np.log(s / s.shift(1)).dropna()


def contract_vol_ratios(closes: pd.DataFrame, specs: List[dict], ref: pd.Series,
                        window: int = VOL_WINDOW) -> pd.DataFrame:
    """Each listed month's realised vol divided by the reference series' realised
    vol, measured on the SAME sessions.

    A thin deferred month that repeats yesterday's settle looks artificially calm,
    so a month needs 40 overlapping sessions and price changes on 70% of them to
    count. The ratio is attached to the tenor the window was measured at — its
    mid-point, T + window/2 sessions."""
    lr_ref = log_returns(ref) if ref is not None and len(ref) else pd.Series(dtype=float)
    rows = []
    for s in specs:
        valid, ratio, rv, n = False, float("nan"), float("nan"), 0
        if s["ticker"] in getattr(closes, "columns", []) and len(lr_ref):
            lr = log_returns(closes[s["ticker"]].dropna())
            j = pd.concat([lr.rename("c"), lr_ref.rename("r")], axis=1, join="inner").dropna()
            j = j.tail(window)
            n = len(j)
            if n >= 40 and (j["c"] != 0).mean() >= 0.7 and j["r"].std() > 0:
                ratio = float(j["c"].std() / j["r"].std())
                rv = float(j["c"].std() * math.sqrt(252))
                valid = 0.2 < ratio < 3.0
        rows.append(dict(label=s["label"], ticker=s["ticker"], T=s["T"],
                         tau=s["T"] + window / 2 / 252, ratio=ratio, rv=rv, n=n, valid=valid))
    return pd.DataFrame(rows)


def series_vol_ratio(a: pd.Series, b: pd.Series, window: int = VOL_WINDOW) -> Optional[float]:
    """Realised vol of a over realised vol of b, on their common recent sessions."""
    if a is None or b is None or not len(a) or not len(b):
        return None
    j = pd.concat([log_returns(a).rename("a"), log_returns(b).rename("b")],
                  axis=1, join="inner").dropna().tail(window)
    if len(j) < 40 or j["b"].std() == 0:
        return None
    return float(j["a"].std() / j["b"].std())


def samuelson_fit(taus: Sequence[float], ratios: Sequence[float]) -> dict:
    """Fit g(τ) = c·(a + (1 − a)·e^(−bτ)) to the measured ratios.

    The Samuelson effect: a future is calm far from delivery and grows more
    volatile as it nears it. c is the level near expiry, c·a the far level, b how
    fast it decays. With fewer than three good months, a flat median ratio is used;
    with none, g ≡ 1 — and the method field says so."""
    t = np.asarray(taus, float)
    y = np.asarray(ratios, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) == 0:
        return dict(c=1.0, a=1.0, b=1.0, n=0, method="none — g ≡ 1")
    if len(y) < 3:
        return dict(c=float(np.median(y)), a=1.0, b=1.0, n=int(len(y)), method="flat median")

    def resid(p):
        c, a, b = p
        return c * (a + (1 - a) * np.exp(-b * t)) - y

    try:
        s = least_squares(resid, [float(y[np.argmin(t)]), 0.7, 1.5],
                          bounds=([0.3, 0.05, 0.05], [2.5, 1.0, 12.0]))
        c, a, b = (float(v) for v in s.x)
        return dict(c=c, a=a, b=b, n=int(len(y)), method="Samuelson fit",
                    rmse=float(np.sqrt(np.mean(s.fun ** 2))))
    except Exception:                                              # noqa: BLE001
        return dict(c=float(np.median(y)), a=1.0, b=1.0, n=int(len(y)), method="flat median")


def samuelson_g(g: dict, tau: float) -> float:
    return g["c"] * (g["a"] + (1 - g["a"]) * math.exp(-g["b"] * max(tau, 0.0)))


def samuelson_scale(g: dict, T_opt: float, T_fut: float) -> float:
    """RMS of g over the option's life. Over the option's life the future's time
    to delivery runs from T_fut down to T_fut − T_opt, and its volatility follows
    g along the way, so the right scale is √(∫g²dτ / T_opt) over that interval —
    closed form below."""
    c, a, b = g["c"], g["a"], g["b"]
    t1 = max(float(T_fut), 0.0)
    t0 = max(t1 - max(float(T_opt), 0.0), 0.0)
    if t1 - t0 < 1e-9:
        return samuelson_g(g, t1)

    def I(t):
        return (a * a * t - 2 * a * (1 - a) * math.exp(-b * t) / b
                - (1 - a) ** 2 * math.exp(-2 * b * t) / (2 * b))

    return c * math.sqrt(max((I(t1) - I(t0)) / (t1 - t0), 1e-12))


# ── The hierarchy ────────────────────────────────────────────────────────────
def build_vol_surface(name: str, r: float, today: date, closes: pd.DataFrame,
                      history: pd.Series, chain_loader: Callable[[str], dict],
                      etf_history: Callable[[str], pd.Series]) -> dict:
    """Implied > forecast > registry, resolved for one contract. Pure given its
    loaders, so the test suite drives it with synthetic chains.

    Returns plain data (cacheable): the source, a label and a one-line detail
    for the header, per-expiry fits, the GARCH forecast, the Samuelson curve and
    the per-month ratios behind it."""
    cfg = CONTRACTS[name]
    specs = strip_specs(name, today)
    warnings: List[str] = []
    garch = garch_fit(log_returns(history).to_numpy()) if history is not None and len(history) else None
    base = dict(name=name, r=r, garch=garch, expiries=[], warnings=warnings,
                registry=cfg["vol"], proxy=None, kind=None, asof=str(today),
                transient=False, garch_ratio=1.0)

    def _garch_ratio(ref):
        """GARCH runs on the continuous front series; the Samuelson curve is
        measured against `ref`. Divide by this to put GARCH in ref's units."""
        x = series_vol_ratio(history, ref) if garch else None
        return float(np.clip(x, 0.5, 2.0)) if x else 1.0

    # 1. IMPLIED — the first proxy whose options give at least one usable expiry.
    for etf, kind in cfg["proxies"]:
        data = chain_loader(etf) or {}
        if data.get("error") and not data.get("chains"):
            warnings.append(f"{etf}: {data['error']}")
            base["transient"] = True
            continue
        fits = []
        for ch in data.get("chains", []):
            try:
                ed = datetime.strptime(ch["expiry"], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            T = year_frac(ed, today)
            if T <= 0:
                continue
            F_etf, pts = chain_implied_vols(ch["calls"], ch["puts"], T, r, data.get("spot"), today)
            fit = fit_expiry_smile(F_etf, T, pts) if F_etf else None
            if fit:
                fit["expiry"] = ch["expiry"]
                fit["points"] = pts.to_dict("records")
                fits.append(fit)
        if not fits:
            warnings.append(f"{etf}: no expiry produced a usable ATM vol")
            continue
        clean = [f for f in fits if f["quality"] != "noisy"]
        for f in fits:
            f["used"] = (f["quality"] != "noisy") or not clean
        if clean and len(clean) < len(fits):
            warnings.append(f"{etf}: {len(fits) - len(clean)} expiry(ies) left out of the term "
                            f"structure — quotes too noisy to fit a smile within 3 vols")
        ref = etf_history(etf)
        ratios = contract_vol_ratios(closes, specs, ref)
        good = ratios[ratios["valid"]] if not ratios.empty else ratios
        if kind == "cross" and good.empty:
            warnings.append(f"{etf}: cross proxy needs this contract's own history to scale "
                            f"the level, and none was usable")
            continue
        g = samuelson_fit(good["tau"], good["ratio"]) if not good.empty else \
            samuelson_fit([], [])
        if good.empty:
            warnings.append("no futures month had enough clean history to map the fund's "
                            "vol onto the strip — fund vol used as is")
        n_quotes = sum(f["n"] for f in fits)
        n_smile = sum(f["has_smile"] for f in fits)
        what = "options" if kind == "direct" else "options, cross-mapped"
        detail = (f"implied from {etf} listed options — {len(fits)} expiries, {n_quotes} quotes, "
                  f"{n_smile} with a fitted smile — mapped to each month by realised ratio "
                  f"({g['method']}, {g['n']} months)")
        if kind == "cross":
            detail = (f"no listed options on this contract: implied from {etf} options and "
                      f"rescaled by this contract's realised vol relative to {etf} "
                      f"({g['method']}, {g['n']} months)")
        base["transient"] = False
        return dict(base, source="implied", proxy=etf, kind=kind, garch_ratio=_garch_ratio(ref),
                    label=f"IMPLIED · {etf} {what}", detail=detail,
                    expiries=sorted(fits, key=lambda f: f["T"]), g=g,
                    ratios=ratios.to_dict("records"), spot=data.get("spot"))

    # 2. FORECAST — GARCH on the front, mapped by the strip's own ratios.
    if garch:
        front = None
        for s in specs:
            if s["ticker"] in getattr(closes, "columns", []) and closes[s["ticker"]].notna().sum() > 60:
                front = closes[s["ticker"]].dropna()
                break
        ratios = contract_vol_ratios(closes, specs, front) if front is not None else pd.DataFrame()
        good = ratios[ratios["valid"]] if not ratios.empty else ratios
        g = samuelson_fit(good["tau"], good["ratio"]) if not good.empty else samuelson_fit([], [])
        why = "no listed options on this contract or a related fund" if not cfg["proxies"] \
            else "the options feed returned nothing usable (" + "; ".join(warnings) + ")"
        return dict(base, source="forecast", garch_ratio=_garch_ratio(front),
                    label="FORECAST · GARCH(1,1)",
                    detail=(f"{why}: GARCH(1,1) on {garch['n']} sessions of front-month history "
                            f"(now {garch['vol_now']:.1%}, long-run {garch['vol_lr']:.1%}, half-life "
                            f"{garch['half_life']:.0f} sessions), mapped by realised ratio "
                            f"({g['method']}). Flat smile: nothing to measure one from."),
                    g=g, ratios=ratios.to_dict("records") if not ratios.empty else [])

    # 3. REGISTRY — nothing reachable.
    return dict(base, source="registry", label="REGISTRY DEFAULT",
                detail="no option prices and no price history reachable — a constant from "
                       "this file. Do not trade off it.",
                g=samuelson_fit([], []), ratios=[])


def vol_surface(name: str, r: float) -> dict:
    """The live surface for one contract. Cached for 15 minutes — unless it had
    to fall back because an options feed failed, in which case it is rebuilt on
    the next run (cheap: its inputs are cached) and upgrades itself to IMPLIED
    as soon as the feed answers."""
    try:
        return _vol_surface_cached(name, round(float(r), 6))
    except FeedUnavailable as e:
        return e.args[0]


@st.cache_data(ttl=900, show_spinner=False)
def _vol_surface_cached(name: str, r: float) -> dict:
    surf = build_vol_surface(name, r, date.today(), fetch_strip_closes(name),
                             fetch_history(name, "5y"), fetch_option_chains,
                             lambda etf: fetch_etf_history(etf, "1y"))
    if surf.get("transient"):
        raise FeedUnavailable(surf)
    return surf


def surface_ref_atm(surf: dict, T_opt: float) -> float:
    """The reference ATM vol before mapping: the fund's implied term structure,
    or the GARCH forecast, or the registry constant."""
    T_opt = max(float(T_opt), 1 / 365)
    if surf["source"] == "implied" and surf["expiries"]:
        ex = [e for e in surf["expiries"] if e.get("used", True)] or surf["expiries"]
        return interp_total_variance([e["T"] for e in ex], [e["atm"] for e in ex], T_opt)
    if surf["source"] == "forecast" and surf.get("garch"):
        return garch_term_vol(surf["garch"], T_opt) / surf.get("garch_ratio", 1.0)
    return float(surf["registry"])


def surface_forecast(surf: dict, T_opt: float, T_fut: Optional[float] = None) -> Optional[float]:
    """GARCH forecast of REALISED vol for one futures month over an option's life,
    in the same units as the implied ATM — for the implied-minus-forecast gap."""
    g = surf.get("garch")
    if not g:
        return None
    T_opt = max(float(T_opt), 1 / 365)
    return (garch_term_vol(g, T_opt) / surf.get("garch_ratio", 1.0)
            * samuelson_scale(surf["g"], T_opt, T_fut if T_fut else T_opt))


def surface_atm(surf: dict, T_opt: float, T_fut: Optional[float] = None) -> float:
    """ATM vol for an option of life T_opt on the future delivering at T_fut."""
    T_opt = max(float(T_opt), 1 / 365)
    T_fut = float(T_fut) if T_fut else T_opt
    ref = surface_ref_atm(surf, T_opt)
    if surf["source"] == "registry":
        return ref
    return float(min(max(ref * samuelson_scale(surf["g"], T_opt, T_fut), VOL_FLOOR), VOL_CAP))


def surface_smile(surf: dict, T_opt: float) -> Optional[Tuple[float, float]]:
    """(ρ, ν) at a tenor, linear in T between fitted expiries, flat beyond.
    None where nothing was fitted — the surface is then flat, and says so."""
    fits = [e for e in surf.get("expiries", []) if e.get("has_smile")]
    if not fits:
        return None
    Ts = [e["T"] for e in fits]
    return (float(np.interp(T_opt, Ts, [e["rho"] for e in fits])),
            float(np.interp(T_opt, Ts, [e["nu"] for e in fits])))


def surface_vol_fn(surf: dict, F: float, T_fut: Optional[float] = None) -> Callable:
    """σ(K, T) for one futures month: ATM from the surface, smile from SABR with
    α re-solved at THIS forward so the ATM is reproduced exactly. SABR pinned
    this way is invariant to the price level, so a fund's smile in moneyness
    carries straight over to the future's."""
    cache: Dict[float, tuple] = {}

    def f(K, T):
        T = max(float(T), 1 / 365)
        key = round(T, 7)
        if key not in cache:
            atm = surface_atm(surf, T, T_fut if T_fut else T)
            sm = surface_smile(surf, T)
            if sm is None:
                cache[key] = (atm, None)
            else:
                rho, nu = sm
                cache[key] = (atm, (sabr_alpha_from_atm(F, T, atm, SABR_BETA, rho, nu), rho, nu))
        atm, p = cache[key]
        if p is None:
            return atm
        v = sabr_vol(F, max(float(K), 1e-9), T, p[0], SABR_BETA, p[1], p[2])
        if not np.isfinite(v):
            v = atm
        return float(min(max(v, VOL_FLOOR), VOL_CAP))

    f.smile = surface_smile(surf, 0.25) is not None        # type: ignore[attr-defined]
    return f


def as_vol_fn(v):
    """Normalise a volatility input into a callable σ(K, T): a number gives a flat
    surface (a MODEL choice, used by the what-if tools and the tests), a function
    is used as is."""
    if callable(v):
        return v
    x = float(v)
    return lambda K, T, _x=x: _x


def shifted_vol_fn(vol_fn, shift: float):
    """Parallel shift of a whole surface in vol points — what scenarios move."""
    base = as_vol_fn(vol_fn)
    return lambda K, T: max(base(K, T) + shift, 0.005)


def correlation_from_closes(closes: pd.DataFrame, a: str, b: str,
                            window: int = 120) -> Optional[float]:
    """Correlation of daily log returns of two tickers over their common sessions.
    None when fewer than 40 overlap — the caller then says it used a default."""
    if closes is None or a not in getattr(closes, "columns", []) or b not in closes.columns:
        return None
    j = pd.concat([log_returns(closes[a].dropna()).rename("a"),
                   log_returns(closes[b].dropna()).rename("b")], axis=1, join="inner").dropna()
    j = j.tail(window)
    if len(j) < 40 or j["a"].std() == 0 or j["b"].std() == 0:
        return None
    return float(np.clip(j["a"].corr(j["b"]), -0.999, 0.999))


def leg_returns(name: str, T: float) -> pd.Series:
    """Daily log returns of the listed month closest to tenor T (no roll jumps);
    the continuous series if the strip has nothing usable."""
    closes = fetch_strip_closes(name)
    specs = strip_specs(name)
    for s in sorted(specs, key=lambda s: abs(s["T"] - T)):
        if s["ticker"] in getattr(closes, "columns", []):
            ser = closes[s["ticker"]].dropna()
            if len(ser) > 60 and (log_returns(ser) != 0).mean() > 0.7:
                return log_returns(ser)
    hist = fetch_history(name, "1y")
    return log_returns(hist) if len(hist) else pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  5. STRATEGIES, BOOK AND SCENARIOS
#  Nobody trades one option. A desk trades structures, holds a book of them, and
#  looks at the whole thing across a grid of price and volatility.
# ══════════════════════════════════════════════════════════════════════════════

#  Each template is a list of (quantity, type, strike spec). A strike spec is
#  either ("pct", x) — percent of the forward — or ("delta", d), resolved through
#  the delta solver so the structure is built the way it is quoted.
STRATEGIES: Dict[str, dict] = {
    "Long call": dict(legs=[(+1, "call", ("pct", 100))],
                      note="The simplest long: unlimited upside, premium at risk."),
    "Long put": dict(legs=[(+1, "put", ("pct", 100))],
                     note="Downside protection with a known maximum cost."),
    "Short call": dict(legs=[(-1, "call", ("pct", 105))],
                       note="Premium income against a view that the market stalls. "
                            "Unlimited loss above the strike."),
    "Short put": dict(legs=[(-1, "put", ("pct", 95))],
                      note="Paid to be long lower. The classic way a producer "
                           "monetises a floor they do not think will be hit."),
    "Straddle": dict(legs=[(+1, "call", ("pct", 100)), (+1, "put", ("pct", 100))],
                     note="Long volatility, direction-neutral. Pays if the market "
                          "moves more than the premium in either direction."),
    "Strangle": dict(legs=[(+1, "call", ("delta", 0.25)), (+1, "put", ("delta", 0.25))],
                     note="A cheaper straddle built on the wings — needs a bigger "
                          "move but costs materially less."),
    "Call spread": dict(legs=[(+1, "call", ("pct", 100)), (-1, "call", ("pct", 110))],
                        note="Upside with a cap, and a fraction of the premium. "
                             "The workhorse of a directional options book."),
    "Put spread": dict(legs=[(+1, "put", ("pct", 100)), (-1, "put", ("pct", 90))],
                       note="Protection down to a level, then nothing — a cheaper "
                            "hedge when the tail is not the worry."),
    "Risk reversal (bullish)": dict(
        legs=[(+1, "call", ("delta", 0.25)), (-1, "put", ("delta", 0.25))],
        note="Long the 25-delta call funded by the 25-delta put. Near zero cost, "
             "and its price IS the skew — which is why the desk quotes it in vols."),
    "Butterfly": dict(legs=[(+1, "call", ("pct", 90)), (-2, "call", ("pct", 100)),
                            (+1, "call", ("pct", 110))],
                      note="A bet that the market pins the middle strike. Long the "
                           "wings, short the body: short volatility, defined risk."),
    "Producer collar (fence)": dict(
        legs=[(+1, "put", ("delta", 0.25)), (-1, "call", ("delta", 0.25))],
        note="THE commodity hedge: a producer buys a floor and sells a ceiling to "
             "pay for it. Often struck for zero premium — which is exactly what "
             "the delta-strike solver is for."),
    "Consumer collar": dict(
        legs=[(+1, "call", ("delta", 0.25)), (-1, "put", ("delta", 0.25))],
        note="The mirror: a consumer caps their cost and sells a floor to fund it. "
             "An airline's fuel programme in one line."),
    "3-way producer": dict(
        legs=[(+1, "put", ("delta", 0.35)), (-1, "call", ("delta", 0.25)),
              (-1, "put", ("delta", 0.15))],
        note="A collar with the deep put sold back. Cheaper still, and the "
             "producer is unprotected below the lowest strike — the structure "
             "that hurt a lot of shale in 2020."),
    "Iron condor": dict(
        legs=[(-1, "call", ("delta", 0.25)), (+1, "call", ("delta", 0.10)),
              (-1, "put", ("delta", 0.25)), (+1, "put", ("delta", 0.10))],
        note="Short both wings with the tails bought back. Income while the market "
             "stays in a range, defined loss when it does not."),
}


def resolve_strike(spec, F, T, r, sigma, option_type) -> float:
    kind, val = spec
    if kind == "pct":
        return float(F * val / 100.0)
    return strike_from_delta(F, T, r, sigma, val, option_type)


def build_strategy(name: str, F, T, r, sigma, lots=1, american=False,
                   n_steps=200, contract=None) -> dict:
    """Price a template and return its legs, net Greeks and payoff function.

    `sigma` may be a number or a σ(K,T) callable. Each leg is priced at ITS OWN
    volatility, which is what makes a risk reversal cost something: with a flat
    surface the 25-delta call and put carry the same vol and the structure is
    free, which is never true in a real market. Strikes are snapped onto the
    contract's listed grid when a contract is given.
    """
    cfg = STRATEGIES[name]
    vf = as_vol_fn(sigma)
    atm = vf(F, T)
    legs = []
    for qty, otype, spec in cfg["legs"]:
        K = resolve_strike(spec, F, T, r, atm, otype)
        if contract:
            K = snap_strike(contract, K)
        sig_k = vf(K, T)
        b = Black76(F, K, T, r, sig_k, otype)
        if american:
            g = american_greeks(F, K, T, r, sig_k, otype, n_steps)
        else:
            g = b.greeks()
            g["early_premium"] = 0.0
        legs.append(dict(qty=qty * lots, type=otype, strike=K, vol=sig_k,
                         spec=f"{spec[0]} {spec[1]}", **g))
    net = {k: float(sum(l["qty"] * l[k] for l in legs))
           for k in ("price", "delta", "gamma", "vega", "theta")}
    return dict(name=name, note=cfg["note"], legs=legs, net=net,
                american=american)


def strategy_payoff(legs: List[dict], spots: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(spots, dtype=float))
    for l in legs:
        pay = (np.maximum(spots - l["strike"], 0.0) if l["type"] == "call"
               else np.maximum(l["strike"] - spots, 0.0))
        out += l["qty"] * pay
    return out


# ── The book ─────────────────────────────────────────────────────────────────
#  A line is a dict that remembers WHAT it is — contract, delivery month, option
#  expiry date, strike, type, style, quantity, entry premium — and nothing about
#  the market. Every revaluation looks the market up again for that line's own
#  contract and month, so a WTI Dec-27 call stays a WTI Dec-27 call whatever the
#  sidebar is showing, and its tenor shortens every day on its own.
#
#  The market a line is valued against is a small dict:
#      F      forward of that contract's delivery month
#      T      option life remaining (years); ≤ 0 means expired → intrinsic
#      fn     σ(K, T) for that month, off that contract's automatic surface

BOOK_PATH = Path(os.environ.get("CODAP_BOOK", str(Path.home() / ".codap" / "book.json")))


def load_book(path: Path = BOOK_PATH) -> List[dict]:
    try:
        data = json.loads(Path(path).read_text())
        return [migrate_line(x) for x in data if isinstance(x, dict)]
    except FileNotFoundError:
        return []
    except Exception as e:                                         # noqa: BLE001
        LOG.warning("could not read book %s: %s", path, e)
        return []


def save_book(book: List[dict], path: Path = BOOK_PATH) -> bool:
    """Atomic write: a crash mid-save leaves the previous book, not half a file."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(book, indent=2, default=str))
        tmp.replace(path)
        return True
    except Exception as e:                                         # noqa: BLE001
        LOG.warning("could not save book %s: %s", path, e)
        return False


def migrate_line(line: dict) -> dict:
    """Bring a line from an older export up to date. The previous version stored
    only a tenor `T` frozen at booking time; it is converted to an expiry date as
    of today and flagged, since the original booking date is unknown."""
    line = dict(line)
    line.setdefault("id", f"{datetime.now():%Y%m%d%H%M%S%f}-{abs(hash(str(line))) % 10_000}")
    line.setdefault("style", "European")
    if line.get("kind") == "option" and not line.get("opt_expiry"):
        T = float(line.get("T", 0.25))
        line["opt_expiry"] = str(date.today() + timedelta(days=int(round(T * 365.25))))
        line["legacy"] = True
    if not line.get("fut_expiry") and line.get("opt_expiry"):
        line["fut_expiry"] = line["opt_expiry"]
    return line


def make_line(kind: str, contract: str, spec: dict, qty: int, entry: float,
              otype: str = "call", strike: float = 0.0, style: str = "European",
              tag: str = "") -> dict:
    """A book line from a strip spec (one row of strip_specs / the curve)."""
    return dict(
        id=f"{datetime.now():%Y%m%d%H%M%S%f}",
        kind=kind, contract=contract, type=otype, strike=float(strike), qty=int(qty),
        entry=float(entry), style=style, tag=tag,
        label=spec["label"], delivery=spec["delivery"],
        opt_expiry=str(spec["opt_expiry"]), fut_expiry=str(spec["expiry"]),
        booked=str(date.today()))


def _to_date(x) -> Optional[date]:
    if isinstance(x, date):
        return x
    try:
        return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def line_tenors(pos: dict, today: Optional[date] = None) -> Tuple[float, float]:
    """(option life left, future life left) in years, from the stored dates."""
    today = today or date.today()
    oe, fe = _to_date(pos.get("opt_expiry")), _to_date(pos.get("fut_expiry"))
    T_opt = year_frac(oe, today) if oe else float(pos.get("T", 0.0))
    T_fut = year_frac(fe, today) if fe else T_opt
    return T_opt, T_fut


def line_label(p: dict) -> str:
    mon = p.get("label", "")
    if p["kind"] == "future":
        return f"{p['contract'].split('(')[-1].rstrip(')')} {mon} future × {p['qty']:+d}"
    return (f"{p['contract'].split('(')[-1].rstrip(')')} {mon} {p['type']} "
            f"{p['strike']:,.4g} {p.get('style', 'European')[0]} × {p['qty']:+d}")


def option_unit_price(pos: dict, F: float, T: float, r: float, sigma: float,
                      n_steps: int = 120) -> float:
    if T <= 0:
        return max(F - pos["strike"], 0.0) if pos["type"] == "call" else max(pos["strike"] - F, 0.0)
    if pos.get("style") == "American":
        return crr_price(F, pos["strike"], T, r, sigma, pos["type"], n_steps)
    return Black76(F, pos["strike"], T, r, sigma, pos["type"]).price()


def line_value(pos: dict, mkt: dict, r: float, dF_pct: float = 0.0, dvol: float = 0.0,
               n_steps: int = 120) -> float:
    """Cash P&L of one line against its entry, with an optional shock: the forward
    moved by dF_pct percent and the whole surface by dvol (in vol units, 0.05 =
    five points)."""
    F = mkt["F"] * (1 + dF_pct / 100.0)
    mult = price_multiplier(pos["contract"])
    if pos["kind"] == "future":
        return pos["qty"] * (F - pos["entry"]) * mult
    T = mkt["T"]
    sig = max(as_vol_fn(mkt["fn"])(pos["strike"], max(T, 1e-6)) + dvol, 0.005)
    return pos["qty"] * (option_unit_price(pos, F, T, r, sig, n_steps) - pos["entry"]) * mult


def book_greeks(lines: List[Tuple[dict, dict]], r: float, n_steps: int = 150) -> dict:
    """Cash Greeks for the whole book, and vega bucketed by tenor.

    Units: delta in dollars of futures-equivalent exposure; gamma as the change
    in that dollar delta for a 1% move; vega in dollars per vol point; theta in
    dollars per calendar day. They add across contracts because they are dollars.
    The buckets are the point: a book is never simply "long vega".
    """
    buckets = {"0-3M": 0.0, "3-6M": 0.0, "6-12M": 0.0, "12M+": 0.0}
    tot = dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, value=0.0, premium=0.0)
    rows = []
    for p, m in lines:
        mult = price_multiplier(p["contract"])
        F, T = m["F"], m["T"]
        status = "expired" if (p["kind"] == "option" and T <= 0) else "live"
        if p["kind"] == "future":
            g = dict(price=F, delta=1.0, gamma=0.0, vega=0.0, theta=0.0)
            sig = float("nan")
        elif T <= 0:
            px = option_unit_price(p, F, 0.0, r, 0.0)
            itm = px > 0
            g = dict(price=px, delta=(1.0 if p["type"] == "call" else -1.0) * itm,
                     gamma=0.0, vega=0.0, theta=0.0)
            sig = float("nan")
        else:
            sig = as_vol_fn(m["fn"])(p["strike"], T)
            if p.get("style") == "American":
                g = american_greeks(F, p["strike"], T, r, sig, p["type"], n_steps)
            else:
                g = Black76(F, p["strike"], T, r, sig, p["type"]).greeks()
        q = p["qty"]
        d_cash = q * g["delta"] * mult * F
        ga_cash = q * g["gamma"] * mult * F ** 2 / 100
        v_cash = q * g["vega"] * mult
        t_cash = q * g["theta"] * mult
        val = q * ((g["price"] if p["kind"] == "option" else F) - p["entry"]) * mult
        tot["delta"] += d_cash
        tot["gamma"] += ga_cash
        tot["vega"] += v_cash
        tot["theta"] += t_cash
        tot["value"] += val
        if p["kind"] == "option":
            tot["premium"] += q * g["price"] * mult
        Tb = max(T, 0.0)
        b = ("0-3M" if Tb <= 0.25 else "3-6M" if Tb <= 0.5 else
             "6-12M" if Tb <= 1.0 else "12M+")
        buckets[b] += v_cash
        rows.append(dict(Line=line_label(p), Status=status, F=F, Days=max(T, 0) * 365.25,
                         Vol=sig, Price=g["price"], Entry=p["entry"], DeltaCash=d_cash,
                         GammaCash=ga_cash, VegaCash=v_cash, ThetaCash=t_cash, MTM=val,
                         Source=m.get("source", "")))
    return dict(total=tot, buckets=buckets, rows=rows)


def scenario_matrix(lines: List[Tuple[dict, dict]], r: float,
                    price_pct=(-20, -10, -5, 0, 5, 10, 20),
                    vol_pts=(-10, -5, 0, 5, 10), n_steps: int = 100) -> pd.DataFrame:
    """Book P&L across a grid of price and volatility moves. Every forward moves
    by the same percentage and every surface by the same number of points — a
    uniform shock, stated, which is what a single-desk grid conventionally shows."""
    base = sum(line_value(p, m, r, 0, 0, n_steps) for p, m in lines)
    data = {}
    for dv in vol_pts:
        data[f"{dv:+d} vols"] = {
            f"{dp:+d}%": sum(line_value(p, m, r, dp, dv / 100.0, n_steps) for p, m in lines) - base
            for dp in price_pct}
    return pd.DataFrame(data).T


def span_margin(lines: List[Tuple[dict, dict]], r: float, scan_pct: float = 5.0,
                vol_pts: float = 10.0, n_steps: int = 100) -> dict:
    """SPAN-style scan risk: the worst loss across sixteen scenarios — the price
    range in thirds up and down, each with vol up and down, plus two extreme moves
    at twice the range counted at 35%. A PROXY: real SPAN adds inter-month and
    inter-commodity credits and exchange-set ranges per contract."""
    base = sum(line_value(p, m, r, 0, 0, n_steps) for p, m in lines)
    scen = [(f * scan_pct, dv, 1.0) for f in (0.0, 1 / 3, -1 / 3, 2 / 3, -2 / 3, 1.0, -1.0)
            for dv in (vol_pts, -vol_pts)]
    scen += [(2 * scan_pct, 0.0, 0.35), (-2 * scan_pct, 0.0, 0.35)]
    worst, detail = 0.0, []
    for dp, dv, w in scen:
        pnl = sum(line_value(p, m, r, dp, dv / 100.0, n_steps) for p, m in lines) - base
        loss = -pnl * w
        detail.append(dict(price=f"{dp:+.2f}%", vol=f"{dv:+.0f}", weight=w, pnl=pnl,
                           weighted_loss=loss))
        worst = max(worst, loss)
    return dict(margin=worst, scenarios=pd.DataFrame(detail), base=base,
                scan_pct=scan_pct, vol_pts=vol_pts)


# ══════════════════════════════════════════════════════════════════════════════
#  6. SELF-VALIDATION
#  Every model here is checked against something computed a different way, and
#  the residual is put on screen. A pricer that cannot show its own error is a
#  pricer you have to take on faith — and these four checks are exactly the ones
#  that caught the errors this version fixes.
# ══════════════════════════════════════════════════════════════════════════════

def validate_models(F, K, T, r, sigma, n_obs=12, quick=True,
                    vol_fn=None, atm=None) -> List[dict]:
    """Run the independent checks and return one row per check.

    `quick` trades Monte Carlo precision for speed on page load; the tolerances
    scale with it, so a PASS means the same thing either way.
    """
    rows: List[dict] = []
    n_mc = 60_000 if quick else 400_000

    # 1. Put-call parity — analytic identity, must hold to machine precision.
    err = put_call_parity_error(F, K, T, r, sigma)
    rows.append(dict(check="Put-call parity", detail="C − P = e^(−rT)(F − K)",
                     error=err, tol=1e-8,
                     independent="closed-form identity"))

    # 2. Theta against a finite difference of the price itself.
    b = Black76(F, K, T, r, sigma, "call")
    h = 1 / 365
    fd = Black76(F, K, T - h, r, sigma, "call").price() - b.price()
    rows.append(dict(check="Theta (1 day)", detail=f"analytic {b.theta():+.6f} vs "
                                                   f"finite difference {fd:+.6f}",
                     error=abs(b.theta() - fd), tol=max(abs(fd) * 0.02, 1e-6),
                     independent="finite difference"))

    # 3. Kemna-Vorst against Monte Carlo on the same observation grid.
    kv = kemna_vorst(F, K, T, r, sigma, n_obs, "call")
    mcg = asian_mc(F, K, T, r, sigma, n_obs, "call", average="geometric",
                   n_paths=n_mc, seed=3, control_variate=False)
    rows.append(dict(check="Geometric Asian",
                     detail=f"analytic {kv['price']:.4f} vs MC {mcg['price']:.4f}",
                     error=abs(kv["price"] - mcg["price"]),
                     tol=3 * max(mcg["std_error"], 1e-9),
                     independent="Monte Carlo (3σ band)"))

    # 4. Barrier knock-in / knock-out identity: KI + KO = vanilla, zero rebate.
    H = F * 0.85
    ki = barrier_analytic(F, K, H, T, r, sigma, "call", "Down-and-In")
    ko = barrier_analytic(F, K, H, T, r, sigma, "call", "Down-and-Out")
    van = Black76(F, K, T, r, sigma, "call").price()
    rows.append(dict(check="Barrier KI + KO", detail=f"{ki:.4f} + {ko:.4f} = {ki + ko:.4f} "
                                                     f"vs vanilla {van:.4f}",
                     error=abs(ki + ko - van), tol=1e-8,
                     independent="closed-form identity"))
    # 5. The binomial tree must reproduce Black-76 when early exercise is off.
    eur_tree = crr_price(F, K, T, r, sigma, "call", 1200, american=False)
    van = Black76(F, K, T, r, sigma, "call").price()
    rows.append(dict(check="CRR tree → Black-76",
                     detail=f"European tree {eur_tree:.5f} vs closed form {van:.5f}",
                     error=abs(eur_tree - van), tol=max(van * 0.001, 1e-4),
                     independent="closed form (1200 steps)"))

    # 6. American must be worth at least European, and at least intrinsic.
    amer = crr_price(F, K, T, r, sigma, "call", 600, american=True)
    slack = min(amer - van, amer - max(F - K, 0.0))
    rows.append(dict(check="American ≥ European",
                     detail=f"American {amer:.5f}, European {van:.5f}, "
                            f"early-exercise value {amer - van:+.5f}",
                     error=max(-slack, 0.0), tol=1e-9,
                     independent="no-arbitrage bound"))

    # 7. SABR must reproduce the ATM volatility it was pinned to.
    a = sabr_alpha_from_atm(F, T, sigma, 0.5, -0.3, 0.8)
    rows.append(dict(check="SABR ATM pinning",
                     detail=f"target {sigma:.4f} vs model "
                            f"{sabr_vol(F, F, T, a, 0.5, -0.3, 0.8):.4f}",
                     error=abs(sabr_vol(F, F, T, a, 0.5, -0.3, 0.8) - sigma), tol=1e-6,
                     independent="analytic inversion"))

    # 8. Delta-strike solver must round-trip through the Black-76 delta.
    kc = strike_from_delta(F, T, r, sigma, 0.25, "call")
    got = Black76(F, kc, T, r, sigma, "call").delta()
    rows.append(dict(check="25Δ strike solver",
                     detail=f"asked 0.2500, strike {kc:,.3f} prices to Δ {got:.4f}",
                     error=abs(got - 0.25), tol=1e-6, independent="forward recomputation"))

    # 9. Implied-vol inversion must return the vol a price was made with — the
    #    whole automatic surface rests on this.
    errs = []
    for kk in (0.8 * F, F, 1.25 * F):
        for typ in ("call", "put"):
            px = Black76(F, kk, T, r, sigma, typ).price()
            errs.append(abs(Black76(F, kk, T, r, 0.5, typ).implied_vol(px) - sigma))
    rows.append(dict(check="Implied-vol inversion",
                     detail=f"6 options at 80/100/125% of F, worst error {max(errs):.2e}",
                     error=max(errs), tol=1e-7, independent="price → vol → price"))

    # 10. The live surface must reproduce its own ATM at the selected expiry —
    #     SABR's α is re-solved at the future's forward, and this checks it.
    if vol_fn is not None and atm is not None:
        got = vol_fn(F, T)
        rows.append(dict(check="Surface ATM pinning",
                         detail=f"surface ATM {atm:.4%} vs σ(F) from the smile {got:.4%}",
                         error=abs(got - atm), tol=1e-6, independent="analytic α inversion"))

    for row in rows:
        row["pass"] = bool(row["error"] <= row["tol"])
    return rows



STEP_MULTIPLES = (1, 2, 4, 5, 10, 20, 25, 50, 100, 200)


def default_step_multiple(name: str, F: float) -> int:
    """The listed-increment multiple closest to 1% of the forward — so a 15-line
    ladder spans about ±7% whether the increment is 0.5 on $80 oil or 5 on
    $3,500 gold."""
    inc = CONTRACTS[name]["strike_inc"]
    return min(STEP_MULTIPLES, key=lambda m: abs(math.log(m * inc / (0.01 * F))))


def option_chain(name: str, F: float, T: float, r: float, vol_fn,
                 n_strikes: int = 15, american: bool = False,
                 step_multiple: int = 1) -> pd.DataFrame:
    """The screen an options trader lives on: every listed strike around the money,
    calls on one side, puts on the other.

    Strikes come off the contract's own increment grid rather than a percentage
    sweep, so every row is an option that actually exists. Each is priced at its
    own σ(K) from the active surface.
    """
    vf = as_vol_fn(vol_fn)
    inc = CONTRACTS[name]["strike_inc"] * max(int(step_multiple), 1)
    atm = snap_strike(name, F)
    half = max(int(n_strikes) // 2, 1)
    strikes = [round(atm + i * inc, 10) for i in range(-half, half + 1) if atm + i * inc > 0]
    mult = price_multiplier(name)
    rows = []
    for K in strikes:
        sig = vf(K, T)
        c = Black76(F, K, T, r, sig, "call")
        p = Black76(F, K, T, r, sig, "put")
        cp = crr_price(F, K, T, r, sig, "call", 160) if american else c.price()
        pp = crr_price(F, K, T, r, sig, "put", 160) if american else p.price()
        rows.append(dict(
            call_px=cp, call_cash=cp * mult, call_delta=c.delta(),
            vega=c.vega(), vol=sig, strike=K,
            moneyness=K / F * 100,
            put_delta=p.delta(), put_cash=pp * mult, put_px=pp,
            atm=abs(K - atm) < CONTRACTS[name]["strike_inc"] / 2))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  7. UI FOUNDATION
#  Explanations are everywhere, but folded: one visible line per screen, the
#  detail one click away, and a tooltip on every input. A wall of prose is
#  excellent the first time and an obstacle the thirtieth.
# ══════════════════════════════════════════════════════════════════════════════

def css(dark: bool = True) -> str:
    """The page styling. The dark sidebar is forced only under a dark theme: on a
    light theme it would leave Streamlit's own dark labels unreadable."""
    sidebar = (f'[data-testid="stSidebar"]{{background:linear-gradient(180deg,#0A0E17 0%,{BG} 100%);'
               'border-right:1px solid #1C2333}' if dark else "")
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif}}
code,pre{{font-family:'JetBrains Mono',monospace}}
.block-container{{padding-top:3.2rem;max-width:1500px}}
{sidebar}
[data-testid="stSidebar"] label{{font-size:.70rem;font-weight:500;color:#6E7681;
    text-transform:uppercase;letter-spacing:.07em}}
.stTabs [data-baseweb="tab-list"]{{background:{BG};border-bottom:1px solid #1C2333;gap:3px}}
.stTabs [data-baseweb="tab"]{{font-size:.82rem;font-weight:500;color:#6E7681;padding:9px 16px}}
.stTabs [aria-selected="true"]{{color:inherit;background:{PANEL};border-bottom:2px solid {AMBER}}}
hr{{border:none;border-top:1px solid #1C2333;margin:10px 0}}
.kpi{{background:{PANEL};border:.5px solid {BORDER};border-radius:9px;padding:11px 13px}}
.kpi-l{{font-size:.64rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;
    margin-bottom:4px}}
.kpi-v{{font-family:'JetBrains Mono',monospace;font-size:.95rem;color:{TEXT};white-space:nowrap}}
.kpi-s{{font-size:.66rem;color:#6E7681;margin-top:3px}}
.badge{{display:inline-block;padding:2px 8px;border:1px solid {BORDER};border-radius:5px;
    font-family:'JetBrains Mono',monospace;font-size:.71rem}}
.lead{{font-size:.85rem;color:inherit;opacity:.78;line-height:1.55;margin:2px 0 10px 0}}
.note{{font-size:.80rem;color:#9BA6B2;line-height:1.6;padding:8px 12px;background:{PANEL};
    border-radius:6px;border-left:2px solid {BORDER};margin-bottom:12px}}
.note b{{color:{TEXT}}}
</style>
"""


def kpi(col, label, value, sub="", color=GRAY):
    col.markdown(f'<div class="kpi" style="border-left:3px solid {color}">'
                 f'<div class="kpi-l" style="color:{color}">{label}</div>'
                 f'<div class="kpi-v">{value}</div>'
                 f'<div class="kpi-s">{sub}</div></div>', unsafe_allow_html=True)


def badge(text, color=GRAY) -> str:
    return f'<span class="badge" style="color:{color}">{text}</span>'


def styled(fig, height=320, title=None, legend_below=False):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter", size=11, color=TEXT),
        title=(dict(text=title, x=0.5, xanchor="center",
                    font=dict(size=12, color=GRAY)) if title else None),
        margin=dict(l=55, r=25, t=55 if title else 25, b=95 if legend_below else 45),
        legend=(dict(orientation="h", yanchor="top", y=-0.22, font=dict(size=10)) if legend_below
                else dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10))),
        xaxis=dict(gridcolor="#1C2333", zerolinecolor=BORDER),
        yaxis=dict(gridcolor="#1C2333", zerolinecolor=BORDER))
    return fig


def vline(fig, x, color, text, y=0.96):
    fig.add_vline(x=x, line=dict(color=color, dash="dash", width=1),
                  annotation_text=text, annotation_position="top left",
                  annotation=dict(yref="paper", y=y,
                                  font=dict(color=color, size=10, family="JetBrains Mono"),
                                  bgcolor=BG, bordercolor=color, borderwidth=1))
    return fig


def lead(text: str, detail: str = "", label: str = "How to read this screen"):
    """One visible line, the rest folded. Every screen opens the same way, so a
    returning user's eye can skip it and a first-time user knows where to look."""
    st.markdown(f'<div class="lead">{text}</div>', unsafe_allow_html=True)
    if detail:
        with st.expander(f"ⓘ  {label}"):
            st.markdown(detail)


def section(title: str, note: str = ""):
    st.markdown(f'<div style="border-left:3px solid {AMBER};padding-left:9px;'
                f'margin:16px 0 7px 0"><span style="font-size:.98rem;font-weight:500;'
                f'color:{INK}">{title}</span></div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)



def book_state() -> List[dict]:
    """The book lives on disk and is loaded once per session."""
    if "book" not in st.session_state:
        st.session_state["book"] = load_book()
    return st.session_state["book"]


def add_lines(*lines: dict) -> None:
    book = book_state()
    book.extend(lines)
    if not save_book(book):
        st.warning(f"Booked for this session, but the book could not be saved to {BOOK_PATH}.")


def replace_book(lines: List[dict]) -> None:
    st.session_state["book"] = [migrate_line(x) for x in lines]
    save_book(st.session_state["book"])


def _smap(styler, fn, subset=None):
    """Styler.map on pandas ≥ 2.1, applymap before it."""
    m = getattr(styler, "map", None) or styler.applymap
    return m(fn, subset=subset)


def as_text(df: pd.DataFrame, fmts: Dict[str, str], na: str = "—") -> pd.DataFrame:
    """Pre-format the columns that can hold blanks. Streamlit draws a missing value
    as 'None' whatever the Styler's na_rep says, so a blank that MEANS something
    ("no solution", "negative variance") has to be written in as text."""
    out = df.copy()
    for c, f in fmts.items():
        if c in out:
            out[c] = [na if (v is None or (isinstance(v, float) and not np.isfinite(v)))
                      else f.format(v) for v in out[c]]
    return out


def pnl_cell_style(v) -> str:
    """Green/red cell shading scaled by magnitude — no matplotlib dependency."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(x) or x == 0:
        return ""
    return f"background-color: rgba({'63,185,80' if x > 0 else '255,123,114'},0.22)"


def nav(options: List[str], key: str, default: Optional[str] = None) -> str:
    """Top-level navigation that renders ONLY the screen you are on. Tabs render
    every screen on every interaction — with Monte Carlo behind half of them that
    is the difference between a desk and a waiting room."""
    default = default or options[0]
    kw = {} if key in st.session_state else {"default": default}
    if hasattr(st, "segmented_control"):
        v = st.segmented_control("Screen", options, key=key, label_visibility="collapsed", **kw)
    else:                                                          # pragma: no cover
        v = st.radio("Screen", options, key=key, horizontal=True,
                     label_visibility="collapsed")
    if v is None:                      # a second click on the active segment deselects it
        v = st.session_state.get(key + "__last", default)
    st.session_state[key + "__last"] = v
    return v


def quotes_vs_desk(F: float, T: float, r: float, quotes: pd.DataFrame, vol_fn) -> pd.DataFrame:
    """Your broker's premiums against the desk: each premium inverted to an
    implied vol and compared with the desk's σ(K). Nothing here changes the
    surface — it is a check, not an input."""
    out = []
    for row in quotes.itertuples():
        try:
            K, px, typ = float(row.Strike), float(row.Premium), str(row.Type).strip().lower()
        except (TypeError, ValueError, AttributeError):
            continue
        if K <= 0 or not np.isfinite(px) or typ not in ("call", "put"):
            continue
        iv = Black76(F, K, T, r, 0.3, typ).implied_vol(px)
        desk = vol_fn(K, T)
        desk_px = Black76(F, K, T, r, desk, typ).price()
        out.append(dict(Strike=K, Type=typ, Premium=px, Implied=iv, Desk=desk,
                        Diff_vols=(iv - desk) * 100 if np.isfinite(iv) else np.nan,
                        Desk_premium=desk_px, Diff_premium=px - desk_px))
    return pd.DataFrame(out)


# ══════════════════════════════════════════════════════════════════════════════
#  8. SCREENS
#  One function per screen. Each reads the context built once per run (contract,
#  month, forward, surface) and renders; nothing on a screen asks for a vol.
# ══════════════════════════════════════════════════════════════════════════════

def _vol_badges(cx: dict) -> str:
    vs = cx["vs"]
    col = {"implied": GREEN, "forecast": AMBER, "registry": RED}.get(vs["source"], GRAY)
    return (badge(f"σ ATM {vs['atm'] * 100:.1f}%", PURPLE) + " " + badge(vs["label"], col) + " " +
            badge("SABR smile" if vs["smile"] else "flat smile", PURPLE))


def screen_chain(cx: dict) -> None:
    lead("Every listed strike around the money for the month picked in the sidebar — calls "
         "left, puts right — priced off the automatic volatility surface.",
         """
**The ladder.** Strikes come off the contract's increment grid (0.50 on WTI, 5 on gold, 10
cents on corn), so every line is an option that exists. `σ(K)` is the volatility used for
that row: the surface's ATM at this month's option expiry, bent by the fitted smile.

**Where σ comes from** is in the badge: *IMPLIED* means listed option prices were inverted
for it; *FORECAST* means no options exist anywhere for this market and a GARCH forecast of
realised volatility is used instead. Nothing on this screen is typed in.

**Tenor** is the option's own expiry, not the future's. On metals, grains and softs the
option dies weeks before the future stops trading, and the chain prices that.

**Cash columns** are per lot: `Call $/lot` is what one option actually costs.
""")
    name, F, T, r, unit, spec = cx["name"], cx["F"], cx["T_opt"], cx["r"], cx["unit"], cx["spec"]
    c1, c2, c3 = st.columns([2, 1, 1])
    width = c1.slider("Strikes shown", 5, 41, 15, 2, help="How far either side of the money.")
    inc0 = CONTRACTS[name]["strike_inc"]
    step = c2.select_slider("Strike step", options=list(STEP_MULTIPLES),
                            value=default_step_multiple(name, F), key=f"ch_step_{name}",
                            format_func=lambda m: f"{m * inc0:g}",
                            help="Multiples of the listed increment, so every strike exists. "
                                 "Defaults to about 1% of the forward.")
    amer = c3.radio("Exercise", ["European", "American"], horizontal=True, key="ch_style",
                    help="Most listed commodity options are American. Early exercise is "
                         "priced on a binomial tree.") == "American"

    ch = option_chain(name, F, T, r, cx["vol_fn"], width, amer, step)
    st.markdown(
        badge(f"{spec['label']} @ {F:,.4f} {unit}", INK) + " " +
        badge(f"option expires {spec['opt_expiry']:%d %b %Y} · {T * 365.25:.0f}d", INK) + " " +
        _vol_badges(cx), unsafe_allow_html=True)

    disp = ch[["call_cash", "call_px", "call_delta", "vol", "strike", "moneyness",
               "put_delta", "put_px", "put_cash"]].copy()
    disp.columns = ["Call $/lot", "Call", "Call Δ", "σ(K)", "STRIKE", "% of F",
                    "Put Δ", "Put", "Put $/lot"]
    atm_k = snap_strike(name, F)
    inc = CONTRACTS[name]["strike_inc"]

    def _hl(row):
        near = abs(row["STRIKE"] - atm_k) < inc / 2
        return ["background-color: rgba(240,165,0,0.14)" if near else "" for _ in row]

    st.dataframe(disp.style.format(
        {"Call $/lot": "${:,.0f}", "Call": "{:,.4f}", "Call Δ": "{:+.3f}", "σ(K)": "{:.2%}",
         "STRIKE": "{:,.4f}", "% of F": "{:.1f}%", "Put Δ": "{:+.3f}", "Put": "{:,.4f}",
         "Put $/lot": "${:,.0f}"}).apply(_hl, axis=1),
        **FULL, hide_index=True, height=min(560, 42 + 35 * len(disp)))

    section("Ticket")
    t1, t2, t3, t4, t5 = st.columns([1.6, 1, 1, 1, 2])
    K = t1.selectbox("Strike", ch["strike"].tolist(),
                     index=int(np.argmin(np.abs(ch["strike"] - F))),
                     format_func=lambda k: f"{k:,.4f}")
    otype = t2.radio("Type", ["call", "put"], horizontal=True, key="ck_t")
    sidew = t3.radio("Side", ["Buy", "Sell"], horizontal=True, key="ck_s")
    lots = int(t4.number_input("Lots", 1, 999, 1, key="ck_l"))
    row = ch[ch["strike"] == K].iloc[0]
    px = float(row["call_px"] if otype == "call" else row["put_px"])
    dlt = float(row["call_delta"] if otype == "call" else row["put_delta"])
    mult = price_multiplier(name)
    sgn = 1 if sidew == "Buy" else -1
    t5.markdown(f'<div class="note" style="margin:0">{sidew} {lots} × {spec["label"]} '
                f'{K:,.4f} {otype}<br><b>{px:,.4f}</b> {unit} = <b>${px * mult * lots:,.0f}</b> · '
                f'Δ {dlt * lots * sgn:+.2f} lots · σ {row["vol"]:.2%}</div>',
                unsafe_allow_html=True)
    if st.button("➕  Add to book", type="primary", **FULL):
        add_lines(make_line("option", name, spec, lots * sgn, px, otype, float(K),
                            "American" if amer else "European", "chain"))
        st.success(f"{sidew} {lots} × {spec['label']} {K:,.4f} {otype} booked and saved.")

    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure(go.Scatter(x=ch["strike"], y=ch["vol"] * 100, mode="lines+markers",
                                   line=dict(color=AMBER, width=2.2)))
        vline(fig, F, GREEN, f"F {F:,.2f}")
        fig.update_xaxes(title=f"Strike ({unit})")
        fig.update_yaxes(title="σ (%)")
        st.plotly_chart(styled(fig, 280, "Volatility across the chain"), **FULL)
    with g2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ch["strike"], y=ch["call_delta"], name="Call Δ",
                                 line=dict(color=GREEN, width=2)))
        fig.add_trace(go.Scatter(x=ch["strike"], y=ch["put_delta"], name="Put Δ",
                                 line=dict(color=RED, width=2)))
        for d in (0.25, -0.25):
            fig.add_hline(y=d, line=dict(color=GRAY, dash="dot", width=1))
        vline(fig, F, BLUE, "F")
        fig.update_xaxes(title=f"Strike ({unit})")
        fig.update_yaxes(title="Delta")
        st.plotly_chart(styled(fig, 280, "Delta ladder — dotted lines are the 25s"), **FULL)

    with st.expander("Check a broker quote against the desk"):
        st.caption("Type premiums you see on a screen. Each is inverted to an implied vol and "
                   "compared with the desk's σ(K) — a sanity check on the automatic surface. "
                   "It does not change any price here.")
        seed = pd.DataFrame({"Strike": [float(atm_k - 2 * inc), float(atm_k), float(atm_k + 2 * inc)],
                             "Type": ["put", "call", "call"], "Premium": [np.nan] * 3})
        q = st.data_editor(seed, **FULL, hide_index=True, num_rows="dynamic", key="broker_q",
                           column_config={
                               "Strike": st.column_config.NumberColumn(format="%.4f"),
                               "Type": st.column_config.SelectboxColumn(options=["call", "put"]),
                               "Premium": st.column_config.NumberColumn(format="%.4f")})
        cmp_ = quotes_vs_desk(F, T, r, q.dropna(subset=["Premium"]), cx["vol_fn"])
        if not cmp_.empty:
            cmp_ = as_text(cmp_, {"Implied": "{:.2%}", "Diff_vols": "{:+.2f}"}, "no solution")
            st.dataframe(cmp_.style.format(
                {"Strike": "{:,.4f}", "Premium": "{:,.4f}", "Desk": "{:.2%}",
                 "Desk_premium": "{:,.4f}", "Diff_premium": "{:+,.4f}"}), **FULL, hide_index=True)


def line_market(p: dict, cx: dict) -> Optional[dict]:
    """The market one book line is valued against: ITS contract's curve, ITS
    delivery month, ITS contract's surface. None if that contract has no curve."""
    name = p.get("contract")
    if name not in CONTRACTS:
        return None
    curve = cx["curve"] if name == cx["name"] else fetch_curve(name)
    if curve is None or curve.empty:
        return None
    T_opt, T_fut = line_tenors(p)
    row = curve[curve["delivery"] == p.get("delivery")] if "delivery" in curve else curve.iloc[:0]
    F = float(row["price"].iloc[0]) if len(row) else forward_at(curve, max(T_fut, 0.0))
    if not F or F <= 0:
        return None
    if p["kind"] == "future":
        return dict(F=F, T=T_fut, fn=None, source="curve")
    surf = cx["surface"] if name == cx["name"] else vol_surface(name, cx["r"])
    return dict(F=F, T=T_opt, fn=surface_vol_fn(surf, F, T_fut), source=surf["label"])


def screen_book(cx: dict) -> None:
    lead("Everything you hold, in cash terms, saved to disk — each line repriced off its own "
         "contract, month and volatility surface every time you open it.",
         f"""
**Persistence.** The book is written to `{BOOK_PATH}` on every change and reloaded on start.
Export/Import moves it between machines.

**Each line is its own market.** A WTI Dec-27 call is valued off the WTI Dec-27 settle and
WTI's surface at that option's remaining life — whichever contract the sidebar shows. Its
tenor is recomputed from its expiry date every day, so theta is real. Expired options show
at intrinsic until you remove them.

**Cash Greeks.** Delta is dollars of futures-equivalent exposure; gamma is how much that
moves for a 1% move in the forward; vega is dollars per vol point; theta dollars per day.

**Vega buckets.** A book is never simply "long vega" — it is long the front and short the
back, or the reverse. The buckets show the calendar risk a single number hides.

**Scenario grid and SPAN.** Every forward moves by the same percentage and every surface by
the same number of points. The SPAN figure is the worst of sixteen such scenarios — a proxy
without the exchange's inter-month and inter-commodity credits.
""")
    book = book_state()
    name, curve = cx["name"], cx["curve"]

    with st.expander("➕  Add a line manually", expanded=not book):
        a = st.columns(6)
        kind = a[0].selectbox("Kind", ["option", "future"], key="bk_kind")
        rows = (curve if kind == "future" else curve[curve["option_live"]]).reset_index(drop=True)
        if rows.empty:
            st.caption("No listed month available for this contract.")
            rows = cx["curve"].iloc[[0]].reset_index(drop=True)
        idx = a[1].selectbox("Month", range(len(rows)), key=f"bk_month_{name}",
                             format_func=lambda i: str(rows["label"].iloc[i]))
        spec = rows.iloc[min(idx, len(rows) - 1)].to_dict()
        F_m = float(spec["price"])
        otype = a[2].selectbox("Type", ["call", "put"], key="bk_type", disabled=kind == "future")
        raw_k = a[3].number_input("Strike", value=float(snap_strike(name, F_m)),
                                  step=float(CONTRACTS[name]["strike_inc"]), key="bk_K",
                                  disabled=kind == "future", format="%.4f")
        qty = int(a[4].number_input("Lots (− = short)", -999, 999, 1, key="bk_q"))
        style = a[5].selectbox("Style", ["European", "American"], key="bk_style",
                               disabled=kind == "future")
        K = snap_strike(name, raw_k)
        if kind == "option":
            fn = surface_vol_fn(cx["surface"], F_m, spec["T"])
            fair = Black76(F_m, K, spec["T_opt"], cx["r"], fn(K, spec["T_opt"]), otype).price()
        else:
            fair = F_m
        entry = st.number_input("Entry price (what you actually paid or received)",
                                value=float(round(fair, 6)), format="%.4f", key="bk_e",
                                help="Defaults to the desk's fair value. Mark-to-market is "
                                     "measured against this.")
        if st.button("Add line", **FULL):
            add_lines(make_line(kind, name, spec, qty, entry, otype, K, style, "manual"))
            st.rerun()

    io1, io2, io3 = st.columns([1, 1.3, 1])
    io1.download_button("⬇ Export book", json.dumps(book, indent=2, default=str),
                        file_name="codap_book.json", **FULL)
    up = io2.file_uploader("Import a book", type="json", label_visibility="collapsed")
    if up is not None and io2.button("Load the imported book", **FULL):
        try:
            data = json.loads(up.read().decode())
            assert isinstance(data, list)
            replace_book(data)
            st.rerun()
        except Exception as e:                                     # noqa: BLE001
            st.error(f"Import failed: {e}")
    if not book:
        st.info("The book is empty. Add a line from the **Chain** ticket, the **Trade** screen, "
                "or the manual entry above.")
        return
    if io3.button("🗑  Flatten the book", **FULL):
        replace_book([])
        st.rerun()

    with st.spinner("Repricing every line off its own market…"):
        priced, unpriced = [], []
        for p in book:
            m = line_market(p, cx)
            (priced if m else unpriced).append((p, m))
        lines = [(p, m) for p, m in priced]
        bg = book_greeks(lines, cx["r"])
        mtx = scenario_matrix(lines, cx["r"]) if lines else pd.DataFrame()
        span = span_margin(lines, cx["r"]) if lines else dict(margin=0.0, scan_pct=5, vol_pts=10,
                                                              scenarios=pd.DataFrame())
    if unpriced:
        st.warning("No market data for: " + ", ".join(line_label(p) for p, _ in unpriced) +
                   " — left out of every total below rather than priced off nothing.")
    if any(p.get("legacy") for p in book):
        st.caption("Lines marked legacy came from an older export that stored a tenor, not an "
                   "expiry date; their expiry was set from the day they were imported.")
    tot = bg["total"]

    k = st.columns(6)
    kpi(k[0], "Mark to market", f"${tot['value']:+,.0f}", "against your entries",
        GREEN if tot["value"] >= 0 else RED)
    kpi(k[1], "Delta cash", f"${tot['delta']:+,.0f}", "futures-equivalent", GREEN)
    kpi(k[2], "Gamma cash", f"${tot['gamma']:+,.0f}", "per 1% move", BLUE)
    kpi(k[3], "Vega cash", f"${tot['vega']:+,.0f}", "per vol point", PURPLE)
    kpi(k[4], "Theta", f"${tot['theta']:+,.0f}", "per calendar day",
        RED if tot["theta"] < 0 else GREEN)
    kpi(k[5], "SPAN margin", f"${span['margin']:,.0f}",
        f"worst of 16 · ±{span['scan_pct']:.0f}% / ±{span['vol_pts']:.0f} vols", AMBER)
    if span["margin"] > 1e-9:
        rom = tot["value"] / span["margin"] * 100
        st.markdown(f'<div class="note">Margin, not notional, caps your size: this book ties up '
                    f'<b>${span["margin"]:,.0f}</b> and is <b>{rom:+.1f}%</b> on margin. '
                    f'Option premium held: ${tot["premium"]:+,.0f}.</div>',
                    unsafe_allow_html=True)

    section("Vega by tenor")
    bk = bg["buckets"]
    bc = st.columns(len(bk))
    for i, (bn, v) in enumerate(bk.items()):
        kpi(bc[i], bn, f"${v:+,.0f}", "vega cash", PURPLE if abs(v) > 1 else GRAY)
    net_v, gross_v = sum(bk.values()), sum(abs(v) for v in bk.values())
    if gross_v > 1e-9 and abs(net_v) < gross_v * 0.6:
        st.caption(f"Net vega ${net_v:+,.0f} against gross ${gross_v:,.0f} — a **calendar "
                   f"position**: long volatility in one bucket, short in another.")

    section("Positions")
    pdf = pd.DataFrame(bg["rows"])
    if not pdf.empty:
        pdf = as_text(pdf, {"Vol": "{:.2%}"})
        st.dataframe(pdf.style.format(
            {"F": "{:,.4f}", "Days": "{:.0f}", "Price": "{:,.4f}",
             "Entry": "{:,.4f}", "DeltaCash": "{:+,.0f}", "GammaCash": "{:+,.0f}",
             "VegaCash": "{:+,.0f}", "ThetaCash": "{:+,.0f}", "MTM": "{:+,.0f}"})
            .pipe(_smap, pnl_cell_style, subset=["MTM"]),
            **FULL, hide_index=True)
    rm = st.multiselect("Remove lines", [p["id"] for p in book],
                        format_func=lambda i: next(line_label(p) for p in book if p["id"] == i),
                        key="bk_rm")
    if rm and st.button(f"Remove {len(rm)} line(s)", **FULL):
        replace_book([p for p in book if p["id"] not in rm])
        st.rerun()

    if mtx.empty:
        return
    section("Scenario grid — price × volatility")
    st.dataframe(mtx.style.format("{:+,.0f}").pipe(_smap, pnl_cell_style), **FULL)
    fig = go.Figure(go.Heatmap(z=mtx.values, x=list(mtx.columns), y=list(mtx.index),
                               colorscale="RdYlGn", zmid=0,
                               hovertemplate="%{x} · %{y}<br>$%{z:,.0f}<extra></extra>"))
    fig.update_xaxes(title="Move in every forward")
    fig.update_yaxes(title="Shift in every surface")
    st.plotly_chart(styled(fig, 300, "Book P&L surface"), **FULL)
    with st.expander("The sixteen SPAN scenarios"):
        st.dataframe(span["scenarios"].style.format(
            {"weight": "{:.2f}", "pnl": "{:+,.0f}", "weighted_loss": "{:+,.0f}"}),
            **FULL, hide_index=True)


def screen_vol_surface(cx: dict) -> None:
    surf, F, T, r, name = cx["surface"], cx["F"], cx["T_opt"], cx["r"], cx["name"]
    section("The automatic volatility surface",
            "Nothing here is an input — " + surf["detail"] + ".")
    fn = cx["vol_fn"]
    q = rr_bf_from_fn(F, T, r, fn, 0.25)
    garch = surf.get("garch")
    fc = surface_forecast(surf, T, cx["T"])
    k = st.columns(5)
    kpi(k[0], f"ATM · {cx['spec']['label']}", f"{q['atm'] * 100:.2f}%",
        f"{T * 365.25:.0f}-day option", PURPLE)
    kpi(k[1], "25Δ risk reversal", f"{q['rr'] * 100:+.2f} vols", "call vol − put vol",
        RED if q["rr"] < 0 else GREEN)
    kpi(k[2], "25Δ butterfly", f"{q['bf'] * 100:+.2f} vols", "wings over the body", BLUE)
    if fc is not None and surf["source"] == "implied":
        kpi(k[3], "Forecast realised", f"{fc * 100:.1f}%", "GARCH over the same life", AMBER)
        kpi(k[4], "Implied − forecast", f"{(q['atm'] - fc) * 100:+.1f} vols",
            "what the market charges over the forecast",
            RED if q["atm"] > fc else GREEN)
    elif garch:
        kpi(k[3], "GARCH now", f"{garch['vol_now'] * 100:.1f}%", "next-session vol", AMBER)
        kpi(k[4], "GARCH long-run", f"{garch['vol_lr'] * 100:.1f}%",
            f"half-life {garch['half_life']:.0f} sessions", AMBER)
    else:
        kpi(k[3], "Source", "REGISTRY", "no data reachable", RED)

    if surf["source"] == "implied":
        section(f"{surf['proxy']} option chains — market against fit")
        exps = surf["expiries"]
        i = st.selectbox("Fund expiry", range(len(exps)),
                         index=int(np.argmin([abs(e["T"] - T) for e in exps])),
                         format_func=lambda j: f"{exps[j]['expiry']}  ·  {exps[j]['T'] * 365.25:.0f}d"
                                               f"  ·  {exps[j]['n']} quotes")
        e = exps[i]
        fig = go.Figure()
        pts = pd.DataFrame(e.get("points", []))
        if not pts.empty:
            for typ, col in (("put", RED), ("call", GREEN)):
                sub = pts[pts["type"].str.contains(typ)]
                fig.add_trace(go.Scatter(x=np.exp(sub["k"]) * 100, y=sub["iv"] * 100, mode="markers",
                                         name=f"{typ}s (mid)", marker=dict(size=8, color=col)))
        if e["has_smile"]:
            ks = np.linspace(min(e["k"]) - 0.03, max(e["k"]) + 0.03, 120)
            a_ = sabr_alpha_from_atm(e["F"], e["T"], e["atm"], e["beta"], e["rho"], e["nu"])
            fig.add_trace(go.Scatter(
                x=np.exp(ks) * 100,
                y=[sabr_vol(e["F"], e["F"] * math.exp(x), e["T"], a_, e["beta"], e["rho"], e["nu"]) * 100
                   for x in ks], mode="lines", name="SABR fit", line=dict(color=BLUE, width=2)))
        fig.add_hline(y=e["atm"] * 100, line=dict(color=PURPLE, dash="dot"))
        fig.update_xaxes(title="Strike, % of the fund's forward")
        fig.update_yaxes(title="Implied vol (%)")
        st.plotly_chart(styled(fig, 320, f"{surf['proxy']} {e['expiry']} — "
                                         + (f"SABR ρ {e['rho']:+.2f}, ν {e['nu']:.2f}, RMSE "
                                            f"{e['rmse'] * 100:.2f} vols" if e["has_smile"]
                                            else "ATM only: too few strikes for a smile")), **FULL)
        st.dataframe(as_text(pd.DataFrame([dict(Expiry=x["expiry"], Days=x["T"] * 365.25, Fwd=x["F"],
                                         ATM=x["atm"], rho=x["rho"] if x["has_smile"] else np.nan,
                                         nu=x["nu"] if x["has_smile"] else np.nan, Quotes=x["n"],
                                         Dropped=x["dropped"],
                                         Status=(x.get("quality", "") if x.get("used", True)
                                                 else "rejected — noisy"),
                                         RMSE=x["rmse"] * 100 if x["has_smile"] else np.nan)
                                    for x in exps]),
                             {"rho": "{:+.3f}", "nu": "{:.3f}", "RMSE": "{:.2f}"}).style.format(
            {"Days": "{:.0f}", "Fwd": "{:,.3f}", "ATM": "{:.2%}"}), **FULL, hide_index=True)

    # ── the strip ────────────────────────────────────────────────────────────
    section(f"{name} — every listed month")
    curve = cx["curve"]
    live = curve[curve["option_live"]] if "option_live" in curve else curve
    rows = []
    for rr in live.itertuples():
        f_m = surface_vol_fn(surf, float(rr.price), float(rr.T))
        qq = rr_bf_from_fn(float(rr.price), float(rr.T_opt), r, f_m, 0.25)
        rows.append(dict(Month=rr.label, Opt_expiry=rr.opt_expiry, Days=rr.T_opt * 365.25,
                         F=rr.price, ATM=qq["atm"], Put25=qq["vol_put"], Call25=qq["vol_call"],
                         RR=qq["rr"] * 100, BF=qq["bf"] * 100,
                         Scale=samuelson_scale(surf["g"], rr.T_opt, rr.T),
                         T_opt=rr.T_opt, T_fut=rr.T))
    sdf = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        if not sdf.empty:
            fig.add_trace(go.Scatter(x=sdf["Days"], y=sdf["ATM"] * 100, mode="lines+markers",
                                     name=f"{name.split('(')[-1].rstrip(')')} ATM (mapped)",
                                     line=dict(color=AMBER, width=2.4)))
        if surf["source"] == "implied":
            fig.add_trace(go.Scatter(x=[x["T"] * 365.25 for x in surf["expiries"]],
                                     y=[x["atm"] * 100 for x in surf["expiries"]], mode="markers",
                                     name=f"{surf['proxy']} ATM (market)",
                                     marker=dict(size=10, color=BLUE, symbol="diamond")))
        if garch:
            dd = np.linspace(7, max(400, float(sdf["Days"].max()) if not sdf.empty else 400), 60)
            fig.add_trace(go.Scatter(x=dd, y=[garch_term_vol(garch, d / 365.25) * 100 for d in dd],
                                     mode="lines", name="GARCH front-month forecast",
                                     line=dict(color=GRAY, dash="dot")))
        fig.update_xaxes(title="Days to option expiry")
        fig.update_yaxes(title="ATM vol (%)")
        st.plotly_chart(styled(fig, 360, "Term structure", legend_below=True), **FULL)
    with c2:
        rat = pd.DataFrame(surf.get("ratios", []))
        fig = go.Figure()
        if not rat.empty:
            good = rat[rat["valid"]]
            bad = rat[~rat["valid"] & rat["ratio"].notna()]
            fig.add_trace(go.Scatter(x=good["tau"], y=good["ratio"], mode="markers", name="measured",
                                     marker=dict(size=9, color=AMBER), text=good["label"]))
            if len(bad):
                fig.add_trace(go.Scatter(x=bad["tau"], y=bad["ratio"],
                                         mode="markers", name="too thin to count",
                                         marker=dict(size=7, color=GRAY, symbol="x"),
                                         text=bad["label"]))
        tt = np.linspace(0, max(2.0, float(rat["tau"].max()) if not rat.empty else 2.0), 80)
        fig.add_trace(go.Scatter(x=tt, y=[samuelson_g(surf["g"], x) for x in tt], mode="lines",
                                 name=f"g(τ) — {surf['g']['method']}", line=dict(color=BLUE, width=2)))
        fig.update_xaxes(title="Years to delivery")
        fig.update_yaxes(title="Realised vol ÷ reference")
        ref = surf["proxy"] if surf["source"] == "implied" else "front month"
        st.plotly_chart(styled(fig, 360, f"Samuelson curve — each month against {ref}",
                               legend_below=True), **FULL)
    if not sdf.empty:
        st.dataframe(sdf.drop(columns=["T_opt", "T_fut"]).style.format(
            {"Days": "{:.0f}", "F": "{:,.4f}", "ATM": "{:.2%}", "Put25": "{:.2%}",
             "Call25": "{:.2%}", "RR": "{:+.2f}", "BF": "{:+.2f}", "Scale": "{:.3f}"}),
            **FULL, hide_index=True)
        st.caption("RR and BF in vol points. Scale is the Samuelson factor applied to the "
                   "reference vol for that month: the RMS of g(τ) over the option's life.")

        Ks = np.linspace(0.70, 1.30, 25)
        Z = np.array([[surface_vol_fn(surf, float(x.F), float(x.T_fut))(float(x.F) * m, float(x.T_opt)) * 100
                       for m in Ks] for x in sdf.itertuples()])
        fig = go.Figure(go.Surface(z=Z, x=Ks * 100, y=sdf["Month"], colorscale="RdYlGn_r",
                                   opacity=0.93,
                                   hovertemplate="K %{x:.0f}% of F<br>%{y}<br>σ %{z:.2f}%<extra></extra>"))
        fig.update_layout(template="plotly_dark", height=430, paper_bgcolor=BG,
                          font=dict(family="Inter", size=11, color=TEXT),
                          margin=dict(l=10, r=10, t=30, b=10),
                          scene=dict(bgcolor=BG,
                                     xaxis=dict(title="Strike, % of F", gridcolor=BORDER, backgroundcolor=BG),
                                     yaxis=dict(title="Month", gridcolor=BORDER, backgroundcolor=BG),
                                     zaxis=dict(title="σ (%)", gridcolor=BORDER, backgroundcolor=BG),
                                     camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8))))
        st.plotly_chart(fig, **FULL)

        fw = []
        for a_, b_ in zip(rows[:-1], rows[1:]):
            fw.append(dict(From=a_["Month"], To=b_["Month"], ATM_from=a_["ATM"], ATM_to=b_["ATM"],
                           Forward=forward_vol(a_["ATM"], a_["T_opt"], b_["ATM"], b_["T_opt"])))
        if fw:
            st.markdown("##### Forward volatility between option expiries")
            st.dataframe(as_text(pd.DataFrame(fw), {"Forward": "{:.2%}"}, "negative variance")
                         .style.format({"ATM_from": "{:.2%}", "ATM_to": "{:.2%}"}),
                         **FULL, hide_index=True)
            st.caption("Each month is a different future, so this is a forward vol across the "
                       "strip rather than for one underlying — read it as the term structure's "
                       "slope, not as a calendar option price.")

    if surf.get("warnings"):
        with st.expander(f"Feed notes ({len(surf['warnings'])})"):
            for w in surf["warnings"]:
                st.markdown(f"- {w}")
    with st.expander("ⓘ  How the surface is built"):
        st.markdown(f"""
1. **Listed options.** For each fund expiry the desk takes bid/ask mids (last trades only
   when the market is closed and no two-sided quote exists), infers the fund's forward from
   put-call parity, and inverts each out-of-the-money option with Black-76.
2. **One smile per expiry.** ATM is read straight off the quotes at k = 0. With five or more
   strikes across both wings, SABR (β = {SABR_BETA}) fits ρ and ν with α pinned to that ATM;
   bad prints the first pass misses by more than three vols are dropped once.
3. **Term structure.** ATM is interpolated linearly in total variance σ²T, forced
   non-decreasing (a dip would be a calendar arbitrage), flat beyond the last expiry.
4. **Onto the futures strip.** Each listed month's realised vol is divided by the fund's,
   over the same {VOL_WINDOW} sessions, and g(τ) = c·(a + (1−a)e^(−bτ)) is fitted across the
   strip — the Samuelson effect. A month's ATM is the fund's ATM at that option's life
   times the RMS of g over the life. The smile carries over in moneyness, α re-solved at
   the future's forward.
5. **No options anywhere** (coffee, cocoa, livestock): GARCH(1,1) on five years of the
   continuous front month gives a mean-reverting forecast, mapped the same way, flat smile.

*What it assumes:* the fund's implied-to-realised relationship carries over to each futures
month. That is a proxy — a good one for gold, a looser one for funds that roll a ladder of
months — and the label on every screen says which source priced the number in front of you.
""")


def rr_bf_from_fn(F: float, T: float, r: float, fn, delta: float = 0.25) -> dict:
    """25-delta risk reversal and butterfly off any σ(K, T): strikes solved at
    their own vols (iterated), quoted in vol units."""
    atm = fn(F, T)
    kc = strike_from_delta(F, T, r, atm, delta, "call")
    kp = strike_from_delta(F, T, r, atm, delta, "put")
    for _ in range(6):
        kc = strike_from_delta(F, T, r, fn(kc, T), delta, "call")
        kp = strike_from_delta(F, T, r, fn(kp, T), delta, "put")
    vc, vp = fn(kc, T), fn(kp, T)
    return dict(atm=atm, K_call=kc, K_put=kp, vol_call=vc, vol_put=vp,
                rr=vc - vp, bf=0.5 * (vc + vp) - atm)


def screen_structure(cx: dict) -> None:
    section("Spread options on refining and processing margins — Kirk (1995)",
            "A <b>crack</b> is the refiner's gross margin, a <b>crush</b> the soybean "
            "processor's. Every leg's volatility comes from <i>its own</i> automatic surface, "
            "and the correlations between legs are measured from their daily returns — "
            "nothing on this screen is typed in except the trade.")
    c0, c1 = st.columns([2, 1])
    struct = c0.selectbox("Structure", list(STRUCTURES.keys()))
    months = c1.select_slider("Option life", options=[1, 2, 3, 4, 6, 9, 12, 18], value=3,
                              format_func=lambda m: f"{m}M")
    T = months / 12
    cfg = STRUCTURES[struct]
    st.markdown(f'<div class="note">{cfg["note"]} &nbsp;·&nbsp; quoted {cfg["basis"]} '
                f'({cfg["unit"]}).</div>', unsafe_allow_html=True)

    legs = [n for n, _, _ in cfg["long"]] + [n for n, _, _ in cfg["short"]]
    prices, vols, srcs, missing = {}, {}, {}, []
    with st.spinner("Loading every leg's curve and surface…"):
        for leg in legs:
            cur = cx["curves"].get(leg)
            if cur is None:
                cur = cx["curve"] if leg == cx["name"] else fetch_curve(leg)
                cx["curves"][leg] = cur
            px = forward_at(cur, T) if (cur is not None and not cur.empty) else None
            if px is None:
                missing.append(leg)
                continue
            prices[leg] = px
            surf = vol_surface(leg, cx["r"])
            vols[leg] = surface_atm(surf, T, T)
            srcs[leg] = surf["label"]
    if missing:
        st.error(f"**NO MARKET DATA** for {', '.join(missing)} — a structure is only as real as "
                 f"its thinnest leg, so nothing is priced in its place.")
        return

    rets = {leg: leg_returns(leg, T) for leg in legs}
    j = pd.concat([s.rename(n) for n, s in rets.items() if len(s)], axis=1, join="inner").dropna().tail(120)
    corr = j.corr() if len(j) >= 40 and j.shape[1] == len(legs) else None

    agg = structure_legs(struct, prices, vols, corr)
    c1, c2 = st.columns(2)
    K = c1.number_input(f"Strike on the spread ({cfg['unit']})",
                        value=float(round(agg["spread"], 3)), step=0.25, format="%.3f")
    side = c2.radio("Option", ["Call ▲ (margin widens)", "Put ▼ (margin compresses)"],
                    key="sp_side", horizontal=True)
    otype = "call" if "Call" in side else "put"
    rho = agg["rho"]

    so = SpreadOption(agg["F_long"], agg["F_short"], K, T, cx["r"],
                      agg["sigma_long"], agg["sigma_short"], rho, otype)
    res = so.price()
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Spread now", f"{agg['spread']:,.3f}", cfg["unit"], GREEN if agg["spread"] > 0 else RED)
    kpi(k2, "Option price", f"{res['price']:.4f}", f"Kirk σ {res['sigma_kirk'] * 100:.1f}%", AMBER)
    kpi(k3, "Intrinsic", f"{res['intrinsic']:.4f}", f"time value {res['time_value']:.4f}")
    kpi(k4, "Correlation", f"{rho:.3f}", agg["rho_source"], BLUE)
    kpi(k5, "Δ long / short", f"{res['delta_long']:+.3f} / {res['delta_short']:+.3f}",
        "products / input", GREEN)

    st.markdown("##### Legs, converted to the common unit")
    ldf = pd.DataFrame(agg["legs"])
    ldf["side"] = ["long"] * len(cfg["long"]) + ["short"] * len(cfg["short"])
    ldf["vol source"] = [srcs.get(x, "") for x in ldf["leg"]]
    st.dataframe(ldf[["side", "leg", "qty", "conv", "price", "value", "vol", "vol source"]].style.format(
        {"qty": "{:.1f}", "conv": "{:.4f}", "price": "{:,.4f}", "value": "{:,.4f}",
         "vol": "{:.1%}"}), **FULL, hide_index=True)
    if corr is not None:
        st.caption(f"Leg correlations measured on {len(j)} common sessions of the month nearest "
                   f"{months}M. Long side vol {agg['sigma_long']:.1%}, short side "
                   f"{agg['sigma_short']:.1%}, built from the full correlation matrix.")
        st.dataframe(corr.style.format("{:.3f}"), **FULL)
    else:
        st.caption(f"Not enough overlapping history to measure leg correlations: each side's vol "
                   f"is value-weighted and the cross-correlation is the registry default "
                   f"{cfg['rho']:.2f}.")

    with st.spinner("Benchmarking Kirk against a two-factor simulation…"):
        mc = so.mc_price(200_000)
    err = res["price"] - mc["price"]
    ok = abs(err) <= max(3 * mc["std_error"], 0.02 * max(mc["price"], 1e-9))
    st.markdown(
        f'<div class="note" style="border-left:2px solid {GREEN if ok else RED}">'
        f'<b>Kirk vs Monte Carlo:</b> {res["price"]:.4f} against {mc["price"]:.4f} ± '
        f'{mc["std_error"]:.4f} — difference <b>{err:+.4f}</b> '
        f'({err / max(mc["price"], 1e-9) * 100:+.2f}%). '
        f'{"Within simulation noise." if ok else "Outside tolerance — treat the closed form with caution here."}'
        f'</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        width = max(abs(agg["spread"]) * 0.8, 10.0)
        rng = np.linspace(agg["spread"] - width, agg["spread"] + width, 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rng, y=so.payoff(rng) - res["price"], name="P&L",
                                 line=dict(color=AMBER, width=2.4)))
        fig.add_hline(y=0, line=dict(color=GRAY, width=0.8))
        vline(fig, K, GREEN, f"K {K:,.1f}")
        vline(fig, agg["spread"], RED, f"Spread {agg['spread']:,.1f}", y=0.82)
        fig.update_xaxes(title=f"Spread at expiry ({cfg['unit']})")
        fig.update_yaxes(title=f"P&L ({cfg['unit']})")
        st.plotly_chart(styled(fig, 330, "P&L at expiry"), **FULL)
    with g2:
        Ks = np.linspace(K - width / 2, K + width / 2, 13)
        Rs = np.linspace(0.4, 0.98, 9)
        Z = np.array([[SpreadOption(agg["F_long"], agg["F_short"], kk, T, cx["r"],
                                    agg["sigma_long"], agg["sigma_short"], rr, otype).price()["price"]
                       for kk in Ks] for rr in Rs])
        fig = go.Figure(go.Heatmap(z=Z, x=[f"{x:,.1f}" for x in Ks], y=[f"{x:.2f}" for x in Rs],
                                   colorscale="YlOrRd",
                                   hovertemplate="K %{x}<br>ρ %{y}<br>Price %{z:.4f}<extra></extra>"))
        fig.update_xaxes(title=f"Strike ({cfg['unit']})")
        fig.update_yaxes(title="Correlation")
        st.plotly_chart(styled(fig, 330, f"Sensitivity to ρ — measured {rho:.2f}"), **FULL)


def screen_calendar(cx: dict) -> None:
    section("Calendar spread options — an option on the shape of the curve",
            "The same commodity in two delivery months. A <b>call</b> pays when the near "
            "month gains on the far one (backwardation widens), a <b>put</b> when contango "
            "deepens. Both legs' vols come off the automatic surface — the far month seen "
            "over the near option's life — and their correlation is measured from the two "
            "contracts' own daily settles.")
    curve, unit, name, surf = cx["curve"], cx["unit"], cx["name"], cx["surface"]
    near_rows = curve[curve["option_live"]].reset_index(drop=True)
    if len(near_rows) < 1 or len(curve) < 2:
        st.error("**NO CURVE** — a calendar spread needs at least two dated contracts.")
        return
    lab = lambda df, i: f"{df['label'].iloc[i]}  (option {df['T_opt'].iloc[i] * 365.25:.0f}d)"
    c1, c2 = st.columns(2)
    ni = c1.selectbox("Near month", range(len(near_rows)), index=0,
                      format_func=lambda i: lab(near_rows, i))
    near = near_rows.iloc[ni]
    far_rows = curve[curve["T"] > near["T"]].reset_index(drop=True)
    if far_rows.empty:
        st.warning("No later month listed.")
        return
    fi = c2.selectbox("Far month", range(len(far_rows)), index=min(4, len(far_rows) - 1),
                      format_func=lambda i: str(far_rows["label"].iloc[i]))
    far = far_rows.iloc[fi]

    Fn, Ff = float(near["price"]), float(far["price"])
    T_opt = max(float(near["T_opt"]), 1 / 365)
    spread = Fn - Ff
    vn = surface_atm(surf, T_opt, float(near["T"]))
    vf = surface_atm(surf, T_opt, float(far["T"]))
    closes = fetch_strip_closes(name)
    rho = correlation_from_closes(closes, near["ticker"], far["ticker"])
    rho_src = "measured, last 120 sessions" if rho is not None else "default 0.95 — too little history"
    rho = 0.95 if rho is None else rho

    d1, d2 = st.columns(2)
    K = d1.number_input(f"Strike on the spread ({unit})", value=float(round(spread, 4)),
                        step=0.01, format="%.4f")
    otype = "call" if "Call" in d2.radio(
        "Option", ["Call ▲ (near gains on far)", "Put ▼ (far gains on near)"],
        horizontal=True, key="cs_side") else "put"

    so = SpreadOption(Fn, Ff, K, T_opt, cx["r"], vn, vf, rho, otype)
    res = so.price()
    mult = price_multiplier(name)
    k = st.columns(6)
    kpi(k[0], "Spread", f"{spread:+,.4f}", f"{near['label']} − {far['label']}",
        GREEN if spread > 0 else RED)
    kpi(k[1], "Price", f"{res['price']:.5f}", f"${res['price'] * mult:,.0f} per lot", AMBER)
    kpi(k[2], "σ near / far", f"{vn:.1%} / {vf:.1%}", "automatic, over the near option's life", PURPLE)
    kpi(k[3], "Correlation", f"{rho:.3f}", rho_src, BLUE)
    kpi(k[4], "Kirk σ", f"{res['sigma_kirk'] * 100:.2f}%", "spread vol", PURPLE)
    kpi(k[5], "Δ near / far", f"{res['delta_long']:+.3f} / {res['delta_short']:+.3f}", "per unit", GREEN)

    g1, g2 = st.columns(2)
    with g1:
        w = max(abs(spread), Fn * 0.05)
        rng = np.linspace(spread - w, spread + w, 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rng, y=so.payoff(rng) - res["price"], name="P&L",
                                 line=dict(color=AMBER, width=2.4)))
        fig.add_hline(y=0, line=dict(color=GRAY, width=0.8))
        vline(fig, K, GREEN, f"K {K:,.3f}")
        vline(fig, spread, RED, f"Now {spread:,.3f}", y=0.82)
        fig.update_xaxes(title=f"Spread at expiry ({unit})")
        fig.update_yaxes(title=f"P&L ({unit})")
        st.plotly_chart(styled(fig, 320, "P&L at expiry"), **FULL)
    with g2:
        rows = []
        for i in range(len(near_rows)):
            a = near_rows.iloc[i]
            later = curve[curve["T"] > a["T"]]
            if later.empty:
                continue
            b = later.iloc[0]
            t = max(float(a["T_opt"]), 1 / 365)
            rr_ = correlation_from_closes(closes, a["ticker"], b["ticker"]) or 0.95
            p = SpreadOption(float(a["price"]), float(b["price"]), float(a["price"] - b["price"]), t,
                             cx["r"], surface_atm(surf, t, float(a["T"])),
                             surface_atm(surf, t, float(b["T"])), rr_, "call").price()["price"]
            rows.append(dict(pair=f"{a['label'][:3]}{a['label'][-2:]}/{b['label'][:3]}{b['label'][-2:]}",
                             spread=float(a["price"] - b["price"]), price=p))
        if rows:
            sdf = pd.DataFrame(rows).head(16)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sdf["pair"], y=sdf["price"], name="ATM call",
                                 marker_color=[GREEN if s > 0 else RED for s in sdf["spread"]],
                                 opacity=0.8))
            fig.add_trace(go.Scatter(x=sdf["pair"], y=sdf["spread"], name="Spread", yaxis="y2",
                                     mode="lines+markers", line=dict(color=AMBER, width=1.5)))
            fig.update_layout(yaxis2=dict(title="Spread", overlaying="y", side="right", showgrid=False))
            fig.update_yaxes(title=f"ATM price ({unit})")
            st.plotly_chart(styled(fig, 320, "At-the-money one-month-apart calendars"), **FULL)


def screen_trade(cx: dict) -> None:
    lead("Price one option or a whole structure on the month picked in the sidebar — a "
         "single option is just a structure with one leg.",
         """
**Strikes** can be given three ways. *Exact* is how a listed option is named and how a
broker statement reads. *% of forward* is a quick way to say "10% out of the money".
*Delta* is how a desk quotes: "the 25-delta put". All three are snapped onto the
contract's listed increment, so the strike you price is one you could deal.

**Every leg carries its own σ(K)** from the automatic surface. This matters more than it
looks: on a flat volatility the 25-delta call and put carry the same number and a risk
reversal comes out nearly free, which is never true in a market that charges for skew.

**The Greeks** are per unit of the commodity; the cash line underneath converts them to
dollars per lot. Delta is futures-equivalent exposure, gamma is how fast that changes,
vega is per one volatility point, theta is per calendar day.

**European or American.** Most listed commodity options are American. The tree prices the
right to exercise early, worth about 0.4% of the premium at a 2% rate and 2.6% at 10%.
""")
    F, T, r, unit, name = cx["F"], cx["T_opt"], cx["r"], cx["unit"], cx["name"]
    vs = cx["vs"]
    vf = vs["fn"]
    mult = price_multiplier(name)
    st.markdown(badge(f"{cx['spec']['label']} @ {F:,.4f} {unit}", INK) + " " +
                badge(f"option expires {cx['spec']['opt_expiry']:%d %b %Y} · {T * 365.25:.0f}d", INK)
                + " " + _vol_badges(cx), unsafe_allow_html=True)

    mode = st.radio("Build", ["Single option", "Structure"], horizontal=True, key="tr_mode")

    if mode == "Single option":
        c1, c2, c3, c4, c5 = st.columns([1.3, 1.4, 1, 1, 1])
        how = c1.selectbox("Strike by", ["Exact", "% of forward", "Delta"],
                           help="Exact is the way a listed option is named; the others "
                                "resolve to a strike and are snapped to the grid.")
        if how == "Exact":
            raw = c2.number_input(f"Strike ({unit})", min_value=1e-6, value=float(snap_strike(name, F)),
                                  step=float(CONTRACTS[name]["strike_inc"]), format="%.4f")
            K = snap_strike(name, raw)
        elif how == "% of forward":
            raw = c2.number_input("Strike (% of F)", 20.0, 300.0, 100.0, 1.0, format="%.1f")
            K = snap_strike(name, F * raw / 100)
        else:
            raw = c2.slider("Delta", 5, 50, 25, 1) / 100
            K = snap_strike(name, strike_from_delta(F, T, r, vf(F, T), raw, "call"))
        otype = c3.radio("Type", ["call", "put"], horizontal=True, key="tr_ty")
        lots = c4.number_input("Lots", -999, 999, 1, key="tr_lots")
        amer = c5.radio("Exercise", ["European", "American"], key="tr_ex") == "American"

        sig = vf(K, T)
        g = (american_greeks(F, K, T, r, sig, otype, 300) if amer
             else Black76(F, K, T, r, sig, otype).greeks() | {"early_premium": 0.0})
        k = st.columns(6)
        kpi(k[0], "Premium", f"{g['price']:,.5f}",
            f"${g['price'] * mult * abs(lots):,.0f} for {abs(lots)} lot(s)", AMBER)
        kpi(k[1], "σ(K)", f"{sig:.2%}", f"ATM {vs['atm']:.2%}", PURPLE)
        kpi(k[2], "Delta", f"{g['delta'] * lots:+.4f}",
            f"${g['delta'] * lots * mult * F:+,.0f} cash", GREEN)
        kpi(k[3], "Gamma", f"{g['gamma'] * lots:+.5f}", "delta change per unit", BLUE)
        kpi(k[4], "Vega", f"{g['vega'] * lots:+.4f}",
            f"${g['vega'] * lots * mult:+,.0f} per vol point", PURPLE)
        kpi(k[5], "Theta", f"{g['theta'] * lots:+.5f}",
            f"${g['theta'] * lots * mult:+,.0f} per day",
            RED if g["theta"] * lots < 0 else GREEN)
        if amer and g["early_premium"] > 0:
            st.caption(f"Early exercise is worth {g['early_premium']:+.5f} {unit} "
                       f"(${g['early_premium'] * mult:,.0f} per lot) — priced on the tree, "
                       f"not assumed away.")
        legs = [dict(qty=lots, type=otype, strike=K, vol=sig, **g)]
        label = f"{otype} {K:,.4f}"
    else:
        c1, c2, c3 = st.columns([2.2, 1, 1])
        sname = c1.selectbox("Structure", list(STRATEGIES.keys()), index=10)
        lots = c2.number_input("Lots", -999, 999, 1, key="ts_lots")
        amer = c3.radio("Exercise", ["European", "American"], key="ts_ex") == "American"
        strat = build_strategy(sname, F, T, r, vf, lots, amer, contract=name)
        st.markdown(f'<div class="note">{strat["note"]}</div>', unsafe_allow_html=True)
        legs, net, label = strat["legs"], strat["net"], sname
        k = st.columns(5)
        kpi(k[0], "Net premium", f"{net['price']:+,.5f}",
            f"${net['price'] * mult:+,.0f} — {'paid' if net['price'] > 0 else 'received'}",
            RED if net["price"] > 0 else GREEN)
        kpi(k[1], "Delta", f"{net['delta']:+.4f}", f"${net['delta'] * mult * F:+,.0f}", GREEN)
        kpi(k[2], "Gamma", f"{net['gamma']:+.5f}", "per unit²", BLUE)
        kpi(k[3], "Vega", f"{net['vega']:+.4f}", f"${net['vega'] * mult:+,.0f} / vol pt", PURPLE)
        kpi(k[4], "Theta", f"{net['theta']:+.5f}", f"${net['theta'] * mult:+,.0f} / day",
            RED if net["theta"] < 0 else GREEN)
        ldf = pd.DataFrame(legs)[["qty", "type", "spec", "strike", "vol", "price",
                                  "delta", "gamma", "vega", "theta"]]
        ldf.columns = ["Qty", "Type", "Spec", "Strike", "σ(K)", "Price", "Δ", "Γ", "Vega", "Θ"]
        st.dataframe(ldf.style.format({"Strike": "{:,.3f}", "σ(K)": "{:.2%}", "Price": "{:,.4f}",
                                       "Δ": "{:+.4f}", "Γ": "{:.5f}", "Vega": "{:.4f}",
                                       "Θ": "{:+.5f}"}),
                     **FULL, hide_index=True)

    prem = sum(l["qty"] * l["price"] for l in legs)
    rng = np.linspace(F * 0.55, F * 1.45, 400)
    pnl = strategy_payoff(legs, rng) - prem
    g1, g2 = st.columns([3, 1.4])
    with g1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rng, y=pnl, name="P&L at expiry",
                                 line=dict(color=AMBER, width=2.5),
                                 fill="tozeroy", fillcolor="rgba(240,165,0,0.10)"))
        fig.add_hline(y=0, line=dict(color=GRAY, width=0.8))
        for l in legs:
            fig.add_vline(x=l["strike"], line=dict(color=GREEN if l["qty"] > 0 else RED,
                                                   dash="dot", width=1))
        vline(fig, F, BLUE, f"F {F:,.2f}")
        fig.update_xaxes(title=f"Price at expiry ({unit})")
        fig.update_yaxes(title=f"P&L ({unit})")
        st.plotly_chart(styled(fig, 350, "Green lines bought, red sold"),
                        **FULL)
    with g2:
        be = rng[np.where(np.diff(np.sign(pnl)) != 0)[0]]
        for lbl, val, col in (
                ("Max profit shown", f"{np.max(pnl):+,.4f}", GREEN),
                ("Max loss shown", f"{np.min(pnl):+,.4f}", RED),
                ("Break-evens", ", ".join(f"{b:,.2f}" for b in be[:4]) or "none in range", INK),
                ("Cash per lot", f"${prem * mult:+,.0f}", AMBER)):
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                        f'border-bottom:.5px solid {BORDER}"><span style="color:{GRAY};'
                        f'font-size:.75rem">{lbl}</span><span style="font-family:JetBrains Mono,'
                        f'monospace;font-size:.75rem;color:{col}">{val}</span></div>',
                        unsafe_allow_html=True)
        st.caption("Maxima are read off the plotted range: an unbounded structure shows the "
                   "edge of the chart, which is the honest answer to 'how much can this lose'.")
        if st.button("➕  Book it", type="primary", **FULL):
            add_lines(*[make_line("option", name, cx["spec"], int(l["qty"]), float(l["price"]),
                                  l["type"], float(l["strike"]),
                                  "American" if amer else "European", label)
                        for l in legs if int(l["qty"]) != 0])
            st.success(f"{len(legs)} leg(s) booked and saved.")



def screen_vol_analytics(cx: dict) -> None:
    section("Volatility analytics — is this quote rich?",
            "The two questions a vol trader asks before trading: what has this market "
            "actually delivered, and does hedging the option pay? The implied side is the "
            "automatic surface; the realised side is measured from price history.")
    name, F, T, r, vol = cx["name"], cx["F"], cx["T_opt"], cx["r"], cx["vs"]["atm"]
    surf = cx["surface"]
    garch = surf.get("garch")

    hist = fetch_history(name, "5y")
    if hist.empty:
        st.error("**NO PRICE HISTORY** — the continuous series returned nothing, so no cone "
                 "can be built. Nothing is drawn in its place.")
    else:
        cone = vol_cone(hist)
        if cone.empty:
            st.warning("Not enough history for a cone.")
        else:
            k = st.columns(4)
            r20 = cone[cone["window"] == 20]
            cur = float(r20["current"].iloc[0]) if len(r20) else float(cone["current"].iloc[0])
            rank = float(r20["rank"].iloc[0]) if len(r20) else float(cone["rank"].iloc[0])
            kpi(k[0], "Realised 20d", f"{cur * 100:.1f}%", "annualised", AMBER)
            kpi(k[1], "Percentile", f"{rank:.0f}th", "of its own 5-year range",
                RED if rank > 80 else GREEN if rank < 20 else GRAY)
            kpi(k[2], "Implied ATM", f"{vol * 100:.1f}%", surf["label"].split(" · ")[0].title()
                + f" · {cx['spec']['label']}", PURPLE)
            prem = (vol - cur) * 100
            kpi(k[3], "Implied − realised", f"{prem:+.1f} vols",
                "you are paying up" if prem > 0 else "you are being paid",
                RED if prem > 0 else GREEN)

            fig = go.Figure()
            for lo, hi, op in (("p5", "p95", 0.12), ("p25", "p75", 0.22)):
                fig.add_trace(go.Scatter(x=cone["window"], y=cone[hi] * 100, mode="lines",
                                         line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=cone["window"], y=cone[lo] * 100, mode="lines",
                                         line=dict(width=0), fill="tonexty",
                                         fillcolor=f"rgba(88,166,255,{op})",
                                         name=f"{lo}–{hi}"))
            fig.add_trace(go.Scatter(x=cone["window"], y=cone["p50"] * 100, mode="lines",
                                     name="Median", line=dict(color=BLUE, width=1.6, dash="dash")))
            fig.add_trace(go.Scatter(x=cone["window"], y=cone["current"] * 100,
                                     mode="lines+markers", name="Today",
                                     line=dict(color=AMBER, width=2.6), marker=dict(size=7)))
            fig.add_hline(y=vol * 100, line=dict(color=PURPLE, dash="dot"))
            if garch:
                fig.add_trace(go.Scatter(
                    x=cone["window"], y=[garch_term_vol(garch, w / 252) * 100 for w in cone["window"]],
                    mode="lines", name="GARCH forecast", line=dict(color=TEAL, width=1.6, dash="dash")))
            fig.update_xaxes(title="Window (trading days)", type="log")
            fig.update_yaxes(title="Annualised volatility (%)")
            st.plotly_chart(styled(fig, 360, "Volatility cone — purple dotted is the implied ATM"),
                            **FULL)
            st.dataframe(cone[["window", "p5", "p25", "p50", "p75", "p95", "current", "rank"]]
                         .style.format({c: "{:.1%}" for c in ("p5", "p25", "p50", "p75", "p95", "current")}
                                       | {"rank": "{:.0f}"}),
                         **FULL, hide_index=True)
            st.caption("Built on the continuous front-month series, which is NOT roll-adjusted: "
                       "each roll injects one artificial jump that inflates realised vol slightly. "
                       "The right series for a cone, the wrong one for a P&L — stated rather than "
                       "left for you to discover.")

    section("Gamma scalping — implied against realised",
            "You buy a straddle at one volatility and delta-hedge it while the market delivers "
            "another. Theory says the hedged position earns ½∫Γ·F²(σ_real² − σ_impl²)dt. "
            "Simulation adds what theory hides: <b>the dispersion</b>. Discrete hedging leaves "
            "path risk, so a correct view still loses on some paths — and that spread is the "
            "real risk of a gamma book.")
    s_imp = vol
    if garch:
        s_real = surface_forecast(surf, T, cx["T"])
        real_src = "GARCH forecast over the option's life"
    else:
        rv = realised_vol(hist, 20).dropna() if not hist.empty else pd.Series(dtype=float)
        s_real = float(rv.iloc[-1]) if len(rv) else vol
        real_src = "last 20-day realised" if len(rv) else "no history — set equal to implied"
    st.markdown(badge(f"implied paid {s_imp:.1%}", PURPLE) + " " +
                badge(f"realised expected {s_real:.1%} · {real_src}", TEAL),
                unsafe_allow_html=True)
    g3, g4 = st.columns(2)
    rebal = g3.select_slider("Rebalance", options=[12, 21, 63, 126, 252], value=63,
                             format_func=lambda x: f"{x}× over the life")
    pos = g4.radio("Position", ["long", "short"], horizontal=True, key="gs_p")

    with st.spinner("Simulating the hedged path…"):
        gs = gamma_scalp(F, F, T, r, s_imp, s_real, rebal, min(cx["n_paths"], 8000), position=pos)
    mult = price_multiplier(cx["name"])
    k = st.columns(5)
    kpi(k[0], "Mean P&L", f"{gs['mean']:+,.4f}", f"${gs['mean'] * mult:+,.0f} per lot",
        GREEN if gs["mean"] > 0 else RED)
    kpi(k[1], "Theory", f"{gs['theory']:+,.4f}", "½Γ F²(σr²−σi²)T", BLUE)
    kpi(k[2], "5th–95th", f"[{gs['p5']:+,.2f}, {gs['p95']:+,.2f}]", "path dispersion", AMBER)
    kpi(k[3], "Win rate", f"{gs['win_rate']:.0f}%", "paths finishing positive",
        GREEN if gs["win_rate"] > 50 else RED)
    kpi(k[4], "Premium", f"{gs['premium']:+,.4f}", "straddle paid or received", PURPLE)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(x=gs["pnl"], nbinsx=60, marker_color=AMBER, opacity=0.8))
        fig.add_vline(x=0, line=dict(color=GRAY, dash="dash"))
        vline(fig, gs["mean"], GREEN, f"mean {gs['mean']:+.2f}")
        fig.update_xaxes(title="Hedged P&L per unit")
        fig.update_yaxes(title="Paths")
        st.plotly_chart(styled(fig, 300, "Distribution of the delta-hedged P&L"),
                        **FULL)
    with c2:
        edge = np.linspace(max(s_imp - 0.20, 0.02), s_imp + 0.20, 11)
        means = [gamma_scalp(F, F, T, r, s_imp, float(sr), rebal, 1200,
                             position=pos)["mean"] for sr in edge]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge * 100, y=means, mode="lines+markers",
                                 line=dict(color=AMBER, width=2.4)))
        fig.add_hline(y=0, line=dict(color=GRAY, width=0.8))
        vline(fig, s_imp * 100, PURPLE, f"implied {s_imp * 100:.0f}%")
        vline(fig, s_real * 100, TEAL, f"expected {s_real * 100:.0f}%", y=0.82)
        fig.update_xaxes(title="Realised volatility delivered (%)")
        fig.update_yaxes(title="Mean hedged P&L")
        st.plotly_chart(styled(fig, 300, "Break-even sits at the implied you paid"),
                        **FULL)



def _strike_and_side(cx: dict, key: str) -> Tuple[float, str]:
    """Strike (as % of the forward, snapped to the grid) and call/put for the exotic
    screens. Volatility is read off the surface at that strike — not asked for."""
    c1, c2 = st.columns([2, 1])
    pct = c1.slider("Strike, % of forward", 50, 150, 100, 1, key=f"{key}_pct")
    side = c2.radio("Type", ["call", "put"], horizontal=True, key=f"{key}_side")
    return snap_strike(cx["name"], cx["F"] * pct / 100), side


def screen_asian(cx: dict) -> None:
    section("Asian options — payoff on the average price",
            "An Asian option settles against the <b>average</b> of the forward over a set of "
            "fixings, not a single closing price. Airlines, refiners and utilities buy them "
            "because their physical exposure is itself an average: they consume every day of "
            "the month, not on the third Wednesday. Averaging suppresses volatility, so an "
            "Asian is always <b>cheaper than the equivalent vanilla</b> — the page proves it "
            "rather than asserting it.")
    F, T, r, unit = cx["F"], cx["T_opt"], cx["r"], cx["unit"]
    K, side = _strike_and_side(cx, "as")
    vol = cx["vol_fn"](K, T)
    a1, a2, a3 = st.columns(3)
    avg = a1.radio("Average", ["arithmetic", "geometric"], horizontal=True)
    n_obs = a2.select_slider("Fixings", options=[4, 12, 22, 52, 126, 252], value=12,
                             format_func=lambda x: {4: "Quarterly (4)", 12: "Monthly (12)",
                                                    22: "Weekly (22)", 52: "Weekly (52)",
                                                    126: "Daily (126)", 252: "Daily (252)"}[x])
    use_cv = a3.toggle("Variance reduction", value=True,
                       help="Antithetic paths plus the geometric average as a control "
                            "variate. The geometric price is known exactly, so the error it "
                            "makes on each path can be subtracted from the arithmetic one.")

    with st.spinner(f"Simulating {cx['n_paths']:,} paths…"):
        res = asian_mc(F, K, T, r, vol, n_obs, side, avg,
                       cx["n_paths"], control_variate=use_cv, antithetic=use_cv)
        kv = kemna_vorst(F, K, T, r, vol, n_obs, side)
        plain = asian_mc(F, K, T, r, vol, n_obs, side, avg,
                         cx["n_paths"], control_variate=False, antithetic=False)
    van = Black76(F, K, T, r, vol, side).price()

    k1, k2, k3, k4, k5 = st.columns(5)
    st.markdown(badge(f"{cx['spec']['label']} · K {K:,.4f} · σ(K) {vol:.2%}", INK) + " " +
                _vol_badges(cx), unsafe_allow_html=True)
    kpi(k1, f"Asian ({avg})", f"{res['price']:.6f}", f"± {res['std_error']:.6f} · {unit}", AMBER)
    kpi(k2, "95% interval", f"[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]",
        f"width {res['ci_hi'] - res['ci_lo']:.5f}")
    kpi(k3, "Geometric, exact", f"{kv['price']:.6f}",
        f"Kemna-Vorst · σ_G {kv['sigma_g'] * 100:.1f}%", BLUE)
    kpi(k4, "Vanilla Black-76", f"{van:.6f}", "same F, K, T, σ", GREEN)
    disc = (1 - res["price"] / van) * 100 if van > 0 else 0
    kpi(k5, "Asian discount", f"{disc:.1f}%", "cheaper than vanilla",
        GREEN if disc > 0 else RED)

    if res["control_variate"]:
        gain = plain["std_error"] / max(res["std_error"], 1e-12)
        st.markdown(
            f'<div class="note">Control variate active (β = {res["beta"]:.3f}): standard error '
            f'{plain["std_error"]:.6f} → <b>{res["std_error"]:.6f}</b>, a <b>{gain:.0f}×</b> '
            f'reduction for the same {res["n_paths"]:,} paths. Matching that precision by brute '
            f'force would take roughly <b>{gain ** 2:,.0f}×</b> more simulation.</div>',
            unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        paths = np.asarray(res["sample_paths"])
        times = np.linspace(T / n_obs, T, n_obs)
        fig = go.Figure()
        for row in paths[:30]:
            fig.add_trace(go.Scatter(x=times, y=row, mode="lines", opacity=0.22,
                                     line=dict(width=0.6, color=AMBER), showlegend=False))
        fig.add_trace(go.Scatter(x=times, y=paths.mean(axis=0), mode="lines",
                                 name="Mean path", line=dict(color=RED, width=2)))
        fig.add_hline(y=K, line=dict(color=GREEN, dash="dash"))
        fig.update_xaxes(title="Time (years)")
        fig.update_yaxes(title=f"Forward ({unit})")
        st.plotly_chart(styled(fig, 300, f"Simulated paths — {n_obs} fixings (30 shown)"),
                        **FULL)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["conv_paths"], y=res["conv_prices"],
                                 mode="lines+markers", name="Estimate",
                                 line=dict(color=AMBER, width=2), marker=dict(size=4)))
        fig.add_hline(y=res["price"], line=dict(color=GREEN, dash="dash"))
        fig.add_hline(y=kv["price"], line=dict(color=BLUE, dash="dot"))
        fig.update_xaxes(title="Paths", type="log")
        fig.update_yaxes(title=f"Price ({unit})")
        st.plotly_chart(styled(fig, 300, "Convergence — green: final, blue: geometric exact"),
                        **FULL)

    avgs = np.asarray(res["sample_avgs"])
    fig = go.Figure(go.Histogram(x=avgs, nbinsx=60, marker_color=AMBER, opacity=0.75,
                                 histnorm="probability density"))
    vline(fig, K, GREEN, f"K {K:,.2f}")
    vline(fig, F, BLUE, f"F {F:,.2f}", y=0.82)
    fig.update_xaxes(title=f"Average price at expiry ({unit})")
    fig.update_yaxes(title="Density")
    st.plotly_chart(styled(fig, 260, "Distribution of the average — narrower than the terminal "
                                     "price, which is exactly why the Asian is cheaper"),
                    **FULL)



def screen_barrier(cx: dict) -> None:
    section("Barrier options — knock-in and knock-out",
            "A barrier option switches on (<b>knock-in</b>) or dies (<b>knock-out</b>) if the "
            "forward touches a level. It is cheaper than the vanilla because you give something "
            "up, and that is the point: an airline buys a knock-out call to cap fuel cost while "
            "accepting the cap disappears in a spike. Priced three ways here — closed form, "
            "discrete-monitoring correction, and Monte Carlo — because they must agree.")
    F, T, r, unit = cx["F"], cx["T_opt"], cx["r"], cx["unit"]
    K, otype = _strike_and_side(cx, "br")
    vol = cx["vol_fn"](K, T)
    b1, b2, b3 = st.columns(3)
    btype = b1.selectbox("Barrier", BARRIER_TYPES)
    default_pct = 85.0 if btype.lower().startswith("down") else 115.0
    bpct = b2.number_input("Barrier (% of forward)", 20.0, 300.0, default_pct, 1.0, format="%.1f")
    H = round(F * bpct / 100, 6)
    rebate = b2.number_input(f"Rebate if knocked ({unit})", 0.0, float(F), 0.0, 0.01, format="%.4f")
    fixings = b3.select_slider("Monitoring", options=[12, 52, 252, 1000], value=252,
                               format_func=lambda x: {12: "Monthly", 52: "Weekly",
                                                      252: "Daily", 1000: "Near-continuous"}[x])
    b3.markdown(f'<div class="note" style="margin-top:6px">F {F:,.3f} · K {K:,.3f} · '
                f'B <b>{H:,.3f}</b> ({H / F * 100:.1f}% of F) · σ {vol * 100:.1f}% · '
                f'T {T:.3f}y</div>', unsafe_allow_html=True)

    n_fix = max(int(round(T * fixings)), 1)
    with st.spinner(f"Simulating {cx['n_paths']:,} paths over {n_fix} fixings…"):
        mc = barrier_mc(F, K, H, T, r, vol, otype, btype, rebate,
                        cx["n_paths"], n_fixings=n_fix)
    cont = barrier_analytic(F, K, H, T, r, vol, otype, btype, rebate)
    H_adj = barrier_discrete_correction(H, vol, T, n_fix, btype)
    corrected = barrier_analytic(F, K, H_adj, T, r, vol, otype, btype, rebate)
    van = Black76(F, K, T, r, vol, otype).price()

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Monte Carlo", f"{mc['price']:.5f}", f"± {mc['std_error']:.5f} · {n_fix} fixings", AMBER)
    kpi(k2, "Closed form", f"{corrected:.5f}", "continuous + BGK correction", BLUE)
    kpi(k3, "Vanilla", f"{van:.5f}", "same F, K, T, σ", GREEN)
    d = (1 - mc["price"] / van) * 100 if van > 0 else 0
    kpi(k4, "Discount", f"{d:+.1f}%", "vs vanilla", GREEN if d > 0 else RED)
    kpi(k5, "Knock probability", f"{mc['knock_prob'] * 100:.1f}%", "barrier touched", RED)

    gap = corrected - mc["price"]
    ok = abs(gap) <= max(3 * mc["std_error"], 0.02 * max(mc["price"], 1e-9))
    st.markdown(
        f'<div class="note" style="border-left:2px solid {GREEN if ok else AMBER}">'
        f'<b>Three prices, one option.</b> The continuous closed form gives {cont:.5f}; a '
        f'discretely monitored barrier is <i>harder</i> to breach, so Broadie-Glasserman-Kou '
        f'shifts it to {H_adj:,.4f} and the price becomes <b>{corrected:.5f}</b>, against '
        f'{mc["price"]:.5f} ± {mc["std_error"]:.5f} from simulation — a gap of {gap:+.5f}. '
        f'{"They agree." if ok else "They disagree by more than simulation noise; at these parameters trust the Monte Carlo."} '
        f'Quoting the continuous formula for a daily-fixing contract, as if the difference did '
        f'not exist, would misprice this option by {abs(cont - mc["price"]):.5f}.</div>',
        unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        paths = np.asarray(mc["paths"])
        breached = np.asarray(mc["breached"])
        t = np.linspace(T / n_fix, T, paths.shape[1])
        fig = go.Figure()
        for i in range(min(40, len(paths))):
            fig.add_trace(go.Scatter(x=t, y=paths[i], mode="lines", opacity=0.3,
                                     line=dict(width=0.6, color=RED if breached[i] else AMBER),
                                     showlegend=False))
        fig.add_hline(y=H, line=dict(color=RED, width=2, dash="dash"))
        fig.add_hline(y=K, line=dict(color=GREEN, width=1.2, dash="dot"))
        fig.update_xaxes(title="Time (years)")
        fig.update_yaxes(title=f"Forward ({unit})")
        st.plotly_chart(styled(fig, 320, "Paths — red touched the barrier, amber survived"),
                        **FULL)
    with g2:
        Hs = np.linspace(F * 0.5, F * 1.5, 40)
        vals = [barrier_analytic(F, K, float(h),
                                 T, r, vol, otype, btype, rebate) for h in Hs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Hs, y=vals, name="Barrier price",
                                 line=dict(color=AMBER, width=2.4)))
        fig.add_hline(y=van, line=dict(color=BLUE, dash="dot"))
        vline(fig, F, GREEN, f"F {F:,.2f}")
        vline(fig, H, RED, f"B {H:,.2f}", y=0.82)
        fig.update_xaxes(title=f"Barrier level ({unit})")
        fig.update_yaxes(title=f"Price ({unit})")
        st.plotly_chart(styled(fig, 320, "Price against barrier — blue is the vanilla ceiling"),
                        **FULL)

    rng = np.linspace(K * 0.45, K * 1.55, 400)
    vpnl = Black76(F, K, T, r, vol, otype).payoff(rng) - van
    bpnl = Black76(F, K, T, r, vol, otype).payoff(rng) - mc["price"]
    hit = rng <= H if btype.lower().startswith("down") else rng >= H
    if btype.lower().endswith("out"):
        bpnl = np.where(hit, rebate - mc["price"], bpnl)
    else:
        bpnl = np.where(hit, bpnl, rebate - mc["price"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rng, y=vpnl, name="Vanilla",
                             line=dict(color=GRAY, width=1.4, dash="dot"), opacity=0.7))
    fig.add_trace(go.Scatter(x=rng, y=bpnl, name="Barrier", line=dict(color=AMBER, width=2.4)))
    fig.add_hline(y=0, line=dict(color=GRAY, width=0.8))
    vline(fig, H, RED, f"B {H:,.2f}")
    vline(fig, K, GREEN, f"K {K:,.2f}", y=0.82)
    fig.update_xaxes(title=f"Price at expiry ({unit})")
    fig.update_yaxes(title=f"P&L ({unit})")
    st.plotly_chart(styled(fig, 270, "P&L at expiry — the terminal picture ignores the path, "
                                     "which is precisely what a barrier does not"),
                    **FULL)


def screen_curve(cx: dict) -> None:
    curve, unit, name = cx["curve"], cx["unit"], cx["name"]
    section("Forward curve",
            "The strip of listed delivery months, each with its real settle date and its "
            "<b>calendar</b> tenor. Everything on this desk prices off these points.")
    if curve.empty:
        st.error("**NO MARKET DATA** — no listed contract returned a settle. Nothing is drawn "
                 "in its place; switch the sidebar to a model curve if you want to price a "
                 "scenario instead.")
        return

    stt = curve_stats(curve)
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Front", f"{stt['f1']:,.4f}", f"{stt['front_label']} · {unit}", AMBER)
    kpi(k2, "Back", f"{stt['fn']:,.4f}", f"{stt['back_label']} · T={stt['tn']:.2f}y")
    kpi(k3, "Structure", stt["structure"],
        f"{stt['fn'] - stt['f1']:+,.4f} front to back",
        GREEN if stt["structure"] == "BACKWARDATION" else RED)
    kpi(k4, "Slope, annualised", f"{stt['slope_ann']:+.2f}%",
        "what a long pays (or earns) to roll",
        RED if stt["slope_ann"] > 0 else GREEN)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["label"], y=curve["price"], mode="lines+markers",
                             name=name, line=dict(color=AMBER, width=2.4), marker=dict(size=7)))
    if not cx["is_model"]:
        cfg = CONTRACTS[name]
        theo = [stt["f1"] * math.exp((cx["r"] + cfg["storage"] - cfg["convenience"])
                                     * (float(t) - stt["t1"])) for t in curve["T"]]
        fig.add_trace(go.Scatter(x=curve["label"], y=theo, mode="lines",
                                 name="Cost of carry (model)",
                                 line=dict(color=BLUE, width=1.4, dash="dot")))
    fig.update_yaxes(title=unit)
    st.plotly_chart(styled(fig, 360, f"{name} — live strip against a cost-of-carry shape"),
                    **FULL)
    if not cx["is_model"]:
        st.caption("The dotted line is what pure storage economics would imply from the front, "
                   "at the registry's storage and convenience defaults. The gap between it and "
                   "the market is the market's own view of scarcity — and it is a MODEL line, "
                   "which is why it is dotted and labelled.")

    carry = implied_carry(curve)
    disp = curve[["label", "expiry", "opt_expiry", "T", "price", "asof"]].copy()
    disp = disp.merge(carry[["label", "spread", "carry_ann"]], on="label", how="left") \
        if not carry.empty else disp.assign(spread=np.nan, carry_ann=np.nan)
    disp.columns = ["Contract", "Future expires", "Option expires", "T (yrs)",
                    f"Price ({unit})", "Settle date", "vs front", "Carry % ann."]
    disp = as_text(disp, {"Carry % ann.": "{:+.2f}", "Settle date": "{}"})
    st.dataframe(disp.style.format({"T (yrs)": "{:.3f}", f"Price ({unit})": "{:,.4f}",
                                    "vs front": "{:+,.4f}"}),
                 **FULL, hide_index=True, height=300)
    cfg = CONTRACTS[name]
    st.caption(f"T is the real year-fraction to the future's expiry. The future stops trading "
               f"{FUT_RULES[cfg['fut_rule']]}; its option expires {OPT_RULES[cfg['opt_rule']]}. "
               f"Weekends are handled, exchange holidays are not — a date can be a day early. "
               f"Verify against the exchange calendar before relying on one to the day.")


def screen_checks(cx: dict) -> None:
    section("Model self-validation",
            "Every model on this desk is priced a second way and the residual is shown. These "
            "are not decoration: a theta sign error and an inverted Kirk factor were both "
            "invisible to inspection and obvious to a check like these.")
    quick = st.toggle("Quick mode (fewer Monte Carlo paths)", value=True)
    with st.spinner("Running independent checks…"):
        rows = validate_models(cx["F"], cx["K"], cx["T_opt"], cx["r"], cx["vs"]["atm"], quick=quick,
                               vol_fn=cx["vol_fn"], atm=cx["vs"]["atm"])

    n_pass = sum(r["pass"] for r in rows)
    k1, k2 = st.columns([1, 3])
    kpi(k1, "Checks passed", f"{n_pass} / {len(rows)}", "independent recomputation",
        GREEN if n_pass == len(rows) else RED)
    with k2:
        df = pd.DataFrame([dict(Check=r["check"], Detail=r["detail"],
                                Residual=r["error"], Tolerance=r["tol"],
                                Benchmark=r["independent"],
                                Result="PASS" if r["pass"] else "FAIL") for r in rows])
        st.dataframe(df.style.format({"Residual": "{:.3e}", "Tolerance": "{:.3e}"}),
                     **FULL, hide_index=True)

    section("Kirk against simulation, across the parameter space")
    st.markdown('<div class="note">Kirk is an approximation, so the honest question is not '
                'whether it is exact but <b>where it stops being good</b>. Each row prices the '
                'same spread option both ways.</div>', unsafe_allow_html=True)
    rows2 = []
    for FL, FS, K, rho in [(100.0, 80.0, 15.0, 0.85), (100.0, 80.0, 25.0, 0.85),
                           (100.0, 80.0, 15.0, 0.40), (90.0, 88.0, 0.0, 0.95),
                           (100.0, 80.0, -5.0, 0.85)]:
        so = SpreadOption(FL, FS, K, max(cx["T_opt"], 0.1), cx["r"], 0.36, 0.32, rho, "call")
        kk = so.price()["price"]
        mm = so.mc_price(120_000)
        rows2.append(dict(F_long=FL, F_short=FS, K=K, rho=rho, Kirk=kk,
                          MonteCarlo=mm["price"], se=mm["std_error"],
                          err_pct=(kk - mm["price"]) / max(mm["price"], 1e-9) * 100))
    st.dataframe(pd.DataFrame(rows2).style.format(
        {"F_long": "{:,.1f}", "F_short": "{:,.1f}", "K": "{:+,.1f}", "rho": "{:.2f}",
         "Kirk": "{:.4f}", "MonteCarlo": "{:.4f}", "se": "{:.4f}", "err_pct": "{:+.2f}%"}),
        **FULL, hide_index=True)
    st.caption("Kirk is at its best when the effective strike stays well away from zero — deep "
               "in-the-money spreads with a large negative strike are where it drifts, and the "
               "simulation column is there so you can see it happen rather than assume it does not.")

    section("What this desk does not model, and why")
    st.markdown(
        '<div class="note">'
        '<b>Implied vol through a fund.</b> Free feeds carry no futures-option chains, so the '
        'surface is implied from options on the exchange-traded fund that holds the commodity '
        'and mapped to each futures month by realised-vol ratios. It assumes the fund\'s '
        'implied-to-realised relationship carries over — close for gold, looser for funds that '
        'roll a ladder of months. Contracts with no fund at all are forecast, not implied.'
        '<br><b>Exchange holidays.</b> Expiry dates follow each family\'s rule on weekdays; a '
        'holiday can move the real date a day earlier.'
        '<br><b>Discrete monitoring.</b> Barrier prices use daily fixings by default; the '
        'closed form is continuous, so the two are reconciled explicitly rather than blurred.'
        '<br><b>Flat rates.</b> One discount rate (the 13-week T-bill), no curve. For tenors '
        'out to two years on a commodity book the error is small; for longer it is not.'
        '<br><b>SPAN.</b> The margin figure is a scan-risk proxy with uniform shocks — no '
        'inter-month or inter-commodity credits, no exchange-set ranges.</div>',
        unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([dict(Instrument=k, Reason=v) for k, v in EXCLUSIONS.items()]),
                 **FULL, hide_index=True)



# ══════════════════════════════════════════════════════════════════════════════
#  9. NAVIGATION AND MAIN
#  The sidebar picks WHAT (contract, curve, month); everything else is fetched.
#  Only the screen you are on is computed.
# ══════════════════════════════════════════════════════════════════════════════

SCREENS = {
    "⛓ Chain": "screen_chain",
    "🧩 Trade": "screen_trade",
    "📚 Book": "screen_book",
    "🌊 Volatility": None,
    "🧱 Structures": None,
    "📈 Market": "screen_curve",
    "🔬 Checks": "screen_checks",
}


def _setup_page() -> None:
    st.set_page_config(page_title="CODAP — Commodity Options Desk",
                       page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
    theme = getattr(getattr(st, "context", None), "theme", None)
    st.markdown(css(getattr(theme, "type", None) != "light"), unsafe_allow_html=True)


def render_sidebar() -> dict:
    """Contract, curve, month — then the things that are fetched rather than set."""
    with st.sidebar:
        st.markdown(f'<div style="padding:8px 0 4px"><span style="font-size:1.3rem;'
                    f'font-weight:700;letter-spacing:-.03em;background:linear-gradient'
                    f'(90deg,{AMBER},{RED});-webkit-background-clip:text;'
                    f'-webkit-text-fill-color:transparent">CODAP</span>'
                    f'<div style="font-size:.7rem;color:#6E7681">Commodity options desk</div>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown("---")
        sector = st.selectbox("Sector", SECTORS, key="sb_sector")
        name = st.selectbox("Contract", contracts_in(sector), key="sb_contract")
        cfg = CONTRACTS[name]
        st.caption(f"{cfg['unit']} · {cfg['size']:,} {cfg['size_unit']}/lot · "
                   f"strikes every {cfg['strike_inc']:g}")

        rinfo = fetch_rate()
        r = rinfo["rate"]
        source = st.radio("Curve", ["Live market", "Model (cost of carry)"], key="sb_curve",
                          help="Live pulls the listed strip. Model builds a cost-of-carry curve "
                               "from a spot you enter — for offline pricing or a scenario.")
        is_model = source.startswith("Model")
        spot_in, storage, convenience = 0.0, cfg["storage"], cfg["convenience"]
        if is_model:
            guess = last_known_spot(name)
            if guess is None:
                st.caption("No live price reachable — type the spot you want to price from.")
            spot_in = st.number_input(f"Spot ({cfg['unit']})", min_value=1e-6,
                                      value=float(guess or 100.0), step=0.01, format="%.4f",
                                      key=f"sb_spot_{name}")
            storage = st.number_input("Storage (%/yr)", 0.0, 60.0, cfg["storage"] * 100, 0.1,
                                      key=f"sb_sto_{name}") / 100
            convenience = st.number_input("Convenience yield (%/yr)", 0.0, 60.0,
                                          cfg["convenience"] * 100, 0.1, key=f"sb_cy_{name}") / 100

        with st.spinner("Fetching the strip…"):
            curve = (model_curve(name, spot_in, r, storage, convenience, n=cfg["n_liquid"])
                     if is_model else fetch_curve(name))
        spec = None
        if not curve.empty:
            live = curve[curve["option_live"]].reset_index(drop=True)
            if not live.empty:
                days = (live["T_opt"] * 365.25).round().astype(int)
                default = int(np.argmax(days.to_numpy() >= 20)) if (days >= 20).any() else 0
                i = st.selectbox(
                    "Expiry", range(len(live)), index=default, key=f"sb_exp_{name}_{is_model}",
                    format_func=lambda j: f"{live['label'].iloc[j]} · option "
                                          f"{live['opt_expiry'].iloc[j]:%d %b} · {days.iloc[j]}d",
                    help="Listed months with a live option. The option's own expiry — "
                         "not the future's — sets its life.")
                spec = live.iloc[i].to_dict()

        st.markdown("---")
        st.markdown(f'<div style="font-size:.72rem;font-weight:600;color:{PURPLE};'
                    f'text-transform:uppercase;letter-spacing:.09em">Volatility · automatic</div>',
                    unsafe_allow_html=True)
        vol_slot = st.empty()

        with st.expander("Rate & simulation"):
            st.caption(f"Discount rate: {rinfo['detail']}" +
                       (f" ({rinfo['asof']})" if rinfo.get("asof") else ""))
            if st.checkbox("Override the rate", key="sb_r_ovr"):
                r = st.number_input("Discount rate (%)", -5.0, 30.0, round(rinfo["rate"] * 100, 2),
                                    0.05, format="%.2f", key="sb_r") / 100
            n_paths = st.select_slider("Monte Carlo paths",
                                       options=[10_000, 25_000, 50_000, 100_000], value=25_000,
                                       key="sb_paths")
        if st.button("↻  Refresh market data", **FULL):
            st.cache_data.clear()
            _failed_fetches().clear()
            st.rerun()
        with st.expander("🔧 Diagnostics"):
            st.code("\n".join(list(FEED_LOG)[-25:]) or "nothing logged", language="text")
            st.caption(f"Book file: {BOOK_PATH}")

    return dict(name=name, cfg=cfg, unit=cfg["unit"], sector=sector, is_model=is_model,
                spot_in=spot_in, storage=storage, convenience=convenience, r=r, r_info=rinfo,
                n_paths=n_paths, curve=curve, curves={name: curve}, spec=spec,
                vol_slot=vol_slot)


def build_context(cx: dict) -> dict:
    """Everything a screen needs, computed once per run."""
    spec = cx["spec"]
    cx["F"] = float(spec["price"])
    cx["T"] = float(spec["T"])
    cx["T_opt"] = max(float(spec["T_opt"]), 1 / 365)
    surf = vol_surface(cx["name"], cx["r"])
    cx["surface"] = surf
    cx["vol_fn"] = surface_vol_fn(surf, cx["F"], cx["T"])
    atm = surface_atm(surf, cx["T_opt"], cx["T"])
    cx["vs"] = dict(atm=atm, fn=cx["vol_fn"], source=surf["source"], label=surf["label"],
                    detail=surf["detail"], smile=surface_smile(surf, cx["T_opt"]) is not None)
    cx["K"] = snap_strike(cx["name"], cx["F"])
    cx["vol_K"] = cx["vol_fn"](cx["K"], cx["T_opt"])
    col = {"implied": GREEN, "forecast": AMBER, "registry": RED}[surf["source"]]
    cx["vol_slot"].markdown(
        f'<div style="font-size:.78rem;color:{INK};margin-top:4px">'
        f'<span style="color:{col};font-weight:600">{surf["label"]}</span><br>'
        f'ATM {atm:.1%} for {spec["label"]} · '
        f'{"SABR smile" if cx["vs"]["smile"] else "flat smile"}</div>', unsafe_allow_html=True)
    return cx


def render_header(cx: dict) -> None:
    curve, vs, spec = cx["curve"], cx.get("vs"), cx.get("spec")
    if cx["is_model"]:
        state, sub = badge("MODEL CURVE", AMBER), "cost of carry from your spot"
    elif curve.empty:
        state, sub = badge("NO MARKET DATA", RED), "no listed contract returned a settle"
    else:
        asof = max(d for d in curve["asof"] if d is not None)
        stale = curve_is_stale(curve)
        state = badge(f"{'STALE' if stale else 'LIVE'} · {asof}", AMBER if stale else GREEN)
        sub = f"{len(curve)} dated contracts marked"
    bits = [badge(cx["name"], INK), badge(cx["unit"]), state]
    if spec is not None and vs is not None:
        col = {"implied": GREEN, "forecast": AMBER, "registry": RED}[vs["source"]]
        bits += [badge(f"{spec['label']} {cx['F']:,.4f} · {cx['T_opt'] * 365.25:.0f}d", INK),
                 badge(f"σ {vs['atm'] * 100:.1f}%", PURPLE), badge(vs["label"], col),
                 badge("smile" if vs["smile"] else "flat smile", PURPLE)]
    bits.append(badge(f"r {cx['r'] * 100:.2f}%"))
    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 2px">'
                + "".join(bits) + "</div>", unsafe_allow_html=True)
    if vs is not None:
        st.caption(f"{sub} · volatility: {vs['detail']}")
    else:
        st.caption(sub)


def guard(label: str, fn, cx: dict) -> None:
    try:
        fn(cx)
    except Exception as e:                                         # noqa: BLE001
        LOG.exception("SCREEN ERROR %s: %s: %s", label.strip(), type(e).__name__, e)
        st.error(f"**This screen stopped.** The rest of the app still works.\n\n"
                 f"`{type(e).__name__}: {e}`")
        with st.expander("Details"):
            st.code(traceback.format_exc(), language="text")


def main() -> None:
    _setup_page()
    if not YF_AVAILABLE:
        st.warning("yfinance is not installed — no live data. `pip install yfinance`, or use "
                   "the model curve.")
    cx = render_sidebar()
    if cx["curve"].empty or cx["spec"] is None:
        render_header(cx)
        st.error("**No curve with a live option, so nothing can be priced.** Every model here "
                 "prices off a forward. Switch to a model curve in the sidebar to price a "
                 "scenario, or open Diagnostics in the sidebar.")
        return
    with st.spinner("Building the volatility surface from listed option prices…"):
        cx = build_context(cx)
    render_header(cx)

    screen = nav(list(SCREENS.keys()), "nav")
    if screen == "🌊 Volatility":
        sub = nav(["Surface", "Cones & gamma scalping"], "nav_vol")
        guard(sub, screen_vol_surface if sub == "Surface" else screen_vol_analytics, cx)
    elif screen == "🧱 Structures":
        subs = {"Cracks & crush": screen_structure, "Calendar spreads": screen_calendar,
                "Asian": screen_asian, "Barrier": screen_barrier}
        sub = nav(list(subs.keys()), "nav_struct")
        guard(sub, subs[sub], cx)
    else:
        guard(screen, globals()[SCREENS[screen]], cx)

    st.markdown("---")
    st.caption("CODAP · every price rests on a forward that was fetched and a volatility "
               "whose source is named on screen · by Adam El Gbouri")


if __name__ == "__main__":
    main()
