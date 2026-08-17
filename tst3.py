"""
CODAP — Commodity Options Desk
==============================
A working desk for paper options traders: an option chain you can deal off, a
trade ticket, a book with cash Greeks and margin, and the volatility work that
decides whether any of it is worth doing.

HOW THE VOLATILITY GETS THERE — the one thing worth reading first. An implied
volatility is not market data anybody can download; it is an option PRICE turned
inside out. So this app never pretends to fetch one. It ranks the ways of getting
one and tells you on every screen which is live:

    1. YOUR QUOTES    paste the premiums on your broker screen; each is inverted
                      to an implied vol and SABR is fitted to them. From then on
                      every price here comes off your market.
    2. REALISED SEED  the volatility this contract actually delivered over the
                      last 30 sessions, computed from free price history. Not an
                      implied vol, and labelled so — but a defensible start.
    3. REGISTRY       a constant in this file. The weakest source, named as such.

You can override the level by hand at any time, and turn the smile off to price
every strike flat — both are choices the header reports rather than hides.

DESIGN CONTRACT — market data is fetched, dated and never invented; a dead feed
says NO MARKET DATA and the screen stands down. Model inputs are labelled
wherever they touch a number. Eight independent checks run on demand and show
their residuals, because a pricer that cannot show its own error is one you have
to take on faith.

SCREENS, in the order a trader opens them:

    Chain        the strike ladder on the contract's real increment grid, the
                 ticket, and the paste-your-prices calibration
    Trade        one option or a structure, strikes by price / % / delta
    Book         cash Greeks, vega bucketed by tenor, scenario grid, SPAN margin
    Volatility   SABR surface · realised-vol cones · gamma scalping
    Structures   cracks and crush · calendar spreads · Asians · barriers
    Market       the forward curve behind every price
    Checks       model validation, stated limits, and what was excluded

LAYOUT — reads top-down in dependency order; nothing below is needed above:

    1. REGISTRY & CALENDAR    contracts, expiry rules, calendar tenor, strike grid
    2. MARKET DATA            grouped download, dated marks, no fabrication
    3. PRICING MODELS         European, American, Asian, spreads, barriers, SABR
    4. VOLATILITY ANALYTICS   realised-vol cones, forward vol, gamma scalping
    5. STRATEGIES & BOOK      structures, cash Greeks, vega buckets, scenarios
    6. SELF-VALIDATION        independent checks on every model
    6b. VOLATILITY SOURCE     quotes > realised > registry, resolved and named
    7. UI FOUNDATION          chrome, and the folded-explanation pattern
    8. SCREENS                one function per screen
    9. NAVIGATION & MAIN      router, with a per-screen guard

Run:    streamlit run codap.py
Tests:  pytest test_codap.py -q

by Adam EL GBOURI
"""
from __future__ import annotations

import logging
import math
import traceback
from calendar import monthrange
from collections import deque
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  1. REGISTRY & CALENDAR
#  Static truth: palette, the contracts, their delivery cycles and expiry rules.
#  Validated at import — a typo here would misprice every option downstream, so
#  it fails loudly at start-up rather than quietly on screen.
# ══════════════════════════════════════════════════════════════════════════════

BG, PANEL, BORDER = "#0D1117", "#161B22", "#30363D"
TEXT, GRAY = "#E6EDF3", "#8B949E"
AMBER, BLUE, GREEN, RED, PURPLE, TEAL = (
    "#F0A500", "#58A6FF", "#3FB950", "#FF7B72", "#BC8CFF", "#39D0D8")

MONTH_CODES = list("FGHJKMNQUVXZ")
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Expiry rules. Energy dies BEFORE its delivery month, Chicago grains trade INTO
# it, COMEX metals almost to its end, and ICE Brent expires two months early —
# which is why its front contract is never the front calendar month.
_EXPIRY_RULES = {
    "prec_25":   "~25th of the month before delivery",
    "prec_eom":  "last day of the month before delivery",
    "prec2_eom": "last day of month M-2 (ICE Brent)",
    "del_15":    "~15th of the delivery month",
    "del_20":    "~20th of the delivery month",
    "del_eom":   "last day of the delivery month",
}

CONTRACTS: Dict[str, dict] = {
    # ── Energy ────────────────────────────────────────────────────────────────
    "WTI Crude (CL)": dict(
        sector="Energy", unit="$/bbl", yf="CL{M}{YY}.NYM", size=1_000,
        size_unit="bbl", divisor=1.0, months="FGHJKMNQUVXZ", n_liquid=18,
        expiry="prec_25", vol=0.32, storage=0.096, convenience=0.08, strike_inc=0.5),
    "Brent Crude (BZ)": dict(
        sector="Energy", unit="$/bbl", yf="BZ{M}{YY}.NYM", size=1_000,
        size_unit="bbl", divisor=1.0, months="FGHJKMNQUVXZ", n_liquid=18,
        expiry="prec2_eom", vol=0.30, storage=0.096, convenience=0.07, strike_inc=0.5),
    "Henry Hub Gas (NG)": dict(
        sector="Energy", unit="$/MMBtu", yf="NG{M}{YY}.NYM", size=10_000,
        size_unit="MMBtu", divisor=1.0, months="FGHJKMNQUVXZ", n_liquid=18,
        expiry="prec_eom", vol=0.55, storage=0.120, convenience=0.10, strike_inc=0.05),
    "RBOB Gasoline (RB)": dict(
        sector="Energy", unit="$/gal", yf="RB{M}{YY}.NYM", size=42_000,
        size_unit="gal", divisor=1.0, months="FGHJKMNQUVXZ", n_liquid=12,
        expiry="prec_eom", vol=0.36, storage=0.084, convenience=0.07, strike_inc=0.01),
    "ULSD Heating Oil (HO)": dict(
        sector="Energy", unit="$/gal", yf="HO{M}{YY}.NYM", size=42_000,
        size_unit="gal", divisor=1.0, months="FGHJKMNQUVXZ", n_liquid=12,
        expiry="prec_eom", vol=0.34, storage=0.084, convenience=0.07, strike_inc=0.01),
    # ── Metals ────────────────────────────────────────────────────────────────
    "Gold (GC)": dict(
        sector="Metals", unit="$/troy oz", yf="GC{M}{YY}.CMX", size=100,
        size_unit="troy oz", divisor=1.0, months="GJMQVZ", n_liquid=8,
        expiry="del_eom", vol=0.15, storage=0.024, convenience=0.005, strike_inc=5.0),
    "Silver (SI)": dict(
        sector="Metals", unit="$/troy oz", yf="SI{M}{YY}.CMX", size=5_000,
        size_unit="troy oz", divisor=1.0, months="HKNUZ", n_liquid=6,
        expiry="del_eom", vol=0.28, storage=0.036, convenience=0.010, strike_inc=0.25),
    "Copper (HG)": dict(
        sector="Metals", unit="$/lb", yf="HG{M}{YY}.CMX", size=25_000,
        size_unit="lb", divisor=1.0, months="HKNUZ", n_liquid=8,
        expiry="del_eom", vol=0.22, storage=0.048, convenience=0.030, strike_inc=0.05),
    "Platinum (PL)": dict(
        sector="Metals", unit="$/troy oz", yf="PL{M}{YY}.NYM", size=50,
        size_unit="troy oz", divisor=1.0, months="FJNV", n_liquid=6,
        expiry="del_eom", vol=0.20, storage=0.030, convenience=0.015, strike_inc=5.0),
    "Palladium (PA)": dict(
        sector="Metals", unit="$/troy oz", yf="PA{M}{YY}.NYM", size=100,
        size_unit="troy oz", divisor=1.0, months="HMUZ", n_liquid=6,
        expiry="del_eom", vol=0.30, storage=0.030, convenience=0.020, strike_inc=5.0),
    # ── Grains & oilseeds (quoted in CENTS — divisor 100) ─────────────────────
    "Corn (ZC)": dict(
        sector="Grains", unit="c/bu", yf="ZC{M}{YY}.CBT", size=5_000,
        size_unit="bu", divisor=100.0, months="HKNUZ", n_liquid=8,
        expiry="del_15", vol=0.25, storage=0.060, convenience=0.04, strike_inc=10.0),
    "Wheat CBOT (ZW)": dict(
        sector="Grains", unit="c/bu", yf="ZW{M}{YY}.CBT", size=5_000,
        size_unit="bu", divisor=100.0, months="HKNUZ", n_liquid=8,
        expiry="del_15", vol=0.28, storage=0.060, convenience=0.04, strike_inc=10.0),
    "Soybeans (ZS)": dict(
        sector="Grains", unit="c/bu", yf="ZS{M}{YY}.CBT", size=5_000,
        size_unit="bu", divisor=100.0, months="FHKNQUX", n_liquid=8,
        expiry="del_15", vol=0.23, storage=0.060, convenience=0.05, strike_inc=20.0),
    "Soybean Meal (ZM)": dict(
        sector="Grains", unit="$/short ton", yf="ZM{M}{YY}.CBT", size=100,
        size_unit="short ton", divisor=1.0, months="FHKNQUVZ", n_liquid=8,
        expiry="del_15", vol=0.24, storage=0.060, convenience=0.05, strike_inc=5.0),
    "Soybean Oil (ZL)": dict(
        sector="Grains", unit="c/lb", yf="ZL{M}{YY}.CBT", size=60_000,
        size_unit="lb", divisor=100.0, months="FHKNQUVZ", n_liquid=8,
        expiry="del_15", vol=0.26, storage=0.060, convenience=0.05, strike_inc=0.5),
    # ── Softs & livestock ────────────────────────────────────────────────────
    "Sugar #11 (SB)": dict(
        sector="Softs", unit="c/lb", yf="SB{M}{YY}.NYB", size=112_000,
        size_unit="lb", divisor=100.0, months="HKNV", n_liquid=6,
        expiry="prec_eom", vol=0.30, storage=0.048, convenience=0.04, strike_inc=0.25),
    "Arabica Coffee (KC)": dict(
        sector="Softs", unit="c/lb", yf="KC{M}{YY}.NYB", size=37_500,
        size_unit="lb", divisor=100.0, months="HKNUZ", n_liquid=6,
        expiry="del_20", vol=0.35, storage=0.048, convenience=0.05, strike_inc=2.5),
    "Cocoa (CC)": dict(
        sector="Softs", unit="$/mt", yf="CC{M}{YY}.NYB", size=10,
        size_unit="mt", divisor=1.0, months="HKNUZ", n_liquid=6,
        expiry="del_15", vol=0.32, storage=0.048, convenience=0.04, strike_inc=50.0),
    "Live Cattle (LE)": dict(
        sector="Livestock", unit="c/lb", yf="LE{M}{YY}.CME", size=40_000,
        size_unit="lb", divisor=100.0, months="GJMQVZ", n_liquid=8,
        expiry="del_eom", vol=0.18, storage=0.036, convenience=0.03, strike_inc=1.0),
    "Lean Hogs (HE)": dict(
        sector="Livestock", unit="c/lb", yf="HE{M}{YY}.CME", size=40_000,
        size_unit="lb", divisor=100.0, months="GJKMNQVZ", n_liquid=6,
        expiry="del_15", vol=0.25, storage=0.036, convenience=0.03, strike_inc=1.0),
}

# What was deliberately dropped from the previous registry, and why. Stated here
# rather than left as broken entries: every one of these had a ticker that does
# not resolve, so the app silently fell back to randomly generated prices and
# presented them in the same charts as real ones.
EXCLUSIONS = {
    "LME Copper / Aluminium / Zinc / Nickel / Lead / Tin":
        "LME contracts do not price off dated month codes the way these tickers "
        "assumed, and no free feed returns them — every curve was fabricated.",
    "Capesize / Panamax freight (BCI, BPI)":
        "Baltic Exchange assessments are licensed data. No feed, no curve.",
    "EU / UK Carbon (EUA, UKA)":
        "ICE carbon futures were listed twice under two different families and "
        "neither ticker resolved.",
    "Coal API2, Uranium UxC, Gasoil ICE, Jet Kero CIF NWE":
        "Platts and Argus assessments are licensed; the TradingView symbols used "
        "for them returned nothing and fell through to synthetic prices.",
    "Fixed-for-floating swaps":
        "Removed with their tab: a commodity swap is corporate treasury, not a "
        "paper trader's instrument, and the option chain that replaced it is the "
        "screen this audience actually works on.",
    "Parametric vol surface (polynomial)":
        "Replaced by SABR, which extrapolates sensibly into the wings and whose "
        "parameters mean something a trader can argue about. The old surface was "
        "left unreachable behind the new tab — dead code, now deleted.",
    "Dutch TTF Natural Gas":
        "Referenced by the spark-spread definition but absent from the registry, "
        "so it silently priced off a $100 placeholder.",
}

_REQUIRED_KEYS = {"sector", "unit", "yf", "size", "size_unit", "divisor",
                  "months", "n_liquid", "expiry", "vol", "storage", "convenience",
                  "strike_inc"}


def _validate_registry(reg: Dict[str, dict]) -> None:
    """Fail at import on a malformed contract. The previous registry carried
    silent typos (a spark spread pointing at a commodity that did not exist);
    a strict check is cheaper than discovering it in a price."""
    assert reg, "contract registry is empty"
    for name, c in reg.items():
        missing = _REQUIRED_KEYS - set(c)
        assert not missing, f"{name}: missing keys {sorted(missing)}"
        unknown = set(c) - _REQUIRED_KEYS
        assert not unknown, f"{name}: unknown keys {sorted(unknown)}"
        assert c["expiry"] in _EXPIRY_RULES, f"{name}: bad expiry rule {c['expiry']!r}"
        assert set(c["months"]) <= set(MONTH_CODES), f"{name}: bad month codes"
        assert c["months"], f"{name}: no delivery months"
        assert c["divisor"] in (1.0, 100.0), f"{name}: divisor must be 1 or 100"
        assert c["size"] > 0 and c["n_liquid"] > 0, f"{name}: bad size/n_liquid"
        assert 0 < c["vol"] < 3, f"{name}: implausible default vol"
        assert 0 <= c["storage"] < 1 and 0 <= c["convenience"] < 1, f"{name}: bad carry"
        assert c["strike_inc"] > 0, f"{name}: bad strike increment"


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
    """Round a strike onto the contract's listed grid.

    Listed options do not trade at 82.4713. They trade on an increment — half a
    dollar on WTI, five cents on gas, five dollars on gold, ten cents on corn —
    and a strike off that grid is a strike you cannot actually deal. Everything
    the platform derives from a percentage or a delta is snapped before it is
    priced, so what you see is quotable.

    The increments are INDICATIVE registry defaults: exchanges change them, and
    they vary by expiry and by how far out of the money the strike sits. Verify
    against the current contract specs before dealing on one.
    """
    inc = CONTRACTS[name]["strike_inc"]
    return round(round(float(K) / inc) * inc, 10)


def notional_per_lot(name: str, price: float) -> float:
    c = CONTRACTS[name]
    return price / c["divisor"] * c["size"]


def _eom(y: int, m: int) -> date:
    return date(y, m, monthrange(y, m)[1])


def _shift_month(y: int, m: int, k: int) -> Tuple[int, int]:
    idx = (y * 12 + (m - 1)) + k
    return idx // 12, idx % 12 + 1


def estimate_expiry(rule: str, dy: int, dm: int) -> date:
    """Last trading day for a delivery month, per contract family. Approximate by
    design (exchange calendars move for holidays) and used only to drop expired
    contracts and to compute tenor — both tolerant of a day or two."""
    if rule == "prec_25":
        y, m = _shift_month(dy, dm, -1)
        return date(y, m, min(25, monthrange(y, m)[1]))
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
    return _eom(dy, dm)


def strip_specs(name: str, today: Optional[date] = None) -> List[dict]:
    """The live strip: every listed delivery month with its ticker, its expiry and
    its CALENDAR tenor.

    T is the real year-fraction to expiry, not the contract's position in the
    list. That distinction is the single most consequential fix in this file:
    platinum lists only Jan/Apr/Jul/Oct, so its second contract is roughly five
    months out, not two. Pricing it at 2/12 understates the tenor by ~60% and,
    since an option is worth roughly √T, underprices it by about a third.
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
        exp = estimate_expiry(c["expiry"], y, m)
        if exp <= today:
            continue
        T = (exp - today).days / 365.25
        out.append(dict(
            ticker=c["yf"].replace("{M}", code).replace("{YY}", f"{y % 100:02d}"),
            label=f"{MONTH_NAMES[m - 1]}-{y}",
            delivery=f"{y}-{m:02d}",
            expiry=exp,
            T=round(T, 6),
            seq=len(out) + 1,
        ))
    return out


# ── Multi-leg structures. Every leg carries its own conversion to the common
#    quote unit, so a 3-2-1 crack is the whole 3-2-1 crack — the previous version
#    dropped the heating-oil leg entirely and printed -12.80 $/bbl where the
#    correct margin was +22.20.
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
#  The previous version fell back to np.random prices on any failure and drew
#  them in the same charts as real settles; here a dead feed produces an empty
#  curve and a message, and a model curve is available only on explicit request.
# ══════════════════════════════════════════════════════════════════════════════

LOG = logging.getLogger("codap")
LOG.setLevel(logging.INFO)
FEED_LOG: deque = deque(maxlen=80)


class _RingHandler(logging.Handler):
    def emit(self, record):
        FEED_LOG.append(f"{datetime.now():%H:%M:%S}  {record.levelname:<7} "
                        f"{record.getMessage()}")


if not any(isinstance(h, _RingHandler) for h in LOG.handlers):
    LOG.addHandler(_RingHandler())

STALE_DAYS = 4


@st.cache_data(ttl=900, show_spinner=False)
def fetch_curve(name: str) -> pd.DataFrame:
    """Live forward curve for one commodity — ONE grouped request for the whole
    strip. Returns an empty frame when nothing comes back; the caller shows that
    as NO MARKET DATA rather than inventing a shape.

    Every row keeps the settle date it actually printed, because a two-day-old
    settle on a thin deferred month is honest information and a two-day-old
    settle presented as today's is not.
    """
    specs = strip_specs(name)
    if not YF_AVAILABLE or not specs:
        LOG.warning("no yfinance / no listed contracts for %s", name)
        return pd.DataFrame()
    tickers = [s["ticker"] for s in specs]
    try:
        raw = yf.download(tickers, period="10d", auto_adjust=True,
                          progress=False, group_by="column", threads=True)
    except Exception as e:                                        # noqa: BLE001
        LOG.warning("download failed for %s: %s", name, e)
        return pd.DataFrame()
    if raw is None or raw.empty:
        LOG.warning("empty payload for %s", name)
        return pd.DataFrame()

    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex) and len(tickers) == 1:
        closes.columns = tickers

    rows = []
    for s in specs:
        if s["ticker"] not in closes.columns:
            continue
        ser = closes[s["ticker"]].dropna()
        if ser.empty:
            continue
        rows.append({**s, "price": float(ser.iloc[-1]),
                     "asof": ser.index[-1].date()})
    if len(rows) < 2:
        LOG.warning("%s: only %d dated contract(s) returned a settle", name, len(rows))
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
    df["seq"] = range(1, len(df) + 1)
    LOG.info("%s: %d/%d contracts marked", name, len(df), len(specs))
    return df


def curve_is_stale(df: pd.DataFrame) -> bool:
    if df.empty or "asof" not in df:
        return False
    return (date.today() - max(df["asof"])).days > STALE_DAYS


def model_curve(name: str, spot: float, r: float, storage: float,
                convenience: float, n: int = 12) -> pd.DataFrame:
    """A cost-of-carry curve from a spot price YOU supply.

    F(T) = S · exp((r + storage − convenience) · T)

    This is a model, and the app labels every screen it touches MODEL CURVE. It
    exists so the pricer stays usable with no connection, or to price a scenario
    that is not the market — never to stand in for a market price behind the
    user's back. Tenors are the real listed ones, so the shape still respects the
    contract's delivery cycle.
    """
    specs = strip_specs(name)[:n]
    if not specs:
        return pd.DataFrame()
    rows = [{**s, "price": float(spot * math.exp((r + storage - convenience) * s["T"])),
             "asof": None} for s in specs]
    return pd.DataFrame(rows)


def curve_stats(df: pd.DataFrame) -> dict:
    """Front, back, slope and structure label for a curve of any length."""
    if df.empty:
        return dict(available=False)
    f1 = float(df["price"].iloc[0])
    fn = float(df["price"].iloc[-1])
    t1 = float(df["T"].iloc[0])
    tn = float(df["T"].iloc[-1])
    dT = max(tn - t1, 1e-9)
    slope_ann = (fn - f1) / f1 / dT * 100
    if abs(fn - f1) / f1 < 0.005:
        structure = "FLAT"
    elif fn > f1:
        structure = "CONTANGO"
    else:
        structure = "BACKWARDATION"
    return dict(available=True, f1=f1, fn=fn, t1=t1, tn=tn,
                slope_ann=slope_ann, structure=structure,
                front_label=str(df["label"].iloc[0]),
                back_label=str(df["label"].iloc[-1]))


def forward_at(df: pd.DataFrame, T: float) -> Optional[float]:
    """Forward price at an arbitrary tenor, log-linearly interpolated on calendar
    T and held flat beyond the last listed month. Interpolating in log space
    keeps a contango curve monotone instead of bending it through zero."""
    if df.empty:
        return None
    Ts = df["T"].to_numpy(dtype=float)
    Ps = df["price"].to_numpy(dtype=float)
    if np.any(Ps <= 0):
        return float(np.interp(T, Ts, Ps))
    return float(np.exp(np.interp(T, Ts, np.log(Ps))))


def implied_carry(df: pd.DataFrame) -> pd.DataFrame:
    """Annualised carry from the front to each listed month, on calendar tenor.

    Positive = contango, and a long position PAYS that rate to roll; negative =
    backwardation, and the roll pays the long. Annualising on the sequence index
    instead of real tenor — the old behaviour — roughly doubles the number on any
    contract that does not list every month.
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame()
    f1 = float(df["price"].iloc[0])
    t1 = float(df["T"].iloc[0])
    rows = []
    for r_ in df.itertuples():
        dT = float(r_.T) - t1
        rows.append(dict(
            label=r_.label, T=float(r_.T), price=float(r_.price),
            spread=float(r_.price) - f1,
            carry_ann=((float(r_.price) - f1) / f1 / dT * 100) if dT > 1e-9 else np.nan))
    return pd.DataFrame(rows)


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
                   vols: Optional[Dict[str, float]] = None) -> dict:
    """Collapse a multi-leg structure into the two synthetic forwards Kirk needs.

    Each leg is converted to the structure's common unit by its own factor, so a
    3-2-1 crack carries BOTH products: (2·RB·42 + 1·HO·42 − 3·CL)/3. Dropping the
    distillate leg — the previous behaviour — turned a +22.20 $/bbl refining
    margin into −12.80 $/bbl, a sign error large enough to invert the trade.

    The blended volatility is a value-weighted average across the legs of a side.
    That is an approximation (it ignores the correlation between the two products)
    and it is stated as one on the page.
    """
    cfg = STRUCTURES[struct]
    vols = vols or {}
    div = float(cfg["divide_by"])

    def side(legs):
        total, wvol, missing = 0.0, 0.0, []
        detail = []
        for name, qty, conv in legs:
            px = prices.get(name)
            if px is None or not np.isfinite(px):
                missing.append(name)
                continue
            val = px * qty * conv / div
            total += val
            v = vols.get(name, CONTRACTS[name]["vol"])
            wvol += abs(val) * v
            detail.append(dict(leg=name, qty=qty, conv=conv, price=px, value=val, vol=v))
        blended = (wvol / abs(total)) if abs(total) > 1e-12 else 0.0
        return total, blended, missing, detail

    FL, sL, miss_l, det_l = side(cfg["long"])
    FS, sS, miss_s, det_s = side(cfg["short"])
    missing = miss_l + miss_s
    return dict(available=not missing, missing=missing,
                F_long=FL, F_short=FS, sigma_long=sL, sigma_short=sS,
                spread=FL - FS, unit=cfg["unit"], rho=cfg["rho"],
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


def sabr_calibrate(F, T, strikes, vols, beta=0.5) -> dict:
    """Least-squares fit of ρ and ν to a strip of quotes, α pinned to ATM.

    Started from several initial guesses because the objective has local minima —
    a single start converges to a plausible-looking wrong answer often enough to
    matter."""
    strikes = np.asarray(strikes, dtype=float)
    vols = np.asarray(vols, dtype=float)
    order = np.argsort(strikes)
    strikes, vols = strikes[order], vols[order]
    atm = float(np.interp(F, strikes, vols))

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
    """Continuous front-month closes, for realised-volatility work only.

    Stated caveat: this series is NOT roll-adjusted, so each roll injects one
    artificial jump. That inflates realised vol slightly — a few tenths of a
    point on a monthly-rolling contract. It is the right series for a vol cone
    (which needs a decade of consistent history) and the wrong one for a P&L.
    """
    tk = CONTINUOUS_TICKER.get(name)
    if not YF_AVAILABLE or not tk:
        return pd.Series(dtype=float)
    try:
        raw = yf.download(tk, period=period, auto_adjust=True, progress=False)
    except Exception as e:                                         # noqa: BLE001
        LOG.warning("history failed for %s: %s", name, e)
        return pd.Series(dtype=float)
    if raw is None or raw.empty:
        LOG.warning("no history for %s (%s)", name, tk)
        return pd.Series(dtype=float)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


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
#  5. STRATEGIES, BOOK AND SCENARIOS
#  Nobody trades one option. A desk trades structures, holds a book of them, and
#  looks at the whole thing across a grid of price and volatility.
# ══════════════════════════════════════════════════════════════════════════════

def as_vol_fn(v):
    """Normalise a volatility input into a callable σ(K, T).

    The whole platform prices through this. Pass a number and every strike gets
    the same volatility — a flat surface, which is a MODEL choice, not a
    simplification to be made silently. Pass a function and each strike gets its
    own σ(K,T) off the SABR surface, which is the point of having built one: a
    25-delta put priced at the at-the-money volatility is exactly the error a
    smile exists to prevent, and before this refactor the platform made it in
    every tab except the surface itself.
    """
    if callable(v):
        return v
    x = float(v)
    return lambda K, T, _x=x: _x


def sabr_vol_fn(params: dict, F: float, shift: float = 0.0):
    """σ(K,T) off a SABR parameter set, with an optional parallel shift in vol
    points — the shift is what the scenario grid moves."""
    def f(K, T):
        v = sabr_vol(F, max(float(K), 1e-9), max(float(T), 1e-9),
                     params["alpha"], params["beta"], params["rho"], params["nu"])
        if not np.isfinite(v):
            v = params.get("atm", 0.3)
        return max(v + shift, 0.005)
    return f


def shifted_vol_fn(vol_fn, shift: float):
    base = as_vol_fn(vol_fn)
    return lambda K, T: max(base(K, T) + shift, 0.005)


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


def position_value(pos: dict, F: float, sigma, T: float, r: float,
                   n_steps: int = 150) -> float:
    """Value one book line at an arbitrary market state. `sigma` is a number or a
    σ(K,T) callable."""
    if pos["kind"] == "future":
        return pos["qty"] * (F - pos["entry"]) * price_multiplier(pos["contract"])
    mult = price_multiplier(pos["contract"])
    sig = as_vol_fn(sigma)(pos["strike"], T)
    if pos.get("style") == "American":
        px = crr_price(F, pos["strike"], T, r, sig, pos["type"], n_steps)
    else:
        px = Black76(F, pos["strike"], T, r, sig, pos["type"]).price()
    return pos["qty"] * (px - pos["entry"]) * mult


def book_greeks(positions: List[dict], F: float, sigma, r: float,
                n_steps: int = 150) -> dict:
    """Aggregate the book into cash Greeks, and bucket vega by tenor.

    The bucketing is the point. A book is never simply "long vega": it is long
    the front and short the back, or the reverse, and a single net number hides
    the calendar risk that actually moves a vol desk's P&L.
    """
    vf = as_vol_fn(sigma)
    buckets = {"0-3M": 0.0, "3-6M": 0.0, "6-12M": 0.0, "12M+": 0.0}
    tot = dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, value=0.0, premium=0.0)
    rows = []
    for p in positions:
        mult = price_multiplier(p["contract"])
        T = max(float(p["T"]), 1e-6)
        if p["kind"] == "future":
            g = dict(price=F, delta=1.0, gamma=0.0, vega=0.0, theta=0.0)
            sig = float("nan")
        else:
            sig = vf(p["strike"], T)
            if p.get("style") == "American":
                g = american_greeks(F, p["strike"], T, r, sig, p["type"], n_steps)
            else:
                g = Black76(F, p["strike"], T, r, sig, p["type"]).greeks()
        q = p["qty"]
        d_cash = q * g["delta"] * mult * F
        ga_cash = q * g["gamma"] * mult * F ** 2 / 100
        v_cash = q * g["vega"] * mult
        t_cash = q * g["theta"] * mult
        val = q * (g["price"] - p["entry"]) * mult
        tot["delta"] += d_cash
        tot["gamma"] += ga_cash
        tot["vega"] += v_cash
        tot["theta"] += t_cash
        tot["value"] += val
        tot["premium"] += q * g["price"] * mult
        b = ("0-3M" if T <= 0.25 else "3-6M" if T <= 0.5 else
             "6-12M" if T <= 1.0 else "12M+")
        buckets[b] += v_cash
        rows.append(dict(Line=_pos_label(p), Qty=q, Strike=p.get("strike"),
                         T=T, Vol=sig, Bucket=b, Price=g["price"], DeltaCash=d_cash,
                         GammaCash=ga_cash, VegaCash=v_cash, ThetaCash=t_cash,
                         MTM=val))
    return dict(total=tot, buckets=buckets, rows=rows)


def _pos_label(p: dict) -> str:
    if p["kind"] == "future":
        return f"{p['contract']} future × {p['qty']:+d}"
    return (f"{p['contract']} {p['type']} {p['strike']:,.2f} "
            f"{p['T'] * 12:.1f}M {p.get('style', 'European')[0]} × {p['qty']:+d}")


def scenario_matrix(positions: List[dict], F: float, sigma, r: float,
                    price_pct=(-20, -10, -5, 0, 5, 10, 20),
                    vol_pts=(-10, -5, 0, 5, 10), n_steps: int = 120) -> pd.DataFrame:
    """Book P&L across a grid of price and volatility moves.

    This is the screen an options desk actually watches. A single delta number
    says nothing about a book that is short the wings; the grid shows where the
    position breaks, in both dimensions at once, and it is the same computation
    the exchange uses to set margin — which is why the margin function below
    reads straight off it.
    """
    vf = as_vol_fn(sigma)
    base = sum(position_value(p, F, vf, max(p["T"], 1e-6), r, n_steps)
               for p in positions)
    data = {}
    for dv in vol_pts:
        s2 = shifted_vol_fn(vf, dv / 100.0)     # parallel shift of the WHOLE surface
        row = {}
        for dp in price_pct:
            F2 = F * (1 + dp / 100.0)
            v = sum(position_value(p, F2, s2, max(p["T"], 1e-6), r, n_steps)
                    for p in positions)
            row[f"{dp:+d}%"] = v - base
        data[f"{dv:+d} vols"] = row
    return pd.DataFrame(data).T


def span_margin(positions: List[dict], F: float, sigma, r: float,
                scan_pct: float = 5.0, vol_pts: float = 10.0,
                n_steps: int = 120) -> dict:
    """SPAN-style scan risk: the worst loss across the exchange's standard grid.

    Sixteen scenarios — the price range in thirds, up and down, each with
    volatility up and down, plus two extreme moves at double the range counted
    at 35% weight. The margin is the largest loss. This is a PROXY: real SPAN
    adds inter-month and inter-commodity credits and uses exchange-set ranges, so
    treat the number as an order of magnitude, which is what it is labelled.

    It matters because a trader's binding constraint is margin, not notional —
    and return on margin is how a structure is actually judged.
    """
    vf = as_vol_fn(sigma)
    base = sum(position_value(p, F, vf, max(p["T"], 1e-6), r, n_steps)
               for p in positions)
    scen = []
    for frac in (0.0, 1 / 3, -1 / 3, 2 / 3, -2 / 3, 1.0, -1.0):
        for dv in (vol_pts, -vol_pts):
            scen.append((frac * scan_pct, dv, 1.0))
    scen.append((2 * scan_pct, 0.0, 0.35))
    scen.append((-2 * scan_pct, 0.0, 0.35))
    worst, detail = 0.0, []
    for dp, dv, w in scen:
        F2 = F * (1 + dp / 100.0)
        s2 = shifted_vol_fn(vf, dv / 100.0)
        pnl = sum(position_value(p, F2, s2, max(p["T"], 1e-6), r, n_steps)
                  for p in positions) - base
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

def validate_models(F, K, T, r, sigma, n_obs=12, quick=True) -> List[dict]:
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

    for row in rows:
        row["pass"] = bool(row["error"] <= row["tol"])
    return rows



def option_chain(name: str, F: float, T: float, r: float, vol_fn,
                 n_strikes: int = 15, american: bool = False) -> pd.DataFrame:
    """The screen an options trader lives on: every listed strike around the money,
    calls on one side, puts on the other.

    Strikes come off the contract's own increment grid rather than a percentage
    sweep, so every row is an option that actually exists. Each is priced at its
    own σ(K) from the active surface.
    """
    vf = as_vol_fn(vol_fn)
    inc = CONTRACTS[name]["strike_inc"]
    atm = snap_strike(name, F)
    half = max(int(n_strikes) // 2, 1)
    strikes = [atm + i * inc for i in range(-half, half + 1) if atm + i * inc > 0]
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
            atm=abs(K - atm) < inc / 2))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  6b. VOLATILITY SOURCE — where the number actually comes from
#  An implied volatility is not market data you can download; it is an option
#  PRICE turned inside out. So the platform never pretends to fetch one. Instead
#  it ranks the ways of getting one, uses the best available, and says on screen
#  which one is live:
#
#    1. YOUR QUOTES     you paste option prices, they are inverted to implied
#                       vols and SABR is calibrated to them. Real market data,
#                       because you supplied the market half.
#    2. REALISED SEED   the volatility this contract has actually delivered over
#                       the last 30 trading days, computed from free price
#                       history. Not an implied vol — a defensible starting point.
#    3. REGISTRY        a constant written into this file. The weakest source,
#                       labelled as such, and only used when the other two fail.
#
#  The old default was 3 with no explanation, which is exactly the fabricated
#  input this project refuses everywhere else.
# ══════════════════════════════════════════════════════════════════════════════

VOL_SOURCES = {
    "quotes": ("YOUR QUOTES", "implied from the option prices you pasted"),
    "realised": ("REALISED SEED", "30-day realised volatility, computed from price history"),
    "registry": ("REGISTRY DEFAULT", "a constant in this file — replace it"),
}


def realised_seed(name: str, window: int = 30) -> Optional[dict]:
    """Realised volatility over the last `window` sessions, with its percentile.

    Honest about what it is: realised is what the market DID, implied is what it
    charges for what it MIGHT do. They are different numbers and the gap between
    them is itself a trade — which is why the app shows both rather than quietly
    using one as the other.
    """
    hist = fetch_history(name, "2y")
    if hist.empty or len(hist) < window + 20:
        return None
    series = realised_vol(hist, window).dropna()
    if series.empty:
        return None
    cur = float(series.iloc[-1])
    return dict(vol=cur, window=window, n=len(series),
                pct=float((series < cur).mean() * 100),
                median=float(series.median()))


def implied_from_quotes(F: float, T: float, r: float,
                        quotes: pd.DataFrame) -> pd.DataFrame:
    """Invert a table of option PRICES into implied volatilities.

    Expects columns Strike, Type, Price. Each row is solved independently and a
    row that cannot be solved — a price below intrinsic, above the discounted
    forward, or simply mistyped — comes back as NaN with a reason rather than a
    number that would silently poison the calibration.
    """
    out = []
    for row in quotes.itertuples():
        try:
            K = float(row.Strike)
            px = float(row.Price)
            typ = str(row.Type).strip().lower()
        except (TypeError, ValueError):
            continue
        if K <= 0 or not np.isfinite(px) or typ not in ("call", "put"):
            continue
        b = Black76(F, K, T, r, 0.3, typ)
        iv = b.implied_vol(px)
        disc = math.exp(-r * T)
        intr = disc * (max(F - K, 0.0) if typ == "call" else max(K - F, 0.0))
        reason = ""
        if math.isnan(iv):
            reason = ("below intrinsic" if px < intr else
                      "above the no-arbitrage ceiling" if px > disc * (F if typ == "call" else K)
                      else "no solution")
        out.append(dict(Strike=K, Type=typ, Price=px, Implied=iv,
                        Intrinsic=intr, Note=reason))
    return pd.DataFrame(out)


def resolve_vol_source(name: str, F: Optional[float], T: float, r: float,
                       manual: Optional[float] = None) -> dict:
    """Pick the volatility source, best first, and report which one won.

    Returns the ATM level, a σ(K,T) function, the source key and a one-line
    explanation for the header. Everything downstream reads this and nothing
    downstream needs to know how the number was obtained.
    """
    cal = st.session_state.get("sabr_cal")
    shape = st.session_state.get("sabr_shape", dict(beta=0.5, rho=-0.30, nu=0.70))

    if manual is not None:
        atm, key, detail = float(manual), "manual", "your override"
    elif cal and cal.get("ok") and abs(cal.get("F", F or 0) - (F or 0)) < 1e-6:
        atm, key, detail = float(cal["atm"]), "quotes", VOL_SOURCES["quotes"][1]
    else:
        seed = realised_seed(name)
        if seed:
            atm, key = seed["vol"], "realised"
            detail = (f"{seed['window']}-day realised, {seed['pct']:.0f}th percentile "
                      f"of its own 2-year range")
        else:
            atm, key = CONTRACTS[name]["vol"], "registry"
            detail = VOL_SOURCES["registry"][1]

    if F and cal and cal.get("ok") and key in ("quotes", "manual"):
        p = dict(alpha=cal["alpha"], beta=cal["beta"], rho=cal["rho"], nu=cal["nu"], atm=atm)
        if key == "manual":
            p["alpha"] = sabr_alpha_from_atm(F, T, atm, p["beta"], p["rho"], p["nu"])
        return dict(atm=atm, source=key, detail=detail, params=p,
                    fn=sabr_vol_fn(p, F), smile=True)
    if F and st.session_state.get("use_smile", True):
        p = dict(beta=shape["beta"], rho=shape["rho"], nu=shape["nu"], atm=atm)
        p["alpha"] = sabr_alpha_from_atm(F, T, atm, p["beta"], p["rho"], p["nu"])
        return dict(atm=atm, source=key, detail=detail, params=p,
                    fn=sabr_vol_fn(p, F), smile=True)
    return dict(atm=atm, source=key, detail=detail, params=None,
                fn=as_vol_fn(atm), smile=False)


# ══════════════════════════════════════════════════════════════════════════════
#  7. UI FOUNDATION
#  Explanations are everywhere, but folded: one visible line per screen, the
#  detail one click away, and a tooltip on every input. A wall of prose is
#  excellent the first time and an obstacle the thirtieth.
# ══════════════════════════════════════════════════════════════════════════════

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif}}
code,pre{{font-family:'JetBrains Mono',monospace}}
.block-container{{padding-top:.8rem;max-width:1500px}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,#0A0E17 0%,{BG} 100%);
    border-right:1px solid #1C2333}}
[data-testid="stSidebar"] label{{font-size:.70rem;font-weight:500;color:#6E7681;
    text-transform:uppercase;letter-spacing:.07em}}
.stTabs [data-baseweb="tab-list"]{{background:{BG};border-bottom:1px solid #1C2333;gap:3px}}
.stTabs [data-baseweb="tab"]{{font-size:.82rem;font-weight:500;color:#6E7681;padding:9px 16px}}
.stTabs [aria-selected="true"]{{color:{TEXT};background:{PANEL};border-bottom:2px solid {AMBER}}}
hr{{border:none;border-top:1px solid #1C2333;margin:10px 0}}
.kpi{{background:{PANEL};border:.5px solid {BORDER};border-radius:9px;padding:11px 13px}}
.kpi-l{{font-size:.64rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;
    margin-bottom:4px}}
.kpi-v{{font-family:'JetBrains Mono',monospace;font-size:.95rem;color:{TEXT};white-space:nowrap}}
.kpi-s{{font-size:.66rem;color:#6E7681;margin-top:3px}}
.badge{{display:inline-block;padding:2px 8px;border:1px solid {BORDER};border-radius:5px;
    font-family:'JetBrains Mono',monospace;font-size:.71rem}}
.lead{{font-size:.85rem;color:#B9C4CF;line-height:1.55;margin:2px 0 10px 0}}
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


def styled(fig, height=320, title=None):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter", size=11, color=TEXT),
        title=(dict(text=title, x=0.5, xanchor="center",
                    font=dict(size=12, color=GRAY)) if title else None),
        margin=dict(l=55, r=25, t=55 if title else 25, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
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
                f'color:{TEXT}">{title}</span></div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)


def add_to_book(**line) -> None:
    st.session_state.setdefault("book", []).append(line)


def book_state() -> List[dict]:
    return st.session_state.setdefault("book", [])


# ══════════════════════════════════════════════════════════════════════════════
#  8. SCREENS
#  Ordered by how often a trader opens them: the chain, then a trade, then the
#  book. Everything else is grouped behind those three.
# ══════════════════════════════════════════════════════════════════════════════

def screen_chain(cx: dict) -> None:
    lead("Every listed strike around the money, calls left, puts right — and the place "
         "to turn the prices on your broker screen into a volatility surface.",
         """
**The ladder.** Strikes come off the contract's real increment grid (0.50 on WTI, 5 on
gold, 10 cents on corn), so every line is an option that exists. `Call Δ` and `Put Δ` are
the futures-equivalent exposure per unit; `σ(K)` is the volatility used for that row.

**The cash columns** are per lot, so `Call $/lot` is what you would actually pay.

**Pasting quotes is the important part.** An implied volatility is not something anyone
can download — it is an option price turned inside out. Type the premiums you can see,
and the app inverts each one, then fits SABR to them. From that moment every screen on
the platform prices off *your* market rather than an assumption.

**Rows that fail to invert** show a reason instead of a number: a premium below intrinsic
or above the discounted forward has no volatility that reproduces it, and silently
dropping it would poison the fit.
""")
    F_curve, r, unit, name = cx["curve"], cx["r"], cx["unit"], cx["name"]
    c1, c2, c3 = st.columns([2, 1, 1])
    labels = [f"{x.label}  ({option_tenor(x.T):.2f}y)" for x in F_curve.itertuples()]
    ei = c1.selectbox("Expiry", range(len(F_curve)),
                      index=min(cx["T_months"] - 1, len(F_curve) - 1),
                      format_func=lambda i: labels[i],
                      help="Each row of the forward curve is a delivery month. The option "
                           "expires a few days before the future it settles into.")
    T = option_tenor(float(F_curve["T"].iloc[ei]))
    F = float(F_curve["price"].iloc[ei])
    width = c2.slider("Strikes shown", 5, 41, 15, 2,
                      help="How far either side of the money to list.")
    amer = c3.radio("Exercise", ["European", "American"], horizontal=True, key="ch_style",
                    help="Most listed commodity options are American. Early exercise is "
                         "priced on a binomial tree, and is worth more the higher the "
                         "discount rate.") == "American"

    vs = resolve_vol_source(name, F, T, r, cx["manual_vol"])
    ch = option_chain(name, F, T, r, vs["fn"], width, amer)
    src_col = {"quotes": GREEN, "realised": AMBER, "manual": BLUE, "registry": RED}[vs["source"]]
    st.markdown(
        badge(f"{F_curve['label'].iloc[ei]} @ {F:,.4f} {unit}", TEXT) + " " +
        badge(f"σ ATM {vs['atm'] * 100:.1f}%", PURPLE) + " " +
        badge(VOL_SOURCES.get(vs["source"], ("MANUAL", ""))[0], src_col) + " " +
        badge("SABR smile" if vs["smile"] else "flat", PURPLE) +
        f'<span style="color:{GRAY};font-size:.76rem;margin-left:8px">{vs["detail"]}</span>',
        unsafe_allow_html=True)

    disp = ch[["call_cash", "call_px", "call_delta", "vol", "strike", "moneyness",
               "put_delta", "put_px", "put_cash"]].copy()
    disp.columns = ["Call $/lot", "Call", "Call Δ", "σ(K)", "STRIKE", "% of F",
                    "Put Δ", "Put", "Put $/lot"]
    atm_k = snap_strike(name, F)

    def _hl(row):
        near = abs(row["STRIKE"] - atm_k) < CONTRACTS[name]["strike_inc"] / 2
        return ["background-color: rgba(240,165,0,0.14)" if near else "" for _ in row]

    st.dataframe(disp.style.format(
        {"Call $/lot": "${:,.0f}", "Call": "{:,.4f}", "Call Δ": "{:+.3f}", "σ(K)": "{:.2%}",
         "STRIKE": "{:,.4f}", "% of F": "{:.1f}%", "Put Δ": "{:+.3f}", "Put": "{:,.4f}",
         "Put $/lot": "${:,.0f}"}).apply(_hl, axis=1),
        use_container_width=True, hide_index=True, height=min(540, 42 + 35 * len(disp)))

    # ── ticket ───────────────────────────────────────────────────────────────
    section("Ticket")
    t1, t2, t3, t4, t5 = st.columns([1.6, 1, 1, 1, 2])
    K = t1.selectbox("Strike", ch["strike"].tolist(),
                     index=int(np.argmin(np.abs(ch["strike"] - F))),
                     format_func=lambda k: f"{k:,.4f}")
    otype = t2.radio("Type", ["call", "put"], horizontal=True, key="ck_t")
    sidew = t3.radio("Side", ["Buy", "Sell"], horizontal=True, key="ck_s")
    lots = t4.number_input("Lots", 1, 999, 1, key="ck_l")
    row = ch[ch["strike"] == K].iloc[0]
    px = float(row["call_px"] if otype == "call" else row["put_px"])
    dlt = float(row["call_delta"] if otype == "call" else row["put_delta"])
    mult = price_multiplier(name)
    t5.markdown(f'<div class="note" style="margin:0">{sidew} {lots} × '
                f'{F_curve["label"].iloc[ei]} {K:,.4f} {otype}<br>'
                f'<b>{px:,.4f}</b> {unit} = <b>${px * mult * lots:,.0f}</b> · '
                f'Δ {dlt * lots * (1 if sidew == "Buy" else -1):+.2f} · '
                f'σ {row["vol"]:.2%}</div>', unsafe_allow_html=True)
    if st.button("➕  Add to book", type="primary", use_container_width=True):
        add_to_book(kind="option", contract=name, type=otype, strike=float(K),
                    qty=int(lots * (1 if sidew == "Buy" else -1)), T=float(T),
                    entry=px, style="American" if amer else "European", tag="chain")
        st.success(f"{sidew} {lots} × {K:,.4f} {otype} booked.")

    # ── quotes → implied vols → SABR ─────────────────────────────────────────
    section("Paste market prices → implied volatility → surface",
            "This is the direction a trader actually works in: you have prices on a screen, "
            "not volatilities. Fill in the premiums you can see and the platform does the "
            "rest. Leave it empty and it falls back to realised volatility, which it says.")
    seed = pd.DataFrame({
        "Strike": [float(x) for x in ch["strike"].iloc[::max(len(ch) // 5, 1)][:6]],
        "Type": ["put", "put", "call", "call", "call", "call"][:len(ch.iloc[::max(len(ch) // 5, 1)][:6])],
        "Price": [np.nan] * len(ch.iloc[::max(len(ch) // 5, 1)][:6])})
    q = st.data_editor(st.session_state.get("quote_table", seed),
                       use_container_width=True, hide_index=True, num_rows="dynamic",
                       key="quotes_editor",
                       column_config={
                           "Strike": st.column_config.NumberColumn(format="%.4f"),
                           "Type": st.column_config.SelectboxColumn(options=["call", "put"]),
                           "Price": st.column_config.NumberColumn(
                               format="%.4f", help="The premium you can deal at, "
                                                   "in the contract's quote units.")})
    a1, a2 = st.columns([1, 3])
    if a1.button("Invert & calibrate", type="primary", use_container_width=True):
        st.session_state["quote_table"] = q
        rows = implied_from_quotes(F, T, r, q.dropna(subset=["Price"]))
        st.session_state["implied_rows"] = rows
        good = rows.dropna(subset=["Implied"]) if not rows.empty else rows
        if len(good) >= 3:
            beta = st.session_state.get("sabr_shape", {}).get("beta", 0.5)
            cal = sabr_calibrate(F, T, good["Strike"], good["Implied"], beta)
            if cal.get("ok"):
                cal["F"] = F
                cal["atm"] = float(np.interp(F, np.sort(good["Strike"]),
                                             good.sort_values("Strike")["Implied"]))
                st.session_state["sabr_cal"] = cal
                st.success(f"Calibrated on {len(good)} quotes — RMSE "
                           f"{cal['rmse'] * 100:.3f} vol points. This surface now drives "
                           f"every screen.")
        elif not good.empty:
            st.warning(f"Only {len(good)} quote(s) inverted — SABR needs at least three. "
                       f"The implied vols are shown below and are usable on their own.")
    if a2.button("Clear the calibration", use_container_width=True):
        st.session_state.pop("sabr_cal", None)
        st.session_state.pop("implied_rows", None)
        st.rerun()

    rows = st.session_state.get("implied_rows")
    if rows is not None and not rows.empty:
        st.dataframe(rows.style.format({"Strike": "{:,.4f}", "Price": "{:,.4f}",
                                        "Implied": "{:.2%}", "Intrinsic": "{:,.4f}"},
                                       na_rep="—"),
                     use_container_width=True, hide_index=True)
        good = rows.dropna(subset=["Implied"])
        if not good.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=good["Strike"], y=good["Implied"] * 100,
                                     mode="markers", name="Your quotes",
                                     marker=dict(size=11, color=AMBER)))
            cal = st.session_state.get("sabr_cal")
            if cal and cal.get("ok"):
                ks = np.linspace(min(good["Strike"]) * 0.92, max(good["Strike"]) * 1.08, 100)
                fig.add_trace(go.Scatter(x=ks, y=[sabr_vol(F, k_, T, cal["alpha"], cal["beta"],
                                                           cal["rho"], cal["nu"]) * 100
                                                  for k_ in ks],
                                         mode="lines", name="SABR fit",
                                         line=dict(color=BLUE, width=2)))
            vline(fig, F, GREEN, "F")
            fig.update_xaxes(title=f"Strike ({unit})")
            fig.update_yaxes(title="Implied vol (%)")
            st.plotly_chart(styled(fig, 300, "Your market, inverted"), use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure(go.Scatter(x=ch["strike"], y=ch["vol"] * 100, mode="lines+markers",
                                   line=dict(color=AMBER, width=2.2)))
        vline(fig, F, GREEN, f"F {F:,.2f}")
        fig.update_xaxes(title=f"Strike ({unit})")
        fig.update_yaxes(title="σ (%)")
        st.plotly_chart(styled(fig, 280, "Volatility across the chain"),
                        use_container_width=True)
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
        st.plotly_chart(styled(fig, 280, "Delta ladder — dotted lines are the 25s"),
                        use_container_width=True)


def screen_trade(cx: dict) -> None:
    lead("Price one option or a whole structure, on the same screen — a single option is "
         "just a structure with one leg.",
         """
**Strikes** can be given three ways. *Exact* is how a listed option is named and how a
broker statement reads. *% of forward* is a quick way to say "10% out of the money".
*Delta* is how a desk quotes: "the 25-delta put". All three are snapped onto the
contract's listed increment, so the strike you price is one you could deal.

**Every leg carries its own σ(K)** from the active surface. This matters more than it
looks: on a flat volatility the 25-delta call and put carry the same number and a risk
reversal comes out nearly free, which is never true in a market that charges for skew.

**The Greeks** are per unit of the commodity; the cash line underneath converts them to
dollars per lot. Delta is futures-equivalent exposure, gamma is how fast that changes,
vega is per one volatility point, theta is per calendar day.

**European or American.** Most listed commodity options are American. The tree prices the
right to exercise early, worth about 0.4% of the premium at a 2% rate and 2.6% at 10%.
""")
    F, T, r, unit, name = cx["F_T"], cx["T_opt"], cx["r"], cx["unit"], cx["name"]
    vs = cx["vs"]
    vf = vs["fn"]
    mult = price_multiplier(name)

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
                     use_container_width=True, hide_index=True)

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
                        use_container_width=True)
    with g2:
        be = rng[np.where(np.diff(np.sign(pnl)) != 0)[0]]
        for lbl, val, col in (
                ("Max profit shown", f"{np.max(pnl):+,.4f}", GREEN),
                ("Max loss shown", f"{np.min(pnl):+,.4f}", RED),
                ("Break-evens", ", ".join(f"{b:,.2f}" for b in be[:4]) or "none in range", TEXT),
                ("Cash per lot", f"${prem * mult:+,.0f}", AMBER)):
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                        f'border-bottom:.5px solid {BORDER}"><span style="color:{GRAY};'
                        f'font-size:.75rem">{lbl}</span><span style="font-family:JetBrains Mono,'
                        f'monospace;font-size:.75rem;color:{col}">{val}</span></div>',
                        unsafe_allow_html=True)
        st.caption("Maxima are read off the plotted range: an unbounded structure shows the "
                   "edge of the chart, which is the honest answer to 'how much can this lose'.")
        if st.button("➕  Book it", type="primary", use_container_width=True):
            for l in legs:
                add_to_book(kind="option", contract=name, type=l["type"],
                            strike=float(l["strike"]), qty=int(l["qty"]), T=float(T),
                            entry=float(l["price"]),
                            style="American" if amer else "European", tag=label)
            st.success(f"{len(legs)} leg(s) booked.")


def screen_book(cx: dict) -> None:
    lead("Everything you hold, in cash terms — with the two views that decide whether a "
         "position is safe: vega by tenor, and P&L across a grid of price and volatility.",
         """
**Cash Greeks.** Delta is your futures-equivalent exposure in dollars; gamma is how much
that delta moves for a 1% move in the forward; vega is dollars per volatility point;
theta is dollars lost per calendar day.

**Vega buckets are the point.** A book is never simply "long vega" — it is long the front
month and short the back, or the reverse. A single net number hides that entirely, and
calendar risk is what actually moves a volatility book's P&L.

**The scenario grid** is the screen an options desk watches. Rows are volatility shifts in
points, columns are moves in the forward, cells are book P&L. A position can be perfectly
delta-flat and still lose in every column.

**SPAN margin** is read off that same grid: sixteen scenarios, the worst weighted loss.
It is a proxy — the exchange adds credits this does not model — but margin, not notional,
is what actually caps your size, which is why return on margin is shown next to it.
""")
    book = book_state()
    F, r, unit, name = cx["F_T"], cx["r"], cx["unit"], cx["name"]
    vf = cx["vs"]["fn"]

    if not book:
        st.info("The book is empty. Add a line from the **Chain** ticket or the **Trade** "
                "screen, or use the manual entry below.")
    with st.expander("➕  Add a line manually", expanded=not book):
        a = st.columns(6)
        kind = a[0].selectbox("Kind", ["option", "future"], key="bk_kind")
        otype = a[1].selectbox("Type", ["call", "put"], key="bk_type",
                               disabled=kind == "future")
        raw_k = a[2].number_input("Strike", value=float(snap_strike(name, F or 100)),
                                  step=float(CONTRACTS[name]["strike_inc"]), key="bk_K",
                                  disabled=kind == "future")
        months = a[3].number_input("Tenor (months)", 1, 60, cx["T_months"], key="bk_T")
        qty = a[4].number_input("Lots", -999, 999, 1, key="bk_q")
        style = a[5].selectbox("Style", ["European", "American"], key="bk_style",
                               disabled=kind == "future")
        K = snap_strike(name, raw_k)
        Tl = option_tenor(months / 12) if kind == "option" else months / 12
        default_entry = (Black76(F, K, Tl, r, vf(K, Tl), otype).price()
                         if kind == "option" and F else (F or 0.0))
        entry = st.number_input("Entry price", value=float(round(default_entry, 4)),
                                step=0.01, format="%.4f", key="bk_e",
                                help="What you actually paid or received. Mark-to-market is "
                                     "measured against this.")
        if st.button("Add line", use_container_width=True):
            add_to_book(kind=kind, contract=name, type=otype, strike=K, qty=int(qty),
                        T=float(Tl), entry=float(entry), style=style, tag="manual")
            st.rerun()

    if not book:
        return

    io1, io2, io3 = st.columns([1, 1, 2])
    io1.download_button("⬇ Export book", json.dumps(book, indent=2, default=str),
                        file_name="codap_book.json", use_container_width=True)
    up = io2.file_uploader("Import", type="json", label_visibility="collapsed")
    if up is not None and io2.button("Load", use_container_width=True):
        try:
            st.session_state["book"] = json.loads(up.read().decode())
            st.rerun()
        except Exception as e:                                     # noqa: BLE001
            st.error(f"Import failed: {e}")
    if io3.button("🗑  Flatten the book", use_container_width=True):
        st.session_state["book"] = []
        st.rerun()

    with st.spinner("Repricing…"):
        bg = book_greeks(book, F, vf, r)
        mtx = scenario_matrix(book, F, vf, r)
        span = span_margin(book, F, vf, r)
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

    rom = tot["value"] / span["margin"] * 100 if span["margin"] > 1e-9 else float("nan")
    st.markdown(f'<div class="note">Margin, not notional, is what caps your size: this book '
                f'ties up <b>${span["margin"]:,.0f}</b> and is <b>{rom:+.1f}%</b> return on '
                f'margin. Premium at risk is ${abs(tot["premium"]):,.0f}.</div>',
                unsafe_allow_html=True)

    section("Vega by tenor")
    bk = bg["buckets"]
    bc = st.columns(len(bk))
    for i, (bn, v) in enumerate(bk.items()):
        kpi(bc[i], bn, f"${v:+,.0f}", "vega cash", PURPLE if abs(v) > 1 else GRAY)
    net_v, gross_v = sum(bk.values()), sum(abs(v) for v in bk.values())
    if gross_v > 1e-9 and abs(net_v) < gross_v * 0.6:
        st.caption(f"Net vega ${net_v:+,.0f} against gross ${gross_v:,.0f} — this is a "
                   f"**calendar position**: long volatility in one bucket, short in another. "
                   f"A single net number would have shown almost nothing.")

    st.markdown("##### Positions")
    st.dataframe(pd.DataFrame(bg["rows"]).style.format(
        {"Strike": "{:,.2f}", "T": "{:.3f}", "Vol": "{:.2%}", "Price": "{:,.4f}",
         "DeltaCash": "{:+,.0f}", "GammaCash": "{:+,.0f}", "VegaCash": "{:+,.0f}",
         "ThetaCash": "{:+,.0f}", "MTM": "{:+,.0f}"}, na_rep="—"),
        use_container_width=True, hide_index=True)

    section("Scenario grid — price × volatility")
    st.dataframe(mtx.style.format("{:+,.0f}").background_gradient(cmap="RdYlGn", axis=None),
                 use_container_width=True)
    fig = go.Figure(go.Heatmap(z=mtx.values, x=list(mtx.columns), y=list(mtx.index),
                               colorscale="RdYlGn", zmid=0,
                               hovertemplate="%{x} · %{y}<br>$%{z:,.0f}<extra></extra>"))
    fig.update_xaxes(title="Move in the forward")
    fig.update_yaxes(title="Shift in volatility")
    st.plotly_chart(styled(fig, 300, "Book P&L surface"), use_container_width=True)
    with st.expander("The sixteen SPAN scenarios"):
        st.dataframe(span["scenarios"].style.format(
            {"weight": "{:.2f}", "pnl": "{:+,.0f}", "weighted_loss": "{:+,.0f}"}),
            use_container_width=True, hide_index=True)


def screen_vol_analytics(cx: dict) -> None:
    section("Volatility analytics — is this quote rich?",
            "The two questions a vol trader asks before trading: what has this market "
            "actually delivered, and does hedging the option pay? Both are answered from free "
            "price history — no options feed required.")
    name, F, T, r, vol = cx["name"], cx["F_T"], cx["T_opt"], cx["r"], cx["vs"]["atm"]

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
            kpi(k[2], "Your implied", f"{vol * 100:.1f}%", "MODEL input", PURPLE)
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
            fig.update_xaxes(title="Window (trading days)", type="log")
            fig.update_yaxes(title="Annualised volatility (%)")
            st.plotly_chart(styled(fig, 360, "Volatility cone — purple dotted is your implied"),
                            use_container_width=True)
            st.dataframe(cone[["window", "p5", "p25", "p50", "p75", "p95", "current", "rank"]]
                         .style.format({c: "{:.1%}" for c in ("p5", "p25", "p50", "p75", "p95", "current")}
                                       | {"rank": "{:.0f}"}),
                         use_container_width=True, hide_index=True)
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
    g1, g2, g3, g4 = st.columns(4)
    s_imp = g1.slider("Implied you pay (%)", 5, 150, int(vol * 100), 1, key="gs_i") / 100
    s_real = g2.slider("Realised delivered (%)", 5, 150, int(vol * 100), 1, key="gs_r") / 100
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
                        use_container_width=True)
    with c2:
        edge = np.linspace(max(s_imp - 0.20, 0.02), s_imp + 0.20, 11)
        means = [gamma_scalp(F, F, T, r, s_imp, float(sr), rebal, 1200,
                             position=pos)["mean"] for sr in edge]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge * 100, y=means, mode="lines+markers",
                                 line=dict(color=AMBER, width=2.4)))
        fig.add_hline(y=0, line=dict(color=GRAY, width=0.8))
        vline(fig, s_imp * 100, PURPLE, f"implied {s_imp * 100:.0f}%")
        fig.update_xaxes(title="Realised volatility delivered (%)")
        fig.update_yaxes(title="Mean hedged P&L")
        st.plotly_chart(styled(fig, 300, "Break-even sits at the implied you paid"),
                        use_container_width=True)


def screen_sabr(cx: dict) -> None:
    section("SABR surface and the skew quotes",
            "SABR is the market standard for commodity option surfaces. Four parameters, each "
            "with a job a trader can argue about: <b>α</b> the level, <b>β</b> the backbone, "
            "<b>ρ</b> the skew, <b>ν</b> the smile. Unlike a polynomial it extrapolates "
            "sensibly into the wings, which is where the quotes that matter live.")
    F, T, r, unit = cx["F_T"], cx["T_opt"], cx["r"], cx["unit"]
    c1, c2, c3, c4 = st.columns(4)
    beta = c1.select_slider("β backbone", options=[0.0, 0.25, 0.5, 0.75, 1.0], value=0.5,
                            help="0 = normal dynamics, 1 = lognormal. Commodities usually "
                                 "sit near 0.5; it is a choice, not a fit.")
    atm = c2.slider("ATM vol (%)", 5, 150, int(cx["vs"]["atm"] * 100), 1, key="sb_atm") / 100
    rho = c3.slider("ρ skew", -95, 95, -35, 1, key="sb_rho") / 100
    nu = c4.slider("ν vol-of-vol", 1, 300, 85, 1, key="sb_nu") / 100
    alpha = sabr_alpha_from_atm(F, T, atm, beta, rho, nu)
    params = dict(alpha=alpha, beta=beta, rho=rho, nu=nu)

    rrbf = risk_reversal_butterfly(F, T, r, params, 0.25)
    k = st.columns(5)
    kpi(k[0], "α (level)", f"{alpha:.4f}", "solved so ATM is exact", AMBER)
    kpi(k[1], "ATM", f"{rrbf['atm'] * 100:.2f}%", "reproduced by construction", TEXT)
    kpi(k[2], "25Δ risk reversal", f"{rrbf['rr'] * 100:+.2f} vols",
        "call vol − put vol", RED if rrbf["rr"] < 0 else GREEN)
    kpi(k[3], "25Δ butterfly", f"{rrbf['bf'] * 100:+.2f} vols", "wings over the body", BLUE)
    kpi(k[4], "25Δ strikes", f"{rrbf['K_put']:,.1f} / {rrbf['K_call']:,.1f}",
        f"put / call ({unit})", PURPLE)
    st.markdown(
        f'<div class="note">That third number is how a desk speaks: <b>"the 25-delta risk '
        f'reversal is at {rrbf["rr"] * 100:+.1f} vols"</b> describes this entire skew in one '
        f'phrase. Negative means puts are bid over calls — the normal commodity shape outside a '
        f'supply squeeze, when it inverts.</div>', unsafe_allow_html=True)

    section("Calibrate to quotes")
    st.markdown('<div class="note">Paste a strip of market volatilities and SABR fits ρ and ν to '
                'them, with α pinned so ATM is matched exactly. This is the one place the '
                'platform can consume real option quotes — there is no free feed for them, so '
                'they are typed in and labelled as your data.</div>', unsafe_allow_html=True)
    default = pd.DataFrame({
        "Strike": [round(F * x, 2) for x in (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)],
        "Market vol %": [round(sabr_vol(F, F * x, T, alpha, beta, rho, nu) * 100, 2)
                         for x in (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)]})
    edited = st.data_editor(default, use_container_width=True, hide_index=True,
                            num_rows="dynamic", key="sabr_quotes")
    if st.button("Calibrate SABR", type="primary"):
        try:
            cal = sabr_calibrate(F, T, edited["Strike"].astype(float),
                                 edited["Market vol %"].astype(float) / 100, beta)
        except Exception as e:                                     # noqa: BLE001
            st.error(f"Calibration failed: {e}")
            cal = dict(ok=False)
        if cal.get("ok"):
            st.session_state["sabr_cal"] = cal
    cal = st.session_state.get("sabr_cal")
    if cal and cal.get("ok"):
        kk = st.columns(5)
        kpi(kk[0], "α", f"{cal['alpha']:.4f}", "level", AMBER)
        kpi(kk[1], "β", f"{cal['beta']:.2f}", "fixed by you")
        kpi(kk[2], "ρ", f"{cal['rho']:+.4f}", "skew", BLUE)
        kpi(kk[3], "ν", f"{cal['nu']:.4f}", "vol of vol", PURPLE)
        kpi(kk[4], "Fit RMSE", f"{cal['rmse'] * 100:.3f} vols",
            f"worst {cal['max_err'] * 100:.3f}",
            GREEN if cal["rmse"] < 0.005 else AMBER)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal["strikes"], y=cal["market"] * 100, mode="markers",
                                 name="Quotes", marker=dict(size=10, color=AMBER)))
        ks = np.linspace(min(cal["strikes"]) * 0.9, max(cal["strikes"]) * 1.1, 120)
        fig.add_trace(go.Scatter(x=ks, y=[sabr_vol(F, k_, T, cal["alpha"], cal["beta"],
                                                   cal["rho"], cal["nu"]) * 100 for k_ in ks],
                                 mode="lines", name="SABR", line=dict(color=BLUE, width=2)))
        vline(fig, F, GREEN, "ATM")
        fig.update_xaxes(title=f"Strike ({unit})")
        fig.update_yaxes(title="Implied vol (%)")
        st.plotly_chart(styled(fig, 320, "Calibrated smile against the quotes"),
                        use_container_width=True)
        params = dict(alpha=cal["alpha"], beta=cal["beta"], rho=cal["rho"], nu=cal["nu"])

    section("Surface and term structure")
    Ks = np.linspace(F * 0.6, F * 1.45, 30)
    Ts = np.array([1 / 12, 3 / 12, 6 / 12, 1.0, 1.5, 2.0])
    Tl = ["1M", "3M", "6M", "1Y", "18M", "2Y"]
    Z = np.array([[sabr_vol(F, float(k_), float(t), params["alpha"], params["beta"],
                            params["rho"], params["nu"]) * 100 for k_ in Ks] for t in Ts])
    fig = go.Figure(go.Surface(z=Z, x=Ks, y=Tl, colorscale="RdYlGn_r", opacity=0.93,
                               hovertemplate="K %{x:,.1f}<br>T %{y}<br>σ %{z:.2f}%<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=420, paper_bgcolor=BG,
                      font=dict(family="Inter", size=11, color=TEXT),
                      margin=dict(l=10, r=10, t=40, b=10),
                      scene=dict(bgcolor=BG,
                                 xaxis=dict(title=f"Strike ({unit})", gridcolor=BORDER,
                                            backgroundcolor=BG),
                                 yaxis=dict(title="Maturity", gridcolor=BORDER, backgroundcolor=BG),
                                 zaxis=dict(title="σ (%)", gridcolor=BORDER, backgroundcolor=BG),
                                 camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8))))
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for i in range(len(Ts) - 1):
        a_ = sabr_vol(F, F, float(Ts[i]), **params)
        b_ = sabr_vol(F, F, float(Ts[i + 1]), **params)
        rows.append(dict(From=Tl[i], To=Tl[i + 1], ATM_from=a_, ATM_to=b_,
                         Forward=forward_vol(a_, float(Ts[i]), b_, float(Ts[i + 1]))))
    st.markdown("##### Forward volatility between expiries")
    st.dataframe(pd.DataFrame(rows).style.format(
        {"ATM_from": "{:.2%}", "ATM_to": "{:.2%}", "Forward": "{:.2%}"}, na_rep="negative variance"),
        use_container_width=True, hide_index=True)
    st.caption("The volatility implied BETWEEN two expiries — what a calendar option actually "
               "trades. When the term structure falls steeply enough the forward variance goes "
               "negative, which is an arbitrage rather than a number, and the table says so "
               "instead of printing a square root of a negative.")


def screen_asian(cx: dict) -> None:
    section("Asian options — payoff on the average price",
            "An Asian option settles against the <b>average</b> of the forward over a set of "
            "fixings, not a single closing price. Airlines, refiners and utilities buy them "
            "because their physical exposure is itself an average: they consume every day of "
            "the month, not on the third Wednesday. Averaging suppresses volatility, so an "
            "Asian is always <b>cheaper than the equivalent vanilla</b> — the page proves it "
            "rather than asserting it.")
    F, K, T, r, unit = cx["F_T"], cx["K"], cx["T_opt"], cx["r"], cx["unit"]
    vol = cx["vol_K"]
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
        res = asian_mc(F, K, T, r, vol, n_obs, cx["side"].lower(), avg,
                       cx["n_paths"], control_variate=use_cv, antithetic=use_cv)
        kv = kemna_vorst(F, K, T, r, vol, n_obs, cx["side"].lower())
        plain = asian_mc(F, K, T, r, vol, n_obs, cx["side"].lower(), avg,
                         cx["n_paths"], control_variate=False, antithetic=False)
    van = Black76(F, K, T, r, vol, cx["side"].lower()).price()

    k1, k2, k3, k4, k5 = st.columns(5)
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
                        use_container_width=True)
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
                        use_container_width=True)

    avgs = np.asarray(res["sample_avgs"])
    fig = go.Figure(go.Histogram(x=avgs, nbinsx=60, marker_color=AMBER, opacity=0.75,
                                 histnorm="probability density"))
    vline(fig, K, GREEN, f"K {K:,.2f}")
    vline(fig, F, BLUE, f"F {F:,.2f}", y=0.82)
    fig.update_xaxes(title=f"Average price at expiry ({unit})")
    fig.update_yaxes(title="Density")
    st.plotly_chart(styled(fig, 260, "Distribution of the average — narrower than the terminal "
                                     "price, which is exactly why the Asian is cheaper"),
                    use_container_width=True)


def screen_structure(cx: dict) -> None:
    section("Spread options on refining and processing margins — Kirk (1995)",
            "A <b>crack</b> is the refiner's gross margin: crude in, products out. A "
            "<b>crush</b> is the same idea for soybeans. An option on that spread is what a "
            "refiner buys to protect a margin, and Kirk's approximation prices it by folding "
            "the short leg into an effective strike. Every leg of the structure is carried and "
            "converted to a common unit — a 3-2-1 crack means <i>both</i> products.")
    struct = st.selectbox("Structure", list(STRUCTURES.keys()))
    cfg = STRUCTURES[struct]
    st.markdown(f'<div class="note">{cfg["note"]} &nbsp;·&nbsp; quoted {cfg["basis"]} '
                f'({cfg["unit"]}).</div>', unsafe_allow_html=True)

    legs = [n for n, _, _ in cfg["long"]] + [n for n, _, _ in cfg["short"]]
    T = cx["T"]
    prices, vols, missing = {}, {}, []
    for leg in legs:
        cur = cx["curves"].get(leg)
        if cur is None:
            cur = fetch_curve(leg)
            cx["curves"][leg] = cur
        px = forward_at(cur, T) if (cur is not None and not cur.empty) else None
        if px is None:
            missing.append(leg)
        else:
            prices[leg] = px
        vols[leg] = CONTRACTS[leg]["vol"]

    if missing:
        st.error(f"**NO MARKET DATA** for {', '.join(missing)} — a structure is only as "
                 f"real as its thinnest leg, so nothing is priced in its place. "
                 f"Check the feed diagnostics in the sidebar.")
        return

    st.markdown(f'<div class="note">Legs are priced at a <b>matched tenor</b> of '
                f'{T:.3f} years, interpolated on each curve. Pairing the front months instead '
                f'would silently compare different delivery dates whenever two contracts have '
                f'different cycles.</div>', unsafe_allow_html=True)

    ed = st.expander("Override leg volatilities (model inputs)")
    with ed:
        cols = st.columns(len(legs))
        for i, leg in enumerate(legs):
            vols[leg] = cols[i].slider(f"σ {leg.split('(')[0].strip()}", 5, 120,
                                       int(CONTRACTS[leg]["vol"] * 100), 1,
                                       key=f"sv_{struct}_{leg}") / 100

    agg = structure_legs(struct, prices, vols)
    c1, c2, c3 = st.columns(3)
    K = c1.number_input(f"Strike on the spread ({cfg['unit']})",
                        value=float(round(agg["spread"], 3)), step=0.25, format="%.3f")
    rho = c2.slider("Correlation between the two sides", -95, 99, int(cfg["rho"] * 100), 1,
                    help="Cracks live at 0.80–0.90: products follow crude closely, and the "
                         "spread's own volatility is what is left over.") / 100
    side = c3.radio("Option", ["Call ▲ (margin widens)", "Put ▼ (margin compresses)"],
                    key="sp_side")
    otype = "call" if "Call" in side else "put"

    so = SpreadOption(agg["F_long"], agg["F_short"], K, T, cx["r"],
                      agg["sigma_long"], agg["sigma_short"], rho, otype)
    res = so.price()

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Spread now", f"{agg['spread']:,.3f}", cfg["unit"],
        GREEN if agg["spread"] > 0 else RED)
    kpi(k2, "Option price", f"{res['price']:.4f}", f"Kirk σ {res['sigma_kirk'] * 100:.1f}%", AMBER)
    kpi(k3, "Intrinsic", f"{res['intrinsic']:.4f}", f"time value {res['time_value']:.4f}")
    kpi(k4, "Δ long side", f"{res['delta_long']:+.4f}", "products", GREEN)
    kpi(k5, "Δ short side", f"{res['delta_short']:+.4f}", "crude / beans", RED)

    st.markdown("##### Legs, converted to the common unit")
    ldf = pd.DataFrame(agg["legs"])
    ldf["side"] = ["long"] * len(cfg["long"]) + ["short"] * len(cfg["short"])
    st.dataframe(ldf[["side", "leg", "qty", "conv", "price", "value", "vol"]].style.format(
        {"qty": "{:.1f}", "conv": "{:.4f}", "price": "{:,.4f}", "value": "{:,.4f}",
         "vol": "{:.0%}"}), use_container_width=True, hide_index=True)
    st.caption(f"Long side aggregates to {agg['F_long']:,.3f} {cfg['unit']} at a blended "
               f"{agg['sigma_long'] * 100:.1f}% vol; short side to {agg['F_short']:,.3f} at "
               f"{agg['sigma_short'] * 100:.1f}%. The blend is value-weighted and ignores the "
               f"correlation *between* the products — an approximation, stated.")

    with st.spinner("Benchmarking Kirk against a two-factor simulation…"):
        mc = so.mc_price(200_000)
    err = res["price"] - mc["price"]
    ok = abs(err) <= max(3 * mc["std_error"], 0.02 * max(mc["price"], 1e-9))
    st.markdown(
        f'<div class="note" style="border-left:2px solid {GREEN if ok else RED}">'
        f'<b>Kirk vs Monte Carlo:</b> {res["price"]:.4f} against {mc["price"]:.4f} ± '
        f'{mc["std_error"]:.4f} — difference <b>{err:+.4f}</b> '
        f'({err / max(mc["price"], 1e-9) * 100:+.2f}%). '
        f'{"Within simulation noise: the approximation is doing its job." if ok else "Outside tolerance — treat the closed form with caution at these parameters."}'
        f'</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        rng = np.linspace(agg["spread"] - 25, agg["spread"] + 25, 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rng, y=so.payoff(rng) - res["price"], name="P&L",
                                 line=dict(color=AMBER, width=2.4)))
        fig.add_trace(go.Scatter(x=rng, y=so.payoff(rng), name="Payoff",
                                 line=dict(color=AMBER, width=1, dash="dot"), opacity=0.45))
        fig.add_hline(y=0, line=dict(color=GRAY, width=0.8))
        vline(fig, K, GREEN, f"K {K:,.1f}")
        vline(fig, agg["spread"], RED, f"Spread {agg['spread']:,.1f}", y=0.82)
        fig.update_xaxes(title=f"Spread at expiry ({cfg['unit']})")
        fig.update_yaxes(title=f"P&L ({cfg['unit']})")
        st.plotly_chart(styled(fig, 330, "P&L at expiry"), use_container_width=True)
    with g2:
        Ks = np.linspace(K - 12, K + 12, 13)
        Rs = np.linspace(0.4, 0.98, 9)
        Z = np.array([[SpreadOption(agg["F_long"], agg["F_short"], kk, T, cx["r"],
                                    agg["sigma_long"], agg["sigma_short"], rr,
                                    otype).price()["price"] for kk in Ks] for rr in Rs])
        fig = go.Figure(go.Heatmap(z=Z, x=[f"{k:,.1f}" for k in Ks],
                                   y=[f"{rr:.2f}" for rr in Rs], colorscale="YlOrRd",
                                   hovertemplate="K %{x}<br>ρ %{y}<br>Price %{z:.4f}<extra></extra>"))
        fig.update_xaxes(title=f"Strike ({cfg['unit']})")
        fig.update_yaxes(title="Correlation")
        st.plotly_chart(styled(fig, 330, "Higher correlation → quieter spread → cheaper option"),
                        use_container_width=True)


def screen_calendar(cx: dict) -> None:
    section("Calendar spread options — an option on the shape of the curve",
            "A calendar spread is the same commodity in two delivery months. A <b>call</b> pays "
            "when backwardation widens (the front pulls away), a <b>put</b> when contango "
            "deepens. It is how a trader expresses a view on the curve without betting on the "
            "level — and how a hedger prices the risk in a roll.")
    curve, unit = cx["curve"], cx["unit"]
    if curve.empty or len(curve) < 2:
        st.error("**NO CURVE** — a calendar spread needs at least two dated contracts.")
        return

    labels = [f"{r_.label}  (T={r_.T:.2f}y)" for r_ in curve.itertuples()]
    c1, c2 = st.columns(2)
    ni = c1.selectbox("Near month", range(len(curve)), index=0, format_func=lambda i: labels[i])
    fi = c2.selectbox("Far month", range(len(curve)),
                      index=min(5, len(curve) - 1), format_func=lambda i: labels[i])
    if ni == fi:
        st.warning("Pick two different months.")
        return

    Fn = float(curve["price"].iloc[ni])
    Ff = float(curve["price"].iloc[fi])
    Tn = float(curve["T"].iloc[ni])
    spread = Fn - Ff
    # A calendar spread option cannot outlive its near leg.
    T_opt = max(min(cx["T"], Tn), 1 / 365)

    d1, d2, d3, d4 = st.columns(4)
    K = d1.number_input(f"Strike on the spread ({unit})", value=float(round(spread, 4)),
                        step=0.01, format="%.4f")
    vn = d2.slider("σ near (%)", 5, 150, int(cx["vs"]["atm"] * 105), 1, key="cs_vn") / 100
    vf = d3.slider("σ far (%)", 5, 150, int(cx["vs"]["atm"] * 90), 1, key="cs_vf") / 100
    rho = d4.slider("Correlation", 50, 99, 95, 1, key="cs_rho",
                    help="Two months of the same commodity move almost together — 0.93–0.98 "
                         "is typical, and the spread's volatility is the small residual.") / 100
    otype = "call" if "Call" in st.radio(
        "Option", ["Call ▲ (backwardation widens)", "Put ▼ (contango deepens)"],
        horizontal=True, key="cs_side") else "put"

    so = SpreadOption(Fn, Ff, K, T_opt, cx["r"], vn, vf, rho, otype)
    res = so.price()
    mult = price_multiplier(cx["name"])

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Spread", f"{spread:+,.4f}",
        f"{curve['label'].iloc[ni]} − {curve['label'].iloc[fi]}",
        GREEN if spread > 0 else RED)
    kpi(k2, "Price", f"{res['price']:.6f}", f"${res['price'] * mult:,.0f} per lot", AMBER)
    kpi(k3, "Kirk σ", f"{res['sigma_kirk'] * 100:.2f}%",
        "spread vol, far below either leg", PURPLE)
    kpi(k4, "Δ near", f"{res['delta_long']:+.4f}", "long leg", GREEN)
    kpi(k5, "Δ far", f"{res['delta_short']:+.4f}", "short leg", RED)
    if T_opt < cx["T"] - 1e-9:
        st.caption(f"Option tenor capped at {T_opt:.3f}y — a calendar spread option cannot "
                   f"outlive its near leg, which expires {curve['expiry'].iloc[ni]}.")

    g1, g2 = st.columns(2)
    with g1:
        rng = np.linspace(spread - abs(spread) - 5, spread + abs(spread) + 5, 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rng, y=so.payoff(rng) - res["price"], name="P&L",
                                 line=dict(color=AMBER, width=2.4)))
        fig.add_hline(y=0, line=dict(color=GRAY, width=0.8))
        vline(fig, K, GREEN, f"K {K:,.3f}")
        vline(fig, spread, RED, f"Now {spread:,.3f}", y=0.82)
        fig.update_xaxes(title=f"Spread at expiry ({unit})")
        fig.update_yaxes(title=f"P&L ({unit})")
        st.plotly_chart(styled(fig, 320, "P&L at expiry"), use_container_width=True)
    with g2:
        rows = []
        for i in range(len(curve) - 1):
            for j in (i + 1, min(i + 2, len(curve) - 1), min(i + 5, len(curve) - 1)):
                if j <= i:
                    continue
                a, b = float(curve["price"].iloc[i]), float(curve["price"].iloc[j])
                t = max(float(curve["T"].iloc[i]), 1 / 365)
                p = SpreadOption(a, b, a - b, t, cx["r"], vn, vf, rho, "call").price()["price"]
                rows.append(dict(pair=f"{curve['label'].iloc[i][:3]}-{curve['label'].iloc[j][:3]}",
                                 spread=a - b, price=p))
        if rows:
            sdf = pd.DataFrame(rows).drop_duplicates("pair").head(18)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sdf["pair"], y=sdf["price"], name="ATM price",
                                 marker_color=[GREEN if s > 0 else RED for s in sdf["spread"]],
                                 opacity=0.8))
            fig.add_trace(go.Scatter(x=sdf["pair"], y=sdf["spread"], name="Spread",
                                     mode="lines+markers", yaxis="y2",
                                     line=dict(color=AMBER, width=1.5)))
            fig.update_layout(yaxis2=dict(title="Spread", overlaying="y", side="right",
                                          showgrid=False))
            fig.update_yaxes(title=f"ATM price ({unit})")
            st.plotly_chart(styled(fig, 320, "At-the-money calendar options across the curve"),
                            use_container_width=True)
        else:
            st.info("Not enough listed months to build a ladder of pairs.")


def screen_barrier(cx: dict) -> None:
    section("Barrier options — knock-in and knock-out",
            "A barrier option switches on (<b>knock-in</b>) or dies (<b>knock-out</b>) if the "
            "forward touches a level. It is cheaper than the vanilla because you give something "
            "up, and that is the point: an airline buys a knock-out call to cap fuel cost while "
            "accepting the cap disappears in a spike. Priced three ways here — closed form, "
            "discrete-monitoring correction, and Monte Carlo — because they must agree.")
    F, K, T, r, unit = cx["F_T"], cx["K"], cx["T_opt"], cx["r"], cx["unit"]
    vol = cx["vol_K"]
    b1, b2, b3 = st.columns(3)
    btype = b1.selectbox("Barrier", BARRIER_TYPES)
    otype = cx["side"].lower()
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
                        use_container_width=True)
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
                        use_container_width=True)

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
                    use_container_width=True)


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
                    use_container_width=True)
    if not cx["is_model"]:
        st.caption("The dotted line is what pure storage economics would imply from the front, "
                   "at the registry's storage and convenience defaults. The gap between it and "
                   "the market is the market's own view of scarcity — and it is a MODEL line, "
                   "which is why it is dotted and labelled.")

    carry = implied_carry(curve)
    disp = curve[["label", "delivery", "T", "price", "asof"]].copy()
    disp = disp.merge(carry[["label", "spread", "carry_ann"]], on="label", how="left")
    disp.columns = ["Contract", "Delivery", "T (yrs)", f"Price ({unit})", "Settle date",
                    "vs front", "Carry % ann."]
    st.dataframe(disp.style.format({"T (yrs)": "{:.3f}", f"Price ({unit})": "{:,.4f}",
                                    "vs front": "{:+,.4f}", "Carry % ann.": "{:+.2f}"},
                                   na_rep="—"),
                 use_container_width=True, hide_index=True, height=300)
    st.caption("T is the real year-fraction to expiry, not the contract's position in the list. "
               "On a commodity that lists only four delivery months — platinum trades Jan, Apr, "
               "Jul, Oct — treating the second contract as two months out understates its tenor "
               "by roughly 60% and, since an option is worth about √T, underprices it by a third.")


def screen_checks(cx: dict) -> None:
    section("Model self-validation",
            "Every model on this desk is priced a second way and the residual is shown. These "
            "are not decoration: the theta sign error and the inverted Kirk factor that this "
            "version fixes were both invisible to inspection and obvious to a check like these.")
    quick = st.toggle("Quick mode (fewer Monte Carlo paths)", value=True)
    with st.spinner("Running independent checks…"):
        rows = validate_models(cx["F_T"], cx["K"], cx["T_opt"], cx["r"], cx["vs"]["atm"], quick=quick)

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
                     use_container_width=True, hide_index=True)

    section("Kirk against simulation, across the parameter space")
    st.markdown('<div class="note">Kirk is an approximation, so the honest question is not '
                'whether it is exact but <b>where it stops being good</b>. Each row prices the '
                'same spread option both ways.</div>', unsafe_allow_html=True)
    rows2 = []
    for FL, FS, K, rho in [(100.0, 80.0, 15.0, 0.85), (100.0, 80.0, 25.0, 0.85),
                           (100.0, 80.0, 15.0, 0.40), (90.0, 88.0, 0.0, 0.95),
                           (100.0, 80.0, -5.0, 0.85)]:
        so = SpreadOption(FL, FS, K, cx["T"], cx["r"], 0.36, 0.32, rho, "call")
        kk = so.price()["price"]
        mm = so.mc_price(120_000)
        rows2.append(dict(F_long=FL, F_short=FS, K=K, rho=rho, Kirk=kk,
                          MonteCarlo=mm["price"], se=mm["std_error"],
                          err_pct=(kk - mm["price"]) / max(mm["price"], 1e-9) * 100))
    st.dataframe(pd.DataFrame(rows2).style.format(
        {"F_long": "{:,.1f}", "F_short": "{:,.1f}", "K": "{:+,.1f}", "rho": "{:.2f}",
         "Kirk": "{:.4f}", "MonteCarlo": "{:.4f}", "se": "{:.4f}", "err_pct": "{:+.2f}%"}),
        use_container_width=True, hide_index=True)
    st.caption("Kirk is at its best when the effective strike stays well away from zero — deep "
               "in-the-money spreads with a large negative strike are where it drifts, and the "
               "simulation column is there so you can see it happen rather than assume it does not.")

    section("What this desk does not model, and why")
    st.markdown(
        f'<div class="note">'
        f'<b>No listed-options feed.</b> Every volatility here is your input or a registry '
        f'default. The surface is a parametrisation, not a calibration, and cannot mark a book.'
        f'<br><b>Discrete monitoring.</b> Barrier prices use daily fixings by default; the '
        f'closed form is continuous, so the two are reconciled explicitly rather than blurred.'
        f'<br><b>Blended structure vols.</b> A multi-product crack collapses its legs into one '
        f'synthetic forward with a value-weighted vol — the correlation between gasoline and '
        f'distillate is not modelled.'
        f'<br><b>Flat rates.</b> One discount rate, no curve. For tenors out to two years on a '
        f'commodity book the error is small; for anything longer it is not.'
        f'<br><b>Instruments removed from the previous version</b> — LME metals, Baltic freight, '
        f'carbon, coal, uranium, Platts assessments — had no resolvable feed, so their "prices" '
        f'were randomly generated and drawn in the same charts as real ones. They are gone '
        f'rather than patched.</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([dict(Instrument=k, Reason=v) for k, v in EXCLUSIONS.items()]),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  9. NAVIGATION AND MAIN
#  Seven screens, ordered by how often they are opened. The three a trader uses
#  every day are top-level; the rest are grouped behind them, because twelve
#  equal tabs make the hierarchy of use invisible.
# ══════════════════════════════════════════════════════════════════════════════

SCREENS = [
    ("⛓  Chain", "the ladder, and where you turn prices into a surface", screen_chain),
    ("🧩  Trade", "price one option or a structure", screen_trade),
    ("📚  Book", "what you hold: Greeks, scenarios, margin", screen_book),
    ("🌊  Volatility", "surface, cones, and whether hedging pays", None),
    ("🧱  Structures", "cracks, calendar spreads, Asians, barriers", None),
    ("📈  Market", "the forward curve behind every price", screen_curve),
    ("🔬  Checks", "what the models get right, and their limits", screen_checks),
]


def _setup_page() -> None:
    st.set_page_config(page_title="CODAP — Commodity Options Desk",
                       page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)


def render_sidebar() -> dict:
    """Contract, curve, tenor, volatility, rate — in that order, because that is
    the order in which each one stops mattering if the one above it is wrong."""
    with st.sidebar:
        st.markdown(f'<div style="padding:8px 0 4px"><span style="font-size:1.3rem;'
                    f'font-weight:700;letter-spacing:-.03em;background:linear-gradient'
                    f'(90deg,{AMBER},{RED});-webkit-background-clip:text;'
                    f'-webkit-text-fill-color:transparent">CODAP</span>'
                    f'<div style="font-size:.7rem;color:#6E7681">Commodity options desk</div>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown("---")

        sector = st.selectbox("Sector", SECTORS)
        name = st.selectbox("Contract", contracts_in(sector))
        cfg = CONTRACTS[name]
        st.caption(f"{cfg['unit']} · {cfg['size']:,} {cfg['size_unit']}/lot · "
                   f"strikes every {cfg['strike_inc']:g}")

        source = st.radio("Curve", ["Live market", "Model (cost of carry)"],
                          help="Live pulls the listed strip. Model builds a cost-of-carry "
                               "curve from a spot you enter — for pricing offline or a "
                               "scenario that is not today's market.")
        is_model = source.startswith("Model")
        spot_in, storage, convenience = 0.0, cfg["storage"], cfg["convenience"]
        if is_model:
            spot_in = st.number_input(f"Spot ({cfg['unit']})", min_value=1e-6,
                                      value=float(_last_known_spot(name)), step=0.01,
                                      format="%.4f")
            storage = st.number_input("Storage (%/yr)", 0.0, 60.0,
                                      cfg["storage"] * 100, 0.1) / 100
            convenience = st.number_input("Convenience yield (%/yr)", 0.0, 60.0,
                                          cfg["convenience"] * 100, 0.1) / 100
        T_months = st.slider("Default maturity (months)", 1, 36, 6,
                             help="Screens that need one expiry use this; the Chain lets "
                                  "you pick any listed month.")
        st.markdown("---")

        st.markdown(f'<div style="font-size:.72rem;font-weight:600;color:{PURPLE};'
                    f'text-transform:uppercase;letter-spacing:.09em">Volatility</div>',
                    unsafe_allow_html=True)
        st.caption("An implied vol cannot be downloaded — it is an option price inverted. "
                   "Paste prices in the Chain and this becomes your market.")
        override = st.checkbox("Override the level manually", value=False,
                               help="Off: the app uses your calibrated quotes if you have "
                                    "pasted any, otherwise the realised volatility it "
                                    "computes from price history.")
        manual_vol = None
        if override:
            manual_vol = st.slider("ATM volatility (%)", 1, 200,
                                   int(cfg["vol"] * 100), 1) / 100
        st.session_state["use_smile"] = st.checkbox(
            "Apply a smile (SABR)", value=True,
            help="On: each strike prices at its own σ(K). Off: one flat number for every "
                 "strike, which is visibly wrong in the wings.")
        if st.session_state["use_smile"] and not st.session_state.get("sabr_cal"):
            with st.expander("Smile shape"):
                b_ = st.select_slider("β backbone", options=[0.0, 0.25, 0.5, 0.75, 1.0],
                                      value=0.5)
                rho_ = st.slider("ρ skew", -95, 95, -30, 1) / 100
                nu_ = st.slider("ν vol-of-vol", 1, 300, 70, 1) / 100
                st.session_state["sabr_shape"] = dict(beta=b_, rho=rho_, nu=nu_)
        st.markdown("---")

        r = st.number_input("Discount rate (%)", -5.0, 30.0, 4.25, 0.05, format="%.2f",
                            help="Used for discounting and for American early exercise.") / 100
        n_paths = st.select_slider("Monte Carlo paths",
                                   options=[10_000, 25_000, 50_000, 100_000], value=25_000,
                                   help="Only the simulated screens use these.")
        if st.button("↻  Refresh market data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        with st.expander("🔧 Diagnostics"):
            st.code("\n".join(list(FEED_LOG)[-20:]) or "nothing logged", language="text")

    return dict(name=name, cfg=cfg, unit=cfg["unit"], sector=sector, is_model=is_model,
                spot_in=spot_in, storage=storage, convenience=convenience,
                T_months=T_months, manual_vol=manual_vol, r=r, n_paths=n_paths,
                side="Call", curves={})


def build_context(cx: dict) -> dict:
    name = cx["name"]
    T = cx["T_months"] / 12
    curve = (model_curve(name, cx["spot_in"], cx["r"], cx["storage"], cx["convenience"],
                         n=max(cx["T_months"], 12)) if cx["is_model"]
             else fetch_curve(name))
    cx["curve"] = curve
    cx["curves"][name] = curve
    cx["T"] = T
    cx["T_opt"] = option_tenor(T)
    cx["F_T"] = forward_at(curve, T) if not curve.empty else None
    cx["vs"] = resolve_vol_source(name, cx["F_T"], cx["T_opt"], cx["r"], cx["manual_vol"])
    cx["vol"] = cx["vs"]["atm"]
    cx["vol_fn"] = cx["vs"]["fn"]
    cx["sabr_params"] = cx["vs"]["params"]
    cx["K"] = snap_strike(name, cx["F_T"]) if cx["F_T"] else None
    cx["vol_K"] = cx["vs"]["fn"](cx["K"], cx["T_opt"]) if cx["K"] else None
    return cx


def render_header(cx: dict) -> None:
    curve, vs = cx["curve"], cx["vs"]
    if cx["is_model"]:
        state, sub = badge("MODEL CURVE", AMBER), "cost of carry from your spot"
    elif curve.empty:
        state, sub = badge("NO MARKET DATA", RED), "no listed contract returned a settle"
    else:
        asof = max(d for d in curve["asof"] if d is not None)
        stale = curve_is_stale(curve)
        state = badge(f"{'STALE' if stale else 'LIVE'} · {asof}", AMBER if stale else GREEN)
        sub = f"{len(curve)} dated contracts marked"
    src_col = {"quotes": GREEN, "realised": AMBER, "manual": BLUE, "registry": RED}
    bits = [badge(cx["name"], TEXT), badge(cx["unit"]), state,
            badge(f"σ {vs['atm'] * 100:.1f}%", PURPLE),
            badge(VOL_SOURCES.get(vs["source"], ("MANUAL", ""))[0],
                  src_col.get(vs["source"], BLUE)),
            badge("smile" if vs["smile"] else "flat vol", PURPLE),
            badge(f"r {cx['r'] * 100:.2f}%")]
    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 2px">'
                + "".join(bits) + "</div>", unsafe_allow_html=True)
    st.caption(f"{sub} · volatility: {vs['detail']}")


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
        st.warning("yfinance is not installed — live curves unavailable. "
                   "`pip install yfinance`, or use the model curve.")
    cx = build_context(render_sidebar())
    render_header(cx)

    if cx["curve"].empty or not cx["F_T"]:
        st.error("**No curve, so nothing can be priced.** Every model here prices off a "
                 "forward. Switch to a model curve in the sidebar to price a scenario, or "
                 "check the diagnostics.")
        return

    tabs = st.tabs([s[0] for s in SCREENS])
    with tabs[0]:
        guard("Chain", screen_chain, cx)
    with tabs[1]:
        guard("Trade", screen_trade, cx)
    with tabs[2]:
        guard("Book", screen_book, cx)
    with tabs[3]:
        sub = st.tabs(["Surface (SABR)", "Cones & gamma scalping"])
        with sub[0]:
            guard("SABR", screen_sabr, cx)
        with sub[1]:
            guard("Vol analytics", screen_vol_analytics, cx)
    with tabs[4]:
        sub = st.tabs(["Cracks & Crush", "Calendar spreads", "Asian", "Barrier"])
        for s_, fn in zip(sub, (screen_structure, screen_calendar,
                                screen_asian, screen_barrier)):
            with s_:
                guard("Structures", fn, cx)
    with tabs[5]:
        guard("Market", screen_curve, cx)
    with tabs[6]:
        guard("Checks", screen_checks, cx)

    st.markdown("---")
    st.caption("CODAP · every price rests on a forward that was fetched and a volatility "
               "whose source is named on screen · by Adam EL GBOURI")


if __name__ == "__main__":
    main()
