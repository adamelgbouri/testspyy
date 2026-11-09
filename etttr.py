"""Meridian Energy Desk: European gas and power trading and risk.

Run with:   streamlit run etttr.py          (needs Python 3.10+ and: pip install -U streamlit pandas plotly numpy)
Test with:  pytest etttr.py

Everything is in this one file, in order:
  1 Calculation engine      units, calendars, curves, trades, valuation
  2 Analytics engine        tenors, curve building, P&L explain, risk, spreads, spot, seasonal,
                            Hedge Advisor and Market Read
  3 Saved data              SQLite file beside this script (meridian_desk.db)
  4 Market data connectors  ENTSO-E, GIE, ENTSOG, Open-Meteo, ECB, licensed providers
  5 Design system           colours, styles, chart template, shared components
  6 Navigation and session
  7 Model checks
  8 Pages
  9 App entry point

The sections above the pages import without opening the app. Built-in curves and trades
are synthetic demo data, not market quotes. This is an educational prototype, not a
production ETRM or valuation system. API keys live in meridian_keys.toml beside this file: never share that one.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, time, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter, sleep
from zoneinfo import ZoneInfo
import gzip
import hashlib
import html
import inspect
import io
import json
import math
import os
import re
import sqlite3
import textwrap
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

MIN_STREAMLIT_VERSION = (1, 50)

# pandas 3 always copies on write. Earlier versions need it switched on, which makes
# sharing large frames between calculations safe.
if pd.__version__.split(".")[0] == "2":
    pd.options.mode.copy_on_write = True

# =============================================================================
# SECTION 1 · CALCULATION ENGINE
# Units, delivery calendars, curves, trades, mark-to-market and operations.
# =============================================================================
MMBTU_PER_MWH = 3.412141633
THERM_PER_MWH = 34.12141633

PRODUCTS = {
    "TTF Natural Gas": {"commodity": "Gas", "zone": "TTF", "currency": "EUR", "unit": "EUR/MWh", "default_price": 35.0, "timezone": "Europe/Amsterdam", "profile": "BASE"},
    "NBP Natural Gas": {"commodity": "Gas", "zone": "NBP", "currency": "GBP", "unit": "GBP/MWh", "default_price": 29.0, "timezone": "Europe/London", "profile": "BASE"},
    "German Power Base": {"commodity": "Power", "zone": "Germany", "currency": "EUR", "unit": "EUR/MWh", "default_price": 85.0, "timezone": "Europe/Berlin", "profile": "BASE"},
    "German Power Peak": {"commodity": "Power", "zone": "Germany", "currency": "EUR", "unit": "EUR/MWh", "default_price": 99.0, "timezone": "Europe/Berlin", "profile": "PEAK"},
    "French Power Base": {"commodity": "Power", "zone": "France", "currency": "EUR", "unit": "EUR/MWh", "default_price": 78.0, "timezone": "Europe/Paris", "profile": "BASE"},
    "French Power Peak": {"commodity": "Power", "zone": "France", "currency": "EUR", "unit": "EUR/MWh", "default_price": 92.0, "timezone": "Europe/Paris", "profile": "PEAK"},
}

# Reference curves feed spread calculations. They can't be traded in the MW-based book.
REFERENCE_PRODUCTS = {
    "EUA Carbon": {"commodity": "Carbon", "zone": "EU ETS", "currency": "EUR", "unit": "EUR/tCO2", "default_price": 72.0, "timezone": "Europe/Brussels", "seasonality": 0.0},
    "API2 Coal": {"commodity": "Coal", "zone": "ARA", "currency": "USD", "unit": "USD/t", "default_price": 105.0, "timezone": "Europe/Amsterdam", "seasonality": 0.02},
}
CURVE_PRODUCTS = {**PRODUCTS, **REFERENCE_PRODUCTS}
DEMO_CURVE_MONTHS = 36

TRADE_COLUMNS = [
    "trade_id", "trade_date", "counterparty", "product", "instrument", "side",
    "delivery_start", "delivery_end", "profile", "quantity_mw", "quantity_mwh",
    "trade_price", "currency", "settlement_type", "status", "notes",
]


def mw_to_mwh(power_mw: float, hours: float) -> float:
    """Convert constant power in MW over a delivery duration into energy in MWh."""
    if not math.isfinite(power_mw) or not math.isfinite(hours):
        raise ValueError("Power and hours must be finite numbers.")
    if hours < 0:
        raise ValueError("Hours cannot be negative.")
    return power_mw * hours


def mwh_to_mmbtu(energy_mwh: float) -> float:
    """Convert energy MWh to MMBtu using 1 MWh = 3.412141633 MMBtu."""
    if not math.isfinite(energy_mwh):
        raise ValueError("Energy must be finite.")
    return energy_mwh * MMBTU_PER_MWH


def mmbtu_to_mwh(energy_mmbtu: float) -> float:
    if not math.isfinite(energy_mmbtu):
        raise ValueError("Energy must be finite.")
    return energy_mmbtu / MMBTU_PER_MWH


def price_eur_mwh_to_gbp_therm(price_eur_mwh: float, eur_per_gbp: float) -> float:
    """Convert EUR/MWh to GBP/therm. eur_per_gbp means EUR per GBP (e.g. 1.17)."""
    if eur_per_gbp <= 0 or not math.isfinite(eur_per_gbp):
        raise ValueError("EUR per GBP must be a positive finite FX rate.")
    if not math.isfinite(price_eur_mwh):
        raise ValueError("Price must be finite.")
    # 1 MWh = 34.12141633 therms; first convert EUR to GBP, then per MWh to per therm.
    return (price_eur_mwh / eur_per_gbp) / THERM_PER_MWH


def mtm_linear(signed_quantity_mwh: float, trade_price: float, market_price: float,
               discount_factor: float = 1.0) -> float:
    """Undiscounted/discounted linear MtM in quote currency.

    Signed quantity convention: positive = long/buy fixed price; negative = short/sell.
    P&L = signed MWh * (market forward - contracted fixed price) * discount factor.
    Both prices must be in the same currency per MWh and for the same delivery product.
    """
    vals = [signed_quantity_mwh, trade_price, market_price, discount_factor]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("MtM inputs must be finite.")
    if discount_factor < 0 or discount_factor > 1.5:
        raise ValueError("Discount factor must be between 0 and 1.5.")
    return signed_quantity_mwh * (market_price - trade_price) * discount_factor


def daily_pnl(previous_mtm: float, current_mtm: float, realised_pnl: float = 0.0) -> float:
    """Simplified daily P&L: change in unrealised MtM plus separately supplied realised P&L."""
    if not all(math.isfinite(x) for x in [previous_mtm, current_mtm, realised_pnl]):
        raise ValueError("P&L inputs must be finite.")
    return (current_mtm - previous_mtm) + realised_pnl


def quote_price(reference_price: float, margin_per_mwh: float, side: str) -> float:
    """Indicative client quote with an explicit desk-side margin convention.

    SELL_TO_CLIENT: desk sells energy to client, quote = reference + margin.
    BUY_FROM_CLIENT: desk buys energy from client, quote = reference - margin.
    This is an illustrative commercial convention, not a universal market rule.
    """
    if not math.isfinite(reference_price) or not math.isfinite(margin_per_mwh):
        raise ValueError("Quote inputs must be finite.")
    if margin_per_mwh < 0:
        raise ValueError("Margin cannot be negative.")
    s = str(side).strip().upper()
    if s == "SELL_TO_CLIENT":
        return reference_price + margin_per_mwh
    if s == "BUY_FROM_CLIENT":
        return reference_price - margin_per_mwh
    raise ValueError("side must be SELL_TO_CLIENT or BUY_FROM_CLIENT")


def build_sample_trades(valuation_date: date | None = None) -> pd.DataFrame:
    valuation_date = valuation_date or date.today()
    d0 = valuation_date.replace(day=1)
    rows = [
        {"trade_id": "TR-1001", "trade_date": valuation_date.isoformat(), "counterparty": "Northstar Utilities (Demo)",
         "product": "TTF Natural Gas", "instrument": "SWAP", "settlement_type": "PHYSICAL", "side": "BUY_FIXED", "delivery_start": _month_start(1, d0).isoformat(),
         "delivery_end": _month_start(4, d0).isoformat(), "profile": "BASE", "quantity_mw": 10.0,
         "trade_price": 35.5, "currency": "EUR", "status": "EXECUTED", "notes": "Illustrative long fixed-price gas swap"},
        {"trade_id": "TR-1002", "trade_date": valuation_date.isoformat(), "counterparty": "Alpine Power (Demo)",
         "product": "French Power Peak", "instrument": "FORWARD", "settlement_type": "PHYSICAL", "side": "SELL_FIXED", "delivery_start": _month_start(1, d0).isoformat(),
         "delivery_end": _month_start(2, d0).isoformat(), "profile": "PEAK", "quantity_mw": 5.0,
         "trade_price": 90.0, "currency": "EUR", "status": "EXECUTED", "notes": "Illustrative short fixed-price peak power"},
        {"trade_id": "TR-1003", "trade_date": valuation_date.isoformat(), "counterparty": "BlueGrid Energy (Demo)",
         "product": "German Power Base", "instrument": "SWAP", "settlement_type": "FINANCIAL", "side": "BUY_FIXED", "delivery_start": _month_start(2, d0).isoformat(),
         "delivery_end": _month_start(5, d0).isoformat(), "profile": "BASE", "quantity_mw": 8.0,
         "trade_price": 82.0, "currency": "EUR", "status": "EXECUTED", "notes": "Illustrative long fixed-price base power"},
    ]
    return normalise_trades(pd.DataFrame(rows), allow_missing_quantity_mwh=True)


def _month_start(offset: int, current_month_start: date) -> date:
    p = pd.Timestamp(current_month_start).to_period("M") + offset
    return p.to_timestamp().date()


def position_summary_by_delivery(trades: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate executed net/gross positions and MtM by product, tenor and profile."""
    columns = ["product", "delivery_month", "profile", "currency", "net_mwh", "gross_mwh",
               "mtm", "unmarked_buckets", "trade_count"]
    detail = trade_delivery_breakdown(trades, curves)
    if detail.empty:
        return pd.DataFrame(columns=columns)
    grouped_rows = []
    keys = ["product", "delivery_month", "profile", "currency"]
    for key, group in detail.groupby(keys, dropna=False, sort=True):
        all_marks = bool(group["mark_status"].eq("MARKED").all())
        available_mtm = group.loc[group["mark_status"].eq("MARKED"), "mtm"].sum(min_count=1)
        grouped_rows.append({
            **dict(zip(keys, key)),
            "net_mwh": float(group["net_mwh"].sum(min_count=1)),
            "gross_mwh": float(group["gross_mwh"].sum(min_count=1)),
            "mtm": float(available_mtm) if all_marks and pd.notna(available_mtm) else float("nan"),
            "unmarked_buckets": int(group["mark_status"].ne("MARKED").sum()),
            "trade_count": int(group["trade_id"].nunique()),
        })
    return pd.DataFrame(grouped_rows, columns=columns).sort_values(["delivery_month", "product", "profile"]).reset_index(drop=True)


def position_summary(marked_trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successfully marked executed delivery volume by product/currency."""
    columns = ["product", "signed_mwh", "mtm", "currency"]
    if marked_trades.empty:
        return pd.DataFrame(columns=columns)
    eligible = marked_trades.loc[marked_trades["status"].astype(str).str.upper().eq("EXECUTED")].copy()
    if "mark_status" in eligible.columns:
        eligible = eligible.loc[eligible["mark_status"].eq("MARKED")]
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    return (eligible.groupby(["product", "currency"], as_index=False)
            .agg(signed_mwh=("signed_quantity_mwh", "sum"), mtm=("mtm", "sum")))


def filter_trade_book(marked_trades: pd.DataFrame, product_filter: str = "All",
                      status_filter: str = "All") -> pd.DataFrame:
    """Apply the blotter's product and lifecycle filters consistently to all displayed KPIs."""
    result = marked_trades.copy()
    if product_filter != "All" and "product" in result.columns:
        result = result.loc[result["product"].eq(product_filter)]
    if status_filter != "All" and "status" in result.columns:
        result = result.loc[result["status"].astype(str).str.upper().eq(status_filter.upper())]
    return result.reset_index(drop=True)


def calculate_book_metrics(marked_trades: pd.DataFrame) -> dict[str, float | int]:
    """Metrics derived from exactly the rows shown in a filtered trade-book view."""
    if marked_trades.empty:
        return {"trade_count": 0, "executed_count": 0, "unmarked_count": 0,
                "eur_mtm": 0.0, "gbp_mtm": 0.0}
    active = marked_trades.loc[~marked_trades["status"].astype(str).str.upper().isin(["CANCELLED", "EXPIRED"])]
    marked = active.loc[active.get("mark_status", pd.Series("MARKED", index=active.index)).eq("MARKED")]
    executed = marked.loc[marked["status"].astype(str).str.upper().eq("EXECUTED")]
    unmarked = active.loc[~active.index.isin(marked.index)]
    return {
        "trade_count": int(len(marked_trades)),
        "executed_count": int(executed.shape[0]),
        "unmarked_count": int(unmarked.shape[0]),
        "eur_mtm": float(executed.loc[executed["currency"].eq("EUR"), "mtm"].sum()),
        "gbp_mtm": float(executed.loc[executed["currency"].eq("GBP"), "mtm"].sum()),
    }


def add_trade_to_book(existing: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Validate a new trade and append it without mutating the existing blotter."""
    if existing.empty:
        base = pd.DataFrame(columns=TRADE_COLUMNS)
    else:
        base = normalise_trades(existing)
    new = normalise_trades(candidate)
    existing_ids = set(base["trade_id"].astype(str)) if not base.empty else set()
    collisions = existing_ids.intersection(set(new["trade_id"].astype(str)))
    if collisions:
        raise ValueError(f"Trade ID already exists: {sorted(collisions)}")
    combined = pd.concat([base, new], ignore_index=True, sort=False)
    # Recompute enriched quantities after validation; preserve a stable schema.
    return normalise_trades(combined)


def set_trade_status(trades: pd.DataFrame, trade_id: str, new_status: str) -> pd.DataFrame:
    """Update a trade lifecycle status while retaining the record for auditability."""
    target = str(new_status).upper().strip()
    if target not in {"PROPOSED", "CONFIRMED", "EXECUTED", "CANCELLED", "EXPIRED"}:
        raise ValueError(f"Unsupported trade status: {target}")
    out = trades.copy()
    matches = out["trade_id"].astype(str).eq(str(trade_id))
    if int(matches.sum()) != 1:
        raise ValueError(f"Expected exactly one trade with ID {trade_id}; found {int(matches.sum())}.")
    current = str(out.loc[matches, "status"].iloc[0]).upper().strip()
    transitions = {
        "PROPOSED": {"CONFIRMED", "EXECUTED", "CANCELLED", "EXPIRED"},
        "CONFIRMED": {"EXECUTED", "CANCELLED", "EXPIRED"},
        "EXECUTED": {"CANCELLED", "EXPIRED"},
    }
    if target not in transitions.get(current, set()):
        raise ValueError(f"Invalid trade lifecycle transition: {current} → {target}.")
    out.loc[matches, "status"] = target
    return normalise_trades(out)


def load_trade_book_csv(source) -> pd.DataFrame:
    """Load, validate and normalize a CSV blotter from a file-like object or path."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    elif isinstance(source, str) and ("\n" in source or "," in source) and not PathLikeExists(source):
        source = io.StringIO(source)
    candidate = pd.read_csv(source)
    return normalise_trades(candidate)


def PathLikeExists(value: str) -> bool:
    # Avoid requiring pathlib in the public API; existing local files remain supported.
    try:
        from pathlib import Path
        return Path(value).exists()
    except Exception:
        return False


PHYSICAL_OP_COLUMNS = ["contract_id", "trade_id", "product", "counterparty", "delivery_start",
                        "delivery_end", "profile", "contracted_mwh", "nominated_mwh", "delivered_mwh",
                        "operational_status", "invoice_status", "updated_at"]


def build_sample_physical_ops(trades: pd.DataFrame) -> pd.DataFrame:
    """Create illustrative operations records for trades explicitly flagged physical."""
    rows = []
    normalized = normalise_trades(trades) if not trades.empty else pd.DataFrame()
    if normalized.empty:
        return pd.DataFrame(columns=PHYSICAL_OP_COLUMNS)
    eligible = normalized.loc[normalized["settlement_type"].eq("PHYSICAL") & normalized["status"].isin(["CONFIRMED", "EXECUTED"])]
    for _, tr in eligible.iterrows():
        rows.append({"contract_id": f"PHY-{tr['trade_id']}", "trade_id": tr["trade_id"],
                     "product": tr["product"], "counterparty": tr["counterparty"],
                     "delivery_start": tr["delivery_start"], "delivery_end": tr["delivery_end"],
                     "profile": tr["profile"], "contracted_mwh": float(tr["quantity_mwh"]),
                     "nominated_mwh": 0.0, "delivered_mwh": 0.0,
                     "operational_status": "SCHEDULED", "invoice_status": "NOT_DUE",
                     "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return pd.DataFrame(rows, columns=PHYSICAL_OP_COLUMNS)


def sync_physical_ops_with_trades(operations: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Add operations records for newly added physical trades without overwriting progress."""
    ops = operations.copy() if operations is not None else pd.DataFrame(columns=PHYSICAL_OP_COLUMNS)
    for col in PHYSICAL_OP_COLUMNS:
        if col not in ops.columns:
            ops[col] = pd.Series(dtype="object")
    try:
        expected = build_sample_physical_ops(trades)
    except Exception:
        # Keep existing operational work visible if the trade book itself has an issue.
        return ops[PHYSICAL_OP_COLUMNS].copy().reset_index(drop=True)
    existing_ids = set(ops["trade_id"].astype(str))
    additions = expected.loc[~expected["trade_id"].astype(str).isin(existing_ids)]
    if not additions.empty:
        ops = pd.concat([ops[PHYSICAL_OP_COLUMNS], additions], ignore_index=True, sort=False)
    return ops[PHYSICAL_OP_COLUMNS].reset_index(drop=True)


def update_physical_operation(operations: pd.DataFrame, contract_id: str, nominated_mwh: float,
                              delivered_mwh: float, operational_status: str,
                              invoice_status: str) -> pd.DataFrame:
    """Persist validated operation edits by returning an updated operations table."""
    for name, value in [("nominated_mwh", nominated_mwh), ("delivered_mwh", delivered_mwh)]:
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be a non-negative finite number.")
    valid_ops = {"SCHEDULED", "NOMINATED", "IN_DELIVERY", "DELIVERED", "ON_HOLD", "CANCELLED"}
    valid_invoice = {"NOT_DUE", "DRAFT", "ISSUED", "PAID", "DISPUTED"}
    status = str(operational_status).upper().strip()
    invoice = str(invoice_status).upper().strip()
    if status not in valid_ops:
        raise ValueError(f"Unsupported operational status: {status}")
    if invoice not in valid_invoice:
        raise ValueError(f"Unsupported invoice status: {invoice}")
    out = operations.copy()
    matches = out["contract_id"].astype(str).eq(str(contract_id))
    if int(matches.sum()) != 1:
        raise ValueError(f"Expected exactly one operations contract {contract_id}; found {int(matches.sum())}.")
    out.loc[matches, "nominated_mwh"] = float(nominated_mwh)
    out.loc[matches, "delivered_mwh"] = float(delivered_mwh)
    out.loc[matches, "operational_status"] = status
    out.loc[matches, "invoice_status"] = invoice
    out.loc[matches, "updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def reconcile_physical_ops(operations: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Reconcile each physical-operation record to its linked trade-book record."""
    if operations.empty:
        return pd.DataFrame(columns=list(PHYSICAL_OP_COLUMNS) + ["trade_status", "reconciliation",
                          "outstanding_mwh", "nomination_gap_mwh", "delivery_vs_nomination_mwh", "risk_flag"])
    trade_lookup = {}
    if not trades.empty:
        normalized = normalise_trades(trades)
        trade_lookup = {str(row["trade_id"]): row for _, row in normalized.iterrows()}
    rows = []
    for _, op in operations.iterrows():
        trade = trade_lookup.get(str(op["trade_id"]))
        reconciliation = "MATCH"
        trade_status = "MISSING"
        if trade is None:
            reconciliation = "NO LINKED TRADE"
        else:
            trade_status = str(trade["status"])
            if trade_status in {"CANCELLED", "EXPIRED"}:
                reconciliation = f"TRADE {trade_status}"
            elif trade["settlement_type"] != "PHYSICAL":
                reconciliation = "SETTLEMENT TYPE MISMATCH"
            elif trade["product"] != op["product"]:
                reconciliation = "PRODUCT MISMATCH"
            elif trade["counterparty"] != op["counterparty"]:
                reconciliation = "COUNTERPARTY MISMATCH"
            elif trade["delivery_start"] != pd.Timestamp(op["delivery_start"]).date():
                reconciliation = "DELIVERY START MISMATCH"
            elif trade["delivery_end"] != pd.Timestamp(op["delivery_end"]).date():
                reconciliation = "DELIVERY END MISMATCH"
            elif trade["profile"] != op["profile"]:
                reconciliation = "DELIVERY PROFILE MISMATCH"
            elif not math.isclose(float(trade["quantity_mwh"]), float(op["contracted_mwh"]), rel_tol=1e-6, abs_tol=0.1):
                reconciliation = "VOLUME MISMATCH"
        contracted = float(op["contracted_mwh"])
        nominated = float(op["nominated_mwh"])
        delivered = float(op["delivered_mwh"])
        outstanding = contracted - delivered
        nomination_gap = contracted - nominated
        delivery_vs_nomination = delivered - nominated
        if reconciliation != "MATCH":
            risk_flag = reconciliation
        elif delivered > contracted + 1e-6:
            risk_flag = "OVERDELIVERY"
        elif nominated == 0 and contracted > 0:
            risk_flag = "NOMINATION MISSING"
        elif delivered + 1e-6 < nominated:
            risk_flag = "DELIVERY BELOW NOMINATION"
        elif outstanding > 1e-6:
            risk_flag = "DELIVERY OUTSTANDING"
        else:
            risk_flag = "RECONCILED"
        record = op.to_dict()
        record.update({"trade_status": trade_status, "reconciliation": reconciliation,
                       "outstanding_mwh": outstanding, "nomination_gap_mwh": nomination_gap,
                       "delivery_vs_nomination_mwh": delivery_vs_nomination, "risk_flag": risk_flag})
        rows.append(record)
    return pd.DataFrame(rows)


def parallel_shock_pnl(signed_quantity_mwh: float, price_shock_per_mwh: float) -> float:
    """First-order P&L change for an absolute parallel price shift in quote-currency/MWh.

    Absolute shocks are intentionally used instead of percentage shocks so the scenario
    remains interpretable when power prices are zero or negative. Results remain in the
    individual trade's quote currency and must not be combined across currencies.
    """
    if not math.isfinite(signed_quantity_mwh) or not math.isfinite(price_shock_per_mwh):
        raise ValueError("Position and price shock must be finite numbers.")
    return float(signed_quantity_mwh) * float(price_shock_per_mwh)


def active_trade_book(trades: pd.DataFrame) -> pd.DataFrame:
    """Return trades eligible for marking; cancelled and expired trades are excluded."""
    if trades.empty:
        return trades.copy()
    if "status" not in trades.columns:
        raise ValueError("Trade book is missing the required status column.")
    status = trades["status"].astype(str).str.upper().str.strip()
    return trades.loc[~status.isin(["CANCELLED", "EXPIRED"])].copy()


def _check_sign_convention() -> None:
    # Runtime sanity checks for the most important financial sign conventions.
    assert mtm_linear(100.0, 50.0, 55.0) == 500.0
    assert mtm_linear(-100.0, 50.0, 55.0) == -500.0
    assert mtm_linear(100.0, 55.0, 50.0) == -500.0


_check_sign_convention()


def _to_date(value) -> date:
    """Accept date, datetime, Timestamp or ISO text and return a plain date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _add_months(first_of_month: date, months: int) -> date:
    """First day of the month `months` after the month containing the given date."""
    year, month_index = divmod(first_of_month.year * 12 + first_of_month.month - 1 + int(months), 12)
    return date(year, month_index + 1, 1)


@lru_cache(maxsize=50000)
def _delivery_hours_cached(start: date, end_exclusive: date, profile: str, timezone_name: str) -> int:
    try:
        zone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise ValueError(f"Unknown IANA timezone: {timezone_name}") from exc
    if profile == "BASE":
        local_start = datetime.combine(start, time.min, tzinfo=zone).astimezone(timezone.utc)
        local_end = datetime.combine(end_exclusive, time.min, tzinfo=zone).astimezone(timezone.utc)
        return int((local_end - local_start).total_seconds() // 3600)
    if profile == "PEAK":
        return int(np.busday_count(start, end_exclusive)) * 12
    raise ValueError("Profile must be BASE or PEAK.")


def delivery_hours(start: date, end_exclusive: date, profile: str = "BASE",
                    timezone_name: str = "Europe/Paris") -> int:
    """Delivery hours for [start, end_exclusive) in the product's market timezone.

    BASE counts actual elapsed hours between local midnights, so 23/25-hour
    daylight-saving days are handled. PEAK is weekdays 08:00-20:00 local time (12 hours).
    """
    start_day, end_day = _to_date(start), _to_date(end_exclusive)
    if end_day <= start_day:
        raise ValueError("Delivery end must be after delivery start (end is exclusive).")
    return _delivery_hours_cached(start_day, end_day, str(profile).strip().upper(), timezone_name)


@lru_cache(maxsize=50000)
def _month_segments(start: date, end_exclusive: date, profile: str, timezone_name: str) -> tuple:
    """Split a delivery window into (first-of-month, delivery hours) pairs."""
    segments = []
    cursor = start.replace(day=1)
    while cursor < end_exclusive:
        following = _add_months(cursor, 1)
        hours = delivery_hours(max(start, cursor), min(end_exclusive, following), profile, timezone_name)
        if hours > 0:
            segments.append((cursor, hours))
        cursor = following
    return tuple(segments)


def _strip_price(lookup: dict, start: date, end_exclusive: date, profile: str, timezone_name: str) -> float:
    """Delivery-hour-weighted price of a strip from a {month: price} lookup."""
    segments = _month_segments(start, end_exclusive, profile, timezone_name)
    if not segments:
        raise ValueError("Contract has no delivery hours for the selected profile.")
    total_value, total_hours = 0.0, 0
    for month, hours in segments:
        if month not in lookup:
            raise ValueError(f"Missing or ambiguous curve mark for {month:%Y-%m}.")
        total_value += float(lookup[month]) * hours
        total_hours += hours
    return total_value / total_hours


@lru_cache(maxsize=4096)
def _sample_curve_rows(product: str, valuation_date: date, months: int) -> tuple:
    spec = CURVE_PRODUCTS[product]
    base = float(spec["default_price"])
    seasonality = float(spec.get("seasonality", 0.07))
    first = _add_months(valuation_date, 1)
    rows = []
    for i in range(months):
        month = _add_months(first, i)
        # Smooth, deterministic seasonal shape purely for demonstration.
        season = seasonality * math.cos(2 * math.pi * (month.month - 1) / 12)
        contango = 0.0015 * i
        rows.append((month, round(base * (1 + season + contango), 4)))
    return tuple(rows)


def make_sample_curve(product: str, valuation_date: date | None = None, months: int = 18) -> pd.DataFrame:
    """Create transparent illustrative monthly forward prices; not live market data."""
    if product not in CURVE_PRODUCTS:
        raise ValueError(f"Unsupported product: {product}")
    if months < 1 or months > 120:
        raise ValueError("months must be between 1 and 120")
    valuation_date = valuation_date or date.today()
    spec = CURVE_PRODUCTS[product]
    rows = _sample_curve_rows(product, valuation_date, months)
    return pd.DataFrame({
        "delivery_month": [row[0] for row in rows],
        "product": product,
        "price": [row[1] for row in rows],
        "currency": spec["currency"],
        "unit": spec["unit"],
        "data_status": "ILLUSTRATIVE — NOT A LIVE QUOTE",
    })


def validate_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Validate monthly forward marks, canonicalise tenors, and enforce product units."""
    required = {"delivery_month", "product", "price", "currency", "unit"}
    missing = required.difference(curve.columns)
    if missing:
        raise ValueError(f"Curve is missing required columns: {sorted(missing)}")
    if curve.empty:
        raise ValueError("Curve contains no rows.")
    out = curve.copy()
    parsed = pd.to_datetime(out["delivery_month"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("Curve delivery_month values must be valid dates.")
    out["delivery_month"] = parsed.dt.to_period("M").dt.to_timestamp().dt.date
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["product"] = out["product"].astype(str).str.strip()
    out["currency"] = out["currency"].astype(str).str.strip().str.upper()
    out["unit"] = out["unit"].astype(str).str.strip()
    if out["price"].isna().any() or not out["price"].map(math.isfinite).all():
        raise ValueError("Curve prices must be finite and non-missing.")
    unknown = sorted(set(out["product"]) - set(CURVE_PRODUCTS))
    if unknown:
        raise ValueError(f"Unsupported curve product(s): {unknown}")
    if out.duplicated(["product", "delivery_month"]).any():
        raise ValueError("Curve contains duplicate product/delivery-month marks.")
    for product, group in out.groupby("product"):
        expected = CURVE_PRODUCTS[product]
        if not group["currency"].eq(expected["currency"]).all():
            raise ValueError(f"Currency mismatch for {product}: expected {expected['currency']}.")
        if not group["unit"].eq(expected["unit"]).all():
            raise ValueError(f"Price unit mismatch for {product}: expected {expected['unit']}.")
    return out.sort_values(["product", "delivery_month"]).reset_index(drop=True)


def _curve_lookup(validated_curve: pd.DataFrame) -> dict:
    return dict(zip(validated_curve["delivery_month"], validated_curve["price"].astype(float)))


def forward_month_price(curve: pd.DataFrame, product: str, delivery_start: date,
                        delivery_end_exclusive: date) -> float:
    """Return the single monthly mark for one full calendar-month forward."""
    if product not in CURVE_PRODUCTS:
        raise ValueError(f"Unsupported product: {product}")
    start_day, end_day = _to_date(delivery_start), _to_date(delivery_end_exclusive)
    if start_day.day != 1 or end_day != _add_months(start_day, 1):
        raise ValueError("FORWARD valuation requires exactly one full calendar-month delivery period.")
    validated = validate_curve(curve)
    if not validated["product"].eq(product).all():
        raise ValueError(f"Curve product mismatch for {product}.")
    lookup = _curve_lookup(validated)
    if start_day not in lookup:
        raise ValueError(f"Missing or ambiguous forward mark for {start_day:%Y-%m}.")
    return float(lookup[start_day])


def swap_fixed_price(curve: pd.DataFrame, delivery_start: date, delivery_end_exclusive: date,
                     profile: str = "BASE") -> float:
    """Weighted fixed price from monthly marks; weighting by delivery hours in each month.

    The curve must contain one mark for every delivery month touched by the contract.
    """
    c = validate_curve(curve)
    start_day, end_day = _to_date(delivery_start), _to_date(delivery_end_exclusive)
    if end_day <= start_day:
        raise ValueError("Delivery end must be after delivery start.")
    products = c["product"].unique().tolist()
    if len(products) != 1:
        raise ValueError("swap_fixed_price expects a curve for exactly one product.")
    timezone_name = CURVE_PRODUCTS[products[0]]["timezone"]
    return float(_strip_price(_curve_lookup(c), start_day, end_day, str(profile).strip().upper(), timezone_name))


def normalise_trades(trades: pd.DataFrame, allow_missing_quantity_mwh: bool = False) -> pd.DataFrame:
    """Validate and enrich trade rows. End date is exclusive; side is signed explicitly."""
    required = {"trade_id", "product", "side", "delivery_start", "delivery_end", "profile", "quantity_mw", "trade_price", "currency", "status"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"Trades are missing required columns: {sorted(missing)}")
    out = trades.copy()
    # Optional descriptive fields are defaulted so imported blotters can remain compact.
    optional_defaults = {
        "trade_date": date.today().isoformat(),
        "counterparty": "Unspecified counterparty",
        "instrument": "SWAP",
        "settlement_type": "FINANCIAL",
        "notes": "",
    }
    for col, default in optional_defaults.items():
        if col not in out.columns:
            out[col] = default
    for col in ("product", "instrument", "settlement_type", "side", "status", "currency", "profile"):
        cleaned = out[col].astype(str).str.strip()
        out[col] = cleaned if col == "product" else cleaned.str.upper()
    if out["trade_id"].isna().any():
        raise ValueError("Trade IDs must be non-empty and unique.")
    out["trade_id"] = out["trade_id"].astype(str).str.strip()
    invalid_ids = out["trade_id"].str.lower().isin(["", "nan", "none", "null", "nat"])
    if invalid_ids.any() or out["trade_id"].duplicated().any():
        raise ValueError("Trade IDs must be non-empty and unique.")
    if not out["product"].isin(PRODUCTS).all():
        bad = out.loc[~out["product"].isin(PRODUCTS), "product"].unique().tolist()
        raise ValueError(f"Unsupported product(s): {bad}")
    expected_currency = out["product"].map(lambda name: PRODUCTS[name]["currency"])
    wrong_currency = out["currency"].ne(expected_currency)
    if wrong_currency.any():
        row = out.loc[wrong_currency].iloc[0]
        raise ValueError(f"Currency mismatch for {row['product']}: expected {PRODUCTS[row['product']]['currency']}, got {row['currency']}.")
    if not out["instrument"].isin(["FORWARD", "SWAP"]).all():
        raise ValueError("instrument must be FORWARD or SWAP.")
    if not out["settlement_type"].isin(["PHYSICAL", "FINANCIAL"]).all():
        raise ValueError("settlement_type must be PHYSICAL or FINANCIAL.")
    trade_dates = pd.to_datetime(out["trade_date"], errors="coerce")
    if trade_dates.isna().any():
        raise ValueError("trade_date must contain valid dates.")
    out["trade_date"] = trade_dates.dt.date.astype(str)
    out["delivery_start"] = pd.to_datetime(out["delivery_start"], errors="raise").dt.date
    out["delivery_end"] = pd.to_datetime(out["delivery_end"], errors="raise").dt.date
    if any(e <= s for s, e in zip(out["delivery_start"], out["delivery_end"])):
        raise ValueError("Every delivery_end must be after delivery_start (exclusive end date).")
    for trade_id, instrument, s, e in zip(out["trade_id"], out["instrument"], out["delivery_start"], out["delivery_end"]):
        if instrument == "FORWARD" and (s.day != 1 or e != _add_months(s, 1)):
            raise ValueError(
                f"FORWARD trade {trade_id} must cover exactly one full calendar month; "
                "use SWAP for multi-month delivery strips."
            )
    out["quantity_mw"] = pd.to_numeric(out["quantity_mw"], errors="raise")
    out["trade_price"] = pd.to_numeric(out["trade_price"], errors="raise")
    if not out["quantity_mw"].map(math.isfinite).all() or (out["quantity_mw"] <= 0).any():
        raise ValueError("quantity_mw must be positive and finite; direction belongs in side.")
    if not out["trade_price"].map(math.isfinite).all():
        raise ValueError("Trade prices must be finite.")
    if not out["side"].isin(["BUY_FIXED", "SELL_FIXED"]).all():
        raise ValueError("side must be BUY_FIXED or SELL_FIXED.")
    if not out["profile"].isin(["BASE", "PEAK"]).all():
        raise ValueError("profile must be BASE or PEAK.")
    commodity = out["product"].map(lambda name: PRODUCTS[name]["commodity"])
    if (commodity.eq("Gas") & out["profile"].eq("PEAK")).any():
        raise ValueError("Gas products must use the BASE delivery profile; PEAK is only valid for power products.")
    # A product's curve only prices its own delivery shape, so the two must always agree.
    expected_profile = out["product"].map(lambda name: PRODUCTS[name]["profile"])
    wrong_profile = out["profile"].ne(expected_profile)
    if wrong_profile.any():
        row = out.loc[wrong_profile].iloc[0]
        raise ValueError(f"{row['product']} must use the {PRODUCTS[row['product']]['profile']} profile, got {row['profile']}. Choose the matching base or peak product.")
    if not out["status"].isin(["PROPOSED", "CONFIRMED", "EXECUTED", "CANCELLED", "EXPIRED"]).all():
        raise ValueError("Unsupported trade status.")
    out["quantity_mwh"] = [
        mw_to_mwh(float(mw), delivery_hours(s, e, p, PRODUCTS[product]["timezone"]))
        for mw, s, e, p, product in zip(
            out["quantity_mw"], out["delivery_start"], out["delivery_end"], out["profile"], out["product"]
        )
    ]
    out["signed_quantity_mwh"] = [q if side == "BUY_FIXED" else -q for q, side in zip(out["quantity_mwh"], out["side"])]
    return out


class _CurveBook:
    """Validates each product curve once per marking run and serves month lookups."""

    def __init__(self, curves: dict[str, pd.DataFrame]):
        self.curves = curves
        self._cache: dict = {}

    def lookup(self, product: str) -> dict:
        if product not in self._cache:
            try:
                self._cache[product] = self._build(product)
            except Exception as exc:
                self._cache[product] = exc
        cached = self._cache[product]
        if isinstance(cached, Exception):
            raise cached
        return cached

    def _build(self, product: str) -> dict:
        curve = self.curves.get(product)
        if curve is None:
            raise ValueError(f"No curve supplied for {product}.")
        validated = validate_curve(curve)
        spec = CURVE_PRODUCTS[product]
        if not validated["product"].eq(product).all():
            raise ValueError(f"Curve product mismatch for {product}.")
        if not validated["currency"].eq(spec["currency"]).all() or not validated["unit"].eq(spec["unit"]).all():
            raise ValueError(f"Curve currency/unit mismatch for {product}: expected {spec['currency']} and {spec['unit']}.")
        return _curve_lookup(validated)


def _mark_record(trade: dict, book: _CurveBook) -> dict:
    """Mark one already-normalised trade against the prepared curve book."""
    product = trade["product"]
    lookup = book.lookup(product)
    settlement = str(trade["settlement_type"])
    if trade["instrument"] == "FORWARD":
        # A monthly forward maps directly to its single monthly delivery tenor.
        if trade["delivery_start"] not in lookup:
            raise ValueError(f"Missing or ambiguous forward mark for {trade['delivery_start']:%Y-%m}.")
        market_price = float(lookup[trade["delivery_start"]])
        valuation_method = f"Single-month {settlement.lower()} forward proxy"
    else:
        # A swap strip uses each monthly forward weighted by delivery hours.
        market_price = _strip_price(lookup, trade["delivery_start"], trade["delivery_end"], trade["profile"], PRODUCTS[product]["timezone"])
        valuation_method = f"{settlement.title()} delivery-hour-weighted swap-strip proxy"
    mtm = mtm_linear(float(trade["signed_quantity_mwh"]), float(trade["trade_price"]), market_price)
    return {**trade, "market_price": market_price, "mtm": mtm, "mtm_unit": str(trade["currency"]),
            "valuation_method": valuation_method, "mark_status": "MARKED", "mark_error": ""}


def mark_trades(trades: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mark each trade against a weighted market forward for its delivery window.

    Requires a monthly curve mark for every delivery month. Marks must share the
    trade's price unit/currency; this function intentionally does not silently FX-convert.
    """
    book = _CurveBook(curves)
    return pd.DataFrame([_mark_record(trade, book) for trade in normalise_trades(trades).to_dict("records")])


def _normalised_records(records: list[dict]) -> list:
    """Normalise rows together when possible; otherwise isolate failures row by row."""
    if not records:
        return []
    try:
        return normalise_trades(pd.DataFrame(records)).to_dict("records")
    except Exception:
        results = []
        for source in records:
            try:
                results.append(normalise_trades(pd.DataFrame([source])).to_dict("records")[0])
            except Exception as exc:
                results.append(exc)
        return results


def mark_trades_resilient(trades: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mark trades independently so one missing curve tenor does not hide the entire book.

    Invalid/unavailable trades remain visible with mark_status='UNMARKED' and a reason.
    Cancelled and expired trades remain visible in the blotter but are excluded from MtM.
    """
    if trades.empty:
        return pd.DataFrame()
    records = trades.to_dict("records")
    terminal = [str(r.get("status", "")).upper().strip() in {"CANCELLED", "EXPIRED"} for r in records]
    active = iter(_normalised_records([r for r, done in zip(records, terminal) if not done]))
    book = _CurveBook(curves)
    rows = []
    for source, done in zip(records, terminal):
        if done:
            lifecycle = str(source.get("status", "")).upper().strip()
            source = dict(source)
            source.update({"market_price": float("nan"), "mtm": float("nan"),
                           "valuation_method": "Not valued", "mark_status": lifecycle,
                           "mark_error": f"Excluded from valuation: trade status is {lifecycle}."})
            rows.append(source)
            continue
        normalised = next(active)
        try:
            if isinstance(normalised, Exception):
                raise normalised
            rows.append(_mark_record(normalised, book))
        except Exception as exc:
            failed = dict(source)
            failed.setdefault("quantity_mwh", float("nan"))
            failed.setdefault("signed_quantity_mwh", float("nan"))
            failed.update({"market_price": float("nan"), "mtm": float("nan"),
                           "valuation_method": "Unavailable", "mark_status": "UNMARKED",
                           "mark_error": f"{type(exc).__name__}: {exc}"})
            rows.append(failed)
    return pd.DataFrame(rows)


def trade_delivery_breakdown(trades: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Split executed positions and simplified MtM into monthly delivery buckets.

    Net and gross MWh are calculated even when a curve mark is missing. MtM is left
    unavailable for the affected bucket rather than silently assuming a price.
    """
    columns = ["trade_id", "product", "delivery_month", "profile", "currency", "net_mwh",
               "gross_mwh", "market_price", "trade_price", "mtm", "mark_status", "mark_error"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    executed = trades.loc[trades["status"].astype(str).str.upper().eq("EXECUTED")]
    records = executed.to_dict("records")
    book = _CurveBook(curves)
    rows = []
    for raw, trade in zip(records, _normalised_records(records)):
        try:
            if isinstance(trade, Exception):
                raise trade
            product = trade["product"]
            try:
                lookup, curve_error = book.lookup(product), ""
            except ValueError as exc:
                if str(exc).startswith("No curve supplied"):
                    lookup, curve_error = {}, "No curve mark for this delivery bucket."
                else:
                    raise
            sign = 1.0 if trade["side"] == "BUY_FIXED" else -1.0
            segments = _month_segments(trade["delivery_start"], trade["delivery_end"], trade["profile"], PRODUCTS[product]["timezone"])
            for month, hours in segments:
                signed_mwh = float(trade["quantity_mw"]) * hours * sign
                if month in lookup:
                    market_price = float(lookup[month])
                    mtm_value = mtm_linear(signed_mwh, float(trade["trade_price"]), market_price)
                    status, error = "MARKED", ""
                else:
                    market_price, mtm_value = float("nan"), float("nan")
                    status, error = "UNMARKED", curve_error or f"Missing curve mark for {month:%Y-%m}."
                rows.append({"trade_id": trade["trade_id"], "product": product,
                             "delivery_month": month, "profile": trade["profile"],
                             "currency": trade["currency"], "net_mwh": signed_mwh,
                             "gross_mwh": abs(signed_mwh), "market_price": market_price,
                             "trade_price": float(trade["trade_price"]), "mtm": mtm_value,
                             "mark_status": status, "mark_error": error})
        except Exception as exc:
            # Keep a diagnostic row instead of dropping a contract silently.
            rows.append({"trade_id": raw.get("trade_id", "UNKNOWN"), "product": raw.get("product", "UNKNOWN"),
                         "delivery_month": raw.get("delivery_start", None), "profile": raw.get("profile", ""),
                         "currency": raw.get("currency", ""), "net_mwh": float("nan"),
                         "gross_mwh": float("nan"), "market_price": float("nan"),
                         "trade_price": raw.get("trade_price", float("nan")), "mtm": float("nan"),
                         "mark_status": "UNMARKED", "mark_error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows, columns=columns)


# =============================================================================
# SECTION 2 · ANALYTICS ENGINE
# Tenors, curve building, P&L explain, risk, spreads, spot power, seasonal gas.
# =============================================================================
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
CUSTOM_CONTRACT = "Custom dates"
SYNTHETIC_HISTORY_SOURCE = "Synthetic demo history"

COAL_MWH_PER_TONNE = 6.978        # API2 basis: 6,000 kcal/kg
DEFAULT_SPREAD_PARAMS = {"gas_efficiency": 0.50, "coal_efficiency": 0.38,
                         "gas_emission_factor": 0.202, "coal_emission_factor": 0.341}


def month_contract_label(month: date) -> str:
    return f"{MONTH_LABELS[month.month - 1]}-{month.year % 100:02d}"


def contract_window(label: str) -> tuple[date, date]:
    """Delivery window [start, end) for Dec-26, Q1-27, Sum-27, Win-27 or Cal-27."""
    text = str(label).strip()
    try:
        name, year_text = text.rsplit("-", 1)
        year = 2000 + int(year_text)
        key = name.strip().casefold()
        if key == "cal":
            return date(year, 1, 1), date(year + 1, 1, 1)
        if key == "sum":
            return date(year, 4, 1), date(year, 10, 1)
        if key == "win":
            return date(year, 10, 1), date(year + 1, 4, 1)
        if key in {"q1", "q2", "q3", "q4"}:
            start = date(year, 3 * (int(key[1]) - 1) + 1, 1)
            return start, _add_months(start, 3)
        months = [m.casefold() for m in MONTH_LABELS]
        if key in months:
            start = date(year, months.index(key) + 1, 1)
            return start, _add_months(start, 1)
    except ValueError:
        pass
    raise ValueError(f"Unknown contract '{label}'. Use a form like Dec-26, Q1-27, Sum-27, Win-27 or Cal-27.")


def contract_choices(valuation_date: date) -> list[str]:
    """Standard forward contracts that start after the current delivery month."""
    first = _add_months(valuation_date, 1)
    labels = [month_contract_label(_add_months(first, i)) for i in range(12)]
    year = first.year
    candidates = []
    for offset in range(4):
        y = (year + offset) % 100
        candidates += [f"Q{q}-{y:02d}" for q in (1, 2, 3, 4)]
    labels += [c for c in candidates if contract_window(c)[0] >= first][:6]
    seasons = []
    for offset in range(4):
        y = (year + offset) % 100
        seasons += [f"Sum-{y:02d}", f"Win-{y:02d}"]
    labels += [c for c in seasons if contract_window(c)[0] >= first][:4]
    labels += [c for c in (f"Cal-{(year + i) % 100:02d}" for i in range(4)) if contract_window(c)[0] >= first][:2]
    return labels


def decompose_tenor_quotes(quotes: Sequence[tuple[str, float]], product: str,
                           prior_curve: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Turn Month/Quarter/Season/Cal quotes into arbitrage-free monthly marks.

    The result stays as close as possible to the prior monthly shape while every quoted
    contract reprices exactly (delivery-hour weighted). Contradictory quotes are rejected.
    """
    if product not in CURVE_PRODUCTS:
        raise ValueError(f"Unsupported product: {product}")
    if not quotes:
        raise ValueError("Enter at least one contract and price.")
    spec = CURVE_PRODUCTS[product]
    profile = spec.get("profile", "BASE")
    windows = []
    for label, price in quotes:
        value = float(price)
        if not math.isfinite(value):
            raise ValueError(f"Price for {label} must be a finite number.")
        windows.append((str(label), value, _month_segments(*contract_window(label), profile, spec["timezone"])))
    months = sorted({month for _, _, segments in windows for month, _ in segments})
    index = {month: i for i, month in enumerate(months)}
    weights = np.zeros((len(windows), len(months)))
    targets = np.zeros(len(windows))
    for row, (_, value, segments) in enumerate(windows):
        total = sum(hours for _, hours in segments)
        for month, hours in segments:
            weights[row, index[month]] = hours / total
        targets[row] = value
    prior_lookup = _curve_lookup(validate_curve(prior_curve)) if prior_curve is not None else {}
    prior = np.zeros(len(months))
    for month, i in index.items():
        if month in prior_lookup:
            prior[i] = prior_lookup[month]
        else:
            # Flat inside the shortest contract covering this month.
            covering = [(len(segments), value) for _, value, segments in windows if month in dict(segments)]
            prior[i] = min(covering)[1]
    adjusted = prior + weights.T @ np.linalg.pinv(weights @ weights.T) @ (targets - weights @ prior)
    residuals = np.abs(weights @ adjusted - targets)
    if residuals.max() > 1e-6:
        worst = windows[int(residuals.argmax())][0]
        raise ValueError(f"These quotes contradict each other (see {worst}, off by {residuals.max():.3f}). "
                         "A quarter, season or year must equal the hour-weighted average of the contracts inside it.")
    return validate_curve(pd.DataFrame({
        "delivery_month": months, "product": product, "price": np.round(adjusted, 4),
        "currency": spec["currency"], "unit": spec["unit"], "data_status": "Built from contract quotes",
    }))


def synthetic_curve_history(valuation_date: date, products: Sequence[str], business_days: int = 90,
                            months: int = 36, seed: int = 20261104) -> pd.DataFrame:
    """Create a clearly labelled synthetic curve history so P&L and VaR can be explored.

    Prices follow a seeded random walk that ends exactly on today's demo curve. This is
    demonstration data, never a market record.
    """
    rng = np.random.default_rng(seed)
    dates = [d.date() for d in pd.bdate_range(end=valuation_date - timedelta(days=1), periods=int(business_days))]
    common = rng.normal(size=len(dates))
    gas_shared = rng.normal(size=len(dates))
    power_shared = rng.normal(size=len(dates))
    volatility = {"Gas": 0.022, "Power": 0.017, "Carbon": 0.018, "Coal": 0.012}
    rows = []
    for product in products:
        spec = CURVE_PRODUCTS[product]
        commodity = spec["commodity"]
        own = rng.normal(size=len(dates))
        shared = gas_shared if commodity == "Gas" else power_shared if commodity == "Power" else own
        shock = (0.45 * common + 0.75 * shared + 0.48 * own) * volatility.get(commodity, 0.015)
        # Walk backwards from today so the most recent history connects to today's curve.
        levels = np.cumsum(shock[::-1])[::-1]
        for day, level in zip(dates, levels):
            for month, price in _sample_curve_rows(product, day, months):
                tenor = (month.year - day.year) * 12 + month.month - day.month
                damping = 0.25 + 0.75 * math.exp(-tenor / 18.0)  # front of the curve moves most
                rows.append({"snapshot_date": day.isoformat(), "product": product,
                             "delivery_month": month.isoformat(),
                             "price": round(price * math.exp(-level * damping), 4),
                             "source": SYNTHETIC_HISTORY_SOURCE})
    return pd.DataFrame(rows)


def snapshot_lookup(history: pd.DataFrame) -> dict:
    """{date: {product: {delivery month: price}}} from stored curve snapshots."""
    result: dict = {}
    if history is None or history.empty:
        return result
    parsed: dict = {}

    def as_date(value) -> date:
        held = parsed.get(value)
        if held is None:
            held = parsed[value] = _to_date(value)
        return held

    for day, product, month, price in zip(history["snapshot_date"].tolist(), history["product"].tolist(),
                                          history["delivery_month"].tolist(), history["price"].tolist()):
        result.setdefault(as_date(day), {}).setdefault(product, {})[as_date(month)] = float(price)
    # A product with no marks on a day keeps its last ones: no new price isn't the same as leaving the curve.
    latest: dict = {}
    for day in sorted(result):
        for product, marks in latest.items():
            result[day].setdefault(product, marks)
        latest.update(result[day])
    return result


def book_buckets(trades: pd.DataFrame) -> pd.DataFrame:
    """Executed trades as signed monthly delivery buckets, the unit all risk runs on."""
    columns = ["trade_id", "trade_date", "product", "currency", "profile", "delivery_month", "net_mwh", "trade_price"]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=columns)
    executed = trades.loc[trades["status"].astype(str).str.upper().eq("EXECUTED")]
    rows = []
    for trade in _normalised_records(executed.to_dict("records")):
        if isinstance(trade, Exception):
            continue
        sign = 1.0 if trade["side"] == "BUY_FIXED" else -1.0
        for month, hours in _month_segments(trade["delivery_start"], trade["delivery_end"], trade["profile"], PRODUCTS[trade["product"]]["timezone"]):
            rows.append({"trade_id": trade["trade_id"], "trade_date": _to_date(trade["trade_date"]),
                         "product": trade["product"], "currency": trade["currency"], "profile": trade["profile"],
                         "delivery_month": month, "net_mwh": float(trade["quantity_mw"]) * hours * sign,
                         "trade_price": float(trade["trade_price"])})
    return pd.DataFrame(rows, columns=columns)


def to_eur(amount: float, currency: str, eur_per_gbp: float, eur_per_usd: float = 1.0) -> float:
    if currency == "EUR":
        return float(amount)
    if currency == "GBP":
        return float(amount) * float(eur_per_gbp)
    if currency == "USD":
        return float(amount) * float(eur_per_usd)
    raise ValueError(f"No FX rate available for {currency}.")


def pnl_explain(buckets: pd.DataFrame, previous: dict, current: dict,
                previous_date: date, current_date: date) -> pd.DataFrame:
    """Attribute one day's P&L per delivery bucket.

    new trade   = traded since the previous mark: volume × (today's mark − trade price)
    price move  = existing bucket: volume × (today's mark − previous mark)
    first mark  = existing bucket with no previous mark: volume × (today's mark − trade price)
    to delivery = existing bucket that left the curve: its last MtM becomes realised, so it
                  reduces open MtM without changing P&L.
    """
    columns = ["trade_id", "product", "currency", "delivery_month", "net_mwh", "previous_mark", "current_mark",
               "new_trade", "price_move", "first_mark", "to_delivery", "previous_mtm", "current_mtm", "total_pnl", "note"]
    rows = []
    for b in buckets.to_dict("records") if buckets is not None else []:
        if b["trade_date"] > current_date:
            continue
        prev_mark = previous.get(b["product"], {}).get(b["delivery_month"])
        curr_mark = current.get(b["product"], {}).get(b["delivery_month"])
        is_new = b["trade_date"] > previous_date
        volume, fixed = float(b["net_mwh"]), float(b["trade_price"])
        new_trade = price_move = first_mark = to_delivery = previous_mtm = current_mtm = 0.0
        note = ""
        if curr_mark is not None:
            current_mtm = volume * (curr_mark - fixed)
        if not is_new and prev_mark is not None:
            previous_mtm = volume * (prev_mark - fixed)
        if is_new:
            if curr_mark is not None:
                new_trade = current_mtm
            else:
                note = "New trade has no mark today"
        elif prev_mark is not None and curr_mark is not None:
            price_move = volume * (curr_mark - prev_mark)
        elif curr_mark is not None:
            first_mark = current_mtm
            note = "First day with a mark"
        elif prev_mark is not None:
            to_delivery = previous_mtm
            note = "Left the curve, MtM treated as realised"
        else:
            note = "No mark on either day"
        rows.append({"trade_id": b["trade_id"], "product": b["product"], "currency": b["currency"],
                     "delivery_month": b["delivery_month"], "net_mwh": volume,
                     "previous_mark": prev_mark if prev_mark is not None else float("nan"),
                     "current_mark": curr_mark if curr_mark is not None else float("nan"),
                     "new_trade": new_trade, "price_move": price_move, "first_mark": first_mark,
                     "to_delivery": to_delivery, "previous_mtm": previous_mtm, "current_mtm": current_mtm,
                     "total_pnl": daily_pnl(previous_mtm, current_mtm, to_delivery), "note": note})
    return pd.DataFrame(rows, columns=columns)


def pnl_history(buckets: pd.DataFrame, lookups: dict, eur_per_gbp: float) -> pd.DataFrame:
    """Daily P&L and open MtM in EUR for every consecutive pair of saved curve days.

    Same rules as pnl_explain, calculated for all days at once.
    """
    columns = ["date", "pnl_eur", "mtm_eur", "cumulative_pnl_eur"]
    days = sorted(lookups)
    if len(days) < 2:
        return pd.DataFrame(columns=columns)
    if buckets is None or buckets.empty:
        steps = days[1:]
        return pd.DataFrame({"date": steps, "pnl_eur": 0.0, "mtm_eur": 0.0, "cumulative_pnl_eur": 0.0})[columns]
    volume = buckets["net_mwh"].to_numpy(dtype=float)
    fixed = buckets["trade_price"].to_numpy(dtype=float)
    factor = buckets["currency"].map({"EUR": 1.0, "GBP": float(eur_per_gbp)}).to_numpy(dtype=float)
    products, months = buckets["product"].tolist(), buckets["delivery_month"].tolist()
    marks = np.full((len(volume), len(days)), np.nan)
    for column, day in enumerate(days):
        snapshot = lookups[day]
        marks[:, column] = [snapshot.get(p, {}).get(m, np.nan) for p, m in zip(products, months)]
    traded = np.array([d.toordinal() for d in buckets["trade_date"]])
    ordinals = np.array([d.toordinal() for d in days])
    open_value = np.nan_to_num((marks - fixed[:, None]) * volume[:, None] * factor[:, None])
    pnl, mtm = [], []
    for step in range(1, len(days)):
        alive = traded <= ordinals[step]
        existing = traded <= ordinals[step - 1]
        now = np.where(alive, open_value[:, step], 0.0)
        before = np.where(existing, open_value[:, step - 1], 0.0)
        left_curve = existing & np.isnan(marks[:, step]) & ~np.isnan(marks[:, step - 1])
        # A month leaving the curve moves from open MtM to realised, so its value stays in P&L.
        pnl.append(float((now - before + np.where(left_curve, before, 0.0)).sum()))
        mtm.append(float(now.sum()))
    out = pd.DataFrame({"date": days[1:], "pnl_eur": pnl, "mtm_eur": mtm})
    out["cumulative_pnl_eur"] = out["pnl_eur"].cumsum()
    return out[columns]


SCENARIOS = {
    "Parallel move": "Every delivery month of every product moves by the shock.",
    "Curve steepens": "Front months fall by the shock, month 24 and beyond rise by it.",
    "Front-month squeeze": "The first month moves by the full shock, fading to nothing by month 6.",
    "Winter premium": "October to March rises by the shock, April to September falls by half of it.",
    "Gas rally, power follows": "Gas moves by the shock and power by 1.5 times the shock.",
    "TTF over NBP": "TTF moves by the shock while NBP and everything else stays put.",
    "France over Germany": "French power moves by the shock while German power and gas stay put.",
    "Peak over base": "Peak products move by the shock while base products and gas stay put.",
}
POWER_PASS_THROUGH = 1.5


def scenario_shock(name: str, size: float, product: str, delivery_month: date, first_month: date) -> float:
    """Price shock in quote currency per MWh for one product and delivery month."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    tenor = (delivery_month.year - first_month.year) * 12 + delivery_month.month - first_month.month
    commodity = PRODUCTS[product]["commodity"]
    if name == "Parallel move":
        return size
    if name == "Curve steepens":
        return size * (2.0 * min(max(tenor, 0), 23) / 23.0 - 1.0)
    if name == "Front-month squeeze":
        return size * max(0.0, 1.0 - max(tenor, 0) / 6.0)
    if name == "Winter premium":
        return size if delivery_month.month in (10, 11, 12, 1, 2, 3) else -size / 2.0
    if name == "Gas rally, power follows":
        return size if commodity == "Gas" else size * POWER_PASS_THROUGH
    if name == "TTF over NBP":
        return size if product.startswith("TTF") else 0.0
    if name == "France over Germany":
        return size if product.startswith("French") else 0.0
    return size if "Peak" in product else 0.0


def run_scenarios(buckets: pd.DataFrame, size: float, first_month: date, eur_per_gbp: float) -> pd.DataFrame:
    """P&L of every named scenario, by currency and consolidated in EUR."""
    rows = []
    for name in SCENARIOS:
        pnl = {"EUR": 0.0, "GBP": 0.0}
        for b in buckets.to_dict("records"):
            pnl[b["currency"]] += parallel_shock_pnl(float(b["net_mwh"]), scenario_shock(name, size, b["product"], b["delivery_month"], first_month))
        rows.append({"scenario": name, "pnl_eur_book": pnl["EUR"], "pnl_gbp_book": pnl["GBP"],
                     "pnl_total_eur": pnl["EUR"] + pnl["GBP"] * float(eur_per_gbp), "what_it_does": SCENARIOS[name]})
    return pd.DataFrame(rows)


def historical_var(buckets: pd.DataFrame, history: pd.DataFrame, eur_per_gbp: float,
                   confidence_levels: Sequence[float] = (0.95, 0.99), prepared: Optional[tuple] = None) -> dict:
    """One-day historical-simulation VaR and expected shortfall of today's positions.

    Each saved day's move in a contract is mapped to its distance from delivery, then
    applied to the bucket at the same distance today.
    """
    empty = {"pnl": pd.DataFrame(columns=["date", "pnl_eur"]), "var": {}, "es": {}, "uncovered": [], "observations": 0}
    if buckets is None or buckets.empty or history is None or history.empty:
        return empty
    moves, days = prepared or _contract_moves(history)
    if not days:
        return empty
    pnl, uncovered, _ = _book_series(buckets, moves, days, date.today(), eur_per_gbp)
    var, es = {}, {}
    for level in confidence_levels:
        var[float(level)], es[float(level)] = _var_es(pnl, float(level))
    return {"pnl": pd.DataFrame({"date": [d.date() for d in days], "pnl_eur": pnl}), "var": var, "es": es,
            "uncovered": uncovered, "observations": len(days)}


def spreads_table(curves: dict[str, pd.DataFrame], power_product: str, gas_product: str,
                  params: Optional[dict] = None, eur_per_gbp: float = 1.17, eur_per_usd: float = 0.92) -> pd.DataFrame:
    """Clean spark and clean dark spreads by delivery month, all in EUR per MWh of power.

    clean spark = power − gas / gas efficiency − EUA × gas emission factor / gas efficiency
    clean dark  = power − coal per MWh(th) / coal efficiency − EUA × coal factor / coal efficiency
    """
    p = {**DEFAULT_SPREAD_PARAMS, **(params or {})}
    if not (0 < p["gas_efficiency"] <= 1 and 0 < p["coal_efficiency"] <= 1):
        raise ValueError("Plant efficiencies must be between 0 and 1.")
    required = [power_product, gas_product, "EUA Carbon", "API2 Coal"]
    lookups = {name: _curve_lookup(validate_curve(curves[name])) for name in required}
    months = sorted(set.intersection(*(set(lookups[name]) for name in required)))
    gas_fx = to_eur(1.0, CURVE_PRODUCTS[gas_product]["currency"], eur_per_gbp, eur_per_usd)
    coal_fx = to_eur(1.0, CURVE_PRODUCTS["API2 Coal"]["currency"], eur_per_gbp, eur_per_usd)
    rows = []
    for month in months:
        power = lookups[power_product][month]
        gas = lookups[gas_product][month] * gas_fx
        carbon = lookups["EUA Carbon"][month]
        coal_per_mwh = lookups["API2 Coal"][month] * coal_fx / COAL_MWH_PER_TONNE
        carbon_gas = carbon * p["gas_emission_factor"] / p["gas_efficiency"]
        carbon_coal = carbon * p["coal_emission_factor"] / p["coal_efficiency"]
        spark = power - gas / p["gas_efficiency"] - carbon_gas
        dark = power - coal_per_mwh / p["coal_efficiency"] - carbon_coal
        # Gas price at which gas and coal plants earn the same clean margin.
        switching = p["gas_efficiency"] * (coal_per_mwh / p["coal_efficiency"] + carbon_coal - carbon_gas)
        rows.append({"delivery_month": month, "contract": month_contract_label(month), "power_eur_mwh": power,
                     "gas_eur_mwh": gas, "eua_eur_t": carbon, "coal_eur_mwh_th": coal_per_mwh,
                     "clean_spark": spark, "clean_dark": dark, "gas_minus_coal": spark - dark,
                     "switching_gas_price": switching})
    return pd.DataFrame(rows)


def curve_spread(curves: dict[str, pd.DataFrame], long_product: str, short_product: str,
                 eur_per_gbp: float = 1.17) -> pd.DataFrame:
    """Monthly difference between two curves, converted to EUR/MWh."""
    a = _curve_lookup(validate_curve(curves[long_product]))
    b = _curve_lookup(validate_curve(curves[short_product]))
    fa = to_eur(1.0, CURVE_PRODUCTS[long_product]["currency"], eur_per_gbp)
    fb = to_eur(1.0, CURVE_PRODUCTS[short_product]["currency"], eur_per_gbp)
    months = sorted(set(a) & set(b))
    return pd.DataFrame({"delivery_month": months, "contract": [month_contract_label(m) for m in months],
                         "spread_eur_mwh": [a[m] * fa - b[m] * fb for m in months]})


def spot_series(frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Clean a day-ahead price frame and add local time and each price's duration in hours."""
    columns = ["timestamp", "local", "value", "hours"]
    if frame is None or frame.empty or not {"timestamp", "value"}.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    # A slice of a series this function already cleaned needs no second pass.
    if frame.attrs.get("spot_clean") == timezone_name and {"local", "hours"}.issubset(frame.columns):
        return frame[columns]
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out["value"] = _number_series(out["value"])
    out = out.dropna(subset=["timestamp", "value"])
    if out.empty:
        return pd.DataFrame(columns=columns)
    if "resolution" in out.columns:
        durations = {r: pd.to_timedelta(r).total_seconds() / 3600.0 for r in out["resolution"].dropna().unique() if isinstance(r, str) and r}
        out["hours"] = out["resolution"].map(durations).astype(float)
    else:
        out["hours"] = float("nan")
    # Where a slot appears twice, keep the finer price. The sort makes this the same choice every time.
    out = out.sort_values(["timestamp", "hours"], kind="mergesort").drop_duplicates("timestamp")
    fallback = out["timestamp"].diff().dt.total_seconds().div(3600).median()
    out["hours"] = out["hours"].fillna(fallback if pd.notna(fallback) and fallback > 0 else 1.0)
    # A day holding both 15- and 60-minute rows would count the same time twice, so keep the finest full set.
    days = out["timestamp"].dt.floor("D")
    finest = out.groupby(days)["hours"].transform("min")
    if out["hours"].ne(finest).any():
        is_finest = out["hours"].eq(finest)
        covered = (out["hours"] * is_finest).groupby(days).transform("sum")
        out = out.loc[is_finest | covered.lt(18.0)]
    out["local"] = out["timestamp"].dt.tz_convert(timezone_name)
    out = out[columns].reset_index(drop=True)
    out.attrs["spot_clean"] = timezone_name
    return out


def spot_kpis(series: pd.DataFrame) -> dict:
    """Time-weighted base and peak averages, range, negative hours and hourly volatility."""
    if series is None or series.empty:
        return {}
    weights = series["hours"]
    base = float((series["value"] * weights).sum() / weights.sum())
    peak_mask = series["local"].dt.weekday.lt(5) & series["local"].dt.hour.between(8, 19)
    peak = float((series.loc[peak_mask, "value"] * weights[peak_mask]).sum() / weights[peak_mask].sum()) if peak_mask.any() else float("nan")
    return {"base": base, "peak": peak, "peak_minus_base": peak - base, "low": float(series["value"].min()),
            "high": float(series["value"].max()), "negative_hours": float(weights[series["value"].lt(0)].sum()),
            "hours": float(weights.sum()), "volatility": float(series["value"].std()) if len(series) > 1 else 0.0}


def spot_hourly(series: pd.DataFrame) -> pd.DataFrame:
    """Average prices into UTC hours so zones and resolutions line up."""
    if series is None or series.empty:
        return pd.DataFrame(columns=["hour", "value"])
    out = series.assign(hour=series["timestamp"].dt.floor("h"))
    return out.groupby("hour", as_index=False)["value"].mean()


def price_duration_curve(series: pd.DataFrame) -> pd.DataFrame:
    """Prices sorted high to low against the share of hours at or above each price."""
    if series is None or series.empty:
        return pd.DataFrame(columns=["share_of_hours", "price"])
    ordered = series.sort_values("value", ascending=False)
    cumulative = ordered["hours"].cumsum() / ordered["hours"].sum() * 100.0
    return pd.DataFrame({"share_of_hours": cumulative.values, "price": ordered["value"].values})


def capture_prices(prices: pd.DataFrame, generation: pd.DataFrame, baseload: float) -> pd.DataFrame:
    """Output-weighted price earned by wind and solar, and its ratio to baseload."""
    columns = ["technology", "capture_price", "capture_rate_pct", "average_output_mw"]
    if prices is None or prices.empty or generation is None or generation.empty:
        return pd.DataFrame(columns=columns)
    hourly_price = spot_hourly(prices)
    gen = generation.copy()
    gen["hour"] = pd.to_datetime(gen["timestamp"], errors="coerce", utc=True).dt.floor("h")
    gen["value"] = _number_series(gen["value"])
    gen = gen.dropna(subset=["hour", "value"])
    labels = gen["series"].astype(str)
    gen = gen.loc[~labels.str.contains("consumption", case=False)]
    rows = []
    for technology, names in (("Solar", ("Solar",)), ("Wind onshore", ("Wind onshore",)),
                              ("Wind offshore", ("Wind offshore",)), ("All wind", ("Wind onshore", "Wind offshore"))):
        subset = gen.loc[gen["series"].isin(names)]
        if subset.empty:
            continue
        hourly = subset.groupby(["hour", "series"], as_index=False)["value"].mean().groupby("hour", as_index=False)["value"].sum()
        joined = hourly.merge(hourly_price, on="hour", suffixes=("_mw", "_price"))
        output = joined["value_mw"].sum()
        if joined.empty or output <= 0:
            continue
        capture = float((joined["value_mw"] * joined["value_price"]).sum() / output)
        rows.append({"technology": technology, "capture_price": capture,
                     "capture_rate_pct": capture / baseload * 100.0 if baseload else float("nan"),
                     "average_output_mw": float(joined["value_mw"].mean())})
    return pd.DataFrame(rows, columns=columns)


def storage_seasonal(frame: pd.DataFrame, today: Optional[date] = None) -> dict:
    """Storage fill by day of year: current and last year against the prior five-year range."""
    empty = {"lines": pd.DataFrame(), "band": pd.DataFrame(), "current_year": None, "band_years": []}
    if frame is None or frame.empty:
        return empty
    date_col = _find_column(frame, ("observation_date", "gasDayStart", "gas_day"))
    fill_col = _find_column(frame, ("full", "fillLevel", "fill_level"))
    if not date_col or not fill_col:
        return empty
    data = pd.DataFrame({"day": pd.to_datetime(frame[date_col], errors="coerce"), "fill": _number_series(frame[fill_col])})
    data = data.dropna().groupby("day", as_index=False)["fill"].mean().sort_values("day")
    if data.empty:
        return empty
    data = data.assign(year=data["day"].dt.year)
    # A common leap year on the x axis lets every year line up by calendar day.
    data["calendar_day"] = pd.to_datetime({"year": 2000, "month": data["day"].dt.month, "day": data["day"].dt.day})
    current_year = int((today or data["day"].max()).year)
    band_years = [y for y in range(current_year - 5, current_year) if y in set(data["year"])]
    prior = data.loc[data["year"].isin(band_years)]
    band = (prior.groupby("calendar_day")["fill"].agg(low="min", high="max", average="mean").reset_index()
            if not prior.empty else pd.DataFrame())
    lines = data.loc[data["year"].isin([current_year, current_year - 1])]
    return {"lines": lines, "band": band, "current_year": current_year, "band_years": band_years}


def temperature_normals_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Calendar-day mean temperature and degree days from a multi-year daily series."""
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["temperature"] = pd.to_numeric(data["temperature"], errors="coerce")
    data = data.dropna(subset=["date", "temperature"])
    data["month_day"] = data["date"].dt.strftime("%m-%d")
    data["hdd"] = (18.0 - data["temperature"]).clip(lower=0)
    data["cdd"] = (data["temperature"] - 22.0).clip(lower=0)
    return (data.groupby("month_day", as_index=False)
            .agg(normal_temperature_c=("temperature", "mean"), normal_hdd=("hdd", "mean"), normal_cdd=("cdd", "mean")))


def hdd_vs_normal(daily: pd.DataFrame, normals: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """Forecast heating demand against normal for the next few days, by location."""
    columns = ["location", "days", "forecast_hdd", "normal_hdd", "hdd_vs_normal", "temperature_vs_normal_c"]
    if daily is None or daily.empty or normals is None or normals.empty or "location" not in daily.columns:
        return pd.DataFrame(columns=columns)
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if "forecast_period" in data.columns:
        data = data.loc[data["forecast_period"].eq("Forecast")]
    data = data.dropna(subset=["date"]).sort_values("date")
    data["month_day"] = data["date"].dt.strftime("%m-%d")
    rows = []
    for location, group in data.groupby("location", sort=False):
        group = group.head(int(horizon_days)).merge(normals.loc[normals["location"].eq(location)], on="month_day", how="inner")
        if group.empty:
            continue
        rows.append({"location": location, "days": int(len(group)),
                     "forecast_hdd": float(group["heating_degree_day_proxy_18c"].sum()),
                     "normal_hdd": float(group["normal_hdd"].sum()),
                     "hdd_vs_normal": float(group["heating_degree_day_proxy_18c"].sum() - group["normal_hdd"].sum()),
                     "temperature_vs_normal_c": float((group["temperature_mean_c"] - group["normal_temperature_c"]).mean())})
    return pd.DataFrame(rows, columns=columns)


# -----------------------------------------------------------------------------
# Hedge Advisor: which standard contracts remove the most risk from today's book
# -----------------------------------------------------------------------------
DEFAULT_HALF_SPREADS = {"Gas": 0.10, "Power": 0.30}     # assumed cost per MWh to cross half the bid/offer
MIN_HEDGE_OBSERVATIONS = 20


def _contract_moves(history: pd.DataFrame) -> tuple[dict, list]:
    """Daily same-contract price moves by product and distance from delivery.

    Only moves in the same contract are used, so a contract rolling to the next month
    never registers as a price move. A day missing a distance borrows the closest one.
    """
    if history is None or history.empty:
        return {}, []
    data = history[["snapshot_date", "product", "delivery_month", "price"]].copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])
    data["delivery_month"] = pd.to_datetime(data["delivery_month"])
    data = data.sort_values(["product", "delivery_month", "snapshot_date"])
    grouped = data.groupby(["product", "delivery_month"])
    data["move"] = grouped["price"].diff()
    data["gap_days"] = grouped["snapshot_date"].diff().dt.days
    data = data.dropna(subset=["move"]).loc[lambda f: f["gap_days"].le(5)]
    if data.empty:
        return {}, []
    data["tenor"] = ((data["delivery_month"].dt.year - data["snapshot_date"].dt.year) * 12
                     + data["delivery_month"].dt.month - data["snapshot_date"].dt.month)
    moves = {}
    for product, frame in data.groupby("product"):
        table = frame.pivot_table(index="snapshot_date", columns="tenor", values="move").sort_index(axis=1)
        moves[product] = table.ffill(axis=1).bfill(axis=1)
    days = sorted({day for table in moves.values() for day in table.index})
    return moves, days


def _position_series(product: str, delivery_month: date, net_mwh: float, currency: str, moves: dict, days: list,
                     valuation_date: date, eur_per_gbp: float) -> Optional[np.ndarray]:
    """Hypothetical daily EUR P&L of one delivery bucket under each saved day's moves."""
    table = moves.get(product)
    if table is None or table.empty:
        return None
    tenor = (delivery_month.year - valuation_date.year) * 12 + delivery_month.month - valuation_date.month
    nearest = min(table.columns, key=lambda t: abs(int(t) - tenor))
    fx = float(eur_per_gbp) if currency == "GBP" else 1.0
    return table[nearest].reindex(days).fillna(0.0).to_numpy() * float(net_mwh) * fx


def _book_series(buckets: pd.DataFrame, moves: dict, days: list, valuation_date: date, eur_per_gbp: float) -> tuple[np.ndarray, list, pd.DataFrame]:
    """Total daily P&L of the book, products with no history, and each bucket's series."""
    total = np.zeros(len(days))
    uncovered, rows = [], []
    position = buckets.groupby(["product", "currency", "delivery_month"], as_index=False)["net_mwh"].sum()
    for row in position.to_dict("records"):
        series = _position_series(row["product"], row["delivery_month"], row["net_mwh"], row["currency"], moves, days, valuation_date, eur_per_gbp)
        if series is None:
            uncovered.append(row["product"])
            continue
        total = total + series
        rows.append({**row, "series": series})
    return total, sorted(set(uncovered)), pd.DataFrame(rows)


def _var_es(pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    threshold = float(np.quantile(pnl, 1.0 - float(confidence)))
    tail = pnl[pnl <= threshold]
    return max(-threshold, 0.0), max(-float(tail.mean()), 0.0) if tail.size else max(-threshold, 0.0)


def risk_drivers(rows: pd.DataFrame, total: np.ndarray, confidence: float) -> pd.DataFrame:
    """How much each delivery bucket lost, on average, on the book's worst days."""
    columns = ["product", "delivery_month", "contract", "net_mwh", "tail_loss_eur", "share_pct"]
    if rows is None or rows.empty:
        return pd.DataFrame(columns=columns)
    threshold = np.quantile(total, 1.0 - float(confidence))
    tail_days = total <= threshold
    book_tail = -float(total[tail_days].mean()) if tail_days.any() else 0.0
    output = []
    for row in rows.to_dict("records"):
        loss = -float(row["series"][tail_days].mean()) if tail_days.any() else 0.0
        output.append({"product": row["product"], "delivery_month": row["delivery_month"], "contract": month_contract_label(row["delivery_month"]),
                       "net_mwh": float(row["net_mwh"]), "tail_loss_eur": loss,
                       "share_pct": loss / book_tail * 100.0 if abs(book_tail) > 1e-9 else 0.0})
    return pd.DataFrame(output, columns=columns).sort_values("tail_loss_eur", ascending=False).reset_index(drop=True)


def _book_net(buckets: pd.DataFrame) -> dict:
    """{product: {delivery month: net MWh}} ignoring months that net to nothing."""
    net: dict = {}
    for product, month, mwh in zip(buckets["product"], buckets["delivery_month"], buckets["net_mwh"]):
        net.setdefault(product, {})
        net[product][month] = net[product].get(month, 0.0) + float(mwh)
    return {p: {m: v for m, v in months.items() if abs(v) > 1e-6} for p, months in net.items()}


def _hedge_fit(product: str, segments: tuple, net: dict) -> Optional[dict]:
    """Whether a contract can hedge the book, and in which direction and up to what size.

    A contract must sit mostly on months the book actually holds. That rules out hedging
    a winter position with a far-dated month, which would only swap price risk for
    calendar-spread risk. A direct hedge must oppose the position and can't exceed it.
    """
    hours_total = sum(hours for _, hours in segments)
    own = net.get(product, {})
    if own:
        covered = [(m, h) for m, h in segments if m in own]
        if sum(h for _, h in covered) / hours_total < 0.6:
            return None
        window_net = sum(own[m] for m, _ in covered)
        if abs(window_net) < 1e-6:
            return None
        cap = max(abs(own[m]) / h for m, h in covered)
        return {"direction": -1.0 if window_net > 0 else 1.0, "cap": cap, "proxy": False}
    months_held = {m for months in net.values() for m in months}
    if not months_held or sum(h for m, h in segments if m in months_held) / hours_total < 0.6:
        return None
    # A proxy hedge has no position of its own to oppose, so either side can be right.
    cap = max(abs(v) / h for months in net.values() for m, v in months.items() for s, h in segments if s == m) if any(m in months_held for m, _ in segments) else 0.0
    return {"direction": 0.0, "cap": cap, "proxy": True}


def hedge_candidates(curves: dict[str, pd.DataFrame], valuation_date: date, products: Sequence[str], moves: dict,
                     days: list, eur_per_gbp: float, half_spreads: Optional[dict] = None, net: Optional[dict] = None) -> list[dict]:
    """Standard contracts that fit the book and have a mark for every month, with their P&L per 1 MW long."""
    spreads = {**DEFAULT_HALF_SPREADS, **(half_spreads or {})}
    book = _CurveBook(curves)
    net = net or {}
    candidates = []
    for product in products:
        if product not in moves:
            continue
        spec = PRODUCTS[product]
        try:
            lookup = book.lookup(product)
        except ValueError:
            continue
        for label in contract_choices(valuation_date):
            start, end = contract_window(label)
            segments = _month_segments(start, end, spec["profile"], spec["timezone"])
            if not segments or any(month not in lookup for month, _ in segments):
                continue
            fit = _hedge_fit(product, segments, net)
            if fit is None:
                continue
            series = np.zeros(len(days))
            for month, hours in segments:
                part = _position_series(product, month, float(hours), spec["currency"], moves, days, valuation_date, eur_per_gbp)
                if part is not None:
                    series = series + part
            hours_total = sum(hours for _, hours in segments)
            fx = float(eur_per_gbp) if spec["currency"] == "GBP" else 1.0
            candidates.append({"product": product, "contract": label, "start": start, "end": end, "segments": segments,
                               "currency": spec["currency"], "profile": spec["profile"], "hours": hours_total,
                               "mid": _strip_price(lookup, start, end, spec["profile"], spec["timezone"]),
                               "half_spread": float(spreads.get(spec["commodity"], 0.2)), "fx": fx, "series": series, **fit})
    return candidates


def _stepwise_hedges(pnl: np.ndarray, candidates: list[dict], confidence: float, max_contracts: int, lot_mw: float,
                     min_improvement: float, net: Optional[dict] = None) -> list[dict]:
    """Add one contract at a time, resizing the set each step to minimise P&L variance.

    Variance chooses the contracts because it is steadier than a tail quantile on a short
    history. A contract is kept only if it also cuts VaR by at least min_improvement.
    """
    base_var, _ = _var_es(pnl, confidence)
    if base_var <= 0 or not candidates:
        return []
    matrix = np.column_stack([c["series"] for c in candidates])
    chosen: list[int] = []
    steps: list[dict] = []
    current_var = base_var

    def solve(indices: list[int]) -> Optional[np.ndarray]:
        design = matrix[:, indices]
        sizes = np.linalg.lstsq(design, -pnl, rcond=None)[0]
        for position, index in enumerate(indices):
            c = candidates[index]
            limited = float(np.clip(sizes[position], -c["cap"], c["cap"]))
            # A direct hedge must oppose the position it covers.
            sizes[position] = 0.0 if c["direction"] * limited < 0 else limited
        sizes = np.round(sizes / lot_mw) * lot_mw
        if np.any(sizes == 0):
            return None
        return sizes if _reduces_positions(indices, sizes) else None

    def _reduces_positions(indices: list[int], sizes: np.ndarray) -> bool:
        """Hedges together must leave each held product with a smaller total position."""
        after = {p: dict(months) for p, months in (net or {}).items()}
        for index, size in zip(indices, sizes):
            c = candidates[index]
            if c["product"] in after:
                for month, hours in c["segments"]:
                    after[c["product"]][month] = after[c["product"]].get(month, 0.0) + float(size) * hours
        return all(sum(abs(v) for v in after[p].values()) <= sum(abs(v) for v in (net or {})[p].values()) + 1e-6 for p in after)

    for _ in range(int(max_contracts)):
        best = None
        for index in range(len(candidates)):
            if index in chosen or not np.any(matrix[:, index]):
                continue
            trial = chosen + [index]
            sizes = solve(trial)
            if sizes is None:
                continue
            residual = pnl + matrix[:, trial] @ sizes
            variance = float(np.var(residual))
            if best is None or variance < best[0] - 1e-9:
                best = (variance, index, sizes, residual)
        if best is None:
            break
        _, index, sizes, residual = best
        new_var, new_es = _var_es(residual, confidence)
        if current_var - new_var < float(min_improvement) * base_var:
            break
        chosen.append(index)
        current_var = new_var
        steps.append({"indices": list(chosen), "sizes": sizes.copy(), "var": new_var, "es": new_es, "residual": residual})
    return steps


def hedge_cost_eur(candidate: dict, mw: float) -> float:
    return abs(float(mw)) * candidate["hours"] * candidate["half_spread"] * candidate["fx"]


def hedge_plan(buckets: pd.DataFrame, history: pd.DataFrame, curves: dict[str, pd.DataFrame], valuation_date: date,
               eur_per_gbp: float, confidence: float = 0.95, max_contracts: int = 3, lot_mw: float = 1.0,
               half_spreads: Optional[dict] = None, products: Optional[Sequence[str]] = None,
               min_improvement: float = 0.05, prepared: Optional[tuple] = None) -> dict:
    """Recommend the few standard contracts that remove the most one-day risk from the book."""
    plan = {"ready": False, "reason": "", "hedges": [], "steps": [], "observations": 0, "uncovered": [], "drivers": pd.DataFrame(),
            "base_var": 0.0, "base_es": 0.0, "hedged_var": 0.0, "hedged_es": 0.0, "total_cost": 0.0, "days": [],
            "base_pnl": np.array([]), "hedged_pnl": np.array([]), "out_of_sample": None, "candidates": 0, "confidence": float(confidence)}
    if buckets is None or buckets.empty:
        plan["reason"] = "The book has no executed trades, so there is nothing to hedge yet."
        return plan
    moves, days = prepared or _contract_moves(history)
    plan["observations"] = len(days)
    if len(days) < MIN_HEDGE_OBSERVATIONS:
        plan["reason"] = f"The advisor needs at least {MIN_HEDGE_OBSERVATIONS} daily curve moves, and {len(days)} are saved so far."
        return plan
    pnl, uncovered, rows = _book_series(buckets, moves, days, valuation_date, eur_per_gbp)
    plan.update({"uncovered": uncovered, "days": [d.date() for d in days], "base_pnl": pnl, "hedged_pnl": pnl})
    plan["base_var"], plan["base_es"] = _var_es(pnl, confidence)
    plan["hedged_var"], plan["hedged_es"] = plan["base_var"], plan["base_es"]
    plan["drivers"] = risk_drivers(rows, pnl, confidence)
    plan["ready"] = True
    universe = list(products) if products is not None else list(PRODUCTS)
    net = _book_net(buckets)
    candidates = hedge_candidates(curves, valuation_date, universe, moves, days, eur_per_gbp, half_spreads, net)
    plan["candidates"] = len(candidates)
    steps = _stepwise_hedges(pnl, candidates, confidence, max_contracts, lot_mw, min_improvement, net)
    if not steps:
        return plan
    final = steps[-1]
    book_products = set(buckets["product"])
    hedges = []
    for position, (index, size) in enumerate(zip(final["indices"], final["sizes"])):
        c = candidates[index]
        step = steps[position]
        hedges.append({"product": c["product"], "contract": c["contract"], "start": c["start"], "end": c["end"], "mw": float(size),
                       "action": "Buy" if size > 0 else "Sell", "mwh": abs(float(size)) * c["hours"], "mid": c["mid"], "currency": c["currency"],
                       "profile": c["profile"], "half_spread": c["half_spread"], "cost_eur": hedge_cost_eur(c, size),
                       "var_after_step": step["var"], "proxy": c["product"] not in book_products})
    plan.update({"hedges": hedges, "hedged_pnl": final["residual"], "hedged_var": final["var"], "hedged_es": final["es"],
                 "total_cost": float(sum(h["cost_eur"] for h in hedges)),
                 "steps": [{"contracts": i + 1, "var": s["var"], "es": s["es"],
                            "cost": float(sum(hedge_cost_eur(candidates[k], x) for k, x in zip(s["indices"], s["sizes"])))} for i, s in enumerate(steps)]})
    # Choose on the earlier days only, then test on later days the choice never saw.
    if len(days) >= 45:
        cut = int(len(days) * 0.7)
        train = [{**c, "series": c["series"][:cut]} for c in candidates]
        fitted = _stepwise_hedges(pnl[:cut], train, confidence, max_contracts, lot_mw, min_improvement, net)
        if fitted:
            last = fitted[-1]
            held = pnl[cut:] + np.column_stack([candidates[i]["series"][cut:] for i in last["indices"]]) @ last["sizes"]
            plan["out_of_sample"] = {"test_days": len(days) - cut, "train_days": cut,
                                     "unhedged_std": float(np.std(pnl[cut:])), "hedged_std": float(np.std(held)),
                                     "unhedged_worst": float(pnl[cut:].min()), "hedged_worst": float(held.min()),
                                     "contracts": [f"{candidates[i]['product']} {candidates[i]['contract']}" for i in last["indices"]]}
    return plan


def custom_hedge_effect(buckets: pd.DataFrame, history: pd.DataFrame, valuation_date: date, eur_per_gbp: float, product: str,
                        contract: str, mw: float, confidence: float = 0.95, prepared: Optional[tuple] = None) -> dict:
    """VaR of the book with one extra contract of your choosing. mw is positive to buy."""
    moves, days = prepared or _contract_moves(history)
    pnl, _, _ = _book_series(buckets, moves, days, valuation_date, eur_per_gbp)
    spec = PRODUCTS[product]
    start, end = contract_window(contract)
    extra = np.zeros(len(days))
    covered = False
    for month, hours in _month_segments(start, end, spec["profile"], spec["timezone"]):
        part = _position_series(product, month, float(mw) * hours, spec["currency"], moves, days, valuation_date, eur_per_gbp)
        if part is not None:
            extra, covered = extra + part, True
    before, _ = _var_es(pnl, confidence)
    after, _ = _var_es(pnl + extra, confidence)
    return {"before": before, "after": after, "covered": covered}


def hedge_ladder_rows(buckets: pd.DataFrame, hedges: Sequence[dict]) -> pd.DataFrame:
    """Net position by product and delivery month once the hedges are added."""
    rows = buckets[["product", "delivery_month", "net_mwh"]].copy() if buckets is not None and not buckets.empty else pd.DataFrame(columns=["product", "delivery_month", "net_mwh"])
    extra = []
    for hedge in hedges:
        spec = PRODUCTS[hedge["product"]]
        for month, hours in _month_segments(hedge["start"], hedge["end"], spec["profile"], spec["timezone"]):
            extra.append({"product": hedge["product"], "delivery_month": month, "net_mwh": float(hedge["mw"]) * hours})
    combined = pd.concat([rows, pd.DataFrame(extra)], ignore_index=True) if extra else rows
    return combined.groupby(["product", "delivery_month"], as_index=False)["net_mwh"].sum()


def proposed_hedge_trades(hedges: Sequence[dict], valuation_date: date, stamp: str) -> pd.DataFrame:
    """Turn hedges into proposed trades at mid plus or minus the assumed half-spread."""
    rows = []
    for number, hedge in enumerate(hedges, start=1):
        buying = hedge["mw"] > 0
        whole_month = hedge["end"] == _add_months(hedge["start"], 1)
        rows.append({"trade_id": f"H-{stamp}-{number}", "trade_date": valuation_date.isoformat(), "counterparty": "To be agreed",
                     "product": hedge["product"], "instrument": "FORWARD" if whole_month else "SWAP", "settlement_type": "FINANCIAL",
                     "side": "BUY_FIXED" if buying else "SELL_FIXED", "delivery_start": hedge["start"].isoformat(), "delivery_end": hedge["end"].isoformat(),
                     "profile": hedge["profile"], "quantity_mw": abs(float(hedge["mw"])),
                     "trade_price": round(hedge["mid"] + (hedge["half_spread"] if buying else -hedge["half_spread"]), 4),
                     "currency": hedge["currency"], "status": "PROPOSED", "notes": f"Suggested by Hedge Advisor ({hedge['contract']})"})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Market Read: rule-based readings of the evidence, with its reasoning shown
# -----------------------------------------------------------------------------
ZONE_WEATHER = {"France": "Paris / French power proxy", "Germany-Luxembourg": "Berlin / German power proxy",
                "Netherlands": "Amsterdam / TTF proxy", "Belgium": "Brussels / Belgian power proxy",
                "Spain": "Madrid / Iberian power proxy", "Great Britain": "London / NBP proxy"}
GAS_WEATHER_LOCATIONS = ("Amsterdam / TTF proxy", "London / NBP proxy", "Paris / French power proxy",
                         "Berlin / German power proxy", "Brussels / Belgian power proxy")
# Natural gas use by country in 2024, billion cubic metres, rounded. Used only as weights.
GAS_DEMAND_WEIGHTS = {"Berlin / German power proxy": 78.0, "London / NBP proxy": 63.0, "Amsterdam / TTF proxy": 32.0,
                      "Paris / French power proxy": 31.0, "Brussels / Belgian power proxy": 14.0}
# EU storage serves EU demand, so the winter projection leaves out Britain.
EU_STORAGE_LOCATIONS = tuple(name for name in GAS_WEATHER_LOCATIONS if name != "London / NBP proxy")
STALE_DAYS = 10
# Signals built from the same fact share one weight, so one fact can't look like several opinions.
SIGNAL_THEMES = {"storage_level": "storage", "storage_pace": "storage", "winter_outlook": "storage", "lng": "supply", "summer_winter": "curve",
                 "switching": "fuel", "fuel": "fuel", "weather": "weather", "momentum": "price", "negative": "price",
                 "residual": "balance", "wind": "balance", "nuclear": "nuclear"}
THEME_WEIGHT_CAP = 3.0
NUCLEAR_MIN_SHARE = 0.15


def _signal(key: str, name: str, weight: float, z: Optional[float] = None, value: str = "", why: str = "",
            flip: str = "", missing: str = "") -> dict:
    """One piece of evidence. z is in standard-deviation-like units, positive is bullish."""
    available = z is not None and math.isfinite(float(z))
    return {"key": key, "name": name, "weight": float(weight), "available": available,
            "score": float(np.clip(float(z) / 2.0, -1.0, 1.0)) if available else 0.0,
            "value": value, "why": why, "flip": flip, "missing": missing}


def _theme(signal: dict) -> str:
    return SIGNAL_THEMES.get(signal["key"], signal["key"])


def counted_weights(signals: Sequence[dict]) -> list[float]:
    """Each signal's weight after capping its theme, using only the signals that are loaded."""
    totals: dict = {}
    for signal in signals:
        if signal["available"]:
            totals[_theme(signal)] = totals.get(_theme(signal), 0.0) + signal["weight"]
    return [signal["weight"] * min(1.0, THEME_WEIGHT_CAP / totals[_theme(signal)]) if signal["available"] else 0.0 for signal in signals]


def summarise_read(signals: Sequence[dict]) -> dict:
    """Combine signals into a reading. Confidence needs enough evidence, agreement and more than one independent theme."""
    counted = counted_weights(signals)
    theme_totals: dict = {}
    for signal in signals:
        theme_totals[_theme(signal)] = theme_totals.get(_theme(signal), 0.0) + signal["weight"]
    total_weight = sum(min(value, THEME_WEIGHT_CAP) for value in theme_totals.values()) or 1.0
    live_weight = sum(counted)
    live = [s for s in signals if s["available"]]
    result = {"reading": "Not enough data", "score": 0.0, "confidence": "None", "coverage": live_weight / total_weight,
              "agreement": 0.0, "supporting": [], "against": [], "live": len(live), "total": len(signals), "themes": 0,
              "missing": [s for s in signals if not s["available"]]}
    if len(live) < 2:
        return result
    score = sum(weight * s["score"] for weight, s in zip(counted, signals)) / live_weight
    label = ("Bullish" if score >= 0.5 else "Leaning bullish" if score >= 0.15 else
             "Bearish" if score <= -0.5 else "Leaning bearish" if score <= -0.15 else "Balanced")
    bullish = sum(weight for weight, s in zip(counted, signals) if s["score"] >= 0.1)
    bearish = sum(weight for weight, s in zip(counted, signals) if s["score"] <= -0.1)
    agreement = max(bullish, bearish) / (bullish + bearish) if bullish + bearish else 0.0
    coverage = result["coverage"]
    direction = 1.0 if score >= 0 else -1.0
    net: dict = {}
    for weight, signal in zip(counted, signals):
        if signal["available"]:
            held = net.setdefault(_theme(signal), [0.0, 0.0])
            held[0] += weight * signal["score"]
            held[1] += weight
    themes = sum(1 for value, weight in net.values() if value / weight * direction >= 0.1)
    confidence = "High" if coverage >= 0.75 and agreement >= 0.75 else "Medium" if coverage >= 0.5 and agreement >= 0.6 else "Low"
    # High confidence needs independent support: signals about one fact aren't a second opinion.
    if confidence == "High" and themes < 2:
        confidence = "Medium"
    # A balanced reading is never a confident call: either signals disagree or none is strong.
    if label == "Balanced":
        confidence = "Low" if bullish and bearish else "Medium"
    order = sorted(zip(counted, signals), key=lambda pair: -pair[0] * abs(pair[1]["score"]))
    ranked = [s for weight, s in order if s["available"]]
    result.update({"reading": label, "score": float(score), "confidence": confidence, "agreement": agreement, "themes": themes,
                   "supporting": [s for s in ranked if s["score"] * direction >= 0.1],
                   "against": [s for s in ranked if s["score"] * direction <= -0.1]})
    return result


def _storage_daily(frame: pd.DataFrame) -> pd.DataFrame:
    date_col = _find_column(frame, ("observation_date", "gasDayStart", "gas_day")) if frame is not None and not frame.empty else None
    fill_col = _find_column(frame, ("full", "fillLevel", "fill_level")) if frame is not None and not frame.empty else None
    if not date_col or not fill_col:
        return pd.DataFrame(columns=["day", "fill"])
    data = pd.DataFrame({"day": pd.to_datetime(frame[date_col], errors="coerce"), "fill": _number_series(frame[fill_col])}).dropna()
    return data.groupby("day", as_index=False)["fill"].mean().sort_values("day")


def _heating_signal(daily: pd.DataFrame, normals: pd.DataFrame, locations: Sequence[str], weight: float, market: str) -> dict:
    name = "Weather against normal"
    missing = "Fetch the weather forecast and load 10-year normals."
    anomaly = hdd_vs_normal(daily, normals)
    anomaly = anomaly.loc[anomaly["location"].isin(locations)] if not anomaly.empty else anomaly
    if anomaly.empty:
        return _signal("weather", name, weight, missing=missing)
    weights = anomaly["location"].map(GAS_DEMAND_WEIGHTS).fillna(1.0).to_numpy(float) if len(locations) > 1 else np.ones(len(anomaly))
    forecast = float(np.average(anomaly["forecast_hdd"], weights=weights))
    normal = float(np.average(anomaly["normal_hdd"], weights=weights))
    days = int(anomaly["days"].max())
    if normal < 3.0:
        return _signal("weather", name, weight, 0.0, f"Outside the heating season: only {normal:.0f} normal degree days ahead.",
                       "Temperature has little effect on heating demand at this time of year.", "This becomes relevant when heating demand returns.")
    gap = (forecast - normal) / normal
    return _signal("weather", name, weight, gap / 0.15,
                   f"Next {days} days: {forecast:.0f} heating degree days against a normal of {normal:.0f} ({gap * 100:+.0f}%).",
                   f"Colder than normal lifts {market} demand, and milder than normal reduces it.",
                   f"Reverses if the forecast moves {'below' if gap > 0 else 'above'} {normal:.0f} degree days.")


def gas_signals(history: pd.DataFrame, alsi: pd.DataFrame, daily: pd.DataFrame, normals: pd.DataFrame,
                imported: dict, curves: dict, today: date, eur_per_usd: float = 0.92, outlook: Optional[dict] = None) -> list[dict]:
    """Evidence for European gas. Curve signals use recent imported marks only, never demo prices."""
    signals = []
    store = _storage_daily(history)
    level = _signal("storage_level", "Storage against the five-year average", 3.0, missing="Load 6 years of storage history.")
    pace = _signal("storage_pace", "Storage pace against normal", 2.0, missing="Load 6 years of storage history.")
    if not store.empty:
        latest = store.iloc[-1]
        age = (pd.Timestamp(today) - latest["day"]).days
        if age > STALE_DAYS:
            level["missing"] = pace["missing"] = f"Storage data is {age} days old. Load it again."
        else:
            seasonal = storage_seasonal(history, today=today)
            band = seasonal["band"]
            if not band.empty:
                target = pd.Timestamp(year=2000, month=latest["day"].month, day=latest["day"].day)
                near = band.iloc[(band["calendar_day"] - target).abs().argsort()[:1]].iloc[0]
                gap = float(latest["fill"]) - float(near["average"])
                spread = max((float(near["high"]) - float(near["low"])) / 2.0, 3.0)
                level = _signal("storage_level", level["name"], 3.0, -gap / spread,
                                f"{latest['fill']:.1f}% full, {abs(gap):.1f} points {'above' if gap >= 0 else 'below'} the five-year average of {near['average']:.1f}% for this date.",
                                "Less gas in store than usual leaves less cushion for the rest of the season.",
                                f"Reverses if fill moves {'below' if gap >= 0 else 'above'} {near['average']:.1f}% at this time of year.")
            lookup = dict(zip(store["day"], store["fill"]))
            then = lookup.get(latest["day"] - pd.Timedelta(days=7))
            history_changes = []
            for offset in range(1, 6):
                a = lookup.get(latest["day"] - pd.DateOffset(years=offset))
                b = lookup.get(latest["day"] - pd.DateOffset(years=offset) - pd.Timedelta(days=7))
                if a is not None and b is not None:
                    history_changes.append(float(a) - float(b))
            if then is not None and len(history_changes) >= 2:
                change = float(latest["fill"]) - float(then)
                usual = float(np.mean(history_changes))
                scale = max(float(np.std(history_changes)), 0.3)
                pace = _signal("storage_pace", pace["name"], 2.0, -(change - usual) / scale,
                               f"Changed {change:+.1f} points in 7 days, against {usual:+.1f} usually at this time of year.",
                               "Storage falling faster, or filling slower, than normal points to a tighter balance.",
                               f"Reverses if the weekly change moves {'below' if change > usual else 'above'} {usual:+.1f} points.")
    signals += [level, pace, _heating_signal(daily, normals, GAS_WEATHER_LOCATIONS, 2.5, "gas")]

    lng = _signal("lng", "LNG send-out against the past month", 1.5, missing="Fetch at least 14 days of ALSI LNG data.")
    if alsi is not None and not alsi.empty:
        date_col = _find_column(alsi, ("observation_date", "gasDayStart", "gas_day"))
        send_col = _find_column(alsi, ("sendOut", "sendout", "send_out"))
        if date_col and send_col:
            flow = pd.DataFrame({"day": pd.to_datetime(alsi[date_col], errors="coerce"), "value": _number_series(alsi[send_col])}).dropna()
            flow = flow.groupby("day", as_index=False)["value"].mean().sort_values("day")
            if len(flow) >= 14:
                recent, prior = flow["value"].tail(7), flow["value"].iloc[:-7].tail(23)
                base = float(prior.mean())
                scale = max(float(prior.std()), abs(base) * 0.01, 1.0)
                gap = float(recent.mean()) - base
                lng = _signal("lng", lng["name"], 1.5, -gap / scale,
                              f"{recent.mean():,.0f} GWh/d over the last week, against {base:,.0f} GWh/d before that.",
                              "More LNG reaching Europe adds supply, and less removes it.",
                              f"Reverses if send-out moves {'below' if gap >= 0 else 'above'} {base:,.0f} GWh/d.")
    signals.append(lng)

    season = _signal("summer_winter", "Summer-to-winter spread", 1.5, missing="Import a real TTF curve in Forward Curves.")
    switching = _signal("switching", "TTF against the coal switching price", 1.0, missing="Import real TTF, EUA and API2 coal curves in Forward Curves.")
    if "TTF Natural Gas" in imported:
        lookup = _curve_lookup(validate_curve(imported["TTF Natural Gas"]))
        spec = PRODUCTS["TTF Natural Gas"]
        for year in range(today.year, today.year + 3):
            summer, winter = f"Sum-{year % 100:02d}", f"Win-{year % 100:02d}"
            if contract_window(summer)[0] > today:
                try:
                    s_price = _strip_price(lookup, *contract_window(summer), "BASE", spec["timezone"])
                    w_price = _strip_price(lookup, *contract_window(winter), "BASE", spec["timezone"])
                except ValueError:
                    break
                spread = w_price - s_price
                season = _signal("summer_winter", season["name"], 1.5, -spread / 2.0,
                                 f"{winter} minus {summer} is {spread:+.2f} EUR/MWh.",
                                 "A thin or inverted winter premium gives little reason to inject, which can tighten next winter.",
                                 "Reverses if the spread moves through zero." if spread < 0 else f"Weakens as the spread narrows from {spread:+.2f} EUR/MWh.")
                break
        if {"EUA Carbon", "API2 Coal"} <= set(imported):
            table = spreads_table(curves, "French Power Base", "TTF Natural Gas", None, 1.17, eur_per_usd)
            if not table.empty:
                row = table.iloc[0]
                gap = float(row["gas_eur_mwh"]) - float(row["switching_gas_price"])
                switching = _signal("switching", switching["name"], 1.0, -gap / max(float(row["switching_gas_price"]) * 0.1, 1.0),
                                    f"{row['contract']} TTF is {abs(gap):.2f} EUR/MWh {'above' if gap >= 0 else 'below'} the {row['switching_gas_price']:.2f} switching price.",
                                    "Gas above the switching price loses power-sector demand to coal, and gas below it gains that demand.",
                                    f"Reverses if TTF moves through {row['switching_gas_price']:.2f} EUR/MWh.")
    forward = outlook_signal(outlook)
    return signals + [season, switching] + ([forward] if forward else [])


def _hourly_residual(load: pd.DataFrame, generation: pd.DataFrame) -> pd.DataFrame:
    if load is None or load.empty or generation is None or generation.empty:
        return pd.DataFrame(columns=["hour", "residual"])
    demand = pd.DataFrame({"hour": pd.to_datetime(load["timestamp"], errors="coerce", utc=True).dt.floor("h"), "load": _number_series(load["value"])}).dropna()
    demand = demand.groupby("hour", as_index=False)["load"].mean()
    labels = generation["series"].astype(str).str.casefold()
    subset = generation.loc[labels.str.contains("wind onshore|wind offshore|solar", regex=True) & ~labels.str.contains("consumption")]
    if subset.empty:
        return pd.DataFrame(columns=["hour", "residual"])
    output = pd.DataFrame({"hour": pd.to_datetime(subset["timestamp"], errors="coerce", utc=True).dt.floor("h"), "series": subset["series"], "mw": _number_series(subset["value"])}).dropna()
    output = output.groupby(["hour", "series"], as_index=False)["mw"].mean().groupby("hour", as_index=False)["mw"].sum()
    joined = demand.merge(output, on="hour")
    joined["residual"] = joined["load"] - joined["mw"]
    return joined[["hour", "residual"]]


def _nuclear_signal(generation: pd.DataFrame, tz: str, today: date) -> Optional[dict]:
    """Nuclear output against recent weeks, in zones where nuclear is a large share. Otherwise no signal."""
    if generation is None or generation.empty or not {"series", "value", "timestamp"}.issubset(generation.columns):
        return None
    labels = generation["series"].astype(str).str.casefold()
    produced = generation.loc[~labels.str.contains("consumption")]
    if produced.empty:
        return None
    frame = pd.DataFrame({"day": pd.to_datetime(produced["timestamp"], errors="coerce", utc=True).dt.tz_convert(tz).dt.date,
                          "hour": pd.to_datetime(produced["timestamp"], errors="coerce", utc=True).dt.floor("h"),
                          "series": produced["series"].astype(str), "mw": _number_series(produced["value"])}).dropna()
    hourly = frame.groupby(["day", "hour", "series"], as_index=False)["mw"].mean()
    is_nuclear = hourly["series"].str.casefold().str.contains("nuclear")
    if not is_nuclear.any():
        return None
    recent_days = sorted(hourly["day"].unique())[-28:]
    window = hourly.loc[hourly["day"].isin(recent_days)]
    share = float(window.loc[window["series"].str.casefold().str.contains("nuclear"), "mw"].sum()) / max(float(window["mw"].sum()), 1.0)
    if share < NUCLEAR_MIN_SHARE:
        return None
    name, weight = "Nuclear output against the past weeks", 2.5
    daily = hourly.loc[is_nuclear].groupby(["day", "hour"], as_index=False)["mw"].sum().groupby("day")["mw"].mean()
    if len(daily) < 14:
        return _signal("nuclear", name, weight, missing="Fetch at least 14 days of actual generation.")
    if (today - daily.index.max()).days > STALE_DAYS:
        return _signal("nuclear", name, weight, missing="Actual generation is out of date. Fetch it again.")
    recent, prior = daily.tail(7), daily.iloc[:-7].tail(21)
    gap = float(recent.mean()) - float(prior.mean())
    scale = max(float(prior.std()), float(prior.mean()) * 0.03, 1.0)
    return _signal("nuclear", name, weight, -gap / scale,
                   f"{recent.mean() / 1000:.1f} GW over the last 7 days, {gap / float(prior.mean()) * 100:+.1f}% against {prior.mean() / 1000:.1f} GW before that.",
                   "Lower nuclear output means dearer plants or imports must fill the gap. This shows output so far, not planned outages ahead.",
                   f"Reverses if output moves {'above' if gap < 0 else 'below'} {prior.mean() / 1000:.1f} GW.")


def power_signals(zone: str, prices: pd.DataFrame, load: pd.DataFrame, generation: pd.DataFrame, daily: pd.DataFrame,
                  normals: pd.DataFrame, imported: dict, today: date) -> list[dict]:
    """Evidence for one power zone. Prices come from day-ahead data, never forward demo marks."""
    tz = ENTSOE_ZONES[zone]["timezone"]
    series = spot_series(prices, tz)
    momentum = _signal("momentum", "Day-ahead price against the past weeks", 2.0, missing="Fetch at least 14 days of day-ahead prices.")
    negative = _signal("negative", "Negative-price hours", 1.5, missing="Fetch at least 14 days of day-ahead prices.")
    if not series.empty:
        series_day = series["local"].dt.date
        by_day = series.assign(day=series_day).groupby("day", as_index=False)["value"].mean()
        age = (today - by_day["day"].max()).days
        if age > STALE_DAYS:
            momentum["missing"] = negative["missing"] = f"Day-ahead prices are {age} days old. Fetch them again."
        elif len(by_day) >= 14:
            recent, prior = by_day["value"].tail(7), by_day["value"].iloc[:-7].tail(21)
            gap = float(recent.mean()) - float(prior.mean())
            momentum = _signal("momentum", momentum["name"], 2.0, gap / max(float(prior.std()), 3.0),
                               f"{recent.mean():.1f} EUR/MWh over the last 7 days, against {prior.mean():.1f} before that.",
                               "Shows the direction prices are already moving in, not why.",
                               f"Reverses if the 7-day average moves {'below' if gap >= 0 else 'above'} {prior.mean():.1f} EUR/MWh.")
            cut = series_day >= by_day["day"].iloc[-7]
            earlier = series_day >= by_day["day"].iloc[max(len(by_day) - 28, 0)]
            share = lambda part: float(part.loc[part["value"] < 0, "hours"].sum() / part["hours"].sum() * 100.0) if not part.empty else 0.0
            recent_share, prior_share = share(series.loc[cut]), share(series.loc[earlier & ~cut])
            hours = float(series.loc[cut & series["value"].lt(0), "hours"].sum())
            negative = _signal("negative", negative["name"], 1.5, -(recent_share - prior_share) / 2.5,
                               f"{hours:.0f} negative hours in the last 7 days, {recent_share:.1f}% of the time against {prior_share:.1f}% before.",
                               "More negative hours mean renewables are oversupplying the system, which pushes prices down.",
                               f"Reverses if the share moves {'below' if recent_share > prior_share else 'above'} {prior_share:.1f}%.")
    residual = _signal("residual", "Residual load against the past weeks", 2.5, missing=f"Fetch actual load and generation for {zone}.")
    hourly = _hourly_residual(load, generation)
    if not hourly.empty:
        days = hourly.assign(day=hourly["hour"].dt.tz_convert(tz).dt.date).groupby("day", as_index=False)["residual"].mean()
        if len(days) >= 14 and (today - days["day"].max()).days <= STALE_DAYS:
            recent, prior = days["residual"].tail(7), days["residual"].iloc[:-7].tail(21)
            gap = float(recent.mean()) - float(prior.mean())
            scale = max(float(prior.std()), abs(float(prior.mean())) * 0.02, 1.0)
            residual = _signal("residual", residual["name"], 2.5, gap / scale,
                               f"{recent.mean() / 1000:.1f} GW over the last 7 days, against {prior.mean() / 1000:.1f} GW before that.",
                               "Residual load is what wind and solar leave for other plants. More of it brings dearer plants into the merit order.",
                               f"Reverses if it moves {'below' if gap >= 0 else 'above'} {prior.mean() / 1000:.1f} GW.")
    location = ZONE_WEATHER.get(zone)
    heating = _heating_signal(daily, normals, (location,) if location else (), 2.0, "power")
    wind = _signal("wind", "Wind outlook against recent days", 1.5, missing="Fetch the weather forecast.")
    if daily is not None and not daily.empty and {"location", "forecast_period", "wind_speed_100m_mean_kmh"}.issubset(daily.columns) and location:
        here = daily.loc[daily["location"].eq(location)]
        ahead = pd.to_numeric(here.loc[here["forecast_period"].eq("Forecast"), "wind_speed_100m_mean_kmh"], errors="coerce").head(7).dropna()
        before = pd.to_numeric(here.loc[here["forecast_period"].ne("Forecast"), "wind_speed_100m_mean_kmh"], errors="coerce").dropna()
        if len(ahead) >= 3 and len(before) >= 2 and before.mean() > 0:
            gap = float(ahead.mean()) - float(before.mean())
            wind = _signal("wind", wind["name"], 1.5, -gap / (0.2 * float(before.mean())),
                           f"{ahead.mean():.0f} km/h forecast at 100 m, against {before.mean():.0f} km/h in recent days.",
                           "Windier weather adds cheap output and calmer weather removes it. This uses one reference city, so treat it as a rough guide.",
                           f"Reverses if the forecast moves {'above' if gap < 0 else 'below'} {before.mean():.0f} km/h.")
    fuel = _signal("fuel", "Day-ahead against gas plant cost", 1.5, missing="Import real TTF and EUA curves in Forward Curves.")
    if {"TTF Natural Gas", "EUA Carbon"} <= set(imported) and not series.empty:
        gas = _curve_lookup(validate_curve(imported["TTF Natural Gas"]))
        carbon = _curve_lookup(validate_curve(imported["EUA Carbon"]))
        shared = sorted(set(gas) & set(carbon))
        if shared:
            eta, factor = DEFAULT_SPREAD_PARAMS["gas_efficiency"], DEFAULT_SPREAD_PARAMS["gas_emission_factor"]
            cost = gas[shared[0]] / eta + carbon[shared[0]] * factor / eta
            spot = float(series.tail(int(7 * 24 / max(float(series["hours"].median()), 0.25)))["value"].mean())
            gap = spot - cost
            fuel = _signal("fuel", fuel["name"], 1.5, -gap / 15.0,
                           f"{spot:.1f} EUR/MWh over the last week, against a {cost:.1f} EUR/MWh cost for a 50% efficient gas plant.",
                           "Prices far above gas plant cost tend to attract supply, and prices far below it leave little room to fall.",
                           f"Reverses if the price moves through {cost:.1f} EUR/MWh.")
    nuclear = _nuclear_signal(generation, tz, today)
    return [residual] + ([nuclear] if nuclear else []) + [momentum, heating, negative, wind, fuel]


# -----------------------------------------------------------------------------
# Forward curve files: marks dropped in a folder, with their true date
# -----------------------------------------------------------------------------
CURVE_FOLDER_NAME = "meridian_curves"
CURVE_FILE_TYPES = (".csv", ".xlsx")
CURVE_STALE_DAYS = 3            # a Friday close is still current on Monday
CURVE_JUMP_LIMIT = 2.0          # a price this many times the last marks is probably a unit error
PRODUCT_ALIASES = {"ttf": "TTF Natural Gas", "nbp": "NBP Natural Gas", "eua": "EUA Carbon", "carbon": "EUA Carbon",
                   "api2": "API2 Coal", "coal": "API2 Coal", "fr base": "French Power Base", "fr peak": "French Power Peak",
                   "french base": "French Power Base", "french peak": "French Power Peak", "de base": "German Power Base",
                   "de peak": "German Power Peak", "german base": "German Power Base", "german peak": "German Power Peak"}


def _pick_column(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    return next((name for name in names if name in frame.columns), None)


def _curve_label(value: object) -> str:
    """Accept Dec-26, Q1-27, 2026-12, 2026-12-01 or 01/12/2026. Anything ambiguous is refused."""
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return month_contract_label(date(value.year, value.month, 1))
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?(?:[ T].*)?", text)
    if match:
        return month_contract_label(date(int(match[1]), int(match[2]), 1))
    match = re.fullmatch(r"0?1/(\d{1,2})/(\d{4})", text)
    if match:
        return month_contract_label(date(int(match[2]), int(match[1]), 1))
    contract_window(text)
    return text


def _curve_day(value: object) -> date:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return date(value.year, value.month, value.day)
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T].*)?", text)
    if match:
        return date(int(match[1]), int(match[2]), int(match[3]))
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if match:
        return date(int(match[3]), int(match[2]), int(match[1]))
    raise ValueError(f"Can't read the date '{text}'. Use 2026-11-04 or 04/11/2026.")


def curve_file_day(file_name: str, fallback: date) -> date:
    """A date in the file name wins, otherwise the day the file was last saved."""
    match = re.search(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})", file_name)
    if match:
        try:
            return date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            pass
    return fallback


def parse_curve_table(raw: pd.DataFrame, file_name: str, fallback_day: date, shapes: dict, latest: dict, today: date) -> list[dict]:
    """Turn one file into monthly curves, one for each product and date in it.

    shapes holds the curve used to shape quarters and years into months. latest holds
    {product: (date, curve)} of the last imported marks, used to catch unit errors.
    """
    frame = raw.copy()
    frame.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().casefold()).strip("_") for c in frame.columns]
    product_col = _pick_column(frame, ("product", "commodity", "instrument"))
    price_col = _pick_column(frame, ("price", "settlement_price", "settlement", "settle", "mid", "close", "mark"))
    period_col = _pick_column(frame, ("contract", "tenor", "delivery_month", "delivery", "month", "period"))
    day_col = _pick_column(frame, ("curve_date", "snapshot_date", "as_of", "asof", "date"))
    missing = [name for name, col in (("product", product_col), ("price", price_col), ("contract or delivery_month", period_col)) if col is None]
    if missing:
        raise ValueError(f"Missing column: {', '.join(missing)}.")
    frame = frame.dropna(subset=[price_col, period_col])
    if frame.empty:
        raise ValueError("The file has no prices.")
    names = {p.casefold(): p for p in CURVE_PRODUCTS} | PRODUCT_ALIASES
    labels = frame[product_col].astype(str).str.strip().str.casefold()
    unknown = sorted(set(frame.loc[~labels.isin(names), product_col].astype(str)))
    if unknown:
        raise ValueError(f"Unknown product: {', '.join(unknown[:3])}.")
    data = pd.DataFrame({"product": labels.map(names), "label": [_curve_label(v) for v in frame[period_col]],
                         "price": _number_series(frame[price_col]),
                         "day": [_curve_day(v) for v in frame[day_col]] if day_col else curve_file_day(file_name, fallback_day)})
    bad = data.loc[data["price"].isna(), "label"].tolist()
    if bad:
        raise ValueError(f"No readable price for {', '.join(bad[:3])}.")
    if max(data["day"]) > today:
        raise ValueError(f"The file is dated {max(data['day']):%d %b %Y}, which is in the future.")
    results = []
    for (product, day), group in data.groupby(["product", "day"], sort=True):
        spec = CURVE_PRODUCTS[product]
        if group["label"].duplicated().any():
            raise ValueError(f"{product} has the same contract twice on {day:%d %b %Y}.")
        months_only = all(_add_months(contract_window(label)[0], 1) == contract_window(label)[1] for label in group["label"])
        if months_only:
            curve = validate_curve(pd.DataFrame({"delivery_month": [contract_window(label)[0] for label in group["label"]], "product": product,
                                                 "price": group["price"].to_numpy(), "currency": spec["currency"], "unit": spec["unit"]}))
        else:
            curve = decompose_tenor_quotes(list(zip(group["label"], group["price"])), product, shapes.get(product))
        known = latest.get(product)
        if known is not None and abs((day - known[0]).days) <= 10:
            before = _curve_lookup(validate_curve(known[1]))
            ratios = [price / before[month] for month, price in zip(curve["delivery_month"], curve["price"]) if before.get(month, 0.0) > 0 and price > 0]
            if ratios and not 1.0 / CURVE_JUMP_LIMIT <= float(np.median(ratios)) <= CURVE_JUMP_LIMIT:
                raise ValueError(f"{product} prices are {float(np.median(ratios)):.1f} times the marks of {known[0]:%d %b}. Check the unit and the product name.")
        results.append({"product": product, "day": day, "curve": curve})
    return results


def curve_mark_date(metadata: Optional[dict]) -> Optional[date]:
    """The market date of imported marks, which can be older than the import."""
    for key in ("curve_date", "imported_at_utc"):
        text = str((metadata or {}).get(key, ""))[:10]
        try:
            return date.fromisoformat(text)
        except ValueError:
            continue
    return None


def curve_state(product: str, custom: dict, metadata: dict, today: date) -> dict:
    if product not in custom:
        return {"state": "Demo", "day": None, "age": None}
    day = curve_mark_date(metadata.get(product))
    age = (today - day).days if day else None
    return {"state": "Unknown date" if age is None else "Current" if age <= CURVE_STALE_DAYS else "Stale", "day": day, "age": age}


def fresh_curves(custom: dict, metadata: dict, today: date, max_age: int = 10) -> dict:
    """Imported curves recent enough to count as evidence of today's market."""
    fresh = {}
    for product, curve in custom.items():
        age = curve_state(product, custom, metadata, today)["age"]
        if age is not None and age <= max_age:
            fresh[product] = curve
    return fresh


# -----------------------------------------------------------------------------
# Market Read track record: was the read right, compared with chance?
# -----------------------------------------------------------------------------
READ_METHOD_VERSION = 2
PRICE_SERIES_VERSION = 1
READ_METHOD_DATE = date(2026, 11, 6)
TRACK_HORIZONS = (1, 5, 20)
TRACK_MIN_INDEPENDENT = 20
CALL_THRESHOLD = 0.15
SIGNAL_THRESHOLD = 0.10
REPLAY_LOOKBACK_DAYS = 40
MARKET_ZONES = {"French power": "France", "German power": "Germany-Luxembourg", "Dutch power": "Netherlands",
                "Belgian power": "Belgium", "Spanish power": "Spain", "British power": "Great Britain"}
REAL_MARK_EXCLUSIONS = ("Demo marks", "Part real marks", SYNTHETIC_HISTORY_SOURCE)


def read_row(read_date: date, market: str, signals: Sequence[dict], summary: dict, price_through: date, source: str) -> dict:
    return {"read_date": read_date.isoformat(), "market": market, "source": source, "reading": summary["reading"], "score": float(summary["score"]),
            "confidence": summary["confidence"], "price_through": price_through.isoformat(),
            "signals": json.dumps({s["key"]: [round(s["score"], 4), s["name"]] for s in signals if s["available"]})}


def wilson_interval(hits: float, n: float, z: float = 1.645) -> tuple[float, float]:
    """The likely range of a true hit rate. z=1.645 gives a 90% range."""
    if n <= 0:
        return 0.0, 1.0
    rate = hits / n
    denominator = 1.0 + z * z / n
    middle = (rate + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, middle - half), min(1.0, middle + half)


def working_days_after(day: date, horizon: int) -> date:
    return np.busday_offset(np.datetime64(day), int(horizon), roll="backward").astype("datetime64[D]").astype(object)


def independent_call_count(anchors: Sequence[date], horizon: int) -> int:
    """Calls whose outcome windows don't overlap. Twenty 20-day calls a day apart are one test, not twenty."""
    count, free_from = 0, None
    for day in sorted(set(anchors)):
        if free_from is None or day >= free_from:
            count += 1
            free_from = working_days_after(day, max(int(horizon), 1))
    return count


def spot_path(series: pd.DataFrame) -> dict:
    """Daily day-ahead base price, with the usual weekday pattern removed for scoring."""
    if series is None or series.empty:
        return {"kind": "spot", "values": {}, "raw": pd.Series(dtype=float)}
    daily = series.assign(day=series["local"].dt.date).groupby("day")["value"].mean()
    weekdays = np.array([d.weekday() for d in daily.index])
    overall = float(daily.mean())
    effect = {w: float(daily.to_numpy()[weekdays == w].mean()) - overall for w in set(weekdays.tolist())}
    adjusted = daily.to_numpy() - np.array([effect[w] for w in weekdays])
    values = dict(zip(daily.index, adjusted.tolist()))
    return {"kind": "spot", "values": values, "raw": daily}


def marks_path(history: pd.DataFrame, product: str) -> dict:
    """Saved real marks for one product. Demo and synthetic prices never count as outcomes."""
    table: dict = {}
    if history is not None and not history.empty:
        rows = history.loc[history["product"].eq(product) & ~history["source"].isin(REAL_MARK_EXCLUSIONS)]
        for day, month, price in zip(rows["snapshot_date"], rows["delivery_month"], rows["price"]):
            table.setdefault(_to_date(day), {})[_to_date(month)] = float(price)
    return {"kind": "marks", "table": table, "days": sorted(table)}


def path_move(path: dict, anchor: date, horizon: int) -> Optional[float]:
    """How far the price moved after a read, using only prices later than it."""
    target = working_days_after(anchor, horizon)
    if path["kind"] == "spot":
        # Against the price when the call was made. An average of past days would flatter
        # any call that follows a trend, even in a market with no trend to find.
        later, base = path["values"].get(target), path["values"].get(anchor)
        return None if later is None or base is None else later - base
    days = path["days"]
    start = next((d for d in reversed(days) if d <= anchor), None)
    end = next((d for d in days if d >= target), None)
    if start is None or end is None or end <= start or (anchor - start).days > 3 or (end - target).days > 3:
        return None
    # The same delivery month at both dates, so a contract rolling never counts as a move.
    first_month = _add_months(date(end.year, end.month, 1), 1)
    months = sorted(m for m in path["table"][start] if m >= first_month and m in path["table"][end])
    return path["table"][end][months[0]] - path["table"][start][months[0]] if months else None


def path_anchors(path: dict) -> list[date]:
    return list(path["values"]) if path["kind"] == "spot" else list(path["days"])


def up_share(path: dict, horizon: int, first: Optional[date] = None, last: Optional[date] = None) -> float:
    """How often the price rose over this horizon on any day, whatever the read said."""
    moves = [path_move(path, day, horizon) for day in path_anchors(path) if (first is None or day >= first) and (last is None or day <= last)]
    moves = [m for m in moves if m]
    return sum(1 for m in moves if m > 0) / len(moves) if moves else 0.5


def score_calls(calls: pd.DataFrame, path: dict, horizon: int) -> dict:
    """calls has anchor and direction (+1 or -1). The result compares them with always guessing the same way."""
    outcomes = [(a, d, path_move(path, a, horizon)) for a, d in zip(calls["anchor"], calls["direction"])] if not calls.empty else []
    outcomes = [row for row in outcomes if row[2]]
    result = {"calls": len(outcomes), "independent": 0, "right": None, "chance": None, "low": None, "high": None, "average_move": None, "verdict": "No results yet"}
    if not outcomes:
        return result
    anchors = [row[0] for row in outcomes]
    hits = sum(1 for _, d, move in outcomes if d * move > 0)
    independent = independent_call_count(anchors, horizon)
    rate = hits / len(outcomes)
    rising = up_share(path, horizon, min(anchors), max(anchors))
    bullish = sum(1 for _, d, _m in outcomes if d > 0)
    chance = (bullish * rising + (len(outcomes) - bullish) * (1.0 - rising)) / len(outcomes)
    low, high = wilson_interval(rate * independent, independent)
    verdict = ("Too few to judge" if independent < TRACK_MIN_INDEPENDENT else "Better than chance" if low > chance else
               "Worse than chance" if high < chance else "No clear edge")
    result.update({"independent": independent, "right": rate, "chance": chance, "low": low, "high": high,
                   "average_move": float(np.mean([d * move for _, d, move in outcomes])), "verdict": verdict})
    return result


def _direction(score: float, threshold: float) -> int:
    return 1 if score >= threshold else -1 if score <= -threshold else 0


def track_tables(reads: pd.DataFrame, paths: dict) -> dict:
    """Score every saved read, overall, by confidence and by signal."""
    overall, by_confidence, by_signal = [], [], []
    if reads is None or reads.empty:
        return {"overall": pd.DataFrame(), "confidence": pd.DataFrame(), "signals": pd.DataFrame()}
    data = reads.copy()
    data["anchor"] = data["price_through"].map(date.fromisoformat)
    # One read per price date: reopening the app before new prices arrive isn't a new call.
    data = data.sort_values("read_date").drop_duplicates(["market", "source", "anchor"], keep="last")
    for (market, source), group in data.groupby(["market", "source"], sort=False):
        path = paths.get(market)
        if not path or not path_anchors(path):
            continue
        calls = group.assign(direction=group["score"].map(lambda s: _direction(float(s), CALL_THRESHOLD)))
        directional = calls.loc[calls["direction"].ne(0)]
        decoded = [(a, json.loads(text)) for a, text in zip(group["anchor"], group["signals"])]
        names = {key: value[1] for _, signals in decoded for key, value in signals.items()}
        for horizon in TRACK_HORIZONS:
            overall.append({"market": market, "source": source, "horizon": horizon, "balanced": int(calls["direction"].eq(0).sum()), **score_calls(directional, path, horizon)})
            for level, part in directional.groupby("confidence"):
                by_confidence.append({"market": market, "source": source, "horizon": horizon, "confidence": level, **score_calls(part, path, horizon)})
            for key, name in names.items():
                rows = [(a, _direction(signals[key][0], SIGNAL_THRESHOLD)) for a, signals in decoded if key in signals]
                frame = pd.DataFrame([r for r in rows if r[1]], columns=["anchor", "direction"])
                by_signal.append({"market": market, "source": source, "horizon": horizon, "signal": name, **score_calls(frame, path, horizon)})
    return {"overall": pd.DataFrame(overall), "confidence": pd.DataFrame(by_confidence), "signals": pd.DataFrame(by_signal)}


def replay_power_reads(market: str, zone: str, prices: pd.DataFrame, load: pd.DataFrame, generation: pd.DataFrame, skip: set, today: date) -> list[dict]:
    """Rebuild past reads using only what was known at the time. Forecast and curve signals are left out."""
    if prices is None or prices.empty or "timestamp" not in prices.columns:
        return []
    tz = ENTSOE_ZONES[zone]["timezone"]
    # Clean and parse everything once; each day then only slices it.
    prices = spot_series(prices, tz)
    if prices.empty:
        return []
    stamps = prices["timestamp"]
    days = sorted(set(prices["local"].dt.date))
    if len(days) < 30:
        return []
    frames = []
    for index, frame in enumerate((load, generation)):
        held = frame is not None and not frame.empty and "timestamp" in frame.columns
        if not held:
            frames.append((None, None))
            continue
        frame = frame.assign(timestamp=pd.to_datetime(frame["timestamp"], errors="coerce", utc=True), value=_number_series(frame["value"])).dropna(subset=["timestamp", "value"])
        if index == 1 and "series" in frame.columns:
            # Hourly means per series give the same signals as raw quarter-hours, at a fraction of the rows.
            frame = frame.assign(timestamp=frame["timestamp"].dt.floor("h")).groupby(["timestamp", "series"], as_index=False)["value"].mean()
        frames.append((frame, frame["timestamp"]))
    rows = []
    for day in days[28:-1]:
        if day.weekday() >= 5 or (day.isoformat(), market) in skip or day >= today:
            continue
        end = pd.Timestamp(day + timedelta(days=1), tz=tz).tz_convert("UTC")
        begin = end - pd.Timedelta(days=REPLAY_LOOKBACK_DAYS)
        known_until = end - pd.Timedelta(days=1)   # actual load and output for this day aren't known yet
        window = [frame.loc[(times >= begin) & (times < known_until)] if frame is not None else None for frame, times in frames]
        signals = power_signals(zone, prices.loc[(stamps >= begin) & (stamps < end)], window[0], window[1], None, None, {}, day)
        summary = summarise_read(signals)
        if summary["confidence"] != "None":
            rows.append(read_row(day, market, signals, summary, day, "Replayed"))
    return rows


def replay_gas_reads(history: pd.DataFrame, alsi: pd.DataFrame, days: Sequence[date], skip: set) -> list[dict]:
    """Past gas reads on the days with real TTF marks to score them against."""
    if history is None or history.empty:
        return []
    date_col = _find_column(history, ("observation_date", "gasDayStart", "gas_day"))
    if not date_col:
        return []
    stamps = pd.to_datetime(history[date_col], errors="coerce")
    alsi_col = _find_column(alsi, ("observation_date", "gasDayStart", "gas_day")) if alsi is not None and not alsi.empty else None
    alsi_stamps = pd.to_datetime(alsi[alsi_col], errors="coerce") if alsi_col else None
    rows = []
    for day in days:
        if (day.isoformat(), "European gas") in skip:
            continue
        # A gas day is published the day after it ends, so the read on day t sees up to t-1.
        cut = pd.Timestamp(day)
        lng = alsi.loc[alsi_stamps < cut] if alsi_col else None
        signals = gas_signals(history.loc[stamps < cut], lng, None, None, {}, {}, day)
        summary = summarise_read(signals)
        if summary["confidence"] != "None":
            rows.append(read_row(day, "European gas", signals, summary, day, "Replayed"))
    return rows


# -----------------------------------------------------------------------------
# Winter storage projection: where EU storage ends March under different weather
# -----------------------------------------------------------------------------
WINTER_MONTHS = (10, 11, 12, 1, 2, 3)
OUTLOOK_MIN_WINTERS = 2
OUTLOOK_MIN_WEATHER_YEARS = 3
BIAS_DAYS = 14
BIAS_HALF_LIFE_DAYS = 30.0


def winter_of(day: date) -> int:
    return day.year if day.month >= 7 else day.year - 1


def winter_label(year: int) -> str:
    return f"{year}-{(year + 1) % 100:02d}"


def _in_other_winter(day: date, year: int, winter: int) -> date:
    """The same calendar day in another winter. 29 February falls back to the 28th."""
    target = day.year + (winter - year)
    return date(target, day.month, 28 if day.month == 2 and day.day == 29 else day.day)


def daily_fill(history: pd.DataFrame) -> pd.Series:
    """Fill level by gas day, with gaps of up to three days bridged."""
    daily = _storage_daily(history)
    if daily.empty:
        return pd.Series(dtype=float)
    series = daily.set_index("day")["fill"].sort_index()
    return series.asfreq("D").interpolate(limit=3)


def _demand_weighted(frame: pd.DataFrame) -> pd.Series:
    """Heating degree days by date, weighted by each country's gas use. A missing city drops out that day."""
    weights = frame["location"].map(GAS_DEMAND_WEIGHTS).fillna(1.0)
    return (frame["hdd"] * weights).groupby(frame["date"]).sum() / weights.groupby(frame["date"]).sum()


def archive_hdd(archive: pd.DataFrame, locations: Sequence[str] = EU_STORAGE_LOCATIONS) -> pd.Series:
    """Heating degree days across EU reference cities, weighted by each country's gas use."""
    if archive is None or archive.empty or not {"date", "location", "hdd"}.issubset(archive.columns):
        return pd.Series(dtype=float)
    part = archive.loc[archive["location"].isin(locations)]
    if part.empty:
        return pd.Series(dtype=float)
    out = pd.DataFrame({"date": pd.to_datetime(part["date"], errors="coerce"), "location": part["location"], "hdd": pd.to_numeric(part["hdd"], errors="coerce")}).dropna()
    return _demand_weighted(out).sort_index()


def _weather_terms(days: pd.DatetimeIndex, hdd: np.ndarray) -> np.ndarray:
    """Month of winter, heating demand and weekend. Fill level is added separately."""
    months = np.array([days.month == m for m in WINTER_MONTHS], dtype=float).T
    return np.column_stack([months, np.asarray(hdd, dtype=float), (days.dayofweek >= 5).astype(float)])


def fit_storage_model(fill: pd.Series, hdd: pd.Series, exclude: Optional[int] = None, until: Optional[pd.Timestamp] = None) -> Optional[dict]:
    """Daily fill change = month + heating demand + weekend + fill level, on winter days.

    The fill term lets withdrawals slow as storage empties. Older winters count half.
    """
    frame = pd.DataFrame({"change": fill.diff(), "before": fill.shift(1)}).join(hdd.rename("hdd"), how="inner").dropna()
    frame = frame.loc[frame.index.month.isin(WINTER_MONTHS)]
    if until is not None:
        frame = frame.loc[frame.index <= until]
    winters = np.array([winter_of(d) for d in frame.index.date]) if not frame.empty else np.array([])
    if exclude is not None and len(frame):
        keep = winters != exclude
        frame, winters = frame.loc[keep], winters[keep]
    full = [w for w in sorted(set(winters.tolist())) if (winters == w).sum() >= 100]
    if len(full) < OUTLOOK_MIN_WINTERS:
        return None
    weather = _weather_terms(frame.index, frame["hdd"].to_numpy())
    design = np.column_stack([weather, frame["before"].to_numpy() - 60.0])
    weights = np.where(winters >= max(full) - 2, 1.0, 0.5)
    root = np.sqrt(weights)
    target = frame["change"].to_numpy()
    beta = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
    # Fuller storage can draw faster, but emptier storage must not draw faster still. That would feed on itself.
    if beta[8] > 0:
        beta = np.append(np.linalg.lstsq(weather * root[:, None], target * root, rcond=None)[0], 0.0)
    fitted = design @ beta
    spread = float(np.sum((target - target.mean()) ** 2))
    return {"beta": beta, "n": int(len(frame)), "winters": full, "hdd_effect": float(beta[6]), "weekend_effect": float(beta[7]), "fill_effect": float(beta[8]),
            "r2": 1.0 - float(np.sum((target - fitted) ** 2)) / spread if spread > 0 else 0.0,
            "low": float(target.min()), "high": float(target.max())}


def recent_model_bias(model: dict, fill: pd.Series, hdd: pd.Series, start: pd.Timestamp) -> float:
    """How far storage has recently run above or below the model, for example from strong LNG supply."""
    frame = pd.DataFrame({"change": fill.diff(), "before": fill.shift(1)}).join(hdd.rename("hdd"), how="inner").dropna()
    frame = frame.loc[(frame.index <= start) & (frame.index > start - pd.Timedelta(days=BIAS_DAYS)) & frame.index.month.isin(WINTER_MONTHS)]
    if len(frame) < 7:
        return 0.0
    design = np.column_stack([_weather_terms(frame.index, frame["hdd"].to_numpy()), frame["before"].to_numpy() - 60.0])
    return float((frame["change"].to_numpy() - design @ model["beta"]).mean())


def simulate_fill(model: dict, start_fill: float, days: pd.DatetimeIndex, hdd: Sequence[float], bias: float = 0.0) -> np.ndarray:
    """Step forward one day at a time, so each day's withdrawal depends on what's left."""
    base = _weather_terms(days, np.asarray(hdd, dtype=float)) @ model["beta"][:8]
    effect = float(model["beta"][8])
    path = np.empty(len(days))
    level = float(start_fill)
    for i, value in enumerate(base):
        step = value + effect * (level - 60.0) + bias * 0.5 ** ((i + 1) / BIAS_HALF_LIFE_DAYS)
        level = float(np.clip(level + np.clip(step, model["low"], model["high"]), 0.0, 100.0))
        path[i] = level
    return path


def _fill_on(fill: pd.Series, day: date) -> Optional[float]:
    value = fill.get(pd.Timestamp(day))
    return float(value) if value is not None and pd.notna(value) else None


def past_winter_changes(fill: pd.Series, start: date, year: int) -> dict:
    """Change in fill from this calendar day to 31 March, for each finished winter."""
    changes = {}
    for winter in range(year - 12, year):
        begin, end = _fill_on(fill, _in_other_winter(start, year, winter)), _fill_on(fill, date(winter + 1, 3, 31))
        if begin is not None and end is not None:
            changes[winter] = end - begin
    return changes


def backtest_outlook(fill: pd.Series, hdd: pd.Series, start: date, year: int) -> pd.DataFrame:
    """Test the method on each past winter, using a model that never saw that winter."""
    columns = ["Winter", "Fill at this date", "Actual end", "Model end", "Model miss", "Average-winter end", "Average-winter miss"]
    changes = past_winter_changes(fill, start, year)
    rows = []
    for winter in sorted(changes):
        model = fit_storage_model(fill, hdd, exclude=winter, until=pd.Timestamp(start))
        others = [v for w, v in changes.items() if w != winter]
        if model is None or len(others) < 2:
            continue
        begin = _in_other_winter(start, year, winter)
        days = pd.date_range(begin + timedelta(days=1), date(winter + 1, 3, 31), freq="D")
        weather = hdd.reindex(days)
        if weather.isna().mean() > 0.02:
            continue
        begin_fill = _fill_on(fill, begin)
        bias = recent_model_bias(model, fill, hdd, pd.Timestamp(begin))
        modelled = float(simulate_fill(model, begin_fill, days, weather.ffill().bfill().to_numpy(), bias)[-1])
        actual = begin_fill + changes[winter]
        average = begin_fill + float(np.mean(others))
        rows.append({"Winter": winter_label(winter), "Fill at this date": begin_fill, "Actual end": actual, "Model end": modelled,
                     "Model miss": modelled - actual, "Average-winter end": average, "Average-winter miss": average - actual})
    return pd.DataFrame(rows, columns=columns)


def winter_band(fill: pd.Series, year: int) -> pd.DataFrame:
    """The five previous winters on this winter's dates."""
    axis = pd.date_range(date(year, 10, 1), date(year + 1, 3, 31), freq="D")
    columns = {}
    for winter in range(year - 5, year):
        columns[winter] = [_fill_on(fill, _in_other_winter(day.date(), year, winter)) for day in axis]
    table = pd.DataFrame(columns, index=axis).astype(float)
    usable = table.notna().sum(axis=1) >= 3
    return pd.DataFrame({"day": axis, "low": table.min(axis=1), "high": table.max(axis=1), "average": table.mean(axis=1)}).loc[usable].reset_index(drop=True)


def forecast_hdd(daily: pd.DataFrame, locations: Sequence[str] = EU_STORAGE_LOCATIONS) -> pd.Series:
    if daily is None or daily.empty or not {"date", "location", "forecast_period", "heating_degree_day_proxy_18c"}.issubset(daily.columns):
        return pd.Series(dtype=float)
    part = daily.loc[daily["forecast_period"].eq("Forecast") & daily["location"].isin(locations)]
    frame = pd.DataFrame({"date": pd.to_datetime(part["date"], errors="coerce"), "location": part["location"], "hdd": pd.to_numeric(part["heating_degree_day_proxy_18c"], errors="coerce")}).dropna()
    return _demand_weighted(frame).sort_index() if not frame.empty else pd.Series(dtype=float)


def project_winter(history: pd.DataFrame, archive: pd.DataFrame, daily_weather: pd.DataFrame, today: date) -> dict:
    """Project EU storage to 31 March under cold, normal and mild weather."""
    if today.month in (4, 5, 6, 7, 8, 9):
        return {"ready": False, "off_season": True, "reason": "The winter projection runs from October to March. It starts again on 1 October."}
    fill = daily_fill(history)
    if fill.dropna().empty:
        return {"ready": False, "off_season": False, "reason": "Select Update market data to load storage history."}
    fill = fill.dropna()
    fill = fill.asfreq("D")
    start_stamp = fill.dropna().index[-1]
    start = start_stamp.date()
    if (today - start).days > STALE_DAYS:
        return {"ready": False, "off_season": False, "reason": f"Storage data is {(today - start).days} days old. Select Update market data."}
    year = winter_of(today)
    end = date(year + 1, 3, 31)
    days = pd.date_range(start + timedelta(days=1), end, freq="D")
    if days.empty:
        return {"ready": False, "off_season": False, "reason": "This winter has finished. The next projection starts on 1 October."}
    start_fill = float(fill.loc[start_stamp])
    band = winter_band(fill, year)
    band_end = band.loc[band["day"].eq(pd.Timestamp(end))].iloc[0].to_dict() if band["day"].eq(pd.Timestamp(end)).any() else None
    actual = fill.loc[pd.Timestamp(year, 10, 1):start_stamp].dropna()
    result = {"ready": True, "off_season": False, "reason": "", "start": start, "start_fill": start_fill, "end": end, "year": year, "band": band, "band_end": band_end,
              "actual": actual, "model": None, "backtest": pd.DataFrame(), "error": None, "average_error": None, "forecast_days": 0, "bias": 0.0, "years": pd.DataFrame()}
    hdd = archive_hdd(archive)
    model = fit_storage_model(fill, hdd) if not hdd.empty else None
    weather_years = {}
    if model is not None and model["hdd_effect"] < 0:
        ahead = forecast_hdd(daily_weather)
        for winter in range(year - 12, year):
            mapped = pd.DatetimeIndex([_in_other_winter(d.date(), year, winter) for d in days])
            values = hdd.reindex(mapped)
            if values.notna().all():
                series = pd.Series(values.to_numpy(), index=days)
                known = ahead.reindex(days).dropna()
                series.loc[known.index] = known.to_numpy()
                weather_years[winter] = series
    if len(weather_years) >= OUTLOOK_MIN_WEATHER_YEARS:
        totals = {w: float(s.sum()) for w, s in weather_years.items()}
        bias = recent_model_bias(model, fill, hdd, start_stamp)
        coldest, mildest = max(totals, key=totals.get), min(totals, key=totals.get)
        normal = pd.concat(weather_years.values(), axis=1).mean(axis=1)
        def path(weather: pd.Series) -> np.ndarray:
            return simulate_fill(model, start_fill, days, weather.to_numpy(), bias)
        ends = {w: float(path(s)[-1]) for w, s in weather_years.items()}
        result["end_without_recent_gap"] = float(simulate_fill(model, start_fill, days, normal.to_numpy(), 0.0)[-1])
        result["scenarios"] = [
            {"name": "Cold", "note": f"Weather like {winter_label(coldest)}, the coldest of the {len(weather_years)} past winters", "path": path(weather_years[coldest])},
            {"name": "Normal", "note": f"Average weather of the {len(weather_years)} past winters", "path": path(normal)},
            {"name": "Mild", "note": f"Weather like {winter_label(mildest)}, the mildest of the {len(weather_years)} past winters", "path": path(weather_years[mildest])}]
        result["days"] = days
        result["central"] = 1
        result["basis"] = "weather"
        result["model"] = model
        result["bias"] = bias
        result["forecast_days"] = int(ahead.reindex(days).notna().sum())
        result["years"] = pd.DataFrame([{"Weather of winter": winter_label(w), "Heating degree days ahead": totals[w], "March end, %": ends[w]}
                                        for w in sorted(ends, key=lambda w: -totals[w])])
    else:
        changes = past_winter_changes(fill, start, year)
        if len(changes) < OUTLOOK_MIN_WINTERS:
            return {"ready": False, "off_season": False, "reason": "Select Update market data to load six years of storage history and the weather archive."}
        biggest, smallest = min(changes, key=changes.get), max(changes, key=changes.get)
        def analog(winter: int) -> np.ndarray:
            mapped = pd.DatetimeIndex([_in_other_winter(d.date(), year, winter) for d in days])
            base = _fill_on(fill, _in_other_winter(start, year, winter))
            return np.clip(start_fill + fill.reindex(mapped).ffill().bfill().to_numpy() - base, 0.0, 100.0)
        average = np.mean([analog(w) for w in changes], axis=0)
        result["scenarios"] = [
            {"name": "Largest draw", "note": f"Storage falls as it did in {winter_label(biggest)}", "path": analog(biggest)},
            {"name": "Average draw", "note": f"Storage falls as it did on average over {len(changes)} past winters", "path": average},
            {"name": "Smallest draw", "note": f"Storage falls as it did in {winter_label(smallest)}", "path": analog(smallest)}]
        result["days"] = days
        result["central"] = 1
        result["basis"] = "past winters"
    for scenario in result["scenarios"]:
        scenario["end"] = float(scenario["path"][-1])
    past_levels = [level for level in (_fill_on(fill, _in_other_winter(start, year, winter)) for winter in range(year - 12, year)) if level is not None]
    result["lowest_past_level"] = min(past_levels) if past_levels else None
    if not hdd.empty:
        test = backtest_outlook(fill, hdd, start, year)
        if not test.empty:
            result["backtest"] = test
            result["error"] = float(test["Model miss"].abs().mean())
            result["average_error"] = float(test["Average-winter miss"].abs().mean())
    return result


def versus_range(level: float, band_end: Optional[dict]) -> str:
    """Compare a March level with the five-year range, in words."""
    if not band_end:
        return ""
    if level < band_end["low"] - 0.5:
        return f"{band_end['low'] - level:.0f} points below the five-year range"
    if level > band_end["high"] + 0.5:
        return f"{level - band_end['high']:.0f} points above the five-year range"
    gap = level - band_end["average"]
    if abs(gap) < 0.5:
        return "in line with the five-year average"
    return f"{abs(gap):.0f} points {'above' if gap > 0 else 'below'} the five-year average"


def outlook_signal(outlook: Optional[dict]) -> Optional[dict]:
    """The projection as forward-looking evidence for gas. Absent outside the season."""
    name = "Winter storage outlook"
    if not outlook or outlook.get("off_season"):
        return None
    if not outlook["ready"]:
        return _signal("winter_outlook", name, 2.0, missing=outlook["reason"])
    band = outlook.get("band_end")
    if not band or outlook.get("basis") != "weather":
        return _signal("winter_outlook", name, 2.0, missing="Select Update market data to load the weather archive and the five-year range.")
    central = outlook["scenarios"][outlook["central"]]["end"]
    gap = central - band["average"]
    return _signal("winter_outlook", name, 2.0, -gap / max((band["high"] - band["low"]) / 2.0, 5.0),
                   f"Normal weather ends March at {central:.0f}%, against a five-year average of {band['average']:.0f}%.",
                   "A low end to winter means more gas to buy back next summer, and a high end means less.",
                   f"Reverses if the projected end moves {'below' if gap >= 0 else 'above'} {band['average']:.0f}%.")


# -----------------------------------------------------------------------------
# Next-day power price estimate
# -----------------------------------------------------------------------------
ESTIMATE_TRAIN_DAYS = 365
ESTIMATE_HALF_LIFE_DAYS = 120          # recent days count more: a day 120 days back weighs half as much
ESTIMATE_MIN_DAYS = 60
ESTIMATE_TEST_DAYS = 180
ESTIMATE_REFIT_EVERY = 5
ESTIMATE_RANGE = (0.1, 0.9)            # an 80% range
ESTIMATE_RANGE_HALF_LIFE_DAYS = 10     # how fast the range follows recent accuracy
ESTIMATE_RANGE_MIN_ERRORS = 30
CALENDAR_COLUMNS = ("same", "work", "lag1", "avg7", "lag7", "saturday", "sunday", "after_off")
WEATHER_COLUMNS = ("hdd", "wind", "solar", "hdd_before", "wind_before", "solar_before")
ESTIMATE_COLUMNS = CALENDAR_COLUMNS + WEATHER_COLUMNS
FORECAST_COLUMNS = CALENDAR_COLUMNS + ("load_gw", "wind_gw", "solar_gw")
# Points spread across each zone, latitude and longitude. Wind in one city says little about a whole
# country's wind output: on real German prices and forecasts, a dozen points cut the average miss by about a sixth.
ZONE_WEATHER_POINTS = {
    "France": ((50.6, 2.8), (49.4, 0.6), (48.86, 2.35), (49.0, 4.2), (48.6, 6.2), (48.1, -2.8), (47.3, 5.0),
               (46.6, -0.3), (45.76, 4.84), (45.8, 2.6), (43.9, 1.4), (43.6, 4.1)),
    "Germany-Luxembourg": ((54.4, 9.1), (53.8, 11.6), (53.1, 7.9), (52.8, 10.4), (52.5, 13.4), (51.9, 7.6), (51.7, 11.8),
                           (51.3, 14.2), (50.6, 9.6), (50.0, 7.0), (49.7, 11.3), (48.5, 9.6), (48.3, 12.2)),
}


def _easter(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    l_ = (32 + 2 * e + 2 * (c // 4) - h - c % 4) % 7
    m = (a + 11 * h + 22 * l_) // 451
    return date(year, (h + l_ - 7 * m + 114) // 31, (h + l_ - 7 * m + 114) % 31 + 1)


def public_holidays(zone: str, year: int) -> set:
    """National public holidays that move power demand, by zone. Regional holidays are left out."""
    e = _easter(year)
    day = lambda m, d: date(year, m, d)
    monday = lambda m, nth: [day(m, 1) + timedelta(days=i) for i in range(31) if (day(m, 1) + timedelta(days=i)).month == m and (day(m, 1) + timedelta(days=i)).weekday() == 0][nth]
    if zone == "France":
        return {day(1, 1), e + timedelta(1), day(5, 1), day(5, 8), e + timedelta(39), e + timedelta(50), day(7, 14), day(8, 15), day(11, 1), day(11, 11), day(12, 25)}
    if zone == "Germany-Luxembourg":
        return {day(1, 1), e - timedelta(2), e + timedelta(1), day(5, 1), e + timedelta(39), e + timedelta(50), day(10, 3), day(12, 25), day(12, 26)}
    if zone == "Netherlands":
        kings = day(4, 27) if day(4, 27).weekday() != 6 else day(4, 26)
        return {day(1, 1), e + timedelta(1), kings, e + timedelta(39), e + timedelta(50), day(12, 25), day(12, 26)}
    if zone == "Belgium":
        return {day(1, 1), e + timedelta(1), day(5, 1), e + timedelta(39), e + timedelta(50), day(7, 21), day(8, 15), day(11, 1), day(11, 11), day(12, 25)}
    if zone == "Spain":
        return {day(1, 1), day(1, 6), e - timedelta(2), day(5, 1), day(8, 15), day(10, 12), day(11, 1), day(12, 6), day(12, 8), day(12, 25)}
    if zone == "Great Britain":
        new_year = day(1, 1) + timedelta(days={5: 2, 6: 1}.get(day(1, 1).weekday(), 0))
        christmas = {5: (day(12, 27), day(12, 28)), 6: (day(12, 27), day(12, 26))}.get(day(12, 25).weekday(), (day(12, 25), day(12, 26) + timedelta(days=2 if day(12, 26).weekday() == 5 else 0)))
        return {new_year, e - timedelta(2), e + timedelta(1), monday(5, 0), monday(5, -1), monday(8, -1), *christmas}
    return set()


def day_classes(days: pd.DatetimeIndex, zone: str) -> np.ndarray:
    """0 for a working day, 1 for a Saturday, 2 for a Sunday or public holiday."""
    holidays = set().union(*(public_holidays(zone, y) for y in range(days.year.min(), days.year.max() + 1))) if len(days) else set()
    off = np.fromiter((d in holidays for d in days.date), bool, len(days)) | (days.dayofweek == 6)
    return np.where(off, 2, np.where(days.dayofweek == 5, 1, 0))


def daily_base_prices(series: pd.DataFrame) -> pd.Series:
    """Average day-ahead price by delivery day, leaving out days with few published prices."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    data = series.assign(day=pd.to_datetime(series["local"].dt.date), weighted=series["value"] * series["hours"])
    grouped = data.groupby("day")
    hours = grouped["hours"].sum()
    # Judged by hours covered, so a change of resolution can't hide a whole period.
    return (grouped["weighted"].sum() / hours).loc[hours >= 19.0].sort_index()


def power_weather(history: pd.DataFrame, daily: pd.DataFrame, location: str) -> pd.DataFrame:
    """Heating demand, wind and solar by day for one city. Recent and forecast days replace the archive."""
    names = {"temperature_mean_c": "temperature", "wind_speed_100m_mean_kmh": "wind", "solar_radiation_mean_wm2": "solar"}
    frames = []
    for frame in (history, daily):
        if frame is not None and not frame.empty and {"date", "location", *names}.issubset(frame.columns):
            part = frame.loc[frame["location"].eq(location), ["date", *names]].rename(columns=names)
            part = part.assign(date=pd.to_datetime(part["date"], errors="coerce")).dropna(subset=["date"])
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["hdd", "wind", "solar"])
    joined = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").set_index("date").sort_index().astype(float)
    return pd.DataFrame({"hdd": (18.0 - joined["temperature"]).clip(lower=0), "wind": joined["wind"], "solar": joined["solar"]})


def zone_weather(history: pd.DataFrame, forecast: pd.DataFrame, zone: str) -> pd.DataFrame:
    """Heating demand, wind and solar by day, averaged over the zone's points. Forecast days replace the archive."""
    frames = [f.loc[f["zone"].eq(zone), ["date", "hdd", "wind", "solar"]] for f in (history, forecast)
              if f is not None and not f.empty and {"zone", "date", "hdd", "wind", "solar"}.issubset(f.columns)]
    if not frames:
        return pd.DataFrame(columns=["hdd", "wind", "solar"])
    joined = pd.concat(frames, ignore_index=True).assign(date=lambda f: pd.to_datetime(f["date"], errors="coerce")).dropna(subset=["date"])
    return joined.drop_duplicates("date", keep="last").set_index("date").sort_index().astype(float)


def estimate_features(base: pd.Series, weather: pd.DataFrame, last_day: pd.Timestamp, zone: str = "") -> pd.DataFrame:
    """One row per delivery day. Prices come only from earlier days, as at auction time.

    same is the latest earlier day of the same kind (working day, Saturday, Sunday or holiday), and work the
    latest earlier working day, so a Monday starts from Friday rather than from a quiet Sunday.
    """
    days = pd.date_range(base.index.min(), last_day, freq="D")
    prices = base.reindex(days)
    values, classes = prices.to_numpy(float), day_classes(days, zone)
    same, work, latest = np.full(len(days), np.nan), np.full(len(days), np.nan), [np.nan, np.nan, np.nan]
    for i, kind in enumerate(classes):
        same[i], work[i] = latest[kind], latest[0]
        if not np.isnan(values[i]):
            latest[kind] = values[i]
    frame = pd.DataFrame({"base": prices, "same": same, "work": work, "lag1": prices.shift(1),
                          "avg7": prices.shift(1).rolling(7, min_periods=5).mean(), "lag7": prices.shift(7)}, index=days)
    frame["saturday"], frame["sunday"] = (classes == 1).astype(float), (classes == 2).astype(float)
    frame["after_off"] = ((classes == 0) & (np.r_[0, classes[:-1]] == 2)).astype(float)
    weather = weather.reindex(days) if weather is not None and not weather.empty else pd.DataFrame(index=days)
    for column in ("hdd", "wind", "solar"):
        frame[column] = weather[column] if column in weather.columns else np.nan
        # The day before's weather explains part of the day before's price, which the fit starts from.
        frame[f"{column}_before"] = frame[column].shift(1)
    return frame


def entsoe_forecast_days(load_forecast: pd.DataFrame, renewable_forecast: pd.DataFrame, tz: str) -> pd.DataFrame:
    """ENTSO-E's day-ahead forecasts by delivery day, in GW. A day needs at least 20 hours to count."""
    columns = ["load_gw", "wind_gw", "solar_gw"]
    needed = {"timestamp", "value"}
    if load_forecast is None or load_forecast.empty or renewable_forecast is None or renewable_forecast.empty or not needed.issubset(load_forecast.columns) or not {"series", *needed}.issubset(renewable_forecast.columns):
        return pd.DataFrame(columns=columns)

    def daily(frame: pd.DataFrame) -> pd.Series:
        hours = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.floor("h")
        parts = pd.DataFrame({"hour": hours, "series": frame["series"].astype(str) if "series" in frame.columns else "x", "mw": _number_series(frame["value"])}).dropna()
        if parts.empty:
            return pd.Series(dtype=float)
        total = parts.groupby(["hour", "series"], as_index=False)["mw"].mean().groupby("hour")["mw"].sum()
        days = pd.DatetimeIndex(total.index).tz_convert(tz).tz_localize(None).normalize()
        grouped = total.groupby(days)
        return (grouped.mean() / 1000.0).loc[grouped.count() >= 20]

    labels = renewable_forecast["series"].astype(str).str.casefold()
    wind, solar = daily(renewable_forecast.loc[labels.str.contains("wind")]), daily(renewable_forecast.loc[labels.str.contains("solar")])
    table_ = pd.DataFrame({"load_gw": daily(load_forecast), "wind_gw": wind, "solar_gw": solar})
    # A zone that publishes no solar or wind forecast at all has none to count.
    for column, series in (("wind_gw", wind), ("solar_gw", solar)):
        if series.empty:
            table_[column] = 0.0
    return table_


def _weighted_beta(features: np.ndarray, target: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Least squares where a day ESTIMATE_HALF_LIFE_DAYS older counts half as much."""
    root = np.sqrt(0.5 ** (ages / ESTIMATE_HALF_LIFE_DAYS))
    return np.linalg.lstsq(features * root[:, None], target * root, rcond=None)[0]


def _design(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame[list(columns)].to_numpy(float)])


def _fit_estimate(frame: pd.DataFrame, columns: Sequence[str] = ESTIMATE_COLUMNS, as_of: Optional[pd.Timestamp] = None) -> np.ndarray:
    as_of = as_of if as_of is not None else frame.index.max() + pd.Timedelta(days=1)
    kept = frame.loc[frame.index >= as_of - pd.Timedelta(days=ESTIMATE_TRAIN_DAYS)]
    return _weighted_beta(_design(kept, columns), kept["base"].to_numpy(float), (as_of - kept.index).days.to_numpy(float))


def _apply_estimate(beta: np.ndarray, frame: pd.DataFrame, columns: Sequence[str] = ESTIMATE_COLUMNS) -> np.ndarray:
    return _design(frame, columns) @ beta


def _walk_forward(known: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Estimate each past day from earlier days only, refitting regularly."""
    first = max(ESTIMATE_MIN_DAYS, len(known) - ESTIMATE_TEST_DAYS)
    features, target = _design(known, columns), known["base"].to_numpy(float)
    day_number = ((known.index - known.index[0]).days).to_numpy(float)
    estimates = np.empty(max(len(known) - first, 0))
    # Refit every few days on earlier days only, then apply that fit to the next block in one step.
    for block_start in range(first, len(known), ESTIMATE_REFIT_EVERY):
        train = np.flatnonzero((day_number < day_number[block_start]) & (day_number >= day_number[block_start] - ESTIMATE_TRAIN_DAYS))
        beta = _weighted_beta(features[train], target[train], day_number[block_start] - day_number[train])
        block = slice(block_start, min(block_start + ESTIMATE_REFIT_EVERY, len(known)))
        estimates[block.start - first:block.stop - first] = features[block] @ beta
    test = known.iloc[first:]
    return pd.DataFrame({"day": test.index, "actual": target[first:], "estimate": estimates, "same_type": test["same"].to_numpy(float),
                         "last_week": test["lag7"].to_numpy(float), "yesterday": test["lag1"].to_numpy(float)})


def estimate_range(errors: np.ndarray) -> dict:
    """An 80% range for the next day from past misses, widened or narrowed by how accurate recent days were.

    Misses are divided by the recent typical miss before taking quantiles, so a calm spell gives a narrower
    range and a volatile one a wider range. held is the share of tested days whose price landed inside the
    range built only from earlier days.
    """
    errors = np.asarray(errors, float)
    scale = pd.Series(np.abs(errors)).ewm(halflife=ESTIMATE_RANGE_HALF_LIFE_DAYS, min_periods=5).mean()
    before = scale.shift(1).to_numpy()
    scaled = errors / before
    usable = ~np.isnan(scaled)
    low_q, high_q = ESTIMATE_RANGE
    if usable.sum() < ESTIMATE_RANGE_MIN_ERRORS:
        return {"low": float(np.quantile(errors, low_q)), "high": float(np.quantile(errors, high_q)), "held": None, "held_days": 0, "adaptive": False}
    now = float(scale.iloc[-1])
    hits = []
    for i in np.flatnonzero(usable):
        past = scaled[:i][usable[:i]]
        if len(past) >= ESTIMATE_RANGE_MIN_ERRORS:
            low, high = np.quantile(past, [low_q, high_q]) * before[i]
            hits.append(low <= errors[i] <= high)
    low, high = np.quantile(scaled[usable], [low_q, high_q]) * now
    return {"low": float(low), "high": float(high), "held": float(np.mean(hits)) if hits else None, "held_days": len(hits), "adaptive": True}


def _weather_for_estimate(zone: str, history: pd.DataFrame, daily: pd.DataFrame, grid_history: Optional[pd.DataFrame], grid_forecast: Optional[pd.DataFrame],
                          target: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    """Weather across the zone's points when held, otherwise the one reference city."""
    location = ZONE_WEATHER.get(zone)
    options = []
    if zone in ZONE_WEATHER_POINTS:
        options.append((zone_weather(grid_history, grid_forecast, zone), f"{len(ZONE_WEATHER_POINTS[zone])} points across the zone"))
    if location:
        options.append((power_weather(history, daily, location), location.split(" / ")[0]))
    enough = [(w, name) for w, name in options if not w.empty and w["wind"].notna().sum() >= ESTIMATE_MIN_DAYS]
    ready = [(w, name) for w, name in enough if target in w.index and not w.loc[[target], ["hdd", "wind", "solar"]].isna().any(axis=None)]
    return (ready or enough or [(pd.DataFrame(columns=["hdd", "wind", "solar"]), "")])[0]


def next_day_estimate(zone: str, series: pd.DataFrame, history: pd.DataFrame, daily: pd.DataFrame,
                      load_forecast: Optional[pd.DataFrame] = None, renewable_forecast: Optional[pd.DataFrame] = None,
                      grid_history: Optional[pd.DataFrame] = None, grid_forecast: Optional[pd.DataFrame] = None) -> dict:
    """Estimate the next unpublished delivery day, and test the methods one day at a time.

    The weather method can run before the auction. ENTSO-E's wind and solar forecast is usually
    published after it, so that method mainly checks published prices, and sets the estimate only
    when the forecast for the target day already exists and it has tested at least as well.
    """
    base = daily_base_prices(series)
    if zone not in ZONE_WEATHER or len(base) < ESTIMATE_MIN_DAYS + 20:
        return {"ready": False, "reason": f"This needs at least {ESTIMATE_MIN_DAYS + 20} days of day-ahead prices. Select Update market data."}
    target = base.index[-1] + pd.Timedelta(days=1)
    weather, basis = _weather_for_estimate(zone, history, daily, grid_history, grid_forecast, target)
    if weather.empty:
        return {"ready": False, "reason": "This needs the wind and solar history. Select Update market data."}
    frame = estimate_features(base, weather, target, zone)
    known = frame.iloc[:-1].dropna(subset=["base", *ESTIMATE_COLUMNS])
    ahead = frame.iloc[[-1]]
    weather_ready = not ahead[list(ESTIMATE_COLUMNS)].isna().any(axis=None)
    if len(known) < ESTIMATE_MIN_DAYS + 10:
        return {"ready": False, "reason": "Prices and weather overlap on too few days so far."}
    test = _walk_forward(known, ESTIMATE_COLUMNS)
    final = _fit_estimate(known, ESTIMATE_COLUMNS, target)

    tz = ENTSOE_ZONES[zone]["timezone"]
    published = entsoe_forecast_days(load_forecast, renewable_forecast, tz)
    forecast_frame = frame[["base", *CALENDAR_COLUMNS]].join(published.reindex(frame.index)) if not published.empty else None
    forecast_known = forecast_frame.iloc[:-1].dropna() if forecast_frame is not None else pd.DataFrame()
    forecast_test, forecast_final = pd.DataFrame(), None
    if len(forecast_known) >= ESTIMATE_MIN_DAYS + 30:
        forecast_test = _walk_forward(forecast_known, FORECAST_COLUMNS)
        forecast_final = _fit_estimate(forecast_known, FORECAST_COLUMNS, target)
    comparison = None
    if not forecast_test.empty:
        shared = test.merge(forecast_test[["day", "estimate"]].rename(columns={"estimate": "forecast_estimate"}), on="day")
        if len(shared) >= 20:
            miss = lambda column: float((shared["actual"] - shared[column]).abs().mean())
            comparison = {"days": len(shared), "weather": miss("estimate"), "forecasts": miss("forecast_estimate"),
                          "same_type": miss("same_type"), "last_week": miss("last_week"), "yesterday": miss("yesterday")}
    target_has_forecast = forecast_final is not None and not forecast_frame.iloc[[-1]][list(FORECAST_COLUMNS)].isna().any(axis=None)
    use_forecasts = target_has_forecast and comparison is not None and comparison["forecasts"] <= comparison["weather"]
    if not use_forecasts and not weather_ready:
        return {"ready": False, "reason": f"There's no weather forecast for {target:%d %b}. Select Update market data."}
    chosen = forecast_test if use_forecasts else test
    errors = (chosen["actual"] - chosen["estimate"]).to_numpy(float)
    value = float(_apply_estimate(forecast_final, forecast_frame.iloc[[-1]], FORECAST_COLUMNS)[0]) if use_forecasts else float(_apply_estimate(final, ahead, ESTIMATE_COLUMNS)[0])
    spread = estimate_range(errors)
    # Judge the latest published price with ENTSO-E's forecasts when they exist for that day.
    checker = forecast_test if not forecast_test.empty and forecast_test["day"].iloc[-1] == base.index[-1] else test
    latest = checker.iloc[-1]
    gap = float(latest["actual"] - latest["estimate"])
    check_errors = (checker["actual"] - checker["estimate"]).abs()
    miss = lambda column: float((chosen["actual"] - chosen[column]).abs().mean())
    result = {"ready": True, "reason": "", "zone": zone, "day": target.date(), "estimate": value, "method": "ENTSO-E forecasts" if use_forecasts else "Weather",
              "weather_basis": basis, "low": value + spread["low"], "high": value + spread["high"], "held": spread["held"], "held_days": spread["held_days"],
              "adaptive_range": spread["adaptive"], "test": chosen, "n": len(chosen), "error": float(np.abs(errors).mean()),
              "error_same_type": miss("same_type"), "error_last_week": miss("last_week"), "error_yesterday": miss("yesterday"),
              "recent_error": float(np.abs(errors[-30:]).mean()),
              "latest_day": latest["day"].date(), "latest_actual": float(latest["actual"]), "latest_estimate": float(latest["estimate"]),
              "latest_gap": gap, "unusual": abs(gap) > float(check_errors.quantile(0.9)), "check_method": "ENTSO-E forecasts" if checker is forecast_test else "Weather",
              "comparison": comparison, "target_has_forecast": target_has_forecast, "forecast_days": int(published.dropna().shape[0]),
              "wind_effect": float(final[ESTIMATE_COLUMNS.index("wind") + 1]), "hdd_effect": float(final[ESTIMATE_COLUMNS.index("hdd") + 1]),
              "solar_effect": float(final[ESTIMATE_COLUMNS.index("solar") + 1]), "forecast_effects": None}
    best_simple = min(result["error_same_type"], result["error_last_week"], result["error_yesterday"])
    result["skill"] = 1.0 - result["error"] / best_simple if best_simple > 0 else 0.0
    if forecast_final is not None:
        result["forecast_effects"] = {"load": float(forecast_final[FORECAST_COLUMNS.index("load_gw") + 1]), "wind": float(forecast_final[FORECAST_COLUMNS.index("wind_gw") + 1]),
                                      "solar": float(forecast_final[FORECAST_COLUMNS.index("solar_gw") + 1])}
    return result


def score_recorded_estimates(log: pd.DataFrame, base: pd.Series) -> dict:
    """Estimates saved on the day, using real forecasts, against the prices that followed."""
    empty = {"n": 0, "error": None, "error_last_week": None, "held": None, "held_n": 0}
    if log is None or log.empty or base is None or base.empty:
        return empty
    days = pd.to_datetime(log["delivery_date"])
    actual = base.reindex(days).to_numpy(float)
    last_week = base.reindex(days - pd.Timedelta(days=7)).to_numpy(float)
    estimate = log["estimate"].to_numpy(float)
    both = ~np.isnan(actual) & ~np.isnan(last_week)
    if not both.any():
        return empty
    out = {"n": int(both.sum()), "error": float(np.abs(actual[both] - estimate[both]).mean()), "error_last_week": float(np.abs(actual[both] - last_week[both]).mean()), "held": None, "held_n": 0}
    if {"low", "high"}.issubset(log.columns):
        low, high = pd.to_numeric(log["low"], errors="coerce").to_numpy(float), pd.to_numeric(log["high"], errors="coerce").to_numpy(float)
        ranged = ~np.isnan(actual) & ~np.isnan(low) & ~np.isnan(high)
        if ranged.any():
            out["held"], out["held_n"] = float(np.mean((actual[ranged] >= low[ranged]) & (actual[ranged] <= high[ranged]))), int(ranged.sum())
    return out


# -----------------------------------------------------------------------------
# P&L drivers: which product and month made the day's result
# -----------------------------------------------------------------------------
def pnl_drivers(step: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    """Combine a period's P&L by product and delivery month, in EUR, largest first."""
    columns = ["product", "delivery_month", "net_mwh", "previous_mark", "current_mark", "mark_change", "price_move", "new_trades", "first_marks", "total"]
    if step is None or step.empty:
        return pd.DataFrame(columns=columns)
    data = step.assign(price_move=step["price_move"] * factor, new_trades=step["new_trade"] * factor, first_marks=step["first_mark"] * factor, total=step["total_pnl"] * factor)
    grouped = data.groupby(["product", "delivery_month"], as_index=False).agg(net_mwh=("net_mwh", "sum"), previous_mark=("previous_mark", "first"), current_mark=("current_mark", "first"),
                                                                              price_move=("price_move", "sum"), new_trades=("new_trades", "sum"), first_marks=("first_marks", "sum"), total=("total", "sum"))
    grouped["mark_change"] = grouped["current_mark"] - grouped["previous_mark"]
    return grouped.loc[grouped["total"].abs().argsort()[::-1], columns].reset_index(drop=True)


def pnl_sentence(drivers: pd.DataFrame) -> str:
    """The period's result in words, naming what moved it most."""
    if drivers is None or drivers.empty:
        return "There's no P&L to explain in this period."
    total, moves, new = float(drivers["total"].sum()), float(drivers["price_move"].sum()), float(drivers["new_trades"].sum())
    parts = [f"P&L was {fmt_money(total)}: {fmt_money(moves)} from price moves and {fmt_money(new)} from new trades."]
    top = drivers.loc[drivers["price_move"].abs().idxmax()]
    if abs(float(top["price_move"])) > 0.005:
        side = "long" if top["net_mwh"] > 0 else "short"
        parts.append(f"The largest price effect was {top['product']} {month_contract_label(top['delivery_month'])}: {side} {abs(top['net_mwh']):,.0f} MWh while the mark moved {top['mark_change']:+.2f}, worth {fmt_money(float(top['price_move']))}.")
    return " ".join(parts)


# -----------------------------------------------------------------------------
# Crisis replay and VaR check
# -----------------------------------------------------------------------------
CRISIS_EPISODES = (
    {"name": "Invasion spike, Feb to 7 Mar 2022", "gas": (100.0, 345.0), "eua": (98.5, 68.0),
     "what": "Front-month TTF went from about EUR 100/MWh, the top of its January and February range, to an intraday peak of EUR 345. EUA fell from about EUR 98.50/t to about EUR 68/t by late March.",
     "sources": ("Elenger, Q1 2022 gas market overview", "Cemnet, 24 March 2022")},
    {"name": "August squeeze, 16 to 26 Aug 2022", "gas": (225.0, 338.0), "eua": None,
     "what": "Front-month TTF went from about EUR 225/MWh to a record close of about EUR 338. No carbon move is applied.",
     "sources": ("European Gas Hub, 17 August 2022", "Hostile.eco, TTF record")},
    {"name": "Collapse, 26 Aug to Dec 2022", "gas": (338.0, 77.0), "eua": None,
     "what": "Front-month TTF fell from about EUR 338/MWh to about EUR 77 by late December. No carbon move is applied.",
     "sources": ("Hostile.eco, TTF record", "CNBC, 29 December 2022")},
)
CRISIS_GAS_FLOOR = 1.0
CRISIS_DECAY_MONTHS = 6.0


def _nearest_mark(lookup: dict, month: date) -> Optional[float]:
    if not lookup:
        return None
    return lookup.get(month, lookup[min(lookup, key=lambda m: abs((m - month).days))])


def crisis_replay(buckets: pd.DataFrame, curve_set: dict, first_month: date, same_percent: bool = True, far_share: float = 0.5,
                  eur_per_gbp: float = 1.17, params: Optional[dict] = None) -> pd.DataFrame:
    """Each episode's move applied to today's book. Power follows gas and carbon through a gas plant's costs."""
    columns = ["episode", "product", "delivery_month", "net_mwh", "price_change", "pnl_eur"]
    if buckets is None or buckets.empty:
        return pd.DataFrame(columns=columns)
    p = {**DEFAULT_SPREAD_PARAMS, **(params or {})}
    position = buckets.groupby(["product", "currency", "delivery_month"], as_index=False)["net_mwh"].sum()
    ttf = _curve_lookup(validate_curve(curve_set["TTF Natural Gas"]))
    eua = _curve_lookup(validate_curve(curve_set["EUA Carbon"])) if "EUA Carbon" in curve_set else {}
    nbp = _curve_lookup(validate_curve(curve_set["NBP Natural Gas"])) if "NBP Natural Gas" in curve_set else {}
    rows = []
    for episode in CRISIS_EPISODES:
        gas_start, gas_end = episode["gas"]
        for row in position.to_dict("records"):
            month = row["delivery_month"]
            distance = max((month.year - first_month.year) * 12 + month.month - first_month.month, 0)
            scale = far_share + (1.0 - far_share) * math.exp(-distance / CRISIS_DECAY_MONTHS)
            def gas_change(price: float, per_unit: float = 1.0) -> float:
                change = price * (gas_end / gas_start - 1.0) * scale if same_percent else (gas_end - gas_start) * scale * per_unit
                return max(change, CRISIS_GAS_FLOOR * per_unit - price)
            ttf_price = _nearest_mark(ttf, month) or 0.0
            if row["product"] == "TTF Natural Gas":
                change = gas_change(ttf_price)
            elif row["product"] == "NBP Natural Gas":
                change = gas_change(_nearest_mark(nbp, month) or ttf_price / eur_per_gbp, 1.0 / eur_per_gbp)
            else:
                carbon = 0.0
                if episode["eua"]:
                    carbon_price = _nearest_mark(eua, month) or 0.0
                    carbon = (carbon_price * (episode["eua"][1] / episode["eua"][0] - 1.0) if same_percent else episode["eua"][1] - episode["eua"][0]) * scale
                change = (gas_change(ttf_price) + carbon * p["gas_emission_factor"]) / p["gas_efficiency"]
            fx = float(eur_per_gbp) if row["currency"] == "GBP" else 1.0
            rows.append({"episode": episode["name"], "product": row["product"], "delivery_month": month, "net_mwh": float(row["net_mwh"]), "price_change": change, "pnl_eur": float(row["net_mwh"]) * change * fx})
    return pd.DataFrame(rows, columns=columns)


def worst_stretch(pnl: pd.DataFrame, days: int) -> Optional[dict]:
    """The worst run of consecutive saved days for today's book."""
    if pnl is None or len(pnl) < days:
        return None
    rolling = pd.Series(pnl["pnl_eur"].to_numpy(float)).rolling(days).sum()
    last = int(rolling.idxmin())
    return {"days": days, "pnl_eur": float(rolling.iloc[last]), "start": pnl["date"].iloc[last - days + 1], "end": pnl["date"].iloc[last]}


VAR_CHECK_WINDOW = 250
VAR_CHECK_MIN_HISTORY = 60
VAR_CHECK_MIN_TESTS = 100


def kupiec_test(days: int, exceptions: int, confidence: float) -> tuple[float, float]:
    """Kupiec's test: does the share of losses beyond VaR match what the confidence level promises?"""
    expected = 1.0 - float(confidence)
    def log_likelihood(share: float) -> float:
        total = 0.0
        if exceptions:
            total += exceptions * math.log(share)
        if days - exceptions:
            total += (days - exceptions) * math.log(1.0 - share)
        return total
    observed = exceptions / days if days else 0.0
    statistic = max(-2.0 * (log_likelihood(expected) - log_likelihood(observed if 0 < observed < 1 else expected)), 0.0) if 0 < exceptions < days else (
        -2.0 * days * math.log(1.0 - expected) if exceptions == 0 and days else -2.0 * days * math.log(expected) if days else 0.0)
    return statistic, math.erfc(math.sqrt(statistic / 2.0))


def var_backtest(pnl: pd.DataFrame, confidence: float = 0.99, window: int = VAR_CHECK_WINDOW, min_history: int = VAR_CHECK_MIN_HISTORY) -> dict:
    """Each day's VaR uses only earlier days. The day's result then either stays inside it or doesn't."""
    values = pnl["pnl_eur"].to_numpy(float) if pnl is not None and not pnl.empty else np.array([])
    empty = {"tests": 0, "exceptions": 0, "expected": 0.0, "statistic": 0.0, "p_value": 1.0, "verdict": "Not enough history", "table": pd.DataFrame(), "pairs": 0, "zone": ""}
    if len(values) <= min_history:
        return empty
    rows = []
    for i in range(min_history, len(values)):
        var = max(-float(np.quantile(values[max(0, i - window):i], 1.0 - confidence)), 0.0)
        rows.append({"date": pnl["date"].iloc[i], "pnl_eur": float(values[i]), "var_eur": var, "exception": bool(values[i] < -var)})
    table_ = pd.DataFrame(rows)
    tests, exceptions = len(table_), int(table_["exception"].sum())
    statistic, p_value = kupiec_test(tests, exceptions, confidence)
    if tests < VAR_CHECK_MIN_TESTS:
        verdict = "Too few days to judge"
    elif p_value >= 0.05:
        verdict = "Consistent with the confidence level"
    else:
        verdict = "VaR looks too low" if exceptions / tests > 1.0 - confidence else "VaR looks too high"
    flags = table_["exception"].to_numpy()
    zone = ""
    if abs(confidence - 0.99) < 1e-9 and tests >= 250:
        recent = int(flags[-250:].sum())
        zone = "Green" if recent <= 4 else "Yellow" if recent <= 9 else "Red"
    return {"tests": tests, "exceptions": exceptions, "expected": tests * (1.0 - confidence), "statistic": statistic, "p_value": p_value, "verdict": verdict,
            "table": table_, "pairs": int((flags[1:] & flags[:-1]).sum()), "zone": zone}


# -----------------------------------------------------------------------------
# Thresholds and a PDF writer that needs no extra package
# -----------------------------------------------------------------------------
def threshold_alerts(rules: pd.DataFrame, metrics: dict) -> list[dict]:
    """Rules whose condition holds now, and whether it's new since the previous value."""
    alerts = []
    if rules is None or rules.empty:
        return alerts
    for metric, condition, level in zip(rules["metric"], rules["condition"], rules["level"]):
        reading = metrics.get(metric)
        if not reading or reading.get("now") is None or pd.isna(level):
            continue
        now, before, level = reading["now"], reading.get("before"), float(level)
        check = (lambda v: v > level) if condition == "Above" else (lambda v: v < level)
        if check(now):
            alerts.append({"metric": metric, "condition": condition, "level": level, "now": now, "unit": reading.get("unit", ""),
                           "new": before is not None and not check(before)})
    return alerts


def _pdf_text(text: object) -> bytes:
    raw = str(text).replace("−", "-").replace("–", "-").replace("—", "-").encode("cp1252", "replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def simple_pdf(blocks: Sequence[tuple], title: str = "") -> bytes:
    """Build an A4 PDF from (kind, content) blocks: title, subtitle, heading, text, note, table."""
    width, height, margin = 595, 842, 48
    usable = width - 2 * margin
    pages: list[list[bytes]] = [[]]
    y = height - margin

    def line(text: object, size: float, bold: bool = False, x: float = margin, color: str = "0.10 0.17 0.20") -> None:
        pages[-1].append(b"BT /F%d %s Tf %s rg %s %s Td (" % (2 if bold else 1, str(size).encode(), color.encode(), str(x).encode(), f"{y:.1f}".encode()) + _pdf_text(text) + b") Tj ET\n")

    def room(needed: float) -> None:
        nonlocal y
        if y - needed < margin:
            pages.append([])
            y = height - margin

    def wrapped(text: str, size: float, bold: bool = False, color: str = "0.10 0.17 0.20", gap: float = 1.35) -> None:
        nonlocal y
        for piece in textwrap.wrap(str(text), max(int(usable / (size * 0.50)), 10)) or [""]:
            room(size * gap)
            y -= size * gap
            line(piece, size, bold, color=color)

    for kind, content in blocks:
        if kind == "title":
            wrapped(content, 20, True, "0.05 0.25 0.30")
            y -= 2
        elif kind == "subtitle":
            wrapped(content, 10, False, "0.35 0.42 0.45")
            y -= 6
        elif kind == "heading":
            room(44)
            y -= 10
            wrapped(content, 12, True, "0.05 0.25 0.30")
            y -= 3
            pages[-1].append(b"0.75 0.82 0.84 RG 0.6 w %d %.1f m %d %.1f l S\n" % (margin, y, width - margin, y))
            y -= 2
        elif kind == "text":
            wrapped(content, 10)
            y -= 3
        elif kind == "note":
            wrapped(content, 8.5, False, "0.35 0.42 0.45")
            y -= 2
        elif kind == "table":
            columns, rows, shares = content
            offsets = [margin + usable * sum(shares[:i]) for i in range(len(shares))]
            for index, row in enumerate([columns] + [list(r) for r in rows]):
                cells = [textwrap.wrap(str(cell), max(int(usable * share / (9 * 0.50)) - 1, 4)) or [""] for cell, share in zip(row, shares)]
                depth = max(len(c) for c in cells)
                room(depth * 12.5 + 3)
                top = y
                for depth_index in range(depth):
                    y = top - 12.5 * (depth_index + 1)
                    for cell, offset in zip(cells, offsets):
                        if depth_index < len(cell):
                            line(cell[depth_index], 9, index == 0, x=round(offset, 1), color="0.05 0.25 0.30" if index == 0 else "0.10 0.17 0.20")
                y -= 3
                pages[-1].append(b"0.88 0.91 0.92 RG 0.4 w %d %.1f m %d %.1f l S\n" % (margin, y, width - margin, y))
            y -= 6
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", None, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>", b"<< /Title (" + _pdf_text(title) + b") >>"]
    kids = []
    for number, commands in enumerate(pages, start=1):
        footer = b"BT /F1 8 Tf 0.45 0.50 0.52 rg %d 28 Td (" % margin + _pdf_text(f"Page {number} of {len(pages)}") + b") Tj ET\n"
        stream = b"".join(commands) + footer
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")
        content_id = len(objects)
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents %d 0 R >>" % (width, height, content_id))
        kids.append(len(objects))
    objects[1] = b"<< /Type /Pages /Count %d /Kids [" % len(kids) + b" ".join(b"%d 0 R" % k for k in kids) + b"] >>"
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    start = len(output)
    output += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        output += b"%010d 00000 n \n" % offset
    output += b"trailer\n<< /Size %d /Root 1 0 R /Info 5 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, start)
    return bytes(output)


# -----------------------------------------------------------------------------
# Free real prices for marks, and where every price came from
# -----------------------------------------------------------------------------
SOURCE_IMPORTED = "Imported marks"
SOURCE_DAY_AHEAD = "Day-ahead average, ENTSO-E"
SOURCE_PART_DEMO = "Day-ahead so far, demo for the rest"
SOURCE_PART_IMPORTED = "Day-ahead so far, imported for the rest"
SOURCE_AUCTION = "EUA auction price, EEX"
SOURCE_DEMO = "Demo"
REAL_PRICE_SOURCES = {SOURCE_IMPORTED, SOURCE_DAY_AHEAD, SOURCE_PART_IMPORTED, SOURCE_AUCTION}
PART_REAL_MARKS = "Part real marks"
DEMO_HISTORY_SOURCES = ("Demo marks", PART_REAL_MARKS, SYNTHETIC_HISTORY_SOURCE)
REALISED_COMPLETE_SHARE = 0.98
REALISED_HISTORY_MONTHS = 13
AUCTION_MAX_AGE_DAYS = 10
POWER_PRODUCT_ZONES = {"French Power Base": "France", "French Power Peak": "France",
                       "German Power Base": "Germany-Luxembourg", "German Power Peak": "Germany-Luxembourg"}
EUA_AUCTION_URL = "https://public.eex-group.com/eex/eua-auction-report/emission-spot-primary-market-auction-report-{year}-data.xlsx"


def realised_months(series: pd.DataFrame, profile: str, timezone_name: str, today: date, months_back: int = REALISED_HISTORY_MONTHS) -> dict:
    """{month: (average price, share of the month's delivery hours covered)} from published day-ahead prices.

    A financial power contract settles on the day-ahead average, so every published hour is a real price.
    """
    if series is None or series.empty:
        return {}
    local = series["local"]
    if str(profile).upper() == "PEAK":
        keep = (local.dt.dayofweek < 5) & (local.dt.hour >= 8) & (local.dt.hour < 20)
        series, local = series.loc[keep], local.loc[keep]
    first = _add_months(date(today.year, today.month, 1), -int(months_back))
    index = (local.dt.year * 12 + local.dt.month - 1).to_numpy()
    lowest, highest = first.year * 12 + first.month - 1, today.year * 12 + today.month - 1
    keep = (index >= lowest) & (index <= highest)
    hours = series["hours"].to_numpy(float)[keep]
    weighted = series["value"].to_numpy(float)[keep] * hours
    sums = pd.DataFrame({"m": index[keep], "w": weighted, "h": hours}).groupby("m")[["w", "h"]].sum()
    result = {}
    for code, (total, covered) in zip(sums.index, sums.to_numpy()):
        month = date(int(code) // 12, int(code) % 12 + 1, 1)
        possible = delivery_hours(month, _add_months(month, 1), profile, timezone_name)
        if covered > 0 and possible > 0:
            result[month] = (float(total) / float(covered), min(float(covered) / possible, 1.0))
    return result


def compose_curve(product: str, base: pd.DataFrame, base_source: str, realised: dict, today: date,
                  auction: Optional[tuple] = None) -> tuple[pd.DataFrame, dict]:
    """The curve used for valuation, with each month's source. Real prices replace demo ones wherever they exist."""
    spec = CURVE_PRODUCTS[product]
    lookup = _curve_lookup(validate_curve(base))
    sources = {month: base_source for month in lookup}
    current = date(today.year, today.month, 1)
    if product == "EUA Carbon" and base_source == SOURCE_DEMO and auction is not None:
        # The futures curve is close to flat, so the latest auction price stands in for every month.
        price = float(auction[0])
        lookup = {month: price for month in sorted(set(lookup) | {current})}
        sources = {month: SOURCE_AUCTION for month in lookup}
    for month, (price, share) in sorted(realised.items()):
        if month < current:
            if share >= REALISED_COMPLETE_SHARE:
                lookup[month], sources[month] = price, SOURCE_DAY_AHEAD
        elif month == current:
            if share >= REALISED_COMPLETE_SHARE:
                lookup[month], sources[month] = price, SOURCE_DAY_AHEAD
                continue
            rest = lookup.get(month)
            rest_source = sources.get(month)
            if rest is None:
                following = _add_months(month, 1)
                rest, rest_source = lookup.get(following), sources.get(following)
            if rest is not None:
                lookup[month] = share * price + (1.0 - share) * float(rest)
                sources[month] = SOURCE_PART_IMPORTED if rest_source in REAL_PRICE_SOURCES else SOURCE_PART_DEMO
    months = sorted(lookup)
    curve = pd.DataFrame({"delivery_month": months, "product": product, "price": [lookup[m] for m in months],
                          "currency": spec["currency"], "unit": spec["unit"]})
    return validate_curve(curve), sources


def price_state(labels: Sequence[str]) -> str:
    """Real when every price is real, Demo when none is, otherwise Part real."""
    labels = list(labels)
    if not labels:
        return "None"
    real = sum(1 for label in labels if label in REAL_PRICE_SOURCES)
    partial = sum(1 for label in labels if label == SOURCE_PART_DEMO)
    if real == len(labels):
        return "Real"
    if real == 0 and partial == 0:
        return "Demo"
    return "Part real"


def trade_price_state(product: str, start: object, end: object, profile: str, sources: dict) -> str:
    """Whether a trade's mark uses real prices, demo prices or both, month by month over its delivery."""
    try:
        spec = CURVE_PRODUCTS[product]
        segments = _month_segments(_to_date(start), _to_date(end), str(profile or spec.get("profile", "BASE")).upper(), spec["timezone"])
    except Exception:
        return "None"
    held = sources.get(product, {})
    return price_state([held.get(month, "Missing") for month, _ in segments if month in held])


def parse_eua_auctions(content: bytes) -> pd.DataFrame:
    """Auction date and clearing price from EEX's yearly EUA auction report. Aviation (EUAA) auctions are left out."""
    raw = pd.read_excel(io.BytesIO(content), header=None, dtype=str)
    header_row = None
    for index in range(min(len(raw), 30)):
        cells = [str(v).strip().casefold() for v in raw.iloc[index].tolist()]
        if "date" in cells and any("auction price" in c for c in cells):
            header_row = index
            break
    if header_row is None:
        raise ValueError("The auction report's layout wasn't recognised: no Date and Auction Price columns.")
    frame = raw.iloc[header_row + 1:].copy()
    frame.columns = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    price_col = next(c for c in frame.columns if "auction price" in c.casefold())
    contract_col = next((c for c in frame.columns if c.casefold() in ("contract", "product")), None)
    out = pd.DataFrame({"date": pd.to_datetime(frame["Date"], errors="coerce", dayfirst=True), "price": _number_series(frame[price_col])})
    if contract_col:
        contract = frame[contract_col].astype(str).str.strip().str.upper()
        out = out.loc[~contract.str.startswith("EUAA").to_numpy()]
    out = out.dropna()
    out = out.loc[out["price"].between(1.0, 500.0)]
    if out.empty:
        raise ValueError("The auction report holds no usable prices.")
    return out.groupby(out["date"].dt.date, as_index=False)["price"].mean().rename(columns={"date": "date"}).sort_values("date").reset_index(drop=True)


def fetch_eua_auctions(today: Optional[date] = None, _http_fn=None) -> pd.DataFrame:
    """EEX's public EUA primary auction results for this year, and last year's in January."""
    today = today or date.today()
    get = _http_fn or _http_get_bytes
    frames, errors = [], []
    for year in ([today.year, today.year - 1] if today.month == 1 else [today.year]):
        try:
            content, _ = get(EUA_AUCTION_URL.format(year=year))
            frames.append(parse_eua_auctions(content))
        except ImportError as exc:
            raise RuntimeError("Reading the EEX auction report needs the openpyxl package: pip install openpyxl") from exc
        except Exception as exc:
            errors.append(f"{year}: {exc}")
    if not frames:
        raise RuntimeError("; ".join(errors) or "No auction report found.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)


# =============================================================================
# SECTION 3 · SAVED DATA (SQLite)
# Everything the desk needs tomorrow lives in one local database file.
# =============================================================================
def default_db_path() -> Path:
    configured = os.getenv("MERIDIAN_DB_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().with_name("meridian_desk.db")


class DeskStore:
    """Small persistence layer. Holds no credentials and is safe to back up as one file."""

    MAX_LIVE_ROWS = 400_000

    def __init__(self, path) -> None:
        self.path = str(path)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS custom_curves (
                    product TEXT PRIMARY KEY, payload TEXT NOT NULL, metadata TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS curve_snapshots (
                    snapshot_date TEXT NOT NULL, product TEXT NOT NULL, delivery_month TEXT NOT NULL,
                    price REAL NOT NULL, source TEXT NOT NULL,
                    PRIMARY KEY (snapshot_date, product, delivery_month));
                CREATE TABLE IF NOT EXISTS live_data (
                    key TEXT PRIMARY KEY, label TEXT NOT NULL, metadata TEXT NOT NULL,
                    fetched_at TEXT NOT NULL, payload BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS read_log (
                    read_date TEXT NOT NULL, market TEXT NOT NULL, source TEXT NOT NULL, reading TEXT NOT NULL, score REAL NOT NULL,
                    confidence TEXT NOT NULL, price_through TEXT NOT NULL, signals TEXT NOT NULL,
                    PRIMARY KEY (read_date, market, source));
                CREATE TABLE IF NOT EXISTS price_estimates (
                    delivery_date TEXT NOT NULL, zone TEXT NOT NULL, estimate REAL NOT NULL, low REAL NOT NULL, high REAL NOT NULL, made_on TEXT NOT NULL, method TEXT NOT NULL DEFAULT 'Weather',
                    PRIMARY KEY (delivery_date, zone));
                CREATE TABLE IF NOT EXISTS alert_rules (metric TEXT NOT NULL, condition TEXT NOT NULL, level REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS curve_files (
                    name TEXT PRIMARY KEY, sha TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL, processed_at TEXT NOT NULL);
            """)
        self._upgrade()

    def _upgrade(self) -> None:
        """Bring a database from an earlier version up to date. Replayed reads are rebuilt with the current method."""
        with self._connect() as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(price_estimates)")}
            if "method" not in columns:
                conn.execute("ALTER TABLE price_estimates ADD COLUMN method TEXT NOT NULL DEFAULT 'Weather'")
            row = conn.execute("SELECT value FROM settings WHERE key='read_method'").fetchone()
            if row is None or float(row[0]) < READ_METHOD_VERSION:
                conn.execute("DELETE FROM read_log WHERE source='Replayed'")
                conn.execute("INSERT OR REPLACE INTO settings VALUES ('read_method',?)", (str(float(READ_METHOD_VERSION)),))
            # Earlier versions mixed the two German auctions, so those prices are fetched again.
            row = conn.execute("SELECT value FROM settings WHERE key='price_series'").fetchone()
            if row is None or float(row[0]) < PRICE_SERIES_VERSION:
                conn.execute("DELETE FROM live_data WHERE key IN ('entsoe_price|Germany-Luxembourg', 'entsoe_price|Austria')")
                conn.execute("DELETE FROM read_log WHERE source='Replayed' AND market='German power'")
                conn.execute("INSERT OR REPLACE INTO settings VALUES ('price_series',?)", (str(float(PRICE_SERIES_VERSION)),))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=15)

    @staticmethod
    def _table_ready(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in out.columns:
            if out[column].dtype == object:
                out[column] = out[column].map(lambda v: v.isoformat() if isinstance(v, (date, datetime)) else v)
        return out

    def save_table(self, name: str, frame: pd.DataFrame) -> None:
        if name not in {"trades", "physical_ops"}:
            raise ValueError(f"Unknown table: {name}")
        with self._connect() as conn:
            self._table_ready(frame).to_sql(name, conn, if_exists="replace", index=False)

    def load_table(self, name: str) -> Optional[pd.DataFrame]:
        if name not in {"trades", "physical_ops"}:
            raise ValueError(f"Unknown table: {name}")
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
            return pd.read_sql_query(f'SELECT * FROM "{name}"', conn) if exists else None

    def save_custom_curve(self, product: str, curve: pd.DataFrame, metadata: dict) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO custom_curves VALUES (?,?,?,?)",
                         (product, self._table_ready(curve).to_json(orient="records"), json.dumps(metadata),
                          datetime.now(timezone.utc).isoformat(timespec="seconds")))

    def load_custom_curves(self) -> tuple[dict, dict]:
        curves, metadata = {}, {}
        with self._connect() as conn:
            for product, payload, meta, _ in conn.execute("SELECT * FROM custom_curves"):
                curves[product] = validate_curve(pd.read_json(io.StringIO(payload), orient="records", convert_dates=False))
                metadata[product] = json.loads(meta)
        return curves, metadata

    def clear_custom_curves(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM custom_curves")

    def save_snapshot(self, snapshot_date: date, curve: pd.DataFrame, source: str, replace: bool = True) -> None:
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        rows = [(snapshot_date.isoformat(), product, _to_date(month).isoformat(), float(price), source)
                for product, month, price in zip(curve["product"], curve["delivery_month"], curve["price"])]
        with self._connect() as conn:
            conn.executemany(f"{verb} INTO curve_snapshots VALUES (?,?,?,?,?)", rows)

    def add_snapshot_rows(self, rows: pd.DataFrame, replace: bool = False) -> int:
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        payload = list(rows[["snapshot_date", "product", "delivery_month", "price", "source"]].itertuples(index=False, name=None))
        with self._connect() as conn:
            before = conn.total_changes
            conn.executemany(f"{verb} INTO curve_snapshots VALUES (?,?,?,?,?)", payload)
            return conn.total_changes - before

    def remove_demo_marks(self, product: str) -> None:
        """Real marks replace a product's demo history, so demo prices never become a day's P&L or risk."""
        with self._connect() as conn:
            conn.execute("DELETE FROM curve_snapshots WHERE product=? AND source='Demo marks'", (product,))

    def load_snapshots(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM curve_snapshots ORDER BY snapshot_date, product, delivery_month", conn)

    def delete_snapshots(self, source: Optional[str] = None) -> None:
        with self._connect() as conn:
            if source:
                conn.execute("DELETE FROM curve_snapshots WHERE source=?", (source,))
            else:
                conn.execute("DELETE FROM curve_snapshots")

    def save_live_item(self, key: str, item: dict) -> bool:
        """Save a fetched dataset. Very large frames stay in the session only."""
        frame = item.get("data")
        if not isinstance(frame, pd.DataFrame) or len(frame) > self.MAX_LIVE_ROWS:
            return False
        payload = self._encode_frame(frame)
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO live_data VALUES (?,?,?,?,?)",
                         (key, str(item.get("label", key)), json.dumps(item.get("metadata", {}), default=str),
                          str(item.get("fetched_at", "")), payload))
        return True

    @staticmethod
    def _encode_frame(frame: pd.DataFrame) -> bytes:
        """Parquet is far faster and smaller than JSON and keeps column types. JSON is the fallback."""
        try:
            buffer = io.BytesIO()
            DeskStore._table_ready(frame).to_parquet(buffer, index=False, compression="zstd")
            return buffer.getvalue()
        except Exception:
            return gzip.compress(frame.to_json(orient="split", date_format="iso", date_unit="s", index=False).encode("utf-8"), compresslevel=5)

    @staticmethod
    def _decode_frame(payload: bytes) -> pd.DataFrame:
        if payload[:4] == b"PAR1":
            return pd.read_parquet(io.BytesIO(payload))
        # Data saved by earlier versions, or data Parquet couldn't hold.
        frame = pd.read_json(io.StringIO(gzip.decompress(payload).decode("utf-8")), orient="split", convert_dates=False)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        for column in ("date", "local_today"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame

    def load_live_items(self) -> dict:
        items = {}
        with self._connect() as conn:
            for key, label, metadata, fetched_at, payload in conn.execute("SELECT * FROM live_data"):
                try:
                    frame = self._decode_frame(bytes(payload))
                except Exception:
                    continue
                items[key] = {"label": label, "data": frame, "fetched_at": fetched_at, "metadata": json.loads(metadata)}
        return items

    READ_COLUMNS = ("read_date", "market", "source", "reading", "score", "confidence", "price_through", "signals")

    def save_reads(self, rows: Sequence[dict]) -> None:
        """Today's read can be refreshed. Earlier dates are never rewritten, so the record stays honest."""
        today = date.today().isoformat()
        with self._connect() as conn:
            for row in rows:
                verb = "INSERT OR REPLACE" if row["source"] == "Recorded" and row["read_date"] == today else "INSERT OR IGNORE"
                conn.execute(f"{verb} INTO read_log VALUES (?,?,?,?,?,?,?,?)", tuple(row[c] for c in self.READ_COLUMNS))

    def load_reads(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM read_log ORDER BY read_date, market", conn)

    def save_price_estimates(self, rows: Sequence[tuple]) -> None:
        """The first estimate for a day stands, so the record can't be improved afterwards."""
        with self._connect() as conn:
            conn.executemany("INSERT OR IGNORE INTO price_estimates (delivery_date, zone, estimate, low, high, made_on, method) VALUES (?,?,?,?,?,?,?)", rows)

    def load_price_estimates(self, zone: str) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM price_estimates WHERE zone=? ORDER BY delivery_date", conn, params=(zone,))

    def load_alerts(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT metric, condition, level FROM alert_rules", conn)

    def save_alerts(self, rules: pd.DataFrame) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM alert_rules")
            conn.executemany("INSERT INTO alert_rules VALUES (?,?,?)", [(str(m), str(c), float(l)) for m, c, l in zip(rules["metric"], rules["condition"], rules["level"])])

    def curve_file_seen(self) -> dict:
        with self._connect() as conn:
            return dict(conn.execute("SELECT name, sha FROM curve_files").fetchall())

    def record_curve_file(self, name: str, sha: str, status: str, detail: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO curve_files VALUES (?,?,?,?,?)", (name, sha, status, detail, datetime.now(timezone.utc).isoformat(timespec="seconds")))

    def load_curve_files(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM curve_files ORDER BY processed_at DESC", conn)

    def get_setting(self, key: str, default: float) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        try:
            return float(row[0]) if row else float(default)
        except ValueError:
            return float(default)

    def set_setting(self, key: str, value: float) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, str(float(value))))


def desk_store() -> Optional[DeskStore]:
    """Open the database once per session. If the folder is read-only, work in memory."""
    if "desk_store" not in st.session_state:
        try:
            st.session_state.desk_store = DeskStore(default_db_path())
            st.session_state.desk_store_error = ""
        except Exception as exc:
            st.session_state.desk_store = None
            st.session_state.desk_store_error = f"{type(exc).__name__}: {exc}"
    return st.session_state.desk_store


def _save(action) -> None:
    """Run a save without ever interrupting the desk; failures appear in the sidebar."""
    store = desk_store()
    if store is None:
        return
    try:
        action(store)
        st.session_state.desk_store_error = ""
    except Exception as exc:
        st.session_state.desk_store_error = f"{type(exc).__name__}: {exc}"


def bump(name: str) -> None:
    """Record that a kind of data changed, so results derived from it are recalculated."""
    versions = st.session_state.setdefault("data_versions", {})
    versions[name] = versions.get(name, 0) + 1


def data_version(*names: str) -> tuple:
    versions = st.session_state.get("data_versions", {})
    return tuple(versions.get(name, 0) for name in names)


def session_memo(name: str, key: object, build: Callable[[], object]) -> object:
    """Keep one derived result per name until its key changes. Callers must not modify it."""
    memo = st.session_state.setdefault("derived_memo", {})
    held = memo.get(name)
    if held is not None and held[0] == key:
        return held[1]
    value = build()
    memo[name] = (key, value)
    return value


def commit_trades(trades: pd.DataFrame) -> None:
    st.session_state.trades = trades
    bump("trades")
    st.session_state.physical_ops = sync_physical_ops_with_trades(st.session_state.get("physical_ops"), trades)
    _save(lambda s: (s.save_table("trades", trades), s.save_table("physical_ops", st.session_state.physical_ops)))


def commit_ops(operations: pd.DataFrame) -> None:
    st.session_state.physical_ops = operations
    _save(lambda s: s.save_table("physical_ops", operations))


def commit_custom_curve(product: str, curve: pd.DataFrame, metadata: dict, curve_date: Optional[date] = None) -> None:
    """Apply an imported curve for pricing and save it under the market date of its marks."""
    day = curve_date or date.today()
    metadata = {**metadata, "curve_date": day.isoformat()}
    st.session_state.custom_curves[product] = curve
    st.session_state.curve_metadata[product] = metadata
    st.session_state.pop("last_quote", None)
    bump("curves")
    bump("snapshots")
    _save(lambda s: (s.save_custom_curve(product, curve, metadata), s.remove_demo_marks(product),
                     s.save_snapshot(day, curve, str(metadata.get("source_label", "Imported curve")), replace=True)))


def fx_rate(key: str) -> float:
    return float(st.session_state.get("settings", {}).get(key, DEFAULT_SETTINGS[key]))


def set_fx_rate(key: str, value: float, ecb: bool = False) -> None:
    """Save a rate. A rate typed in by hand marks the FX source as yours rather than the ECB's."""
    settings = st.session_state.setdefault("settings", {})
    settings[key] = float(value)
    _save(lambda s: s.set_setting(key, float(value)))
    if not ecb and key in ("eur_per_gbp", "eur_per_usd"):
        settings["fx_ecb_epoch"] = -1.0
        _save(lambda s: s.set_setting("fx_ecb_epoch", -1.0))


DEFAULT_SETTINGS = {"eur_per_gbp": 1.17, "eur_per_usd": 0.92, "auto_update": 1.0, "fx_ecb_epoch": 0.0}


# =============================================================================
# SECTION 4 · MARKET DATA CONNECTORS
# Keys are never stored in this file. They come from Streamlit secrets, environment
# variables or a keys file beside the app. Keep that file out of any shared folder or repository.
# =============================================================================
HTTP_TIMEOUT_SECONDS = 25
MAX_API_RESPONSE_BYTES = 20 * 1024 * 1024
KEYS_FILE_NAME = "meridian_keys.toml"
ENTSOE_WINDOW_DAYS = 360
ENTSOE_GENERATION_WINDOW_DAYS = 31
ENTSOE_FALLBACK_WINDOW_DAYS = 30
RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
RETRY_WAIT_SECONDS = 1.5
ENTSOE_MAX_DAYS = 1096
ECB_FX_BASE = "https://data-api.ecb.europa.eu/service/data/EXR"
OPEN_METEO_ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
GIE_AGSI_BASE = "https://agsi.gie.eu/api"
GIE_ALSI_BASE = "https://alsi.gie.eu/api"
ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"
ENTSOG_BASE = "https://transparency.entsog.eu/api/v1"
OPEN_METEO_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_CUSTOMER_BASE = "https://customer-api.open-meteo.com/v1/forecast"

ENTSOE_ZONES = {
    "France": {"eic": "10YFR-RTE------C", "timezone": "Europe/Paris"},
    "Germany-Luxembourg": {"eic": "10Y1001A1001A82H", "timezone": "Europe/Berlin"},
    "Netherlands": {"eic": "10YNL----------L", "timezone": "Europe/Amsterdam"},
    "Belgium": {"eic": "10YBE----------2", "timezone": "Europe/Brussels"},
    "Spain": {"eic": "10YES-REE------0", "timezone": "Europe/Madrid"},
    "Italy North": {"eic": "10Y1001A1001A73I", "timezone": "Europe/Rome"},
    "Austria": {"eic": "10YAT-APG------L", "timezone": "Europe/Vienna"},
    "Denmark West": {"eic": "10YDK-1--------W", "timezone": "Europe/Copenhagen"},
    "Denmark East": {"eic": "10YDK-2--------M", "timezone": "Europe/Copenhagen"},
    "Poland": {"eic": "10YPL-AREA-----S", "timezone": "Europe/Warsaw"},
    "Great Britain": {"eic": "10YGB----------A", "timezone": "Europe/London"},
    "Switzerland": {"eic": "10YCH-SWISSGRIDZ", "timezone": "Europe/Zurich"},
}

WEATHER_LOCATIONS = {
    "Amsterdam / TTF proxy": {"latitude": 52.3676, "longitude": 4.9041, "timezone": "Europe/Amsterdam"},
    "London / NBP proxy": {"latitude": 51.5072, "longitude": -0.1276, "timezone": "Europe/London"},
    "Paris / French power proxy": {"latitude": 48.8566, "longitude": 2.3522, "timezone": "Europe/Paris"},
    "Berlin / German power proxy": {"latitude": 52.5200, "longitude": 13.4050, "timezone": "Europe/Berlin"},
    "Brussels / Belgian power proxy": {"latitude": 50.8503, "longitude": 4.3517, "timezone": "Europe/Brussels"},
    "Madrid / Iberian power proxy": {"latitude": 40.4168, "longitude": -3.7038, "timezone": "Europe/Madrid"},
}


ENTSOE_PSR_NAMES = {
    "B01": "Biomass", "B02": "Lignite", "B03": "Coal-derived gas", "B04": "Natural gas",
    "B05": "Hard coal", "B06": "Oil", "B07": "Oil shale", "B08": "Peat",
    "B09": "Geothermal", "B10": "Hydro pumped storage", "B11": "Hydro run-of-river",
    "B12": "Hydro reservoir", "B13": "Marine", "B14": "Nuclear", "B15": "Other renewable",
    "B16": "Solar", "B17": "Waste", "B18": "Wind offshore", "B19": "Wind onshore",
    "B20": "Other", "B21": "AC link", "B22": "DC link", "B23": "Substation", "B24": "Transformer", "B25": "Energy storage",
}


def _secret_or_env(secret_name: str, env_name: str) -> str:
    """Read optional credentials without printing or logging them."""
    try:
        value = st.secrets.get(secret_name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return str(os.getenv(env_name, "")).strip()


def _build_url(url: str, params: Optional[dict] = None) -> str:
    clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
    if not clean:
        return url
    separator = "&" if "?" in url else "?"
    return url + separator + urlencode(clean, doseq=True)


def _json_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> dict | list:
    body, _ = _http_get_bytes(url, params=params, headers=headers)
    try:
        return json.loads(body.decode("utf-8-sig"))
    except Exception as exc:
        raise RuntimeError("The API response was not valid JSON. Confirm that the URL returns JSON data.") from exc


def _payload_rows(payload) -> list[dict]:
    """Extract record lists from common API response envelopes."""
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    # Several providers change the envelope's capitalisation over time.
    # Match envelope names case-insensitively, but preserve record field names.
    lower_keys = {str(k).casefold(): v for k, v in payload.items()}
    record_keys = {
        "gasdaystart", "periodfrom", "periodto", "value", "pointdirection",
        "pointdirectionkey", "operatorlabel", "pointlabel", "full", "price",
        "gasinstorage", "name", "timestamp",
    }
    envelope_keys = (
        "data", "results", "items", "records", "result", "operatorpointdirections",
        "operatorpointdirection", "aggregateddata", "operationaldata", "operationaldatas",
    )
    for key in envelope_keys:
        value = lower_keys.get(key.casefold())
        if isinstance(value, list):
            return [dict(x) for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            nested = _payload_rows(value)
            if nested:
                return nested
            # Data may be a single observation rather than a list.
            if any(str(k).casefold() in record_keys for k in value):
                return [value]
    if any(str(k).casefold() in record_keys for k in payload):
        return [payload]
    return []


def _number_series(series: pd.Series) -> pd.Series:
    """Parse provider numbers, including decimal commas and missing-value markers."""
    def one(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        s = str(value).strip()
        if s in {"", "-", "--", "N/A", "n/a", "null", "None"}:
            return float("nan")
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return float("nan")
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    # Each distinct text value is parsed once, which matters for long repetitive columns.
    return series.map({value: one(value) for value in series.dropna().unique()}).astype(float)


def _flatten_records(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    try:
        return pd.json_normalize(rows, sep="_")
    except Exception:
        return pd.DataFrame(rows)


def fetch_gie_history(dataset: str, api_key: str, scope: str,
                      start_date: date, end_date: date, country_code: str = "DE",
                      _json_get_fn=None) -> pd.DataFrame:
    """Fetch GIE AGSI+ storage or ALSI LNG observations for a country/European aggregate."""
    if not api_key.strip():
        raise ValueError("Enter your GIE API key. Register at https://agsi.gie.eu/account.")
    if end_date < start_date:
        raise ValueError("End date must be on or after start date.")
    if (end_date - start_date).days > 2300:
        raise ValueError("Select a period of no more than six years to limit API calls.")
    dataset_key = dataset.strip().upper()
    if dataset_key not in {"AGSI", "ALSI"}:
        raise ValueError("GIE dataset must be AGSI or ALSI.")
    endpoint = GIE_AGSI_BASE if dataset_key == "AGSI" else GIE_ALSI_BASE
    params_base = {"from": start_date.isoformat(), "to": end_date.isoformat(), "size": 300}
    # GIE's documented selector is `type=EU` (not `continent=EU`).
    # Country filters use the provider's two-letter country slug.
    is_europe_scope = scope in {"Europe aggregate", "Europe / EU datasets"}
    if is_europe_scope:
        params_base["type"] = "eu"
    else:
        params_base["country"] = country_code.lower()
    request_json = _json_get_fn or _json_get
    all_rows: list[dict] = []
    # A 2-year European dataset can span many pages. Never silently return a partial curve.
    max_pages = 100
    pagination_complete = False
    for page in range(1, max_pages + 1):
        params = dict(params_base, page=page)
        payload = request_json(endpoint, params=params, headers={"x-key": api_key, "Accept": "application/json"})
        if isinstance(payload, dict):
            error = str(payload.get("error", "") or payload.get("message", ""))
            if error and ("denied" in error.lower() or "api key" in error.lower() or "invalid" in error.lower()):
                raise RuntimeError("GIE rejected the API key or dataset access. Check the key and its AGSI+/ALSI permissions.")
        rows = _payload_rows(payload)
        all_rows.extend(rows)
        last_page = None
        if isinstance(payload, dict):
            lower_meta = {str(k).casefold(): v for k, v in payload.items()}
            for key in ("last_page", "lastpage"):
                if lower_meta.get(key) is not None:
                    try: last_page = int(lower_meta[key])
                    except Exception: pass
                    break
        if not rows:
            pagination_complete = True
            break
        if last_page is not None and page >= last_page:
            pagination_complete = True
            break
        if last_page is None and len(rows) < int(params_base["size"]):
            pagination_complete = True
            break
    if not pagination_complete:
        raise RuntimeError(f"GIE query exceeded the {max_pages}-page safety limit and may be incomplete. Narrow the date range and retry.")
    if not all_rows:
        raise RuntimeError("GIE returned no records for this scope and period. Try another country/date range or check dataset access.")
    frame = _flatten_records(all_rows)
    date_col = next((c for c in ("gasDayStart", "gas_day", "date", "gasDay") if c in frame.columns), None)
    if date_col:
        frame["observation_date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.date
    numeric_cols = [c for c in frame.columns if c.lower() in {
        "full", "gasinstorage", "workinggasvolume", "injection", "withdrawal",
        "injectioncapacity", "withdrawalcapacity", "inventory", "sendout", "dtmi", "dtrs",
        "gasinstoragegwh", "inventory_gwh", "sendout_gwh", "workinggasvolume_twh"
    } or c.lower().endswith(("_gwh", "_twh"))]
    for col in numeric_cols:
        frame[col] = _number_series(frame[col])
    frame["_source"] = f"GIE {dataset_key}+"
    frame["_fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame["_scope"] = "Europe / EU datasets" if is_europe_scope else f"{country_code.upper()} aggregate / facilities"
    return frame


@lru_cache(maxsize=512)
def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_text(parent, names: Sequence[str]) -> Optional[str]:
    wanted = set(names)
    for node in parent.iter():
        if _xml_local_name(node.tag) in wanted and node.text and node.text.strip():
            return node.text.strip()
    return None


def _parse_duration(value: str) -> pd.Timedelta:
    # ENTSO-E uses ISO 8601 durations such as PT60M, PT15M and PT1H.
    td = pd.to_timedelta(value)
    if pd.isna(td) or td <= pd.Timedelta(0):
        raise ValueError(f"Unsupported ENTSO-E resolution: {value}")
    return td


def normalise_entsog_point_catalogue(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the catalogue has the exact pointDirection identifier ENTSOG expects.

    Depending on response representation, the catalogue may expose a combined
    pointDirection/key field or the separate operatorKey + pointKey + directionKey
    components. ENTSOG's API identifier is the exact concatenation of those components.
    """
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    def norm(name: object) -> str:
        return "".join(ch for ch in str(name).casefold() if ch.isalnum())
    lookup = {norm(column): column for column in out.columns}

    combined_candidates = ("pointdirection", "pointdirectionkey", "pointdirectionid")
    combined_col = next((lookup[k] for k in combined_candidates if k in lookup), None)
    if combined_col is not None:
        out["pointDirection"] = out[combined_col].astype("string").str.strip()
        return out

    operator_col = next((lookup[k] for k in ("operatorkey", "operatorid") if k in lookup), None)
    point_col = next((lookup[k] for k in ("pointkey", "connectionpointkey", "pointid") if k in lookup), None)
    direction_col = next((lookup[k] for k in ("directionkey", "directionid", "flowdirectionkey") if k in lookup), None)
    if operator_col is not None and point_col is not None and direction_col is not None:
        components = (out[operator_col], out[point_col], out[direction_col])
        valid = pd.Series(True, index=out.index)
        for column in components:
            valid &= column.notna() & column.astype(str).str.strip().ne("")
        ids = components[0].astype("string").str.strip().fillna("") + components[1].astype("string").str.strip().fillna("") + components[2].astype("string").str.strip().fillna("")
        out["pointDirection"] = ids.where(valid, pd.NA)
        return out

    # Some API representations expose a generic key/id only; use it as a last resort.
    generic_col = next((lookup[k] for k in ("key", "id") if k in lookup), None)
    if generic_col is not None:
        out["pointDirection"] = out[generic_col].astype("string").str.strip()
    return out


def fetch_entsog_point_catalogue(limit: int = 1000) -> pd.DataFrame:
    """Fetch a bounded list of gas transmission point directions with published data."""
    payload = _json_get(f"{ENTSOG_BASE}/operatorpointdirections", params={"hasData": 1, "limit": min(max(int(limit), 1), 1000)})
    rows = _payload_rows(payload)
    if not rows:
        raise RuntimeError("ENTSOG returned no point-direction catalogue rows. Try the official platform catalogue.")
    frame = _flatten_records(rows)
    frame = normalise_entsog_point_catalogue(frame)
    if "pointDirection" not in frame.columns or frame["pointDirection"].dropna().empty:
        raise RuntimeError("ENTSOG returned catalogue rows without a recognised pointDirection identifier. Paste an identifier from the official catalogue or update the mapping for the provider's current response fields.")
    return frame


def fetch_entsog_flow(point_direction: str, start_date: date,
                      end_date: date, indicator: str = "Physical Flow") -> pd.DataFrame:
    """Fetch physical gas flows for one exact ENTSOG pointDirection identifier.

    ENTSOG's date filters are passed as the selected inclusive calendar dates.
    """
    if not point_direction.strip():
        raise ValueError("Enter/select a pointDirection identifier from the ENTSOG catalogue.")
    if end_date < start_date:
        raise ValueError("End date must be on or after start date.")
    if (end_date - start_date).days > 180:
        raise ValueError("Select a period of no more than 180 days to avoid slow provider queries.")
    params = {
        "pointDirection": point_direction.strip(),
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "indicator": indicator,
        "periodType": "day",
        "timeZone": "CET",
        "limit": 1000,
        "sorting": "periodFrom",
    }
    payload = _json_get(f"{ENTSOG_BASE}/aggregatedData", params=params)
    rows = _payload_rows(payload)
    if not rows:
        raise RuntimeError("ENTSOG returned no records. pointDirection and indicator values are case-sensitive; verify the catalogue selection and date range.")
    frame = _flatten_records(rows)
    date_col = next((c for c in ("periodFrom", "periodTo", "from", "date") if c in frame.columns), None)
    value_col = next((c for c in ("value", "quantity", "physicalFlow") if c in frame.columns), None)
    if date_col:
        frame["timestamp"] = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    if value_col:
        frame["value_numeric"] = _number_series(frame[value_col])
    frame["_source"] = "ENTSOG Transparency Platform"
    frame["_fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame["_point_direction"] = point_direction.strip()
    return frame


def fetch_open_meteo(latitude: float, longitude: float, timezone_name: str,
                     past_days: int = 3, forecast_days: int = 7,
                     api_key: str = "", commercial_endpoint: bool = False,
                     _json_get_fn=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch weather forecast/hourly observations and aggregate daily energy proxies."""
    if not (-90 <= float(latitude) <= 90 and -180 <= float(longitude) <= 180):
        raise ValueError("Weather coordinates must be valid latitude/longitude values.")
    if not (0 <= int(past_days) <= 7 and 1 <= int(forecast_days) <= 16):
        raise ValueError("Use past_days 0–7 and forecast_days 1–16 for this dashboard request.")
    endpoint = OPEN_METEO_CUSTOMER_BASE if commercial_endpoint else OPEN_METEO_FORECAST_BASE
    params = {
        "latitude": latitude, "longitude": longitude,
        # 100 m wind is a more relevant wind-generation proxy than 10 m wind,
        # but neither variable is a generation forecast without turbine/power-curve data.
        "hourly": "temperature_2m,wind_speed_10m,wind_speed_100m,shortwave_radiation,precipitation,cloud_cover",
        "past_days": int(past_days), "forecast_days": int(forecast_days),
        "timezone": timezone_name, "timeformat": "unixtime",
    }
    if commercial_endpoint:
        if not api_key.strip():
            raise ValueError("Enter your Open-Meteo commercial API key or select the public endpoint.")
        params["apikey"] = api_key.strip()
    request_json = _json_get_fn or _json_get
    payload = request_json(endpoint, params=params)
    hourly_payload = payload.get("hourly", {}) if isinstance(payload, dict) else {}
    if not hourly_payload or "time" not in hourly_payload:
        raise RuntimeError("Open-Meteo response has no hourly observations. Check coordinates and request parameters.")
    hourly = pd.DataFrame(hourly_payload)
    time_values = hourly.pop("time")
    numeric_times = pd.to_numeric(time_values, errors="coerce")
    if numeric_times.notna().all():
        # Unix timestamps avoid ambiguous local wall times on DST fall-back days.
        hourly["timestamp"] = pd.to_datetime(numeric_times, unit="s", utc=True).dt.tz_convert(timezone_name)
    else:
        # Fallback supports saved fixtures/provider-compatible ISO timestamps.
        hourly["timestamp"] = pd.to_datetime(time_values, errors="coerce")
        if getattr(hourly["timestamp"].dt, "tz", None) is None:
            try:
                hourly["timestamp"] = hourly["timestamp"].dt.tz_localize(timezone_name, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                hourly["timestamp"] = hourly["timestamp"].dt.tz_localize(timezone_name, ambiguous="NaT", nonexistent="shift_forward")
    hourly["date"] = hourly["timestamp"].dt.date
    hourly["timezone"] = timezone_name
    numeric_weather_columns = (
        "temperature_2m", "wind_speed_10m", "wind_speed_100m",
        "shortwave_radiation", "precipitation", "cloud_cover",
    )
    for col in numeric_weather_columns:
        if col in hourly:
            hourly[col] = pd.to_numeric(hourly[col], errors="coerce")

    # Average weather variables by local calendar day; precipitation is accumulated.
    mean_cols = [c for c in numeric_weather_columns if c in hourly.columns and c != "precipitation"]
    agg = {c: "mean" for c in mean_cols}
    if "precipitation" in hourly.columns:
        agg["precipitation"] = "sum"
    daily = hourly.groupby("date", as_index=False).agg(agg) if agg else pd.DataFrame()
    if not daily.empty:
        daily = daily.rename(columns={
            "temperature_2m": "temperature_mean_c",
            "wind_speed_10m": "wind_speed_mean_kmh",
            "wind_speed_100m": "wind_speed_100m_mean_kmh",
            "shortwave_radiation": "solar_radiation_mean_wm2",
            "precipitation": "precipitation_sum_mm",
            "cloud_cover": "cloud_cover_mean_pct",
        })
        daily["date"] = pd.to_datetime(daily["date"])
        daily["timezone"] = timezone_name
        if "shortwave_radiation" in hourly.columns:
            solar_sum = hourly.groupby("date", as_index=False)["shortwave_radiation"].sum(min_count=1)
            solar_sum = solar_sum.rename(columns={"shortwave_radiation": "solar_irradiation_sum_whm2"})
            solar_sum["date"] = pd.to_datetime(solar_sum["date"], errors="coerce")
            daily = daily.merge(solar_sum, on="date", how="left")
        hours_per_day = hourly.groupby("date").size()
        daily["hours_covered"] = daily["date"].dt.date.map(hours_per_day).fillna(0).astype(int)
        local_today = datetime.now(ZoneInfo(timezone_name)).date()
        daily["local_today"] = pd.Timestamp(local_today)
        daily["forecast_period"] = daily["date"].dt.date.map(
            lambda d: "Historical / recent" if d < local_today else ("Today (may be partial)" if d == local_today else "Forecast")
        )
        # Simple degree-day proxies (not official national demand indices): use daily mean temperature.
        if "temperature_mean_c" in daily.columns:
            daily["heating_degree_day_proxy_18c"] = (18.0 - daily["temperature_mean_c"]).clip(lower=0)
            daily["cooling_degree_day_proxy_22c"] = (daily["temperature_mean_c"] - 22.0).clip(lower=0)
    hourly["_source"] = "Open-Meteo"
    hourly["_fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not daily.empty:
        daily["_source"] = "Open-Meteo"
    return hourly, daily


def fetch_open_meteo_market_pack(
    location_names: Sequence[str], past_days: int = 3, forecast_days: int = 10,
    api_key: str = "", commercial_endpoint: bool = False, _fetch_fn=None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fetch a selected set of representative European energy-market weather locations.

    Requests are made only when invoked by a user action. Individual location failures are
    returned as warnings so one unavailable endpoint does not discard successful locations.
    The location set is a commercial-context proxy, not a population- or load-weighted forecast.
    """
    selected = list(dict.fromkeys(str(name) for name in location_names))
    unknown = [name for name in selected if name not in WEATHER_LOCATIONS]
    if unknown:
        raise ValueError(f"Unknown weather location(s): {unknown}")
    if not selected:
        raise ValueError("Select at least one energy-market weather location.")

    def fetch_one(name: str):
        loc = WEATHER_LOCATIONS[name]
        if _fetch_fn is not None:
            hourly_frame, daily_frame = _fetch_fn(name, loc, past_days, forecast_days, api_key, commercial_endpoint)
        else:
            hourly_frame, daily_frame = fetch_open_meteo(
                loc["latitude"], loc["longitude"], loc["timezone"],
                past_days=past_days, forecast_days=forecast_days,
                api_key=api_key, commercial_endpoint=commercial_endpoint,
            )
        hourly_frame = hourly_frame.copy()
        daily_frame = daily_frame.copy()
        for frame in (hourly_frame, daily_frame):
            if not frame.empty:
                frame["location"] = name
                frame["market_area"] = name.split(" / ", 1)[-1].replace(" proxy", "")
                frame["latitude"] = float(loc["latitude"])
                frame["longitude"] = float(loc["longitude"])
                frame["timezone"] = loc["timezone"]
        return name, hourly_frame, daily_frame

    hourly_results, daily_results, failures = {}, {}, []
    # Concurrent I/O keeps a six-zone refresh responsive; the user still controls refresh cadence.
    with ThreadPoolExecutor(max_workers=min(4, len(selected))) as executor:
        future_map = {executor.submit(fetch_one, name): name for name in selected}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                _, hourly_frame, daily_frame = future.result()
                if not hourly_frame.empty:
                    hourly_results[name] = hourly_frame
                if not daily_frame.empty:
                    daily_results[name] = daily_frame
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")

    # Preserve user-selected ordering in charts and tables, not network completion order.
    hourly = pd.concat([hourly_results[name] for name in selected if name in hourly_results], ignore_index=True) if hourly_results else pd.DataFrame()
    daily = pd.concat([daily_results[name] for name in selected if name in daily_results], ignore_index=True) if daily_results else pd.DataFrame()
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily.sort_values(["location", "date"]).reset_index(drop=True)
    if not hourly.empty:
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], errors="coerce")
        hourly = hourly.sort_values(["location", "timestamp"]).reset_index(drop=True)
    return hourly, daily, failures


def weather_forecast_summary(daily: pd.DataFrame, horizon_days: int = 7,
                             fallback_location: str = "Selected location") -> pd.DataFrame:
    """Summarise upcoming future local forecast dates by reference location."""
    if daily is None or daily.empty or "date" not in daily.columns:
        return pd.DataFrame()
    if int(horizon_days) < 1 or int(horizon_days) > 14:
        raise ValueError("Weather summary horizon must be between 1 and 14 days.")
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
    if "location" not in data.columns:
        data["location"] = fallback_location
    out_rows = []
    for location, group in data.groupby("location", dropna=False, sort=False):
        group = group.dropna(subset=["date"]).copy()
        if group.empty:
            continue
        if "local_today" in group.columns and group["local_today"].notna().any():
            today = pd.to_datetime(group["local_today"].dropna().iloc[0], errors="coerce").date()
        else:
            tz_name = str(group["timezone"].dropna().iloc[0]) if "timezone" in group.columns and group["timezone"].notna().any() else "Europe/Paris"
            try:
                today = datetime.now(ZoneInfo(tz_name)).date()
            except Exception:
                today = date.today()
        future = group.loc[(group["date"] > today) & (group["date"] <= today + timedelta(days=int(horizon_days)))].copy()
        row = {"Location": str(location), "Forecast days": int(future["date"].nunique()),
               "Forecast start": min(future["date"]).isoformat() if not future.empty else "—",
               "Forecast end": max(future["date"]).isoformat() if not future.empty else "—"}
        aliases = {
            "temperature_mean_c": "Mean temperature (°C)",
            "heating_degree_day_proxy_18c": "Heating degree-day proxy (base 18°C)",
            "cooling_degree_day_proxy_22c": "Cooling degree-day proxy (base 22°C)",
            "wind_speed_100m_mean_kmh": "Mean wind at 100 m (km/h)",
            "wind_speed_mean_kmh": "Mean wind at 10 m (km/h)",
            "solar_radiation_mean_wm2": "Mean solar radiation (W/m²)",
            "solar_irradiation_sum_whm2": "Daily solar irradiation (Wh/m²)",
        }
        for col, label in aliases.items():
            if col in future.columns:
                values = pd.to_numeric(future[col], errors="coerce")
                row[label] = float(values.sum()) if col.startswith(("heating_degree_day", "cooling_degree_day", "solar_irradiation_sum")) else (float(values.mean()) if values.notna().any() else float("nan"))
            else:
                row[label] = float("nan")
        row["Interpretation"] = "Weather proxy only; not a national weighted index or price forecast."
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def fetch_provider_table(url: str, api_key: str = "", auth_mode: str = "Bearer token",
                         header_name: str = "Authorization", response_format: str = "Auto",
                         query_params: Optional[dict] = None, _http_get_bytes_fn=None) -> pd.DataFrame:
    """Generic adapter for a licensed market-data REST endpoint; vendor-specific terms apply."""
    if not url.strip().lower().startswith("https://"):
        raise ValueError("Use an HTTPS endpoint URL supplied by your licensed data vendor.")
    headers = {"Accept": "application/json,text/csv,*/*"}
    if auth_mode == "Bearer token":
        if api_key.strip(): headers[header_name.strip() or "Authorization"] = f"Bearer {api_key.strip()}"
    elif auth_mode == "API-key header":
        if api_key.strip(): headers[header_name.strip() or "X-API-Key"] = api_key.strip()
    elif auth_mode == "No authentication":
        pass
    else:
        raise ValueError("Unsupported provider authentication mode.")
    if query_params is not None and not isinstance(query_params, dict):
        raise ValueError("Provider query parameters must be a JSON object of key/value pairs.")
    request_http = _http_get_bytes_fn or _http_get_bytes
    body, content_type = request_http(url.strip(), params=query_params, headers=headers)
    fmt = response_format
    if fmt == "Auto":
        fmt = "CSV" if ("csv" in content_type.lower() or body[:100].lower().find(b"delivery_month") >= 0) else "JSON"
    try:
        if fmt == "CSV":
            frame = pd.read_csv(io.BytesIO(body))
        else:
            payload = json.loads(body.decode("utf-8-sig"))
            rows = _payload_rows(payload)
            if not rows and isinstance(payload, dict):
                # If the vendor returns a keyed object of arrays, try normalisation.
                for v in payload.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        rows = v
                        break
            frame = _flatten_records(rows)
            if frame.empty and isinstance(payload, dict):
                frame = pd.json_normalize(payload)
    except Exception as exc:
        raise RuntimeError("Could not parse the provider response as the selected JSON/CSV format.") from exc
    if frame.empty:
        raise RuntimeError("Provider returned an empty table. Check endpoint, permissions, response format and query parameters.")
    frame["_provider_fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return frame


def normalise_provider_curve(raw: pd.DataFrame, product: str, delivery_col: str,
                             price_col: str) -> pd.DataFrame:
    """Map a vendor response to the desk's strict monthly forward-curve schema."""
    if product not in CURVE_PRODUCTS:
        raise ValueError(f"Unsupported canonical product: {product}")
    if delivery_col not in raw.columns or price_col not in raw.columns:
        raise ValueError("Choose valid delivery date and price columns from the provider response.")
    out = pd.DataFrame({
        "delivery_month": pd.to_datetime(raw[delivery_col], errors="coerce"),
        "product": product,
        "price": _number_series(raw[price_col]),
        "currency": CURVE_PRODUCTS[product]["currency"],
        "unit": CURVE_PRODUCTS[product]["unit"],
    })
    if out["delivery_month"].isna().any():
        raise ValueError("Some delivery dates could not be parsed. Map a column containing ISO dates such as YYYY-MM-01, and convert contract labels before import.")
    if out["price"].isna().any():
        raise ValueError("Some provider prices are missing or non-numeric. No implicit conversion was applied.")
    return validate_curve(out)


def _normalised_col_name(value: object) -> str:
    """Case/punctuation-insensitive column identifier for provider schemas."""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if frame is None or frame.empty:
        return None
    columns = list(frame.columns)
    normalised = {_normalised_col_name(c): c for c in columns}
    for candidate in candidates:
        key = _normalised_col_name(candidate)
        if key in normalised:
            return normalised[key]
    # Some provider payloads flatten nested objects (e.g. inventory_gwh).
    for candidate in candidates:
        key = _normalised_col_name(candidate)
        matches = [c for c in columns if _normalised_col_name(c).endswith(key)]
        if len(matches) == 1:
            return matches[0]
    return None


def _numeric_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    col = _find_column(frame, candidates)
    if col is None:
        return None
    parsed = _number_series(frame[col])
    return col if parsed.notna().any() else None


def _frame_with_time(frame: pd.DataFrame, date_candidates: Optional[Sequence[str]] = None) -> tuple[pd.DataFrame, Optional[str]]:
    """Return a copy with a UTC sort key, supporting date-only and timestamp columns."""
    if frame is None or frame.empty:
        return pd.DataFrame(), None
    candidates = date_candidates or (
        "observation_date", "gasDayStart", "gas_day", "date", "timestamp", "periodFrom", "periodfrom"
    )
    col = _find_column(frame, candidates)
    if col is None:
        return frame.copy(), None
    out = frame.copy()
    parsed = pd.to_datetime(out[col], errors="coerce", utc=True)
    out["_meridian_sort_time"] = parsed
    out = out.loc[parsed.notna()].sort_values("_meridian_sort_time").reset_index(drop=True)
    return out, col


def _latest_row(frame: pd.DataFrame, date_candidates: Optional[Sequence[str]] = None,
                prefer_aggregate: bool = False) -> Optional[pd.Series]:
    """Get the latest observation without accidentally summing facilities or series."""
    out, time_col = _frame_with_time(frame, date_candidates)
    if out.empty:
        return None
    if prefer_aggregate:
        name_col = _find_column(out, ("name", "country", "label", "dataset"))
        code_col = _find_column(out, ("code", "countryCode", "country_code"))
        mask = pd.Series(False, index=out.index)
        if name_col:
            mask |= out[name_col].astype(str).str.strip().str.casefold().isin({"eu", "europe", "european union", "europe aggregate"})
        if code_col:
            mask |= out[code_col].astype(str).str.strip().str.casefold().isin({"eu", "europe"})
        if mask.any():
            out = out.loc[mask].copy()
    return out.iloc[-1]


def _latest_metric(frame: pd.DataFrame, value_candidates: Sequence[str],
                   date_candidates: Optional[Sequence[str]] = None,
                   prefer_aggregate: bool = False) -> tuple[Optional[float], Optional[pd.Series], Optional[str]]:
    """Return latest numeric value, source row and matched column."""
    row = _latest_row(frame, date_candidates, prefer_aggregate=prefer_aggregate)
    if row is None:
        return None, None, None
    col = _numeric_column(frame, value_candidates)
    if col is None:
        return None, row, None
    value = _number_series(pd.Series([row.get(col)])).iloc[0]
    if pd.isna(value) or not math.isfinite(float(value)):
        return None, row, col
    return float(value), row, col


def _latest_observation_date(frame: pd.DataFrame,
                             date_candidates: Optional[Sequence[str]] = None) -> Optional[str]:
    if frame is None or frame.empty:
        return None
    column = _find_column(frame, date_candidates or ("observation_date", "gasDayStart", "gas_day", "date", "timestamp", "periodFrom", "periodfrom"))
    if column is None:
        return None
    stamp = pd.to_datetime(frame[column], errors="coerce", utc=True).max()
    if pd.isna(stamp):
        return None
    return stamp.strftime("%Y-%m-%d %H:%M UTC" if (stamp.hour or stamp.minute or stamp.second) else "%Y-%m-%d")


def build_live_snapshot() -> dict[str, dict]:
    """Summarise current session feeds for dashboards, without fabricating missing values."""
    snapshot: dict[str, dict] = {}
    # GIE AGSI+ gas storage. EU type requests contain an aggregate; country requests stay country-level.
    agsi = _live_dataset_frame("gie_agsi")
    if not agsi.empty:
        full, row, _ = _latest_metric(agsi, ("full", "fillLevel", "fill_level"), prefer_aggregate=True)
        storage, _, _ = _latest_metric(agsi, ("gasInStorage", "gas_in_storage"), prefer_aggregate=True)
        capacity, _, _ = _latest_metric(agsi, ("workingGasVolume", "working_gas_volume"), prefer_aggregate=True)
        if full is None and storage is not None and capacity and capacity > 0:
            full = storage / capacity * 100.0
        date_label = _latest_observation_date(agsi, ("observation_date", "gasDayStart", "gas_day"))
        scope = str(_live_item_meta("gie_agsi").get("scope", "selected API scope"))
        change = None
        date_col = _find_column(agsi, ("observation_date", "gasDayStart", "gas_day"))
        storage_col = _find_column(agsi, ("gasInStorage", "gas_in_storage"))
        if date_col and storage_col:
            hist = agsi.copy()
            hist["_day"] = pd.to_datetime(hist[date_col], errors="coerce")
            hist["_storage"] = _number_series(hist[storage_col])
            name_col = _find_column(hist, ("name", "country"))
            code_col = _find_column(hist, ("code", "countryCode", "country_code"))
            if scope == "Europe / EU datasets":
                aggregate_mask = pd.Series(False, index=hist.index)
                if name_col: aggregate_mask |= hist[name_col].astype(str).str.casefold().isin({"eu", "europe", "european union"})
                if code_col: aggregate_mask |= hist[code_col].astype(str).str.casefold().isin({"eu", "europe"})
                if aggregate_mask.any(): hist = hist.loc[aggregate_mask]
            hist = hist.dropna(subset=["_day", "_storage"]).sort_values("_day")
            if len(hist) >= 8:
                change = float(hist.iloc[-1]["_storage"] - hist.iloc[-8]["_storage"])
        snapshot["storage"] = {"fill_pct": full, "inventory_twh": storage, "capacity_twh": capacity,
                               "date": date_label, "scope": scope, "change_7_reports_twh": change,
                               "fetched_at": _live_item_fetched_at("gie_agsi")}

    # GIE ALSI+ LNG fields have different units: inventory is commonly 10^3 m3 LNG, send-out GWh/d.
    alsi = _live_dataset_frame("gie_alsi")
    if not alsi.empty:
        inventory, row, inventory_col = _latest_metric(alsi, ("inventory", "inventory_lng", "inventory_gwh", "inventory_twh"), prefer_aggregate=True)
        sendout, _, _ = _latest_metric(alsi, ("sendOut", "sendout", "send_out", "sendout_gwh"), prefer_aggregate=True)
        dtmi, _, _ = _latest_metric(alsi, ("dtmi",), prefer_aggregate=True)
        inventory_unit = "10³ m³ LNG" if inventory_col and _normalised_col_name(inventory_col) == "inventory" else ("GWh" if inventory_col and "gwh" in inventory_col.casefold() else ("TWh" if inventory_col and "twh" in inventory_col.casefold() else "provider-defined unit"))
        snapshot["lng"] = {"inventory": inventory, "inventory_unit": inventory_unit, "sendout": sendout,
                           "sendout_unit": "GWh/d" if sendout is not None else None, "capacity": dtmi,
                           "date": _latest_observation_date(alsi, ("observation_date", "gasDayStart", "gas_day")),
                           "scope": str(_live_item_meta("gie_alsi").get("scope", "selected API scope")),
                           "fetched_at": _live_item_fetched_at("gie_alsi")}

    price_frame = _live_dataset_frame("entsoe_price")
    if not price_frame.empty and "value" in price_frame.columns:
        p = price_frame.copy()
        p["timestamp"] = pd.to_datetime(p.get("timestamp"), errors="coerce", utc=True)
        p["value"] = _number_series(p["value"])
        p = p.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
        if not p.empty:
            recent_cutoff = p["timestamp"].max() - pd.Timedelta(days=7)
            recent = p.loc[p["timestamp"].ge(recent_cutoff)]
            meta = _live_item_meta("entsoe_price")
            snapshot["power_price"] = {"value": float(p.iloc[-1]["value"]), "unit": str(p.iloc[-1].get("unit", "EUR/MWh")),
                "timestamp": p.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
                "zone": str(meta.get("zone", "selected bidding zone")),
                "sample_mean": float(recent["value"].mean()) if not recent.empty else None,
                "sample_min": float(recent["value"].min()) if not recent.empty else None,
                "sample_max": float(recent["value"].max()) if not recent.empty else None,
                "negative_observations": int((recent["value"] < 0).sum()),
                "fetched_at": _live_item_fetched_at("entsoe_price")}

    load_frame = _live_dataset_frame("entsoe_load")
    if not load_frame.empty and "value" in load_frame.columns:
        load = load_frame.copy()
        load["timestamp"] = pd.to_datetime(load.get("timestamp"), errors="coerce", utc=True)
        load["value"] = _number_series(load["value"])
        load = load.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
        if not load.empty:
            snapshot["power_load"] = {"value": float(load.iloc[-1]["value"]), "unit": str(load.iloc[-1].get("unit", "MW")),
                "timestamp": load.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
                "zone": str(_live_item_meta("entsoe_load").get("zone", "selected bidding zone")),
                "fetched_at": _live_item_fetched_at("entsoe_load")}

    gen = _live_dataset_frame("entsoe_generation")
    if not gen.empty and {"timestamp", "value", "series"}.issubset(gen.columns):
        gen = gen.copy()
        gen["timestamp"] = pd.to_datetime(gen["timestamp"], errors="coerce", utc=True)
        gen["value"] = _number_series(gen["value"])
        gen = gen.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
        if not gen.empty:
            latest_t = gen["timestamp"].max()
            latest_gen = gen.loc[gen["timestamp"].eq(latest_t)]
            labels = latest_gen["series"].astype(str).str.casefold()
            renewable = latest_gen.loc[labels.str.contains("wind onshore|wind offshore|solar", regex=True) & ~labels.str.contains("consumption"), "value"].sum()
            snapshot["generation"] = {"timestamp": latest_t.strftime("%Y-%m-%d %H:%M UTC"),
                "wind_solar_mw": float(renewable) if not latest_gen.empty else None,
                "series_count": int(latest_gen["series"].nunique()),
                "zone": str(_live_item_meta("entsoe_generation").get("zone", "selected bidding zone")),
                "fetched_at": _live_item_fetched_at("entsoe_generation")}

    flow = _live_dataset_frame("entsog_flow")
    if not flow.empty:
        value_col = _numeric_column(flow, ("value_numeric", "value", "quantity", "flow"))
        row = _latest_row(flow, ("timestamp", "periodFrom", "period_from", "date"))
        if row is not None and value_col:
            val = _number_series(pd.Series([row.get(value_col)])).iloc[0]
            snapshot["gas_flow"] = {"value": float(val) if pd.notna(val) else None,
                "unit": str(row.get("unit", row.get("unitName", "check provider unit"))),
                "timestamp": str(row.get("timestamp", row.get("periodFrom", row.get("date", "")))),
                "label": str(_live_item_meta("entsog_flow").get("indicator", "Selected ENTSOG indicator")),
                "point_direction": str(_live_item_meta("entsog_flow").get("point_direction", "selected point")),
                "fetched_at": _live_item_fetched_at("entsog_flow")}

    weather = _live_dataset_frame("weather_daily")
    if not weather.empty:
        weather_for_snapshot = weather.copy()
        # Prefer a current-day row when the dataset contains both recent days and forecasts;
        # choosing the maximum date would incorrectly report the end of the forecast horizon.
        if "forecast_period" in weather_for_snapshot.columns:
            current_rows = weather_for_snapshot.loc[weather_for_snapshot["forecast_period"].astype(str).eq("Today (may be partial)")]
            if not current_rows.empty:
                weather_for_snapshot = current_rows
        sort_col = "date" if "date" in weather_for_snapshot.columns else "timestamp"
        if sort_col in weather_for_snapshot.columns:
            weather_for_snapshot[sort_col] = pd.to_datetime(weather_for_snapshot[sort_col], errors="coerce")
            weather_for_snapshot = weather_for_snapshot.sort_values(sort_col)
        row = weather_for_snapshot.iloc[0] if not weather_for_snapshot.empty else None
        if row is not None:
            def row_num(aliases):
                col = _find_column(weather_for_snapshot, aliases)
                if not col: return None
                x = _number_series(pd.Series([row.get(col)])).iloc[0]
                return float(x) if pd.notna(x) and math.isfinite(float(x)) else None
            metadata_location = _live_item_meta("weather_daily").get("location", "selected location")
            snapshot["weather"] = {"date": str(row.get("date", "")),
                "temperature_c": row_num(("temperature_mean_c", "temperature_2m")),
                "wind_kmh": row_num(("wind_speed_mean_kmh", "wind_speed_10m")),
                "wind_100m_kmh": row_num(("wind_speed_100m_mean_kmh", "wind_speed_100m")),
                "solar_wm2": row_num(("solar_radiation_mean_wm2", "shortwave_radiation")),
                "location": str(row.get("location", metadata_location)),
                "forecast_period": str(row.get("forecast_period", "Weather data")),
                "fetched_at": _live_item_fetched_at("weather_daily")}
    return snapshot


class ProviderHTTPError(RuntimeError):
    """HTTP failure that keeps the provider's response body for a useful message."""

    def __init__(self, message: str, status: int, body: bytes = b"") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def _http_get_bytes(url: str, params: Optional[dict] = None,
                    headers: Optional[dict] = None,
                    timeout: int = HTTP_TIMEOUT_SECONDS) -> tuple[bytes, str]:
    """Small dependency-free HTTP GET with one retry. Errors never echo query tokens or headers."""
    final_url = _build_url(url, params)
    req_headers = {"User-Agent": "MERIDIAN-Energy-Desk/2.0", "Accept": "application/json,*/*", "Accept-Encoding": "gzip"}
    if headers:
        req_headers.update({str(k): str(v) for k, v in headers.items() if v is not None})
    request = Request(final_url, headers=req_headers, method="GET")
    for attempt in (1, 2):
        try:
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                encoding = response.headers.get("Content-Encoding", "")
                body = response.read(MAX_API_RESPONSE_BYTES + 1)
                if len(body) > MAX_API_RESPONSE_BYTES:
                    raise RuntimeError("API response exceeded the 20 MB safety limit. Narrow the requested dates or filters.")
                return (gzip.decompress(body) if encoding == "gzip" else body), content_type
        except HTTPError as exc:
            # Busy or briefly failing servers often recover, so these get one more try.
            if attempt == 1 and exc.code in RETRY_HTTP_CODES:
                sleep(RETRY_WAIT_SECONDS)
                continue
            # Do not include exc.url because some vendors require tokens in URL query strings.
            try:
                detail = exc.read(8192)
            except Exception:
                detail = b""
            raise ProviderHTTPError(f"API returned HTTP {exc.code}. Check access rights, credentials, dates and query parameters.", exc.code, detail) from None
        except URLError as exc:
            if attempt == 1:
                sleep(RETRY_WAIT_SECONDS)
                continue
            reason = getattr(exc, "reason", "network error")
            raise RuntimeError(f"Could not reach the data provider ({reason}). Check your internet connection and provider availability.") from None
        except TimeoutError:
            if attempt == 1:
                sleep(RETRY_WAIT_SECONDS)
                continue
            raise RuntimeError("The data provider timed out. Try a shorter date range.") from None
    raise RuntimeError("The data provider could not be reached.")


def _direct_children(parent, name: str) -> list:
    return [child for child in parent if _xml_local_name(child.tag) == name]


def _child_text(parent, name: str) -> Optional[str]:
    for child in parent:
        if _xml_local_name(child.tag) == name and child.text and child.text.strip():
            return child.text.strip()
    return None


def parse_entsoe_xml(content: bytes, series_kind: str = "price") -> pd.DataFrame:
    """Parse standard ENTSO-E Publication/GL MarketDocument time series into rows.

    Handles compressed A03 series, which omit a point when the value is unchanged, and
    separates generation from consumption reported under the same production type.
    """
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            xml_names = [n for n in zf.namelist() if n.lower().endswith((".xml", ".xml.gz"))]
            if not xml_names:
                raise ValueError("ENTSO-E returned a ZIP without an XML document.")
            content = zf.read(xml_names[0])
            if xml_names[0].lower().endswith(".xml.gz") or content[:2] == b"\x1f\x8b":
                content = gzip.decompress(content)
    elif content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError("ENTSO-E response could not be parsed as XML.") from exc
    if _xml_local_name(root.tag) == "Acknowledgement_MarketDocument":
        reason = _xml_text(root, ["text"]) or _xml_text(root, ["code"]) or "No matching market document was returned."
        raise RuntimeError(f"ENTSO-E acknowledgement: {reason}")
    kind = series_kind.lower()
    wanted = ("price.amount", "quantity") if kind == "price" else ("quantity", "price.amount")
    frames = []
    all_series = _direct_children(root, "TimeSeries")
    if kind == "price":
        # Germany-Luxembourg and Austria publish two auctions: the official SDAC price is sequence 1 and
        # EXAA is sequence 2, with the same timestamps. Keep one auction, and treat unlabelled series as 1.
        def auction(ts) -> int:
            text = next((child.text for child in ts if _xml_local_name(child.tag).startswith("classificationSequence")), None)
            try:
                return int(str(text).strip())
            except ValueError:
                return 1
        wanted_auction = min((auction(ts) for ts in all_series), default=1)
        all_series = [ts for ts in all_series if auction(ts) == wanted_auction]
    for ts in all_series:
        psr_node = _direct_children(ts, "MktPSRType")
        psr_code = (_child_text(psr_node[0], "psrType") if psr_node else "") or ""
        curve_type = _child_text(ts, "curveType") or ""
        series_name = ENTSOE_PSR_NAMES.get(psr_code, psr_code or ("Day-ahead price" if kind == "price" else kind.title()))
        if kind == "generation" and _direct_children(ts, "outBiddingZone_Domain.mRID"):
            series_name = f"{series_name} (consumption)"
        for period in _direct_children(ts, "Period"):
            interval = _direct_children(period, "timeInterval")
            start_txt = _child_text(interval[0], "start") if interval else _child_text(period, "start")
            end_txt = _child_text(interval[0], "end") if interval else None
            resolution = _child_text(period, "resolution")
            if not start_txt or not resolution:
                continue
            try:
                period_start = pd.Timestamp(start_txt)
                period_start = period_start.tz_localize("UTC") if period_start.tzinfo is None else period_start.tz_convert("UTC")
                delta = _parse_duration(resolution)
            except Exception:
                continue
            positions, values = [], []
            for point in period:
                if _xml_local_name(point.tag) != "Point":
                    continue
                found: dict = {}
                for child in point:
                    found[_xml_local_name(child.tag)] = child.text
                value_text = found.get(wanted[0]) or found.get(wanted[1])
                try:
                    positions.append(int(found["position"]))
                    values.append(float(value_text.replace(",", ".")))
                except (KeyError, AttributeError, TypeError, ValueError):
                    if len(positions) > len(values):
                        positions.pop()
            if not positions:
                continue
            order = np.argsort(positions, kind="stable")
            positions_array, values_array = np.asarray(positions)[order], np.asarray(values)[order]
            if curve_type == "A03":
                # A missing position repeats the previous value until the period ends.
                last_position = int(positions_array[-1])
                if end_txt:
                    try:
                        period_end = pd.Timestamp(end_txt)
                        period_end = period_end.tz_localize("UTC") if period_end.tzinfo is None else period_end.tz_convert("UTC")
                        last_position = max(last_position, int((period_end - period_start) / delta))
                    except Exception:
                        pass
                full = np.arange(int(positions_array[0]), last_position + 1)
                values_array = values_array[np.searchsorted(positions_array, full, side="right") - 1]
                positions_array = full
            frames.append(pd.DataFrame({
                "timestamp": period_start + pd.to_timedelta(positions_array - 1, unit="s") * delta.total_seconds(),
                "value": values_array, "series": series_name, "psr_code": psr_code, "resolution": resolution,
                "unit": "EUR/MWh" if kind == "price" else "MW", "_step": delta.total_seconds(),
            }))
    if not frames:
        msg = _xml_text(root, ["text"]) or "No observations found in ENTSO-E XML response."
        raise RuntimeError(f"ENTSO-E returned no usable observations. {msg}")
    # Where a document repeats a slot (for example 15- and 60-minute auctions), keep the finest.
    result = (pd.concat(frames, ignore_index=True).sort_values(["timestamp", "series", "_step"])
              .drop_duplicates(subset=["timestamp", "series"], keep="first").drop(columns="_step"))
    return result.reset_index(drop=True)


def _entsoe_error_message(exc: ProviderHTTPError) -> str:
    """Turn an ENTSO-E HTTP failure into its own explanation. Never includes the token."""
    if exc.status == 401:
        return "ENTSO-E rejected the token (HTTP 401). Check that it is correct and that REST access is enabled on your account."
    if exc.status == 429:
        return "ENTSO-E is rate-limiting requests (HTTP 429). Wait a minute and try again."
    try:
        root = ET.fromstring(exc.body)
        reason = _xml_text(root, ["text"])
        if reason:
            return f"ENTSO-E: {reason}"
    except Exception:
        pass
    return f"ENTSO-E returned HTTP {exc.status}. Check the zone, dates and that this dataset exists for the selection."


def fetch_entsoe_series(token: str, series_kind: str, zone_name: str,
                        start_date: date, end_date_exclusive: date,
                        source_zone: Optional[str] = None,
                        target_zone: Optional[str] = None,
                        _http_fn=None, progress=None) -> pd.DataFrame:
    """Fetch ENTSO-E day-ahead prices, actual load, generation or cross-border flow.

    Long periods are split into requests of up to 360 days and joined, so up to three years can be
    requested at once. A window with no published data is skipped rather than failing the rest.
    """
    if not token.strip():
        raise ValueError("Enter your ENTSO-E Web API token. REST access must be enabled on your account.")
    if end_date_exclusive <= start_date:
        raise ValueError("End date must be later than start date (end date is exclusive).")
    if (end_date_exclusive - start_date).days > ENTSOE_MAX_DAYS:
        raise ValueError(f"Select at most {ENTSOE_MAX_DAYS} delivery days per request.")
    if series_kind == "Physical flow":
        if source_zone not in ENTSOE_ZONES or target_zone not in ENTSOE_ZONES:
            raise ValueError("Select both a source and destination bidding zone.")
        if source_zone == target_zone:
            raise ValueError("Physical flow requires two different bidding zones.")
        document_type = "A11"
        # ENTSO-E convention: in_Domain receives the flow, out_Domain sends it.
        base_params = {"in_Domain": ENTSOE_ZONES[target_zone]["eic"], "out_Domain": ENTSOE_ZONES[source_zone]["eic"]}
        parser_kind = "flow"
    else:
        if zone_name not in ENTSOE_ZONES:
            raise ValueError("Select a supported ENTSO-E bidding zone.")
        eic = ENTSOE_ZONES[zone_name]["eic"]
        if series_kind == "Day-ahead price":
            document_type, parser_kind = "A44", "price"
            base_params = {"in_Domain": eic, "out_Domain": eic, "contract_MarketAgreement.type": "A01"}
        elif series_kind == "Actual load":
            document_type, parser_kind = "A65", "load"
            base_params = {"processType": "A16", "outBiddingZone_Domain": eic}
        elif series_kind == "Actual generation by type":
            document_type, parser_kind = "A75", "generation"
            base_params = {"processType": "A16", "in_Domain": eic}
        elif series_kind == "Day-ahead load forecast":
            document_type, parser_kind = "A65", "load"
            base_params = {"processType": "A01", "outBiddingZone_Domain": eic}
        elif series_kind == "Day-ahead wind and solar forecast":
            document_type, parser_kind = "A69", "generation"
            base_params = {"processType": "A01", "in_Domain": eic}
        else:
            raise ValueError("Unsupported ENTSO-E series type.")
    # ENTSO-E expects UTC period boundaries in yyyyMMddHHmm format.
    zone = ZoneInfo(ENTSOE_ZONES.get(zone_name, {}).get("timezone", "Europe/Paris"))
    request_http = _http_fn or _http_get_bytes
    # ENTSO-E serves up to a year per request. Generation holds many series, so it uses shorter windows.
    span = ENTSOE_GENERATION_WINDOW_DAYS if parser_kind == "generation" and document_type == "A75" else ENTSOE_WINDOW_DAYS

    def split(first: date, last: date, days: int) -> list[tuple[date, date]]:
        parts, cursor = [], first
        while cursor < last:
            following = min(cursor + timedelta(days=days), last)
            parts.append((cursor, following))
            cursor = following
        return parts

    def request(window_start: date, window_end: date):
        params = dict(base_params)
        params.update({
            "documentType": document_type,
            "periodStart": datetime.combine(window_start, time.min, tzinfo=zone).astimezone(timezone.utc).strftime("%Y%m%d%H%M"),
            "periodEnd": datetime.combine(window_end, time.min, tzinfo=zone).astimezone(timezone.utc).strftime("%Y%m%d%H%M"),
            "securityToken": token.strip(),
        })
        try:
            body, _ = request_http(ENTSOE_BASE, params=params, headers={"Accept": "application/xml, application/zip, */*"})
            return parse_entsoe_xml(body, parser_kind), ""
        except ProviderHTTPError as exc:
            message = _entsoe_error_message(exc)
            if "no matching data" not in message.casefold():
                raise RuntimeError(message) from None
            return None, message
        except RuntimeError as exc:
            if "no matching data" not in str(exc).casefold():
                raise
            return None, str(exc)

    windows = split(start_date, end_date_exclusive, span)
    frames, no_data = [], ""
    for index, (window_start, window_end) in enumerate(windows):
        try:
            results = [request(window_start, window_end)]
        except RuntimeError:
            # If a long window is refused, ask again in month-sized pieces before giving up.
            if (window_end - window_start).days <= ENTSOE_FALLBACK_WINDOW_DAYS:
                raise
            results = [request(a, b) for a, b in split(window_start, window_end, ENTSOE_FALLBACK_WINDOW_DAYS)]
        for frame, message in results:
            if frame is not None:
                frames.append(frame)
            elif message:
                no_data = message
        if progress is not None:
            progress((index + 1) / len(windows))
    if not frames:
        raise RuntimeError(no_data or "ENTSO-E returned no usable observations for this selection.")
    frame = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["timestamp", "series"]).sort_values(["timestamp", "series"]).reset_index(drop=True))
    frame["_source"] = "ENTSO-E Transparency Platform"
    return frame


def fetch_ecb_rate(quote_currency: str, _http_fn=None) -> tuple[float, str]:
    """Latest ECB euro reference rate: units of quote_currency per 1 EUR, and its date."""
    code = str(quote_currency).strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError("Use a three-letter currency code.")
    request_http = _http_fn or _http_get_bytes
    body, _ = request_http(f"{ECB_FX_BASE}/D.{code}.EUR.SP00.A", params={"lastNObservations": 1, "format": "csvdata"},
                           headers={"Accept": "text/csv"})
    frame = pd.read_csv(io.BytesIO(body))
    if frame.empty or "OBS_VALUE" not in frame.columns:
        raise RuntimeError(f"ECB returned no {code} reference rate.")
    rate = float(frame["OBS_VALUE"].iloc[-1])
    if not math.isfinite(rate) or rate <= 0:
        raise RuntimeError(f"ECB returned an invalid {code} reference rate.")
    return rate, str(frame["TIME_PERIOD"].iloc[-1]) if "TIME_PERIOD" in frame.columns else ""


def fetch_weather_archive(location_names: Sequence[str], years: int = 10, _json_get_fn=None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Daily temperatures from Open-Meteo's archive, with calendar-day normals made from them."""
    request_json = _json_get_fn or _json_get
    end = date.today() - timedelta(days=7)
    start = date(end.year - int(years), end.month, 1)

    def fetch_one(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        loc = WEATHER_LOCATIONS[name]
        payload = request_json(OPEN_METEO_ARCHIVE_BASE, params={
            "latitude": loc["latitude"], "longitude": loc["longitude"], "start_date": start.isoformat(),
            "end_date": end.isoformat(), "daily": "temperature_2m_mean", "timezone": loc["timezone"]})
        daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
        if not daily.get("time"):
            raise RuntimeError("no archive data returned")
        raw = pd.DataFrame({"date": daily["time"], "temperature": daily.get("temperature_2m_mean", [])})
        normals = temperature_normals_from_daily(raw)
        normals["location"] = name
        kept = raw.assign(temperature=pd.to_numeric(raw["temperature"], errors="coerce")).dropna()
        kept = kept.assign(location=name, hdd=(18.0 - kept["temperature"]).clip(lower=0))
        return kept, normals

    days, normals, failures = [], [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(fetch_one, name): name for name in location_names}
        for future in as_completed(future_map):
            try:
                kept, normal = future.result()
                days.append(kept)
                normals.append(normal)
            except Exception as exc:
                failures.append(f"{future_map[future]}: {exc}")
    return (pd.concat(days, ignore_index=True) if days else pd.DataFrame(),
            pd.concat(normals, ignore_index=True) if normals else pd.DataFrame(), failures)


def fetch_power_weather(location_names: Sequence[str], days: int = 430, _json_get_fn=None) -> tuple[pd.DataFrame, list[str]]:
    """Daily temperature, wind at 100 m and solar radiation from Open-Meteo's archive, for the price estimate."""
    request_json = _json_get_fn or _json_get
    end = date.today() - timedelta(days=5)
    start = end - timedelta(days=int(days))
    def fetch_one(name: str) -> pd.DataFrame:
        loc = WEATHER_LOCATIONS[name]
        payload = request_json(OPEN_METEO_ARCHIVE_BASE, params={
            "latitude": loc["latitude"], "longitude": loc["longitude"], "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": "temperature_2m,wind_speed_100m,shortwave_radiation", "timezone": loc["timezone"]})
        hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}
        if not hourly.get("time"):
            raise RuntimeError("no archive data returned")
        frame = pd.DataFrame({"date": [str(t)[:10] for t in hourly["time"]], "temperature_mean_c": pd.to_numeric(pd.Series(hourly.get("temperature_2m", [])), errors="coerce"),
                              "wind_speed_100m_mean_kmh": pd.to_numeric(pd.Series(hourly.get("wind_speed_100m", [])), errors="coerce"),
                              "solar_radiation_mean_wm2": pd.to_numeric(pd.Series(hourly.get("shortwave_radiation", [])), errors="coerce")})
        return frame.groupby("date", as_index=False).mean().assign(location=name)
    frames, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(fetch_one, name): name for name in location_names}
        for future in as_completed(future_map):
            try:
                frames.append(future.result())
            except Exception as exc:
                failures.append(f"{future_map[future]}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), failures


def fetch_zone_weather(zone: str, start: Optional[date] = None, end: Optional[date] = None, forecast: bool = False,
                       api_key: str = "", commercial_endpoint: bool = False, _json_get_fn=None) -> pd.DataFrame:
    """Daily heating demand, wind at 100 m and solar radiation averaged over a zone's points, in one request.

    The archive gives past days. The forecast gives the last week and the next three days, which also
    covers the days the archive hasn't published yet.
    """
    points = ZONE_WEATHER_POINTS[zone]
    timezone_name = ENTSOE_ZONES[zone]["timezone"]
    params = {"latitude": ",".join(f"{lat:.2f}" for lat, _ in points), "longitude": ",".join(f"{lon:.2f}" for _, lon in points),
              "hourly": "temperature_2m,wind_speed_100m,shortwave_radiation", "timezone": timezone_name}
    if forecast:
        url = OPEN_METEO_CUSTOMER_BASE if commercial_endpoint else OPEN_METEO_FORECAST_BASE
        params.update({"past_days": 7, "forecast_days": 3})
        if api_key.strip():
            params["apikey"] = api_key.strip()
    else:
        url = OPEN_METEO_ARCHIVE_BASE
        params.update({"start_date": start.isoformat(), "end_date": end.isoformat()})
    payload = (_json_get_fn or _json_get)(url, params=params)
    answers = payload if isinstance(payload, list) else [payload]
    per_point = []
    for answer in answers:
        hourly = answer.get("hourly", {}) if isinstance(answer, dict) else {}
        if not hourly.get("time"):
            continue
        frame = pd.DataFrame({"date": [str(t)[:10] for t in hourly["time"]],
                              "temperature": pd.to_numeric(pd.Series(hourly.get("temperature_2m", [])), errors="coerce"),
                              "wind": pd.to_numeric(pd.Series(hourly.get("wind_speed_100m", [])), errors="coerce"),
                              "solar": pd.to_numeric(pd.Series(hourly.get("shortwave_radiation", [])), errors="coerce")})
        # A day counts only with most of its hours, so a partly published day can't pass as a calm one.
        counts = frame.groupby("date")[["temperature", "wind", "solar"]].count().min(axis=1)
        daily = frame.groupby("date")[["temperature", "wind", "solar"]].mean().loc[lambda f: counts.reindex(f.index) >= 20]
        per_point.append(daily.assign(hdd=(18.0 - daily["temperature"]).clip(lower=0)))
    if not per_point:
        raise RuntimeError("Open-Meteo returned no weather for this zone.")
    joined = pd.concat(per_point)
    grouped = joined.groupby(level=0)
    means = grouped[["temperature", "hdd", "wind", "solar"]].mean().loc[grouped.size() >= max(1, len(points) // 2)]
    if means.empty:
        complete = sum(not part.empty for part in per_point)
        raise RuntimeError(f"Open-Meteo gave complete days for {complete} of {len(points)} points, too few to stand for the zone.")
    out = means.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out.assign(zone=zone, points=len(per_point))[["zone", "date", "temperature", "hdd", "wind", "solar", "points"]]


# TradingView through tvdatafeed, an unofficial library: it can break when TradingView changes, prices are
# delayed, and TradingView's terms don't allow automated access. Personal use, at the user's choice.
TRADINGVIEW_CURVES = {"TTF Natural Gas": {"exchange": "NYMEX", "root": "TTF", "name": "NYMEX Dutch TTF month futures"}}
TRADINGVIEW_MONTH_CODES = "FGHJKMNQUVXZ"
TRADINGVIEW_INSTALL = "pip install tradingview-datafeed"   # or the maintained fork: pip install git+https://github.com/rongardF/tvdatafeed.git
TRADINGVIEW_HISTORY_BARS = 60          # first fetch: about three months of daily closes, so P&L and risk start with real history
TRADINGVIEW_UPDATE_BARS = 7
TRADINGVIEW_MAX_MONTHS = 48
TRADINGVIEW_CARRY_DAYS = 5             # a contract with no trade that day keeps its last close for up to this many days


def tradingview_available() -> bool:
    import importlib.util
    import sys
    if sys.modules.get("tvDatafeed") is not None:
        return True
    try:
        return importlib.util.find_spec("tvDatafeed") is not None
    except (ImportError, ValueError):
        return False


def tradingview_symbol(root: str, month: date) -> str:
    return f"{root}{TRADINGVIEW_MONTH_CODES[month.month - 1]}{month.year}"


def fetch_tradingview_curve(product: str, today: date, n_bars: int, username: str = "", password: str = "", _client=None) -> pd.DataFrame:
    """Daily closes of each month contract, one row per day and delivery month, ready for parse_curve_table.

    A contract is used only on days before its delivery month starts. A day where fewer than half the
    contracts have a price, such as a holiday, is left out.
    """
    spec = TRADINGVIEW_CURVES[product]
    if _client is None:
        try:
            from tvDatafeed import TvDatafeed, Interval
        except ImportError:
            raise RuntimeError(f"tvdatafeed isn't installed. Run: {TRADINGVIEW_INSTALL}") from None
        import logging
        logging.getLogger("tvDatafeed").setLevel(logging.CRITICAL)   # its own messages are replaced by the summary below
        client, interval = TvDatafeed(username or None, password or None), Interval.in_daily
    else:
        client, interval = _client, "1D"
    # With history, start from contracts that were the front month at the start of that history.
    first = _add_months(date(today.year, today.month, 1), 1 - (math.ceil(n_bars / 21) if n_bars > TRADINGVIEW_UPDATE_BARS else 0))
    closes, misses, failures = {}, 0, 0
    for step in range(TRADINGVIEW_MAX_MONTHS + 4):
        month = _add_months(first, step)
        try:
            bars = client.get_hist(symbol=tradingview_symbol(spec["root"], month), exchange=spec["exchange"], interval=interval, n_bars=int(n_bars))
        except Exception:
            bars, failures = None, failures + 1
            if failures >= 3 and not closes:
                raise RuntimeError("Can't reach TradingView. Check the connection and try again later.") from None
        if bars is None or len(bars) == 0 or "close" not in getattr(bars, "columns", []):
            if month > today:
                misses += 1
                if misses >= 3 and closes:
                    break   # past the last listed contract
            continue
        misses = 0
        days = [min(pd.Timestamp(stamp).date(), today) for stamp in bars.index]
        prices = pd.to_numeric(bars["close"], errors="coerce").to_numpy(float)
        kept = {d: p for d, p in zip(days, prices) if d < month and np.isfinite(p) and p > 0}
        if kept:
            closes[month] = pd.Series(kept)
    if not closes:
        raise RuntimeError("TradingView returned no TTF prices. Check the connection, or add a TradingView login to the keys file.")
    table = pd.DataFrame(closes).sort_index()
    table.index = pd.to_datetime(table.index)
    for month in table.columns:   # carry a quiet contract's last close, only while it was listed
        listed = table[month].first_valid_index()
        filled = table[month].ffill(limit=TRADINGVIEW_CARRY_DAYS)
        table[month] = filled.where(table.index >= listed)
    for day in table.index:
        table.loc[day, [m for m in table.columns if pd.Timestamp(m) <= day]] = np.nan
    live = table.notna().sum(axis=1)
    listed_then = pd.Series([sum(1 for m in table.columns if pd.Timestamp(m) > d) for d in table.index], index=table.index)
    table = table.loc[live >= 0.5 * listed_then.clip(lower=1)]
    rows = table.stack().dropna().rename("price").reset_index()   # pandas 3 keeps empty cells when stacking
    rows.columns = ["curve_date", "contract", "price"]
    return pd.DataFrame({"product": product, "contract": [pd.Timestamp(m).date() for m in rows["contract"]],
                         "curve_date": [pd.Timestamp(d).date() for d in rows["curve_date"]], "price": rows["price"].to_numpy(float)})


def fetch_temperature_normals(location_names: Sequence[str], years: int = 10, _json_get_fn=None) -> tuple[pd.DataFrame, list[str]]:
    """Calendar-day temperature and degree-day normals from Open-Meteo's historical archive."""
    _, normals, failures = fetch_weather_archive(location_names, years, _json_get_fn)
    return normals, failures

def _stored_live_item(source_key: str, label: str, data, metadata: Optional[dict] = None) -> None:
    item = {
        "label": label,
        "data": data,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": metadata or {},
    }
    st.session_state.setdefault("live_data", {})[source_key] = item
    bump("live")
    _save(lambda store: store.save_live_item(source_key, item))


def _resolve_live_key(key: str) -> str:
    """A plain dataset name resolves to its most recently fetched zone or variant."""
    items = st.session_state.get("live_data", {})
    if key in items:
        return key
    variants = [k for k in items if k.startswith(f"{key}|")]
    return max(variants, key=lambda k: str(items[k].get("fetched_at", ""))) if variants else key


def _live_variants(key: str) -> dict:
    """{zone or variant: stored item} for one dataset name."""
    items = st.session_state.get("live_data", {})
    return {k.split("|", 1)[1]: v for k, v in items.items() if k.startswith(f"{key}|")}


def _live_dataset_frame(key: str) -> pd.DataFrame:
    item = st.session_state.get("live_data", {}).get(_resolve_live_key(key), {})
    data = item.get("data") if isinstance(item, dict) else None
    # Copy-on-write is enabled, so this shallow copy is safe and avoids duplicating large frames.
    return data.copy(deep=False) if isinstance(data, pd.DataFrame) else pd.DataFrame()


def _live_item_fetched_at(key: str) -> str:
    item = st.session_state.get("live_data", {}).get(_resolve_live_key(key), {})
    return str(item.get("fetched_at", "")) if isinstance(item, dict) else ""


def _live_item_meta(key: str) -> dict:
    item = st.session_state.get("live_data", {}).get(_resolve_live_key(key), {})
    return item.get("metadata", {}) if isinstance(item, dict) and isinstance(item.get("metadata", {}), dict) else {}


def residual_ready_zones() -> list[str]:
    """Zones with price, load and generation all loaded, the only valid residual-load inputs."""
    sets = [set(_live_variants(k)) for k in ("entsoe_price", "entsoe_load", "entsoe_generation")]
    return sorted(set.intersection(*sets)) if all(sets) else []


def _build_residual_load_proxy(zone: Optional[str] = None) -> pd.DataFrame:
    """Join one zone's load, wind/solar output and day-ahead price into hourly context.

    Residual load = actual load minus actual wind and solar output, for overlapping
    timestamps. Inputs from different bidding zones are refused rather than combined.
    """
    suffix = f"|{zone}" if zone else ""
    keys = [f"entsoe_price{suffix}", f"entsoe_load{suffix}", f"entsoe_generation{suffix}"]
    zones = {_live_item_meta(k).get("zone") for k in keys} - {None}
    if len(zones) > 1:
        return pd.DataFrame()
    prices, load, generation = (_live_dataset_frame(k) for k in keys)
    if any(frame.empty for frame in (prices, load, generation)):
        return pd.DataFrame()

    def to_hourly(frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
        if "timestamp" not in frame.columns or "value" not in frame.columns:
            return pd.DataFrame(columns=["hour", column_name])
        out = frame[["timestamp", "value"]].copy()
        out["hour"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True).dt.floor("h")
        out[column_name] = _number_series(out["value"])
        out = out.dropna(subset=["hour", column_name])
        return out.groupby("hour", as_index=False)[column_name].mean()

    price_hourly = to_hourly(prices, "day_ahead_eur_mwh")
    load_hourly = to_hourly(load, "actual_load_mw")
    if price_hourly.empty or load_hourly.empty or "series" not in generation.columns:
        return pd.DataFrame()
    gen = generation.copy()
    labels = gen["series"].astype(str).str.casefold()
    gen = gen.loc[labels.str.contains("wind onshore|wind offshore|solar", regex=True) & ~labels.str.contains("consumption")].copy()
    if gen.empty:
        return pd.DataFrame()
    gen["hour"] = pd.to_datetime(gen["timestamp"], errors="coerce", utc=True).dt.floor("h")
    gen["generation_mw"] = _number_series(gen["value"])
    gen = gen.dropna(subset=["hour", "generation_mw"])
    # Sum onshore wind + offshore wind + solar at each source timestamp, then average
    # these sub-hourly/hourly totals into a common UTC-hour bucket.
    renewable_hourly = (gen.groupby(["hour", "timestamp"], as_index=False)["generation_mw"].sum()
                        .groupby("hour", as_index=False)["generation_mw"].mean()
                        .rename(columns={"generation_mw": "wind_solar_mw"}))
    if renewable_hourly.empty:
        return pd.DataFrame()
    joined = price_hourly.merge(load_hourly, on="hour", how="inner").merge(renewable_hourly, on="hour", how="inner")
    if joined.empty:
        return joined
    joined["residual_load_proxy_mw"] = joined["actual_load_mw"] - joined["wind_solar_mw"]
    return joined.sort_values("hour").reset_index(drop=True)


# =============================================================================
# SECTION 5 · DESIGN SYSTEM
# North Sea at dusk: deep petrol surfaces, a sodium-lamp accent for actions, and
# one fixed colour per market so gas, power and carbon read the same on every chart.
# =============================================================================
BG = "#0B202A"          # deepest layer of the page
PANEL = "#16353F"       # body of a raised surface
PANEL_SOFT = "#20485A"  # raised controls
PLATE_HI = "#1F4656"    # lit top-left edge of a plate
PLATE_LO = "#122B36"    # shaded bottom-right edge of a plate
BORDER = "#2F515C"      # hairlines
TEXT = "#F0F6F5"        # primary ink
GRAY = "#A9BEC2"        # secondary ink
SODIUM = "#F7BC45"      # actions
GAIN = "#5EE0A8"        # profit, long
LOSS = "#FF8F7E"        # loss, short

# Each part of the app has its own light, so you can tell where you are at a glance.
SECTION_COLORS = {"Market": "#4FB3F6", "Pricing": "#F7BC45", "Book and risk": "#34D6B4", "Data and controls": "#A893F5"}
_NAV_SECTION_CSS = "\n".join(f".st-key-nav_{i} {{{{ --sec:{c}; }}}}".replace("{{", "{").replace("}}", "}") for i, c in enumerate(SECTION_COLORS.values()))

# Chart series, validated for colour-blind separation on the page colour.
SERIES = ["#4C97E6", "#C2870C", "#17A891", "#E0703C", "#9180EE", "#6EA62E", "#DA6694"]
BLUE, AMBER, TEAL, ORANGE, PURPLE, GREEN, PINK = SERIES

ENTITY_COLORS = {
    "TTF Natural Gas": BLUE, "NBP Natural Gas": TEAL,
    "French Power Base": AMBER, "French Power Peak": ORANGE,
    "German Power Base": PURPLE, "German Power Peak": PINK,
    "EUA Carbon": GREEN, "API2 Coal": "#8FA3A8",
    "France": AMBER, "Germany-Luxembourg": PURPLE, "Netherlands": TEAL, "Belgium": ORANGE,
    "Spain": PINK, "Italy North": GREEN, "Austria": BLUE, "Great Britain": "#8FA3A8",
    "EUR": BLUE, "GBP": TEAL,
}

MAX_LINE_POINTS = 2000    # more than a chart's width can show
MAX_TABLE_ROWS = 2000
UI_FONT = "Archivo, 'Helvetica Neue', Arial, sans-serif"
NOTE_FONT = "Newsreader, Georgia, 'Times New Roman', serif"

# Streamlit re-runs this script on every interaction, so the template is built only once.
if "meridian" not in pio.templates:
    pio.templates["meridian"] = go.layout.Template(layout=dict(
    colorway=SERIES,
    font=dict(family=UI_FONT, size=13, color=TEXT),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    title=dict(font=dict(size=14, color=TEXT), x=0.015, xanchor="left", xref="container", y=0.975, yanchor="top"),
    xaxis=dict(showgrid=False, linecolor=BORDER, tickcolor=BORDER, ticks="outside", ticklen=4, automargin=True,
               zeroline=False, separatethousands=True, title=dict(font=dict(size=12, color=GRAY)), tickfont=dict(color=GRAY)),
    yaxis=dict(gridcolor="rgba(190,214,218,0.12)", zeroline=True, zerolinecolor=BORDER, linecolor="rgba(0,0,0,0)",
               automargin=True, separatethousands=True, title=dict(font=dict(size=12, color=GRAY)), tickfont=dict(color=GRAY)),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)",
                title=dict(text=""), font=dict(size=12, color=GRAY)),
    hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(family=UI_FONT, size=13, color=TEXT)),
    bargap=0.25, barcornerradius=4,
    ))
px.defaults.template = "meridian"
px.defaults.color_discrete_sequence = SERIES

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100,400;100,500;100,600;100,700&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
:root {{
  color-scheme: dark;
  --accent: {SODIUM};
  --plate: linear-gradient(158deg, {PLATE_HI} 0%, {PANEL} 52%, {PLATE_LO} 100%);
  --raise: inset 0 1px 0 rgba(255,255,255,0.09), inset 0 -1px 0 rgba(0,0,0,0.30), 0 12px 26px rgba(2,12,18,0.42), 0 2px 5px rgba(2,12,18,0.34);
  --raise-small: inset 0 1px 0 rgba(255,255,255,0.11), inset 0 -1px 0 rgba(0,0,0,0.28), 0 4px 10px rgba(2,12,18,0.40);
  --well: inset 0 2px 5px rgba(2,12,18,0.55), inset 0 -1px 0 rgba(255,255,255,0.05);
}}
html, body, .stApp, button, input, textarea, select, .stMarkdown, [data-testid="stWidgetLabel"] {{ font-family:{UI_FONT}; }}
.stApp {{ color:{TEXT}; font-size:15px; font-feature-settings:"tnum" 1;
  background:
    radial-gradient(1150px 620px at 8% -12%, rgba(52,214,180,0.17), transparent 62%),
    radial-gradient(1050px 560px at 96% -8%, rgba(247,188,69,0.11), transparent 60%),
    radial-gradient(1300px 800px at 60% 118%, rgba(79,179,246,0.09), transparent 62%),
    linear-gradient(180deg, #0F2B37 0%, {BG} 100%); }}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stDeployButton"], [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"] {{ display:none; }}
[data-testid="stMainBlockContainer"] {{ max-width:1400px; padding:1.5rem 2.5rem 3.5rem; }}
:focus-visible {{ outline:2px solid {SODIUM} !important; outline-offset:2px; }}

h1, h2, h3, h4 {{ color:{TEXT}; font-weight:600; letter-spacing:-0.005em; font-family:{UI_FONT}; }}
.stApp h2, .stApp h3 {{ font-size:1.14rem; font-weight:600; padding:1.25rem 0 0.45rem; line-height:1.3; }}
.stApp h2::before, .stApp h3::before {{ content:""; display:inline-block; width:4px; height:0.95em; margin-right:9px; border-radius:3px; vertical-align:-0.1em; background:var(--accent); box-shadow:0 0 10px var(--accent); }}
p, li {{ line-height:1.55; }}
a {{ color:{SODIUM}; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color:{GRAY}; font-size:0.86rem; line-height:1.5; max-width:92ch; }}
hr {{ border-color:{BORDER}; margin:1.2rem 0; }}

/* Page header: a lit band in the colour of this part of the app */
.st-key-page_head {{ position:relative; padding:18px 22px; margin-bottom:0.9rem; border-radius:16px; overflow:hidden;
  background:linear-gradient(100deg, color-mix(in srgb, var(--accent) 24%, {PANEL}) 0%, {PANEL} 46%, {PLATE_LO} 100%);
  box-shadow:var(--raise); }}
.st-key-page_head::after {{ content:""; position:absolute; left:0; right:0; bottom:0; height:2px; background:linear-gradient(90deg, var(--accent), transparent 70%); }}
.head-row {{ display:flex; gap:16px; align-items:center; }}
.head-icon {{ flex:none; width:54px; height:54px; border-radius:14px; display:flex; align-items:center; justify-content:center; color:#0D2029;
  background:linear-gradient(150deg, color-mix(in srgb, var(--accent) 80%, white) 0%, var(--accent) 62%, color-mix(in srgb, var(--accent) 72%, black) 100%);
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.45), inset 0 -2px 0 rgba(0,0,0,0.18), 0 8px 18px color-mix(in srgb, var(--accent) 34%, transparent); }}
.head-icon span {{ font-family:'Material Symbols Rounded'; font-size:30px; line-height:1; font-weight:400; }}
.stApp h1.page-title {{ font-family:{NOTE_FONT}; font-weight:500; font-size:2.15rem; line-height:1.12; color:{TEXT}; margin:0; padding:0; letter-spacing:-0.012em; }}
.page-sub {{ color:#C2D3D5; font-size:0.95rem; margin:0.25rem 0 0; max-width:78ch; }}
.pill {{ display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:999px; font-size:0.8rem; color:{TEXT}; white-space:nowrap; background:rgba(5,20,27,0.42); box-shadow:var(--well); }}
.pill i {{ width:8px; height:8px; border-radius:50%; display:inline-block; background:{GRAY}; }}
.pill.good i {{ background:{GAIN}; box-shadow:0 0 8px {GAIN}; }} .pill.warn i {{ background:{SODIUM}; box-shadow:0 0 8px {SODIUM}; }}

/* The desk note panel */
.st-key-hero {{ padding:24px 26px 20px; border-radius:18px; position:relative; overflow:hidden;
  background:
    radial-gradient(620px 300px at 100% 0%, rgba(79,179,246,0.20), transparent 70%),
    radial-gradient(520px 320px at 0% 100%, rgba(52,214,180,0.16), transparent 70%),
    var(--plate);
  box-shadow:var(--raise); }}
.st-key-hero [data-testid="stPlotlyChart"] {{ background:none; box-shadow:none; }}
.desk-note {{ font-family:{NOTE_FONT}; font-size:1.42rem; line-height:1.42; color:{TEXT}; max-width:34em; margin:0.2rem 0 1rem; }}
.desk-note p {{ margin:0 0 0.7rem; font-size:inherit; line-height:inherit; }}
.desk-note b {{ font-weight:600; font-family:{UI_FONT}; font-size:0.92em; white-space:nowrap; }}
.desk-note .soft {{ color:{GRAY}; font-size:0.86em; }}
.src-row {{ display:flex; flex-wrap:wrap; align-items:center; gap:0.4rem 0.45rem; margin:0.85rem 0 0.1rem; }}
.src-title {{ font-family:{UI_FONT}; font-size:0.78rem; font-weight:600; color:{GRAY}; margin-right:0.25rem; }}
.src {{ display:inline-flex; align-items:baseline; gap:0.35rem; padding:0.2rem 0.6rem; border-radius:999px; font-family:{UI_FONT}; font-size:0.78rem; color:{TEXT};
        border:1px solid rgba(169,190,194,0.25); background:rgba(255,255,255,0.03); }}
.src b {{ font-weight:700; font-size:0.72rem; }}
.src.real b, .src.user b {{ color:{GAIN}; }} .src.real {{ border-color:rgba(70,200,150,0.45); }}
.src.part b, .src.old b {{ color:{AMBER}; }} .src.part, .src.old {{ border-color:rgba(242,184,75,0.45); }}
.src.demo b {{ color:{AMBER}; }} .src.demo {{ border-color:rgba(242,184,75,0.65); background:rgba(242,184,75,0.08); }}
.src.missing b, .src.info b {{ color:{GRAY}; }}
.desk-note ul {{ margin:0 0 0.7rem; padding-left:1.1em; list-style:disc; }}
.desk-note li {{ margin:0 0 0.28rem; padding:0; font-size:1.12rem; line-height:1.38; color:{TEXT}; }}
.desk-note li::marker {{ color:{GRAY}; }}
.desk-note .desk-label {{ margin:0.9rem 0 0.3rem; font-family:{UI_FONT}; font-size:0.86rem; font-weight:600; color:{GRAY}; }}
.desk-note b.gain {{ color:{GAIN}; }} .desk-note b.loss {{ color:{LOSS}; }}

/* Figures: raised tiles lit along the top edge */
.fig, [data-testid="stMetric"] {{ position:relative; overflow:hidden; border-radius:13px; padding:13px 15px 12px; margin-bottom:0.65rem; min-height:92px;
  background:var(--plate); box-shadow:var(--raise); }}
.fig::before, [data-testid="stMetric"]::before {{ content:""; position:absolute; left:0; right:0; top:0; height:3px; background:linear-gradient(90deg, var(--tile, var(--accent)), transparent 85%); }}
.fig .l, [data-testid="stMetricLabel"] p {{ color:{GRAY}; font-size:0.84rem; line-height:1.3; font-weight:400; }}
.fig .v, [data-testid="stMetricValue"] {{ color:{TEXT}; font-size:1.5rem; font-weight:600; line-height:1.3; position:relative; }}
[data-testid="stMetricValue"] > div {{ font-size:1.5rem; }}
.fig .n {{ color:{GRAY}; font-size:0.8rem; margin-top:1px; line-height:1.35; position:relative; }}
.fig .v.gain {{ color:{GAIN}; }} .fig .v.loss {{ color:{LOSS}; }}
.fig svg.spark {{ position:absolute; right:0; bottom:0; width:48%; height:38px; opacity:0.92; pointer-events:none; }}
.fig .meter {{ height:7px; border-radius:6px; margin-top:8px; background:rgba(4,17,24,0.6); box-shadow:var(--well); overflow:hidden; }}
.fig .meter i {{ display:block; height:100%; border-radius:6px; background:linear-gradient(90deg, color-mix(in srgb, var(--tile) 62%, black), var(--tile)); box-shadow:0 0 10px var(--tile); }}

/* Charts and tables */
[data-testid="stPlotlyChart"] {{ background:var(--plate); border-radius:15px; box-shadow:var(--raise); overflow:hidden; }}
[data-testid="stDataFrame"] {{ border-radius:12px; box-shadow:0 8px 20px rgba(2,12,18,0.36); }}

/* Sidebar */
section[data-testid="stSidebar"] {{ width:256px !important; border-right:1px solid rgba(255,255,255,0.05);
  background:linear-gradient(180deg, #17394A 0%, #102B38 55%, #0C232E 100%); box-shadow:6px 0 22px rgba(2,12,18,0.40); }}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding:0 !important; background:none; }}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding:0 12px 24px !important; margin:0 !important; }}
[data-testid="stSidebarHeader"] {{ padding:12px 12px 2px; }}
.brand {{ display:flex; align-items:center; gap:11px; padding:0 4px; }}
.brand .mark {{ flex:none; width:40px; height:40px; border-radius:11px; display:flex; align-items:center; justify-content:center; color:#14262C;
  background:linear-gradient(150deg, #FFE08A 0%, {SODIUM} 58%, #C98A16 100%); box-shadow:inset 0 1px 0 rgba(255,255,255,0.55), inset 0 -2px 0 rgba(0,0,0,0.16), 0 6px 16px rgba(247,188,69,0.32); }}
.brand .mark span {{ font-family:'Material Symbols Rounded'; font-size:25px; line-height:1; }}
.brand .name {{ font-family:{NOTE_FONT}; font-size:1.42rem; font-weight:500; color:{TEXT}; line-height:1.05; }}
.brand small {{ display:block; font-family:{UI_FONT}; font-size:0.74rem; font-weight:400; color:{GRAY}; margin-top:3px; }}
section[data-testid="stSidebar"] [data-testid="stRadio"], section[data-testid="stSidebar"] [data-testid="stRadio"] > div, section[data-testid="stSidebar"] [role="radiogroup"] {{ width:100% !important; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] p {{ color:var(--sec); font-size:0.8rem; font-weight:600; padding:0 8px; margin-top:6px; }}
section[data-testid="stSidebar"] [role="radiogroup"] {{ gap:2px; flex-direction:column; }}
section[data-testid="stSidebar"] [role="radiogroup"] > div {{ width:100%; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {{ width:100% !important; box-sizing:border-box; margin:0; padding:5px 8px; border-radius:8px; cursor:pointer; display:block; transition:background 120ms ease; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{ display:none; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ font-size:0.93rem; color:{TEXT}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] [role="img"] {{ color:{GRAY}; margin-right:4px; font-size:1.05rem !important; vertical-align:-3px; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background:rgba(255,255,255,0.06); }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{
  background:linear-gradient(95deg, color-mix(in srgb, var(--sec) 30%, transparent), color-mix(in srgb, var(--sec) 9%, transparent));
  box-shadow:inset 3px 0 0 var(--sec), inset 0 1px 0 rgba(255,255,255,0.09), 0 4px 10px rgba(2,12,18,0.34); }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{ font-weight:600; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] [role="img"] {{ color:var(--sec); }}
.side-foot {{ color:{GRAY}; font-size:0.8rem; line-height:1.5; padding:0 8px; }}
.side-foot b {{ color:{TEXT}; font-weight:500; }}
section[data-testid="stSidebar"] label[data-baseweb="radio"] {{ width:100% !important; margin:0; padding:5px 8px; border-radius:8px; }}
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {{ display:none; }}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {{ background:rgba(255,255,255,0.06); }}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {{ background:linear-gradient(95deg, color-mix(in srgb, var(--sec) 30%, transparent), color-mix(in srgb, var(--sec) 9%, transparent)); box-shadow:inset 3px 0 0 var(--sec), 0 4px 10px rgba(2,12,18,0.34); }}
{_NAV_SECTION_CSS}

/* Views: a recessed tray with a raised active key */
[class*="st-key-tabs_view_"] [role="radiogroup"] {{ gap:3px; padding:4px; border-radius:11px; width:fit-content; max-width:100%; flex-wrap:wrap; background:rgba(4,17,24,0.5); box-shadow:var(--well); }}
[class*="st-key-tabs_view_"] [data-testid="stRadioOption"], [class*="st-key-tabs_view_"] label[data-baseweb="radio"] {{ margin:0; padding:7px 14px; border-radius:8px; cursor:pointer; transition:background 120ms ease; }}
[class*="st-key-tabs_view_"] [data-testid="stRadioOption"] > div > div:first-child, [class*="st-key-tabs_view_"] label[data-baseweb="radio"] > div:first-child {{ display:none; }}
[class*="st-key-tabs_view_"] [data-testid="stRadioOption"] p, [class*="st-key-tabs_view_"] label[data-baseweb="radio"] p {{ font-size:0.93rem; color:{GRAY}; }}
[class*="st-key-tabs_view_"] [data-testid="stRadioOption"]:hover, [class*="st-key-tabs_view_"] label[data-baseweb="radio"]:hover {{ background:rgba(255,255,255,0.05); }}
[class*="st-key-tabs_view_"] [data-testid="stRadioOption"][data-selected="true"], [class*="st-key-tabs_view_"] label[data-baseweb="radio"]:has(input:checked) {{ background:linear-gradient(170deg, {PLATE_HI}, {PANEL}); box-shadow:var(--raise-small), inset 0 -2px 0 var(--accent); }}
[class*="st-key-tabs_view_"] [data-testid="stRadioOption"][data-selected="true"] p, [class*="st-key-tabs_view_"] label[data-baseweb="radio"]:has(input:checked) p {{ color:{TEXT}; font-weight:600; }}

/* Controls: buttons stand proud and press in, fields sit in wells */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{ border-radius:9px; border:0; color:{TEXT}; font-weight:600; min-height:2.5rem;
  background:linear-gradient(170deg, {PANEL_SOFT}, {PANEL}); box-shadow:var(--raise-small); transition:transform 90ms ease, box-shadow 90ms ease, filter 120ms ease; }}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{ color:{TEXT}; filter:brightness(1.16); }}
.stButton > button:active, .stDownloadButton > button:active, [data-testid="stFormSubmitButton"] > button:active {{ transform:translateY(1px); box-shadow:var(--well); }}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind^="primary"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{ color:#14262C;
  background:linear-gradient(170deg, #FFD97A 0%, {SODIUM} 55%, #DD9C22 100%);
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.5), inset 0 -2px 0 rgba(0,0,0,0.14), 0 7px 16px rgba(247,188,69,0.30); }}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button[kind^="primary"]:hover, [data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {{ color:#14262C; filter:brightness(1.07); }}
.stButton > button[kind="primary"] p, [data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p {{ color:#14262C; }}
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"], [data-testid="stTextInputRootElement"], [data-testid="stDateInput"] [role="group"], [data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], .stTextArea textarea, [data-testid="stNumberInputContainer"] {{ background:rgba(5,20,27,0.46) !important; border-color:rgba(255,255,255,0.06) !important; border-radius:9px; box-shadow:var(--well); }}
[data-testid="stWidgetLabel"] p {{ color:{GRAY}; font-size:0.86rem; }}
[data-testid="stTextInputRootElement"] input, [data-testid="stSelectbox"] input, [data-testid="stNumberInputContainer"] input, [data-testid="stDateInput"] input {{ background:transparent !important; }}
[data-tag], [data-baseweb="tag"] {{ background:linear-gradient(170deg, {PANEL_SOFT}, {PANEL}) !important; box-shadow:var(--raise-small); }}
[data-tag] span, [data-tag] button, [data-baseweb="tag"] span, [data-baseweb="tag"] svg {{ color:{TEXT} !important; fill:{TEXT}; }}

/* Panels and notes */
.st-key-panel {{ padding:16px 18px 6px; border-radius:15px; background:var(--plate); box-shadow:var(--raise); margin-bottom:0.8rem; }}
[data-testid="stExpander"] details {{ border:0; border-radius:12px; background:var(--plate); box-shadow:var(--raise); }}
[data-testid="stExpander"] summary {{ padding:0.6rem 0.9rem; }}
[data-testid="stExpander"] summary:hover {{ color:var(--accent); }}
[data-testid="stAlertContainer"] {{ border-radius:11px; border:0; color:{TEXT} !important; background:{PANEL} !important; box-shadow:var(--raise-small); }}
[data-testid="stAlertContainer"] p {{ color:{TEXT}; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ background:linear-gradient(95deg, rgba(247,188,69,0.24), rgba(247,188,69,0.07)) !important; box-shadow:inset 3px 0 0 {SODIUM}, var(--raise-small); }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{ background:linear-gradient(95deg, rgba(79,179,246,0.24), rgba(79,179,246,0.07)) !important; box-shadow:inset 3px 0 0 #4FB3F6, var(--raise-small); }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{ background:linear-gradient(95deg, rgba(255,143,126,0.26), rgba(255,143,126,0.07)) !important; box-shadow:inset 3px 0 0 {LOSS}, var(--raise-small); }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{ background:linear-gradient(95deg, rgba(94,224,168,0.22), rgba(94,224,168,0.06)) !important; box-shadow:inset 3px 0 0 {GAIN}, var(--raise-small); }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:rgba(255,255,255,0.07); }}

/* Quote ticket */
.st-key-ticket {{ position:relative; overflow:hidden; padding:20px 22px; border-radius:16px; box-shadow:var(--raise);
  background:radial-gradient(420px 200px at 100% 0%, rgba(247,188,69,0.22), transparent 70%), var(--plate); }}
.st-key-ticket::before {{ content:""; position:absolute; left:0; right:0; top:0; height:3px; background:linear-gradient(90deg, {SODIUM}, transparent 90%); }}
.ticket-price {{ font-family:{NOTE_FONT}; font-size:2.9rem; line-height:1.1; font-weight:500; margin:0.15rem 0 0.1rem; color:#FFD97A; text-shadow:0 0 26px rgba(247,188,69,0.36); }}
.ticket-unit {{ color:{GRAY}; font-size:0.9rem; margin-bottom:0.9rem; }}
.ticket-row {{ display:flex; justify-content:space-between; gap:18px; padding:6px 0; border-top:1px solid rgba(255,255,255,0.08); font-size:0.94rem; }}
.ticket-row span:first-child {{ color:{GRAY}; }}

/* Market Read */
.dial {{ position:relative; overflow:hidden; border-radius:16px; padding:17px 19px 15px; margin-bottom:0.8rem; min-height:206px; box-shadow:var(--raise);
  background:radial-gradient(420px 170px at 100% 0%, color-mix(in srgb, var(--accent) 20%, transparent), transparent 72%), var(--plate); }}
.dial-top {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
.dial-name {{ color:{GRAY}; font-size:0.92rem; }}
.dial-verdict {{ font-family:{NOTE_FONT}; font-size:1.95rem; font-weight:500; line-height:1.15; margin:0.5rem 0 0.75rem; color:{TEXT}; }}
.dial-verdict.gain {{ color:{GAIN}; }} .dial-verdict.loss {{ color:{LOSS}; }}
.scale {{ position:relative; height:12px; margin:0 8px; }}
.scale .track {{ position:absolute; left:-8px; right:-8px; top:2px; height:8px; border-radius:6px; box-shadow:var(--well); background:linear-gradient(90deg, {LOSS} 0%, #4A5F66 50%, {GAIN} 100%); }}
.scale .knob {{ position:absolute; top:-4px; width:20px; height:20px; margin-left:-10px; border-radius:50%; background:linear-gradient(160deg, #FFFFFF, #B9C9CC); box-shadow:inset 0 -2px 2px rgba(0,0,0,0.22), 0 3px 8px rgba(2,12,18,0.6); }}
.scale-ends {{ display:flex; justify-content:space-between; color:{GRAY}; font-size:0.76rem; margin-top:7px; }}
.dial-line {{ color:#C2D3D5; font-size:0.86rem; line-height:1.4; margin-top:0.55rem; }}
.ev {{ display:flex; gap:22px; justify-content:space-between; align-items:center; padding:13px 17px; margin-bottom:9px; border-radius:12px; background:var(--plate); box-shadow:var(--raise-small); }}
.ev-main {{ min-width:0; flex:1; }}
.ev-name {{ font-weight:600; font-size:0.96rem; }}
.ev-value {{ color:{TEXT}; font-size:0.93rem; margin-top:2px; line-height:1.4; }}
.ev-why {{ color:{GRAY}; font-size:0.84rem; margin-top:2px; line-height:1.4; }}
.ev-side {{ flex:none; width:170px; }}
.ev-tag {{ text-align:right; font-size:0.84rem; font-weight:600; margin-bottom:6px; color:{GRAY}; }}
.ev-tag.gain {{ color:{GAIN}; }} .ev-tag.loss {{ color:{LOSS}; }}
.ev-bar {{ position:relative; height:8px; border-radius:5px; background:rgba(4,17,24,0.6); box-shadow:var(--well); }}
.ev-bar::after {{ content:""; position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:{GRAY}; }}
.ev-bar i {{ position:absolute; top:0; bottom:0; border-radius:5px; background:{GRAY}; }}
.ev-bar i.gain {{ background:{GAIN}; }} .ev-bar i.loss {{ background:{LOSS}; }}
.flip {{ padding:9px 0; border-top:1px solid rgba(255,255,255,0.08); font-size:0.92rem; line-height:1.45; }}
.flip b {{ font-weight:600; }}

@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; }} }}
@media (max-width: 720px) {{
  [data-testid="stMainBlockContainer"] {{ padding:3rem 1rem 2.5rem; }}
  .stApp h1.page-title {{ font-size:1.65rem; }}
  .head-icon {{ width:44px; height:44px; }}
  .desk-note {{ font-size:1.2rem; }}
  .st-key-hero, .st-key-page_head {{ padding:15px; }}
  .ev {{ flex-direction:column; align-items:stretch; gap:10px; }}
  .ev-side {{ width:100%; }}
}}
</style>
"""


def _styled(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    """Apply the house chart style. Entity colours are fixed, so a filter never repaints a series."""
    resolved_height = height or fig.layout.height or 360
    has_title = bool(fig.layout.title and fig.layout.title.text)
    # Set here as well as in the template: Streamlit fills any background the figure itself leaves blank.
    fig.update_layout(template="meridian", height=resolved_height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=18, r=20, t=70 if has_title else 34, b=14),
                      hovermode=fig.layout.hovermode or ("x unified" if any(t.type == "scatter" and "lines" in str(t.mode) for t in fig.data) else "closest"))
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title=dict(x=0.015, xanchor="left", xref="container", y=0.975, yanchor="top", font=dict(size=14, color=TEXT, weight=600)))
    fig.update_xaxes(title_font=dict(size=12, color=GRAY))
    fig.update_yaxes(title_font=dict(size=12, color=GRAY))
    for trace in fig.data:
        name = str(getattr(trace, "name", "") or "")
        color = ENTITY_COLORS.get(name)
        if trace.type == "scatter":
            if "lines" in str(trace.mode):
                _thin_line(trace)
            points = len(trace.x) if trace.x is not None else 0
            if "lines" in str(trace.mode):
                if points > 48 and "markers" in str(trace.mode):
                    trace.mode = "lines"
                if trace.line.width is None:
                    trace.line.width = 2
            if "markers" in str(trace.mode):
                trace.marker.size = trace.marker.size or 8
                trace.marker.line = dict(width=2, color=PANEL)
            if color:
                if "lines" in str(trace.mode):
                    trace.line.color = color
                trace.marker.color = color
        elif trace.type == "bar":
            if color:
                trace.marker.color = color
            trace.marker.line = dict(width=0)
    _add_line_glow(fig)
    return fig


def _thin_line(trace) -> None:
    """Reduce a very long line to each bucket's low and high, which keeps its spikes on screen."""
    if trace.x is None or trace.y is None or trace.customdata is not None or len(trace.x) <= MAX_LINE_POINTS:
        return
    y = np.asarray(trace.y, dtype=float)
    x = np.asarray(trace.x)
    filled = np.where(np.isnan(y), np.nanmean(y), y)
    edges = np.linspace(0, len(y), MAX_LINE_POINTS // 2 + 1).astype(int)
    keep = set()
    for start, end in zip(edges[:-1], edges[1:]):
        window = filled[start:end]
        keep.update((start + int(window.argmin()), start + int(window.argmax())))
    index = np.array(sorted(keep))
    trace.x, trace.y = x[index], y[index]


def _add_line_glow(fig: go.Figure) -> None:
    """Lay a soft, wide stroke under each line so it lifts off the plate. Purely visual."""
    glows = []
    for index, trace in enumerate(fig.data):
        if trace.type != "scatter" or "lines" not in str(trace.mode) or trace.fill not in (None, "none"):
            continue
        if trace.line.dash not in (None, "solid") or trace.line.width == 0 or trace.x is None or len(trace.x) > 800:
            continue
        color = trace.line.color or SERIES[index % len(SERIES)]
        trace.line.color = color
        glows.append(go.Scatter(x=trace.x, y=trace.y, mode="lines", line=dict(color=color, width=7), opacity=0.17,
                                hoverinfo="skip", showlegend=False, xaxis=trace.xaxis, yaxis=trace.yaxis))
    if glows:
        original = len(fig.data)
        fig.add_traces(glows)
        fig.data = tuple(fig.data[original:]) + tuple(fig.data[:original])


def chart(fig: go.Figure, height: Optional[int] = None) -> None:
    # Early 1.5x releases size charts with use_container_width instead of width.
    sizing = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters else {"use_container_width": True}
    # theme=None keeps the house template; Streamlit's theme would paint an opaque plot background.
    st.plotly_chart(_styled(fig, height), theme=None, config={"displayModeBar": False, "responsive": True}, **sizing)


def _pretty_column(name: object) -> str:
    text = str(name)
    if any(ch.isupper() for ch in text[1:]) or " " in text:
        return text
    text = text.replace("_", " ").strip()
    for old, new in (("mwh", "MWh"), ("mw", "MW"), ("mtm", "MtM"), ("eur", "EUR"), ("gbp", "GBP"), ("id", "ID"), ("pnl", "P&L")):
        text = " ".join(new if word == old else word for word in text.split(" "))
    return text[:1].upper() + text[1:]


def table(frame: pd.DataFrame, height: Optional[int] = None, **_ignored) -> None:
    """House table: readable headers, thousands separators, internal columns hidden."""
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    if len(frame) > MAX_TABLE_ROWS:
        st.caption(f"Showing the latest {MAX_TABLE_ROWS:,} of {len(frame):,} rows. Download the data for all of them.")
        frame = frame.tail(MAX_TABLE_ROWS)
    visible = [c for c in frame.columns if not str(c).startswith("_")]
    config = {}
    for column in visible:
        label = _pretty_column(column)
        if pd.api.types.is_float_dtype(frame[column]):
            config[column] = st.column_config.NumberColumn(label, format="localized")
        else:
            config[column] = st.column_config.Column(label)
    display = frame[visible].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].round(2 if display[column].abs().max() >= 100 else 3)
        elif display[column].dtype == object:
            display[column] = display[column].map(lambda v: v.isoformat() if isinstance(v, (date, datetime)) else v)
    arguments = {"width": "stretch", "hide_index": True, "column_config": config}
    if height:
        arguments["height"] = height
    st.dataframe(display, **arguments)


def fmt_money(value: float, currency: str = "EUR", signed: bool = True, decimals: int = 0) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    symbol = {"EUR": "€", "GBP": "£", "USD": "$"}.get(currency, f"{currency} ")
    amount = float(value)
    sign = ("+" if amount > 0 else "−" if amount < 0 else "") if signed else ("−" if amount < 0 else "")
    return f"{sign}{symbol}{abs(amount):,.{decimals}f}"


def tone_of(value: float) -> str:
    return "gain" if value > 0 else "loss" if value < 0 else ""


def _sparkline(values: Sequence[float], color: str) -> str:
    """A small inline trend drawn from real values. Returns nothing if there are too few."""
    points = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(points) < 3:
        return ""
    if len(points) > 90:
        step = len(points) / 90
        points = [points[int(i * step)] for i in range(90)]
    low, high = min(points), max(points)
    span = (high - low) or 1.0
    coords = [(i / (len(points) - 1) * 120, 30 - (p - low) / span * 26) for i, p in enumerate(points)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    gradient = "g" + hashlib.sha1(f"{line}{color}".encode()).hexdigest()[:10]
    return (f'<svg class="spark" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden="true">'
            f'<defs><linearGradient id="{gradient}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity="0.34"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="0,34 {line} 120,34" fill="url(#{gradient})"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>')


def kpi(col, label: str, value: str, sub: str = "", accent: str = "", tone: str = "",
        spark: Optional[Sequence[float]] = None, meter: Optional[float] = None) -> None:
    """A raised figure tile: label, number, optional note, trend line or fill meter."""
    color = accent or {"gain": GAIN, "loss": LOSS}.get(tone, "")
    tile = f' style="--tile:{color}"' if color else ""
    note = f'<div class="n">{html.escape(sub)}</div>' if sub else ""
    trend = _sparkline(spark, color or "#4FB3F6") if spark is not None else ""
    bar = f'<div class="meter"><i style="width:{max(0.0, min(100.0, float(meter))):.1f}%"></i></div>' if meter is not None else ""
    col.markdown(f'<div class="fig"{tile}>{trend}<div class="l">{html.escape(label)}</div>'
                 f'<div class="v {tone}">{html.escape(value)}</div>{note}{bar}</div>', unsafe_allow_html=True)


def render_header(title: str, subtitle: str) -> None:
    label = st.session_state.get("nav_page", DEFAULT_NAV_ITEM)
    icon = label.split("/", 1)[1].split(":", 1)[0] if label.startswith(":material/") else "bolt"
    with st.container(key="page_head"):
        left, right = st.columns([5, 2], vertical_alignment="center")
        left.markdown(f'<div class="head-row"><div class="head-icon"><span aria-hidden="true">{html.escape(icon)}</span></div>'
                      f'<div><h1 class="page-title">{html.escape(title)}</h1><p class="page-sub">{html.escape(subtitle)}</p></div></div>', unsafe_allow_html=True)
        sources = page_sources_html(current_page_name())
        if sources:
            st.markdown(sources, unsafe_allow_html=True)


def page_accent_css(label: str) -> str:
    """Set the accent colour for the section the current page belongs to."""
    for section, items in NAV_SECTIONS.items():
        if label in items:
            return f"<style>:root {{ --accent:{SECTION_COLORS[section]}; }}</style>"
    return ""


def title_block(title: str, subtitle: str) -> None:
    render_header(title, subtitle)


def render_next_step(current_page: str) -> None:
    """A quiet pointer at the foot of a page to the next step in the desk's workflow."""
    next_page = WORKFLOW_NEXT.get(current_page)
    if not next_page:
        return
    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
    if st.button(f"Next: {next_page}", key=f"workflow_next_{current_page}"):
        st.session_state["pending_nav_page"] = next_page
        st.rerun()


def view_tabs(labels: Sequence[str]) -> list[bool]:
    """A switcher that only builds the chosen view, so hidden charts and tables cost nothing."""
    key = "view_" + hashlib.sha1("|".join(labels).encode("utf-8")).hexdigest()[:8]
    with st.container(key=f"tabs_{key}"):
        chosen = st.radio("View", list(labels), horizontal=True, label_visibility="collapsed", key=key)
    return [label == chosen for label in labels]


def need_data_note(message: str) -> None:
    st.info(message, icon=":material/info:")


# =============================================================================
# SECTION 6 · NAVIGATION AND SESSION
# =============================================================================
NAV_SECTIONS = {
    "Market": [":material/dashboard:  Market Overview", ":material/wb_twilight:  Morning Brief", ":material/psychology:  Market Read", ":material/public:  Market Fundamentals",
               ":material/bolt:  Power Spot", ":material/compare_arrows:  Spreads & Margins"],
    "Pricing": [":material/show_chart:  Forward Curves", ":material/request_quote:  OTC Quote Builder"],
    "Book and risk": [":material/menu_book:  Trading Book & MtM", ":material/calendar_today:  Daily P&L",
                      ":material/shield:  Risk & Exposure", ":material/verified_user:  Hedge Advisor", ":material/local_shipping:  Physical Operations"],
    "Data and controls": [":material/cable:  Live Data Integrations", ":material/tune:  Data & Assumptions",
                          ":material/rule:  Model Validation"],
}
ALL_NAV_ITEMS = [label for labels in NAV_SECTIONS.values() for label in labels]
DEFAULT_NAV_ITEM = ALL_NAV_ITEMS[0]
NAV_TO_PAGE = {label: label.split("  ", 1)[1] for label in ALL_NAV_ITEMS}

WORKFLOW_NEXT = {
    "Market Overview": "Morning Brief",
    "Morning Brief": "Market Read",
    "Market Read": "Market Fundamentals",
    "Market Fundamentals": "Power Spot",
    "Power Spot": "Spreads & Margins",
    "Spreads & Margins": "Forward Curves",
    "Forward Curves": "OTC Quote Builder",
    "OTC Quote Builder": "Trading Book & MtM",
    "Trading Book & MtM": "Daily P&L",
    "Daily P&L": "Risk & Exposure",
    "Risk & Exposure": "Hedge Advisor",
    "Hedge Advisor": "Physical Operations",
}

valuation_date = date.today()
curves: dict[str, pd.DataFrame] = {}


def set_nav_page(page_name: str) -> None:
    """Navigate to a route and synchronise the grouped sidebar radios."""
    target_label = next((label for label, name in NAV_TO_PAGE.items() if name == page_name), None)
    if target_label is None:
        raise ValueError(f"Unknown application page: {page_name}")
    st.session_state.nav_page = target_label
    for section, items in NAV_SECTIONS.items():
        st.session_state[f"navsec_{section}"] = target_label if target_label in items else None


def _on_nav(section: str) -> None:
    selected = st.session_state.get(f"navsec_{section}")
    if selected is None:
        return
    st.session_state.nav_page = selected
    for other in NAV_SECTIONS:
        if other != section:
            st.session_state[f"navsec_{other}"] = None


def render_sidebar() -> str:
    if st.session_state.get("nav_page") not in ALL_NAV_ITEMS:
        st.session_state.nav_page = DEFAULT_NAV_ITEM
    current = st.session_state.nav_page
    with st.sidebar:
        st.markdown('<div class="brand"><div class="mark"><span aria-hidden="true">bolt</span></div>'
                    '<div class="name">Meridian<small>European gas and power desk</small></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
        for index, (section, items) in enumerate(NAV_SECTIONS.items()):
            key = f"navsec_{section}"
            if key not in st.session_state:
                st.session_state[key] = current if current in items else None
            with st.container(key=f"nav_{index}"):
                st.radio(section, items, key=key, on_change=_on_nav, args=(section,))
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
        render_update_panel()
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        loaded = sum(1 for value in st.session_state.get("live_data", {}).values()
                     if isinstance(value, dict) and isinstance(value.get("data"), pd.DataFrame) and not value["data"].empty)
        saved = ("Your work saves automatically." if st.session_state.get("desk_store") is not None and not st.session_state.get("desk_store_error")
                 else "Saving is unavailable, so changes last until you close this tab.")
        st.markdown(f'<div class="side-foot">{loaded} datasets held. {saved}<br>For analysis only, not for trade execution.</div>', unsafe_allow_html=True)
    return NAV_TO_PAGE[st.session_state.nav_page]


def init_session() -> None:
    """Load the saved desk once per browser session, creating the demo book the first time."""
    if st.session_state.get("session_ready"):
        return
    store = desk_store()
    saved_trades = saved_ops = None
    custom, metadata, live, settings = {}, {}, {}, dict(DEFAULT_SETTINGS)
    if store is not None:
        try:
            saved_trades = store.load_table("trades")
            saved_ops = store.load_table("physical_ops")
            custom, metadata = store.load_custom_curves()
            live = store.load_live_items()
            settings = {key: store.get_setting(key, value) for key, value in DEFAULT_SETTINGS.items()}
        except Exception as exc:
            st.session_state.desk_store_error = f"{type(exc).__name__}: {exc}"
    st.session_state.custom_curves = custom
    st.session_state.curve_metadata = metadata
    st.session_state.live_data = live
    st.session_state.settings = settings
    if float(settings.get("fx_ecb_epoch", 0.0)) > 0 and "fx_updated_at" not in st.session_state:
        st.session_state.fx_updated_at = datetime.fromtimestamp(float(settings["fx_ecb_epoch"]), timezone.utc)
    if saved_trades is None:
        commit_trades(build_sample_trades())
    else:
        st.session_state.trades = saved_trades
        st.session_state.physical_ops = sync_physical_ops_with_trades(saved_ops, saved_trades)
    st.session_state.session_ready = True


def build_curves() -> dict[str, pd.DataFrame]:
    """Imported marks, then free real prices, then demo prices for whatever is left. See composed_curves."""
    return composed_curves()[0]

def ensure_daily_snapshot() -> None:
    """Record today's marks the first time the desk opens each day. History builds by itself."""
    if st.session_state.get("snapshot_day") == valuation_date:
        return
    custom, metadata = st.session_state.get("custom_curves", {}), st.session_state.get("curve_metadata", {})
    for product, curve in curves.items():
        # Old imported marks aren't today's price. Saving them again would show a false day of no movement.
        marked = curve_mark_date(metadata.get(product)) if product in custom else None
        if marked is not None and marked < valuation_date:
            continue
        source = snapshot_source(product)
        _save(lambda s, c=curve, src=source: s.save_snapshot(valuation_date, c, src, replace=False))
    st.session_state.snapshot_day = valuation_date
    bump("snapshots")


def saved_snapshots() -> pd.DataFrame:
    def load() -> pd.DataFrame:
        store = desk_store()
        if store is None:
            return pd.DataFrame(columns=["snapshot_date", "product", "delivery_month", "price", "source"])
        return store.load_snapshots()
    return session_memo("snapshots", data_version("snapshots"), load)


def saved_lookups() -> dict:
    return session_memo("snapshot_lookup", data_version("snapshots"), lambda: snapshot_lookup(saved_snapshots()))


def saved_moves() -> tuple:
    return session_memo("contract_moves", data_version("snapshots"), lambda: _contract_moves(saved_snapshots()))


def executed_buckets() -> pd.DataFrame:
    return session_memo("buckets", data_version("trades"), lambda: book_buckets(st.session_state.trades))


def zone_prices(zone: str) -> pd.DataFrame:
    """A zone's day-ahead prices, cleaned once until new data arrives."""
    timezone_name = ENTSOE_ZONES[zone]["timezone"]
    return session_memo(f"spot|{zone}", data_version("live"), lambda: spot_series(_live_dataset_frame(f"entsoe_price|{zone}"), timezone_name))


def live_snapshot() -> dict:
    return session_memo("live_snapshot", data_version("live"), build_live_snapshot)


def reference_mark(product: str, start: date, end: date, profile: str) -> float:
    return swap_fixed_price(curves[product], start, end, profile)


def month_start(offset: int = 1) -> date:
    return _add_months(valuation_date, offset)


def month_end_exclusive(start: date, months: int) -> date:
    """Delivery end for a tenor counted in months: 15 Dec plus 3 months ends on 15 Mar."""
    return (pd.Timestamp(start) + pd.DateOffset(months=int(months))).date()


def get_marked_trades() -> pd.DataFrame:
    """Mark the book, reusing the result while trades and curves are unchanged."""
    key = (data_version("trades", "curves"), valuation_date)
    return session_memo("marked", key, lambda: mark_trades_resilient(st.session_state.trades.copy(), curves)).copy()


def curve_source_label(product: str) -> str:
    if product not in st.session_state.get("custom_curves", {}):
        return "Demo marks"
    metadata = st.session_state.get("curve_metadata", {}).get(product, {})
    return str(metadata.get("source_label", "Imported marks")).strip() or "Imported marks"


def keys_file_path() -> Path:
    configured = os.getenv("MERIDIAN_KEYS_FILE", "").strip()
    return Path(configured) if configured else Path(__file__).resolve().with_name(KEYS_FILE_NAME)


def keys_from_file(path: Optional[Path] = None) -> dict[str, str]:
    """Read NAME = "value" lines from the keys file beside the app. The values are never logged."""
    try:
        text = Path(path or keys_file_path()).read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    found = {}
    for line in text.splitlines():
        match = re.match(r"""\s*([A-Za-z0-9_]+)\s*=\s*["']([^"']*)["']""", line)
        if match and match[2].strip():
            found[match[1]] = match[2].strip()
    return found


def resolved_credentials() -> dict[str, str]:
    """Credentials in priority order: entered this session, Streamlit secrets, environment, keys file."""
    overrides = st.session_state.get("credential_overrides", {})
    from_file = keys_from_file()
    def pick(name: str, secret: str) -> str:
        return overrides.get(name) or _secret_or_env(secret, secret) or from_file.get(secret, "")
    return {"gie": pick("gie", "GIE_API_KEY"), "entsoe": pick("entsoe", "ENTSOE_API_TOKEN"), "open_meteo": pick("open_meteo", "OPEN_METEO_API_KEY"),
            "tradingview_user": pick("tradingview_user", "TRADINGVIEW_USERNAME"), "tradingview_password": pick("tradingview_password", "TRADINGVIEW_PASSWORD")}


def missing_keys_text() -> str:
    keys = resolved_credentials()
    absent = [label for label, value in (("ENTSO-E", keys["entsoe"]), ("GIE", keys["gie"])) if not value]
    return f"No {' or '.join(absent)} key found. Put {KEYS_FILE_NAME} in the same folder as this app." if absent else ""


# =============================================================================
# SECTION 6B · AUTOMATIC MARKET DATA UPDATES
# Saved keys, background fetching, and requests only for what isn't held yet.
# =============================================================================
UPDATE_DEFAULT_ZONES = ("France", "Germany-Luxembourg")
PRICE_HISTORY_DAYS = 365
# ENTSO-E's forecasts only need to span the tested days and enough earlier days to fit them.
FORECAST_HISTORY_DAYS = 300
ZONE_WEATHER_HISTORY_DAYS = 430
STALE_HOURS = {"tradingview": 1.0, "prices": 6.0, "fundamentals": 6.0, "weather": 3.0, "fx": 12.0, "normals": 24.0 * 30.0, "power_weather": 72.0}
KEEP_DAYS = {"prices": 400, "load": 90, "generation": 90, "lng": 400}
MAX_UPDATE_WORKERS = 4


@dataclass
class UpdateTask:
    """One source to refresh. fetch runs in a worker thread, so it must not touch the session."""
    name: str
    fetch: Callable[[], object]
    apply: Callable[[object], str]
    start: Optional[date] = None


def merge_observations(old: Optional[pd.DataFrame], new: pd.DataFrame, keys: Sequence[str],
                       time_col: Optional[str] = None, keep_days: Optional[int] = None) -> pd.DataFrame:
    """Add newly fetched rows to held rows. New rows win, so provider revisions are kept."""
    if old is None or old.empty:
        merged = new.copy()
    else:
        merged = pd.concat([old, new], ignore_index=True)
    usable = [k for k in keys if k in merged.columns]
    if usable:
        probe = merged[usable[0]]
        marker = pd.to_datetime(probe, errors="coerce", utc=True).astype("int64") if time_col == usable[0] else probe.astype(str)
        extra = [merged[k].astype(str) for k in usable[1:]]
        merged = merged.loc[~pd.concat([marker.rename("_a")] + [e.rename(f"_b{i}") for i, e in enumerate(extra)], axis=1).duplicated(keep="last")]
    if time_col and time_col in merged.columns:
        times = pd.to_datetime(merged[time_col], errors="coerce", utc=True)
        if keep_days:
            merged, times = merged.loc[times >= times.max() - pd.Timedelta(days=int(keep_days))], times.loc[times >= times.max() - pd.Timedelta(days=int(keep_days))]
        merged = merged.iloc[np.argsort(times.to_numpy(), kind="stable")]
    return merged.reset_index(drop=True)


def _held_frame(key: str) -> Optional[pd.DataFrame]:
    item = st.session_state.get("live_data", {}).get(key)
    data = item.get("data") if isinstance(item, dict) else None
    return data if isinstance(data, pd.DataFrame) and not data.empty else None


def store_observations(key: str, label: str, new: pd.DataFrame, metadata: dict, keys: Sequence[str],
                       time_col: str, keep_days: Optional[int] = None) -> pd.DataFrame:
    merged = merge_observations(_held_frame(key), new, keys, time_col, keep_days)
    _stored_live_item(key, label, merged, metadata)
    return merged


def store_entsoe(key: str, label: str, new: pd.DataFrame, metadata: dict, keep_days: Optional[int] = None) -> pd.DataFrame:
    return store_observations(key, label, new, metadata, ("timestamp", "series"), "timestamp", keep_days)


def incremental_start(held: Optional[pd.DataFrame], time_col: str, first_days: int, overlap_days: int, today: date) -> date:
    """Start of the next request: a full first window, then only the days not held yet."""
    if held is None or time_col not in held.columns:
        return today - timedelta(days=int(first_days))
    last = pd.to_datetime(held[time_col], errors="coerce", utc=True).max()
    if pd.isna(last):
        return today - timedelta(days=int(first_days))
    return max(last.date() - timedelta(days=int(overlap_days)), today - timedelta(days=ENTSOE_MAX_DAYS))


def _hours_since(key: str) -> Optional[float]:
    fetched = st.session_state.get("live_data", {}).get(key, {}).get("fetched_at", "")
    if not fetched:
        return None
    try:
        return (datetime.now(timezone.utc) - pd.Timestamp(fetched).to_pydatetime()).total_seconds() / 3600.0
    except Exception:
        return None


def _needs_update(key: str, kind: str, force: bool) -> bool:
    age = _hours_since(key)
    return force or age is None or age >= STALE_HOURS[kind]


def tradingview_tasks(force: bool = True, today: Optional[date] = None) -> list[UpdateTask]:
    """The TTF curve from TradingView, the default source for TTF marks. Refreshed hourly as prices are delayed anyway."""
    today = today or date.today()
    if not tradingview_available():
        def missing() -> None:
            raise RuntimeError(f"TTF is on demo prices because tradingview-datafeed isn't installed. Run: {TRADINGVIEW_INSTALL}, then restart the app.")
        return [UpdateTask("TTF curve from TradingView", missing, lambda _: "")]
    if not _needs_update("tradingview_curves", "tradingview", force):
        return []
    keys = resolved_credentials()
    bars = TRADINGVIEW_HISTORY_BARS if _held_frame("tradingview_curves") is None else TRADINGVIEW_UPDATE_BARS
    login = (keys["tradingview_user"], keys["tradingview_password"])
    tasks = []
    for tv_product in TRADINGVIEW_CURVES:
        def fetch_tv(product=tv_product) -> pd.DataFrame:
            return fetch_tradingview_curve(product, today, bars, *login)
        def apply_tv(raw: pd.DataFrame, product=tv_product) -> str:
            return apply_tradingview_curve(product, raw)
        tasks.append(UpdateTask(f"{SHORT_NAMES.get(tv_product, tv_product)} curve from TradingView", fetch_tv, apply_tv))
    return tasks


def plan_update(force: bool = True, today: Optional[date] = None) -> list[UpdateTask]:
    """Decide what to request. force ignores recent fetches, except for slow-moving normals."""
    today = today or date.today()
    keys = resolved_credentials()
    tasks: list[UpdateTask] = tradingview_tasks(force, today)   # first, so the TTF marks come in first

    def keyed(kind: str, fetch: Callable[[], object]) -> Callable[[], object]:
        """A source with no key reports what to do, rather than a provider's error."""
        if keys[kind]:
            return fetch
        label = "ENTSO-E" if kind == "entsoe" else "GIE"
        def no_key() -> None:
            raise RuntimeError(f"No {label} key found. Put {KEYS_FILE_NAME} in the same folder as this app.")
        return no_key

    def entsoe_task(kind: str, store_key: str, label: str, series_kind: str, zone: str, keep: int, first_days: int, end: date,
                    source: Optional[str] = None, target: Optional[str] = None, variant: Optional[str] = None) -> None:
        name = variant or zone
        full_key = f"{store_key}|{name}"
        if not _needs_update(full_key, kind, force):
            return
        held = _held_frame(full_key)
        start = incremental_start(held, "timestamp", first_days, 2, today)
        # Prices held for less than a year are extended once, so past reads can be scored straight away.
        if store_key == "entsoe_price" and held is not None:
            earliest = pd.to_datetime(held["timestamp"], errors="coerce", utc=True).min()
            if pd.notna(earliest) and earliest.date() > today - timedelta(days=PRICE_HISTORY_DAYS - 35):
                start = today - timedelta(days=PRICE_HISTORY_DAYS)
        metadata = {"zone": zone}
        if source:
            metadata.update({"source_zone": source, "target_zone": target, "variant": name})
        fetch = keyed("entsoe", lambda: fetch_entsoe_series(keys["entsoe"], series_kind, zone, start, max(end, start + timedelta(days=1)), source_zone=source, target_zone=target))
        def apply(frame: pd.DataFrame) -> str:
            merged = store_entsoe(full_key, f"{label}, {name}", frame, metadata, keep)
            return f"{len(frame):,} new rows, {len(merged):,} held"
        tasks.append(UpdateTask(f"{label}, {name}", fetch, apply, start))

    price_zones = list(dict.fromkeys(list(UPDATE_DEFAULT_ZONES) + [z for z in _live_variants("entsoe_price") if z in ENTSOE_ZONES]))
    for zone in price_zones:
        entsoe_task("prices", "entsoe_price", "Day-ahead price", "Day-ahead price", zone, KEEP_DAYS["prices"], PRICE_HISTORY_DAYS, today + timedelta(days=2))
    for store_key, label, series_kind, keep in (("entsoe_load", "Actual load", "Actual load", KEEP_DAYS["load"]),
                                                ("entsoe_generation", "Actual generation", "Actual generation by type", KEEP_DAYS["generation"])):
        for zone in dict.fromkeys(list(UPDATE_DEFAULT_ZONES) + [z for z in _live_variants(store_key) if z in ENTSOE_ZONES]):
            entsoe_task("prices", store_key, label, series_kind, zone, keep, 30, today + timedelta(days=1))
    for store_key, label, series_kind in (("entsoe_load_forecast", "Load forecast", "Day-ahead load forecast"),
                                          ("entsoe_renewable_forecast", "Wind and solar forecast", "Day-ahead wind and solar forecast")):
        for zone in price_zones:
            if zone in ZONE_WEATHER:
                entsoe_task("prices", store_key, label, series_kind, zone, FORECAST_HISTORY_DAYS, FORECAST_HISTORY_DAYS, today + timedelta(days=2))
    for variant, item in _live_variants("entsoe_flow").items():
        meta = item.get("metadata", {})
        if meta.get("source_zone") in ENTSOE_ZONES and meta.get("target_zone") in ENTSOE_ZONES:
            entsoe_task("prices", "entsoe_flow", "Physical flow", "Physical flow", meta["source_zone"], KEEP_DAYS["load"], 30, today + timedelta(days=1),
                        meta["source_zone"], meta["target_zone"], variant)

    if _needs_update("gie_agsi_history", "fundamentals", force):
        held = _held_frame("gie_agsi_history")
        # Providers revise recent gas days, so each update reloads the last 45.
        start = incremental_start(held, "gasDayStart", 0, 45, today) if held is not None else date(today.year - 5, 1, 1)
        def apply_storage(frame: pd.DataFrame) -> str:
            meta = {"scope": "Europe / EU datasets", "country": ""}
            merged = store_observations("gie_agsi_history", "AGSI gas storage, 6 years", frame, meta, ("gasDayStart", "name"), "gasDayStart")
            _stored_live_item("gie_agsi", "AGSI gas storage", merged.tail(40).reset_index(drop=True), meta)
            return f"{len(frame):,} new rows, {len(merged):,} held"
        tasks.append(UpdateTask("EU gas storage", keyed("gie", lambda: fetch_gie_history("AGSI", keys["gie"], "Europe / EU datasets", start, today)), apply_storage, start))
    if _needs_update("gie_alsi", "fundamentals", force):
        held = _held_frame("gie_alsi")
        lng_start = incremental_start(held, "gasDayStart", 60, 30, today)
        def apply_lng(frame: pd.DataFrame) -> str:
            merged = store_observations("gie_alsi", "ALSI LNG terminals", frame, {"scope": "Europe / EU datasets", "country": ""}, ("gasDayStart", "name"), "gasDayStart", KEEP_DAYS["lng"])
            return f"{len(frame):,} new rows, {len(merged):,} held"
        tasks.append(UpdateTask("EU LNG terminals", keyed("gie", lambda: fetch_gie_history("ALSI", keys["gie"], "Europe / EU datasets", lng_start, today)), apply_lng, lng_start))
    flow_item = st.session_state.get("live_data", {}).get("entsog_flow")
    if isinstance(flow_item, dict) and flow_item.get("metadata", {}).get("point_direction") and _needs_update("entsog_flow", "fundamentals", force):
        point, indicator = flow_item["metadata"]["point_direction"], flow_item["metadata"].get("indicator", "Physical Flow")
        def apply_gas_flow(frame: pd.DataFrame) -> str:
            _stored_live_item("entsog_flow", f"ENTSOG {indicator}", frame, {"point_direction": point, "indicator": indicator})
            return f"{len(frame):,} rows"
        tasks.append(UpdateTask("Gas flow at your saved point", lambda: fetch_entsog_flow(point, today - timedelta(days=60), today - timedelta(days=1), indicator), apply_gas_flow))

    if _needs_update("weather_daily", "weather", force):
        open_meteo_key = keys["open_meteo"]
        fetch_weather = lambda: fetch_open_meteo_market_pack(list(WEATHER_LOCATIONS), 3, 10, open_meteo_key, bool(open_meteo_key))
        def apply_weather(outcome: tuple) -> str:
            hourly, daily, failures = outcome
            if daily.empty:
                raise RuntimeError(f"All {len(failures) or len(WEATHER_LOCATIONS)} locations failed.")
            meta = {"locations": list(WEATHER_LOCATIONS), "location": "Multiple locations", "forecast_days": 10}
            _stored_live_item("weather_hourly", "Open-Meteo hourly weather", hourly, meta)
            _stored_live_item("weather_daily", "Open-Meteo daily weather", daily, meta)
            st.session_state["weather_fetch_failures"] = failures
            return f"{daily['location'].nunique()} of {len(WEATHER_LOCATIONS)} locations"
        tasks.append(UpdateTask("Weather forecast", fetch_weather, apply_weather))
    if _needs_update("weather_normals", "normals", False) or _needs_update("weather_archive", "normals", False):
        def apply_archive(outcome: tuple) -> str:
            archive, normals, failures = outcome
            if normals.empty:
                raise RuntimeError(f"All {len(failures) or len(WEATHER_LOCATIONS)} locations failed.")
            _stored_live_item("weather_normals", "Open-Meteo temperature normals", normals, {"years": 10})
            _stored_live_item("weather_archive", "Open-Meteo daily temperature history", archive, {"years": 10})
            return f"{normals['location'].nunique()} of {len(WEATHER_LOCATIONS)} locations"
        tasks.append(UpdateTask("Weather history", lambda: fetch_weather_archive(list(WEATHER_LOCATIONS)), apply_archive))
    power_zones = [z for z in dict.fromkeys(list(UPDATE_DEFAULT_ZONES) + list(_live_variants("entsoe_price"))) if z in ZONE_WEATHER]
    grid_zones = [z for z in power_zones if z in ZONE_WEATHER_POINTS]
    if grid_zones and _needs_update("zone_weather_history", "power_weather", False):
        held_grid = _held_frame("zone_weather_history")
        archive_end = today - timedelta(days=5)
        def fetch_grid_history() -> tuple[pd.DataFrame, list[str]]:
            frames, failures = [], []
            for zone in grid_zones:
                held_zone = held_grid.loc[held_grid["zone"].eq(zone)] if held_grid is not None and "zone" in held_grid.columns else None
                start = incremental_start(held_zone if held_zone is not None and not held_zone.empty else None, "date", ZONE_WEATHER_HISTORY_DAYS, 3, today)
                if start > archive_end:
                    continue
                try:
                    frames.append(fetch_zone_weather(zone, start, archive_end))
                except Exception as exc:
                    failures.append(f"{zone}: {exc}")
            return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), failures
        def apply_grid_history(outcome: tuple) -> str:
            frame, failures = outcome
            if frame.empty and failures:
                raise RuntimeError("; ".join(failures))
            merged = store_observations("zone_weather_history", "Open-Meteo weather across each zone", frame, {"zones": grid_zones}, ("date", "zone"), "date", ZONE_WEATHER_HISTORY_DAYS + 20) if not frame.empty else held_grid
            return f"{len(frame):,} new days, {0 if merged is None else len(merged):,} held" + (f". Failed: {'; '.join(failures)}" if failures else "")
        tasks.append(UpdateTask("Weather across each power zone", fetch_grid_history, apply_grid_history))
    if grid_zones and _needs_update("zone_weather_forecast", "weather", force):
        open_meteo_key = keys["open_meteo"]
        def fetch_grid_forecast() -> tuple[pd.DataFrame, list[str]]:
            frames, failures = [], []
            for zone in grid_zones:
                try:
                    frames.append(fetch_zone_weather(zone, forecast=True, api_key=open_meteo_key, commercial_endpoint=bool(open_meteo_key)))
                except Exception as exc:
                    failures.append(f"{zone}: {exc}")
            return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), failures
        def apply_grid_forecast(outcome: tuple) -> str:
            frame, failures = outcome
            if frame.empty:
                raise RuntimeError("; ".join(failures) or "No zone returned a forecast.")
            _stored_live_item("zone_weather_forecast", "Open-Meteo forecast across each zone", frame, {"zones": grid_zones})
            return f"{frame['zone'].nunique()} of {len(grid_zones)} zones"
        tasks.append(UpdateTask("Weather forecast across each power zone", fetch_grid_forecast, apply_grid_forecast))
    # Zones with points across them use those; the single reference city is kept for the others.
    power_zones = [z for z in power_zones if z not in ZONE_WEATHER_POINTS]
    if power_zones and _needs_update("weather_power_history", "power_weather", False):
        def apply_power_weather(outcome: tuple) -> str:
            frame, failures = outcome
            if frame.empty:
                raise RuntimeError(f"All {len(failures) or len(power_zones)} locations failed.")
            _stored_live_item("weather_power_history", "Open-Meteo wind and solar history", frame, {"days": 430})
            return f"{frame['location'].nunique()} of {len(power_zones)} locations"
        tasks.append(UpdateTask("Wind and solar history", lambda: fetch_power_weather([ZONE_WEATHER[z] for z in power_zones]), apply_power_weather))
    held_auctions = _live_dataset_frame("eex_eua_auctions")
    auction_age = (today - pd.Timestamp(held_auctions["date"].iloc[-1]).date()).days if not held_auctions.empty and "date" in held_auctions.columns else None
    if force or auction_age is None or (auction_age >= 1 and _needs_update("eex_eua_auctions", "prices", False)):
        def apply_auctions(frame: pd.DataFrame) -> str:
            _stored_live_item("eex_eua_auctions", "EEX EUA primary auction prices", frame, {"source": "EEX public auction report"})
            last = frame.iloc[-1]
            return f"{pd.Timestamp(last['date']):%d %b}: €{float(last['price']):.2f}/t"
        tasks.append(UpdateTask("EUA auction prices", lambda: fetch_eua_auctions(today), apply_auctions))
    fx_age = st.session_state.get("fx_updated_at")
    if force or fx_age is None or (datetime.now(timezone.utc) - fx_age).total_seconds() / 3600.0 >= STALE_HOURS["fx"]:
        def apply_fx(outcome: tuple) -> str:
            (gbp_per_eur, _), (usd_per_eur, _) = outcome
            set_fx_rate("eur_per_gbp", 1.0 / gbp_per_eur, ecb=True)
            set_fx_rate("eur_per_usd", 1.0 / usd_per_eur, ecb=True)
            st.session_state.fx_updated_at = datetime.now(timezone.utc)
            set_fx_rate("fx_ecb_epoch", st.session_state.fx_updated_at.timestamp(), ecb=True)
            for widget in ("risk_gbp", "risk_usd", "spread_gbp", "spread_usd", "pnl_fx", "data_fx"):
                st.session_state.pop(widget, None)
            return f"EUR/GBP {gbp_per_eur:.4f}, EUR/USD {usd_per_eur:.4f}"
        tasks.append(UpdateTask("ECB FX rates", lambda: (fetch_ecb_rate("GBP"), fetch_ecb_rate("USD")), apply_fx))
    return tasks


def start_update(force: bool = True, tasks: Optional[list[UpdateTask]] = None) -> int:
    """Start fetching in the background so the app stays usable. Returns the number of sources."""
    if st.session_state.get("update_job"):
        return 0
    if tasks is None:
        import_curve_folder()
    planned = plan_update(force) if tasks is None else tasks
    if not planned:
        return 0
    executor = ThreadPoolExecutor(max_workers=MAX_UPDATE_WORKERS)
    st.session_state.update_job = {"tasks": planned, "futures": [executor.submit(task.fetch) for task in planned], "executor": executor}
    return len(planned)


def update_progress() -> tuple[int, int]:
    job = st.session_state.get("update_job")
    if not job:
        return 0, 0
    return sum(1 for future in job["futures"] if future.done()), len(job["futures"])


def collect_update() -> bool:
    """Apply a finished update on the main thread. Returns True when one has just finished."""
    job = st.session_state.get("update_job")
    if not job or not all(future.done() for future in job["futures"]):
        return False
    results = []
    for task, future in zip(job["tasks"], job["futures"]):
        try:
            results.append((task.name, True, task.apply(future.result())))
        except Exception as exc:
            results.append((task.name, False, str(exc).split(" Check your internet")[0]))
    job["executor"].shutdown(wait=False)
    st.session_state.update_job = None
    st.session_state.update_results = results
    st.session_state.update_finished_at = datetime.now(timezone.utc)
    st.session_state.reads_recorded = False
    return True


def maybe_auto_update() -> None:
    """Refresh stale data once when the app opens, unless switched off."""
    if st.session_state.get("auto_update_checked"):
        return
    st.session_state.auto_update_checked = True
    if os.getenv("MERIDIAN_NO_AUTO_UPDATE") or not fx_rate("auto_update"):
        return
    start_update(force=False)


def _ago(moment: datetime) -> str:
    minutes = int((datetime.now(timezone.utc) - moment).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    return f"{hours} h ago" if hours < 48 else f"{hours // 24} days ago"


def fetched_text(stamp: object) -> str:
    """A saved fetch time in words, for captions under the data."""
    try:
        return f"Fetched {_ago(pd.Timestamp(str(stamp)).to_pydatetime())}"
    except Exception:
        return "Fetch time unknown"


def _latest_fetch() -> Optional[datetime]:
    stamps = [str(item.get("fetched_at", "")) for item in st.session_state.get("live_data", {}).values() if isinstance(item, dict)]
    stamps = [s for s in stamps if s]
    try:
        return pd.Timestamp(max(stamps)).to_pydatetime() if stamps else None
    except Exception:
        return None


def _update_panel() -> None:
    """Sidebar control. While fetching, it refreshes itself until the data is ready."""
    done, total = update_progress()
    if total:
        if done == total:
            collect_update()
            st.rerun()
        st.progress(done / total, text=f"Updating market data, {done} of {total} sources")
        return
    if st.button("Update market data", type="primary", key="sidebar_update", width="stretch", help="Uses your saved keys and fetches only what's new."):
        count = start_update(force=True)
        if count:
            st.rerun()
    latest = _latest_fetch()
    results = st.session_state.get("update_results") or []
    failed = [r for r in results if not r[1]]
    line = f"Data updated {_ago(latest)}." if latest else "No market data yet."
    st.markdown(f'<div class="side-foot"><b>{html.escape(line)}</b></div>', unsafe_allow_html=True)
    absent = missing_keys_text()
    if absent:
        st.warning(absent, icon=":material/key:")
    if failed:
        with st.expander(f"{len(failed)} of {len(results)} sources failed"):
            for name, _, detail in failed:
                st.caption(f"{name}: {detail}")
    bad_files = [r for r in st.session_state.get("curve_import_results", []) if r["status"] != "Imported"]
    if bad_files:
        st.caption(f"{len(bad_files)} curve file(s) couldn't be imported. See Forward Curves.")


def render_update_panel() -> None:
    running = bool(st.session_state.get("update_job"))
    st.fragment(_update_panel, run_every="1.5s" if running else None)()
    automatic = st.checkbox("Update when I open the app", value=bool(fx_rate("auto_update")), key="auto_update_toggle",
                            help="Fetches stale data in the background each time you open the app.")
    if automatic != bool(fx_rate("auto_update")):
        set_fx_rate("auto_update", 1.0 if automatic else 0.0)


def update_button(key: str, label: str = "Update market data") -> None:
    """The same update, offered where the data is used."""
    done, total = update_progress()
    if total:
        st.button(f"Updating, {done} of {total} sources", key=key, disabled=True)
    elif st.button(label, type="primary", key=key):
        if start_update(force=True):
            st.rerun()


# =============================================================================
# SECTION 6C · CURVE FILES, READ RECORD AND WINTER OUTLOOK
# =============================================================================
def curve_folder() -> Path:
    configured = os.getenv("MERIDIAN_CURVES_DIR", "").strip()
    return Path(configured) if configured else Path(__file__).resolve().with_name(CURVE_FOLDER_NAME)


def _read_curve_bytes(content: bytes, suffix: str) -> pd.DataFrame:
    if suffix == ".xlsx":
        try:
            return pd.read_excel(io.BytesIO(content))
        except ImportError as exc:
            raise ValueError("Excel files need the openpyxl package. Save the file as CSV, or run: pip install openpyxl") from exc
    # sep=None detects commas or semicolons, which French Excel uses.
    return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding="utf-8-sig", dtype=str)


def import_curve_folder() -> list[dict]:
    """Bring in any new file from the curve folder. Each version of a file is processed once."""
    store = desk_store()
    folder = curve_folder()
    if store is None:
        return []
    try:
        folder.mkdir(exist_ok=True)
        files = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.casefold() in CURVE_FILE_TYPES and not p.name.startswith(("~", "."))),
                       key=lambda p: p.stat().st_mtime)
        seen = store.curve_file_seen()
    except Exception:
        return []
    results = []
    for path in files:
        try:
            content = path.read_bytes()
        except OSError as exc:
            results.append({"name": path.name, "status": "Failed", "detail": str(exc)})
            continue
        sha = hashlib.sha256(content).hexdigest()
        if seen.get(path.name) == sha:
            continue
        custom, metadata = st.session_state.get("custom_curves", {}), st.session_state.get("curve_metadata", {})
        today = date.today()
        shapes = {p: custom.get(p, make_sample_curve(p, today, DEMO_CURVE_MONTHS)) for p in CURVE_PRODUCTS}
        latest = {p: (curve_mark_date(metadata.get(p)), custom[p]) for p in custom if curve_mark_date(metadata.get(p))}
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime).date()
            parsed = parse_curve_table(_read_curve_bytes(content, path.suffix.casefold()), path.name, modified, shapes, latest, today)
            source = f"File: {path.name}"
            newest: dict = {}
            for item in parsed:
                store.save_snapshot(item["day"], item["curve"], source, replace=True)
                newest[item["product"]] = item
            applied = []
            for product, item in newest.items():
                held = curve_mark_date(metadata.get(product)) if product in custom else None
                if held is None or item["day"] >= held:
                    commit_custom_curve(product, item["curve"], {"source_label": source, "source_type": "Curve folder", "file_name": path.name,
                                                                 "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, curve_date=item["day"])
                    applied.append(product)
            bump("snapshots")
            days = sorted({item["day"] for item in parsed})
            when = f"{days[0]:%d %b %Y}" if len(days) == 1 else f"{len(days)} days to {days[-1]:%d %b %Y}"
            status, detail = "Imported", f"{len(newest)} product{'s' if len(newest) > 1 else ''}, {when}" + ("" if len(applied) == len(newest) else ". Newer marks kept for the others.")
        except Exception as exc:
            status, detail = "Failed", str(exc)
        store.record_curve_file(path.name, sha, status, detail)
        results.append({"name": path.name, "status": status, "detail": detail})
    if results:
        st.session_state.curve_import_results = results
        imported = sum(1 for r in results if r["status"] == "Imported")
        if imported:
            st.session_state["flash_message"] = f"Imported {imported} curve file(s)."
    return results


def apply_tradingview_curve(product: str, raw: pd.DataFrame) -> str:
    """Save each day's TradingView curve as that day's marks, and price with the latest one."""
    store = desk_store()
    custom, metadata = st.session_state.get("custom_curves", {}), st.session_state.get("curve_metadata", {})
    today = date.today()
    shapes = {p: custom.get(p, make_sample_curve(p, today, DEMO_CURVE_MONTHS)) for p in CURVE_PRODUCTS}
    latest = {p: (curve_mark_date(metadata.get(p)), custom[p]) for p in custom if curve_mark_date(metadata.get(p))}
    parsed = parse_curve_table(raw, "tradingview", today, shapes, latest, today)
    source = f"TradingView, {TRADINGVIEW_CURVES[product]['name']}, delayed"
    if store is not None:
        for item in parsed:
            store.save_snapshot(item["day"], item["curve"], source, replace=True)
    newest = parsed[-1]
    held = curve_mark_date(metadata.get(product)) if product in custom else None
    if held is None or newest["day"] >= held:
        commit_custom_curve(product, newest["curve"], {"source_label": source, "source_type": "TradingView", "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                            curve_date=newest["day"])
    bump("snapshots")
    _stored_live_item("tradingview_curves", "TradingView forward curves", raw.loc[raw["curve_date"].eq(newest["day"])], {"product": product, "days": len(parsed)})
    front = newest["curve"].iloc[0]
    return f"{len(newest['curve'])} months dated {newest['day']:%d %b}, front {pd.Timestamp(front['delivery_month']):%b %y} at {front['price']:.2f}" + (f", plus {len(parsed) - 1} earlier days" if len(parsed) > 1 else "")


def saved_reads() -> pd.DataFrame:
    def load() -> pd.DataFrame:
        store = desk_store()
        return store.load_reads() if store is not None else pd.DataFrame(columns=list(DeskStore.READ_COLUMNS))
    return session_memo("saved_reads", data_version("reads"), load)


def _price_through(market: str, today: date) -> Optional[date]:
    """The last price date behind a read, or None when the data is too old to call it today's read."""
    if market == "European gas":
        fill = _storage_daily(_live_dataset_frame("gie_agsi_history"))
        return today if not fill.empty and (today - fill["day"].iloc[-1].date()).days <= 4 else None
    zone = MARKET_ZONES.get(market)
    series = zone_prices(zone) if zone else pd.DataFrame()
    if series is None or series.empty:
        return None
    last = series["local"].dt.date.max()
    return last if (today - last).days <= 3 else None


def record_market_reads() -> None:
    """Save today's reads once they rest on current data. This builds the track record."""
    if st.session_state.get("reads_recorded"):
        return
    st.session_state.reads_recorded = True
    rows = []
    for market, read in market_reads().items():
        through = _price_through(market, valuation_date)
        if through is not None and read["summary"]["confidence"] != "None":
            rows.append(read_row(valuation_date, market, read["signals"], read["summary"], through, "Recorded"))
    if rows:
        _save(lambda s: s.save_reads(rows))
        bump("reads")
    record_price_estimates()


def track_paths() -> dict:
    def build() -> dict:
        paths = {"European gas": marks_path(saved_snapshots(), "TTF Natural Gas")}
        for market, zone in MARKET_ZONES.items():
            series = zone_prices(zone)
            if series is not None and not series.empty:
                paths[market] = spot_path(series)
        return paths
    return session_memo("track_paths", data_version("live", "snapshots"), build)


def ensure_replayed_reads() -> None:
    """Rebuild past reads from held data. Only dates not done yet are calculated."""
    def run() -> int:
        store = desk_store()
        if store is None:
            return 0
        held = saved_reads()
        replayed = held.loc[held["source"].eq("Replayed")] if not held.empty else held
        skip = set(zip(replayed["read_date"], replayed["market"])) if not replayed.empty else set()
        rows = []
        for market, zone in MARKET_ZONES.items():
            if zone in _live_variants("entsoe_price"):
                rows += replay_power_reads(market, zone, _live_dataset_frame(f"entsoe_price|{zone}"), _live_dataset_frame(f"entsoe_load|{zone}"),
                                           _live_dataset_frame(f"entsoe_generation|{zone}"), skip, valuation_date)
        gas_days = [d for d in track_paths()["European gas"]["days"] if d < valuation_date]
        rows += replay_gas_reads(_live_dataset_frame("gie_agsi_history"), _live_dataset_frame("gie_alsi"), gas_days, skip)
        if rows:
            _save(lambda s: s.save_reads(rows))
            bump("reads")
        return len(rows)
    session_memo("replay_done", data_version("live", "snapshots"), run)


def winter_outlook() -> dict:
    return session_memo("winter_outlook", (data_version("live"), valuation_date),
                        lambda: project_winter(_live_dataset_frame("gie_agsi_history"), _live_dataset_frame("weather_archive"),
                                               _live_dataset_frame("weather_daily"), valuation_date))


# =============================================================================
# SECTION 6D · REAL OR DEMO: WHERE EACH PRICE ON A PAGE COMES FROM
# =============================================================================
def eua_auction_price() -> Optional[tuple]:
    frame = _live_dataset_frame("eex_eua_auctions")
    if frame.empty or not {"date", "price"}.issubset(frame.columns):
        return None
    last = frame.iloc[-1]
    day = pd.Timestamp(last["date"]).date()
    return (float(last["price"]), day) if (valuation_date - day).days <= AUCTION_MAX_AGE_DAYS else None


def book_months() -> dict:
    """{product: delivery months held}, so past months are only added where a trade needs them."""
    trades = st.session_state.get("trades")
    held: dict = {}
    if trades is None or trades.empty:
        return held
    for row in trades.to_dict("records"):
        if str(row.get("status", "")).upper() in ("CANCELLED", "EXPIRED") or row.get("product") not in CURVE_PRODUCTS:
            continue
        try:
            spec = CURVE_PRODUCTS[row["product"]]
            segments = _month_segments(_to_date(row["delivery_start"]), _to_date(row["delivery_end"]), str(row.get("profile") or "BASE").upper(), spec["timezone"])
        except Exception:
            continue
        held.setdefault(row["product"], set()).update(month for month, _ in segments)
    return held


def composed_curves() -> tuple[dict, dict]:
    """Every curve with real prices put in wherever a free source has them, and each month's source."""
    def build() -> tuple[dict, dict]:
        custom = st.session_state.get("custom_curves", {})
        needed = book_months()
        current = date(valuation_date.year, valuation_date.month, 1)
        auction = eua_auction_price()
        curves_, sources = {}, {}
        for product in CURVE_PRODUCTS:
            base = custom[product] if product in custom else make_sample_curve(product, valuation_date, months=DEMO_CURVE_MONTHS)
            realised = {}
            zone = POWER_PRODUCT_ZONES.get(product)
            if zone:
                spec = PRODUCTS[product]
                realised = realised_months(zone_prices(zone), spec["profile"], spec["timezone"], valuation_date)
                # Past months only where the book delivers in them, so forward views still start this month.
                realised = {m: v for m, v in realised.items() if m >= current or m in needed.get(product, set())}
            try:
                curves_[product], sources[product] = compose_curve(product, base, SOURCE_IMPORTED if product in custom else SOURCE_DEMO, realised, valuation_date, auction)
            except Exception:
                curves_[product] = validate_curve(base)
                sources[product] = {m: SOURCE_IMPORTED if product in custom else SOURCE_DEMO for m in curves_[product]["delivery_month"]}
        return curves_, sources
    return session_memo("composed_curves", (data_version("live", "curves", "trades"), valuation_date), build)


def curve_sources() -> dict:
    return composed_curves()[1]


def product_price_state(product: str) -> str:
    state = price_state(curve_sources().get(product, {}).values())
    if state == "Real" and product in st.session_state.get("custom_curves", {}):
        status = curve_state(product, st.session_state["custom_curves"], st.session_state.get("curve_metadata", {}), valuation_date)
        if status["state"] != "Current":
            return "Old"
    return state


def snapshot_source(product: str) -> str:
    """How a day's saved marks are labelled, so history built on demo prices is never mistaken for real."""
    state = product_price_state(product)
    if product in st.session_state.get("custom_curves", {}) and state in ("Real", "Old"):
        return curve_source_label(product)
    return {"Real": "Free real prices", "Part real": PART_REAL_MARKS}.get(state, "Demo marks")


SHORT_NAMES = {"TTF Natural Gas": "TTF", "NBP Natural Gas": "NBP", "German Power Base": "German base", "German Power Peak": "German peak",
               "French Power Base": "French base", "French Power Peak": "French peak", "EUA Carbon": "EUA", "API2 Coal": "API2 coal"}


def _names(products: Sequence[str]) -> str:
    names = [SHORT_NAMES.get(p, p) for p in products]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def price_warning(products: Sequence[str]) -> str:
    """One sentence naming the curves a result relies on that aren't fully real and current."""
    states = {p: product_price_state(p) for p in dict.fromkeys(products)}
    demo = [p for p, s in states.items() if s == "Demo"]
    part = [p for p, s in states.items() if s == "Part real"]
    old = [p for p, s in states.items() if s == "Old"]
    parts = []
    if demo:
        parts.append(f"{_names(demo)} {'uses' if len(demo) == 1 else 'use'} demo prices")
    if part:
        parts.append(f"{_names(part)} {'uses' if len(part) == 1 else 'use'} real day-ahead prices for delivered days but demo prices for later months")
    if old:
        parts.append(f"{_names(old)} {'has' if len(old) == 1 else 'have'} out-of-date imported marks")
    return ("; ".join(parts) + ". Results that depend on these are illustrations, not market values.") if parts else ""


def book_products() -> list[str]:
    trades = st.session_state.get("trades")
    if trades is None or trades.empty:
        return []
    active = trades.loc[~trades["status"].astype(str).str.upper().isin(["CANCELLED", "EXPIRED"])]
    return sorted(active["product"].dropna().astype(str).unique())


# ---- inputs a page can declare: each returns (state, text) pairs ------------------
def _latest(frame: pd.DataFrame, columns: Sequence[str]) -> Optional[date]:
    column = _find_column(frame, columns) if not frame.empty else None
    if not column:
        return None
    stamps = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
    return stamps.max().date() if not stamps.empty else None


def src_spot() -> list:
    zones = [z for z in _live_variants("entsoe_price") if z in ENTSOE_ZONES]
    if not zones:
        return [("missing", "ENTSO-E day-ahead prices")]
    last = max(zone_prices(z)["local"].max() for z in zones if not zone_prices(z).empty) if any(not zone_prices(z).empty for z in zones) else None
    return [("real", f"ENTSO-E day-ahead prices{f' to {last:%d %b}' if last is not None else ''}")]


def src_power_system() -> list:
    loaded = bool(_live_variants("entsoe_load")) or bool(_live_variants("entsoe_generation"))
    return [("real", "ENTSO-E load and generation")] if loaded else [("missing", "ENTSO-E load and generation")]


def src_forecasts() -> list:
    return [("real", "ENTSO-E load, wind and solar forecasts")] if _live_variants("entsoe_load_forecast") else [("missing", "ENTSO-E forecasts")]


def src_storage() -> list:
    last = _latest(_live_dataset_frame("gie_agsi_history"), ("gasDayStart", "observation_date", "gas_day"))
    return [("real", f"GIE storage to {last:%d %b}")] if last else [("missing", "GIE storage")]


def src_lng() -> list:
    last = _latest(_live_dataset_frame("gie_alsi"), ("gasDayStart", "observation_date", "gas_day"))
    return [("real", f"GIE LNG to {last:%d %b}")] if last else [("missing", "GIE LNG")]


def src_weather() -> list:
    return [("real", "Open-Meteo weather")] if not _live_dataset_frame("weather_daily").empty else [("missing", "Open-Meteo weather")]


def src_flows() -> list:
    return [("real", "ENTSO-E and ENTSOG flows")] if _live_variants("entsoe_flow") or _live_variants("entsog") else []


def src_fx() -> list:
    stamp = float(st.session_state.get("settings", {}).get("fx_ecb_epoch", 0.0))
    if stamp > 0:
        return [("real", f"ECB FX rates of {datetime.fromtimestamp(stamp, timezone.utc):%d %b}")]
    if stamp < 0:
        return [("user", "FX rates set by you")]
    return [("demo", "Default FX rates")]


def src_book() -> list:
    trades = st.session_state.get("trades")
    if trades is None or trades.empty:
        return [("missing", "Trades")]
    demo = trades["counterparty"].astype(str).str.contains("(Demo)", regex=False)
    count = lambda n, word: f"{n} {word}{'s' if n != 1 else ''}"
    if demo.all():
        return [("demo", f"{count(len(trades), 'sample trade')}, not yours")]
    if demo.any():
        return [("user", f"{count(int((~demo).sum()), 'trade')} of yours"), ("demo", count(int(demo.sum()), "sample trade"))]
    return [("user", f"{count(len(trades), 'trade')} of yours")]


def src_curves(products: Sequence[str]) -> list:
    groups: dict = {}
    for product in dict.fromkeys(products):
        groups.setdefault(product_price_state(product), []).append(product)
    out = []
    real = groups.get("Real", [])
    if real:
        labels = {label for p in real for label in curve_sources()[p].values()}
        how = "auction price" if labels == {SOURCE_AUCTION} else "imported marks" if labels <= {SOURCE_IMPORTED} else "real marks"
        out.append(("real", f"{_names(real)} {how}"))
    if groups.get("Part real"):
        out.append(("part", f"{_names(groups['Part real'])}: day-ahead for delivered days, demo after"))
    if groups.get("Old"):
        out.append(("old", f"{_names(groups['Old'])} imported marks"))
    if groups.get("Demo"):
        out.append(("demo", f"{_names(groups['Demo'])} forward prices"))
    return out


def src_book_curves() -> list:
    return src_curves(book_products()) if book_products() else []


def src_history() -> list:
    history = saved_snapshots()
    held = book_products()
    if history.empty or not held:
        return [("missing", "Saved price history")]
    part = history.loc[history["product"].isin(held)]
    by_day = part.groupby("snapshot_date")["source"].agg(lambda s: "demo" if s.isin(("Demo marks", SYNTHETIC_HISTORY_SOURCE)).any() else "part" if s.eq(PART_REAL_MARKS).any() else "real")
    counts = by_day.value_counts()
    out = []
    for state, text in (("real", "real"), ("part", "part real"), ("demo", "demo")):
        if counts.get(state, 0):
            out.append((state, f"{int(counts[state])} day{'s' if counts[state] > 1 else ''} of {text} price history"))
    return out


def src_read_curves() -> list:
    fresh = fresh_curves(st.session_state.get("custom_curves", {}), st.session_state.get("curve_metadata", {}), valuation_date)
    if fresh:
        return [("real", f"{_names(sorted(fresh))} imported marks")]
    return [("info", "Curve signals off: they only use real forward marks")]


ALL_PRICED = tuple(CURVE_PRODUCTS)
PAGE_SOURCES = {
    "Market Overview": lambda: src_book() + src_curves(list(dict.fromkeys(book_products() + ["TTF Natural Gas", "French Power Base", "German Power Base"]))) + src_spot() + src_storage() + src_weather(),
    "Morning Brief": lambda: src_spot() + src_storage() + src_weather() + src_book() + src_book_curves() + src_history(),
    "Market Read": lambda: src_storage() + src_lng() + src_weather() + src_spot() + src_power_system() + src_read_curves(),
    "Market Fundamentals": lambda: src_storage() + src_lng() + src_weather() + src_spot() + src_power_system() + src_flows(),
    "Power Spot": lambda: src_spot() + src_weather() + src_forecasts(),
    "Spreads & Margins": lambda: src_curves([p for p in CURVE_PRODUCTS if p != "NBP Natural Gas"]) + src_fx(),
    "Forward Curves": lambda: src_curves(ALL_PRICED),
    "OTC Quote Builder": lambda: src_curves(list(PRODUCTS)),
    "Trading Book & MtM": lambda: src_book() + src_book_curves() + src_fx(),
    "Daily P&L": lambda: src_book() + src_history() + src_fx(),
    "Risk & Exposure": lambda: src_book() + src_curves(list(dict.fromkeys(book_products() + ["TTF Natural Gas", "EUA Carbon"]))) + src_history() + src_fx(),
    "Hedge Advisor": lambda: src_book() + src_book_curves() + src_history(),
    "Physical Operations": lambda: src_book(),
    "Live Data Integrations": lambda: src_spot() + src_power_system() + src_storage() + src_lng() + src_weather() + src_fx(),
    "Data & Assumptions": lambda: src_curves(ALL_PRICED) + src_fx(),
    "Model Validation": lambda: [("info", "Built-in test data, not market prices")],
}
SOURCE_LABELS = {"real": "Real", "part": "Part real", "demo": "Demo", "old": "Old", "missing": "Not loaded", "user": "Yours", "info": "Note"}


def page_sources_html(page: str) -> str:
    settings = st.session_state.get("settings", {})
    key = (data_version("live", "curves", "trades", "snapshots"), valuation_date, settings.get("fx_ecb_epoch"))
    return session_memo(f"sources|{page}", key, lambda: _page_sources_html(page))


def _page_sources_html(page: str) -> str:
    builder = PAGE_SOURCES.get(page)
    if builder is None:
        return ""
    try:
        items = builder()
    except Exception:
        return ""
    order = {"real": 0, "user": 1, "part": 2, "old": 3, "demo": 4, "missing": 5, "info": 6}
    chips = "".join(f'<span class="src {state}"><b>{SOURCE_LABELS[state]}</b>{html.escape(text)}</span>'
                    for state, text in sorted(items, key=lambda item: order.get(item[0], 9)))
    return f'<div class="src-row"><span class="src-title">Data on this page</span>{chips}</div>'


def current_page_name() -> str:
    label = st.session_state.get("nav_page", DEFAULT_NAV_ITEM)
    return label.split("  ", 1)[1].strip() if "  " in label else label


def prices_used_lines(products: Sequence[str]) -> list[tuple[str, str]]:
    """(product, how it's priced) for the brief and its PDF."""
    lines = []
    for product in dict.fromkeys(products):
        state = product_price_state(product)
        detail = {"Real": "real prices", "Part real": "real day-ahead for delivered days, demo prices for later months",
                  "Old": "real but out-of-date imported marks", "Demo": "demo prices"}.get(state, state)
        lines.append((product, detail))
    return lines
# =============================================================================
# SECTION 7 · MODEL CHECKS
# =============================================================================
def run_model_validation(current_trades: pd.DataFrame, current_curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run deterministic unit/integration checks for the app's calculation model.

    The checks use independent expected values for key formulas and separately probe
    that invalid trades/curves are rejected. A failure here is a functional defect or a
    current user-data issue; it is not a market-strategy performance backtest.
    """
    rows: list[dict[str, str]] = []

    def run_case(name: str, area: str, fn) -> None:
        try:
            detail = fn()
            rows.append({"Status": "PASS", "Area": area, "Check": name, "Details": str(detail)})
        except Exception as exc:
            rows.append({"Status": "FAIL", "Area": area, "Check": name, "Details": f"{type(exc).__name__}: {exc}"})

    def assert_close(actual: float, expected: float, tolerance: float = 1e-7) -> None:
        if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
            raise AssertionError(f"Expected {expected}, received {actual}.")

    def expected_rejection(fn, expected_text: str = "") -> str:
        try:
            fn()
        except ValueError as exc:
            if expected_text and expected_text.lower() not in str(exc).lower():
                raise AssertionError(f"Rejected for an unexpected reason: {exc}") from exc
            return f"Invalid input correctly rejected: {exc}"
        raise AssertionError("Invalid input was incorrectly accepted.")

    def check_product_registry() -> str:
        required = {"commodity", "zone", "currency", "unit", "default_price", "timezone"}
        if not PRODUCTS or any(not required.issubset(spec) for spec in PRODUCTS.values()):
            raise AssertionError("A product is missing required metadata.")
        for name, spec in PRODUCTS.items():
            ZoneInfo(spec["timezone"])
            if not math.isfinite(float(spec["default_price"])):
                raise AssertionError(f"Invalid default mark for {name}.")
        return f"{len(PRODUCTS)} products have complete metadata and valid IANA timezones."

    def check_energy_conversion() -> str:
        for mwh in (0.0, 1.0, 1234.567, 1_000_000.0):
            assert_close(mmbtu_to_mwh(mwh_to_mmbtu(mwh)), mwh, 1e-10)
        return "MWh ⇄ MMBtu round-trip passed for four test magnitudes."

    def check_gas_price_conversion() -> str:
        assert_close(price_eur_mwh_to_gbp_therm(THERM_PER_MWH, 1.0), 1.0, 1e-10)
        return "34.12141633 EUR/MWh at 1.0 EUR/GBP converts to 1.0 GBP/therm."

    def check_dst_spring() -> str:
        value = delivery_hours(date(2026, 3, 29), date(2026, 3, 30), "BASE", "Europe/Paris")
        if value != 23:
            raise AssertionError(f"Expected 23 hours on spring-forward day, received {value}.")
        return "Spring daylight-saving day correctly has 23 elapsed hours."

    def check_dst_autumn() -> str:
        value = delivery_hours(date(2026, 10, 25), date(2026, 10, 26), "BASE", "Europe/Paris")
        if value != 25:
            raise AssertionError(f"Expected 25 hours on autumn fallback day, received {value}.")
        return "Autumn daylight-saving day correctly has 25 elapsed hours."

    def check_peak_hours() -> str:
        value = delivery_hours(date(2026, 6, 1), date(2026, 6, 8), "PEAK", "Europe/Paris")
        if value != 60:
            raise AssertionError(f"Expected 60 weekday peak hours, received {value}.")
        return "Monday-to-Monday peak delivery correctly counts five weekdays × 12 hours."

    def check_curves() -> str:
        total_marks = 0
        for product, curve in current_curves.items():
            validated = validate_curve(curve)
            if not validated["product"].eq(product).all():
                raise AssertionError(f"Curve registered under {product} contains another product.")
            total_marks += len(validated)
        missing = set(PRODUCTS) - set(current_curves)
        if missing:
            raise AssertionError(f"No curve available for: {sorted(missing)}")
        return f"{len(current_curves)} product curves and {total_marks} monthly marks validated."

    def check_sample_curves_and_marks() -> str:
        valuation = date.today()
        sample_curves = {p: make_sample_curve(p, valuation, months=24) for p in PRODUCTS}
        sample_trades = build_sample_trades(valuation)
        marked = mark_trades(sample_trades, sample_curves)
        if len(marked) != len(sample_trades):
            raise AssertionError("Some sample trades were lost during marking.")
        if marked["mtm"].isna().any() or not marked["mtm"].map(math.isfinite).all():
            raise AssertionError("Sample MtM contains missing or non-finite values.")
        for _, row in marked.iterrows():
            expected = float(row["signed_quantity_mwh"]) * (float(row["market_price"]) - float(row["trade_price"]))
            assert_close(float(row["mtm"]), expected)
        return f"{len(marked)} sample trades marked with finite MtM and independently checked formula."

    def check_mtm_signs() -> str:
        assert_close(mtm_linear(100.0, 50.0, 55.0), 500.0)
        assert_close(mtm_linear(-100.0, 50.0, 55.0), -500.0)
        assert_close(mtm_linear(100.0, 55.0, 50.0), -500.0)
        assert_close(mtm_linear(100.0, 50.0, 50.0), 0.0)
        return "Long, short, adverse-market and at-market values match hand calculations."

    def check_daily_pnl() -> str:
        assert_close(daily_pnl(previous_mtm=1200.0, current_mtm=1450.0, realised_pnl=-25.0), 225.0)
        return "Daily P&L correctly equals change in MtM plus supplied realised P&L."

    def check_quote_margin() -> str:
        assert_close(quote_price(50.0, 1.25, "SELL_TO_CLIENT"), 51.25)
        assert_close(quote_price(50.0, 1.25, "BUY_FROM_CLIENT"), 48.75)
        assert_close(quote_price(-20.0, 1.25, "SELL_TO_CLIENT"), -18.75)
        return "Buy/sell quote margins behave correctly, including a negative power reference price."

    def check_parallel_shock() -> str:
        assert_close(parallel_shock_pnl(100.0, 5.0), 500.0)
        assert_close(parallel_shock_pnl(-100.0, 5.0), -500.0)
        assert_close(parallel_shock_pnl(100.0, -5.0), -500.0)
        return "Absolute price-shock P&L is linear, signed, and valid independent of the price level."

    def check_sample_trades() -> str:
        valuation = date.today()
        trades = build_sample_trades(valuation)
        validated = normalise_trades(trades)
        if validated["trade_id"].duplicated().any():
            raise AssertionError("Duplicate trade IDs found.")
        if (validated["quantity_mwh"] <= 0).any():
            raise AssertionError("Sample trade has non-positive delivered energy.")
        for _, trade in validated.iterrows():
            zone = PRODUCTS[trade["product"]]["timezone"]
            expected_hours = delivery_hours(trade["delivery_start"], trade["delivery_end"], trade["profile"], zone)
            assert_close(float(trade["quantity_mwh"]), float(trade["quantity_mw"]) * expected_hours)
        return f"{len(validated)} sample trades pass schema, unit, status, sign and delivery-volume checks."

    def check_executed_summary() -> str:
        valuation = date.today()
        sample_curves = {p: make_sample_curve(p, valuation, months=24) for p in PRODUCTS}
        marked = mark_trades(build_sample_trades(valuation), sample_curves)
        summary = position_summary(marked)
        executed = marked.loc[marked["status"].eq("EXECUTED")]
        expected_total = float(executed["signed_quantity_mwh"].sum())
        actual_total = float(summary["signed_mwh"].sum()) if not summary.empty else 0.0
        assert_close(actual_total, expected_total)
        if set(summary["currency"].unique()) - {"EUR", "GBP"}:
            raise AssertionError("Unexpected currency in exposure summary.")
        return "Portfolio net delivery volume reconciles to executed trades; results remain grouped by currency/product."

    def check_active_trade_filter() -> str:
        demo = build_sample_trades(date.today())
        demo.loc[demo.index[0], "status"] = "CANCELLED"
        demo.loc[demo.index[1], "status"] = "EXPIRED"
        active = active_trade_book(demo)
        if len(active) != len(demo) - 2:
            raise AssertionError("Cancelled or expired trades remain eligible for marking.")
        return "Cancelled and expired trades are excluded from the active trade book."

    def check_negative_power_prices() -> str:
        curve = pd.DataFrame([{
            "delivery_month": date(2026, 11, 1), "product": "French Power Base", "price": -35.0,
            "currency": "EUR", "unit": "EUR/MWh"
        }])
        accepted = validate_curve(curve)
        assert_close(float(accepted.iloc[0]["price"]), -35.0)
        return "Negative French power marks are accepted without corrupting price validation."

    def check_bad_trade_rejected() -> str:
        sample = build_sample_trades(date.today()).iloc[[0]].copy()
        sample["currency"] = "GBP"
        return expected_rejection(lambda: normalise_trades(sample), "currency mismatch")

    def check_duplicate_ids_rejected() -> str:
        sample = build_sample_trades(date.today()).iloc[[0, 0]].copy()
        return expected_rejection(lambda: normalise_trades(sample), "unique")

    def check_invalid_dates_rejected() -> str:
        sample = build_sample_trades(date.today()).iloc[[0]].copy()
        sample["delivery_end"] = sample["delivery_start"]
        return expected_rejection(lambda: normalise_trades(sample), "after delivery_start")

    def check_missing_trade_id_rejected() -> str:
        sample = build_sample_trades(date.today()).iloc[[0]].copy()
        sample["trade_id"] = None
        return expected_rejection(lambda: normalise_trades(sample), "trade IDs")

    def check_zero_quantity_rejected() -> str:
        sample = build_sample_trades(date.today()).iloc[[0]].copy()
        sample["quantity_mw"] = 0.0
        return expected_rejection(lambda: normalise_trades(sample), "positive and finite")

    def check_invalid_quote_margin_rejected() -> str:
        return expected_rejection(lambda: quote_price(50.0, -0.01, "SELL_TO_CLIENT"), "cannot be negative")

    def check_invalid_discount_factor_rejected() -> str:
        return expected_rejection(lambda: mtm_linear(100.0, 50.0, 55.0, discount_factor=2.0), "between 0 and 1.5")

    def check_unknown_timezone_rejected() -> str:
        return expected_rejection(lambda: delivery_hours(date(2026, 1, 1), date(2026, 1, 2), "BASE", "Unknown/Zone"), "timezone")

    def check_duplicate_curve_marks_rejected() -> str:
        sample = make_sample_curve("TTF Natural Gas", date.today(), months=3)
        sample = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
        return expected_rejection(lambda: validate_curve(sample), "duplicate")

    def check_incomplete_curve_rejected() -> str:
        valuation = date.today()
        curve = make_sample_curve("TTF Natural Gas", valuation, months=4)
        start = _month_start(1, valuation.replace(day=1))
        end = _month_start(4, valuation.replace(day=1))
        target_month = pd.Timestamp(_month_start(2, valuation.replace(day=1))).date()
        reduced = curve.loc[curve["delivery_month"].ne(target_month)]
        return expected_rejection(lambda: swap_fixed_price(reduced, start, end, "BASE"), "missing")

    def check_navigation_registry() -> str:
        pages = set(NAV_TO_PAGE.values())
        required = {"Market Overview", "Forward Curves", "OTC Quote Builder", "Trading Book & MtM", "Daily P&L", "Risk & Exposure",
                    "Physical Operations", "Market Fundamentals", "Power Spot", "Spreads & Margins", "Market Read", "Hedge Advisor", "Live Data Integrations", "Model Validation", "Data & Assumptions"}
        missing = required - pages
        if missing:
            raise AssertionError(f"Navigation is missing page(s): {sorted(missing)}")
        if len(ALL_NAV_ITEMS) != len(pages):
            raise AssertionError("Navigation labels map to duplicate page names.")
        return f"All {len(required)} expected pages are registered in the sidebar navigation."

    def check_current_book() -> str:
        if current_trades.empty:
            return "Current trade book is empty; marking test is skipped, not treated as a failure."
        marked = mark_trades_resilient(current_trades, current_curves)
        if len(marked) != len(current_trades):
            raise AssertionError("Current trade count does not reconcile after resilient marking.")
        active = marked.loc[~marked["status"].astype(str).str.upper().isin(["CANCELLED", "EXPIRED"])]
        unmarked = active.loc[active["mark_status"].ne("MARKED")]
        if not unmarked.empty:
            reasons = "; ".join(f"{r['trade_id']}: {r['mark_error']}" for _, r in unmarked.head(3).iterrows())
            raise AssertionError(f"{len(unmarked)} current trade(s) cannot be marked: {reasons}")
        return f"Current book successfully marks all {len(active)} active trade(s); terminal trades remain for audit."

    def check_forward_one_month_rule() -> str:
        sample = build_sample_trades(date.today()).iloc[[1]].copy()
        sample["instrument"] = "FORWARD"
        valid = normalise_trades(sample)
        if len(valid) != 1:
            raise AssertionError("A single-month forward was not accepted.")
        invalid = sample.copy()
        invalid["delivery_end"] = (pd.Timestamp(invalid["delivery_end"].iloc[0]).to_period("M") + 2).start_time.date().isoformat()
        return expected_rejection(lambda: normalise_trades(invalid), "one full calendar month")

    def check_isolated_mark_failure() -> str:
        demo = build_sample_trades(date.today())
        good = demo.iloc[[0]].copy()
        missing = demo.iloc[[0]].copy()
        missing["trade_id"] = "TEST-NBP-MISSING-CURVE"
        missing["product"] = "NBP Natural Gas"
        missing["currency"] = "GBP"
        missing["trade_price"] = 30.0
        combined = pd.concat([good, missing], ignore_index=True, sort=False)
        result = mark_trades_resilient(combined, {"TTF Natural Gas": make_sample_curve("TTF Natural Gas", date.today(), 24)})
        statuses = result.set_index("trade_id")["mark_status"].to_dict()
        if statuses.get(good["trade_id"].iloc[0]) != "MARKED" or statuses.get("TEST-NBP-MISSING-CURVE") != "UNMARKED":
            raise AssertionError(f"Unexpected isolated marking result: {statuses}")
        if not str(result.loc[result["trade_id"].eq("TEST-NBP-MISSING-CURVE"), "mark_error"].iloc[0]).strip():
            raise AssertionError("Unmarked trade has no diagnostic reason.")
        return "A missing NBP curve leaves that trade unmarked without blocking the valid TTF trade."

    def check_delivery_month_positions() -> str:
        valuation = date.today()
        demo = build_sample_trades(valuation)
        sample_curves = {p: make_sample_curve(p, valuation, 24) for p in PRODUCTS}
        marked = mark_trades_resilient(demo, sample_curves)
        detail = position_summary_by_delivery(marked, sample_curves)
        if detail.empty or not detail["delivery_month"].map(lambda x: isinstance(x, date)).all():
            raise AssertionError("Delivery-month position buckets were not generated.")
        ttf_months = detail.loc[detail["product"].eq("TTF Natural Gas"), "delivery_month"].nunique()
        if ttf_months != 3:
            raise AssertionError(f"Expected three TTF swap delivery buckets; received {ttf_months}.")
        if (detail["gross_mwh"] < detail["net_mwh"].abs() - 1e-6).any():
            raise AssertionError("Gross positions cannot be less than the absolute net position.")
        return f"Positions split into {len(detail)} product/month/profile buckets with net and gross MWh."

    def check_csv_import_and_lifecycle() -> str:
        sample = build_sample_trades(date.today())
        csv_text = sample.iloc[[0]].drop(columns=["quantity_mwh", "signed_quantity_mwh"], errors="ignore").to_csv(index=False)
        imported = load_trade_book_csv(io.StringIO(csv_text))
        if len(imported) != 1:
            raise AssertionError("CSV import did not produce one validated trade.")
        legacy_csv = imported.drop(columns=["instrument", "settlement_type"], errors="ignore").to_csv(index=False)
        legacy_loaded = load_trade_book_csv(io.StringIO(legacy_csv))
        if legacy_loaded.iloc[0]["instrument"] != "SWAP" or legacy_loaded.iloc[0]["settlement_type"] != "FINANCIAL":
            raise AssertionError("Legacy CSV defaults for instrument/settlement were not applied.")
        proposal = imported.assign(trade_id="TEST-PHYSICAL-LIFECYCLE", status="PROPOSED", settlement_type="PHYSICAL")
        appended = add_trade_to_book(sample, proposal)
        if len(appended) != len(sample) + 1:
            raise AssertionError("New trade was not appended.")
        confirmed = set_trade_status(appended, "TEST-PHYSICAL-LIFECYCLE", "CONFIRMED")
        ops = sync_physical_ops_with_trades(build_sample_physical_ops(sample), confirmed)
        if "TEST-PHYSICAL-LIFECYCLE" not in set(ops["trade_id"].astype(str)):
            raise AssertionError("Confirmed physical trade did not create an operations record.")
        cancelled = set_trade_status(confirmed, "TEST-PHYSICAL-LIFECYCLE", "CANCELLED")
        if cancelled.loc[cancelled["trade_id"].eq("TEST-PHYSICAL-LIFECYCLE"), "status"].iloc[0] != "CANCELLED":
            raise AssertionError("Cancellation was not retained in the blotter.")
        if "TEST-PHYSICAL-LIFECYCLE" in set(active_trade_book(cancelled)["trade_id"].astype(str)):
            raise AssertionError("Cancelled trade remains active for valuation.")
        ops_check = reconcile_physical_ops(ops, cancelled)
        if not ops_check.loc[ops_check["trade_id"].eq("TEST-PHYSICAL-LIFECYCLE"), "reconciliation"].iloc[0].startswith("TRADE CANCELLED"):
            raise AssertionError("Operations-to-trade lifecycle mismatch was not detected.")
        return "CSV load, legacy defaults, append, confirm, physical linkage, cancellation and audit reconciliation passed."

    def check_physical_ops_persistence_and_reconciliation() -> str:
        demo = build_sample_trades(date.today())
        ops = build_sample_physical_ops(demo)
        if ops.empty:
            raise AssertionError("No sample physical operations were generated.")
        contract = ops.iloc[0]
        updated = update_physical_operation(ops, contract["contract_id"],
                                            float(contract["contracted_mwh"]),
                                            float(contract["contracted_mwh"]), "DELIVERED", "ISSUED")
        check = reconcile_physical_ops(updated, demo)
        row = check.loc[check["contract_id"].eq(contract["contract_id"])].iloc[0]
        if row["reconciliation"] != "MATCH" or row["risk_flag"] != "RECONCILED":
            raise AssertionError(f"Physical delivery did not reconcile: {row['reconciliation']}, {row['risk_flag']}")
        if float(updated.loc[updated["contract_id"].eq(contract["contract_id"]), "delivered_mwh"].iloc[0]) != float(contract["contracted_mwh"]):
            raise AssertionError("Delivered quantity was not persisted in the operations table.")
        return "Physical delivery update persists and reconciles against its linked physical trade."

    def check_filtered_book_metrics() -> str:
        sample = build_sample_trades(date.today())
        sample.loc[sample.index[0], "status"] = "EXECUTED"
        sample.loc[sample.index[1], "status"] = "PROPOSED"
        sample_curves = {p: make_sample_curve(p, date.today(), 24) for p in PRODUCTS}
        marked = mark_trades_resilient(sample, sample_curves)
        proposed = filter_trade_book(marked, status_filter="PROPOSED")
        metrics = calculate_book_metrics(proposed)
        if metrics["trade_count"] != len(proposed) or metrics["executed_count"] != 0:
            raise AssertionError("Displayed book metrics do not match the status-filtered rows.")
        return "Filtered blotter KPIs are derived from the same rows as the displayed view."

    def check_physical_settlement_validation() -> str:
        sample = build_sample_trades(date.today()).iloc[[0]].copy()
        sample["settlement_type"] = "OTHER"
        return expected_rejection(lambda: normalise_trades(sample), "PHYSICAL or FINANCIAL")

    def check_forward_and_swap_valuation_methods() -> str:
        valuation = date.today()
        sample = build_sample_trades(valuation)
        sample_curves = {product_name: make_sample_curve(product_name, valuation, 24) for product_name in PRODUCTS}
        marked = mark_trades(sample, sample_curves).set_index("trade_id")
        fwd = marked.loc["TR-1002"]
        swp = marked.loc["TR-1001"]
        expected_fwd = forward_month_price(sample_curves["French Power Peak"], "French Power Peak",
                                           fwd["delivery_start"], fwd["delivery_end"])
        assert_close(float(fwd["market_price"]), expected_fwd)
        if "Single-month" not in str(fwd["valuation_method"]) or "swap-strip" not in str(swp["valuation_method"]).lower():
            raise AssertionError("Instrument-specific valuation methods are not displayed.")
        detail = trade_delivery_breakdown(marked.reset_index(), sample_curves)
        for trade_id in marked.index:
            bucket_mtm = detail.loc[detail["trade_id"].eq(trade_id), "mtm"].sum(min_count=1)
            assert_close(float(bucket_mtm), float(marked.loc[trade_id, "mtm"]), 1e-6)
        return "FORWARD uses its single monthly mark; SWAP uses a delivery-hour-weighted strip and monthly MtM reconciles."

    def check_entsoe_price_parser() -> str:
        fixture = b'''<?xml version="1.0" encoding="UTF-8"?>
        <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
          <TimeSeries><Period><timeInterval><start>2026-01-01T00:00Z</start><end>2026-01-01T02:00Z</end></timeInterval>
            <resolution>PT60M</resolution><Point><position>1</position><price.amount>80.5</price.amount></Point>
            <Point><position>2</position><price.amount>-12.0</price.amount></Point></Period></TimeSeries>
        </Publication_MarketDocument>'''
        parsed = parse_entsoe_xml(fixture, "price")
        if len(parsed) != 2 or list(parsed["value"]) != [80.5, -12.0]:
            raise AssertionError(f"Unexpected parsed values: {parsed.to_dict('records')}")
        if str(parsed.iloc[0]["timestamp"].tz) != "UTC":
            raise AssertionError("ENTSO-E timestamps must be UTC-aware.")
        # Exercise the compressed transport format used by some ENTSO-E responses.
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("market-data.xml.gz", gzip.compress(fixture))
        compressed = parse_entsoe_xml(archive.getvalue(), "price")
        if len(compressed) != 2 or list(compressed["value"]) != [80.5, -12.0]:
            raise AssertionError("ZIP/GZIP-wrapped ENTSO-E XML was not parsed correctly.")
        return "Namespaced XML and ZIP/GZIP-wrapped XML parse two hourly prices, including a negative price, with UTC timestamps."

    def check_provider_curve_normalisation() -> str:
        raw = pd.DataFrame({"delivery_date": ["2026-11-01", "2026-12-01"], "settlement": [72.5, -5.0]})
        mapped = normalise_provider_curve(raw, "French Power Base", "delivery_date", "settlement")
        if len(mapped) != 2 or mapped.iloc[1]["price"] != -5.0:
            raise AssertionError("Provider curve mapping failed or lost a negative power price.")
        return "Mapped vendor fields to canonical product/unit schema; negative electricity prices are preserved."

    def check_provider_numeric_parser() -> str:
        got = _number_series(pd.Series(["1.234,56", "-5,25", "-", "12.5"]))
        expected = [1234.56, -5.25, float("nan"), 12.5]
        for actual, want in zip(got, expected):
            if math.isnan(want):
                if not math.isnan(actual): raise AssertionError("Missing marker was not parsed as NaN.")
            else:
                assert_close(actual, want)
        return "Numeric parsing handles decimal commas, thousands separators, missing markers and negative values."

    def check_gie_request_and_data_mapping() -> str:
        calls = []
        def fake_get(url, params=None, headers=None):
            calls.append((url, params or {}, headers or {}))
            return {"last_page": 1, "data": [{
                "gasDayStart": "2026-10-20", "name": "Germany", "gasInStorage": "45.2",
                "full": "74.0", "workingGasVolume": "61.0", "status": "C"
            }]}
        got = fetch_gie_history("AGSI", "test-key", "Europe / EU datasets", date(2026, 10, 20), date(2026, 10, 20), "DE", _json_get_fn=fake_get)
        if not calls or str(calls[0][1].get("type", "")).casefold() != "eu" or "continent" in calls[0][1]:
            raise AssertionError("GIE Europe selection must use the documented type=EU parameter.")
        if calls[0][2].get("x-key") != "test-key":
            raise AssertionError("GIE API key was not sent in the x-key header.")
        assert_close(float(got.iloc[0]["gasInStorage"]), 45.2)
        if str(got.iloc[0]["observation_date"]) != "2026-10-20":
            raise AssertionError("GIE gasDayStart was not mapped to an observation date.")
        return "Mocked GIE request uses the documented type=eu selector, sends its API key in a header, and normalises dates/numeric fields."

    def check_entsog_envelope_normalisation() -> str:
        directions = _payload_rows({"operatorpointdirections": [{"pointDirection": "sample-id", "pointLabel": "Interconnector"}]})
        aggregated = _payload_rows({"AggregatedData": [{"periodFrom": "2026-10-01", "value": "1,25"}]})
        if not directions or directions[0].get("pointDirection") != "sample-id":
            raise AssertionError("Case-insensitive operator-point-direction envelope was not recognised.")
        if not aggregated or aggregated[0].get("value") != "1,25":
            raise AssertionError("Case-insensitive aggregated-data envelope was not recognised.")
        components = normalise_entsog_point_catalogue(pd.DataFrame([{
            "operatorKey": "FR-TSO-0001", "pointKey": "FR-POINT-0002", "directionKey": "exit"
        }]))
        expected_identifier = "FR-TSO-0001FR-POINT-0002exit"
        if components.iloc[0]["pointDirection"] != expected_identifier:
            raise AssertionError(f"ENTSOG component identifiers were not concatenated correctly: {components.iloc[0]['pointDirection']!r}")
        mixed_case = normalise_entsog_point_catalogue(pd.DataFrame([{
            "operator_key": "DE-TSO-0001", "point_key": "DE-POINT-0002", "direction_key": "entry"
        }]))
        if mixed_case.iloc[0]["pointDirection"] != "DE-TSO-0001DE-POINT-0002entry":
            raise AssertionError("ENTSOG underscored component field variants were not recognised.")
        return "ENTSOG JSON envelopes and both combined/component-derived pointDirection identifiers are handled."

    def check_open_meteo_aggregation_and_timezone() -> str:
        payload = {"hourly": {
            "time": ["2026-10-20T00:00", "2026-10-20T01:00", "2026-10-21T00:00"],
            "temperature_2m": [10, 12, 8], "wind_speed_10m": [4, 6, 5],
            "shortwave_radiation": [0, 20, 0],
        }}
        hourly, daily = fetch_open_meteo(52.3, 4.9, "Europe/Amsterdam", past_days=1, forecast_days=1, _json_get_fn=lambda *args, **kwargs: payload)
        if len(hourly) != 3 or len(daily) != 2:
            raise AssertionError("Open-Meteo observations were not correctly grouped by local delivery date.")
        assert_close(float(daily.iloc[0]["temperature_mean_c"]), 11.0)
        if not hourly["timezone"].eq("Europe/Amsterdam").all():
            raise AssertionError("Weather rows must retain the selected timezone metadata.")
        return "Mocked Open-Meteo hourly observations aggregate into local-date summaries with timezone metadata."

    def check_open_meteo_public_request_variables() -> str:
        calls = []
        payload = {"hourly": {
            "time": ["2026-10-24T00:00"], "temperature_2m": [11.0],
            "wind_speed_10m": [8.0], "wind_speed_100m": [13.0],
            "shortwave_radiation": [35.0], "precipitation": [0.0], "cloud_cover": [65.0],
        }}
        def fake_get(url, params=None, headers=None):
            calls.append((url, dict(params or {})))
            return payload
        fetch_open_meteo(48.8566, 2.3522, "Europe/Paris", past_days=0,
                         forecast_days=1, _json_get_fn=fake_get)
        if len(calls) != 1 or calls[0][0] != OPEN_METEO_FORECAST_BASE:
            raise AssertionError("Public Open-Meteo endpoint was not selected for non-commercial use.")
        params = calls[0][1]
        wanted = {"temperature_2m", "wind_speed_10m", "wind_speed_100m", "shortwave_radiation", "precipitation", "cloud_cover"}
        if not wanted.issubset(set(str(params.get("hourly", "")).split(","))):
            raise AssertionError("One or more energy-relevant weather variables were omitted from the request.")
        if "apikey" in params:
            raise AssertionError("The public endpoint should not require an API key.")
        return "Public endpoint request needs no API key and asks for temperature, 10/100 m wind, solar radiation, precipitation and cloud cover."

    def check_open_meteo_multi_location_forecast_pack() -> str:
        def fake_fetch(name, loc, past_days, forecast_days, api_key, commercial_endpoint):
            zone = loc["timezone"]
            today_local = datetime.now(ZoneInfo(zone)).date()
            t0 = datetime.combine(today_local, time(0, 0)).isoformat()
            t1 = datetime.combine(today_local + timedelta(days=1), time(0, 0)).isoformat()
            if name.startswith("Paris"):
                temps = [5.0, 10.0]
                winds = [12.0, 14.0]
                solar = [40.0, 100.0]
            else:
                temps = [24.0, 26.0]
                winds = [20.0, 25.0]
                solar = [80.0, 160.0]
            payload = {"hourly": {"time": [t0, t1], "temperature_2m": temps,
                "wind_speed_10m": winds, "wind_speed_100m": [x * 1.5 for x in winds],
                "shortwave_radiation": solar, "precipitation": [0.0, 1.0], "cloud_cover": [30.0, 50.0]}}
            h, d = fetch_open_meteo(loc["latitude"], loc["longitude"], zone, past_days=0,
                                    forecast_days=2, _json_get_fn=lambda *args, **kwargs: payload)
            return h, d
        hourly, daily, failures = fetch_open_meteo_market_pack(
            ["Paris / French power proxy", "Berlin / German power proxy"],
            past_days=0, forecast_days=2, _fetch_fn=fake_fetch)
        if failures:
            raise AssertionError(f"Unexpected mocked weather failures: {failures}")
        if set(daily["location"].dropna()) != {"Paris / French power proxy", "Berlin / German power proxy"}:
            raise AssertionError("Multi-location fetch did not preserve location identity.")
        if not {"heating_degree_day_proxy_18c", "cooling_degree_day_proxy_22c", "wind_speed_100m_mean_kmh", "solar_irradiation_sum_whm2"}.issubset(daily.columns):
            raise AssertionError("Weather risk/proxy metrics are missing from daily aggregation.")
        summary = weather_forecast_summary(daily, horizon_days=7)
        if len(summary) != 2 or not summary["Forecast days"].eq(1).all():
            raise AssertionError("Forecast summary did not select upcoming complete local days per location.")
        paris = summary.loc[summary["Location"].str.startswith("Paris")].iloc[0]
        berlin = summary.loc[summary["Location"].str.startswith("Berlin")].iloc[0]
        assert_close(float(paris["Heating degree-day proxy (base 18°C)"]), 8.0)
        assert_close(float(berlin["Cooling degree-day proxy (base 22°C)"]), 4.0)
        assert_close(float(paris["Daily solar irradiation (Wh/m²)"]), 100.0)
        return "Multi-zone Open-Meteo fetch preserves local timezones/locations and derives HDD/CDD, 100 m wind, daily solar irradiation, and per-zone forecast summaries."

    def check_open_meteo_partial_location_failure() -> str:
        zone = "Europe/Paris"
        today_local = datetime.now(ZoneInfo(zone)).date()
        times = [datetime.combine(today_local, time(0, 0)).isoformat(),
                 datetime.combine(today_local + timedelta(days=1), time(0, 0)).isoformat()]
        payload = {"hourly": {"time": times, "temperature_2m": [12.0, 13.0],
            "wind_speed_10m": [8.0, 9.0], "wind_speed_100m": [12.0, 13.5],
            "shortwave_radiation": [30.0, 60.0], "precipitation": [0.0, 0.0], "cloud_cover": [45.0, 40.0]}}
        def fake_fetch(name, loc, past_days, forecast_days, api_key, commercial_endpoint):
            if name.startswith("Berlin"):
                raise RuntimeError("simulated provider outage")
            return fetch_open_meteo(loc["latitude"], loc["longitude"], loc["timezone"], past_days=0,
                forecast_days=2, _json_get_fn=lambda *args, **kwargs: payload)
        hourly, daily, failures = fetch_open_meteo_market_pack(
            ["Paris / French power proxy", "Berlin / German power proxy"], _fetch_fn=fake_fetch)
        if daily.empty or set(daily["location"].dropna()) != {"Paris / French power proxy"}:
            raise AssertionError("One location failure caused successful weather data to be discarded.")
        if len(failures) != 1 or "Berlin" not in failures[0]:
            raise AssertionError("A partial weather-fetch failure was not surfaced against its location.")
        return "A failed weather location is reported separately while successful regional observations remain available."

    def check_generic_provider_rest_csv() -> str:
        calls = []
        def fake_http(url, params=None, headers=None, **kwargs):
            calls.append((url, params or {}, headers or {}))
            return b"delivery_month,settlement_price\n2026-11-01,90.5\n", "text/csv"
        raw = fetch_provider_table("https://provider.example/curves", "test-token", "Bearer token",
                                   query_params={"product": "TTF", "limit": 10}, _http_get_bytes_fn=fake_http)
        if not calls or calls[0][2].get("Authorization") != "Bearer test-token":
            raise AssertionError("Licensed provider REST adapter did not build the expected bearer header.")
        if calls[0][1] != {"product": "TTF", "limit": 10}:
            raise AssertionError("Provider query parameters were not passed through.")
        curve = normalise_provider_curve(raw, "TTF Natural Gas", "delivery_month", "settlement_price")
        assert_close(float(curve.iloc[0]["price"]), 90.5)
        return "Mocked licensed-provider REST CSV is parsed, authenticated, and normalised to the canonical monthly-curve schema."

    def check_live_snapshot_indicators() -> str:
        previous = st.session_state.get("live_data", {})
        try:
            st.session_state.live_data = {
                "gie_agsi": {"label": "AGSI EU", "metadata": {"scope": "Europe / EU datasets"}, "fetched_at": "2026-10-23T19:00:00+00:00", "data": pd.DataFrame([
                    {"gasDayStart": "2026-10-13", "observation_date": date(2026, 10, 13), "name": "EU", "code": "eu", "gasInStorage": 100.0, "workingGasVolume": 120.0, "full": 83.33},
                    {"gasDayStart": "2026-10-20", "observation_date": date(2026, 10, 20), "name": "EU", "code": "eu", "gasInStorage": 95.0, "workingGasVolume": 120.0, "full": 79.17},
                    {"gasDayStart": "2026-10-21", "observation_date": date(2026, 10, 21), "name": "EU", "code": "eu", "gasInStorage": 94.5, "workingGasVolume": 120.0, "full": 78.75},
                    {"gasDayStart": "2026-10-22", "observation_date": date(2026, 10, 22), "name": "EU", "code": "eu", "gasInStorage": 94.0, "workingGasVolume": 120.0, "full": 78.33},
                    {"gasDayStart": "2026-10-23", "observation_date": date(2026, 10, 23), "name": "EU", "code": "eu", "gasInStorage": 93.5, "workingGasVolume": 120.0, "full": 77.92},
                    {"gasDayStart": "2026-10-24", "observation_date": date(2026, 10, 24), "name": "EU", "code": "eu", "gasInStorage": 93.0, "workingGasVolume": 120.0, "full": 77.50},
                    {"gasDayStart": "2026-10-25", "observation_date": date(2026, 10, 25), "name": "EU", "code": "eu", "gasInStorage": 92.5, "workingGasVolume": 120.0, "full": 77.08},
                    {"gasDayStart": "2026-10-26", "observation_date": date(2026, 10, 26), "name": "EU", "code": "eu", "gasInStorage": 92.0, "workingGasVolume": 120.0, "full": 76.67},
                ])},
                "gie_alsi": {"label": "ALSI EU", "metadata": {"scope": "Europe / EU datasets"}, "fetched_at": "2026-10-23T19:00:00+00:00", "data": pd.DataFrame([{"gasDayStart": "2026-10-23", "observation_date": date(2026, 10, 23), "name": "EU", "code": "eu", "inventory": 520.0, "sendOut": 1100.0, "dtmi": 700.0}])},
                "entsoe_price": {"label": "ENTSO-E price", "metadata": {"zone": "France"}, "fetched_at": "2026-10-23T19:00:00+00:00", "data": pd.DataFrame([{"timestamp": "2026-10-23T11:00:00Z", "value": -5.0, "unit": "EUR/MWh", "series": "Day-ahead price"}, {"timestamp": "2026-10-23T12:00:00Z", "value": 80.0, "unit": "EUR/MWh", "series": "Day-ahead price"}])},
                "weather_daily": {"label": "Weather", "metadata": {"location": "Paris"}, "fetched_at": "2026-10-23T19:00:00+00:00", "data": pd.DataFrame([{"date": pd.Timestamp("2026-10-23"), "temperature_mean_c": 13.0, "wind_speed_mean_kmh": 15.0, "solar_radiation_mean_wm2": 90.0}])},
            }
            snap = build_live_snapshot()
            assert_close(snap["storage"]["fill_pct"], 76.67)
            assert_close(snap["storage"]["inventory_twh"], 92.0)
            assert_close(snap["storage"]["change_7_reports_twh"], -8.0)
            assert_close(snap["lng"]["sendout"], 1100.0)
            assert_close(snap["power_price"]["value"], 80.0)
            if snap["weather"]["location"] != "Paris":
                raise AssertionError("Weather location metadata was lost.")
            return "Market snapshot reads AGSI inventory/fill/change, ALSI send-out, latest ENTSO-E price and weather from mocked feed data without hardcoded market values."
        finally:
            st.session_state.live_data = previous

    def check_residual_load_proxy_alignment() -> str:
        previous = st.session_state.get("live_data", {})
        try:
            times = pd.date_range("2026-10-20T10:00:00Z", periods=4, freq="h")
            st.session_state.live_data = {
                "entsoe_price": {"data": pd.DataFrame({"timestamp": times, "value": [50, 55, 60, 65]})},
                "entsoe_load": {"data": pd.DataFrame({"timestamp": times, "value": [1000, 1100, 1200, 1300]})},
                "entsoe_generation": {"data": pd.DataFrame([
                    {"timestamp": t, "value": wind, "series": "Wind onshore"} for t, wind in zip(times, [100, 100, 100, 100])
                ] + [
                    {"timestamp": t, "value": solar, "series": "Solar"} for t, solar in zip(times, [25, 50, 75, 100])
                ] + [
                    {"timestamp": t, "value": 20, "series": "Nuclear"} for t in times
                ])},
            }
            result = _build_residual_load_proxy()
            if len(result) != 4:
                raise AssertionError(f"Expected four aligned hourly rows, got {len(result)}.")
            assert_close(float(result.iloc[0]["wind_solar_mw"]), 125.0)
            assert_close(float(result.iloc[0]["residual_load_proxy_mw"]), 875.0)
            return "Power context joins overlapping hourly timestamps and computes residual-load proxy as load minus wind/solar; unrelated generation types are excluded."
        finally:
            st.session_state.live_data = previous

    # ---- Checks added with the saved desk, risk engine and analytics ----------------
    def check_flow_direction() -> str:
        captured = {}
        def fake_http(url, params=None, headers=None):
            captured.update(params or {})
            return (b'<GL_MarketDocument xmlns="urn:x"><TimeSeries><Period><timeInterval><start>2026-01-01T00:00Z</start></timeInterval>'
                    b'<resolution>PT60M</resolution><Point><position>1</position><quantity>500</quantity></Point></Period></TimeSeries></GL_MarketDocument>'), "text/xml"
        fetch_entsoe_series("test", "Physical flow", "France", date(2026, 1, 1), date(2026, 1, 2),
                            source_zone="France", target_zone="Germany-Luxembourg", _http_fn=fake_http)
        if captured["out_Domain"] != ENTSOE_ZONES["France"]["eic"] or captured["in_Domain"] != ENTSOE_ZONES["Germany-Luxembourg"]["eic"]:
            raise AssertionError("A France to Germany flow must send out_Domain=France and in_Domain=Germany.")
        return "A France to Germany request names France as the sending area and Germany as the receiving area."

    def check_profile_must_match_product() -> str:
        base = build_sample_trades(date(2026, 11, 4)).iloc[[2]].copy()
        base["profile"] = "PEAK"
        expected_rejection(lambda: normalise_trades(base), "must use the BASE profile")
        peak = build_sample_trades(date(2026, 11, 4)).iloc[[1]].copy()
        peak["profile"] = "BASE"
        expected_rejection(lambda: normalise_trades(peak), "must use the PEAK profile")
        marked = mark_trades_resilient(base, {p: make_sample_curve(p, date(2026, 11, 4), 24) for p in PRODUCTS})
        if marked.iloc[0]["mark_status"] != "UNMARKED":
            raise AssertionError("A base product with a peak profile must not receive a mark.")
        return "Base products reject peak profiles and peak products reject base profiles, so no trade is marked on the wrong curve."

    def check_residual_refuses_mixed_zones() -> str:
        previous = st.session_state.get("live_data", {})
        try:
            t = pd.date_range("2026-10-20T10:00:00Z", periods=3, freq="h")
            st.session_state.live_data = {
                "entsoe_price": {"data": pd.DataFrame({"timestamp": t, "value": [50, 55, 60]}), "metadata": {"zone": "France"}},
                "entsoe_load": {"data": pd.DataFrame({"timestamp": t, "value": [1000, 1100, 1200]}), "metadata": {"zone": "Germany-Luxembourg"}},
                "entsoe_generation": {"data": pd.DataFrame({"timestamp": t, "value": [10, 20, 30], "series": "Solar"}), "metadata": {"zone": "Spain"}},
            }
            if not _build_residual_load_proxy().empty:
                raise AssertionError("Residual load combined three bidding zones.")
            return "Residual load returns nothing when price, load and generation come from different zones."
        finally:
            st.session_state.live_data = previous

    def check_parser_a03_consumption_and_errors() -> str:
        points = "".join(f"<Point><position>{p}</position><price.amount>{v}</price.amount></Point>" for p, v in [(1, 50), (5, 60), (20, 70)])
        a03 = (f'<Publication_MarketDocument xmlns="urn:x"><TimeSeries><curveType>A03</curveType><Period><timeInterval><start>2026-10-31T23:00Z</start>'
               f'<end>2026-11-01T23:00Z</end></timeInterval><resolution>PT60M</resolution>{points}</Period></TimeSeries></Publication_MarketDocument>').encode()
        parsed = parse_entsoe_xml(a03, "price")
        if len(parsed) != 24:
            raise AssertionError(f"A03 series should fill to 24 hours, got {len(parsed)}.")
        assert_close(float(parsed.iloc[3]["value"]), 50.0)
        assert_close(float(parsed.iloc[23]["value"]), 70.0)
        assert_close(float(parsed["value"].mean()), (4 * 50 + 15 * 60 + 5 * 70) / 24)
        def series(tag: str, value: int) -> str:
            return (f'<TimeSeries><{tag}>X</{tag}><MktPSRType><psrType>B10</psrType></MktPSRType><Period><timeInterval><start>2026-11-01T00:00Z</start></timeInterval>'
                    f'<resolution>PT60M</resolution><Point><position>1</position><quantity>{value}</quantity></Point></Period></TimeSeries>')
        mixed = ('<GL_MarketDocument xmlns="urn:x">' + series("inBiddingZone_Domain.mRID", 900) + series("outBiddingZone_Domain.mRID", 400) + '</GL_MarketDocument>').encode()
        names = set(parse_entsoe_xml(mixed, "generation")["series"])
        if names != {"Hydro pumped storage", "Hydro pumped storage (consumption)"}:
            raise AssertionError(f"Generation and consumption were not separated: {names}")
        ack = b'<Acknowledgement_MarketDocument xmlns="urn:x"><Reason><code>999</code><text>No matching data found</text></Reason></Acknowledgement_MarketDocument>'
        try:
            parse_entsoe_xml(ack, "price")
        except RuntimeError as exc:
            if "No matching data found" not in str(exc):
                raise AssertionError(f"Acknowledgement lost its reason: {exc}") from exc
        else:
            raise AssertionError("An acknowledgement document was accepted as data.")
        def http_error(url, params=None, headers=None):
            raise ProviderHTTPError("HTTP 400", 400, b'<Acknowledgement_MarketDocument xmlns="urn:x"><Reason><code>999</code><text>Delivered period too long</text></Reason></Acknowledgement_MarketDocument>')
        try:
            fetch_entsoe_series("secret-token", "Actual load", "France", date(2026, 1, 1), date(2026, 1, 2), _http_fn=http_error)
        except RuntimeError as exc:
            if "Delivered period too long" not in str(exc) or "secret-token" in str(exc):
                raise AssertionError(f"HTTP error text is wrong or exposes the token: {exc}") from exc
        else:
            raise AssertionError("An HTTP error was swallowed.")
        return "A03 gaps fill forward, consumption is kept apart from generation, and ENTSO-E's own reason reaches the user without the token."

    def check_long_fetch_is_chunked() -> str:
        windows = []
        def fake_http(url, params=None, headers=None, refuse_long=False):
            start, end = (datetime.strptime(params[k], "%Y%m%d%H%M") for k in ("periodStart", "periodEnd"))
            windows.append((end - start).days)
            if refuse_long and (end - start).days > 40:
                raise RuntimeError("Too long a period")
            return (f'<Publication_MarketDocument xmlns="urn:x"><TimeSeries><Period><timeInterval><start>{start:%Y-%m-%dT%H:%MZ}</start><end>{start + timedelta(hours=1):%Y-%m-%dT%H:%MZ}</end></timeInterval>'
                    f'<resolution>PT60M</resolution><Point><position>1</position><price.amount>{len(windows)}</price.amount></Point></Period></TimeSeries></Publication_MarketDocument>').encode(), "text/xml"
        frame = fetch_entsoe_series("test", "Day-ahead price", "France", date(2025, 1, 1), date(2027, 1, 1), _http_fn=fake_http)
        if len(windows) != 3 or max(windows) > 361 or len(frame) != 3:
            raise AssertionError(f"Two years should take three requests of up to 360 days, got {windows}.")
        windows.clear()
        fetch_entsoe_series("test", "Actual generation by type", "France", date(2026, 1, 1), date(2026, 4, 1), _http_fn=fake_http)
        if len(windows) != 3 or max(windows) > 32:
            raise AssertionError(f"Generation should use month-sized requests, got {windows}.")
        windows.clear()
        frame = fetch_entsoe_series("test", "Day-ahead price", "France", date(2026, 1, 1), date(2026, 4, 1), _http_fn=lambda u, params=None, headers=None: fake_http(u, params, headers, refuse_long=True))
        if windows[0] < 80 or len(windows) != 4 or max(windows[1:]) > 31 or len(frame) != 3:
            raise AssertionError(f"A refused long request should be retried in 30-day pieces, got {windows}.")
        return "Two years take three requests of up to 360 days, generation uses month-sized requests, and a refused long request is retried in 30-day pieces."

    def check_tenors_and_month_end() -> str:
        if contract_window("Q1-27") != (date(2027, 1, 1), date(2027, 4, 1)):
            raise AssertionError("Q1-27 window is wrong.")
        if contract_window("Win-26") != (date(2026, 10, 1), date(2027, 4, 1)) or contract_window("Sum-27") != (date(2027, 4, 1), date(2027, 10, 1)):
            raise AssertionError("Season window is wrong.")
        if contract_window("Cal-27") != (date(2027, 1, 1), date(2028, 1, 1)) or contract_window("Dec-26") != (date(2026, 12, 1), date(2027, 1, 1)):
            raise AssertionError("Year or month window is wrong.")
        if month_end_exclusive(date(2026, 12, 15), 3) != date(2027, 3, 15):
            raise AssertionError("A 3-month tenor from 15 December must end on 15 March.")
        expected_rejection(lambda: contract_window("Spring-27"), "unknown contract")
        choices = contract_choices(date(2026, 11, 4))
        if choices[0] != "Dec-26" or any(contract_window(c)[0] < date(2026, 12, 1) for c in choices):
            raise AssertionError("Contract list includes a contract already in delivery.")
        return "Months, quarters, seasons and years map to correct windows, and tenors keep their full length from any start day."

    def check_curve_decomposition() -> str:
        product = "French Power Base"
        built = decompose_tenor_quotes([("Q1-27", 90.0), ("Jan-27", 95.0), ("Cal-27", 80.0)], product)
        lookup = _curve_lookup(built)
        if len(lookup) != 12:
            raise AssertionError(f"Expected 12 monthly marks, got {len(lookup)}.")
        zone = PRODUCTS[product]["timezone"]
        for label, target in (("Q1-27", 90.0), ("Jan-27", 95.0), ("Cal-27", 80.0)):
            start, end = contract_window(label)
            assert_close(_strip_price(lookup, start, end, "BASE", zone), target, 1e-3)
        expected_rejection(lambda: decompose_tenor_quotes([("Q1-27", 90.0), ("Jan-27", 80.0), ("Feb-27", 80.0), ("Mar-27", 80.0)], product), "contradict")
        return "Contract quotes become monthly marks that reprice each contract exactly, and contradictory quotes are rejected."

    def check_saved_desk_roundtrip() -> str:
        with tempfile.TemporaryDirectory() as folder:
            store = DeskStore(Path(folder) / "test.db")
            trades = build_sample_trades(date(2026, 11, 4))
            store.save_table("trades", trades)
            restored = normalise_trades(store.load_table("trades"))
            if restored["trade_id"].tolist() != trades["trade_id"].tolist():
                raise AssertionError("Trades changed in the database.")
            assert_close(float(restored["signed_quantity_mwh"].sum()), float(trades["signed_quantity_mwh"].sum()))
            curve = make_sample_curve("TTF Natural Gas", date(2026, 11, 4), 6)
            store.save_custom_curve("TTF Natural Gas", curve, {"source_label": "Test"})
            loaded, meta = store.load_custom_curves()
            assert_close(float(loaded["TTF Natural Gas"]["price"].sum()), float(curve["price"].sum()))
            if meta["TTF Natural Gas"]["source_label"] != "Test":
                raise AssertionError("Curve metadata was lost.")
            store.save_snapshot(date(2026, 11, 4), curve, "Test")
            store.save_snapshot(date(2026, 11, 4), curve.assign(price=1.0), "Other", replace=False)
            if float(store.load_snapshots()["price"].iloc[0]) == 1.0:
                raise AssertionError("A first-of-day save overwrote existing marks.")
            frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"), "value": [1.5, -2.0], "series": "Day-ahead price"})
            store.save_live_item("entsoe_price|France", {"label": "x", "data": frame, "fetched_at": "t", "metadata": {"zone": "France"}})
            item = store.load_live_items()["entsoe_price|France"]
            if item["metadata"]["zone"] != "France" or not item["data"]["value"].tolist() == [1.5, -2.0] or str(item["data"]["timestamp"].dt.tz) != "UTC":
                raise AssertionError("Fetched data did not survive saving.")
            store.set_setting("eur_per_gbp", 1.2)
            assert_close(store.get_setting("eur_per_gbp", 1.0), 1.2)
        return "Trades, curves, daily marks, fetched data and settings all survive a save and reload."

    def check_bulk_marking_matches_and_is_fast() -> str:
        day = date(2026, 11, 4)
        book = pd.concat([build_sample_trades(day).assign(trade_id=lambda f: f["trade_id"] + f"-{i}") for i in range(170)], ignore_index=True)
        sample_curves = {p: make_sample_curve(p, day, 24) for p in PRODUCTS}
        began = perf_counter()
        fast = mark_trades_resilient(book, sample_curves)
        elapsed = perf_counter() - began
        one = pd.concat([mark_trades_resilient(book.iloc[[i]], sample_curves) for i in range(3)], ignore_index=True)
        for i in range(3):
            assert_close(float(fast.iloc[i]["mtm"]), float(one.iloc[i]["mtm"]))
        if elapsed > 3.0:
            raise AssertionError(f"Marking {len(book)} trades took {elapsed:.2f}s.")
        return f"{len(book)} trades mark in {elapsed * 1000:.0f} ms and match trade-by-trade results."

    def check_pnl_explain_by_hand() -> str:
        d0, d1 = date(2026, 11, 3), date(2026, 11, 4)
        m1, m2, m3 = date(2026, 11, 1), date(2026, 12, 1), date(2027, 1, 1)
        buckets = pd.DataFrame([
            {"trade_id": "OLD", "trade_date": date(2026, 10, 1), "product": "TTF Natural Gas", "currency": "EUR", "profile": "BASE", "delivery_month": m2, "net_mwh": 100.0, "trade_price": 30.0},
            {"trade_id": "OLD", "trade_date": date(2026, 10, 1), "product": "TTF Natural Gas", "currency": "EUR", "profile": "BASE", "delivery_month": m1, "net_mwh": 100.0, "trade_price": 30.0},
            {"trade_id": "NEW", "trade_date": d1, "product": "TTF Natural Gas", "currency": "EUR", "profile": "BASE", "delivery_month": m3, "net_mwh": -50.0, "trade_price": 40.0},
        ])
        previous = {"TTF Natural Gas": {m1: 31.0, m2: 32.0, m3: 36.0}}
        current = {"TTF Natural Gas": {m2: 35.0, m3: 38.0}}
        result = pnl_explain(buckets, previous, current, d0, d1)
        assert_close(float(result["price_move"].sum()), 300.0)      # 100 × (35 − 32)
        assert_close(float(result["new_trade"].sum()), 100.0)       # −50 × (38 − 40)
        assert_close(float(result["to_delivery"].sum()), 100.0)     # 100 × (31 − 30) leaves open MtM
        assert_close(float(result["previous_mtm"].sum()), 300.0)
        assert_close(float(result["current_mtm"].sum()), 600.0)
        assert_close(float(result["total_pnl"].sum()), 400.0)       # 600 − 300 + 100 realised
        return "Hand-built book reconciles: 300 price move + 100 new trade = 400 P&L, with 100 moving to realised."

    def check_scenarios_by_hand() -> str:
        first = date(2026, 12, 1)
        assert_close(scenario_shock("Parallel move", 5, "TTF Natural Gas", date(2027, 6, 1), first), 5.0)
        assert_close(scenario_shock("Curve steepens", 5, "TTF Natural Gas", first, first), -5.0)
        assert_close(scenario_shock("Curve steepens", 5, "TTF Natural Gas", date(2028, 11, 1), first), 5.0)
        assert_close(scenario_shock("Front-month squeeze", 6, "TTF Natural Gas", date(2027, 3, 1), first), 3.0)
        assert_close(scenario_shock("Winter premium", 4, "TTF Natural Gas", date(2027, 7, 1), first), -2.0)
        assert_close(scenario_shock("Gas rally, power follows", 2, "French Power Base", first, first), 3.0)
        assert_close(scenario_shock("TTF over NBP", 2, "NBP Natural Gas", first, first), 0.0)
        assert_close(scenario_shock("France over Germany", 2, "French Power Peak", first, first), 2.0)
        assert_close(scenario_shock("Peak over base", 2, "French Power Base", first, first), 0.0)
        buckets = pd.DataFrame([{"product": "NBP Natural Gas", "currency": "GBP", "delivery_month": first, "net_mwh": 10.0}])
        result = run_scenarios(buckets, 5.0, first, 1.2).set_index("scenario")
        assert_close(float(result.loc["Parallel move", "pnl_gbp_book"]), 50.0)
        assert_close(float(result.loc["Parallel move", "pnl_total_eur"]), 60.0)
        return "Each scenario's shock matches its definition, and GBP results convert to EUR at the entered rate."

    def check_var_by_hand() -> str:
        days = pd.bdate_range(end="2026-11-03", periods=101)
        month = _add_months(date.today(), 2)
        history = pd.DataFrame({"snapshot_date": [d.date().isoformat() for d in days], "product": "TTF Natural Gas",
                                "delivery_month": month.isoformat(), "price": [50.0 + (i % 2) for i in range(len(days))]})
        buckets = pd.DataFrame([{"product": "TTF Natural Gas", "currency": "EUR", "delivery_month": month, "net_mwh": 100.0}])
        result = historical_var(buckets, history, 1.17)
        if result["observations"] != 100:
            raise AssertionError(f"Expected 100 daily moves, got {result['observations']}.")
        assert_close(result["var"][0.95], 100.0)
        assert_close(result["es"][0.99], 100.0)
        missing = historical_var(pd.DataFrame([{"product": "NBP Natural Gas", "currency": "GBP", "delivery_month": month, "net_mwh": 1.0}]), history, 1.17)
        if missing["uncovered"] != ["NBP Natural Gas"]:
            raise AssertionError("A product without history must be reported as not covered.")
        return "A price alternating by ±1 on 100 MWh gives a VaR of 100, and products with no history are reported."

    def check_spreads_by_hand() -> str:
        month = date(2027, 1, 1)
        make = lambda product, price: pd.DataFrame([{"delivery_month": month, "product": product, "price": price, "currency": CURVE_PRODUCTS[product]["currency"], "unit": CURVE_PRODUCTS[product]["unit"]}])
        sample = {"French Power Base": make("French Power Base", 100.0), "TTF Natural Gas": make("TTF Natural Gas", 30.0),
                  "NBP Natural Gas": make("NBP Natural Gas", 25.0), "EUA Carbon": make("EUA Carbon", 70.0), "API2 Coal": make("API2 Coal", 100.0)}
        params = {"gas_efficiency": 0.5, "coal_efficiency": 0.4, "gas_emission_factor": 0.2, "coal_emission_factor": 0.34}
        row = spreads_table(sample, "French Power Base", "TTF Natural Gas", params, 1.2, 0.9).iloc[0]
        assert_close(float(row["clean_spark"]), 100.0 - 30.0 / 0.5 - 70.0 * 0.2 / 0.5)
        coal = 100.0 * 0.9 / COAL_MWH_PER_TONNE
        assert_close(float(row["clean_dark"]), 100.0 - coal / 0.4 - 70.0 * 0.34 / 0.4)
        # At the switching price, gas and coal must earn the same clean margin.
        at_switch = spreads_table({**sample, "TTF Natural Gas": make("TTF Natural Gas", float(row["switching_gas_price"]))}, "French Power Base", "TTF Natural Gas", params, 1.2, 0.9).iloc[0]
        assert_close(float(at_switch["clean_spark"]), float(at_switch["clean_dark"]))
        nbp = spreads_table(sample, "French Power Base", "NBP Natural Gas", params, 1.2, 0.9).iloc[0]
        assert_close(float(nbp["gas_eur_mwh"]), 30.0)
        return "Clean spark, clean dark, the switching price and NBP conversion all match hand calculations."

    def check_spot_analytics_by_hand() -> str:
        times = pd.date_range("2026-11-02T00:00:00Z", periods=4, freq="h")  # a Monday, 01:00 to 04:00 in Paris
        frame = pd.DataFrame({"timestamp": times, "value": [10.0, -20.0, 30.0, 40.0], "resolution": "PT60M"})
        series = spot_series(frame, "Europe/Paris")
        k = spot_kpis(series)
        assert_close(k["base"], 15.0)
        assert_close(k["negative_hours"], 1.0)
        quarter = spot_series(pd.DataFrame({"timestamp": pd.date_range("2026-11-02T00:00:00Z", periods=4, freq="15min"), "value": [-1.0, -1.0, 2.0, 2.0], "resolution": "PT15M"}), "Europe/Paris")
        assert_close(spot_kpis(quarter)["negative_hours"], 0.5)
        duration = price_duration_curve(series)
        assert_close(float(duration.iloc[0]["price"]), 40.0)
        assert_close(float(duration.iloc[-1]["share_of_hours"]), 100.0)
        generation = pd.DataFrame({"timestamp": times, "value": [0.0, 100.0, 100.0, 0.0], "series": "Solar"})
        captured = capture_prices(series, generation, k["base"]).set_index("technology")
        assert_close(float(captured.loc["Solar", "capture_price"]), 5.0)   # mean of −20 and 30
        assert_close(float(captured.loc["Solar", "capture_rate_pct"]), 5.0 / 15.0 * 100.0)
        return "Time-weighted averages, negative hours at 15-minute resolution, duration curve and capture price match hand calculations."

    def check_multi_zone_store() -> str:
        previous = st.session_state.get("live_data", {})
        try:
            st.session_state.live_data = {
                "entsoe_price|France": {"data": pd.DataFrame({"value": [1.0]}), "metadata": {"zone": "France"}, "fetched_at": "2026-11-01"},
                "entsoe_price|Spain": {"data": pd.DataFrame({"value": [2.0]}), "metadata": {"zone": "Spain"}, "fetched_at": "2026-11-02"},
            }
            if set(_live_variants("entsoe_price")) != {"France", "Spain"}:
                raise AssertionError("Fetching a second zone replaced the first.")
            assert_close(float(_live_dataset_frame("entsoe_price")["value"].iloc[0]), 2.0)
            assert_close(float(_live_dataset_frame("entsoe_price|France")["value"].iloc[0]), 1.0)
            return "Zones are stored side by side, and a plain dataset name resolves to the latest fetched."
        finally:
            st.session_state.live_data = previous

    def check_storage_season_and_hdd() -> str:
        rows = []
        for year in range(2021, 2027):
            rows.append({"gasDayStart": f"{year}-03-01", "full": float(year - 2000)})
        seasonal = storage_seasonal(pd.DataFrame(rows), today=date(2026, 3, 2))
        band = seasonal["band"].iloc[0]
        if seasonal["band_years"] != [2021, 2022, 2023, 2024, 2025]:
            raise AssertionError(f"Wrong comparison years: {seasonal['band_years']}")
        assert_close(float(band["low"]), 21.0)
        assert_close(float(band["high"]), 25.0)
        assert_close(float(band["average"]), 23.0)
        normals = temperature_normals_from_daily(pd.DataFrame({"date": ["2024-12-01", "2025-12-01"], "temperature": [8.0, 12.0]})).assign(location="Paris")
        assert_close(float(normals.iloc[0]["normal_hdd"]), 8.0)
        daily = pd.DataFrame({"date": ["2026-12-01"], "location": ["Paris"], "forecast_period": ["Forecast"],
                              "temperature_mean_c": [5.0], "heating_degree_day_proxy_18c": [13.0]})
        anomaly = hdd_vs_normal(daily, normals)
        assert_close(float(anomaly.iloc[0]["hdd_vs_normal"]), 5.0)
        assert_close(float(anomaly.iloc[0]["temperature_vs_normal_c"]), -5.0)
        return "Storage compares with the previous five years, and forecast heating degree days compare with normal."

    def check_ecb_and_archive_parsers() -> str:
        requested = {}
        def fake_ecb(url, params=None, headers=None):
            requested["url"] = url
            return b"KEY,TIME_PERIOD,OBS_VALUE\nEXR.D.GBP.EUR.SP00.A,2026-11-03,0.8500\n", "text/csv"
        rate, day = fetch_ecb_rate("GBP", _http_fn=fake_ecb)
        assert_close(rate, 0.85)
        if day != "2026-11-03" or "D.GBP.EUR.SP00.A" not in requested["url"]:
            raise AssertionError("ECB request or date is wrong.")
        def fake_archive(url, params=None, headers=None):
            return {"daily": {"time": ["2024-01-01", "2025-01-01"], "temperature_2m_mean": [2.0, 4.0]}}
        normals, failures = fetch_temperature_normals(["Paris / French power proxy"], _json_get_fn=fake_archive)
        if failures or normals.empty:
            raise AssertionError(f"Archive parsing failed: {failures}")
        assert_close(float(normals.iloc[0]["normal_temperature_c"]), 3.0)
        return "Mocked ECB and Open-Meteo archive responses parse into a rate and temperature normals."

    # ---- Hedge Advisor and Market Read ---------------------------------------------
    def _advisor_fixture():
        day = date(2026, 11, 4)
        history = synthetic_curve_history(day, list(PRODUCTS), business_days=80)
        sample_curves = {p: make_sample_curve(p, day, 36) for p in PRODUCTS}
        return day, history, sample_curves

    def _advisor_book(day, product="TTF Natural Gas", contract="Q1-27", mw=10.0, side="BUY_FIXED"):
        start, end = contract_window(contract)
        spec = PRODUCTS[product]
        return normalise_trades(pd.DataFrame([{"trade_id": f"A{side}{product}", "trade_date": day.isoformat(), "product": product, "side": side, "delivery_start": start, "delivery_end": end,
                                               "profile": spec["profile"], "quantity_mw": mw, "trade_price": 30.0, "currency": spec["currency"], "status": "EXECUTED"}]))

    def check_hedge_exact_offset() -> str:
        day, history, sample_curves = _advisor_fixture()
        buckets = book_buckets(_advisor_book(day))
        plan = hedge_plan(buckets, history, sample_curves, day, 1.17, products=["TTF Natural Gas"])
        first = plan["hedges"][0]
        if (first["product"], first["contract"], first["action"]) != ("TTF Natural Gas", "Q1-27", "Sell"):
            raise AssertionError(f"Expected to sell TTF Q1-27, got {first['action']} {first['product']} {first['contract']}.")
        assert_close(first["mw"], -10.0)
        if plan["hedged_var"] > 1.0 or plan["base_var"] < 100.0:
            raise AssertionError(f"An identical offset should remove the risk: {plan['base_var']} to {plan['hedged_var']}.")
        assert_close(first["cost_eur"], 10.0 * 2159 * DEFAULT_HALF_SPREADS["Gas"], 1e-6)
        return "A 10 MW long in TTF Q1-27 gets 'sell 10 MW TTF Q1-27', VaR falls to zero, and the cost matches volume times half-spread."

    def check_hedge_var_matches_engine() -> str:
        day, history, sample_curves = _advisor_fixture()
        moves, days = _contract_moves(history)
        buckets = book_buckets(_advisor_book(day))
        pnl, _, rows = _book_series(buckets, moves, days, day, 1.17)
        by_hand = np.zeros(len(days))
        for month, hours in _month_segments(*contract_window("Q1-27"), "BASE", "Europe/Amsterdam"):
            tenor = (month.year - day.year) * 12 + month.month - day.month
            table_ = moves["TTF Natural Gas"]
            nearest = min(table_.columns, key=lambda t: abs(int(t) - tenor))
            by_hand += table_[nearest].reindex(days).fillna(0.0).to_numpy() * 10.0 * hours
        if not np.allclose(pnl, by_hand):
            raise AssertionError("Book P&L doesn't match a separate calculation.")
        drivers = risk_drivers(rows, pnl, 0.95)
        assert_close(float(drivers["share_pct"].sum()), 100.0, 1e-6)
        return "Book P&L matches a separate calculation, and risk driver shares add to 100%."

    def check_hedge_edge_cases() -> str:
        day, history, sample_curves = _advisor_fixture()
        flat = pd.concat([_advisor_book(day), _advisor_book(day, side="SELL_FIXED")], ignore_index=True)
        plan = hedge_plan(book_buckets(flat), history, sample_curves, day, 1.17)
        if plan["hedges"]:
            raise AssertionError("A flat book must get no hedge.")
        short = history.loc[history["snapshot_date"].isin(sorted(history["snapshot_date"].unique())[:5])]
        if hedge_plan(book_buckets(_advisor_book(day)), short, sample_curves, day, 1.17)["ready"]:
            raise AssertionError("Too little history must not produce a plan.")
        if hedge_plan(pd.DataFrame(), history, sample_curves, day, 1.17)["ready"]:
            raise AssertionError("An empty book must not produce a plan.")
        no_history = hedge_plan(book_buckets(_advisor_book(day)), history.loc[history["product"].ne("TTF Natural Gas")], sample_curves, day, 1.17)
        if no_history["uncovered"] != ["TTF Natural Gas"] or any(h["product"] == "TTF Natural Gas" for h in no_history["hedges"]):
            raise AssertionError("A product with no history must be reported, not hedged.")
        proxy = hedge_plan(book_buckets(_advisor_book(day)), history, sample_curves, day, 1.17, products=["NBP Natural Gas"], min_improvement=0.01)
        if proxy["hedges"] and not all(h["proxy"] for h in proxy["hedges"]):
            raise AssertionError("A hedge in another product must be labelled as a proxy.")
        gbp = hedge_plan(book_buckets(_advisor_book(day, "NBP Natural Gas")), history, sample_curves, day, 2.0, products=["NBP Natural Gas"])
        low = hedge_plan(book_buckets(_advisor_book(day, "NBP Natural Gas")), history, sample_curves, day, 1.0, products=["NBP Natural Gas"])
        assert_close(gbp["base_var"], low["base_var"] * 2.0, 1e-6)
        return "Flat books, thin history, empty books and missing products behave correctly, proxies are labelled and GBP converts."

    def check_hedge_proposed_trades() -> str:
        day, history, sample_curves = _advisor_fixture()
        book = _advisor_book(day)
        plan = hedge_plan(book_buckets(book), history, sample_curves, day, 1.17, products=["TTF Natural Gas"])
        trades = proposed_hedge_trades(plan["hedges"], day, "T")
        row = trades.iloc[0]
        if row["status"] != "PROPOSED" or row["side"] != "SELL_FIXED":
            raise AssertionError("A hedge must enter the book as a proposed sale.")
        assert_close(float(row["trade_price"]), round(plan["hedges"][0]["mid"] - DEFAULT_HALF_SPREADS["Gas"], 4))
        combined = add_trade_to_book(book, trades)
        if len(combined) != len(book) + len(trades):
            raise AssertionError("Proposed hedges didn't join the book.")
        executed = combined.copy()
        executed["status"] = "EXECUTED"
        after = hedge_plan(book_buckets(executed), history, sample_curves, day, 1.17, products=["TTF Natural Gas"])
        if after["base_var"] > 1.0 or after["hedges"]:
            raise AssertionError("Once executed, the hedge should leave almost no risk and need no further hedge.")
        if plan["out_of_sample"] is None or plan["out_of_sample"]["hedged_std"] > plan["out_of_sample"]["unhedged_std"]:
            raise AssertionError("The direct hedge should hold up on unseen days.")
        own = custom_hedge_effect(book_buckets(book), history, day, 1.17, "TTF Natural Gas", "Q1-27", -10.0)
        if own["after"] > 1.0:
            raise AssertionError("Test-your-own should match the recommended hedge.")
        worse = custom_hedge_effect(book_buckets(book), history, day, 1.17, "TTF Natural Gas", "Q1-27", 10.0)
        assert_close(worse["after"], own["before"] * 2.0, 1e-6)
        return "Hedges enter as proposed trades at mid less the spread, remove the risk once executed, and hold up on unseen days."

    def check_hedge_fits_the_book() -> str:
        day, history, sample_curves = _advisor_fixture()
        book = normalise_trades(pd.concat([_advisor_book(day, "TTF Natural Gas", "Q1-27", 10.0), _advisor_book(day, "German Power Base", "Q2-27", 8.0)], ignore_index=True))
        buckets = book_buckets(book)
        net = _book_net(buckets)
        plan = hedge_plan(buckets, history, sample_curves, day, 1.17, products=["TTF Natural Gas", "German Power Base"], max_contracts=5, min_improvement=0.001)
        if not plan["hedges"]:
            raise AssertionError("This book should get a hedge.")
        for hedge in plan["hedges"]:
            spec = PRODUCTS[hedge["product"]]
            segments = _month_segments(hedge["start"], hedge["end"], spec["profile"], spec["timezone"])
            held = sum(h for m, h in segments if m in net[hedge["product"]]) / sum(h for _, h in segments)
            if held < 0.6:
                raise AssertionError(f"{hedge['contract']} doesn't sit on months the book holds.")
            if hedge["action"] != "Sell" or abs(hedge["mw"]) > (10.0 if hedge["product"].startswith("TTF") else 8.0) + 1e-9:
                raise AssertionError(f"{hedge['action']} {hedge['mw']} MW doesn't oppose or exceeds the position.")
        after = hedge_ladder_rows(buckets, plan["hedges"])
        for product in net:
            before_gross = sum(abs(v) for v in net[product].values())
            after_gross = after.loc[after["product"].eq(product), "net_mwh"].abs().sum()
            if after_gross > before_gross + 1e-6:
                raise AssertionError(f"Hedges increased the total {product} position.")
        if _hedge_fit("TTF Natural Gas", _month_segments(*contract_window("Cal-28"), "BASE", "Europe/Amsterdam"), net) is not None:
            raise AssertionError("A contract away from the book's months must not be offered.")
        return "Hedges sit on months the book holds, oppose the position, never exceed it, and leave each total position smaller."

    def check_read_summary_logic() -> str:
        strong = [_signal("a", "A", 2.0, 2.0, flip="x"), _signal("b", "B", 1.0, 2.0), _signal("c", "C", 1.0, 1.0)]
        result = summarise_read(strong)
        assert_close(result["score"], (2 + 1 + 0.5) / 4)
        if result["reading"] != "Bullish" or result["confidence"] != "High":
            raise AssertionError(f"Unanimous evidence read as {result['reading']}, {result['confidence']}.")
        mixed = summarise_read([_signal("a", "A", 1.0, 2.0), _signal("b", "B", 1.0, -2.0), _signal("c", "C", 1.0, 0.0)])
        if mixed["reading"] != "Balanced" or mixed["confidence"] != "Low":
            raise AssertionError(f"Opposing evidence read as {mixed['reading']}, {mixed['confidence']}.")
        bearish = summarise_read([_signal("a", "A", 1.0, -1.0), _signal("b", "B", 1.0, -0.4), _signal("c", "C", 2.0, None, missing="m")])
        if not bearish["reading"].endswith("earish") or bearish["confidence"] == "High" or len(bearish["missing"]) != 1:
            raise AssertionError(f"Partial bearish evidence handled wrongly: {bearish['reading']}, {bearish['confidence']}.")
        weak = summarise_read([_signal("a", "A", 3.0, 0.3), _signal("b", "B", 1.0, 0.0)])
        if weak["reading"] != "Balanced" or weak["confidence"] == "High":
            raise AssertionError("Weak evidence must not read as a high-confidence call.")
        one = summarise_read([_signal("a", "A", 1.0, 2.0), _signal("b", "B", 1.0)])
        if one["reading"] != "Not enough data":
            raise AssertionError("One signal must not produce a reading.")
        if _signal("x", "X", 1.0, 99.0)["score"] != 1.0 or _signal("x", "X", 1.0, float("nan"))["available"]:
            raise AssertionError("Scores must be capped and invalid values refused.")
        return "Weighted scores, labels, confidence, missing data and score limits all behave as defined."

    def check_gas_read_by_hand() -> str:
        today = date(2026, 11, 4)
        rows = []
        for year in range(2021, 2027):
            for offset in range(0, 40):
                day = date(year, 10, 1) + timedelta(days=offset)
                if day <= today:
                    fill = 60.0 + (year - 2021) * 2.0 if year < 2026 else 50.0
                    change = -0.1 * offset if year < 2026 else -0.5 * offset
                    rows.append({"gasDayStart": day.isoformat(), "full": fill + change})
        history = pd.DataFrame(rows)
        lng = pd.DataFrame({"gasDayStart": [(today - timedelta(days=i)).isoformat() for i in range(30)],
                            "sendOut": [5000.0 if i < 7 else 4000.0 + (i % 3) for i in range(30)]})
        signals = {s["key"]: s for s in gas_signals(history, lng, pd.DataFrame(), pd.DataFrame(), {}, {}, today)}
        if not signals["storage_level"]["available"] or signals["storage_level"]["score"] <= 0:
            raise AssertionError("Storage below its average must read as bullish.")
        if not signals["storage_pace"]["available"] or signals["storage_pace"]["score"] <= 0:
            raise AssertionError("A faster draw than normal must read as bullish.")
        if not signals["lng"]["available"] or signals["lng"]["score"] >= 0:
            raise AssertionError("More LNG must read as bearish.")
        for key in ("summer_winter", "switching", "weather"):
            if signals[key]["available"]:
                raise AssertionError(f"{key} must wait for its data. Demo prices must never count.")
        stale = {s["key"]: s for s in gas_signals(history, lng, pd.DataFrame(), pd.DataFrame(), {}, {}, date(2027, 2, 1))}
        if stale["storage_level"]["available"] or "days old" not in stale["storage_level"]["missing"]:
            raise AssertionError("Old storage data must not drive a reading.")
        ttf = pd.DataFrame({"delivery_month": [_add_months(date(2027, 4, 1), i) for i in range(12)], "product": "TTF Natural Gas", "price": [40.0] * 6 + [36.0] * 6, "currency": "EUR", "unit": "EUR/MWh"})
        inverted = {s["key"]: s for s in gas_signals(history, lng, pd.DataFrame(), pd.DataFrame(), {"TTF Natural Gas": ttf}, {}, today)}
        if inverted["summer_winter"]["score"] <= 0:
            raise AssertionError("Winter below summer must read as bullish.")
        return "Low storage, a fast draw and an inverted winter read bullish, more LNG reads bearish, and old data and demo curves never count."

    def check_power_read_by_hand() -> str:
        today = date(2026, 11, 4)
        hours = pd.date_range("2026-10-05T00:00:00Z", periods=24 * 30, freq="h")
        rising = np.where(hours >= pd.Timestamp("2026-10-28", tz="UTC"), 100.0, 60.0) + np.sin(np.arange(len(hours)))
        prices = pd.DataFrame({"timestamp": hours, "value": rising, "resolution": "PT60M"})
        load = pd.DataFrame({"timestamp": hours, "value": np.where(hours >= pd.Timestamp("2026-10-28", tz="UTC"), 60000.0, 50000.0) + 100 * np.sin(np.arange(len(hours)))})
        generation = pd.concat([pd.DataFrame({"timestamp": hours, "value": 5000.0, "series": "Solar"}),
                                pd.DataFrame({"timestamp": hours, "value": 900.0, "series": "Solar (consumption)"})])
        signals = {s["key"]: s for s in power_signals("France", prices, load, generation, pd.DataFrame(), pd.DataFrame(), {}, today)}
        if signals["momentum"]["score"] <= 0 or signals["residual"]["score"] <= 0:
            raise AssertionError("Rising prices and residual load must read as bullish.")
        if "GW" not in signals["residual"]["value"] or signals["heating" if "heating" in signals else "weather"]["available"]:
            raise AssertionError("Residual text or missing weather handled wrongly.")
        residual = _hourly_residual(load, generation)
        assert_close(float(residual.iloc[0]["residual"]), float(load.iloc[0]["value"]) - 5000.0)
        falling = prices.assign(value=np.where(hours >= pd.Timestamp("2026-10-28", tz="UTC"), -5.0, 60.0))
        negative = {s["key"]: s for s in power_signals("France", falling, load, generation, pd.DataFrame(), pd.DataFrame(), {}, today)}["negative"]
        if negative["score"] >= 0:
            raise AssertionError("More negative hours must read as bearish.")
        gas = pd.DataFrame({"delivery_month": [date(2026, 12, 1)], "product": "TTF Natural Gas", "price": [30.0], "currency": "EUR", "unit": "EUR/MWh"})
        eua = pd.DataFrame({"delivery_month": [date(2026, 12, 1)], "product": "EUA Carbon", "price": [70.0], "currency": "EUR", "unit": "EUR/tCO2"})
        fuel = {s["key"]: s for s in power_signals("France", prices, load, generation, pd.DataFrame(), pd.DataFrame(), {"TTF Natural Gas": gas, "EUA Carbon": eua}, today)}["fuel"]
        if "88.3" not in fuel["value"] or fuel["score"] >= 0:
            raise AssertionError(f"Gas plant cost should be 30/0.5 + 70×0.202/0.5 = 88.3: {fuel['value']}")
        return "Rising prices and residual load read bullish, negative hours read bearish, consumption stays out of residual load, and plant cost matches by hand."

    def check_merge_observations() -> str:
        times = pd.date_range("2026-10-01", periods=3, freq="D", tz="UTC")
        old = pd.DataFrame({"timestamp": times, "series": "S", "value": [1.0, 2.0, 3.0]})
        new = pd.DataFrame({"timestamp": times[1:].append(pd.DatetimeIndex([times[-1] + pd.Timedelta(days=1)])), "series": "S", "value": [20.0, 30.0, 40.0]})
        merged = merge_observations(old, new, ("timestamp", "series"), "timestamp")
        if merged["value"].tolist() != [1.0, 20.0, 30.0, 40.0]:
            raise AssertionError(f"Merge lost revisions or order: {merged['value'].tolist()}")
        trimmed = merge_observations(old, new, ("timestamp", "series"), "timestamp", keep_days=1)
        if trimmed["value"].tolist() != [30.0, 40.0]:
            raise AssertionError(f"Retention is wrong: {trimmed['value'].tolist()}")
        two = merge_observations(old, old.assign(series="T"), ("timestamp", "series"), "timestamp")
        if len(two) != 6:
            raise AssertionError("Different series at one time must both be kept.")
        days = merge_observations(pd.DataFrame({"gasDayStart": ["2026-10-01"], "name": ["EU"], "full": [1.0]}), pd.DataFrame({"gasDayStart": ["2026-10-01", "2026-10-02"], "name": ["EU", "EU"], "full": [2.0, 3.0]}), ("gasDayStart", "name"), "gasDayStart")
        if days["full"].tolist() != [2.0, 3.0]:
            raise AssertionError("Gas-day revision was not applied.")
        if merge_observations(None, new, ("timestamp", "series"), "timestamp")["value"].tolist() != [20.0, 30.0, 40.0]:
            raise AssertionError("A first fetch must be kept as it is.")
        return "New rows replace revised ones, other series stay, old rows are trimmed, and rows stay in time order."

    def check_update_plan_is_incremental() -> str:
        previous = st.session_state.get("live_data", {})
        previous_fx = st.session_state.get("fx_updated_at")
        today = date(2026, 11, 4)
        try:
            st.session_state.live_data = {}
            st.session_state.pop("fx_updated_at", None)
            first = {t.name: t for t in plan_update(force=True, today=today)}
            for wanted in ("Day-ahead price, France", "Day-ahead price, Germany-Luxembourg", "Actual load, France", "Actual generation, Germany-Luxembourg", "EU gas storage", "EU LNG terminals", "Weather forecast", "Weather history", "ECB FX rates"):
                if wanted not in first:
                    raise AssertionError(f"{wanted} is missing from a first update.")
            if first["Day-ahead price, France"].start != today - timedelta(days=PRICE_HISTORY_DAYS) or first["EU gas storage"].start != date(2021, 1, 1):
                raise AssertionError("A first update must fetch the full window.")
            now = datetime.now(timezone.utc).isoformat()
            stamp = pd.Timestamp("2026-11-03T22:00:00Z")
            st.session_state.live_data = {
                "entsoe_price|France": {"data": pd.DataFrame({"timestamp": [stamp - pd.Timedelta(days=360), stamp], "value": [1.0, 1.0], "series": ["P", "P"]}), "fetched_at": now, "metadata": {}},
                "entsoe_price|Spain": {"data": pd.DataFrame({"timestamp": [stamp], "value": [1.0], "series": ["P"]}), "fetched_at": "2026-11-01T00:00:00+00:00", "metadata": {}},
                "gie_agsi_history": {"data": pd.DataFrame({"gasDayStart": ["2026-11-03"], "full": [80.0]}), "fetched_at": now, "metadata": {}},
                "weather_normals": {"data": pd.DataFrame({"x": [1]}), "fetched_at": now, "metadata": {}},
                "weather_archive": {"data": pd.DataFrame({"x": [1]}), "fetched_at": now, "metadata": {}},
            }
            again = {t.name: t for t in plan_update(force=True, today=today)}
            if again["Day-ahead price, France"].start != date(2026, 11, 1):
                raise AssertionError(f"Only new days should be requested: {again['Day-ahead price, France'].start}")
            if again["EU gas storage"].start != date(2026, 9, 19):
                raise AssertionError(f"Storage should reload only recent days: {again['EU gas storage'].start}")
            if again["Day-ahead price, Spain"].start != today - timedelta(days=PRICE_HISTORY_DAYS):
                raise AssertionError("Prices held for less than a year should be extended once.")
            if "Day-ahead price, Spain" not in again or "Weather history" in again:
                raise AssertionError("Zones you added must update, and recent normals must not reload.")
            quiet = {t.name for t in plan_update(force=False, today=today)}
            if "Day-ahead price, France" in quiet or "Day-ahead price, Spain" not in quiet:
                raise AssertionError("Opening the app must skip recent data and fetch stale data.")
            return "A first update fetches full windows, later ones only new days, and opening the app skips anything recent."
        finally:
            st.session_state.live_data = previous
            if previous_fx is not None:
                st.session_state.fx_updated_at = previous_fx

    def check_update_job_lifecycle() -> str:
        saved_job = st.session_state.get("update_job")
        saved_results = st.session_state.get("update_results")
        applied = []
        def fail():
            raise RuntimeError("provider down. Check your internet connection")
        try:
            st.session_state.update_job = None
            count = start_update(tasks=[UpdateTask("Good", lambda: 5, lambda v: applied.append(v) or "ok"), UpdateTask("Bad", fail, lambda v: "never"),
                                        UpdateTask("Bad apply", lambda: 1, lambda v: (_ for _ in ()).throw(ValueError("bad frame")))])
            if count != 3 or start_update(tasks=[UpdateTask("Second", lambda: 1, lambda v: "x")]) != 0:
                raise AssertionError("Only one update may run at a time.")
            st.session_state.update_job["executor"].shutdown(wait=True)
            if update_progress() != (3, 3) or not collect_update():
                raise AssertionError("A finished update was not collected.")
            results = {name: (good, detail) for name, good, detail in st.session_state.update_results}
            if applied != [5] or not results["Good"][0] or results["Bad"] != (False, "provider down.") or results["Bad apply"][0]:
                raise AssertionError(f"One failure must not affect the others: {results}")
            if collect_update() or st.session_state.update_job is not None:
                raise AssertionError("The job must clear once collected.")
            return "Fetches run in the background, results apply once, and one failing source doesn't affect the others."
        finally:
            st.session_state.update_job = saved_job
            st.session_state.update_results = saved_results

    def check_pnl_history_matches_explain() -> str:
        day = date(2026, 11, 4)
        lookups = snapshot_lookup(synthetic_curve_history(day, list(PRODUCTS), business_days=40))
        days = sorted(lookups)
        book = build_sample_trades(day)
        book["trade_date"] = [days[5].isoformat(), days[15].isoformat(), days[0].isoformat()]
        buckets = book_buckets(book)
        # Remove one month from the curve for part of the period, to exercise to-delivery and first marks.
        target = buckets.iloc[0]["delivery_month"]
        for moment in days[25:]:
            lookups[moment][buckets.iloc[0]["product"]].pop(target, None)
        fast = pnl_history(buckets, lookups, 1.2)
        for row, (before, after) in zip(fast.to_dict("records"), zip(days, days[1:])):
            step = pnl_explain(buckets, lookups[before], lookups[after], before, after)
            factor = step["currency"].map({"EUR": 1.0, "GBP": 1.2}) if not step.empty else 1.0
            assert_close(row["pnl_eur"], float((step["total_pnl"] * factor).sum()) if not step.empty else 0.0, 1e-6)
            assert_close(row["mtm_eur"], float((step["current_mtm"] * factor).sum()) if not step.empty else 0.0, 1e-6)
        return f"All {len(fast)} days of the fast history equal the detailed day-by-day calculation."

    def check_saved_format_compatibility() -> str:
        with tempfile.TemporaryDirectory() as folder:
            store = DeskStore(Path(folder) / "formats.db")
            frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"), "value": [1.5, float("nan"), -2.0], "series": "S",
                                  "observation_date": [date(2026, 1, 1), date(2026, 1, 2), None]})
            store.save_live_item("new", {"label": "n", "data": frame, "fetched_at": "t", "metadata": {}})
            restored = store.load_live_items()["new"]["data"]
            if str(restored["timestamp"].dt.tz) != "UTC" or restored["value"].isna().sum() != 1 or float(restored["value"].iloc[2]) != -2.0:
                raise AssertionError("Compact format changed the data.")
            old = gzip.compress(frame.drop(columns="observation_date").to_json(orient="split", date_format="iso", date_unit="s", index=False).encode())
            with store._connect() as conn:
                conn.execute("INSERT INTO live_data VALUES (?,?,?,?,?)", ("old", "o", "{}", "t", old))
                conn.execute("INSERT INTO live_data VALUES (?,?,?,?,?)", ("bad", "b", "{}", "t", b"junk"))
            items = store.load_live_items()
            if "bad" in items or float(items["old"]["data"]["value"].iloc[2]) != -2.0 or str(items["old"]["data"]["timestamp"].dt.tz) != "UTC":
                raise AssertionError("Data saved by an earlier version must still load.")
        return "The compact format keeps types and gaps, earlier saves still load, and damaged rows are skipped."

    def check_line_thinning() -> str:
        y = np.sin(np.arange(20000) / 50.0)
        y[12345] = 99.0
        y[777] = -99.0
        trace = go.Scatter(x=np.arange(20000), y=y, mode="lines")
        _thin_line(trace)
        if len(trace.x) > MAX_LINE_POINTS or float(np.max(trace.y)) != 99.0 or float(np.min(trace.y)) != -99.0:
            raise AssertionError("Thinning lost a spike or kept too many points.")
        if np.any(np.diff(np.asarray(trace.x)) <= 0):
            raise AssertionError("Thinning broke time order.")
        short = go.Scatter(x=[1, 2, 3], y=[1.0, 2.0, 3.0], mode="lines")
        _thin_line(short)
        if len(short.x) != 3:
            raise AssertionError("Short lines must be left alone.")
        return "A 20,000-point line reduces to the chart's limit, keeps its extremes and stays in order."

    def check_entsoe_production_types() -> str:
        def series(code: str, value: int) -> str:
            return (f'<TimeSeries><inBiddingZone_Domain.mRID>X</inBiddingZone_Domain.mRID><MktPSRType><psrType>{code}</psrType></MktPSRType><Period><timeInterval><start>2026-11-01T00:00Z</start></timeInterval>'
                    f'<resolution>PT60M</resolution><Point><position>1</position><quantity>{value}</quantity></Point><Point><position>2</position><quantity>{value + 1}</quantity></Point></Period></TimeSeries>')
        codes = [f"B{i:02d}" for i in range(1, 21)] + ["B25"]
        document = ('<GL_MarketDocument xmlns="urn:x">' + "".join(series(code, i) for i, code in enumerate(codes)) + '</GL_MarketDocument>').encode()
        parsed = parse_entsoe_xml(document, "generation")
        if parsed["series"].nunique() != len(codes) or len(parsed) != 2 * len(codes):
            raise AssertionError(f"Production types were merged: {parsed['series'].nunique()} of {len(codes)}.")
        if ENTSOE_PSR_NAMES["B03"] == ENTSOE_PSR_NAMES["B05"]:
            raise AssertionError("Coal-derived gas and hard coal must be separate.")
        return "All ENTSO-E generation types stay separate, including coal-derived gas and hard coal."

    # ---- Keys, curve files, track record and winter outlook ------------------------
    def check_keys_file() -> str:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "keys.toml"
            path.write_text('# my keys\nENTSOE_API_TOKEN = "abc-123"\nGIE_API_KEY=\'xyz\'\nOPEN_METEO_API_KEY = ""\n', encoding="utf-8")
            found = keys_from_file(path)
        if found != {"ENTSOE_API_TOKEN": "abc-123", "GIE_API_KEY": "xyz"}:
            raise AssertionError("The keys file was read incorrectly.")
        if keys_from_file(Path("/nonexistent/keys.toml")) != {}:
            raise AssertionError("A missing keys file must give no keys, not an error.")
        return "Keys are read from the file, blank ones are ignored, and a missing file gives no keys."

    def check_curve_files() -> str:
        today = date(2026, 11, 5)
        shapes = {"TTF Natural Gas": make_sample_curve("TTF Natural Gas", today, 36)}
        monthly = pd.DataFrame({"Product": ["TTF", "TTF"], "Delivery Month": ["2026-12-01", "01/01/2027"], "Settle": ["31,50", "32.25"]})
        parsed = parse_curve_table(monthly, "ttf_2026-11-04.csv", today, shapes, {}, today)
        assert_close(parsed[0]["curve"]["price"].iloc[0], 31.5)
        if parsed[0]["day"] != date(2026, 11, 4) or len(parsed[0]["curve"]) != 2 or parsed[0]["curve"]["delivery_month"].iloc[1] != date(2027, 1, 1):
            raise AssertionError("Monthly file parsed incorrectly.")
        quotes = pd.DataFrame({"curve_date": ["2026-11-03"] * 2 + ["04/11/2026"] * 2, "product": ["TTF Natural Gas"] * 4, "contract": ["Q1-27", "Cal-28"] * 2, "price": [30.0, 28.0, 31.0, 28.5]})
        parsed = parse_curve_table(quotes, "x.csv", today, shapes, {}, today)
        if [p["day"] for p in parsed] != [date(2026, 11, 3), date(2026, 11, 4)] or len(parsed[1]["curve"]) != 15:
            raise AssertionError("A file with two dates and contracts was parsed incorrectly.")
        lookup = _curve_lookup(parsed[1]["curve"])
        spec = PRODUCTS["TTF Natural Gas"]
        assert_close(_strip_price(lookup, *contract_window("Q1-27"), "BASE", spec["timezone"]), 31.0, 1e-3)
        refused = {"unknown product": pd.DataFrame({"product": ["Brent"], "contract": ["Dec-26"], "price": [80]}),
                   "ambiguous month": pd.DataFrame({"product": ["TTF"], "delivery_month": ["12/01/2026"], "price": [30]}),
                   "no price": pd.DataFrame({"product": ["TTF", "TTF"], "contract": ["Dec-26", "Jan-27"], "price": ["31", "n/a"]}),
                   "missing column": pd.DataFrame({"product": ["TTF"], "price": [30]}),
                   "future date": pd.DataFrame({"curve_date": ["2026-12-01"], "product": ["TTF"], "contract": ["Dec-26"], "price": [30]})}
        for name, frame in refused.items():
            try:
                parse_curve_table(frame, "x.csv", today, shapes, {}, today)
            except ValueError:
                continue
            raise AssertionError(f"A file with {name} should be refused.")
        last = pd.DataFrame({"delivery_month": [date(2026, 12, 1)], "product": ["TTF Natural Gas"], "price": [30.0], "currency": ["EUR"], "unit": ["EUR/MWh"]})
        try:
            parse_curve_table(pd.DataFrame({"product": ["TTF"], "contract": ["Dec-26"], "price": [300.0]}), "x.csv", today, shapes, {"TTF Natural Gas": (date(2026, 11, 4), last)}, today)
        except ValueError:
            return "Monthly and contract files, French decimals and several dates all import. Unknown products, unclear dates, future dates and unit errors are refused."
        raise AssertionError("A price ten times the last marks should be refused.")

    def check_curve_dates_and_gaps() -> str:
        today = date(2026, 11, 10)
        custom = {"TTF Natural Gas": 1, "EUA Carbon": 1}
        meta = {"TTF Natural Gas": {"curve_date": "2026-11-09"}, "EUA Carbon": {"curve_date": "2026-11-01"}}
        if curve_state("TTF Natural Gas", custom, meta, today)["state"] != "Current" or curve_state("EUA Carbon", custom, meta, today)["state"] != "Stale" or curve_state("API2 Coal", custom, meta, today)["state"] != "Demo":
            raise AssertionError("Curve status is wrong.")
        if set(fresh_curves(custom, {**meta, "EUA Carbon": {"curve_date": "2026-10-01"}}, today)) != {"TTF Natural Gas"}:
            raise AssertionError("Old marks must not count as today's evidence.")
        month = date(2026, 12, 1)
        rows = pd.DataFrame({"snapshot_date": ["2026-11-03", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-05"], "product": ["TTF Natural Gas", "EUA Carbon", "EUA Carbon", "TTF Natural Gas", "EUA Carbon"],
                             "delivery_month": [month.isoformat()] * 5, "price": [30.0, 70.0, 71.0, 33.0, 72.0], "source": "x"})
        lookups = snapshot_lookup(rows)
        trade = normalise_trades(pd.DataFrame([{"trade_id": "G", "trade_date": "2026-11-01", "product": "TTF Natural Gas", "instrument": "SWAP", "settlement_type": "FINANCIAL", "side": "BUY_FIXED",
                                                "delivery_start": "2026-12-01", "delivery_end": "2027-01-01", "profile": "BASE", "quantity_mw": 1.0, "trade_price": 30.0, "currency": "EUR", "status": "EXECUTED", "counterparty": "X"}]))
        history = pnl_history(book_buckets(trade), lookups, 1.17)
        assert_close(float(history["pnl_eur"].iloc[0]), 0.0)
        assert_close(float(history["pnl_eur"].iloc[1]), 3.0 * 744.0)
        return "Curve status is right, and a product missing for a day keeps its last marks: 0 P&L that day, then the full 3.00 move on 744 MWh."

    def check_track_scoring() -> str:
        low, high = wilson_interval(50, 100)
        assert_close(low, 0.4190, 1e-3)
        assert_close(high, 0.5810, 1e-3)
        monday = date(2026, 11, 2)
        if independent_call_count([monday + timedelta(days=i) for i in range(5)], 5) != 1 or independent_call_count([monday, monday + timedelta(days=7)], 5) != 2:
            raise AssertionError("Overlapping outcome windows must count once.")
        if working_days_after(date(2026, 11, 6), 1) != date(2026, 11, 9) or working_days_after(date(2026, 11, 7), 1) != date(2026, 11, 9):
            raise AssertionError("One working day after a Friday or Saturday is Monday.")
        # Weekends always trade 20 lower. That pattern must not count as a price move.
        days = pd.date_range("2026-01-05", periods=56, freq="D")
        hourly = pd.DataFrame({"local": days, "value": [30.0 if d.weekday() >= 5 else 50.0 for d in days], "hours": 24.0})
        path = spot_path(hourly)
        move = path_move(path, date(2026, 2, 6), 1)
        assert_close(move, 0.0)
        # A path that rises every day: bullish is always right, bearish never.
        days = pd.bdate_range("2026-01-05", periods=200)
        table_ = {d.date(): {date(2028, 1, 1): 30.0 + i} for i, d in enumerate(days)}
        rising = {"kind": "marks", "table": table_, "days": sorted(table_)}
        always = pd.DataFrame({"anchor": [d.date() for d in days[:150]], "direction": 1})
        result = score_calls(always, rising, 1)
        assert_close(result["right"], 1.0)
        assert_close(result["chance"], 1.0)
        assert_close(result["average_move"], 1.0)
        if result["verdict"] != "No clear edge":
            raise AssertionError(f"Always bullish in a rising market isn't skill: {result['verdict']}")
        few = score_calls(always.head(10), rising, 1)
        if few["verdict"] != "Too few to judge":
            raise AssertionError("Ten calls can't produce a verdict.")
        alternating = {d.date(): {date(2028, 1, 1): 30.0 + (i % 2)} for i, d in enumerate(days)}
        swing = {"kind": "marks", "table": alternating, "days": sorted(alternating)}
        good = pd.DataFrame({"anchor": [d.date() for d in days[:150]], "direction": [1 if i % 2 == 0 else -1 for i in range(150)]})
        verdict = score_calls(good, swing, 1)
        assert_close(verdict["right"], 1.0)
        if verdict["verdict"] != "Better than chance" or abs(verdict["chance"] - 0.5) > 0.01:
            raise AssertionError(f"Perfect timing should beat 50%: {verdict}")
        demo = pd.DataFrame({"snapshot_date": ["2026-01-05"], "product": ["TTF Natural Gas"], "delivery_month": ["2027-01-01"], "price": [30.0], "source": ["Demo marks"]})
        if marks_path(demo, "TTF Natural Gas")["days"]:
            raise AssertionError("Demo prices must never score a read.")
        return "Wilson range, overlap count, weekday removal and verdicts match hand results. Always-bullish in a rising market shows no edge, and demo prices are never used."

    def check_track_no_lookahead() -> str:
        zone = "France"
        tz = ENTSOE_ZONES[zone]["timezone"]
        stamps = pd.date_range("2026-06-01", periods=24 * 60, freq="h", tz="UTC")
        rng = np.random.default_rng(5)
        prices = pd.DataFrame({"timestamp": stamps, "value": 50.0 + np.cumsum(rng.normal(size=len(stamps))) * 0.2, "resolution": "PT60M"})
        changed = prices.copy()
        cutoff = pd.Timestamp("2026-07-13", tz=tz).tz_convert("UTC")
        changed.loc[changed["timestamp"] >= cutoff, "value"] = 500.0
        one = replay_power_reads("French power", zone, prices, None, None, set(), date(2026, 12, 1))
        two = replay_power_reads("French power", zone, changed, None, None, set(), date(2026, 12, 1))
        day = date(2026, 7, 10)
        a, b = [next(r for r in rows if r["read_date"] == day.isoformat()) for rows in (one, two)]
        if a["score"] != b["score"] or a["signals"] != b["signals"]:
            raise AssertionError("Changing later prices changed an earlier read.")
        if not any(r["score"] != s["score"] for r, s in zip(one, two) if r["read_date"] >= "2026-07-13"):
            raise AssertionError("Test problem: the change had no effect at all.")
        path, moved = spot_path(spot_series(prices, tz)), spot_path(spot_series(changed, tz))
        if abs(path["raw"][day] - moved["raw"][day]) > 1e-9:
            raise AssertionError("The starting price of a call used later data.")
        return "Replacing every price after a date leaves all earlier reads unchanged."

    def _winter_fixture(hdd_effect: float = -0.04, years: tuple = (2016, 2026)) -> tuple:
        """Synthetic storage driven by a known rule, so the fit can be checked against the truth."""
        rng = np.random.default_rng(11)
        days = pd.date_range(date(years[0], 1, 1), date(years[1], 11, 3), freq="D")
        winter_cold = {y: rng.normal(0, 1.5) for y in range(years[0] - 1, years[1] + 1)}
        seasonal = 9.0 + 8.0 * np.cos(2 * np.pi * (days.dayofyear.to_numpy() - 15) / 365.25)
        hdd = np.clip(seasonal + np.array([winter_cold[winter_of(d)] for d in days.date]) + rng.normal(0, 1.5, len(days)), 0, None)
        fill = np.empty(len(days))
        level = 88.0
        for i, day in enumerate(days):
            if day.month in WINTER_MONTHS:
                step = 0.35 + hdd_effect * hdd[i] - 0.12 * (day.dayofweek >= 5) - 0.002 * (level - 60.0) + rng.normal(0, 0.03)
            else:
                step = 0.02 * (90.0 - level)
            level = float(np.clip(level + step, 2.0, 100.0))
            fill[i] = level
        return pd.Series(fill, index=days), pd.Series(hdd, index=days)

    def check_winter_model() -> str:
        fill, hdd = _winter_fixture()
        model = fit_storage_model(fill, hdd)
        assert_close(model["hdd_effect"], -0.04, 0.003)
        assert_close(model["weekend_effect"], -0.12, 0.01)
        assert_close(model["fill_effect"], -0.002, 0.0015)
        wrong_way = pd.Series(np.cumsum(hdd.reindex(fill.index).fillna(0).to_numpy() * -0.01) + (fill.index.year - 2016) * 8.0 + 70.0, index=fill.index)
        if fit_storage_model(wrong_way, hdd)["fill_effect"] > 0:
            raise AssertionError("Emptier storage must never draw faster.")
        if model["r2"] < 0.8:
            raise AssertionError(f"The model should explain a clean rule: {model['r2']}")
        days = pd.date_range("2020-11-02", "2021-03-31", freq="D")
        one = simulate_fill(model, 80.0, days, hdd.reindex(days).to_numpy())
        two = simulate_fill(model, 80.0, days, hdd.reindex(days).to_numpy() + 1.0)
        if not two[-1] < one[-1]:
            raise AssertionError("Colder weather must leave less in storage.")
        return "On storage built from a known rule, the fit recovers the heating, weekend and fill effects, and colder weather leaves less gas."

    def check_winter_projection() -> str:
        fill, hdd = _winter_fixture()
        today = date(2026, 11, 5)
        history = pd.DataFrame({"gasDayStart": fill.index.strftime("%Y-%m-%d"), "full": fill.to_numpy()})
        archive = pd.DataFrame({"date": hdd.index.strftime("%Y-%m-%d"), "location": GAS_WEATHER_LOCATIONS[0], "hdd": hdd.to_numpy()})
        outlook = project_winter(history, archive, pd.DataFrame(), today)
        if not outlook["ready"] or outlook["basis"] != "weather":
            raise AssertionError(f"Projection not ready: {outlook['reason']}")
        cold, normal, mild = [s["end"] for s in outlook["scenarios"]]
        if not cold < normal < mild:
            raise AssertionError(f"Expected cold < normal < mild, got {cold}, {normal}, {mild}.")
        if outlook["days"][-1] != pd.Timestamp(2027, 3, 31) or outlook["start"] != date(2026, 11, 3):
            raise AssertionError("The projection must run from the latest gas day to 31 March.")
        if outlook["error"] is None or outlook["error"] > 3.0 or outlook["error"] >= outlook["average_error"]:
            raise AssertionError(f"On a clean rule the method should beat an average winter: {outlook['error']} against {outlook['average_error']}.")
        analog = project_winter(history, pd.DataFrame(), pd.DataFrame(), today)
        if analog["basis"] != "past winters" or len(analog["scenarios"]) != 3:
            raise AssertionError("Without weather, the projection should fall back to past winters.")
        summer = project_winter(history, archive, pd.DataFrame(), date(2026, 7, 1))
        if summer["ready"] or not summer["off_season"] or outlook_signal(summer) is not None:
            raise AssertionError("Outside the season there should be no projection and no signal.")
        signal = outlook_signal(outlook)
        band = outlook["band_end"]
        assert_close(signal["score"], float(np.clip(-(normal - band["average"]) / max((band["high"] - band["low"]) / 2.0, 5.0) / 2.0, -1.0, 1.0)))
        if not outlook_signal({"ready": False, "off_season": False, "reason": "x"})["missing"]:
            raise AssertionError("Missing data should show as a signal waiting for data.")
        return f"Cold {cold:.0f}% is below normal {normal:.0f}%, below mild {mild:.0f}%. The past-winter test beats assuming an average winter, and the projection falls back or stops when it should."

    # ---- Next-day estimate, drivers, crisis replay, VaR check, thresholds and PDF ----
    def check_next_day_estimate() -> str:
        rng = np.random.default_rng(21)
        days = pd.date_range("2025-10-01", periods=300, freq="D")
        wind = rng.uniform(5, 40, len(days))
        hdd = np.clip(10 + 8 * np.cos(2 * np.pi * days.dayofyear.to_numpy() / 365.25) + rng.normal(0, 2, len(days)), 0, None)
        solar = rng.uniform(20, 250, len(days))
        price = np.empty(len(days))
        price[0] = 70.0
        for i in range(1, len(days)):
            price[i] = 25 + 0.6 * price[i - 1] - 0.5 * wind[i] + 0.8 * hdd[i] - 0.02 * solar[i] - 12.0 * (days[i].dayofweek >= 5) + rng.normal(0, 1.0)
        hourly = pd.DataFrame({"local": days.repeat(24) + pd.to_timedelta(np.tile(np.arange(24), len(days)), unit="h"), "value": price.repeat(24), "hours": 1.0})
        location = ZONE_WEATHER["France"]
        weather = pd.DataFrame({"date": days, "location": location, "temperature_mean_c": 18.0 - hdd, "wind_speed_100m_mean_kmh": wind, "solar_radiation_mean_wm2": solar})
        forecast = weather.tail(1).assign(date=days[-1] + pd.Timedelta(days=1), wind_speed_100m_mean_kmh=20.0, temperature_mean_c=10.0, solar_radiation_mean_wm2=100.0)
        result = next_day_estimate("France", hourly, weather, forecast)
        if not result["ready"]:
            raise AssertionError(result["reason"])
        assert_close(result["wind_effect"], -0.5, 0.06)
        assert_close(result["hdd_effect"], 0.8, 0.15)
        if not result["error"] < min(result["error_last_week"], result["error_yesterday"]) or not result["low"] < result["estimate"] < result["high"]:
            raise AssertionError("On a clean rule the estimate should beat simple methods and sit inside its range.")
        if result["day"] != (days[-1] + pd.Timedelta(days=1)).date():
            raise AssertionError("The estimate must be for the first unpublished day.")
        changed = hourly.copy()
        changed.loc[changed["local"] >= days[-1], "value"] = 900.0
        again = next_day_estimate("France", changed, weather, forecast)
        if abs(float(again["test"].iloc[-2]["estimate"]) - float(result["test"].iloc[-2]["estimate"])) > 1e-9:
            raise AssertionError("A later price changed an earlier estimate.")
        short = next_day_estimate("France", hourly.head(24 * 30), weather, forecast)
        no_forecast = next_day_estimate("France", hourly, weather, pd.DataFrame())
        if short["ready"] or no_forecast["ready"]:
            raise AssertionError("Without enough prices or a forecast there should be no estimate.")
        log = pd.DataFrame({"delivery_date": [days[-1].date().isoformat(), "2030-01-01"], "estimate": [price[-1] + 2.0, 1.0]})
        score = score_recorded_estimates(log, pd.Series(price, index=days))
        if score["n"] != 1:
            raise AssertionError("Only estimates with a published price can be scored.")
        assert_close(score["error"], 2.0)
        return "On prices built from a known rule, the fit recovers the wind and heating effects and beats simple methods. Later prices never change earlier estimates."

    def check_pnl_drivers() -> str:
        month, other = date(2026, 12, 1), date(2027, 1, 1)
        buckets = pd.DataFrame([{"trade_id": "A", "product": "TTF Natural Gas", "currency": "EUR", "delivery_month": month, "net_mwh": 1000.0, "trade_price": 30.0, "trade_date": date(2026, 11, 1)},
                                {"trade_id": "B", "product": "TTF Natural Gas", "currency": "EUR", "delivery_month": month, "net_mwh": -200.0, "trade_price": 31.0, "trade_date": date(2026, 11, 1)},
                                {"trade_id": "C", "product": "NBP Natural Gas", "currency": "GBP", "delivery_month": other, "net_mwh": 500.0, "trade_price": 25.0, "trade_date": date(2026, 11, 5)}])
        previous = {"TTF Natural Gas": {month: 32.0}, "NBP Natural Gas": {other: 26.0}}
        current = {"TTF Natural Gas": {month: 30.0}, "NBP Natural Gas": {other: 27.0}}
        step = pnl_explain(buckets, previous, current, date(2026, 11, 4), date(2026, 11, 5))
        drivers = pnl_drivers(step, step["currency"].map({"EUR": 1.0, "GBP": 1.2}))
        ttf, nbp = drivers.loc[drivers["product"].eq("TTF Natural Gas")].iloc[0], drivers.loc[drivers["product"].eq("NBP Natural Gas")].iloc[0]
        assert_close(ttf["price_move"], 800.0 * -2.0)
        assert_close(nbp["new_trades"], 500.0 * 2.0 * 1.2)
        assert_close(float(drivers["total"].sum()), -1600.0 + 1200.0)
        assert_close(float((drivers["price_move"] + drivers["new_trades"] + drivers["first_marks"] - drivers["total"]).abs().max()), 0.0)
        sentence = pnl_sentence(drivers)
        if "TTF Natural Gas Dec-26" not in sentence or "long 800 MWh" not in sentence or "-2.00" not in sentence:
            raise AssertionError(sentence)
        return "Parts add up to the total, GBP converts, and the sentence names long 800 MWh of TTF Dec-26 moving -2.00, worth -€1,600."

    def check_crisis_replay() -> str:
        month = date(2026, 12, 1)
        lookup = lambda product, price: pd.DataFrame({"delivery_month": [month], "product": [product], "price": [price], "currency": [PRODUCTS.get(product, REFERENCE_PRODUCTS.get(product))["currency"]], "unit": [PRODUCTS.get(product, REFERENCE_PRODUCTS.get(product))["unit"]]})
        curve_set = {"TTF Natural Gas": lookup("TTF Natural Gas", 30.0), "EUA Carbon": lookup("EUA Carbon", 80.0), "NBP Natural Gas": lookup("NBP Natural Gas", 25.0)}
        book = pd.DataFrame([{"product": "TTF Natural Gas", "currency": "EUR", "delivery_month": month, "net_mwh": 100.0}, {"product": "French Power Base", "currency": "EUR", "delivery_month": month, "net_mwh": 10.0}, {"product": "NBP Natural Gas", "currency": "GBP", "delivery_month": month, "net_mwh": 10.0}])
        first = crisis_replay(book, curve_set, month, True, 1.0, 1.2).query("episode == @CRISIS_EPISODES[0]['name']").set_index("product")
        assert_close(first.loc["TTF Natural Gas", "price_change"], 30.0 * 2.45)
        carbon = 80.0 * (68.0 / 98.5 - 1.0)
        assert_close(first.loc["French Power Base", "price_change"], (73.5 + carbon * 0.202) / 0.5)
        assert_close(first.loc["NBP Natural Gas", "pnl_eur"], 10.0 * 25.0 * 2.45 * 1.2)
        far_book = book.head(1).assign(delivery_month=date(2028, 12, 1))
        far = crisis_replay(far_book, curve_set, month, True, 0.0, 1.2)["price_change"].iloc[0]
        assert_close(far, 73.5 * math.exp(-24.0 / 6.0))
        absolute = crisis_replay(book.head(1), curve_set, month, False, 1.0, 1.2).set_index("episode")["price_change"]
        assert_close(absolute[CRISIS_EPISODES[1]["name"]], 113.0)
        assert_close(absolute[CRISIS_EPISODES[2]["name"]], 1.0 - 30.0)
        stretch = worst_stretch(pd.DataFrame({"date": pd.date_range("2026-01-01", periods=6).date, "pnl_eur": [1, -5, -6, 2, -1, 3]}), 2)
        assert_close(stretch["pnl_eur"], -11.0)
        if stretch["end"] != date(2026, 1, 3):
            raise AssertionError("Worst stretch dates are wrong.")
        return "Gas moves, carbon, the gas-plant link to power, GBP, fading along the curve and the gas price floor all match hand calculations."

    def check_var_backtest() -> str:
        statistic, p_value = kupiec_test(250, 5, 0.99)
        expected = -2.0 * (5 * math.log(0.01) + 245 * math.log(0.99) - 5 * math.log(0.02) - 245 * math.log(0.98))
        assert_close(statistic, expected)
        assert_close(p_value, math.erfc(math.sqrt(expected / 2.0)))
        zero, zero_p = kupiec_test(250, 0, 0.99)
        assert_close(zero, -2.0 * 250 * math.log(0.99))
        if not 0.05 < p_value < 0.2 or zero_p > 0.05 + 0.1:
            raise AssertionError(f"Kupiec p-values look wrong: {p_value}, {zero_p}.")
        rng = np.random.default_rng(4)
        days = pd.bdate_range("2023-01-02", periods=800).date
        steady = pd.DataFrame({"date": days, "pnl_eur": rng.normal(0, 1000, len(days))})
        check = var_backtest(steady, 0.99)
        if check["verdict"] != "Consistent with the confidence level" or check["zone"] == "":
            raise AssertionError(f"Stable markets should pass: {check['verdict']}, {check['exceptions']} of {check['tests']}.")
        shifted = steady["pnl_eur"].to_numpy().copy()
        shifted[500:] *= 4.0
        broken = var_backtest(steady.assign(pnl_eur=shifted), 0.99)
        if broken["verdict"] != "VaR looks too low":
            raise AssertionError(f"A jump in volatility should fail: {broken['verdict']}.")
        later = steady.copy()
        later.loc[700:, "pnl_eur"] = -1e9
        a, b = var_backtest(steady, 0.99)["table"], var_backtest(later, 0.99)["table"]
        assert_close(float(a["var_eur"].iloc[300]), float(b["var_eur"].iloc[300]))
        if var_backtest(steady.head(50))["tests"] != 0 or var_backtest(steady.head(100))["verdict"] != "Too few days to judge":
            raise AssertionError("Short samples should give no verdict.")
        return f"Kupiec matches the formula, stable markets pass ({check['exceptions']} of {check['tests']}), a jump in volatility fails, and later losses never change earlier VaR."

    def check_alerts_and_pdf() -> str:
        rules = pd.DataFrame({"metric": ["A", "A", "B", "Missing"], "condition": ["Above", "Below", "Below", "Above"], "level": [10.0, 10.0, 5.0, 1.0]})
        fired = threshold_alerts(rules, {"A": {"now": 12.0, "before": 9.0, "unit": "%"}, "B": {"now": 4.0, "before": 4.5, "unit": "EUR"}})
        if [(a["metric"], a["condition"], a["new"]) for a in fired] != [("A", "Above", True), ("B", "Below", False)]:
            raise AssertionError(f"Wrong thresholds fired: {fired}")
        blocks = [("title", "Morning brief (test)"), ("subtitle", "Ünïcode, € and back\\slash"), ("heading", "Section"), ("text", "word " * 400),
                  ("table", (["A", "B"], [["one", "two"]] * 90, (0.5, 0.5))), ("note", "End")]
        pdf = simple_pdf(blocks, "Test")
        if not pdf.startswith(b"%PDF-1.4") or not pdf.rstrip().endswith(b"%%EOF"):
            raise AssertionError("The file isn't a complete PDF.")
        start = int(pdf[pdf.rindex(b"startxref") + 10:].split()[0])
        if pdf[start:start + 4] != b"xref":
            raise AssertionError("The PDF's file table is in the wrong place.")
        count = int(re.search(rb"/Count (\d+)", pdf)[1])
        if count < 3 or b"\x80" not in pdf or b"\\(test\\)" not in pdf:
            raise AssertionError("Paging, the euro sign or brackets went wrong.")
        for offset, number in zip(re.findall(rb"(\d{10}) 00000 n", pdf)[:5], range(1, 6)):
            if not pdf[int(offset):].startswith(b"%d 0 obj" % number):
                raise AssertionError("A PDF object is at the wrong position.")
        return f"Thresholds fire only when their condition holds and mark what's new. A long PDF has {count} pages with a valid file table, the euro sign and escaped brackets."

    def check_signal_themes() -> str:
        storage = [_signal("storage_level", "L", 3.0, 2.0), _signal("storage_pace", "P", 2.0, 2.0), _signal("winter_outlook", "W", 2.0, 2.0), _signal("lng", "G", 1.5)]
        alone = summarise_read(storage)
        assert_close(sum(counted_weights(storage)), 3.0)
        assert_close(counted_weights(storage)[0], 3.0 * 3.0 / 7.0)
        assert_close(alone["coverage"], 3.0 / 4.5)
        if alone["confidence"] == "High" or alone["themes"] != 1:
            raise AssertionError(f"Storage alone must not reach high confidence: {alone['confidence']}, {alone['themes']} themes.")
        both = summarise_read(storage + [_signal("weather", "H", 2.5, 2.0)])
        if both["confidence"] != "High" or both["themes"] != 2:
            raise AssertionError(f"Two independent themes that agree should give high confidence: {both['confidence']}.")
        split = summarise_read([_signal("storage_level", "L", 3.0, 2.0), _signal("storage_pace", "P", 2.0, 2.0), _signal("weather", "H", 3.0, -2.0)])
        assert_close(split["score"], 0.0)
        return "Three storage signals count as 3.0 of weight, not 7.0, and can't reach high confidence alone. A second theme that agrees can, and an opposing one of equal weight cancels them."

    def check_nuclear_signal() -> str:
        days = pd.date_range("2026-09-01", periods=40, freq="D", tz="UTC")
        def frame(nuclear: float, recent: float, other: float = 10000.0) -> pd.DataFrame:
            rows = []
            for i, day in enumerate(days):
                rows.append({"timestamp": day + pd.Timedelta(hours=12), "series": "Nuclear", "value": recent if i >= 33 else nuclear})
                rows.append({"timestamp": day + pd.Timedelta(hours=12), "series": "Fossil Gas", "value": other})
                rows.append({"timestamp": day + pd.Timedelta(hours=12), "series": "Nuclear (consumption)", "value": 99999.0})
            return pd.DataFrame(rows)
        today = date(2026, 10, 11)
        fall = _nuclear_signal(frame(40000.0, 36000.0), "Europe/Paris", today)
        if not fall["available"] or fall["score"] <= 0 or "36.0 GW" not in fall["value"] or "-10.0%" not in fall["value"]:
            raise AssertionError(f"Falling output should be bullish: {fall}")
        assert_close(fall["score"], 1.0 * min(4000.0 / max(0.0, 40000.0 * 0.03, 1.0) / 2.0, 1.0))
        rise = _nuclear_signal(frame(40000.0, 44000.0), "Europe/Paris", today)
        if rise["score"] >= 0:
            raise AssertionError("Rising output should be bearish.")
        if _nuclear_signal(frame(100.0, 100.0), "Europe/Berlin", today) is not None:
            raise AssertionError("A zone with little nuclear should have no nuclear signal.")
        if _nuclear_signal(frame(40000.0, 36000.0), "Europe/Paris", date(2027, 1, 1))["available"]:
            raise AssertionError("Old output must not count as current evidence.")
        return "Falling nuclear output is bullish and rising is bearish. Consumption rows are ignored, zones with little nuclear get no signal, and old data doesn't count."

    def check_entsoe_forecast_method() -> str:
        calls = []
        def fake(url, params=None, **kwargs):
            calls.append(dict(params))
            return b'<Doc><TimeSeries><Period><timeInterval><start>2026-01-01T00:00Z</start><end>2026-01-01T01:00Z</end></timeInterval><resolution>PT60M</resolution><Point><position>1</position><quantity>5</quantity></Point></Period></TimeSeries></Doc>', "text/xml"
        for kind, document, process, area in (("Day-ahead load forecast", "A65", "A01", "outBiddingZone_Domain"), ("Day-ahead wind and solar forecast", "A69", "A01", "in_Domain")):
            calls.clear()
            fetch_entsoe_series("t", kind, "France", date(2026, 1, 1), date(2026, 1, 2), _http_fn=fake)
            sent = calls[0]
            if sent["documentType"] != document or sent["processType"] != process or area not in sent:
                raise AssertionError(f"Wrong request for {kind}: {sent}")
        rng = np.random.default_rng(31)
        days = pd.date_range("2025-08-01", periods=320, freq="D")
        wind_gw, load_gw, solar_gw = rng.uniform(1, 15, len(days)), rng.uniform(40, 70, len(days)), rng.uniform(0, 6, len(days))
        price = np.empty(len(days))
        price[0] = 70.0
        for i in range(1, len(days)):
            price[i] = 0.3 * price[i - 1] + 1.5 * load_gw[i] - 2.5 * wind_gw[i] - 1.0 * solar_gw[i] - 10.0 * (days[i].dayofweek >= 5) + rng.normal(0, 1.0)
        hourly = pd.DataFrame({"local": days.repeat(24) + pd.to_timedelta(np.tile(np.arange(24), len(days)), unit="h"), "value": price.repeat(24), "hours": 1.0})
        location = ZONE_WEATHER["France"]
        weather = pd.DataFrame({"date": days, "location": location, "temperature_mean_c": rng.normal(10, 5, len(days)), "wind_speed_100m_mean_kmh": rng.uniform(5, 40, len(days)), "solar_radiation_mean_wm2": rng.uniform(20, 250, len(days))})
        ahead = weather.tail(1).assign(date=days[-1] + pd.Timedelta(days=1))
        stamps = (days.repeat(24) + pd.to_timedelta(np.tile(np.arange(24), len(days)), unit="h")).tz_localize("Europe/Paris", ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
        load_fc = pd.DataFrame({"timestamp": stamps, "value": (load_gw * 1000).repeat(24), "series": "Load"})
        renew = pd.concat([pd.DataFrame({"timestamp": stamps, "value": (wind_gw * 500).repeat(24), "series": name}) for name in ("Wind Onshore", "Wind Offshore")] +
                          [pd.DataFrame({"timestamp": stamps, "value": (solar_gw * 1000).repeat(24), "series": "Solar"})])
        late = next_day_estimate("France", hourly, weather, ahead, load_fc, renew)
        if late["method"] != "Weather" or late["target_has_forecast"] or late["check_method"] != "ENTSO-E forecasts":
            raise AssertionError(f"Before forecasts exist, weather sets the estimate: {late['method']}, {late['check_method']}.")
        if not late["comparison"] or late["comparison"]["forecasts"] >= late["comparison"]["weather"]:
            raise AssertionError(f"On a rule built from forecasts, that method should win: {late['comparison']}")
        assert_close(late["forecast_effects"]["wind"], -2.5, 0.2)
        assert_close(late["forecast_effects"]["load"], 1.5, 0.2)
        next_stamps = pd.date_range(days[-1] + pd.Timedelta(days=1), periods=24, freq="h", tz="Europe/Paris").tz_convert("UTC")
        early_load = pd.concat([load_fc, pd.DataFrame({"timestamp": next_stamps, "value": 55000.0, "series": "Load"})])
        early_renew = pd.concat([renew] + [pd.DataFrame({"timestamp": next_stamps, "value": 5000.0, "series": name}) for name in ("Wind Onshore", "Solar")])
        early = next_day_estimate("France", hourly, weather, ahead, early_load, early_renew)
        if early["method"] != "ENTSO-E forecasts":
            raise AssertionError("When the forecast exists and tested better, it should set the estimate.")
        worse = next_day_estimate("France", hourly, weather, ahead, load_fc.assign(value=rng.uniform(40000, 70000, len(load_fc))), renew.assign(value=rng.uniform(0, 9000, len(renew))))
        if worse["comparison"]["forecasts"] < worse["comparison"]["weather"] and worse["method"] != "Weather":
            raise AssertionError("Method choice is inconsistent.")
        absent = next_day_estimate("France", hourly, weather, ahead)
        if absent["method"] != "Weather" or absent["comparison"] is not None:
            raise AssertionError("Without forecasts the weather method should run alone.")
        changed = hourly.copy()
        changed.loc[changed["local"] >= days[-1], "value"] = 900.0
        again = next_day_estimate("France", changed, weather, ahead, load_fc, renew)
        # The forecast test holds one fewer usable row only if the last day is dropped, so compare by date.
        one = pd.DataFrame(late["test"]).set_index("day")["estimate"]
        two = pd.DataFrame(again["test"]).set_index("day")["estimate"]
        check_day = days[-3]
        assert_close(float(one.loc[check_day]), float(two.loc[check_day]))
        return "Requests use A65/A01 and A69/A01. Weather sets the estimate before ENTSO-E's forecast exists, then the forecast takes over only if it tested better. Effects are recovered and later prices change nothing earlier."

    def check_estimate_calendar() -> str:
        if _easter(2026) != date(2026, 4, 5) or _easter(2027) != date(2027, 3, 28):
            raise AssertionError("Easter date is wrong.")
        fr, de, gb = public_holidays("France", 2026), public_holidays("Germany-Luxembourg", 2026), public_holidays("Great Britain", 2027)
        if not {date(2026, 5, 14), date(2026, 5, 25), date(2026, 7, 14), date(2026, 11, 11)} <= fr or date(2026, 4, 3) in fr:
            raise AssertionError("French holidays are wrong.")
        if not {date(2026, 4, 3), date(2026, 10, 3), date(2026, 12, 26)} <= de or date(2026, 7, 14) in de:
            raise AssertionError("German holidays are wrong.")
        if not {date(2027, 12, 27), date(2027, 12, 28), date(2027, 5, 3), date(2027, 5, 31), date(2027, 8, 30)} <= gb:
            raise AssertionError("British bank holidays and Christmas substitutes are wrong.")
        days = pd.date_range("2026-05-01", "2026-05-18")   # Fri 1 May and Thu 14 May are French holidays
        kinds = day_classes(days, "France")
        if kinds[0] != 2 or kinds[1] != 1 or kinds[2] != 2 or kinds[3] != 0 or kinds[13] != 2:
            raise AssertionError(f"Day kinds are wrong: {kinds[:14]}")
        base = pd.Series(np.arange(len(days), dtype=float) + 100.0, index=days)
        frame = estimate_features(base, pd.DataFrame(), days[-1] + pd.Timedelta(days=1), "France")
        monday = frame.loc["2026-05-11"]   # Friday 8 May is a holiday too, so the last working day is Thursday 7 May
        if monday["work"] != base["2026-05-07"] or monday["same"] != base["2026-05-07"] or monday["lag1"] != base["2026-05-10"] or monday["after_off"] != 1.0:
            raise AssertionError("A working day after a weekend must start from the last working day.")
        sunday = frame.loc["2026-05-17"]
        if sunday["same"] != base["2026-05-14"]:
            raise AssertionError("A Sunday must start from the last Sunday or holiday.")
        return "Easter, French, German and British holidays match the calendar. A Monday after a long weekend starts from the last working day, and a Sunday from the last holiday."

    def check_zone_weather_fetch() -> str:
        calls = []
        def answer(day_values, hours=24):
            times = [f"{d}T{h:02d}:00" for d in day_values for h in range(hours)]
            return times
        def fake(url, params=None):
            calls.append((url, dict(params)))
            n = len(params["latitude"].split(","))
            out = []
            for i in range(n):
                times = answer(["2026-01-01", "2026-01-02"]) + answer(["2026-01-03"], 10)   # the last day is only partly published
                out.append({"hourly": {"time": times, "temperature_2m": [8.0 + i] * len(times), "wind_speed_100m": [10.0 * (i + 1)] * len(times), "shortwave_radiation": [50.0] * len(times)}})
            return out
        frame = fetch_zone_weather("Germany-Luxembourg", date(2026, 1, 1), date(2026, 1, 3), _json_get_fn=fake)
        url, params = calls[-1]
        points = len(ZONE_WEATHER_POINTS["Germany-Luxembourg"])
        if url != OPEN_METEO_ARCHIVE_BASE or len(params["latitude"].split(",")) != points or len(calls) != 1:
            raise AssertionError("All points must go in one archive request.")
        if list(frame["date"].dt.date) != [date(2026, 1, 1), date(2026, 1, 2)]:
            raise AssertionError("A partly published day must be left out.")
        assert_close(float(frame["wind"].iloc[0]), 10.0 * (points + 1) / 2)
        assert_close(float(frame["hdd"].iloc[0]), float(np.mean([max(18.0 - (8.0 + i), 0.0) for i in range(points)])))
        try:
            fetch_zone_weather("France", forecast=True, _json_get_fn=lambda u, params=None: fake(u, params)[0])
            raise AssertionError("One point can't stand for a whole zone.")
        except RuntimeError as exc:
            if "too few" not in str(exc):
                raise
        url, params = calls[-1]
        if url != OPEN_METEO_FORECAST_BASE or params.get("past_days") != 7 or params.get("forecast_days") != 3:
            raise AssertionError("The forecast request is wrong.")
        return f"{points} German points go in one request, partial days are dropped, wind and heating demand are averaged across points, and an answer for too few points is refused."

    def check_estimate_range() -> str:
        rng = np.random.default_rng(5)
        steady = rng.normal(0, 10, 400)
        out = estimate_range(steady)
        if out["held"] is None or not 0.72 <= out["held"] <= 0.88:
            raise AssertionError(f"On steady misses the range should hold about 80% of days: {out['held']}")
        assert_close(out["high"] - out["low"], 2 * 1.2816 * 10, 6.0)
        jumpy = np.r_[rng.normal(0, 5, 200), rng.normal(0, 25, 40)]
        wide = estimate_range(jumpy)
        if not wide["high"] - wide["low"] > 2.5 * (estimate_range(jumpy[:200])["high"] - estimate_range(jumpy[:200])["low"]):
            raise AssertionError("After misses grow, the range must widen.")
        few = estimate_range(rng.normal(0, 10, 20))
        if few["held"] is not None or few["adaptive"]:
            raise AssertionError("With few tested days the range can't claim a hit rate.")
        return f"On steady misses the range held {out['held'] * 100:.0f}% of days for an 80% aim, it widens when misses grow, and with few tested days it claims no hit rate."

    def check_estimate_uses_zone_points() -> str:
        rng = np.random.default_rng(8)
        days = pd.date_range("2025-09-01", periods=330, freq="D")
        zone_wind, city_wind = rng.uniform(5, 40, len(days)), rng.uniform(5, 40, len(days))
        price = 120 - 2.0 * zone_wind - 15.0 * (days.dayofweek >= 5) + rng.normal(0, 2, len(days))
        hourly = pd.DataFrame({"local": days.repeat(24) + pd.to_timedelta(np.tile(np.arange(24), len(days)), unit="h"), "value": price.repeat(24), "hours": 1.0})
        city = pd.DataFrame({"date": days, "location": ZONE_WEATHER["France"], "temperature_mean_c": 12.0, "wind_speed_100m_mean_kmh": city_wind, "solar_radiation_mean_wm2": 100.0})
        grid = pd.DataFrame({"zone": "France", "date": days, "hdd": 6.0, "wind": zone_wind, "solar": 100.0})
        ahead = grid.tail(1).assign(date=days[-1] + pd.Timedelta(days=1), wind=30.0)
        with_grid = next_day_estimate("France", hourly, city, city.tail(1).assign(date=days[-1] + pd.Timedelta(days=1)), grid_history=grid, grid_forecast=ahead)
        city_only = next_day_estimate("France", hourly, city, city.tail(1).assign(date=days[-1] + pd.Timedelta(days=1)))
        if "points" not in with_grid["weather_basis"] or city_only["weather_basis"] != "Paris":
            raise AssertionError(f"Wrong weather source: {with_grid['weather_basis']}, {city_only['weather_basis']}")
        if not with_grid["error"] < 0.5 * city_only["error"]:
            raise AssertionError("When prices follow zone-wide wind, the zone points should test far better than one city.")
        assert_close(with_grid["wind_effect"], -2.0, 0.25)
        no_ahead = next_day_estimate("France", hourly, city, city.tail(1).assign(date=days[-1] + pd.Timedelta(days=1)), grid_history=grid)
        if no_ahead["weather_basis"] != "Paris":
            raise AssertionError("Without a zone forecast for the day, the city forecast should be used.")
        return f"Zone points are used when they have a forecast for the day (miss €{with_grid['error']:.1f} against €{city_only['error']:.1f} for one city), otherwise the city."

    def check_tradingview_curve() -> str:
        today = date(2026, 11, 9)
        screen = {date(2026, 12, 1): 87.010, date(2027, 1, 1): 85.384, date(2027, 2, 1): 84.784, date(2027, 3, 1): 81.755, date(2027, 4, 1): 70.556}
        calls = []
        class FakeClient:
            def get_hist(self, symbol, exchange, interval, n_bars):
                calls.append((symbol, exchange, n_bars))
                code, year = symbol[3], int(symbol[4:])
                month = date(year, TRADINGVIEW_MONTH_CODES.index(code) + 1, 1)
                if month > date(2027, 4, 1):
                    return None   # not listed
                days = pd.bdate_range(end=pd.Timestamp(today), periods=n_bars)
                last = screen.get(month, 90.0)
                closes = last + 0.1 * np.arange(len(days))[::-1]
                frame = pd.DataFrame({"symbol": symbol, "close": closes}, index=days + pd.Timedelta(hours=1))
                if month == date(2027, 3, 1):
                    frame = frame.drop(frame.index[-3])   # no trade one day
                if month == date(2026, 12, 1):
                    frame = frame.drop(frame.index[-6])   # one more gap
                return frame
        raw = fetch_tradingview_curve("TTF Natural Gas", today, 7, _client=FakeClient())
        if calls[0][:2] != ("TTFZ2026", "NYMEX") or calls[1][0] != "TTFF2027":
            raise AssertionError(f"Wrong symbols: {calls[:2]}")
        if len(calls) != 8:
            raise AssertionError(f"It should stop after three months that aren't listed: {len(calls)} requests.")
        latest = raw.loc[raw["curve_date"].eq(today)].set_index("contract")["price"]
        for month, price in screen.items():
            assert_close(float(latest[month]), price)
        parsed = parse_curve_table(raw, "tradingview", today, {}, {}, today)
        if len(parsed) != 7 or parsed[-1]["day"] != today or len(parsed[-1]["curve"]) != 5:
            raise AssertionError("Each day should become one monthly curve.")
        gap_day = pd.bdate_range(end=pd.Timestamp(today), periods=7)[-3].date()
        carried = raw.loc[raw["curve_date"].eq(gap_day)].set_index("contract")["price"]
        assert_close(float(carried[date(2027, 3, 1)]), screen[date(2027, 3, 1)] + 0.3)   # previous day's close carried
        history = fetch_tradingview_curve("TTF Natural Gas", today, 60, _client=FakeClient())
        october = history.loc[history["curve_date"].eq(date(2026, 10, 20))]
        if date(2026, 11, 1) not in set(october["contract"]) or date(2026, 11, 1) in set(history.loc[history["curve_date"] >= date(2026, 11, 1), "contract"]):
            raise AssertionError("A contract counts only before its delivery month starts.")
        class Down:
            def get_hist(self, **kwargs):
                raise ConnectionError("offline")
        try:
            fetch_tradingview_curve("TTF Natural Gas", today, 7, _client=Down())
            raise AssertionError("An unreachable TradingView should be reported.")
        except RuntimeError as exc:
            if "reach" not in str(exc):
                raise
        return "Symbols are NYMEX TTFZ2026, TTFF2027 and on. Your screenshot's closes come through as today's marks, a day with no trade keeps the last close, expired months drop out, and an outage stops after three tries."

    def check_demand_weights() -> str:
        archive = pd.DataFrame({"date": ["2026-01-01"] * 3, "location": ["Berlin / German power proxy", "Paris / French power proxy", "London / NBP proxy"], "hdd": [20.0, 10.0, 99.0]})
        assert_close(float(archive_hdd(archive).iloc[0]), (78.0 * 20.0 + 31.0 * 10.0) / 109.0)
        one = pd.DataFrame({"date": ["2026-01-01"], "location": ["Paris / French power proxy"], "hdd": [10.0]})
        assert_close(float(archive_hdd(one).iloc[0]), 10.0)
        return "German gas use counts for more than French, Britain stays out of the EU storage model, and a missing city simply drops out."

    def check_two_auctions() -> str:
        series = lambda position, price: (f"<TimeSeries>{position}<curveType>A03</curveType><Period><timeInterval><start>2026-11-05T23:00Z</start><end>2026-11-06T23:00Z</end></timeInterval>"
                                          f"<resolution>PT60M</resolution><Point><position>1</position><price.amount>{price}</price.amount></Point></Period></TimeSeries>")
        label = "<classificationSequence_AttributeInstanceComponent.position>{}</classificationSequence_AttributeInstanceComponent.position>"
        for order in ((1, 2), (2, 1)):
            xml = "<Doc>" + "".join(series(label.format(n), 100.0 if n == 1 else 500.0) for n in order) + "</Doc>"
            frame = parse_entsoe_xml(xml.encode(), "price")
            if len(frame) != 24 or set(frame["value"]) != {100.0}:
                raise AssertionError(f"Only the SDAC auction should be kept: {set(frame['value'])}.")
        single = parse_entsoe_xml(("<Doc>" + series("", 70.0) + "</Doc>").encode(), "price")
        if set(single["value"]) != {70.0}:
            raise AssertionError("A zone with no auction label should be kept.")
        day = pd.date_range("2026-11-06", periods=96, freq="15min", tz="UTC")
        mixed = pd.concat([pd.DataFrame({"timestamp": day, "value": 70.0, "resolution": "PT15M"}), pd.DataFrame({"timestamp": day[::4], "value": 700.0, "resolution": "PT60M"})])
        cleaned = spot_series(mixed.sample(frac=1.0, random_state=3), "UTC")
        assert_close(float(spot_kpis(cleaned)["base"]), 70.0)
        assert_close(float(cleaned["hours"].sum()), 24.0)
        hourly_then_quarter = pd.concat([pd.DataFrame({"timestamp": pd.date_range("2026-10-01", periods=24 * 5, freq="h", tz="UTC"), "value": 50.0, "resolution": "PT60M"}),
                                         pd.DataFrame({"timestamp": pd.date_range("2026-10-06", periods=96 * 5, freq="15min", tz="UTC"), "value": 80.0, "resolution": "PT15M"})])
        if len(daily_base_prices(spot_series(hourly_then_quarter, "UTC"))) != 10:
            raise AssertionError("A change from hourly to 15-minute prices must keep both periods.")
        return "Only the official SDAC auction is kept, in any order. Unlabelled zones are kept, overlapping resolutions don't count twice, and a change of resolution loses no days."


    def check_real_marks() -> str:
        day = pd.date_range("2026-10-01", "2026-11-11", freq="15min", tz="Europe/Paris", inclusive="left")
        prices = pd.DataFrame({"local": day, "value": np.where((day.hour >= 8) & (day.hour < 20) & (day.dayofweek < 5), 120.0, 60.0), "hours": 0.25})
        today = date(2026, 11, 10)
        base = realised_months(prices, "BASE", "Europe/Paris", today)
        peak = realised_months(prices, "PEAK", "Europe/Paris", today)
        assert_close(base[date(2026, 10, 1)][1], 1.0)
        weekdays = int(np.busday_count(date(2026, 10, 1), date(2026, 11, 1)))
        assert_close(base[date(2026, 10, 1)][0], (120.0 * weekdays * 12 + 60.0 * (745 - weekdays * 12)) / 745.0, 1e-6)
        assert_close(peak[date(2026, 10, 1)][0], 120.0)
        assert_close(base[date(2026, 11, 1)][1], 240.0 / 720.0)
        demo = make_sample_curve("French Power Base", today, 12)
        curve, sources = compose_curve("French Power Base", demo, SOURCE_DEMO, base, today)
        lookup = _curve_lookup(curve)
        next_demo = _curve_lookup(validate_curve(demo))[date(2026, 12, 1)]
        share, realised = base[date(2026, 11, 1)][1], base[date(2026, 11, 1)][0]
        assert_close(lookup[date(2026, 11, 1)], share * realised + (1 - share) * next_demo)
        if sources[date(2026, 10, 1)] != SOURCE_DAY_AHEAD or sources[date(2026, 11, 1)] != SOURCE_PART_DEMO or sources[date(2026, 12, 1)] != SOURCE_DEMO:
            raise AssertionError(f"Wrong month sources: {sources}")
        if price_state(sources.values()) != "Part real" or price_state([SOURCE_DEMO]) != "Demo" or price_state([SOURCE_DAY_AHEAD, SOURCE_IMPORTED]) != "Real":
            raise AssertionError("Price state summary is wrong.")
        if trade_price_state("French Power Base", "2026-10-01", "2026-11-01", "BASE", {"French Power Base": sources}) != "Real":
            raise AssertionError("A trade that delivered in October should be marked on real prices.")
        if trade_price_state("French Power Base", "2026-10-01", "2027-01-01", "BASE", {"French Power Base": sources}) != "Part real":
            raise AssertionError("A trade spanning delivered and future months is part real.")
        imported = demo.assign(price=100.0)
        _, mixed = compose_curve("French Power Base", imported, SOURCE_IMPORTED, base, today)
        if mixed[date(2026, 11, 1)] != SOURCE_PART_IMPORTED or price_state(mixed.values()) != "Real":
            raise AssertionError("Day-ahead plus imported marks are fully real.")
        eua, eua_sources = compose_curve("EUA Carbon", make_sample_curve("EUA Carbon", today, 12), SOURCE_DEMO, {}, today, (81.5, date(2026, 11, 9)))
        if set(eua["price"]) != {81.5} or set(eua_sources.values()) != {SOURCE_AUCTION} or date(2026, 11, 1) not in eua_sources:
            raise AssertionError("The auction price should replace demo EUA marks.")
        _, kept = compose_curve("EUA Carbon", make_sample_curve("EUA Carbon", today, 12).assign(price=70.0), SOURCE_IMPORTED, {}, today, (81.5, date(2026, 11, 9)))
        if set(kept.values()) != {SOURCE_IMPORTED}:
            raise AssertionError("Imported EUA marks must not be replaced by the auction price.")
        trades = normalise_trades(pd.DataFrame([{"trade_id": "R", "trade_date": "2026-09-20", "product": "French Power Base", "instrument": "SWAP", "settlement_type": "FINANCIAL",
                                                 "side": "BUY_FIXED", "delivery_start": "2026-10-01", "delivery_end": "2026-11-01", "profile": "BASE", "quantity_mw": 1.0,
                                                 "trade_price": 80.0, "currency": "EUR", "status": "EXECUTED", "counterparty": "X"}]))
        marked = mark_trades_resilient(trades, {**{p: make_sample_curve(p, today, 12) for p in CURVE_PRODUCTS}, "French Power Base": curve})
        assert_close(float(marked["mtm"].iloc[0]), (base[date(2026, 10, 1)][0] - 80.0) * 745.0, 1e-3)
        return "October settles on its day-ahead average, November blends delivered days with demo prices, an October trade values on real prices, and the EUA auction replaces only demo marks."

    def check_eua_auction_file() -> str:
        if __import__("importlib.util").util.find_spec("openpyxl") is None:
            return "Skipped: openpyxl isn't installed, so the auction report can't be read here."
        rows = [["EEX EUA Primary Auction"], [], ["Date", "Auction Name", "Contract", "Status", "Auction Price €/tCO2", "Auction Volume tCO2"],
                ["03/11/2026", "EU, 4th period", "EUA", "Ok", "79,10", "3000000"], ["04/11/2026", "EU, 4th period", "EUAA", "Ok", "70.00", "500000"],
                ["05/11/2026", "DE, 4th period", "EUA", "Ok", "80.40", "2000000"], ["05/11/2026", "EU, 4th period", "EUA", "Ok", "80.60", "2000000"]]
        buffer = io.BytesIO()
        pd.DataFrame(rows).to_excel(buffer, header=False, index=False)
        frame = parse_eua_auctions(buffer.getvalue())
        if frame["date"].tolist() != [date(2026, 11, 3), date(2026, 11, 5)]:
            raise AssertionError(f"Wrong auction days: {frame}")
        assert_close(float(frame["price"].iloc[0]), 79.10)
        assert_close(float(frame["price"].iloc[1]), 80.50)
        try:
            buffer = io.BytesIO()
            pd.DataFrame([["nothing"]]).to_excel(buffer, header=False, index=False)
            parse_eua_auctions(buffer.getvalue())
        except ValueError:
            return "Header rows are found, aviation auctions are left out, French decimals are read and same-day auctions are averaged. An unknown layout is refused."
        raise AssertionError("An unknown layout should be refused.")
    cases = [
        ("Product registry and market timezones", "Configuration", check_product_registry),
        ("MWh ↔ MMBtu conversions", "Units & conversions", check_energy_conversion),
        ("EUR/MWh → GBP/therm conversion", "Units & conversions", check_gas_price_conversion),
        ("Spring DST delivery hours", "Delivery calendar", check_dst_spring),
        ("Autumn DST delivery hours", "Delivery calendar", check_dst_autumn),
        ("Power peak-hour delivery profile", "Delivery calendar", check_peak_hours),
        ("Current forward-curve integrity", "Market data", check_curves),
        ("Demo curve → trade → MtM integration", "Integration", check_sample_curves_and_marks),
        ("MtM sign and boundary cases", "Valuation", check_mtm_signs),
        ("Daily P&L formula", "P&L", check_daily_pnl),
        ("OTC quote margin conventions", "Quoting", check_quote_margin),
        ("Absolute parallel price-shock sensitivity", "Risk", check_parallel_shock),
        ("Trade schema and delivery energy", "Trade capture", check_sample_trades),
        ("Executed position aggregation", "Risk & exposure", check_executed_summary),
        ("Cancelled/expired trade filtering", "Trade lifecycle", check_active_trade_filter),
        ("Negative power price support", "Market data", check_negative_power_prices),
        ("Reject incompatible trade currency", "Input validation", check_bad_trade_rejected),
        ("Reject duplicate trade IDs", "Input validation", check_duplicate_ids_rejected),
        ("Reject invalid delivery dates", "Input validation", check_invalid_dates_rejected),
        ("Reject missing trade IDs", "Input validation", check_missing_trade_id_rejected),
        ("Reject zero-volume trades", "Input validation", check_zero_quantity_rejected),
        ("Reject negative quote margins", "Input validation", check_invalid_quote_margin_rejected),
        ("Reject invalid discount factors", "Input validation", check_invalid_discount_factor_rejected),
        ("Reject unknown IANA timezones", "Input validation", check_unknown_timezone_rejected),
        ("Reject duplicate curve tenors", "Input validation", check_duplicate_curve_marks_rejected),
        ("Reject missing delivery marks", "Input validation", check_incomplete_curve_rejected),
        ("Sidebar navigation registry", "Interface structure", check_navigation_registry),
        ("Current trade book end-to-end mark", "Current user data", check_current_book),
        ("FORWARD single-month contract rule", "Instrument conventions", check_forward_one_month_rule),
        ("Per-trade marking failure isolation", "Error handling", check_isolated_mark_failure),
        ("Delivery-month position split", "Risk & exposure", check_delivery_month_positions),
        ("CSV import, append and cancellation lifecycle", "End-to-end workflow", check_csv_import_and_lifecycle),
        ("Physical operations update and trade reconciliation", "Operations integration", check_physical_ops_persistence_and_reconciliation),
        ("Filtered trade-book KPI reconciliation", "Interface integration", check_filtered_book_metrics),
        ("Physical/financial settlement validation", "Input validation", check_physical_settlement_validation),
        ("Distinct FORWARD and SWAP valuation methods", "Valuation", check_forward_and_swap_valuation_methods),
        ("ENTSO-E XML and ZIP/GZIP parser including negative price", "Live data parser", check_entsoe_price_parser),
        ("Licensed vendor curve field mapping", "Live data parser", check_provider_curve_normalisation),
        ("Provider numeric format normalisation", "Live data parser", check_provider_numeric_parser),
        ("GIE EU selector, credential header and data mapping", "Live API integration", check_gie_request_and_data_mapping),
        ("ENTSOG JSON envelope normalisation", "Live API integration", check_entsog_envelope_normalisation),
        ("Open-Meteo timezone and daily aggregation", "Live API integration", check_open_meteo_aggregation_and_timezone),
        ("Open-Meteo public endpoint and market-relevant variables", "Live API integration", check_open_meteo_public_request_variables),
        ("Open-Meteo multi-location forecast and demand proxies", "Live API integration", check_open_meteo_multi_location_forecast_pack),
        ("Open-Meteo partial regional failure handling", "Live API integration", check_open_meteo_partial_location_failure),
        ("Licensed REST-provider CSV connector", "Live API integration", check_generic_provider_rest_csv),
        ("Live dashboard indicator integration", "Dashboard integration", check_live_snapshot_indicators),
        ("Cross-market hourly load/generation/price join", "Fundamentals analytics", check_residual_load_proxy_alignment),
        ("Physical flow direction", "Live API integration", check_flow_direction),
        ("Trade profile must match product", "Input validation", check_profile_must_match_product),
        ("Residual load refuses mixed zones", "Fundamentals analytics", check_residual_refuses_mixed_zones),
        ("ENTSO-E A03 gaps, consumption and error text", "Live data parser", check_parser_a03_consumption_and_errors),
        ("Long ENTSO-E requests are split", "Live API integration", check_long_fetch_is_chunked),
        ("Multi-zone data store", "Live API integration", check_multi_zone_store),
        ("Standard tenors and month ends", "Instrument conventions", check_tenors_and_month_end),
        ("Monthly marks from contract quotes", "Market data", check_curve_decomposition),
        ("Saved desk round trip", "Saved data", check_saved_desk_roundtrip),
        ("Fast marking matches trade-by-trade marking", "Valuation", check_bulk_marking_matches_and_is_fast),
        ("Daily P&L explain by hand", "P&L", check_pnl_explain_by_hand),
        ("Scenario shocks by hand", "Risk", check_scenarios_by_hand),
        ("Historical VaR by hand", "Risk", check_var_by_hand),
        ("Clean spark and dark spreads by hand", "Spreads", check_spreads_by_hand),
        ("Spot analytics by hand", "Power spot", check_spot_analytics_by_hand),
        ("Seasonal storage and degree-day normals", "Fundamentals analytics", check_storage_season_and_hdd),
        ("ECB and weather archive parsers", "Live API integration", check_ecb_and_archive_parsers),
        ("Only the official day-ahead auction is used", "Live data parser", check_two_auctions),
        ("Real day-ahead and auction prices replace demo marks", "Market data", check_real_marks),
        ("EUA auction report", "Live data parser", check_eua_auction_file),
        ("Related signals share one weight", "Market Read", check_signal_themes),
        ("Nuclear output signal", "Market Read", check_nuclear_signal),
        ("Demand weights for heating", "Market Read", check_demand_weights),
        ("ENTSO-E forecast method", "Next-day estimate", check_entsoe_forecast_method),
        ("Next-day estimate recovers known effects", "Next-day estimate", check_next_day_estimate),
        ("TradingView curve through tvdatafeed", "Live API integration", check_tradingview_curve),
        ("Holidays and kinds of day", "Next-day estimate", check_estimate_calendar),
        ("Weather across a zone in one request", "Live API integration", check_zone_weather_fetch),
        ("The 80% range holds about 80%", "Next-day estimate", check_estimate_range),
        ("Zone weather points beat one city", "Next-day estimate", check_estimate_uses_zone_points),
        ("P&L drivers add up", "Daily P&L", check_pnl_drivers),
        ("Crisis replay by hand", "Risk", check_crisis_replay),
        ("VaR check and Kupiec test", "Risk", check_var_backtest),
        ("Thresholds and PDF", "Morning brief", check_alerts_and_pdf),
        ("Keys file reading", "Saved data", check_keys_file),
        ("Curve files: formats and refusals", "Market data", check_curve_files),
        ("Old marks aren't saved as a new day", "Market data", check_curve_dates_and_gaps),
        ("Track record scoring by hand", "Track record", check_track_scoring),
        ("Track record can't see the future", "Track record", check_track_no_lookahead),
        ("Winter model recovers known effects", "Winter outlook", check_winter_model),
        ("Winter scenarios are ordered and dated", "Winter outlook", check_winter_projection),
        ("Hedge advisor: exact offset by hand", "Hedge Advisor", check_hedge_exact_offset),
        ("Hedge advisor: P&L and risk drivers", "Hedge Advisor", check_hedge_var_matches_engine),
        ("Hedge advisor: edge cases", "Hedge Advisor", check_hedge_edge_cases),
        ("Hedge advisor: proposed trades and unseen days", "Hedge Advisor", check_hedge_proposed_trades),
        ("Hedge advisor: hedges fit the book", "Hedge Advisor", check_hedge_fits_the_book),
        ("Market read: scoring and confidence", "Market Read", check_read_summary_logic),
        ("Market read: gas evidence by hand", "Market Read", check_gas_read_by_hand),
        ("Market read: power evidence by hand", "Market Read", check_power_read_by_hand),
        ("Update: merge keeps revisions and trims", "Market data updates", check_merge_observations),
        ("Update: requests only what's missing", "Market data updates", check_update_plan_is_incremental),
        ("Update: background run and failure isolation", "Market data updates", check_update_job_lifecycle),
        ("Fast P&L history matches day-by-day explain", "P&L", check_pnl_history_matches_explain),
        ("Compact saved format and earlier saves", "Saved data", check_saved_format_compatibility),
        ("Long lines thin without losing spikes", "Interface structure", check_line_thinning),
        ("ENTSO-E production types stay separate", "Live data parser", check_entsoe_production_types),
    ]
    for name, area, fn in cases:
        run_case(name, area, fn)
    return pd.DataFrame(rows, columns=["Status", "Area", "Check", "Details"])


# =============================================================================
# SECTION 8 · PAGES
# =============================================================================
def fx_controls(prefix: str) -> tuple[float, float]:
    """FX inputs shared by every page that consolidates currencies. Saved across sessions."""
    c1, c2, c3 = st.columns([1, 1, 1.2], vertical_alignment="bottom")
    gbp = c1.number_input("EUR per 1 GBP", min_value=0.5, max_value=2.5, value=round(fx_rate("eur_per_gbp"), 4), step=0.005, format="%.4f", key=f"{prefix}_gbp")
    usd = c2.number_input("EUR per 1 USD", min_value=0.3, max_value=2.0, value=round(fx_rate("eur_per_usd"), 4), step=0.005, format="%.4f", key=f"{prefix}_usd")
    if c3.button("Use latest ECB rates", key=f"{prefix}_ecb", width="stretch"):
        try:
            gbp_per_eur, day = fetch_ecb_rate("GBP")
            usd_per_eur, _ = fetch_ecb_rate("USD")
            set_fx_rate("eur_per_gbp", 1.0 / gbp_per_eur, ecb=True)
            set_fx_rate("eur_per_usd", 1.0 / usd_per_eur, ecb=True)
            set_fx_rate("fx_ecb_epoch", datetime.now(timezone.utc).timestamp(), ecb=True)
            st.session_state.fx_updated_at = datetime.now(timezone.utc)
            for suffix in ("gbp", "usd"):
                st.session_state.pop(f"{prefix}_{suffix}", None)
            st.session_state["flash_message"] = f"Using ECB reference rates from {day}."
            st.rerun()
        except Exception as exc:
            st.error(f"Could not get ECB rates: {exc}")
    # The boxes show four decimals, so only a change beyond that rounding is a rate typed in by hand.
    if abs(gbp - fx_rate("eur_per_gbp")) > 0.00005:
        set_fx_rate("eur_per_gbp", gbp)
    if abs(usd - fx_rate("eur_per_usd")) > 0.00005:
        set_fx_rate("eur_per_usd", usd)
    return fx_rate("eur_per_gbp"), fx_rate("eur_per_usd")


def contract_picker(prefix: str, product: str) -> tuple[date, date, str, str]:
    """Choose a standard contract or custom dates. Returns start, end, instrument and label."""
    choices = contract_choices(valuation_date) + [CUSTOM_CONTRACT]
    default = next((i for i, c in enumerate(choices) if c.startswith("Q")), 0)
    c1, c2 = st.columns(2)
    label = c1.selectbox("Delivery contract", choices, index=default, key=f"{prefix}_contract",
                         help="Months, quarters, seasons (Sum = April to September, Win = October to March) and calendar years.")
    if label == CUSTOM_CONTRACT:
        d1, d2 = st.columns(2)
        start = d1.date_input("First delivery day", value=month_start(1), key=f"{prefix}_start")
        end = d2.date_input("Last delivery day", value=month_start(2) - timedelta(days=1), key=f"{prefix}_end")
        end = end + timedelta(days=1)  # the engine works with an exclusive end date
        whole_month = start.day == 1 and end == _add_months(start, 1)
    else:
        start, end = contract_window(label)
        whole_month = end == _add_months(start, 1)
    if whole_month:
        instrument = c2.selectbox("Structure", ["FORWARD", "SWAP"], key=f"{prefix}_instrument",
                                  format_func=lambda x: x.title(), help="A forward covers one calendar month. A swap is a fixed-price strip.")
    else:
        c2.selectbox("Structure", ["Swap"], disabled=True, key=f"{prefix}_instrument_fixed", help="Anything longer than one calendar month is a swap strip.")
        instrument = "SWAP"
    return start, end, instrument, label


def desk_note_html(metrics: dict, snapshot: dict, pnl_today: Optional[float], reads: Sequence[tuple] = (), outlook: str = "") -> str:
    """The morning note as short bullets, written from whatever is actually loaded. Missing data stays unsaid."""
    today_items = []
    total = metrics["eur_total"]
    if metrics["executed"]:
        trades = "trade" if metrics["executed"] == 1 else "trades"
        item = f'Book: <b class="{tone_of(total)}">{fmt_money(total)}</b> across {metrics["executed"]} executed {trades}'
        if pnl_today is not None:
            item += f', <b class="{tone_of(pnl_today)}">{fmt_money(pnl_today)}</b> on the day'
        held = executed_buckets()
        state = {product_price_state(p) for p in held["product"].unique()} if not held.empty else set()
        if state & {"Demo", "Old"}:
            item += ", valued partly or fully on demo prices" if "Real" in state or "Part real" in state else ", valued on demo prices"
        elif "Part real" in state:
            item += ", valued on real day-ahead for delivered days and demo prices after"
        if src_book()[0][0] == "demo":
            item += ". These are the sample trades, not yours"
        today_items.append(item)
    else:
        today_items.append("Book: no executed trades yet")
    has_market = False
    storage = snapshot.get("storage", {})
    if storage.get("fill_pct") is not None:
        item = f'EU gas storage: <b>{storage["fill_pct"]:.1f}%</b> full'
        change = storage.get("change_7_reports_twh")
        if change is not None:
            item += f', {"down" if change < 0 else "up"} {abs(change):.1f} TWh in a week'
        today_items.append(item)
        has_market = True
    for zone, label in (("France", "French"), ("Germany-Luxembourg", "German")):
        series = zone_prices(zone)
        if not series.empty:
            today_local = datetime.now(ZoneInfo(ENTSOE_ZONES[zone]["timezone"])).date()
            current = series.loc[series["local"].dt.date.eq(today_local)]
            if not current.empty:
                today_items.append(f'{label} day-ahead: <b>{spot_kpis(current)["base"]:.1f} €/MWh</b> average today')
                has_market = True
    parts = ["<ul>" + "".join(f"<li>{item}</li>" for item in today_items) + "</ul>"]
    read_items = [f"{html.escape(market)}: {html.escape(reading.lower())}" for market, reading in reads]
    if outlook:
        read_items.append(outlook)
    if read_items:
        parts.append('<p class="desk-label">Market read</p><ul>' + "".join(f"<li>{item}</li>" for item in read_items) + "</ul>")
    if not has_market:
        parts.append('<p class="soft">Market data will appear here once it has been updated.</p>')
    if metrics["unmarked"]:
        word = "trade needs" if metrics["unmarked"] == 1 else "trades need"
        parts.append(f'<p class="soft">{metrics["unmarked"]} {word} a price before it can be valued. See Trading Book.</p>')
    return f'<div class="desk-note">{"".join(parts)}</div>'


def _outlook_line() -> str:
    outlook = winter_outlook()
    if not outlook.get("ready") or outlook.get("basis") != "weather":
        return ""
    return f"EU storage at the end of March: about <b>{outlook['scenarios'][outlook['central']]['end']:.0f}%</b> with normal weather"


def page_market_overview():
    title_block("Today at the desk", f"{valuation_date:%A %d %B %Y}. Your book and the market in one view.")
    marked = get_marked_trades()
    executed = marked.loc[marked["status"].eq("EXECUTED") & marked["mark_status"].eq("MARKED")] if not marked.empty else pd.DataFrame()
    unmarked = int(marked.loc[~marked["status"].isin(["CANCELLED", "EXPIRED"]) & ~marked["mark_status"].eq("MARKED")].shape[0]) if not marked.empty else 0
    eur_mtm = float(executed.loc[executed["currency"].eq("EUR"), "mtm"].sum()) if not executed.empty else 0.0
    gbp_mtm = float(executed.loc[executed["currency"].eq("GBP"), "mtm"].sum()) if not executed.empty else 0.0
    eur_per_gbp = fx_rate("eur_per_gbp")
    metrics = {"eur_total": eur_mtm + gbp_mtm * eur_per_gbp, "executed": int(len(executed)), "unmarked": unmarked}

    lookups = saved_lookups()
    days = sorted(lookups)
    recent_days = days[-31:]
    trend = session_memo("overview_trend", (data_version("trades", "snapshots"), eur_per_gbp),
                         lambda: pnl_history(executed_buckets(), {d: lookups[d] for d in recent_days}, eur_per_gbp))
    pnl_today = float(trend["pnl_eur"].iloc[-1]) if not trend.empty and not executed_buckets().empty else None

    snapshot = live_snapshot()
    hero = st.container(key="hero")
    note_col, chart_col = hero.columns([1, 1.05], gap="large")
    with note_col:
        reads = [(name, read["summary"]["reading"]) for name, read in market_reads().items() if read["summary"]["confidence"] != "None"]
        st.markdown(desk_note_html(metrics, snapshot, pnl_today, reads, _outlook_line()), unsafe_allow_html=True)
        update_button("overview_refresh")
    with chart_col:
        names = ["TTF Natural Gas", "French Power Base", "German Power Base"]
        frame = pd.concat([curves[n].assign(delivery_month=lambda f: pd.to_datetime(f["delivery_month"])) for n in names], ignore_index=True)
        frame = frame.loc[frame["delivery_month"].lt(pd.Timestamp(_add_months(valuation_date, 25))) & frame["delivery_month"].ge(pd.Timestamp(valuation_date.replace(day=1)))]
        fig = px.line(frame, x="delivery_month", y="price", color="product", title="Forward curves, EUR/MWh" + (lambda st_: " (demo prices)" if st_ == {"Demo"} else " (partly demo prices)" if "Demo" in st_ or "Part real" in st_ else " (real prices)")({product_price_state(p) for p in names}),
                      labels={"delivery_month": "", "price": "", "product": ""})
        chart(fig, 290)

    mtm_trend = trend["mtm_eur"].tolist() if len(trend) >= 3 else None
    st.subheader("Book")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Open MtM, EUR trades", fmt_money(eur_mtm), tone=tone_of(eur_mtm), accent=BLUE)
    kpi(c2, "Open MtM, GBP trades", fmt_money(gbp_mtm, "GBP"), tone=tone_of(gbp_mtm), accent=TEAL)
    kpi(c3, "Total in EUR", fmt_money(metrics["eur_total"]), "Last 30 saved days" if mtm_trend else f"At {eur_per_gbp:.4f} EUR per GBP", tone=tone_of(metrics["eur_total"]), spark=mtm_trend)
    kpi(c4, "Trades needing a price", f"{unmarked}", "All marked" if not unmarked else "See Trading Book", accent=GAIN if not unmarked else LOSS)

    st.subheader("Market")
    storage, lng, load = snapshot.get("storage", {}), snapshot.get("lng", {}), snapshot.get("power_load", {})
    alsi = _live_dataset_frame("gie_alsi")
    send_col = _find_column(alsi, ("sendOut", "sendout", "send_out"))
    france = zone_prices("France")
    m1, m2, m3, m4 = st.columns(4)
    kpi(m1, "EU gas storage", f"{storage['fill_pct']:.1f}%" if storage.get("fill_pct") is not None else "Not loaded",
        f"{storage['inventory_twh']:.0f} TWh in store" if storage.get("inventory_twh") is not None else "",
        accent=ENTITY_COLORS["TTF Natural Gas"], meter=storage.get("fill_pct"))
    kpi(m2, "LNG send-out", f"{lng['sendout']:,.0f} GWh/d" if lng.get("sendout") is not None else "Not loaded", accent="#17A891",
        spark=_number_series(alsi[send_col]).tail(30).tolist() if send_col else None)
    kpi(m3, "French day-ahead", f"{spot_kpis(france.tail(24))['base']:.1f} €/MWh" if not france.empty else "Not loaded", "Last 24 hours" if not france.empty else "", accent=SODIUM,
        spark=spot_hourly(france)["value"].tail(72).tolist() if not france.empty else None)
    kpi(m4, "Actual power load", f"{load['value'] / 1000:.1f} GW" if load.get("value") is not None else "Not loaded", str(load.get("zone") or ""), accent="#9180EE")

    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
    weather_daily = _live_dataset_frame("weather_daily")
    normals = _live_dataset_frame("weather_normals")
    anomaly = session_memo("hdd_anomaly", data_version("live"), lambda: hdd_vs_normal(weather_daily, normals))
    if not anomaly.empty:
        fig = px.bar(anomaly.sort_values("hdd_vs_normal"), x="hdd_vs_normal", y="location", orientation="h",
                     title="Heating demand over the next 7 days against normal, degree days",
                     labels={"hdd_vs_normal": "", "location": ""})
        fig.update_traces(marker_color=BLUE)
        chart(fig, 280)
    elif not weather_daily.empty:
        summary = weather_forecast_summary(weather_daily, horizon_days=7, fallback_location="Selected location")
        if not summary.empty and "Heating degree-day proxy (base 18°C)" in summary.columns:
            fig = px.bar(summary.sort_values("Heating degree-day proxy (base 18°C)"), x="Heating degree-day proxy (base 18°C)", y="Location", orientation="h",
                         title="Heating degree days forecast for the next 7 days", labels={"Heating degree-day proxy (base 18°C)": "", "Location": ""})
            fig.update_traces(marker_color=BLUE)
            chart(fig, 280)

    with st.expander("Where this data comes from and how fresh it is"):
        rows = []
        for key, item in st.session_state.get("live_data", {}).items():
            data = item.get("data") if isinstance(item, dict) else None
            rows.append({"Dataset": item.get("label", key), "Rows": len(data) if isinstance(data, pd.DataFrame) else 0,
                         "Latest observation": _latest_observation_date(data) if isinstance(data, pd.DataFrame) else None,
                         "Fetched (UTC)": item.get("fetched_at", "")})
        if rows:
            table(pd.DataFrame(rows))
        else:
            st.caption("Nothing fetched yet.")
        st.caption("GIE AGSI+ and ALSI attribution is required. Weather data by Open-Meteo under CC BY 4.0. Providers can delay or revise observations.")


def page_power_spot():
    title_block("Power spot", "Compare day-ahead prices across bidding zones: levels, shape, spreads and what wind and solar actually earn.")
    zones = list(ENTSOE_ZONES)
    with st.container(key="panel"):
        c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
        selected = c1.multiselect("Bidding zones", zones, default=[z for z in ("France", "Germany-Luxembourg", "Netherlands") if z in zones], key="spot_zones")
        horizon = c2.selectbox("Period", [7, 30, 90, 365], index=1, format_func=lambda d: f"Last {d} days", key="spot_days")
        fetch = c3.button("Fetch prices", type="primary", width="stretch", key="spot_fetch", disabled=not selected)
    if fetch:
        token = resolved_credentials()["entsoe"]
        end = date.today() + timedelta(days=2)
        start = date.today() - timedelta(days=int(horizon))
        progress = st.progress(0.0, text="Requesting day-ahead prices")
        failures = []
        for index, zone in enumerate(selected):
            try:
                frame = fetch_entsoe_series(token, "Day-ahead price", zone, start, end)
                store_entsoe(f"entsoe_price|{zone}", f"Day-ahead price, {zone}", frame, {"zone": zone}, KEEP_DAYS["prices"])
            except Exception as exc:
                failures.append(f"{zone}: {exc}")
            progress.progress((index + 1) / len(selected), text=f"Loaded {zone}")
        progress.empty()
        for failure in failures:
            st.error(failure)
    loaded = {z: zone_prices(z) for z in _live_variants("entsoe_price") if z in ENTSOE_ZONES}
    # The period chooses what is shown, not only what is fetched: the update holds a year of prices.
    first_day = date.today() - timedelta(days=int(horizon))
    active = {z: s.loc[s["local"].dt.date > first_day] for z, s in loaded.items() if z in selected and not s.empty}
    active = {z: s for z, s in active.items() if not s.empty}
    if not active:
        need_data_note("Choose your zones and select Fetch prices to load day-ahead data from ENTSO-E.")
        return

    rows = []
    for zone, series in active.items():
        k = spot_kpis(series)
        rows.append({"Zone": zone, "Base avg": k["base"], "Peak avg": k["peak"], "Peak minus base": k["peak_minus_base"], "Low": k["low"], "High": k["high"],
                     "Negative hours": k["negative_hours"], "Hourly volatility": k["volatility"], "Hours": k["hours"]})
    st.caption(f"EUR/MWh. Peak is weekdays 08:00 to 20:00 local time. Averages are weighted by time, so 15-minute and hourly data compare fairly. Showing {min(s['local'].min() for s in active.values()):%d %b %Y} to {max(s['local'].max() for s in active.values()):%d %b %Y}. A zone with less history covers a shorter span.")
    table(pd.DataFrame(rows))

    tabs = view_tabs(["Prices", "Daily shape", "Heatmap", "Duration curve", "Zone spread", "Wind and solar capture", "Next-day estimate"])
    if tabs[6]:
        render_next_day(list(active))
    if tabs[0]:
        long_view = max(k.total_seconds() for k in (s["timestamp"].max() - s["timestamp"].min() for s in active.values())) > 60 * 86400
        frames = []
        for zone, series in active.items():
            if long_view:
                daily = series.assign(day=series["local"].dt.date).groupby("day", as_index=False)["value"].mean().rename(columns={"day": "time"})
            else:
                daily = spot_hourly(series).rename(columns={"hour": "time"})
            frames.append(daily.assign(zone=zone))
        fig = px.line(pd.concat(frames, ignore_index=True), x="time", y="value", color="zone",
                      title="Daily average day-ahead price" if long_view else "Hourly day-ahead price",
                      labels={"time": "", "value": "EUR/MWh", "zone": ""})
        chart(fig, 400)
    if tabs[1]:
        frames = []
        for zone, series in active.items():
            frames.append(series.assign(hour=series["local"].dt.hour).groupby("hour", as_index=False)["value"].mean().assign(zone=zone))
        fig = px.line(pd.concat(frames, ignore_index=True), x="hour", y="value", color="zone", title="Average price by local hour of day",
                      labels={"hour": "Local hour", "value": "EUR/MWh", "zone": ""})
        fig.update_xaxes(dtick=2)
        chart(fig, 400)
    if tabs[2]:
        zone = st.columns([1, 2])[0].selectbox("Zone", list(active), key="spot_heat_zone")
        heat = active[zone].assign(day=active[zone]["local"].dt.date, hour=active[zone]["local"].dt.hour)
        pivot = heat.pivot_table(index="hour", columns="day", values="value", aggfunc="mean")
        limit = float(pivot.abs().quantile(0.98).max()) if pivot.size else 100.0
        fig = go.Figure(go.Heatmap(z=pivot.values, x=[str(d) for d in pivot.columns], y=pivot.index, xgap=1, ygap=1,
                                   colorscale=[[0, LOSS], [0.5, "#23434E"], [1, SODIUM]], zmid=0, zmin=-limit, zmax=limit,
                                   colorbar=dict(title="EUR/MWh", thickness=12, outlinewidth=0),
                                   hovertemplate="%{x}, hour %{y}: %{z:.1f} EUR/MWh<extra></extra>"))
        fig.update_layout(title=f"{zone}: price by day and local hour", yaxis_title="Local hour", hovermode="closest")
        fig.update_yaxes(autorange="reversed", dtick=4)
        chart(fig, 430)
    if tabs[3]:
        frames = [price_duration_curve(series).assign(zone=zone) for zone, series in active.items()]
        fig = px.line(pd.concat(frames, ignore_index=True), x="share_of_hours", y="price", color="zone", title="Share of hours at or above each price",
                      labels={"share_of_hours": "Share of hours (%)", "price": "EUR/MWh", "zone": ""})
        chart(fig, 400)
    if tabs[4]:
        if len(active) < 2:
            need_data_note("Load at least two zones to compare them.")
        else:
            a1, a2 = st.columns(2)
            names = list(active)
            first = a1.selectbox("Zone", names, index=0, key="spot_spread_a")
            second = a2.selectbox("Against", names, index=1, key="spot_spread_b")
            if first == second:
                need_data_note("Choose two different zones.")
            else:
                merged = spot_hourly(active[first]).merge(spot_hourly(active[second]), on="hour", suffixes=("_a", "_b"))
                merged["spread"] = merged["value_a"] - merged["value_b"]
                c1, c2, c3 = st.columns(3)
                kpi(c1, "Average spread", f"{merged['spread'].mean():+.2f} €/MWh")
                kpi(c2, f"Hours {first} was dearer", f"{(merged['spread'] > 0).mean() * 100:.0f}%")
                kpi(c3, "Largest gap", f"{merged['spread'].abs().max():.2f} €/MWh")
                fig = px.line(merged, x="hour", y="spread", title=f"{first} minus {second}", labels={"hour": "", "spread": "EUR/MWh"})
                fig.update_traces(line_color=BLUE)
                chart(fig, 340)
    if tabs[5]:
        zone = st.columns([1, 2])[0].selectbox("Zone", list(active), key="spot_capture_zone")
        generation = _live_dataset_frame(f"entsoe_generation|{zone}")
        if st.button(f"Fetch generation for {zone}", key="spot_capture_fetch"):
            series = active[zone]
            start_day = series["local"].min().date()
            end_day = min(series["local"].max().date() + timedelta(days=1), date.today() + timedelta(days=1))
            try:
                with st.spinner("Requesting actual generation"):
                    generation = fetch_entsoe_series(resolved_credentials()["entsoe"], "Actual generation by type", zone, start_day, end_day)
                generation = store_entsoe(f"entsoe_generation|{zone}", f"Actual generation, {zone}", generation, {"zone": zone}, KEEP_DAYS["generation"])
            except Exception as exc:
                st.error(f"Could not fetch generation: {exc}")
        captured = capture_prices(active[zone], generation, spot_kpis(active[zone])["base"])
        if captured.empty:
            need_data_note(f"Fetch generation for {zone} to see what wind and solar earned against the baseload price.")
        else:
            st.caption(f"Capture price is the output-weighted price. A rate below 100% means the technology earned less than baseload ({spot_kpis(active[zone])['base']:.1f} €/MWh).")
            table(captured)
            fig = px.bar(captured, x="technology", y="capture_rate_pct", title="Capture rate against baseload, %", labels={"technology": "", "capture_rate_pct": ""})
            fig.update_traces(marker_color=BLUE)
            fig.add_hline(y=100, line=dict(color=GRAY, dash="dot"))
            chart(fig, 320)


def page_spreads():
    title_block("Spreads and margins", "What a gas or coal plant earns on the forward curve, and the location, shape and hub spreads around it.")
    note = price_warning([p for p in CURVE_PRODUCTS if p != "NBP Natural Gas"])
    if note:
        st.warning(note)
    tab_margin, tab_other, tab_history = view_tabs(["Clean spark and dark", "Location, shape and hub", "History"])
    if tab_margin:
        c1, c2 = st.columns(2)
        power = c1.selectbox("Power", [p for p, s in PRODUCTS.items() if s["commodity"] == "Power"], key="spread_power")
        gas = c2.selectbox("Gas hub", [p for p, s in PRODUCTS.items() if s["commodity"] == "Gas"], key="spread_gas")
        with st.expander("Plant assumptions and FX"):
            a, b, c, d = st.columns(4)
            params = {"gas_efficiency": a.number_input("Gas plant efficiency", 0.3, 0.7, DEFAULT_SPREAD_PARAMS["gas_efficiency"], 0.01, key="spread_eta_g"),
                      "coal_efficiency": b.number_input("Coal plant efficiency", 0.25, 0.5, DEFAULT_SPREAD_PARAMS["coal_efficiency"], 0.01, key="spread_eta_c"),
                      "gas_emission_factor": c.number_input("Gas emissions, tCO2/MWh(th)", 0.1, 0.4, DEFAULT_SPREAD_PARAMS["gas_emission_factor"], 0.001, format="%.3f", key="spread_ef_g"),
                      "coal_emission_factor": d.number_input("Coal emissions, tCO2/MWh(th)", 0.2, 0.5, DEFAULT_SPREAD_PARAMS["coal_emission_factor"], 0.001, format="%.3f", key="spread_ef_c")}
            eur_per_gbp, eur_per_usd = fx_controls("spread")
        spreads = spreads_table(curves, power, gas, params, eur_per_gbp, eur_per_usd)
        if spreads.empty:
            need_data_note("These curves share no delivery months. Check curve coverage in Forward Curves.")
        else:
            front = spreads.iloc[0]
            k1, k2, k3, k4 = st.columns(4)
            kpi(k1, f"Clean spark, {front['contract']}", f"{front['clean_spark']:+.2f} €/MWh", tone=tone_of(front["clean_spark"]))
            kpi(k2, f"Clean dark, {front['contract']}", f"{front['clean_dark']:+.2f} €/MWh", tone=tone_of(front["clean_dark"]))
            kpi(k3, "Fuel in the money", "Gas" if front["gas_minus_coal"] > 0 else "Coal", f"{abs(front['gas_minus_coal']):.2f} €/MWh ahead")
            kpi(k4, "Gas switching price", f"{front['switching_gas_price']:.2f} €/MWh", "Gas and coal earn the same here")
            long_form = spreads.melt(id_vars=["contract", "delivery_month"], value_vars=["clean_spark", "clean_dark"], var_name="spread", value_name="eur_mwh")
            long_form["spread"] = long_form["spread"].map({"clean_spark": "Clean spark", "clean_dark": "Clean dark"})
            long_form["delivery_month"] = pd.to_datetime(long_form["delivery_month"])
            fig = px.line(long_form, x="delivery_month", y="eur_mwh", color="spread", title=f"{power} against {gas}, EUR per MWh of power",
                          labels={"delivery_month": "", "eur_mwh": "", "spread": ""}, color_discrete_map={"Clean spark": BLUE, "Clean dark": AMBER})
            chart(fig, 380)
            table(spreads.drop(columns=["delivery_month"]).rename(columns={
                "contract": "Contract", "power_eur_mwh": "Power, EUR/MWh", "gas_eur_mwh": "Gas, EUR/MWh", "eua_eur_t": "EUA, EUR/t",
                "coal_eur_mwh_th": "Coal, EUR/MWh(th)", "clean_spark": "Clean spark", "clean_dark": "Clean dark",
                "gas_minus_coal": "Spark minus dark", "switching_gas_price": "Gas switching price"}), height=320)
    if tab_other:
        eur_per_gbp = fx_rate("eur_per_gbp")
        pairs = [("French Power Base", "German Power Base", "France minus Germany, base"),
                 ("French Power Peak", "French Power Base", "French peak minus base"),
                 ("German Power Peak", "German Power Base", "German peak minus base"),
                 ("TTF Natural Gas", "NBP Natural Gas", "TTF minus NBP, in EUR")]
        frames = [curve_spread(curves, a, b, eur_per_gbp).assign(spread=label) for a, b, label in pairs]
        data = pd.concat(frames, ignore_index=True)
        data["delivery_month"] = pd.to_datetime(data["delivery_month"])
        fig = px.line(data, x="delivery_month", y="spread_eur_mwh", color="spread", title="Forward spreads, EUR/MWh",
                      labels={"delivery_month": "", "spread_eur_mwh": "", "spread": ""})
        chart(fig, 420)
        st.caption(f"NBP is converted at {eur_per_gbp:.4f} EUR per GBP.")
    if tab_history:
        history = saved_snapshots()
        lookups = saved_lookups()
        if len(lookups) < 2:
            need_data_note("Spread history appears once at least two days of curves are saved. Curves save automatically each day, or create demo history in Daily P&L.")
        else:
            rows = []
            for day in sorted(lookups):
                daily = lookups[day]
                target = _add_months(day, 1)
                entry = {"date": day}
                for a, b, label in (("French Power Base", "German Power Base", "France minus Germany"),
                                    ("French Power Peak", "French Power Base", "French peak minus base")):
                    if target in daily.get(a, {}) and target in daily.get(b, {}):
                        entry[label] = daily[a][target] - daily[b][target]
                gas, carbon, base = daily.get("TTF Natural Gas", {}), daily.get("EUA Carbon", {}), daily.get("French Power Base", {})
                if target in gas and target in carbon and target in base:
                    eta, factor = DEFAULT_SPREAD_PARAMS["gas_efficiency"], DEFAULT_SPREAD_PARAMS["gas_emission_factor"]
                    entry["French clean spark"] = base[target] - gas[target] / eta - carbon[target] * factor / eta
                rows.append(entry)
            data = pd.DataFrame(rows).melt(id_vars="date", var_name="spread", value_name="eur_mwh").dropna()
            fig = px.line(data, x="date", y="eur_mwh", color="spread", title="Front-month spreads over saved days, EUR/MWh",
                          labels={"date": "", "eur_mwh": "", "spread": ""})
            chart(fig, 400)
            sources = set(history["source"])
            if SYNTHETIC_HISTORY_SOURCE in sources:
                st.warning("This includes synthetic demo history, which is not a market record.")


def page_forward_curves():
    title_block("Forward curves", "Check the monthly marks behind every quote and valuation, or replace them with your own.")
    selected = st.columns([1, 2])[0].selectbox("Curve", list(CURVE_PRODUCTS), key="forward_curve_product",
                            format_func=lambda p: f"{p} ({CURVE_PRODUCTS[p]['unit']})")
    spec = CURVE_PRODUCTS[selected]
    curve = validate_curve(curves[selected])
    curve["delivery_month"] = pd.to_datetime(curve["delivery_month"])
    after_current = curve.loc[curve["delivery_month"] > pd.Timestamp(valuation_date.replace(day=1))]
    first = after_current.iloc[0] if not after_current.empty else curve.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"Front month, {first['delivery_month']:%b %Y}", f"{float(first['price']):,.2f}", spec["unit"])
    kpi(c2, "Curve average", f"{curve['price'].mean():,.2f}", spec["unit"])
    kpi(c3, "Low", f"{curve['price'].min():,.2f}", spec["unit"])
    kpi(c4, "High", f"{curve['price'].max():,.2f}", spec["unit"])

    history = saved_snapshots()
    product_days = sorted(history.loc[history["product"].eq(selected), "snapshot_date"].unique()) if not history.empty else []
    previous = [d for d in product_days if d < valuation_date.isoformat()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["delivery_month"], y=curve["price"], mode="lines+markers", name="Today", line=dict(color=ENTITY_COLORS.get(selected, BLUE), width=2)))
    if previous:
        earlier = history.loc[history["product"].eq(selected) & history["snapshot_date"].eq(previous[-1])].copy()
        earlier["delivery_month"] = pd.to_datetime(earlier["delivery_month"])
        fig.add_trace(go.Scatter(x=earlier["delivery_month"], y=earlier["price"], mode="lines", name=f"Saved {previous[-1]}", line=dict(color=GRAY, width=1.5, dash="dot")))
    fig.update_layout(title=f"{selected}, {spec['unit']}", yaxis_title="")
    chart(fig, 400)

    sources_here = curve_sources().get(selected, {})
    state = product_price_state(selected)
    counts = pd.Series(list(sources_here.values())).value_counts()
    mix = "; ".join(f"{n} month{'s' if n > 1 else ''} {label.lower() if label != 'Demo' else 'demo prices'}" for label, n in counts.items())
    if selected in TRADINGVIEW_CURVES:
        meta = st.session_state.get("curve_metadata", {}).get(selected, {})
        from_tv = meta.get("source_type") == "TradingView" and selected in st.session_state.get("custom_curves", {})
        status, action = st.columns([4, 1])
        with status:
            if not tradingview_available():
                st.warning(f"TTF is on demo prices: the TradingView library isn't installed. Run `{TRADINGVIEW_INSTALL}` and restart the app.")
            elif from_tv:
                st.success(f"Marks from TradingView ({TRADINGVIEW_CURVES[selected]['name']}), dated {curve_mark_date(meta):%d %b %Y}, {fetched_text(meta.get('imported_at_utc')).lower()}. "
                           "Refreshed every hour when the app updates.")
            else:
                st.info("The TTF curve comes from TradingView by default. It's fetched when the app opens and with each update.")
        with action:
            running = bool(st.session_state.get("update_job"))
            if tradingview_available() and st.button("Fetch TTF now", key="tv_fetch_now", disabled=running, width="stretch"):
                if start_update(tasks=tradingview_tasks(True)):
                    st.rerun()
        st.caption("Through tvdatafeed: prices are delayed, the library is unofficial and can stop working when TradingView changes, and TradingView's terms don't allow automated access, so keep this to personal use.")
    if selected in st.session_state.get("custom_curves", {}):
        meta = st.session_state.get("curve_metadata", {}).get(selected, {})
        marked = curve_mark_date(meta)
        st.caption(f"Imported marks{f' dated {marked:%d %b %Y}' if marked else ''}. Check your data licence. Months: {mix}.")
    if state in ("Demo", "Part real", "Old"):
        st.warning(f"This curve isn't fully real. Months: {mix}. Add real marks in the views below to replace the demo prices.")
    elif selected not in st.session_state.get("custom_curves", {}):
        st.caption(f"Real prices from free sources. Months: {mix}.")
    if spec["commodity"] == "Power":
        zone = "France" if spec["zone"] == "France" else "Germany-Luxembourg"
        series = zone_prices(zone)
        if not series.empty:
            st.caption(f"For context, {zone} day-ahead averaged {spot_kpis(series)['base']:.2f} €/MWh over the loaded period. Day-ahead prices never replace forward marks.")

    tab_table, tab_status, tab_build, tab_import, tab_history = view_tabs(["Monthly marks", "Curve status and folder", "Build from contract quotes", "Import a CSV", "Import past curves"])
    if tab_table:
        display_curve = curve.copy()
        display_curve["delivery_month"] = display_curve["delivery_month"].dt.date
        if previous:
            earlier_lookup = dict(zip(pd.to_datetime(earlier["delivery_month"]).dt.date, earlier["price"]))
            display_curve["change"] = [p - earlier_lookup[m] if m in earlier_lookup else float("nan") for m, p in zip(display_curve["delivery_month"], display_curve["price"])]
        display_curve["source"] = [sources_here.get(m, "") for m in display_curve["delivery_month"]]
        table(display_curve[[c for c in display_curve.columns if c in ("delivery_month", "price", "change", "unit", "source")]], height=320)
        st.download_button("Download this curve as CSV", data=display_curve.to_csv(index=False).encode("utf-8"),
                           file_name=f"{selected.lower().replace(' ', '_')}_forward_curve.csv", mime="text/csv")
    if tab_status:
        render_curve_status()
    if tab_build:
        st.caption("Enter the contracts you can see quoted, in any mix of months, quarters, seasons and years. Meridian finds monthly marks that reprice each one exactly while keeping the current curve's shape.")
        default_quotes = pd.DataFrame({"Contract": [c for c in contract_choices(valuation_date) if c.startswith(("Q", "Cal"))][:3], "Price": [None, None, None]})
        default_quotes["Price"] = pd.to_numeric(default_quotes["Price"])
        edited = st.data_editor(default_quotes, num_rows="dynamic", hide_index=True, width="stretch", key=f"tenor_quotes_{selected}",
                                column_config={"Contract": st.column_config.TextColumn("Contract", help="For example Dec-26, Q1-27, Sum-27, Win-27 or Cal-27"),
                                               "Price": st.column_config.NumberColumn("Price", help=spec["unit"])})
        quotes = [(str(r["Contract"]).strip(), float(r["Price"])) for _, r in edited.iterrows() if pd.notna(r["Price"]) and str(r["Contract"]).strip()]
        keep_shape = st.checkbox("Keep the current curve's monthly shape", value=True, key=f"tenor_shape_{selected}")
        if quotes:
            try:
                built = decompose_tenor_quotes(quotes, selected, curves[selected] if keep_shape else None)
                preview = built.copy()
                preview["delivery_month"] = pd.to_datetime(preview["delivery_month"])
                fig = px.bar(preview, x="delivery_month", y="price", title=f"Monthly marks that reprice your {len(quotes)} quotes", labels={"delivery_month": "", "price": spec["unit"]})
                fig.update_traces(marker_color=ENTITY_COLORS.get(selected, BLUE))
                chart(fig, 300)
                existing = curves[selected] if keep_shape else None
                if st.button("Apply these marks", type="primary", key="apply_tenor_curve"):
                    merged = built
                    if existing is not None:
                        untouched = validate_curve(existing)
                        untouched = untouched.loc[~untouched["delivery_month"].isin(set(built["delivery_month"]))]
                        merged = validate_curve(pd.concat([untouched, built], ignore_index=True))
                    commit_custom_curve(selected, merged, {"source_label": "Built from contract quotes", "source_type": "Contract quotes",
                                                           "quotes": dict(quotes), "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    st.session_state["flash_message"] = f"Applied marks for {selected}."
                    st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if tab_import:
        st.caption("Columns: delivery_month, product, price, currency, unit, with one row per product and month. For a vendor's own format, use Live Data Integrations.")
        curve_file = st.file_uploader("Forward curve CSV", type=["csv"], key="forward_curve_upload")
        if curve_file is not None:
            try:
                candidate_curve = validate_curve(pd.read_csv(curve_file))
                import_product = st.selectbox("Product to update", sorted(candidate_curve["product"].unique().tolist()), key="forward_curve_import_product")
                import_subset = candidate_curve.loc[candidate_curve["product"].eq(import_product)].copy()
                st.caption(f"{len(import_subset)} monthly marks in {import_subset['unit'].iloc[0]}.")
                table(import_subset, height=240)
                if st.button("Apply this curve", key="apply_forward_curve_csv", type="primary"):
                    commit_custom_curve(import_product, import_subset, {
                        "source_label": "CSV import", "source_type": "CSV upload",
                        "file_name": getattr(curve_file, "name", "user_curve.csv"),
                        "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    st.session_state["flash_message"] = f"Applied {len(import_subset)} marks for {import_product}."
                    st.rerun()
            except Exception as exc:
                st.error(f"This curve was not applied: {exc}")
    if tab_history:
        st.caption("Load past end-of-day curves to build Daily P&L and VaR straight away. Columns: snapshot_date, delivery_month, product, price.")
        history_file = st.file_uploader("Curve history CSV", type=["csv"], key="curve_history_upload")
        if history_file is not None and st.button("Add to saved curves", key="apply_curve_history", type="primary"):
            try:
                raw = pd.read_csv(history_file)
                missing = {"snapshot_date", "delivery_month", "product", "price"} - set(raw.columns)
                if missing:
                    raise ValueError(f"Missing columns: {sorted(missing)}")
                unknown = sorted(set(raw["product"]) - set(CURVE_PRODUCTS))
                if unknown:
                    raise ValueError(f"Unknown products: {unknown}")
                rows = pd.DataFrame({"snapshot_date": pd.to_datetime(raw["snapshot_date"], errors="raise").dt.date.astype(str),
                                     "product": raw["product"], "delivery_month": pd.to_datetime(raw["delivery_month"], errors="raise").dt.to_period("M").dt.to_timestamp().dt.date.astype(str),
                                     "price": pd.to_numeric(raw["price"], errors="raise"), "source": "CSV history import"})
                store = desk_store()
                if store is None:
                    raise RuntimeError("Saving is unavailable in this folder.")
                added = store.add_snapshot_rows(rows, replace=True)
                bump("snapshots")
                st.success(f"Saved {added:,} past marks across {rows['snapshot_date'].nunique()} days.")
            except Exception as exc:
                st.error(f"History was not added: {exc}")


def page_otc_quote_builder():
    title_block("OTC quote builder", "Price a contract from the current curve. The ticket updates as you type.")
    inputs, ticket = st.columns([1.25, 1], gap="large")
    with inputs:
        c1, c2 = st.columns(2)
        product = c1.selectbox("Product", list(PRODUCTS), key="quote_product")
        side = c2.selectbox("Desk side", ["SELL_TO_CLIENT", "BUY_FROM_CLIENT"], key="quote_side",
                            format_func=lambda x: "Desk sells to client" if x == "SELL_TO_CLIENT" else "Desk buys from client")
        start, end, instrument, contract = contract_picker("quote", product)
        c1, c2 = st.columns(2)
        mw = c1.number_input("Volume (MW)", min_value=0.1, max_value=100000.0, value=10.0, step=0.5, key="quote_mw")
        margin = c2.number_input("Desk margin per MWh", min_value=0.0, max_value=100.0, value=0.35, step=0.05, key="quote_margin")
        c1, c2 = st.columns(2)
        settlement_type = c1.selectbox("Settlement", ["FINANCIAL", "PHYSICAL"], format_func=lambda x: x.title(), key="quote_settlement")
        counterparty = c2.text_input("Counterparty", value="Prospective client", key="quote_counterparty")
        st.caption(f"Profile: {PRODUCTS[product]['profile'].title()}, set by the product. Choose a peak product to quote peak hours.")
    profile = PRODUCTS[product]["profile"]
    with ticket:
        with st.container(key="ticket"):
            try:
                ref = (forward_month_price(curves[product], product, start, end) if instrument == "FORWARD"
                       else reference_mark(product, start, end, profile))
                quoted = quote_price(ref, margin, side)
                hours = delivery_hours(start, end, profile, PRODUCTS[product]["timezone"])
                total_mwh = mw * hours
                currency = PRODUCTS[product]["currency"]
                st.markdown(f'<div class="ticket-price">{quoted:,.3f}</div><div class="ticket-unit">{PRODUCTS[product]["unit"]}, indicative client price</div>', unsafe_allow_html=True)
                rows = [("Contract", f"{contract if contract != CUSTOM_CONTRACT else 'Custom dates'}, {instrument.title()}"),
                        ("Delivery", f"{start:%d %b %Y} to {end - timedelta(days=1):%d %b %Y}"),
                        ("Curve reference", f"{ref:,.3f}"), ("Delivery hours", f"{hours:,}"),
                        ("Energy", f"{total_mwh:,.0f} MWh"), ("Contract value", fmt_money(quoted * total_mwh, currency, signed=False)),
                        ("Desk margin in total", fmt_money(margin * total_mwh, currency, signed=False)),
                        ("Curve source", curve_source_label(product))]
                st.markdown("".join(f'<div class="ticket-row"><span>{html.escape(a)}</span><span>{html.escape(str(b))}</span></div>' for a, b in rows), unsafe_allow_html=True)
                st.markdown('<div style="height:0.7rem"></div>', unsafe_allow_html=True)
                if st.button("Add to book as proposed", type="primary", width="stretch", key="quote_add"):
                    new_id = f"Q-{pd.Timestamp.now().strftime('%y%m%d%H%M%S')}"
                    new = pd.DataFrame([{"trade_id": new_id, "trade_date": valuation_date.isoformat(), "counterparty": counterparty or "Unspecified",
                                         "product": product, "instrument": instrument, "settlement_type": settlement_type,
                                         "side": "BUY_FIXED" if side == "BUY_FROM_CLIENT" else "SELL_FIXED",
                                         "delivery_start": start.isoformat(), "delivery_end": end.isoformat(), "profile": profile,
                                         "quantity_mw": mw, "trade_price": quoted, "currency": currency, "status": "PROPOSED",
                                         "notes": "From OTC Quote Builder"}])
                    try:
                        commit_trades(add_trade_to_book(st.session_state.trades, new))
                        st.session_state["flash_message"] = f"Added {new_id} as proposed. It joins positions once executed."
                        st.session_state["pending_nav_page"] = "Trading Book & MtM"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not add this quote: {exc}")
            except Exception as exc:
                st.error(f"This contract can't be priced: {exc}")
    st.caption("The quote adds your margin to the curve when the desk sells and subtracts it when the desk buys. It is indicative and can't be executed from here.")


def page_trading_book_mtm():
    title_block("Trading book", "Capture trades, follow their status and see what they are worth against the current curves.")
    try:
        marked = get_marked_trades()
        tab_blotter, tab_positions, tab_add, tab_status, tab_import = view_tabs(["Blotter", "Positions by month", "Add a trade", "Change status", "Import or export"])
        if tab_blotter:
            if marked.empty:
                need_data_note("The book is empty. Add a trade, or import a CSV.")
            else:
                fc1, fc2 = st.columns(2)
                products_for_filter = ["All"] + sorted(marked["product"].dropna().unique().tolist())
                statuses_for_filter = ["All"] + sorted(marked["status"].dropna().astype(str).str.upper().unique().tolist())
                product_filter = fc1.selectbox("Product", products_for_filter, key="book_product_filter")
                status_filter = fc2.selectbox("Status", statuses_for_filter, key="book_status_filter", format_func=lambda x: x.title())
                view = filter_trade_book(marked, product_filter, status_filter)
                metrics = calculate_book_metrics(view)
                c1, c2, c3, c4 = st.columns(4)
                kpi(c1, "Trades shown", str(metrics["trade_count"]), f"{metrics['executed_count']} executed and marked")
                kpi(c2, "Need a price", str(metrics["unmarked_count"]))
                kpi(c3, "Executed MtM, EUR", fmt_money(metrics["eur_mtm"]), tone=tone_of(metrics["eur_mtm"]))
                kpi(c4, "Executed MtM, GBP", fmt_money(metrics["gbp_mtm"], "GBP"), tone=tone_of(metrics["gbp_mtm"]))
                if metrics["unmarked_count"]:
                    st.warning("Some active trades have no price yet. The Mark error column says why, and they are left out of the totals.")
                st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
                columns = ["trade_id", "counterparty", "product", "instrument", "settlement_type", "side", "delivery_start", "delivery_end", "profile", "quantity_mw", "quantity_mwh", "trade_price", "market_price", "currency", "mtm", "status", "mark_error"]
                view = view.assign(prices=[trade_price_state(p, a, b, f, curve_sources()) if str(s).upper() == "EXECUTED" else "" for p, a, b, f, s in
                                           zip(view["product"], view["delivery_start"], view["delivery_end"], view["profile"], view["status"])]) if not view.empty else view
                table(view[["prices"] + [c for c in columns if c in view.columns]])
                if not view.empty and "mtm" in view.columns:
                    eur = view.loc[view["prices"].ne("") & view["currency"].eq("EUR")]
                    split = eur.groupby("prices")["mtm"].sum()
                    if len(split):
                        st.caption("EUR MtM by price source: " + ", ".join(f"{fmt_money(float(split[k]))} on {k.lower()} prices" for k in ("Real", "Part real", "Demo") if k in split.index)
                                   + ". Only the real part is a market value.")
        if tab_positions:
            month_positions = position_summary_by_delivery(marked.loc[marked["status"].eq("EXECUTED")].copy() if not marked.empty else marked, curves)
            if month_positions.empty:
                need_data_note("No executed positions yet. Positions appear here when a trade is executed.")
            else:
                month_positions["delivery_month"] = pd.to_datetime(month_positions["delivery_month"]).dt.date
                fig = px.bar(month_positions, x="delivery_month", y="net_mwh", color="product", barmode="group", title="Net position by delivery month, MWh",
                             labels={"delivery_month": "", "net_mwh": "", "product": ""})
                chart(fig, 380)
                st.caption("Positive is long, negative is short. A month with a missing price shows a blank MtM rather than an assumed one.")
                table(month_positions)
        if tab_add:
            c1, c2, c3 = st.columns(3)
            product = c1.selectbox("Product", list(PRODUCTS), key="add_product")
            side = c2.selectbox("Side", ["BUY_FIXED", "SELL_FIXED"], format_func=lambda x: "Buy fixed (long)" if x == "BUY_FIXED" else "Sell fixed (short)", key="add_side")
            trade_id = c3.text_input("Trade ID", value=st.session_state.setdefault("add_default_id", f"TR-{pd.Timestamp.now().strftime('%y%m%d%H%M%S')}"), key="add_trade_id")
            start, end, instrument, _ = contract_picker("add", product)
            c1, c2, c3 = st.columns(3)
            mw = c1.number_input("Volume (MW)", min_value=0.1, max_value=100000.0, value=5.0, step=0.5, key="add_mw")
            try:
                suggested = float(forward_month_price(curves[product], product, start, end) if instrument == "FORWARD" else reference_mark(product, start, end, PRODUCTS[product]["profile"]))
            except Exception:
                suggested = float(PRODUCTS[product]["default_price"])
            price = c2.number_input(f"Fixed price ({PRODUCTS[product]['unit']})", min_value=-500.0, max_value=10000.0, value=round(suggested, 3), step=0.25, key=f"add_price_{product}_{start}_{end}", help="Starts at the current curve price. Negative power prices are allowed.")
            settlement_type = c3.selectbox("Settlement", ["FINANCIAL", "PHYSICAL"], format_func=lambda x: x.title(), key="add_settlement")
            c1, c2 = st.columns(2)
            counterparty = c1.text_input("Counterparty", value="Demo counterparty", key="add_cp")
            status = c2.selectbox("Status", ["PROPOSED", "CONFIRMED", "EXECUTED"], index=2, format_func=lambda x: x.title(), key="add_status")
            if st.button("Add trade", type="primary", key="add_trade_button"):
                row = pd.DataFrame([{"trade_id": trade_id, "trade_date": valuation_date.isoformat(), "counterparty": counterparty,
                                     "product": product, "instrument": instrument, "settlement_type": settlement_type, "side": side,
                                     "delivery_start": start.isoformat(), "delivery_end": end.isoformat(), "profile": PRODUCTS[product]["profile"],
                                     "quantity_mw": mw, "trade_price": price, "currency": PRODUCTS[product]["currency"], "status": status,
                                     "notes": "Entered manually"}])
                try:
                    commit_trades(add_trade_to_book(st.session_state.trades, row))
                    st.session_state.pop("add_default_id", None)
                    st.session_state.pop("add_trade_id", None)
                    st.session_state["flash_message"] = f"Added {trade_id}."
                    st.rerun()
                except Exception as exc:
                    st.error(f"This trade was not added: {exc}")
        if tab_status:
            cancellable = marked.loc[~marked["status"].astype(str).str.upper().isin(["CANCELLED", "EXPIRED"])] if not marked.empty else marked
            if cancellable.empty:
                need_data_note("No open trades to update.")
            else:
                c1, c2 = st.columns(2)
                selected_trade_id = c1.selectbox("Trade", cancellable["trade_id"].astype(str).tolist(), key="cancel_trade_id")
                selected_status = str(cancellable.loc[cancellable["trade_id"].astype(str).eq(selected_trade_id), "status"].iloc[0])
                allowed = {"PROPOSED": ["CONFIRMED", "EXECUTED", "CANCELLED", "EXPIRED"], "CONFIRMED": ["EXECUTED", "CANCELLED", "EXPIRED"], "EXECUTED": ["CANCELLED", "EXPIRED"]}.get(selected_status, [])
                if allowed:
                    target_status = c2.selectbox(f"New status (now {selected_status.title()})", allowed, format_func=lambda x: x.title(), key="target_trade_status")
                    if st.button("Update status", type="primary", key="update_trade_status_button"):
                        try:
                            commit_trades(set_trade_status(st.session_state.trades, selected_trade_id, target_status))
                            st.session_state["flash_message"] = f"{selected_trade_id} is now {target_status.title()}."
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Status was not changed: {exc}")
                st.caption("Cancelled and expired trades stay in the blotter for the record, but leave positions and MtM.")
        if tab_import:
            st.download_button("Download the trade book as CSV", marked.to_csv(index=False).encode("utf-8"), "trade_book.csv", "text/csv", key="book_download")
            st.caption("Required columns: trade_id, product, side, delivery_start, delivery_end, profile, quantity_mw, trade_price, currency, status. Instrument defaults to Swap and settlement to Financial.")
            uploaded = st.file_uploader("Trades CSV", type=["csv"], key="trade_upload")
            if uploaded is not None and st.button("Replace the trade book with this file", key="replace_trade_book"):
                try:
                    imported = load_trade_book_csv(uploaded)
                    commit_trades(imported)
                    st.session_state["flash_message"] = f"Imported {len(imported)} trades."
                    st.rerun()
                except Exception as exc:
                    st.error(f"This file was not imported, and your book is unchanged: {exc}")
    except Exception as exc:
        st.error(f"The book could not be valued: {exc}")
        st.caption("No prices are invented. Check curve coverage, products, delivery dates and units.")


def page_daily_pnl():
    title_block("Daily P&L", "What changed between two saved days of curves, and why.")
    history = saved_snapshots()
    lookups = saved_lookups()
    days = sorted(lookups)
    buckets = executed_buckets()
    synthetic = not history.empty and history["source"].eq(SYNTHETIC_HISTORY_SOURCE).any()
    if len(days) < 2:
        need_data_note("Daily P&L compares two days of curves. Today's curves are saved, and tomorrow's save automatically when you open the app. To explore now, create 90 days of demo history or import past curves in Forward Curves.")
        if st.button("Create demo history", type="primary", key="seed_history_pnl"):
            seed_demo_history()
            st.rerun()
        return
    if synthetic:
        st.warning("Part of this history is synthetic demo data, created to show how the page works. It isn't a market record.")
    c1, c2, c3 = st.columns([1, 1, 1])
    current_day = c2.selectbox("To", list(reversed(days[1:])), key="pnl_to", format_func=lambda d: f"{d:%a %d %b %Y}")
    options = list(reversed([d for d in days if d < current_day]))
    previous_day = c1.selectbox("From", options, key="pnl_from", format_func=lambda d: f"{d:%a %d %b %Y}")
    with c3:
        eur_per_gbp = st.number_input("EUR per 1 GBP", min_value=0.5, max_value=2.5, value=round(fx_rate("eur_per_gbp"), 4), step=0.005, format="%.4f", key="pnl_fx")
    step = pnl_explain(buckets, lookups[previous_day], lookups[current_day], previous_day, current_day)
    if step.empty:
        need_data_note("No executed trades in this period.")
        return
    factor = step["currency"].map({"EUR": 1.0, "GBP": eur_per_gbp})
    total = {col: float((step[col] * factor).sum()) for col in ("new_trade", "price_move", "first_mark", "to_delivery", "previous_mtm", "current_mtm", "total_pnl")}
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "P&L for the period", fmt_money(total["total_pnl"]), tone=tone_of(total["total_pnl"]))
    kpi(k2, "From price moves", fmt_money(total["price_move"]), tone=tone_of(total["price_move"]))
    kpi(k3, "From new trades", fmt_money(total["new_trade"]), tone=tone_of(total["new_trade"]))
    kpi(k4, "Open MtM now", fmt_money(total["current_mtm"]), tone=tone_of(total["current_mtm"]))
    drivers = render_pnl_drivers(step, factor)
    left, right = st.columns(2, gap="large")
    with left:
        fig = go.Figure(go.Waterfall(
            x=["Open MtM<br>before", "New<br>trades", "Price<br>moves", "First<br>marks", "To<br>delivery", "Open MtM<br>now"],
            y=[total["previous_mtm"], total["new_trade"], total["price_move"], total["first_mark"], -total["to_delivery"], total["current_mtm"]],
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            increasing=dict(marker=dict(color=GAIN)), decreasing=dict(marker=dict(color=LOSS)), totals=dict(marker=dict(color=BLUE)),
            connector=dict(line=dict(color=BORDER, width=1)), hovertemplate="€%{y:,.0f}<extra></extra>"))
        fig.update_layout(title="How open MtM moved, EUR", yaxis_title="", hovermode="closest")
        fig.update_xaxes(tickangle=0)
        chart(fig, 360)
    with right:
        by_product = step.assign(pnl_eur=step["total_pnl"] * factor).groupby("product", as_index=False)["pnl_eur"].sum().sort_values("pnl_eur")
        fig = go.Figure(go.Bar(x=by_product["pnl_eur"], y=by_product["product"], orientation="h",
                               marker=dict(color=[GAIN if v >= 0 else LOSS for v in by_product["pnl_eur"]]),
                               hovertemplate="%{y}: €%{x:,.0f}<extra></extra>"))
        fig.update_layout(title="P&L by product, EUR", hovermode="closest", yaxis_title="")
        chart(fig, 360)
    series = session_memo("pnl_series", (data_version("trades", "snapshots"), eur_per_gbp), lambda: pnl_history(buckets, lookups, eur_per_gbp))
    if len(series) >= 2:
        left, right = st.columns(2, gap="large")
        with left:
            fig = go.Figure(go.Bar(x=series["date"], y=series["pnl_eur"], marker=dict(color=[GAIN if v >= 0 else LOSS for v in series["pnl_eur"]]), hovertemplate="%{x}: €%{y:,.0f}<extra></extra>"))
            fig.update_layout(title="Daily P&L, EUR", hovermode="closest")
            chart(fig, 300)
        with right:
            fig = px.line(series, x="date", y="cumulative_pnl_eur", title="Cumulative P&L, EUR", labels={"date": "", "cumulative_pnl_eur": ""})
            fig.update_traces(line_color=BLUE)
            chart(fig, 300)
        st.caption("Trades enter the history from their trade date. Status changes aren't replayed, so a trade cancelled today is left out of past days too.")
    render_driver_table(drivers)
    with st.expander("P&L by trade and delivery month"):
        table(step.drop(columns=["previous_mtm"]))
    st.caption("New trade: today's value of trades done since the first date. Price move: volume times the change in mark. To delivery: a month that left the curve; its value moves from open MtM to realised, so it doesn't change P&L.")


def seed_demo_history() -> None:
    """Fill past days with labelled synthetic marks for products still using demo curves."""
    products = [p for p in CURVE_PRODUCTS if product_price_state(p) != "Real"]
    rows = synthetic_curve_history(valuation_date, products)
    _save(lambda s: s.add_snapshot_rows(rows, replace=False))
    bump("snapshots")
    st.session_state["flash_message"] = f"Created {rows['snapshot_date'].nunique()} days of labelled synthetic history."


def page_risk_exposure():
    title_block("Risk and exposure", "Where the book is exposed by delivery month, what named scenarios would do to it, and how much it could lose in a day.")
    buckets = executed_buckets()
    marked = get_marked_trades()
    unmarked_exec = marked.loc[marked["status"].eq("EXECUTED") & marked["mark_status"].ne("MARKED")] if not marked.empty else pd.DataFrame()
    if buckets.empty:
        need_data_note("No executed trades yet. Risk appears here when a trade is executed.")
        return
    if not unmarked_exec.empty:
        st.warning(f"{len(unmarked_exec)} executed trade(s) have no price. Their volumes count here, but their MtM doesn't. See Trading Book.")
    eur_per_gbp, eur_per_usd = fx_controls("risk")
    executed = marked.loc[marked["status"].eq("EXECUTED") & marked["mark_status"].eq("MARKED")]
    eur_mtm = float(executed.loc[executed["currency"].eq("EUR"), "mtm"].sum())
    gbp_mtm = float(executed.loc[executed["currency"].eq("GBP"), "mtm"].sum())
    gross = float(buckets["net_mwh"].abs().sum())
    net_by_product = buckets.groupby("product")["net_mwh"].sum()
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Open MtM in EUR", fmt_money(eur_mtm + gbp_mtm * eur_per_gbp), tone=tone_of(eur_mtm + gbp_mtm * eur_per_gbp))
    kpi(k2, "Gross volume", f"{gross:,.0f} MWh")
    kpi(k3, "Largest position", f"{net_by_product.abs().max():,.0f} MWh", str(net_by_product.abs().idxmax()))
    kpi(k4, "Value of a +1 move, all products", fmt_money(float((buckets["net_mwh"] * buckets["currency"].map({"EUR": 1.0, "GBP": eur_per_gbp})).sum())), "Per 1 unit per MWh")

    tab_ladder, tab_scenarios, tab_crisis, tab_var, tab_check = view_tabs(["Delivery ladder", "Scenarios", "Crisis replay", "Value at risk", "VaR check"])
    if tab_crisis or tab_check:
        var_outcome = session_memo("var", (data_version("trades", "snapshots"), eur_per_gbp), lambda: historical_var(buckets, saved_snapshots(), eur_per_gbp, prepared=saved_moves()))
    if tab_crisis:
        render_crisis_replay(buckets, eur_per_gbp, var_outcome)
    if tab_check:
        render_var_check(var_outcome)
    if tab_ladder:
        ladder = buckets.groupby(["product", "delivery_month"], as_index=False)["net_mwh"].sum()
        pivot = ladder.pivot(index="product", columns="delivery_month", values="net_mwh").sort_index()
        chart(_ladder_figure(ladder, "Net position by product and delivery month. Blue is long, red is short.", float(pivot.abs().max().max()) or 1.0), max(260, 90 + 44 * len(pivot.index)))
        st.caption("A move of 1 currency unit per MWh in a delivery month changes the value of the book by the volume shown for that month. Volumes count delivery hours, including daylight-saving days.")
        ladder_table = pivot.fillna(0.0)
        ladder_table.columns = [month_contract_label(m) for m in ladder_table.columns]
        table(ladder_table.reset_index().rename(columns={"product": "Product"}))
    if tab_scenarios:
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        size = c1.slider("Shock size (quote currency per MWh)", min_value=-50.0, max_value=50.0, value=5.0, step=1.0, key="risk_shock")
        results = run_scenarios(buckets, float(size), _add_months(valuation_date, 1), eur_per_gbp).sort_values("pnl_total_eur")
        worst = results.iloc[0]
        kpi(c2, "Worst scenario", fmt_money(float(worst["pnl_total_eur"])), str(worst["scenario"]), tone=tone_of(float(worst["pnl_total_eur"])))
        fig = go.Figure(go.Bar(x=results["pnl_total_eur"], y=results["scenario"], orientation="h",
                               marker=dict(color=[GAIN if v >= 0 else LOSS for v in results["pnl_total_eur"]]),
                               hovertemplate="%{y}: €%{x:,.0f}<extra></extra>"))
        fig.update_layout(title=f"P&L in EUR for a {size:+.0f} per MWh shock", hovermode="closest")
        chart(fig, 390)
        table(results.rename(columns={"scenario": "Scenario", "pnl_eur_book": "EUR trades", "pnl_gbp_book": "GBP trades", "pnl_total_eur": "Total in EUR", "what_it_does": "What it does"}))
        st.caption("These are instant first-order moves. Fixed FX, no liquidity cost and no change in volumes.")
    if tab_var:
        history = saved_snapshots()
        outcome = session_memo("var", (data_version("trades", "snapshots"), eur_per_gbp), lambda: historical_var(buckets, history, eur_per_gbp, prepared=saved_moves()))
        if outcome["observations"] < 20:
            need_data_note(f"Value at risk needs at least 20 daily curve moves, and {outcome['observations']} are saved so far. Curves save each day you open the app, or you can create demo history or import past curves in Forward Curves.")
            if st.button("Create demo history", type="primary", key="seed_history_var"):
                seed_demo_history()
                st.rerun()
        else:
            if history["source"].eq(SYNTHETIC_HISTORY_SOURCE).any():
                st.warning("Part of this history is synthetic demo data, so these figures only show how the page works. They say nothing about real risk.")
            k1, k2, k3, k4 = st.columns(4)
            kpi(k1, "1-day VaR, 95%", fmt_money(outcome["var"][0.95], signed=False))
            kpi(k2, "1-day VaR, 99%", fmt_money(outcome["var"][0.99], signed=False))
            kpi(k3, "Expected shortfall, 95%", fmt_money(outcome["es"][0.95], signed=False))
            kpi(k4, "Daily moves used", f"{outcome['observations']}")
            fig = px.histogram(outcome["pnl"], x="pnl_eur", nbins=30, title="Hypothetical daily P&L of today's book, EUR", labels={"pnl_eur": "", "count": ""})
            fig.update_traces(marker_color=BLUE, marker_line_width=0)
            fig.update_yaxes(title="Days")
            fig.add_vline(x=-outcome["var"][0.95], line=dict(color=LOSS, dash="dot", width=2), annotation_text="VaR 95%", annotation_font_color=TEXT, annotation_position="top left")
            fig.add_vline(x=-outcome["var"][0.99], line=dict(color=LOSS, width=2), annotation_text="VaR 99%", annotation_font_color=TEXT, annotation_position="top left")
            fig.update_layout(hovermode="closest")
            chart(fig, 360)
            if outcome["uncovered"]:
                st.warning(f"No saved history for {', '.join(outcome['uncovered'])}, so these products aren't in the figures.")
            st.caption("Historical simulation: each saved daily move is applied to today's position at the same distance from delivery. Results depend entirely on the days saved, and don't cover liquidity or basis risks those days didn't contain.")


def market_reads() -> dict:
    """Readings for gas and each covered power zone, reused while their inputs are unchanged."""
    return session_memo("market_reads", (data_version("live", "curves"), valuation_date), _build_market_reads)


def _build_market_reads() -> dict:
    today = valuation_date
    imported = fresh_curves(st.session_state.get("custom_curves", {}), st.session_state.get("curve_metadata", {}), today)
    daily, normals = _live_dataset_frame("weather_daily"), _live_dataset_frame("weather_normals")
    reads = {"European gas": gas_signals(_live_dataset_frame("gie_agsi_history"), _live_dataset_frame("gie_alsi"), daily, normals,
                                         imported, curves, today, fx_rate("eur_per_usd"), winter_outlook())}
    loaded = set(_live_variants("entsoe_price"))
    for label, zone in MARKET_ZONES.items():
        if zone in ("France", "Germany-Luxembourg") or zone in loaded:
            reads[label] = power_signals(zone, zone_prices(zone), _live_dataset_frame(f"entsoe_load|{zone}"),
                                         _live_dataset_frame(f"entsoe_generation|{zone}"), daily, normals, imported, today)
    return {name: {"signals": signals, "summary": summarise_read(signals)} for name, signals in reads.items()}


def _read_tone(summary: dict) -> str:
    return "gain" if summary["score"] >= 0.15 else "loss" if summary["score"] <= -0.15 else ""


def _read_dial(name: str, read: dict) -> str:
    """A market's verdict on a physical bearish-to-bullish scale."""
    summary = read["summary"]
    position = (summary["score"] + 1.0) / 2.0 * 100.0
    enough = summary["confidence"] != "None"
    lead = summary["supporting"][0]["name"] if summary["supporting"] else ""
    lead_text = re.sub(r"\b(lng|ttf|eua|nbp)\b", lambda m: m[0].upper(), lead.lower()) if lead else ""
    line = f"Led by: {lead_text}." if enough and lead else "Load the missing data to form a reading." if not enough else "The evidence is evenly split."
    knob = f'<div class="knob {_read_tone(summary)}" style="left:{position:.1f}%"></div>' if enough else ""
    confidence = f'{summary["confidence"]} confidence' if enough else f'{summary["live"]} of {summary["total"]} signals'
    return (f'<div class="dial"><div class="dial-top"><span class="dial-name">{html.escape(name)}</span><span class="pill">{html.escape(confidence)}</span></div>'
            f'<div class="dial-verdict {_read_tone(summary)}">{html.escape(summary["reading"])}</div>'
            f'<div class="scale"><div class="track"></div>{knob}</div><div class="scale-ends"><span>Bearish</span><span>Balanced</span><span>Bullish</span></div>'
            f'<div class="dial-line">{html.escape(line)}</div></div>')


def _evidence_row(signal: dict) -> str:
    score = signal["score"]
    label = "Bullish" if score >= 0.1 else "Bearish" if score <= -0.1 else "Neutral"
    tone = "gain" if score >= 0.1 else "loss" if score <= -0.1 else ""
    width = abs(score) * 50.0
    left = 50.0 if score >= 0 else 50.0 - width
    return (f'<div class="ev"><div class="ev-main"><div class="ev-name">{html.escape(signal["name"])}</div>'
            f'<div class="ev-value">{html.escape(signal["value"])}</div><div class="ev-why">{html.escape(signal["why"])}</div></div>'
            f'<div class="ev-side"><div class="ev-tag {tone}">{label}</div><div class="ev-bar"><i class="{tone}" style="left:{left:.1f}%;width:{width:.1f}%"></i></div></div></div>')


def page_market_read():
    title_block("Market read", "What the evidence says for each market, how sure it is, and what would change the view.")
    reads = market_reads()
    columns = st.columns(min(len(reads), 3))
    for index, (name, read) in enumerate(list(reads.items())[:3]):
        columns[index].markdown(_read_dial(name, read), unsafe_allow_html=True)
    missing_count = sum(len(r["summary"]["missing"]) for r in reads.values())
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    with c1:
        update_button("read_load_all")
    c2.caption(f"{missing_count} signal(s) are waiting for data." if missing_count else "All signals have data.")

    tabs = view_tabs(list(reads) + ["Track record"])
    if tabs[-1]:
        render_track_record()
    for tab, (name, read) in zip(tabs, reads.items()):
        summary = read["summary"]
        if tab:
            live = [s for s in read["signals"] if s["available"]]
            if live:
                ranked = sorted(live, key=lambda s: -s["weight"] * abs(s["score"]))
                st.markdown("".join(_evidence_row(s) for s in ranked), unsafe_allow_html=True)
            else:
                need_data_note("No evidence for this market yet. Select Update market data and it will load.")
            left, right = st.columns(2, gap="large")
            with left:
                st.subheader("What would change this read")
                flips = (summary["supporting"] or live)[:3]
                if flips:
                    st.markdown("".join(f'<div class="flip"><b>{html.escape(s["name"])}.</b> {html.escape(s["flip"])}</div>' for s in flips), unsafe_allow_html=True)
                else:
                    st.caption("Nothing to test yet.")
            with right:
                st.subheader("Evidence against it")
                if summary["against"]:
                    st.markdown("".join(f'<div class="flip"><b>{html.escape(s["name"])}.</b> {html.escape(s["value"])}</div>' for s in summary["against"]), unsafe_allow_html=True)
                elif summary["confidence"] != "None":
                    st.caption("No loaded signal points strongly the other way.")
                else:
                    st.caption("Nothing to test yet.")
            if summary["missing"]:
                with st.expander(f"Waiting for data ({len(summary['missing'])} of {summary['total']} signals)"):
                    for signal in summary["missing"]:
                        st.markdown(f"**{signal['name']}.** {signal['missing']}")
    with st.expander("How this reading works"):
        st.markdown("Each signal compares a current figure with its own history and becomes a score from -1 (bearish) to +1 (bullish). "
                    "The reading is the weighted average, with these fixed thresholds: 0.15 for leaning and 0.50 for a full call.\n\n"
                    "Signals built from the same fact, such as the three storage signals, share one capped weight, so one fact can't look like several opinions.\n\n"
                    "Confidence combines how much of the evidence is loaded with how far the signals agree. High confidence also needs support from at least two independent themes.")
        rows = [{"Market": name, "Signal": s["name"], "Weight": s["weight"], "Counted as": round(counted, 2), "Score": round(s["score"], 2) if s["available"] else None, "Status": "Loaded" if s["available"] else "Waiting for data"}
                for name, read in reads.items() for s, counted in zip(read["signals"], counted_weights(read["signals"]))]
        table(pd.DataFrame(rows))
        st.caption("The weights are expert judgement. They aren't fitted to prices, and the Track record view shows how the reads have done, so this organises the evidence rather than forecasting prices. It isn't trading advice. Curve signals only use imported marks, never demo prices.")


def _ladder_figure(rows: pd.DataFrame, title: str, limit: float) -> go.Figure:
    pivot = rows.pivot(index="product", columns="delivery_month", values="net_mwh").sort_index()
    pivot = pivot[sorted(pivot.columns)]
    fig = go.Figure(go.Heatmap(z=pivot.values, x=[month_contract_label(m) for m in pivot.columns], y=list(pivot.index), xgap=2, ygap=2,
                               colorscale=[[0, LOSS], [0.5, "#23434E"], [1, BLUE]], zmid=0, zmin=-limit, zmax=limit,
                               colorbar=dict(title="MWh", thickness=12, outlinewidth=0),
                               hovertemplate="%{y}, %{x}: %{z:,.0f} MWh<extra></extra>"))
    fig.update_layout(title=title, hovermode="closest")
    fig.update_xaxes(type="category")
    return fig


def page_hedge_advisor():
    title_block("Hedge advisor", "The few standard contracts that would remove the most risk from your book, and what they would cost.")
    buckets = executed_buckets()
    if buckets.empty:
        need_data_note("No executed trades yet. Execute a trade in Trading Book and the advisor will size a hedge for it.")
        return
    history = saved_snapshots()
    eur_per_gbp = fx_rate("eur_per_gbp")
    with st.container(key="panel"):
        c1, c2, c3, c4, c5 = st.columns(5, vertical_alignment="bottom")
        confidence = c1.selectbox("Confidence", [0.95, 0.99], format_func=lambda v: f"{v * 100:.0f}%", key="hedge_conf")
        max_contracts = c2.selectbox("Contracts at most", [1, 2, 3, 4, 5], index=2, key="hedge_max")
        lot = c3.selectbox("Round to", [0.5, 1.0, 5.0], index=1, format_func=lambda v: f"{v:g} MW", key="hedge_lot")
        gas_cost = c4.number_input("Gas half-spread, per MWh", min_value=0.0, max_value=5.0, value=DEFAULT_HALF_SPREADS["Gas"], step=0.01, key="hedge_gas_cost",
                                   help="Your assumption for half the bid/offer spread. It isn't a market quote.")
        power_cost = c5.number_input("Power half-spread, per MWh", min_value=0.0, max_value=10.0, value=DEFAULT_HALF_SPREADS["Power"], step=0.05, key="hedge_power_cost",
                                     help="Your assumption for half the bid/offer spread. It isn't a market quote.")
        own_only = st.checkbox("Only use products already in my book", value=True, key="hedge_own_only",
                               help="Untick to allow proxy hedges, which rely on two markets continuing to move together.")
    products = sorted(set(buckets["product"])) if own_only else list(PRODUCTS)
    spreads = {"Gas": gas_cost, "Power": power_cost}
    plan_key = (data_version("trades", "snapshots", "curves"), valuation_date, eur_per_gbp, confidence, max_contracts, lot, gas_cost, power_cost, tuple(products))
    plan = session_memo("hedge_plan", plan_key, lambda: hedge_plan(buckets, history, curves, valuation_date, eur_per_gbp, confidence, max_contracts, lot, spreads, products, prepared=saved_moves()))
    if not plan["ready"]:
        need_data_note(plan["reason"] + " Curves save each day you open the app. To explore now, create demo history, or import past curves in Forward Curves.")
        if st.button("Create demo history", type="primary", key="seed_history_hedge"):
            seed_demo_history()
            st.rerun()
        return
    if not history.empty and history["source"].eq(SYNTHETIC_HISTORY_SOURCE).any():
        st.warning("Part of this history is synthetic demo data, so these hedges only show how the advisor works. Don't act on them.")
    marks_note = price_warning(products)
    if marks_note:
        st.warning(marks_note)
    if plan["uncovered"]:
        st.warning(f"No saved history for {', '.join(plan['uncovered'])}, so these products aren't in the risk figures or the hedges.")

    level = f"{confidence * 100:.0f}%"
    hedges = plan["hedges"]
    hero = st.container(key="hero")
    note, picture = hero.columns([1, 1.05], gap="large")
    with note:
        if hedges:
            actions = [f'<b>{h["action"].lower()} {h["mw"] if h["mw"] > 0 else -h["mw"]:g} MW {html.escape(h["product"])} {html.escape(h["contract"])}</b>' for h in hedges]
            joined = actions[0] if len(actions) == 1 else ", ".join(actions[:-1]) + " and " + actions[-1]
            cut = (1.0 - plan["hedged_var"] / plan["base_var"]) * 100.0 if plan["base_var"] else 0.0
            text = (f'<p>To cut risk the most, {joined}.</p>'
                    f'<p>That would take your one-day VaR from <b>{fmt_money(plan["base_var"], signed=False)}</b> to '
                    f'<b class="gain">{fmt_money(plan["hedged_var"], signed=False)}</b>, a {cut:.0f}% reduction, for about <b>{fmt_money(plan["total_cost"], signed=False)}</b> in spread.</p>')
        else:
            text = (f'<p>No standard contract would cut your one-day VaR of <b>{fmt_money(plan["base_var"], signed=False)}</b> by at least 5%.</p>'
                    '<p class="soft">Your book may already be well balanced, or its risk may sit in products you can allow by unticking the option above.</p>')
        st.markdown(f'<div class="desk-note">{text}</div>', unsafe_allow_html=True)
        if hedges and st.button("Add these hedges to the book as proposed", type="primary", key="hedge_add"):
            proposed = proposed_hedge_trades(hedges, valuation_date, pd.Timestamp.now().strftime("%y%m%d%H%M%S"))
            try:
                commit_trades(add_trade_to_book(st.session_state.trades, proposed))
                st.session_state["flash_message"] = f"Added {len(proposed)} proposed hedge(s). Execute them in Trading Book for them to count in risk."
                st.session_state["pending_nav_page"] = "Trading Book & MtM"
                st.rerun()
            except Exception as exc:
                st.error(f"Could not add these hedges: {exc}")
    with picture:
        dist = pd.DataFrame({"pnl": np.concatenate([plan["base_pnl"], plan["hedged_pnl"]]),
                             "Book": ["As it is"] * len(plan["base_pnl"]) + ["With hedges"] * len(plan["hedged_pnl"])})
        fig = px.histogram(dist.loc[dist["Book"].eq("As it is")] if not hedges else dist, x="pnl", color="Book", barmode="overlay", nbins=30, opacity=0.72,
                           title="Daily P&L of today's book on each saved day, EUR", labels={"pnl": "", "Book": ""},
                           color_discrete_map={"As it is": AMBER, "With hedges": BLUE})
        fig.update_traces(marker_line_width=0)
        fig.update_yaxes(title="Days")
        fig.update_layout(hovermode="closest")
        chart(fig, 300)

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, f"VaR now, {level}", fmt_money(plan["base_var"], signed=False), f"Expected shortfall {fmt_money(plan['base_es'], signed=False)}", accent=AMBER)
    kpi(k2, f"VaR with hedges, {level}", fmt_money(plan["hedged_var"], signed=False), f"Expected shortfall {fmt_money(plan['hedged_es'], signed=False)}", accent=BLUE)
    kpi(k3, "Risk removed", fmt_money(plan["base_var"] - plan["hedged_var"], signed=False), tone="gain" if hedges else "")
    kpi(k4, "Estimated spread cost", fmt_money(plan["total_cost"], signed=False), f"Assumed half-spreads on {plan['observations']} saved days")

    tab_plan, tab_drivers, tab_ladder, tab_own, tab_trust = view_tabs(["The hedges", "What drives the risk", "Positions before and after", "Test your own hedge", "How far to trust this"])
    if tab_plan:
        if hedges:
            rows = [{"Action": h["action"], "Product": h["product"], "Contract": h["contract"], "MW": abs(h["mw"]), "Energy, MWh": h["mwh"],
                     "Mid": h["mid"], "Quote currency": h["currency"], "Spread cost, EUR": h["cost_eur"], "VaR after adding it, EUR": h["var_after_step"],
                     "Type": "Proxy" if h["proxy"] else "Direct"} for h in hedges]
            table(pd.DataFrame(rows))
            steps = pd.DataFrame([{"Contracts": "None", "VaR": plan["base_var"], "Cost": 0.0}] +
                                 [{"Contracts": f"{s['contracts']}", "VaR": s["var"], "Cost": s["cost"]} for s in plan["steps"]])
            fig = go.Figure(go.Bar(x=steps["Contracts"], y=steps["VaR"], marker=dict(color=[AMBER] + [BLUE] * (len(steps) - 1)),
                                   customdata=steps["Cost"], hovertemplate="%{x} contract(s): VaR €%{y:,.0f}, spread cost €%{customdata:,.0f}<extra></extra>"))
            fig.update_layout(title="One-day VaR as each contract is added, EUR", hovermode="closest", xaxis_title="Contracts in the hedge")
            fig.update_xaxes(type="category")
            chart(fig, 320)
            st.caption("Each step resizes the earlier contracts too, so the table shows the final sizes and the VaR reached at each step. Proposed hedges don't change risk until you execute them in Trading Book.")
            before_net = _book_net(buckets)
            grown = []
            for row in hedge_ladder_rows(buckets, hedges).to_dict("records"):
                previous = before_net.get(row["product"], {}).get(row["delivery_month"], 0.0)
                if abs(row["net_mwh"]) > abs(previous) + 1e-6:
                    side = "long" if row["net_mwh"] > 0 else "short"
                    grown.append(f"{row['product']} {month_contract_label(row['delivery_month'])} ({side} {abs(row['net_mwh']):,.0f} MWh)")
            if grown:
                st.warning(f"Check before using this: the hedge makes these months larger, because a quarter or season also covers months you hold less of: {', '.join(grown)}. It works only if those months keep moving with the rest of the contract.")
        else:
            need_data_note("No hedge cleared the 5% improvement bar with these settings.")
    if tab_drivers:
        drivers = plan["drivers"]
        if drivers.empty:
            need_data_note("No delivery months with history to analyse.")
        else:
            shown = drivers.copy()
            shown["Bucket"] = shown["product"] + " " + shown["contract"]
            top = pd.concat([shown.head(6), shown.tail(4)]).drop_duplicates("Bucket").sort_values("tail_loss_eur")
            fig = go.Figure(go.Bar(x=top["tail_loss_eur"], y=top["Bucket"], orientation="h", marker=dict(color=[LOSS if v > 0 else GAIN for v in top["tail_loss_eur"]]),
                                   hovertemplate="%{y}: €%{x:,.0f}<extra></extra>"))
            fig.update_layout(title=f"Average gain or loss on the worst {100 - confidence * 100:.0f}% of days, EUR. Positive bars lose money, negative bars offset.", hovermode="closest")
            chart(fig, max(300, 34 * len(top) + 110))
            lead = drivers.iloc[0]
            st.caption(f"{lead['product']} {lead['contract']} is the largest single source, at {lead['share_pct']:.0f}% of the loss on those days.")
            table(shown[["product", "contract", "net_mwh", "tail_loss_eur", "share_pct"]].rename(columns={"product": "Product", "contract": "Delivery", "net_mwh": "Net MWh", "tail_loss_eur": "Loss on worst days, EUR", "share_pct": "Share, %"}), height=280)
    if tab_ladder:
        before = hedge_ladder_rows(buckets, [])
        after = hedge_ladder_rows(buckets, hedges)
        limit = float(max(before["net_mwh"].abs().max(), after["net_mwh"].abs().max())) or 1.0
        height = max(240, 90 + 40 * max(before["product"].nunique(), after["product"].nunique()))
        chart(_ladder_figure(before, "Before, net MWh by delivery month", limit), height)
        if hedges:
            chart(_ladder_figure(after, "After, net MWh by delivery month", limit), height)
            st.caption("Blue is long and red is short. The hedges cut total risk, which isn't always the same as flattening every month, because some months offset others.")
    if tab_own:
        c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
        own_product = c1.selectbox("Product", list(PRODUCTS), key="own_product")
        own_contract = c2.selectbox("Contract", contract_choices(valuation_date), index=12, key="own_contract")
        own_side = c3.selectbox("Side", ["Sell", "Buy"], key="own_side")
        own_mw = c4.number_input("MW", min_value=0.5, max_value=10000.0, value=5.0, step=0.5, key="own_mw")
        effect = custom_hedge_effect(buckets, history, valuation_date, eur_per_gbp, own_product, own_contract, own_mw if own_side == "Buy" else -own_mw, confidence, prepared=saved_moves())
        if not effect["covered"]:
            need_data_note(f"No saved history for {own_product}, so its effect can't be estimated.")
        else:
            change = effect["after"] - effect["before"]
            r1, r2, r3 = st.columns(3)
            kpi(r1, f"VaR now, {level}", fmt_money(effect["before"], signed=False))
            kpi(r2, "VaR with this trade", fmt_money(effect["after"], signed=False))
            kpi(r3, "Change in VaR", fmt_money(change), "This trade lowers risk" if change < 0 else "This trade adds risk" if change > 0 else "No change", tone="gain" if change < 0 else "loss" if change > 0 else "")
    if tab_trust:
        test = plan["out_of_sample"]
        if test:
            reduction = (1.0 - test["hedged_std"] / test["unhedged_std"]) * 100.0 if test["unhedged_std"] else 0.0
            st.write(f"**Test on unseen days.** A hedge chosen using only the first {test['train_days']} days ({', '.join(test['contracts'])}) was then applied to the last {test['test_days']} days, which it had never seen.")
            t1, t2, t3 = st.columns(3)
            kpi(t1, "Daily volatility on those days", f"{reduction:+.0f}%" if reduction < 0 else f"−{reduction:.0f}%", f"{fmt_money(test['unhedged_std'], signed=False)} down to {fmt_money(test['hedged_std'], signed=False)}" if reduction >= 0 else "The hedge added volatility", tone="gain" if reduction > 0 else "loss")
            kpi(t2, "Worst day unhedged", fmt_money(test["unhedged_worst"]), tone="loss" if test["unhedged_worst"] < 0 else "")
            kpi(t3, "Worst day hedged", fmt_money(test["hedged_worst"]), tone="loss" if test["hedged_worst"] < 0 else "gain")
        else:
            st.write(f"**Test on unseen days.** This needs at least 45 saved days, and you have {plan['observations']}. With no suitable hedge in the earlier days, the test is also skipped.")
        st.markdown(f"**What this uses.** {plan['observations']} saved days and {plan['candidates']} contracts that fit your book. A contract must sit mostly on delivery months you hold, oppose your position, and leave your total position in that product smaller. Sizes minimise day-to-day P&L swings, and a contract is kept only if it cuts VaR by at least 5%.\n\n"
                    "**What it leaves out.** Liquidity and your real bid/offer, margin and collateral, credit limits, trade timing and physical constraints. Spread costs are your own assumptions.\n\n"
                    "**Main risk.** The hedge assumes markets keep moving as they did on the saved days. A short history or a change in market regime can make it wrong, and proxy hedges are the most exposed. It's a starting point for your own judgement, not trading advice.")


# -----------------------------------------------------------------------------
# Curve status, track record and winter outlook views
# -----------------------------------------------------------------------------
def curve_template_csv(today: date) -> bytes:
    """A file to fill in: twelve months and the next two calendar years, with no overlaps."""
    first = _add_months(date(today.year, today.month, 1), 1)
    months = [month_contract_label(_add_months(first, i)) for i in range(12)]
    last_year = _add_months(first, 11).year
    years = [f"Cal-{(last_year + i) % 100:02d}" for i in (1, 2)]
    rows = [{"curve_date": today.isoformat(), "product": product, "contract": contract, "price": ""} for product in CURVE_PRODUCTS for contract in months + years]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def render_curve_status() -> None:
    today = valuation_date
    custom, metadata = st.session_state.get("custom_curves", {}), st.session_state.get("curve_metadata", {})
    rows = []
    for product in CURVE_PRODUCTS:
        status = curve_state(product, custom, metadata, today)
        rows.append({"Curve": product, "Status": {"Demo": "Demo prices", "Current": "Current", "Stale": "Out of date"}.get(status["state"], status["state"]),
                     "Marks dated": f"{status['day']:%d %b %Y}" if status["day"] else "", "Age, days": str(status["age"]) if status["age"] is not None else "",
                     "Source": curve_source_label(product), "Months": str(len(custom[product])) if product in custom else ""})
    table(pd.DataFrame(rows))
    current = sum(1 for r in rows if r["Status"] == "Current")
    st.caption(f"{current} of {len(rows)} curves are current. Marks older than {CURVE_STALE_DAYS} days count as out of date, and those older than {STALE_DAYS} days no longer feed Market Read.")
    st.subheader("Add marks without opening the app")
    st.markdown(f"Save your broker sheet or settlement file in the folder below. New files are imported when you open the app or select Update market data.\n\n`{curve_folder()}`")
    left, right = st.columns(2)
    with left:
        st.download_button("Download a template", data=curve_template_csv(today), file_name="curve_template.csv", mime="text/csv", key="curve_template")
    with right:
        if st.button("Check the folder now", key="curve_folder_check"):
            results = import_curve_folder()
            if results:
                st.rerun()
            st.toast("No new files.", icon=":material/info:")
    st.caption("Columns: product, price, and contract or delivery_month. Add curve_date to set the market date, otherwise a date in the file name or the file's saved date is used. "
               "One file can hold many dates, which also builds Daily P&L and VaR history. CSV works with commas or semicolons, and Excel files need the openpyxl package. "
               "A contract list such as Q1-27 or Cal-27 is split into months using the current curve's shape. Months you don't supply stay unpriced rather than using demo prices.")
    store = desk_store()
    log = store.load_curve_files() if store is not None else pd.DataFrame()
    if not log.empty:
        st.subheader("Files processed")
        table(log.assign(processed_at=log["processed_at"].str[:16].str.replace("T", " "))[["name", "status", "detail", "processed_at"]]
              .rename(columns={"name": "File", "status": "Result", "detail": "Detail", "processed_at": "Processed, UTC"}))


def _track_row(row: dict) -> dict:
    percent = lambda v: None if v is None or pd.isna(v) else f"{v * 100:.0f}%"
    has_range = row.get("low") is not None and pd.notna(row.get("low"))
    return {"Calls": row["calls"], "Independent tests": row["independent"], "Right": percent(row["right"]), "By chance": percent(row["chance"]),
            "Likely range": f"{row['low'] * 100:.0f}% to {row['high'] * 100:.0f}%" if has_range else None,
            "Average move in the call's favour": None if row["average_move"] is None or pd.isna(row["average_move"]) else round(float(row["average_move"]), 2), "Verdict": row["verdict"]}


def _track_figure(market: str, path: dict, reads: pd.DataFrame) -> go.Figure:
    if path["kind"] == "spot":
        prices = path["raw"].tail(250)
        title, lookup = f"{market}: day-ahead base price and the calls made, EUR/MWh", prices.to_dict()
    else:
        fronts = {}
        for day in path["days"]:
            months = sorted(m for m in path["table"][day] if m > date(day.year, day.month, 1))
            if months:
                fronts[day] = path["table"][day][months[0]]
        prices = pd.Series(fronts, dtype=float).tail(250)
        title, lookup = f"{market}: nearest TTF month and the calls made, EUR/MWh", fronts
    fig = go.Figure(go.Scatter(x=list(prices.index), y=prices.to_numpy(), mode="lines", name="Price", line=dict(color=GRAY, width=2)))
    part = reads.loc[reads["market"].eq(market)].copy()
    part["anchor"] = part["price_through"].map(date.fromisoformat)
    part = part.loc[part["anchor"].isin(set(prices.index))]
    styles = (("Recorded", 1, "Bullish call", "triangle-up", GAIN), ("Recorded", -1, "Bearish call", "triangle-down", LOSS),
              ("Replayed", 1, "Bullish, replayed", "triangle-up-open", GAIN), ("Replayed", -1, "Bearish, replayed", "triangle-down-open", LOSS))
    for source, direction, name, symbol, color in styles:
        chosen = part.loc[part["source"].eq(source) & part["score"].map(lambda s: _direction(float(s), CALL_THRESHOLD) == direction)]
        if not chosen.empty:
            fig.add_trace(go.Scatter(x=chosen["anchor"].tolist(), y=[lookup[a] for a in chosen["anchor"]], mode="markers", name=name,
                                     marker=dict(symbol=symbol, size=9, color=color, line=dict(width=1.5, color=color))))
    fig.update_layout(title=title)
    return fig


def render_track_record() -> None:
    with st.spinner("Scoring past reads"):
        ensure_replayed_reads()
        reads = saved_reads()
        paths = track_paths()
        tables = session_memo("track_tables", data_version("reads", "live", "snapshots"), lambda: track_tables(reads, paths))
    recorded = reads.loc[reads["source"].eq("Recorded")] if not reads.empty else reads
    horizon = st.radio("Days ahead", list(TRACK_HORIZONS), index=1, horizontal=True, key="track_horizon", format_func=lambda d: f"{d} working day{'s' if d > 1 else ''} ahead")
    overall = tables["overall"]
    chosen = overall.loc[overall["horizon"].eq(horizon)] if not overall.empty else overall
    def line(row: dict) -> str:
        return (f"<b>{html.escape(row['market'])}:</b> right {row['right'] * 100:.0f}% of the time on {row['calls']} calls, "
                f"against {row['chance'] * 100:.0f}% by chance. {html.escape(row['verdict'])}.")
    started = f"Recording since {date.fromisoformat(recorded['read_date'].min()):%d %b %Y}, with {len(recorded)} reads saved. Saved reads are never rewritten." if not recorded.empty else "Recording starts with your first update of current data."
    rows = chosen.to_dict("records") if not chosen.empty else []
    scored = lambda source: [r for r in rows if r["source"] == source and r["right"] is not None and pd.notna(r["right"])]
    live, replayed = scored("Recorded"), scored("Replayed")
    parts = [f"<p>{html.escape(started)}</p>"]
    parts += [f"<p>{line(r)}</p>" for r in live]
    if not live:
        parts.append(f'<p class="soft">Recorded reads can be scored once {horizon} working day{"s have" if horizon > 1 else " has"} passed and new prices are in.</p>')
    if replayed:
        parts.append("<p>On past data, without forecasts or curve signals:</p>" + "".join(f"<p>{line(r)}</p>" for r in replayed))
    if not rows:
        parts.append('<p class="soft">Select Update market data to load prices. Past reads are then rebuilt and scored here.</p>')
    with st.container(key="hero"):
        st.markdown(f'<div class="desk-note">{"".join(parts)}</div>', unsafe_allow_html=True)
    if not paths["European gas"]["days"]:
        need_data_note("European gas can't be scored yet because it needs real TTF prices to compare against. Add dated TTF marks in Forward Curves and the gas reads will be scored too.")
    if chosen.empty:
        return
    shown = chosen.loc[chosen["calls"].gt(0)]
    if not shown.empty:
        table(pd.DataFrame([{"Market": r["market"], "Reads": r["source"], **_track_row(r), "Balanced, no call": r["balanced"]} for r in shown.to_dict("records")]))
    st.caption("Likely range is a 90% range for the true hit rate, based on independent tests. A verdict needs at least "
               f"{TRACK_MIN_INDEPENDENT} independent tests, and the range must clear the chance rate.")
    options = [m for m in paths if path_anchors(paths[m]) and not reads.empty and reads["market"].eq(m).any()]
    if options:
        market = st.selectbox("Market", options, key="track_market")
        chart(_track_figure(market, paths[market], reads), 380)
        signals = tables["signals"]
        part = signals.loc[signals["horizon"].eq(horizon) & signals["market"].eq(market) & signals["calls"].gt(0)] if not signals.empty else signals
        if not part.empty:
            st.subheader("Which signals have worked")
            table(pd.DataFrame([{"Signal": r["signal"], "Reads": r["source"], **_track_row(r)} for r in part.sort_values(["source", "signal"]).to_dict("records")]))
        levels = tables["confidence"]
        part = levels.loc[levels["horizon"].eq(horizon) & levels["market"].eq(market) & levels["calls"].gt(0)] if not levels.empty else levels
        if not part.empty:
            st.subheader("Does higher confidence mean better calls")
            table(pd.DataFrame([{"Confidence": r["confidence"], "Reads": r["source"], **_track_row(r)} for r in part.sort_values(["source", "confidence"]).to_dict("records")]))
    if not recorded.empty and recorded["read_date"].min() < READ_METHOD_DATE.isoformat():
        st.caption(f"The read's method changed on {READ_METHOD_DATE:%d %b %Y}, when related signals began sharing a capped weight and French nuclear output was added. Recorded reads before that date used the earlier method.")
    with st.expander("How the scoring works"):
        st.markdown("**Recorded** reads were saved on the day, with all the evidence then available. They are the real test.\n\n"
                    "**Replayed** reads are rebuilt from past data using only what was known at the time. They leave out forecasts and curve signals, so they test only part of the method.\n\n"
                    "**A call is right** when the price moves its way. Bullish and bearish count as calls, and a balanced read makes no call.\n\n"
                    "**Power** compares day-ahead base on the later day with its level when the read was made, with the usual weekday pattern removed. "
                    "**Gas** compares the same TTF delivery month at both dates, using real marks only.\n\n"
                    "**By chance** is how often you'd be right by making the same mix of calls at random in this period, which allows for a market that mostly rose or fell.\n\n"
                    "**Independent tests** count calls whose outcome windows don't overlap, because reads a day apart are mostly watching the same price move. Consecutive reads also tend to agree, so even this can flatter the result.")
    st.download_button("Download all saved reads", data=reads.drop(columns=["signals"]).to_csv(index=False).encode("utf-8"), file_name="meridian_read_log.csv", mime="text/csv", key="download_reads")


OUTLOOK_COLORS = (BLUE, AMBER, TEAL)


def _outlook_figure(outlook: dict) -> go.Figure:
    fig = go.Figure()
    band = outlook["band"]
    if not band.empty:
        fig.add_trace(go.Scatter(x=band["day"], y=band["high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=band["day"], y=band["low"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(169,190,194,0.20)", name="Five-year range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=band["day"], y=band["average"], mode="lines", name="Five-year average", line=dict(color=GRAY, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=outlook["actual"].index, y=outlook["actual"].to_numpy(), mode="lines", name="This winter", line=dict(color=TEXT, width=2.5)))
    start_day = pd.Timestamp(outlook["start"])
    for scenario, color in zip(outlook["scenarios"], OUTLOOK_COLORS):
        x = [start_day] + list(outlook["days"])
        y = [outlook["start_fill"]] + scenario["path"].tolist()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=scenario["name"], line=dict(color=color, width=2.5, dash="dash" if scenario["name"] != outlook["scenarios"][outlook["central"]]["name"] else "solid")))
        fig.add_annotation(x=x[-1], y=y[-1], text=f"{scenario['name']} {scenario['end']:.0f}%", showarrow=False, xanchor="left", xshift=8, font=dict(color=TEXT, size=12))
    fig.update_layout(title="EU gas storage this winter, % full", margin=dict(r=90))
    fig.update_yaxes(title="", rangemode="tozero")
    return fig


def render_winter_outlook() -> None:
    st.subheader("Where EU storage could end this winter")
    outlook = winter_outlook()
    if not outlook["ready"]:
        need_data_note(outlook["reason"])
        if not outlook.get("off_season"):
            update_button("outlook_update")
        return
    scenarios = outlook["scenarios"]
    central = scenarios[outlook["central"]]
    band_end = outlook["band_end"]
    comparison = lambda level: f", {versus_range(level, band_end)}" if band_end else ""
    weather_based = outlook["basis"] == "weather"
    first = (f"With normal weather, EU storage ends March at about <b>{central['end']:.0f}%</b>{comparison(central['end'])}."
             if weather_based else f"If storage falls as it has on average, it ends March at about <b>{central['end']:.0f}%</b>{comparison(central['end'])}.")
    if weather_based:
        second = (f"The coldest winter of the past {len(outlook['years'])} would take it to <b>{scenarios[0]['end']:.0f}%</b>{comparison(scenarios[0]['end'])}, "
                  f"and the mildest to <b>{scenarios[2]['end']:.0f}%</b>{comparison(scenarios[2]['end'])}.")
    else:
        second = "This repeats past withdrawals because the weather history isn't loaded, so it doesn't separate weather from anything else. Select Update market data to load it."
    if outlook["error"] is not None and weather_based:
        beat = outlook["error"] < outlook["average_error"]
        third = (f"Tested on {len(outlook['backtest'])} past winters with their actual weather, the method missed the March level by {outlook['error']:.1f} points on average, "
                 f"against {outlook['average_error']:.1f} for assuming an average winter. "
                 + ("" if beat else "It hasn't beaten that simple average, so rely on the range more than the central number."))
    else:
        third = ""
    with st.container(key="hero"):
        paragraph = f'<p class="soft">{html.escape(third)}</p>' if third else ""
        st.markdown(f'<div class="desk-note"><p>{first} {second}</p>{paragraph}</div>', unsafe_allow_html=True)
    lowest = outlook.get("lowest_past_level")
    if lowest is not None and outlook["start_fill"] < lowest:
        st.warning(f"Storage is at {outlook['start_fill']:.1f}%, below every past winter in the data at this date (lowest {lowest:.1f}%). The method is working outside its experience, and it can't see withdrawals slowing because of high prices or demand cuts. Treat the March levels as a warning sign, not a precise figure.")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"Now, {outlook['start']:%d %b}", f"{outlook['start_fill']:.1f}%", "Latest published gas day")
    for column, scenario, color in zip((c2, c3, c4), scenarios, OUTLOOK_COLORS):
        kpi(column, f"{scenario['name']}, 31 March", f"{scenario['end']:.0f}%", scenario["note"], accent=color)
    chart(_outlook_figure(outlook), 440)
    if outlook["forecast_days"]:
        st.caption(f"The first {outlook['forecast_days']} days use the current weather forecast in every scenario. After that, each scenario uses past weather on the same calendar days.")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("How the method did on past winters")
        if not outlook["backtest"].empty:
            shown = outlook["backtest"].round(1).rename(columns={"Fill at this date": "Start", "Actual end": "Actual", "Model end": "Model", "Model miss": "Miss", "Average-winter miss": "Miss if average"}).drop(columns=["Average-winter end"])
            table(shown, height=37 * (len(shown) + 1) + 3)
            st.caption("Fill levels in %. Each winter was predicted by a model fitted without it, using that winter's actual weather. This tests the link between weather and storage, not a weather forecast, so real errors will be larger. "
                       "It's also a small sample, and later winters helped predict earlier ones.")
        else:
            st.caption("This needs storage and weather history for at least three past winters.")
    with right:
        if not outlook["years"].empty:
            st.subheader("March end under each past winter's weather")
            table(outlook["years"].round(1), height=37 * (len(outlook["years"]) + 1) + 3)
    with st.expander("How this projection works"):
        model = outlook["model"]
        if model:
            st.markdown(f"Each winter day's storage change is estimated from the month, heating demand, weekends and how full storage is. The result is then run forward one day at a time.\n\n"
                        f"- Each extra heating degree day changes storage by **{model['hdd_effect']:+.3f} points** that day.\n"
                        f"- A weekend day changes it by **{model['weekend_effect']:+.3f} points**.\n"
                        + (f"- Each extra point of fill changes the daily move by **{model['fill_effect']:+.4f} points**, so withdrawals slow as storage empties.\n" if model["fill_effect"] < 0 else "- The data shows no sign of withdrawals slowing as storage empties, so that effect is left out. In reality they do slow at low levels, so the coldest case may be too low.\n") +
                                                f"- These explain **{model['r2'] * 100:.0f}%** of daily changes across {model['n']:,} days in {len(model['winters'])} winters. The last three winters count fully and older ones count half.\n"
                        f"- Storage has lately run **{outlook['bias']:+.3f} points a day** against the model. That gap is carried forward and halves every {BIAS_HALF_LIFE_DAYS:.0f} days. Without it, normal weather would end March at **{outlook['end_without_recent_gap']:.0f}%**.\n\n")
        st.markdown("Heating demand averages reference cities in Germany, the Netherlands, France and Belgium, weighted by each country's gas use. It's still a rough stand-in, because it leaves out the rest of the EU, including Italy and Spain. "
                    "The projection doesn't know about supply outages, LNG arrivals, prices or policy, and the crisis winters of 2021-22 and 2022-23 sit in the sample. Treat it as a guide to the size of the weather risk, not as a forecast.")


# -----------------------------------------------------------------------------
# Next-day estimate, P&L drivers, crisis replay, VaR check and morning brief
# -----------------------------------------------------------------------------
def zone_estimate(zone: str) -> dict:
    return session_memo(f"estimate|{zone}", data_version("live"),
                        lambda: next_day_estimate(zone, zone_prices(zone), _live_dataset_frame("weather_power_history"), _live_dataset_frame("weather_daily"),
                                                  _live_dataset_frame(f"entsoe_load_forecast|{zone}"), _live_dataset_frame(f"entsoe_renewable_forecast|{zone}"),
                                                  _live_dataset_frame("zone_weather_history"), _live_dataset_frame("zone_weather_forecast")))


def record_price_estimates() -> None:
    rows = []
    store = desk_store()
    for zone in _live_variants("entsoe_price"):
        if zone in ZONE_WEATHER:
            base = daily_base_prices(zone_prices(zone))
            target = (base.index[-1] + pd.Timedelta(days=1)).date().isoformat() if len(base) else None
            held = store.load_price_estimates(zone) if store is not None else pd.DataFrame()
            if target is None or (not held.empty and held["delivery_date"].eq(target).any()):
                continue   # the first estimate for a day stands, so there's nothing to compute
            result = zone_estimate(zone)
            if result["ready"]:
                rows.append((result["day"].isoformat(), zone, result["estimate"], result["low"], result["high"], valuation_date.isoformat(), result["method"]))
    if rows:
        _save(lambda s: s.save_price_estimates(rows))
        bump("estimates")


def render_next_day(zones: Sequence[str]) -> None:
    usable = [z for z in zones if z in ZONE_WEATHER]
    if not usable:
        need_data_note("Choose a zone with a weather reference city, such as France or Germany-Luxembourg.")
        return
    zone = st.selectbox("Zone", usable, key="estimate_zone") if len(usable) > 1 else usable[0]
    result = zone_estimate(zone)
    if not result["ready"]:
        need_data_note(result["reason"])
        update_button("estimate_update")
        return
    beat = result["skill"] > 0
    gap_word = "above" if result["latest_gap"] > 0 else "below"
    weather_words = f"the weather forecast at {result['weather_basis']}" if "points" in result["weather_basis"] else f"the weather forecast for {result['weather_basis'] or 'one city'}"
    basis = ("It uses ENTSO-E's published load, wind and solar forecasts for that day." if result["method"] != "Weather" else
             f"It uses {weather_words}, which has tested better here than ENTSO-E's forecasts." if result["target_has_forecast"] and result["comparison"] else
             f"It uses {weather_words}, because ENTSO-E's wind and solar forecast is usually published by 18:00 the day before delivery, after the auction.")
    euros = lambda value: f"{'+' if value >= 0 else '-'}€{abs(value):.2f}"
    shown_gap = round(result["latest_actual"]) - round(result["latest_estimate"])   # agrees with the rounded prices shown
    against = (f"in line with the €{result['latest_estimate']:.0f} suggested by" if shown_gap == 0 else
               f"€{abs(shown_gap):.0f} {gap_word} the €{result['latest_estimate']:.0f} suggested by")
    check_basis = "ENTSO-E's forecasts for that day" if result["check_method"] != "Weather" else "the weather"
    held = (f" The range is built the same way each day and held the price on <b>{result['held'] * 100:.0f}%</b> of {result['held_days']} tested days, against the 80% it aims for."
            if result["held"] is not None else " There are too few tested days yet to say how often the range holds.")
    trend = (" Recent days have been harder than usual, so the range is wider." if result["recent_error"] > 1.25 * result["error"] else
             " Recent days have been calmer than usual, so the range is narrower." if result["recent_error"] < 0.8 * result["error"] else "")
    with st.container(key="hero"):
        st.markdown(
            f'<div class="desk-note"><p>Base price for {result["day"]:%a %d %b} in {html.escape(zone)}: about <b>€{result["estimate"]:.0f}/MWh</b>, '
            f'with an 80% chance of landing between <b>€{result["low"]:.0f} and €{result["high"]:.0f}</b>. {html.escape(basis)}</p>'
            f'<p>The latest published day, {result["latest_day"]:%a %d %b}, came in at €{result["latest_actual"]:.0f}, {against} {check_basis}. '
            f'{"That is an unusually large gap, so check for something the method can&#39;t see." if result["unusual"] else "That is within the usual range of error."}</p>'
            f'<p class="soft">Over the last {result["n"]} days, each estimated only from earlier days, it missed by €{result["error"]:.1f} on average, '
            f'{abs(result["skill"]) * 100:.0f}% {"less" if beat else "more"} than the best simple method (€{min(result["error_same_type"], result["error_last_week"], result["error_yesterday"]):.1f}).{held}{trend} '
            f'{"" if beat else "It has not beaten the simple methods, so don&#39;t rely on it over them."}</p></div>', unsafe_allow_html=True)
    test = result["test"].tail(90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["day"], y=test["actual"], mode="lines", name="Published price", line=dict(color=TEXT, width=2)))
    fig.add_trace(go.Scatter(x=test["day"], y=test["estimate"], mode="lines", name="Estimate made the day before", line=dict(color=AMBER, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[pd.Timestamp(result["day"])], y=[result["estimate"]], mode="markers", name="Next day, with 80% range", marker=dict(color=AMBER, size=11, symbol="diamond"),
                             error_y=dict(type="data", symmetric=False, array=[result["high"] - result["estimate"]], arrayminus=[result["estimate"] - result["low"]], color=AMBER, thickness=2)))
    fig.update_layout(title=f"{zone}: day-ahead base price, EUR/MWh")
    chart(fig, 380)
    comparison = result["comparison"]
    st.subheader("How the methods compare")
    rows = [{"Method": f"Weather at {result['weather_basis']}" if "points" in result["weather_basis"] else f"Weather in {result['weather_basis'] or 'one city'}", "Average miss, EUR/MWh": round(comparison["weather"] if comparison else result["error"], 2)}]
    if comparison:
        rows.insert(0, {"Method": "ENTSO-E forecasts", "Average miss, EUR/MWh": round(comparison["forecasts"], 2)})
    source = comparison or {"same_type": result["error_same_type"], "last_week": result["error_last_week"], "yesterday": result["error_yesterday"]}
    rows += [{"Method": "Last day of the same kind", "Average miss, EUR/MWh": round(source["same_type"], 2)},
             {"Method": "Same weekday last week", "Average miss, EUR/MWh": round(source["last_week"], 2)}, {"Method": "Previous day", "Average miss, EUR/MWh": round(source["yesterday"], 2)}]
    table(pd.DataFrame(rows))
    if comparison:
        st.caption(f"On the same {comparison['days']} days. The ENTSO-E method isn't fully comparable: its wind and solar forecasts are usually published after the auction, so it shows how good the forecast information is, not what you could have estimated beforehand.")
    else:
        st.caption(f"Over {result['n']} days. A day of the same kind means a working day, a Saturday, or a Sunday or public holiday. ENTSO-E's forecasts cover {result['forecast_days']} days so far, and comparing with them needs at least {ESTIMATE_MIN_DAYS + 30}.")
    store = desk_store()
    log = store.load_price_estimates(zone) if store is not None else pd.DataFrame()
    live = score_recorded_estimates(log, daily_base_prices(zone_prices(zone)))
    if live["n"]:
        st.caption(f"Estimates saved on the day, before prices were known: {live['n']} scored so far, missing by €{live['error']:.1f} on average, against €{live['error_last_week']:.1f} for the same weekday last week."
                   + (f" Their ranges held the price on {live['held'] * 100:.0f}% of {live['held_n']} days." if live["held"] is not None else "") + " This is the truest test, as nothing here could see the outcome.")
    else:
        st.caption("Each estimate is also saved on the day, before the price is known. They're scored here once the prices are published, which is the truest test of the method.")
    with st.expander("How this estimate works"):
        effects = result["forecast_effects"]
        grid = "points" in result["weather_basis"]
        st.markdown("Each method is a regression on up to a year of days, where recent days count more: a day four months back counts half as much as yesterday. "
                    "It starts from the last day of the same kind (a working day, a Saturday, or a Sunday or public holiday), the last working day, the previous day, the last 7 days' average and the same weekday last week, "
                    "so a Monday starts from Friday's price, not Sunday's, and national public holidays are priced like Sundays.\n\n"
                    + (f"**Weather method.** It adds heating demand, wind at 100 m and solar radiation averaged over {len(ZONE_WEATHER_POINTS.get(zone, ()))} points spread across the zone, for the day and the day before. "
                       "On real German prices and pre-auction weather forecasts, a dozen points instead of one city cut the average miss by about a sixth. " if grid else
                       f"**Weather method.** It adds heating demand, wind and solar in {result['weather_basis'] or 'one reference city'}, for the day and the day before. ")
                    + f"10 km/h more wind changes the price by **{euros(result['wind_effect'] * 10)}**, one more heating degree day by **{euros(result['hdd_effect'])}**, and 10 W/m² more solar radiation by **{euros(result['solar_effect'] * 10)}**. "
                    "Its test uses the weather that actually happened, because past forecasts aren't stored, so real errors run a little larger. The saved estimates above don't have that advantage.\n\n"
                    "**The range.** It comes from this method's own misses on the tested days, scaled by how large recent misses were, so it widens when the market turns volatile and narrows when it calms. "
                    "It aims to hold the price on 8 days in 10. After a sudden jump in volatility it takes a week or two to catch up.\n\n"
                    "**ENTSO-E method.** It adds the system operators' day-ahead forecasts of load, wind and solar for the whole zone, and its test uses those forecasts rather than what happened. "
                    + (f"1 GW more forecast load changes the price by **{euros(effects['load'])}**, 1 GW more wind by **{euros(effects['wind'])}**, and 1 GW more solar by **{euros(effects['solar'])}**. " if effects else "")
                    + "The load forecast is due two hours before the auction deadline, but the wind and solar forecast is only due by 18:00 the day before delivery. "
                    "So it usually can't set the estimate before the auction. Its main use is checking whether a published price looks high or low against those forecasts. It sets the estimate only when the forecast already exists and it has tested at least as well as the weather method.\n\n"
                    f"ENTSO-E's forecasts cover **{result['forecast_days']} days** for this zone so far, and some operators may revise them after first publishing.\n\n"
                    "Neither method knows about gas or carbon prices, plant outages, interconnector limits or the auction itself. A sharp move in gas between one day and the next is the most common reason for a large miss.")


def render_pnl_drivers(step: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    drivers = pnl_drivers(step, factor)
    with st.container(key="hero"):
        st.markdown(f'<div class="desk-note"><p>{html.escape(pnl_sentence(drivers))}</p></div>', unsafe_allow_html=True)
    return drivers


def render_driver_table(drivers: pd.DataFrame) -> None:
    if drivers.empty:
        return
    st.subheader("What drove it, by delivery month")
    sources = curve_sources()
    drivers = drivers.assign(prices=[{SOURCE_DEMO: "Demo"}.get(sources.get(p, {}).get(m, ""), sources.get(p, {}).get(m, "")) for p, m in zip(drivers["product"], drivers["delivery_month"])])
    shown = drivers.assign(delivery_month=drivers["delivery_month"].map(month_contract_label)).rename(columns={"prices": "Prices now",
        "product": "Product", "delivery_month": "Month", "net_mwh": "Net MWh", "previous_mark": "Mark before", "current_mark": "Mark now", "mark_change": "Change",
        "price_move": "Price move, EUR", "new_trades": "New trades, EUR", "first_marks": "First marks, EUR", "total": "Total, EUR"})
    table(shown.round({"Net MWh": 0, "Mark before": 2, "Mark now": 2, "Change": 2, "Price move, EUR": 0, "New trades, EUR": 0, "First marks, EUR": 0, "Total, EUR": 0}))


def render_crisis_replay(buckets: pd.DataFrame, eur_per_gbp: float, outcome: dict) -> None:
    c1, c2 = st.columns(2, vertical_alignment="bottom")
    mode = c1.radio("Apply the move as", ["The same percentage", "The same EUR/MWh"], horizontal=True, key="crisis_mode")
    far = c2.slider("How much far-dated months move, as a share of the nearest month", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="crisis_far")
    same_percent = mode == "The same percentage"
    detail = crisis_replay(buckets, curves, _add_months(valuation_date, 1), same_percent, float(far), eur_per_gbp)
    totals = detail.groupby("episode", sort=False)["pnl_eur"].sum()
    var99 = outcome.get("var", {}).get(0.99)
    worst_name = totals.idxmin()
    multiple = f", or {abs(totals.min()) / var99:.0f} times the 99% one-day VaR" if var99 and totals.min() < 0 else ""
    with st.container(key="hero"):
        st.markdown(f'<div class="desk-note"><p>The worst of these for today\'s book is <b>{html.escape(worst_name)}</b>: <span style="white-space:nowrap">{fmt_money(float(totals.min()))}</span>{multiple}.</p></div>', unsafe_allow_html=True)
    marks_note = price_warning(sorted(buckets["product"].unique()) + ["TTF Natural Gas"])
    if marks_note:
        st.warning(marks_note)
    fig = go.Figure(go.Bar(x=totals.to_numpy(), y=list(totals.index), orientation="h", marker=dict(color=[GAIN if v >= 0 else LOSS for v in totals]), hovertemplate="%{y}: €%{x:,.0f}<extra></extra>"))
    fig.update_layout(title="P&L of today's book in each episode, EUR", hovermode="closest", yaxis_title="", margin=dict(l=240))
    chart(fig, 280)
    pivot = detail.pivot_table(index="episode", columns="product", values="pnl_eur", aggfunc="sum", sort=False).round(0)
    pivot["Total"] = pivot.sum(axis=1)
    table(pivot.reset_index().rename(columns={"episode": "Episode"}))
    stretches = [s for s in (worst_stretch(outcome.get("pnl"), n) for n in (5, 10)) if s]
    if stretches:
        st.subheader("Worst stretches in your own saved marks")
        table(pd.DataFrame([{"Length": f"{s['days']} days", "P&L of today's book, EUR": round(s["pnl_eur"]), "From": s["start"], "To": s["end"]} for s in stretches]))
        if saved_snapshots()["source"].eq(SYNTHETIC_HISTORY_SOURCE).any():
            st.warning("Part of this history is synthetic demo data, so this table says nothing about real risk.")
    with st.expander("What each episode applies, and the limits"):
        for episode in CRISIS_EPISODES:
            st.markdown(f"**{episode['name']}.** {episode['what']} Figures are rounded and come from public reports: {'; '.join(episode['sources'])}.")
        st.markdown("Only the nearest-month gas move is from those reports. **These parts are assumptions:** far-dated months use the share you set, fading over about six months. "
                    f"NBP moves with TTF. Power is implied through a gas plant at {DEFAULT_SPREAD_PARAMS['gas_efficiency'] * 100:.0f}% efficiency, not taken from 2022 power prices, and real power moved differently, especially in France.\n\n"
                    "Moves are applied at once, with no hedging, margin calls or lost liquidity, all of which made 2022 harder than this shows. "
                    "Same EUR/MWh on today's lower prices is far harsher than the same percentage, and gas is held above EUR 1/MWh.")


def render_var_check(outcome: dict) -> None:
    level = st.radio("Confidence level", [0.95, 0.99], index=1, horizontal=True, key="var_check_level", format_func=lambda v: f"{v * 100:.0f}%")
    check = var_backtest(outcome.get("pnl"), float(level))
    if not check["tests"]:
        need_data_note(f"This needs more than {VAR_CHECK_MIN_HISTORY} days of saved curve moves, and {outcome.get('observations', 0)} are saved so far.")
        return
    if saved_snapshots()["source"].isin(DEMO_HISTORY_SOURCES).any():
        st.warning("Part of this history uses demo or synthetic prices, so this result only shows how the check works. It says nothing about real risk.")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Days tested", f"{check['tests']}")
    kpi(k2, "Losses beyond VaR", f"{check['exceptions']}", f"{check['expected']:.1f} expected")
    kpi(k3, "Kupiec p-value", f"{check['p_value']:.2f}", "Below 0.05 rejects VaR" if check["tests"] >= VAR_CHECK_MIN_TESTS else "Not judged on so few days")
    kpi(k4, "Result", check["verdict"], f"Basel zone: {check['zone']}" if check["zone"] else "")
    data = check["table"]
    fig = go.Figure(go.Bar(x=data["date"], y=data["pnl_eur"], name="Daily P&L of today's book", marker=dict(color=GRAY)))
    fig.add_trace(go.Scatter(x=data["date"], y=-data["var_eur"], mode="lines", name="VaR that day", line=dict(color=AMBER, width=2)))
    hits = data.loc[data["exception"]]
    fig.add_trace(go.Scatter(x=hits["date"], y=hits["pnl_eur"], mode="markers", name="Loss beyond VaR", marker=dict(color=LOSS, size=9, symbol="x")))
    fig.update_layout(title="Daily P&L against the VaR known the day before, EUR")
    chart(fig, 380)
    if check["pairs"]:
        st.caption(f"Losses beyond VaR came on consecutive days {check['pairs']} time{'s' if check['pairs'] > 1 else ''}. Clustering means VaR is slow to react when markets turn, even if the total count looks fine.")
    with st.expander("How this check works"):
        st.markdown(f"Each day's VaR uses only the {VAR_CHECK_WINDOW} days before it, with at least {VAR_CHECK_MIN_HISTORY}. That day's result is then compared with it. "
                    "This applies today's book to past price moves, so it tests the method rather than your past trading.\n\n"
                    "Kupiec's test asks whether the number of losses beyond VaR is plausible for the confidence level. It has little power on short samples, which is why fewer than "
                    f"{VAR_CHECK_MIN_TESTS} days gets no verdict. The Basel zone counts the last 250 days at 99%: green up to 4, yellow 5 to 9, red 10 or more.")


BRIEF_METRICS = ("EU storage fill", "Winter outlook, normal weather", "TTF nearest month", "French day-ahead base", "German day-ahead base", "Daily P&L", "1-day VaR, 95%")


def _base_pair(zone: str) -> tuple[Optional[float], Optional[float], Optional[date]]:
    base = daily_base_prices(zone_prices(zone))
    if len(base) < 2:
        return None, None, None
    return float(base.iloc[-1]), float(base.iloc[-2]), base.index[-1].date()


def brief_metrics() -> dict:
    metrics: dict = {}
    fill = _storage_daily(_live_dataset_frame("gie_agsi_history"))
    if len(fill) >= 2:
        metrics["EU storage fill"] = {"now": float(fill["fill"].iloc[-1]), "before": float(fill["fill"].iloc[-2]), "unit": "%", "as_of": fill["day"].iloc[-1].date()}
    outlook = winter_outlook()
    if outlook.get("ready") and outlook.get("basis") == "weather":
        metrics["Winter outlook, normal weather"] = {"now": outlook["scenarios"][outlook["central"]]["end"], "before": None, "unit": "%", "as_of": outlook["start"]}
    marks = marks_path(saved_snapshots(), "TTF Natural Gas")
    if len(marks["days"]) >= 2:
        last, previous = marks["days"][-1], marks["days"][-2]
        months = sorted(m for m in marks["table"][last] if m > date(last.year, last.month, 1) and m in marks["table"][previous])
        if months:
            metrics["TTF nearest month"] = {"now": marks["table"][last][months[0]], "before": marks["table"][previous][months[0]], "unit": "EUR/MWh", "as_of": last}
    for name, zone in (("French day-ahead base", "France"), ("German day-ahead base", "Germany-Luxembourg")):
        now, before, day = _base_pair(zone)
        if now is not None:
            metrics[name] = {"now": now, "before": before, "unit": "EUR/MWh", "as_of": day}
    lookups, buckets = saved_lookups(), executed_buckets()
    days = sorted(lookups)
    if len(days) >= 2 and not buckets.empty:
        series = session_memo("pnl_series", (data_version("trades", "snapshots"), fx_rate("eur_per_gbp")), lambda: pnl_history(buckets, lookups, fx_rate("eur_per_gbp")))
        if len(series) >= 1:
            metrics["Daily P&L"] = {"now": float(series["pnl_eur"].iloc[-1]), "before": float(series["pnl_eur"].iloc[-2]) if len(series) >= 2 else None, "unit": "EUR", "as_of": days[-1]}
        outcome = session_memo("var", (data_version("trades", "snapshots"), fx_rate("eur_per_gbp")), lambda: historical_var(buckets, saved_snapshots(), fx_rate("eur_per_gbp"), prepared=saved_moves()))
        if outcome["observations"] >= 20:
            metrics["1-day VaR, 95%"] = {"now": outcome["var"][0.95], "before": None, "unit": "EUR at risk", "as_of": days[-1]}
    return metrics


def _metric_text(value: Optional[float], unit: str) -> str:
    if value is None:
        return ""
    return {"%": f"{value:.1f}%", "EUR": fmt_money(value), "EUR at risk": fmt_money(value, signed=False), "EUR/MWh": f"€{value:.2f}"}.get(unit, f"{value:,.2f}")


def build_brief() -> dict:
    """Everything the morning brief says, used by both the page and the PDF."""
    metrics = brief_metrics()
    store = desk_store()
    rules = store.load_alerts() if store is not None else pd.DataFrame(columns=["metric", "condition", "level"])
    changes = []
    for name in BRIEF_METRICS:
        reading = metrics.get(name)
        if reading:
            change = reading["now"] - reading["before"] if reading.get("before") is not None else None
            changes.append({"Item": name, "Now": _metric_text(reading["now"], reading["unit"]),
                            "Before": _metric_text(reading.get("before"), reading["unit"]),
                            "Change": ("" if change is None else f"{change:+.1f} pts" if reading["unit"] == "%" else fmt_money(change) if reading["unit"] == "EUR" else f"{change:+.2f}"),
                            "Date": f"{reading['as_of']:%d %b}"})
    held = saved_reads()
    earlier = held.loc[held["source"].eq("Recorded") & held["read_date"].lt(valuation_date.isoformat())] if not held.empty else held
    reads = []
    for market, read in market_reads().items():
        previous = earlier.loc[earlier["market"].eq(market)] if not earlier.empty else earlier
        before = previous["reading"].iloc[-1] if not previous.empty else ""
        summary = read["summary"]
        reads.append({"Market": market, "Read": summary["reading"], "Confidence": summary["confidence"], "Previous": before or "None saved", "Changed": bool(before) and before != summary["reading"]})
    pnl_text, drivers = "", pd.DataFrame()
    lookups, buckets = saved_lookups(), executed_buckets()
    days = sorted(lookups)
    if len(days) >= 2 and not buckets.empty:
        step = pnl_explain(buckets, lookups[days[-2]], lookups[days[-1]], days[-2], days[-1])
        if not step.empty:
            drivers = pnl_drivers(step, step["currency"].map({"EUR": 1.0, "GBP": fx_rate("eur_per_gbp")}))
            pnl_text = pnl_sentence(drivers)
    estimates = []
    for zone in ("France", "Germany-Luxembourg"):
        if zone in _live_variants("entsoe_price"):
            result = zone_estimate(zone)
            if result["ready"]:
                estimates.append({"Zone": zone, "Day": f"{result['day']:%a %d %b}", "Estimate": f"€{result['estimate']:.0f}", "80% range": f"€{result['low']:.0f} to €{result['high']:.0f}", "Latest day": "Unusually large gap" if result["unusual"] else "Within usual error"})
    outlook = winter_outlook()
    outlook_text = ""
    if outlook.get("ready") and outlook.get("basis") == "weather":
        scenarios = outlook["scenarios"]
        outlook_text = f"EU storage ends March at about {scenarios[1]['end']:.0f}% with normal weather, {scenarios[0]['end']:.0f}% in the coldest past winter and {scenarios[2]['end']:.0f}% in the mildest."
    warnings_ = []
    held_products = sorted(buckets["product"].unique()) if not buckets.empty else []
    marks_note = price_warning(held_products)
    if marks_note:
        warnings_.append(marks_note)
    if not metrics.get("EU storage fill"):
        warnings_.append("There's no storage data yet. Select Update market data.")
    priced = list(dict.fromkeys(held_products + ["TTF Natural Gas", "French Power Base", "German Power Base"]))
    return {"prices_used": prices_used_lines(priced), "book_note": src_book(), "alerts": threshold_alerts(rules, metrics), "changes": changes, "reads": reads, "pnl_text": pnl_text, "drivers": drivers, "estimates": estimates,
            "outlook": outlook_text, "warnings": warnings_}


def brief_pdf(brief: dict) -> bytes:
    blocks: list[tuple] = [("title", "Morning brief"), ("subtitle", f"Meridian, {valuation_date:%A %d %B %Y}. Screening analysis, not investment advice.")]
    if brief["warnings"]:
        blocks += [("note", w) for w in brief["warnings"]]
    if brief["alerts"]:
        blocks.append(("heading", "Thresholds crossed"))
        blocks += [("text", f"{a['metric']}: {_metric_text(a['now'], a['unit'])}, {a['condition'].lower()} your level of {_metric_text(a['level'], a['unit'])}" + (". New today." if a["new"] else ".")) for a in brief["alerts"]]
    if brief["changes"]:
        blocks.append(("heading", "What changed"))
        columns = ["Item", "Now", "Before", "Change", "Date"]
        blocks.append(("table", (columns, [[row[c] for c in columns] for row in brief["changes"]], (0.40, 0.17, 0.17, 0.15, 0.11))))
    if brief["reads"]:
        blocks.append(("heading", "Market read"))
        columns = ["Market", "Read", "Confidence", "Previous"]
        blocks.append(("table", (columns, [[row[c] for c in columns] for row in brief["reads"]], (0.28, 0.28, 0.18, 0.26))))
    if brief["pnl_text"]:
        blocks += [("heading", "P&L"), ("text", brief["pnl_text"])]
    if brief["estimates"]:
        blocks.append(("heading", "Next-day power price estimate"))
        columns = ["Zone", "Day", "Estimate", "80% range", "Latest day"]
        blocks.append(("table", (columns, [[row[c] for c in columns] for row in brief["estimates"]], (0.28, 0.16, 0.13, 0.19, 0.24))))
    if brief["outlook"]:
        blocks += [("heading", "Winter storage"), ("text", brief["outlook"])]
    blocks.append(("heading", "Prices used"))
    blocks += [("text", f"{product}: {detail}.") for product, detail in brief["prices_used"]]
    blocks += [("text", f"Book: {text}.") for state, text in brief["book_note"] if state == "demo"]
    blocks.append(("note", "Day-ahead, storage, weather and FX figures come from ENTSO-E, GIE, Open-Meteo and the ECB. Reads, estimates and projections are simple models with the limits shown in the app. Check their track record before relying on them."))
    return simple_pdf(blocks, "Meridian morning brief")


def page_morning_brief() -> None:
    title_block("Morning brief", f"{valuation_date:%A %d %B %Y}. What changed since the last update, on one page you can save as a PDF.")
    brief = build_brief()
    left, right = st.columns([1, 4], vertical_alignment="center")
    with left:
        update_button("brief_update")
    with right:
        st.download_button("Download as PDF", data=brief_pdf(brief), file_name=f"meridian_brief_{valuation_date.isoformat()}.pdf", mime="application/pdf", key="brief_pdf")
    for note in brief["warnings"]:
        st.warning(note)
    for alert in brief["alerts"]:
        st.error(f"{alert['metric']}: {_metric_text(alert['now'], alert['unit'])}, {alert['condition'].lower()} your level of {_metric_text(alert['level'], alert['unit'])}." + (" New today." if alert["new"] else ""), icon=":material/notifications_active:")
    if brief["changes"]:
        st.subheader("What changed")
        table(pd.DataFrame(brief["changes"]))
    if brief["reads"]:
        st.subheader("Market read")
        table(pd.DataFrame(brief["reads"]).assign(Changed=lambda f: f["Changed"].map({True: "Yes", False: ""})))
    if brief["pnl_text"]:
        st.subheader("P&L")
        st.markdown(brief["pnl_text"])
    if brief["estimates"]:
        st.subheader("Next-day power price estimate")
        table(pd.DataFrame(brief["estimates"]))
    if brief["outlook"]:
        st.subheader("Winter storage")
        st.markdown(brief["outlook"])
    st.subheader("Prices used")
    table(pd.DataFrame(brief["prices_used"], columns=["Product", "Priced with"]))
    st.subheader("Your thresholds")
    store = desk_store()
    if store is None:
        st.caption("Thresholds need the saved desk database, which isn't available in this folder.")
        return
    rules = store.load_alerts()
    edited = st.data_editor(rules, num_rows="dynamic", hide_index=True, key="alert_rules", width="stretch", column_config={
        "metric": st.column_config.SelectboxColumn("Item", options=list(BRIEF_METRICS), required=True),
        "condition": st.column_config.SelectboxColumn("Condition", options=["Above", "Below"], required=True),
        "level": st.column_config.NumberColumn("Level", required=True)})
    if st.button("Save thresholds", key="save_alerts"):
        store.save_alerts(edited.dropna())
        st.session_state["flash_message"] = "Thresholds saved."
        st.rerun()
    st.caption("A threshold shows at the top of this page and in the PDF whenever its condition holds. Levels use each item's own unit.")


def page_market_fundamentals():
    title_block("Market fundamentals", "Storage, LNG, power, gas flows and weather, read together. These describe the market. They are not price forecasts.")
    data_items = st.session_state.get("live_data", {})
    snapshot = build_live_snapshot()

    top_a, top_b, top_c = st.columns([1.2, 1.2, 2])
    top_a.metric("Loaded datasets", sum(1 for x in data_items.values() if isinstance(x, dict) and isinstance(x.get("data"), pd.DataFrame) and not x["data"].empty))
    latest_fetch_raw = max((str(x.get("fetched_at", "")) for x in data_items.values() if isinstance(x, dict)), default="")
    try:
        latest_fetch_display = pd.Timestamp(latest_fetch_raw).strftime("%d %b %H:%M UTC") if latest_fetch_raw else "—"
    except Exception:
        latest_fetch_display = "—"
    top_b.metric("Latest fetch (UTC)", latest_fetch_display)
    if top_c.button("Check data sources"):
        st.session_state["pending_nav_page"] = "Live Data Integrations"
        st.rerun()

    if not data_items:
        st.info("No market data yet. Select Update market data in the sidebar.")
    else:
        status_rows = []
        for key, item in data_items.items():
            frame = item.get("data") if isinstance(item, dict) else None
            status_rows.append({
                "Dataset": item.get("label", key),
                "Observations": len(frame) if isinstance(frame, pd.DataFrame) else 0,
                "Latest observation": _latest_observation_date(frame) if isinstance(frame, pd.DataFrame) else None,
                "Last fetched (UTC)": item.get("fetched_at", ""),
                "Source": (frame["_source"].iloc[0] if isinstance(frame, pd.DataFrame) and not frame.empty and "_source" in frame.columns else item.get("label", key)),
            })
        with st.expander("Data coverage, timestamps and provenance", expanded=False):
            table(pd.DataFrame(status_rows))
            st.caption("The observation timestamp describes the market/fundamental period; Last fetched describes when this app retrieved it. These are not interchangeable. GIE AGSI+/ALSI attribution is mandatory.")

    st.subheader("What the data shows")
    d1, d2, d3, d4, d5 = st.columns(5)
    storage = snapshot.get("storage", {})
    lng = snapshot.get("lng", {})
    power = snapshot.get("power_price", {})
    load = snapshot.get("power_load", {})
    weather = snapshot.get("weather", {})
    d1.metric("Storage fill", f"{storage['fill_pct']:.1f}%" if storage.get("fill_pct") is not None else "Not loaded",
              help=f"GIE AGSI+ · {storage.get('scope', 'selected scope')} · {storage.get('date', 'n/a')}")
    change = storage.get("change_7_reports_twh")
    d2.metric("Storage change", f"{change:+.2f} TWh" if change is not None else "Not enough history",
              help="Change across latest and eighth-latest available reporting observations; assumes the same aggregate series and unit.")
    d3.metric("LNG send-out", f"{lng['sendout']:.1f} GWh/d" if lng.get("sendout") is not None else "Not loaded",
              help=f"GIE ALSI+ · {lng.get('date', 'n/a')}")
    d4.metric("Latest day-ahead", f"{power['value']:.2f} €/MWh" if power.get("value") is not None else "Not loaded",
              help=f"ENTSO-E · {power.get('zone', 'selected zone')} · {power.get('timestamp', 'n/a')}")
    d5.metric("Actual load", f"{load['value']/1000:.2f} GW" if load.get("value") is not None else "Not loaded",
              help=f"ENTSO-E · {load.get('zone', 'selected zone')} · {load.get('timestamp', 'n/a')}")

    insights = []
    if change is not None:
        if change < 0:
            interpretation = "Observed storage draw over the latest seven reporting steps. Check weather, consumption, net imports and LNG send-out before forming a price view."
        elif change > 0:
            interpretation = "Observed storage build over the latest seven reporting steps. Check whether the change reflects injections, lower demand or supply flows."
        else:
            interpretation = "Little net inventory change over the selected reporting window."
        insights.append({"Topic": "Gas storage", "Observation": f"{change:+.2f} TWh versus seven reporting steps ago", "Commercial reading": interpretation})
    if power:
        sample_mean = power.get("sample_mean")
        relative = ""
        if sample_mean is not None:
            relative = f"; latest observation is {power['value'] - sample_mean:+.2f} {power.get('unit', 'EUR/MWh')} versus fetched-window mean"
        insights.append({"Topic": "Day-ahead power", "Observation": f"{power.get('zone', 'Selected zone')}: {power['value']:.2f} {power.get('unit', 'EUR/MWh')}{relative}", "Commercial reading": f"{power.get('negative_observations', 0)} negative price observations in the most recent 7-day part of the fetched sample. This is a count of observations, not necessarily hours; the data resolution can vary."})
    weather_summary_for_insights = weather_forecast_summary(
        _live_dataset_frame("weather_daily"), horizon_days=7,
        fallback_location=str(_live_item_meta("weather_daily").get("location", "Selected location")))
    if not weather_summary_for_insights.empty:
        for _, weather_row in weather_summary_for_insights.iterrows():
            notes = []
            temp = weather_row.get("Mean temperature (°C)")
            hdd = weather_row.get("Heating degree-day proxy (base 18°C)")
            cdd = weather_row.get("Cooling degree-day proxy (base 22°C)")
            wind = weather_row.get("Mean wind at 100 m (km/h)")
            solar = weather_row.get("Mean solar radiation (W/m²)")
            if pd.notna(temp): notes.append(f"mean temperature {float(temp):.1f} °C")
            if pd.notna(hdd): notes.append(f"HDD proxy {float(hdd):.1f} over {int(weather_row.get('Forecast days', 0))} days")
            if pd.notna(cdd) and float(cdd) > 0: notes.append(f"CDD proxy {float(cdd):.1f}")
            if pd.notna(wind): notes.append(f"mean wind at 100 m {float(wind):.1f} km/h")
            if pd.notna(solar): notes.append(f"mean solar radiation {float(solar):.1f} W/m²")
            commercial_reading = "Use the temperature/HDD proxy to contextualise weather-sensitive heating demand and gas burn; compare with storage, LNG send-out and flows. Wind/solar proxies indicate resource conditions only; compare with ENTSO-E generation and load. Reference-point weather is not a weighted national forecast and does not directly predict hub prices."
            if notes:
                insights.append({"Topic": f"Weather, {weather_row['Location']}", "Observation": "; ".join(notes), "Commercial reading": commercial_reading})
    elif weather:
        w = []
        if weather.get("temperature_c") is not None: w.append(f"mean temperature {weather['temperature_c']:.1f} °C")
        if weather.get("wind_kmh") is not None: w.append(f"mean wind {weather['wind_kmh']:.1f} km/h")
        if weather.get("solar_wm2") is not None: w.append(f"mean solar radiation {weather['solar_wm2']:.1f} W/m²")
        if w:
            insights.append({"Topic": "Weather", "Observation": f"{weather.get('location', 'Selected location')}: " + "; ".join(w), "Commercial reading": "Use as a local demand/renewables covariate. One city is not a weighted national forecast and does not directly predict hub prices."})
    if insights:
        table(pd.DataFrame(insights)[["Topic", "Observation"]])
        st.caption("These describe observations only. Check weather, consumption, imports and LNG together before forming a price view. Reference-city weather isn't a national forecast.")
    else:
        st.info("Select Update market data in the sidebar to see what the data shows.")

    fund_tabs = view_tabs(["Gas storage", "Winter outlook", "LNG", "Power", "Gas flows", "Weather", "Cross-market analysis"])
    if fund_tabs[1]:
        render_winter_outlook()
    if fund_tabs[0]:
        st.subheader("EU gas storage")
        history_item = _live_dataset_frame("gie_agsi_history")
        seasonal = storage_seasonal(history_item, today=date.today())
        if not seasonal["lines"].empty:
            fig = go.Figure()
            band = seasonal["band"]
            if not band.empty:
                fig.add_trace(go.Scatter(x=band["calendar_day"], y=band["high"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=band["calendar_day"], y=band["low"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(167,187,190,0.18)",
                                         name=f"Range {seasonal['band_years'][0]} to {seasonal['band_years'][-1]}", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=band["calendar_day"], y=band["average"], mode="lines", line=dict(color=GRAY, width=1.5, dash="dot"), name="Five-year average"))
            current = seasonal["current_year"]
            for year, color, width in ((current - 1, TEAL, 1.5), (current, BLUE, 2.5)):
                line = seasonal["lines"].loc[seasonal["lines"]["year"].eq(year)]
                if not line.empty:
                    fig.add_trace(go.Scatter(x=line["calendar_day"], y=line["fill"], mode="lines", line=dict(color=color, width=width), name=str(year)))
            fig.update_layout(title="Storage fill by time of year, %", xaxis=dict(tickformat="%b", dtick="M1"), yaxis_title="")
            chart(fig, 400)

        data = _live_dataset_frame("gie_agsi")
        if not data.empty:
            frame, date_col = _frame_with_time(data, ("observation_date", "gasDayStart", "gas_day"))
            value_col = _find_column(frame, ("gasInStorage", "gas_in_storage"))
            fill_col = _find_column(frame, ("full", "fillLevel", "fill_level"))
            cap_col = _find_column(frame, ("workingGasVolume", "working_gas_volume"))
            scope = _live_item_meta("gie_agsi").get("scope", "selected API scope")
            st.caption(f"Source: GIE AGSI+ · scope: {scope} · updated daily by GIE; latest fetched time {_live_item_fetched_at('gie_agsi')}")
            if date_col and value_col:
                frame[value_col] = _number_series(frame[value_col])
                if fill_col: frame[fill_col] = _number_series(frame[fill_col])
                if cap_col: frame[cap_col] = _number_series(frame[cap_col])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=frame[date_col], y=frame[value_col], mode="lines+markers", name="Gas in storage (TWh)", line=dict(color=AMBER, width=2)))
                fig.update_layout(title="Gas inventory trend", xaxis_title="Gas day", yaxis_title="GIE reported inventory unit (normally TWh)")
                chart(fig, 360)
                if fill_col:
                    fig_fill = px.line(frame, x=date_col, y=fill_col, markers=True, title="Reported storage fill level", labels={fill_col: "Fill (%)", date_col: "Gas day"})
                    fig_fill.update_layout(height=300)
                    chart(fig_fill)
                elif cap_col:
                    frame["derived_fill_pct"] = frame[value_col] / frame[cap_col] * 100.0
                    st.caption("Fill percentage is derived as inventory / working gas volume because the feed did not provide a usable full field.")
                    fig_fill = px.line(frame, x=date_col, y="derived_fill_pct", markers=True, title="Derived storage fill level", labels={"derived_fill_pct": "Fill (%)", date_col: "Gas day"})
                    fig_fill.update_layout(height=300)
                    chart(fig_fill)
            table(data)
        else:
            st.info("Select Update market data in the sidebar to load storage.")

    if fund_tabs[2]:
        data = _live_dataset_frame("gie_alsi")
        st.subheader("LNG terminals")
        if not data.empty:
            frame, date_col = _frame_with_time(data, ("observation_date", "gasDayStart", "gas_day"))
            inv_col = _find_column(frame, ("inventory", "inventory_lng", "inventory_gwh", "inventory_twh"))
            send_col = _find_column(frame, ("sendOut", "sendout", "send_out", "sendout_gwh"))
            st.caption(f"Source: GIE ALSI+ · scope: {_live_item_meta('gie_alsi').get('scope', 'selected scope')} · inventory units follow the field returned by GIE (often 10³ m³ LNG); send-out is commonly GWh/d.")
            if date_col and inv_col:
                frame[inv_col] = _number_series(frame[inv_col])
                fig = px.line(frame, x=date_col, y=inv_col, color="name" if "name" in frame.columns and frame["name"].nunique() > 1 else None, markers=True, title=f"LNG inventory · {inv_col}")
                fig.update_layout(height=330, xaxis_title="Gas day", yaxis_title="Provider field unit")
                chart(fig)
            if date_col and send_col:
                frame[send_col] = _number_series(frame[send_col])
                fig = px.line(frame, x=date_col, y=send_col, color="name" if "name" in frame.columns and frame["name"].nunique() > 1 else None, markers=True, title="LNG send-out")
                fig.update_layout(height=330, xaxis_title="Gas day", yaxis_title="GWh/d (confirm selected field/unit)")
                chart(fig)
            table(data)
        else:
            st.info("Select Update market data in the sidebar to load LNG data.")

    if fund_tabs[3]:
        st.subheader("Power prices, load and generation")
        power_keys = [("entsoe_price", "Day-ahead price"), ("entsoe_load", "Actual load"), ("entsoe_generation", "Actual generation by type"), ("entsoe_flow", "Physical flow")]
        available = [(key, label) for key, label in power_keys if _live_variants(key) or not _live_dataset_frame(key).empty]
        if available:
            c1, c2 = st.columns(2)
            selected_power_key = c1.selectbox("Dataset", [x[0] for x in available], format_func=lambda x: dict(available)[x], key="fundamentals_entsoe_display_key")
            variants = _live_variants(selected_power_key)
            variant = c2.selectbox("Zone", list(variants), key=f"fundamentals_zone_{selected_power_key}") if variants else None
            data_key = f"{selected_power_key}|{variant}" if variant else selected_power_key
            data = _live_dataset_frame(data_key).copy()
            st.caption(f"ENTSO-E Transparency Platform. Fetched {_live_item_fetched_at(data_key)[:16].replace('T', ' ')} UTC. Times are UTC.")
            if {"timestamp", "value"}.issubset(data.columns):
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
                data["value"] = _number_series(data["value"])
                data = data.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
                unit = str(data["unit"].iloc[0]) if "unit" in data.columns and not data.empty else "Value"
                fig = px.line(data, x="timestamp", y="value", color="series" if "series" in data.columns and data["series"].nunique() > 1 else None,
                              title=dict(available)[selected_power_key], labels={"timestamp": "", "value": unit, "series": ""})
                chart(fig, 380)
                if selected_power_key == "entsoe_price":
                    k = spot_kpis(spot_series(data, ENTSOE_ZONES.get(variant or "France", {}).get("timezone", "Europe/Paris")))
                    p1, p2, p3, p4 = st.columns(4)
                    kpi(p1, "Average", f"{k['base']:.2f} €/MWh")
                    kpi(p2, "Low", f"{k['low']:.2f} €/MWh")
                    kpi(p3, "High", f"{k['high']:.2f} €/MWh")
                    kpi(p4, "Negative hours", f"{k['negative_hours']:.1f}")
                if selected_power_key == "entsoe_generation" and "series" in data:
                    produced = data.loc[~data["series"].astype(str).str.contains("consumption", case=False)]
                    mix = produced.groupby("series", as_index=False)["value"].mean().sort_values("value")
                    fig = px.bar(mix, x="value", y="series", orientation="h", title="Average output by production type, MW", labels={"series": "", "value": ""})
                    fig.update_traces(marker_color=BLUE)
                    chart(fig, max(300, 30 * len(mix) + 90))
                table(data, height=300)
        else:
            need_data_note("Select Update market data in the sidebar to load prices, load and generation.")

    if fund_tabs[4]:
        data = _live_dataset_frame("entsog_flow")
        st.subheader("Gas flows")
        if not data.empty:
            value_col = _numeric_column(data, ("value_numeric", "value", "quantity", "flow"))
            date_col = _find_column(data, ("timestamp", "periodFrom", "period_from", "date"))
            meta = _live_item_meta("entsog_flow")
            st.caption(f"Source: ENTSOG Transparency Platform · indicator {meta.get('indicator', 'selected indicator')} · point direction {meta.get('point_direction', 'selected point')} · last fetched {_live_item_fetched_at('entsog_flow')}")
            if date_col and value_col:
                chart_frame = data.copy()
                chart_frame[date_col] = pd.to_datetime(chart_frame[date_col], errors="coerce")
                chart_frame[value_col] = _number_series(chart_frame[value_col])
                chart_frame = chart_frame.dropna(subset=[date_col, value_col])
                if not chart_frame.empty:
                    fig = px.line(chart_frame, x=date_col, y=value_col, markers=True, title="Selected point-direction flow / indicator")
                    fig.update_layout(height=350, xaxis_title="Provider observation period", yaxis_title="Provider value — verify unit")
                    chart(fig)
            table(data)
            st.caption("A network flow at a specific point is not automatically a delivery to a contractual counterparty; direction and physical units must be reconciled.")
        else:
            st.info("Fetch the ENTSOG point-direction catalogue, select an exact identifier, and fetch the chosen indicator in Live Data Integrations.")

    if fund_tabs[5]:
        data = _live_dataset_frame("weather_daily")
        st.subheader("Weather")
        weather_for_normals = _live_dataset_frame("weather_daily")
        anomaly = hdd_vs_normal(weather_for_normals, _live_dataset_frame("weather_normals"))
        if not anomaly.empty:
            view = anomaly.sort_values("hdd_vs_normal")
            fig = go.Figure(go.Bar(x=view["hdd_vs_normal"], y=view["location"], orientation="h", marker=dict(color=BLUE), hovertemplate="%{y}: %{x:+.1f} degree days<extra></extra>"))
            fig.update_layout(title="Next 7 days: heating degree days above or below normal", hovermode="closest")
            chart(fig, 300)
            table(anomaly)

        if not data.empty:
            meta = _live_item_meta("weather_daily")
            st.caption(f"Source: Open-Meteo · locations: {', '.join(meta.get('locations', [meta.get('location', 'selected location')]))} · last fetched {_live_item_fetched_at('weather_daily')}")
            weather_summary = weather_forecast_summary(data, horizon_days=7,
                fallback_location=str(meta.get("location", "Selected location")))
            if not weather_summary.empty:
                st.subheader("Commercial weather indicators — next 7 future local days")
                table(weather_summary)
            chart_data = data.copy()
            chart_data["date"] = pd.to_datetime(chart_data["date"], errors="coerce")
            location_field = "location" if "location" in chart_data.columns else None
            if "temperature_mean_c" in chart_data.columns:
                fig = px.line(chart_data, x="date", y="temperature_mean_c", color=location_field, markers=True,
                    title="Daily mean temperature by reference location",
                    labels={"date": "Local date", "temperature_mean_c": "Temperature (°C)", "location": "Location"})
                fig.update_layout(height=330)
                chart(fig)
            if "wind_speed_100m_mean_kmh" in chart_data.columns:
                fig = px.line(chart_data, x="date", y="wind_speed_100m_mean_kmh", color=location_field, markers=True,
                    title="Wind-resource proxy at 100 m", labels={"date": "Local date", "wind_speed_100m_mean_kmh": "Wind speed (km/h)", "location": "Location"})
                fig.update_layout(height=330)
                chart(fig)
            if "solar_radiation_mean_wm2" in chart_data.columns:
                fig = px.line(chart_data, x="date", y="solar_radiation_mean_wm2", color=location_field, markers=True,
                    title="Solar-resource proxy", labels={"date": "Local date", "solar_radiation_mean_wm2": "Mean shortwave radiation (W/m²)", "location": "Location"})
                fig.update_layout(height=330)
                chart(fig)
            with st.expander("Hourly and daily source data"):
                table(data)
                hourly_weather = _live_dataset_frame("weather_hourly")
                if not hourly_weather.empty:
                    st.download_button("Download hourly weather (CSV)", hourly_weather.to_csv(index=False).encode("utf-8"), file_name="open_meteo_hourly_weather.csv", mime="text/csv", key="fundamentals_download_weather_hourly")
            st.caption("HDD/CDD use simple daily-mean-temperature bases of 18°C/22°C. Wind at 100 m and shortwave radiation are renewable-resource proxies—not generation forecasts. Representative city points are not national weighted averages. Source: Open-Meteo (CC BY 4.0; attribution required).")
        else:
            st.info("Select Update market data in the sidebar to load the forecast.")

    if fund_tabs[6]:
        st.subheader("Residual load against day-ahead price")
        ready = residual_ready_zones()
        if ready:
            residual_zone = st.selectbox("Zone", ready, key="residual_zone")
            residual = session_memo(f"residual|{residual_zone}", data_version("live"), lambda: _build_residual_load_proxy(residual_zone))
        else:
            residual_zone = None
            residual = _build_residual_load_proxy()
        if not residual.empty:
            st.caption("Actual load minus reported wind and solar, in UTC hours. It leaves out other generation, trade, storage, curtailment and losses, so treat it as a proxy.")
            fig = px.line(residual, x="hour", y="residual_load_proxy_mw", title="Residual load, MW", labels={"hour": "", "residual_load_proxy_mw": ""})
            fig.update_traces(line_color=BLUE)
            chart(fig, 330)
            scatter = px.scatter(residual, x="residual_load_proxy_mw", y="day_ahead_eur_mwh", title="Day-ahead price against residual load",
                                 labels={"residual_load_proxy_mw": "Residual load (MW)", "day_ahead_eur_mwh": "EUR/MWh"})
            scatter.update_traces(marker_color=BLUE, marker_opacity=0.65)
            chart(scatter, 360)
            corr = residual[["residual_load_proxy_mw", "day_ahead_eur_mwh"]].corr().iloc[0, 1] if len(residual) >= 3 else float("nan")
            st.metric("Correlation in this sample", f"{corr:.3f}" if pd.notna(corr) else "Not enough overlap", help="Describes this sample only. It doesn't show cause and isn't a signal.")
            st.download_button("Download as CSV", residual.to_csv(index=False).encode("utf-8"), file_name="meridian_power_context.csv", mime="text/csv", key="download_residual_load_context")
        else:
            need_data_note("Select Update market data in the sidebar. This view needs prices, load and generation for the same zone, and never combines different zones.")

    with st.expander("How to read these indicators"):
        st.markdown("- **GIE AGSI+ and ALSI:** storage and LNG data give gas supply context. Attribution to GIE is required.\n"
                    "- **ENTSO-E:** day-ahead, load and generation are spot observations. They never replace monthly forward marks.\n"
                    "- **ENTSOG:** a flow at a point keeps its direction, period and unit, and isn't the same as a contract delivery.\n"
                    "- **Open-Meteo:** weather gives context for demand and renewables. It isn't a price forecast.\n"
                    "- **ICE Endex and EEX:** market-referenced MtM needs a licensed curve or an authorised export.")
        st.caption("All indicators are descriptive. Providers can delay or revise data, and this app doesn't certify what you import.")


def page_physical_operations():
    title_block("Physical operations", "Nominations, deliveries and invoices for trades flagged as physical, checked against the book.")
    st.caption("A simulator of the operations workflow. It isn't connected to a real nomination or logistics system.")
    ops = reconcile_physical_ops(st.session_state.physical_ops, st.session_state.trades)
    if ops.empty:
        need_data_note("No physical trades yet. Add a confirmed or executed trade with Physical settlement to create an operations record.")
    else:
        open_mask = ~ops["operational_status"].isin(["DELIVERED", "CANCELLED"])
        exceptions = ~ops["risk_flag"].eq("RECONCILED")
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Open contracts", str(int(open_mask.sum())))
        kpi(c2, "Outstanding volume", f"{ops['outstanding_mwh'].clip(lower=0).sum():,.0f} MWh")
        kpi(c3, "Exceptions", str(int(exceptions.sum())))
        kpi(c4, "Mismatches with the book", str(int((ops["reconciliation"] != "MATCH").sum())))
        show_cols = ["contract_id", "trade_id", "product", "counterparty", "delivery_start", "delivery_end",
                     "profile", "contracted_mwh", "nominated_mwh", "delivered_mwh", "outstanding_mwh",
                     "nomination_gap_mwh", "delivery_vs_nomination_mwh", "operational_status", "invoice_status",
                     "trade_status", "reconciliation", "risk_flag", "updated_at"]
        table(ops[show_cols].round(3))
        st.download_button("Download physical operations (CSV)", ops.to_csv(index=False).encode("utf-8"),
                            "physical_operations_reconciliation.csv", "text/csv")
        st.subheader("Update nominations, delivery and invoice status")
        selected_contract = st.columns([1, 2])[0].selectbox("Operations contract", ops["contract_id"].astype(str).tolist(), key="ops_selected_contract")
        selected_row = ops.loc[ops["contract_id"].astype(str).eq(selected_contract)].iloc[0]
        max_volume = max(1_000_000.0, float(selected_row["contracted_mwh"]) * 2.0 + 1000.0,
                         float(selected_row["nominated_mwh"]), float(selected_row["delivered_mwh"]))
        with st.form("update_physical_operations"):
            c1, c2, c3 = st.columns(3)
            nominated = c1.number_input("Nominated volume (MWh)", min_value=0.0, max_value=max_volume,
                                        value=float(selected_row["nominated_mwh"]), step=100.0)
            delivered = c2.number_input("Delivered / allocated volume (MWh)", min_value=0.0, max_value=max_volume,
                                        value=float(selected_row["delivered_mwh"]), step=100.0)
            operational_status = c3.selectbox("Operational status", ["SCHEDULED", "NOMINATED", "IN_DELIVERY", "DELIVERED", "ON_HOLD", "CANCELLED"],
                                               index=["SCHEDULED", "NOMINATED", "IN_DELIVERY", "DELIVERED", "ON_HOLD", "CANCELLED"].index(str(selected_row["operational_status"])))
            invoice_status = st.selectbox("Invoice status", ["NOT_DUE", "DRAFT", "ISSUED", "PAID", "DISPUTED"],
                                          index=["NOT_DUE", "DRAFT", "ISSUED", "PAID", "DISPUTED"].index(str(selected_row["invoice_status"])))
            save_ops = st.form_submit_button("Save operations update", type="primary")
        if save_ops:
            try:
                commit_ops(update_physical_operation(
                    st.session_state.physical_ops, selected_contract, nominated, delivered,
                    operational_status, invoice_status
                ))
                st.session_state["flash_message"] = f"Saved updates for {selected_contract}."
                st.rerun()
            except Exception as exc:
                st.error(f"Operations update rejected: {exc}")
        if (ops["reconciliation"] != "MATCH").any():
            st.warning("At least one operations record does not match a linked physical-settlement trade. Correct the trade reference, product, settlement convention or contracted volume.")
        st.caption("Outstanding = contracted minus delivered. Nomination gap = contracted minus nominated. Delivery vs nomination = delivered minus nominated. Overdelivery and missing nominations are flagged rather than silently clipped.")


def page_live_data_integrations():
    title_block("Live data", "Every source updates with your saved keys. Use this page to check them, or to fetch something specific.")
    with st.container(key="panel"):
        c1, c2 = st.columns([1, 2.6], vertical_alignment="center")
        with c1:
            update_button("live_update_all", "Update all sources")
        c2.caption("Fetches only what's new from ENTSO-E, GIE, Open-Meteo and the ECB. You don't need to enter any key.")
        results = st.session_state.get("update_results") or []
        if results:
            table(pd.DataFrame([{"Source": n, "Result": "Updated" if good else "Failed", "Detail": d} for n, good, d in results]))
    saved = resolved_credentials()
    with st.expander("Replace an access key", expanded=False):
        st.caption("You only need this to use a different key. Your saved keys come from the keys file beside the app and are never sent to your browser, so these fields stay empty. A key pasted here lasts for this session. To change one permanently, edit meridian_keys.toml.")
        c1, c2, c3 = st.columns(3)
        gie_api_key = c1.text_input("GIE key (AGSI+ and ALSI)", value="", type="password", key="live_gie_api_key", placeholder="Saved key in use" if saved["gie"] else "No key saved")
        entsoe_api_token = c2.text_input("ENTSO-E token", value="", type="password", key="live_entsoe_token", placeholder="Saved token in use" if saved["entsoe"] else "No token saved")
        openmeteo_api_key = c3.text_input("Open-Meteo commercial key (optional)", value="", type="password", key="live_openmeteo_key", placeholder="Saved key in use" if saved["open_meteo"] else "Public endpoint in use")
        overrides = st.session_state.setdefault("credential_overrides", {})
        for name, typed in (("gie", gie_api_key), ("entsoe", entsoe_api_token), ("open_meteo", openmeteo_api_key)):
            if typed.strip():
                overrides[name] = typed.strip()
        saved = resolved_credentials()
        effective_gie_api_key, effective_entsoe_api_token, openmeteo_api_key = saved["gie"], saved["entsoe"], saved["open_meteo"]
        st.markdown("[AGSI+](https://agsi.gie.eu/#/api), [ALSI](https://alsi.gie.eu/#/api), [ENTSO-E](https://transparency.entsoe.eu/), [ENTSOG manual](https://transparency.entsog.eu/api/archiveDirectories/8/api-manual/ENTSOG_TP_API_UserManual_v3.0.pdf), [Open-Meteo](https://open-meteo.com/en/docs), [ICE Endex](https://developer.ice.com/fixed-income-data-services/catalog/ice-endex)")
        st.caption("Open-Meteo's public endpoint is for non-commercial use. Check its plans before commercial use.")

    tab_gie, tab_power, tab_gasflows, tab_weather, tab_forward = view_tabs(["Gas storage and LNG", "Power", "Gas flows", "Weather", "Licensed forward curves"])

    if tab_gie:
        st.subheader("GIE storage inventories and LNG terminal data")
        st.write("AGSI+ provides daily storage inventory/fill/injection/withdrawal data; ALSI provides LNG-terminal inventory and send-out data. The Europe/EU selector requests GIE's documented EU aggregate series (`type=eu`); the country selector requests a country-level dataset. Do not sum facility rows blindly. Observations can be estimated or retroactively corrected.")
        with st.form("gie_live_fetch_form"):
            g1, g2, g3 = st.columns(3)
            gie_dataset = g1.selectbox("Dataset", ["AGSI — Gas storage", "ALSI — LNG terminals"], key="gie_dataset_choice")
            gie_scope = g2.selectbox("Scope", ["Europe / EU datasets", "Selected country"], key="gie_scope_choice")
            gie_country = g3.selectbox("Country", ["DE", "FR", "NL", "IT", "ES", "BE", "AT", "PL", "CZ", "DK", "GB"], key="gie_country_choice")
            g1, g2 = st.columns(2)
            gie_from = g1.date_input("From gas day", value=date.today() - timedelta(days=30), key="gie_from_date")
            gie_to = g2.date_input("To gas day", value=date.today(), key="gie_to_date")
            gie_submit = st.form_submit_button("Fetch GIE history", type="primary")
        if gie_submit:
            try:
                dataset_code = "AGSI" if gie_dataset.startswith("AGSI") else "ALSI"
                with st.spinner("Requesting GIE observations…"):
                    frame = fetch_gie_history(dataset_code, effective_gie_api_key, gie_scope, gie_from, gie_to, gie_country)
                _stored_live_item(f"gie_{dataset_code.lower()}", gie_dataset, frame, {"scope": gie_scope, "country": gie_country})
                st.success(f"Loaded {len(frame):,} GIE observation(s).")
            except Exception as exc:
                st.error(f"GIE request failed: {exc}")
        gie_show_dataset = "gie_agsi" if gie_dataset.startswith("AGSI") else "gie_alsi"
        gie_frame = _live_dataset_frame(gie_show_dataset)
        if not gie_frame.empty:
            st.caption(f"{fetched_text(st.session_state.live_data[gie_show_dataset]['fetched_at'])} · Source: {st.session_state.live_data[gie_show_dataset]['label']}")
            date_axis = "observation_date" if "observation_date" in gie_frame.columns else None
            value_axis = next((c for c in (("gasInStorage", "gasinstorage", "workingGasVolume", "full") if gie_show_dataset == "gie_agsi" else ("inventory_gwh", "gasInStorage", "inventory", "sendOut", "sendout")) if c in gie_frame.columns), None)
            color_axis = "name" if "name" in gie_frame.columns and gie_frame["name"].nunique() > 1 else None
            if date_axis and value_axis:
                chart_data = gie_frame.copy()
                chart_data[date_axis] = pd.to_datetime(chart_data[date_axis], errors="coerce")
                fig = px.line(chart_data, x=date_axis, y=value_axis, color=color_axis, markers=True, title=f"{gie_dataset} · {value_axis}")
                fig.update_layout(height=350)
                chart(fig)
            table(gie_frame)
            st.download_button("Download GIE data (CSV)", gie_frame.to_csv(index=False).encode("utf-8"), file_name=f"{gie_show_dataset}.csv", mime="text/csv", key=f"download_{gie_show_dataset}")
        else:
            st.info("No GIE dataset has been fetched in this session yet.")

    if tab_power:
        st.subheader("ENTSO-E electricity data")
        st.write("Day-ahead prices, actual load, generation by type and cross-border flows. Each zone is kept, so you can load several and compare them in Power Spot. Periods up to three years are split into smaller requests for you.")
        kinds = ["Day-ahead price", "Actual load", "Actual generation by type", "Physical flow"]
        keys_by_kind = {"Day-ahead price": "entsoe_price", "Actual load": "entsoe_load", "Actual generation by type": "entsoe_generation", "Physical flow": "entsoe_flow"}
        zones = list(ENTSOE_ZONES)
        e0, e1, e2 = st.columns(3)
        entsoe_kind = e0.selectbox("Dataset", kinds, key="entsoe_kind_choice")
        if entsoe_kind == "Physical flow":
            entsoe_source = e1.selectbox("Flow from", zones, index=zones.index("France"), key="entsoe_flow_source")
            entsoe_target = e2.selectbox("Flow to", zones, index=zones.index("Germany-Luxembourg"), key="entsoe_flow_target")
            selected_zone = entsoe_source
        else:
            selected_zone = e1.selectbox("Bidding zone", zones, index=zones.index("France"), key="entsoe_zone_choice")
            entsoe_source, entsoe_target = selected_zone, None
        e0, e1, e2 = st.columns(3, vertical_alignment="bottom")
        entsoe_from = e0.date_input("First delivery day", value=date.today() - timedelta(days=7), key="entsoe_start_date")
        entsoe_to = e1.date_input("Last delivery day", value=date.today(), key="entsoe_end_date")
        if e2.button("Fetch from ENTSO-E", type="primary", key="entsoe_fetch", width="stretch"):
            try:
                progress = st.progress(0.0, text="Requesting ENTSO-E data")
                frame = fetch_entsoe_series(effective_entsoe_api_token, entsoe_kind, selected_zone, entsoe_from, entsoe_to + timedelta(days=1),
                                            source_zone=entsoe_source, target_zone=entsoe_target,
                                            progress=lambda share: progress.progress(share, text="Requesting ENTSO-E data"))
                progress.empty()
                variant = f"{entsoe_source} to {entsoe_target}" if entsoe_kind == "Physical flow" else selected_zone
                store_entsoe(f"{keys_by_kind[entsoe_kind]}|{variant}", f"{entsoe_kind}, {variant}", frame,
                             {"zone": selected_zone, "source_zone": entsoe_source, "target_zone": entsoe_target, "variant": variant})
                st.success(f"Loaded {len(frame):,} ENTSO-E observations for {variant}.")
            except Exception as exc:
                st.error(f"ENTSO-E request failed: {exc}")
        variants = _live_variants(keys_by_kind[entsoe_kind])
        if variants:
            chosen = st.columns([1, 2])[0].selectbox("Show", list(variants), key=f"entsoe_show_{keys_by_kind[entsoe_kind]}")
            item = variants[chosen]
            entsoe_display = item["data"].copy()
            entsoe_display["timestamp"] = pd.to_datetime(entsoe_display["timestamp"], errors="coerce", utc=True)
            fig = px.line(entsoe_display, x="timestamp", y="value", color="series" if entsoe_display["series"].nunique() > 1 else None,
                          title=str(item["label"]), labels={"timestamp": "", "value": str(entsoe_display["unit"].iloc[0]), "series": ""})
            chart(fig, 360)
            st.caption(f"Fetched {str(item['fetched_at'])[:16].replace('T', ' ')} UTC. Times are UTC.")
            table(entsoe_display, height=300)
            st.download_button("Download as CSV", entsoe_display.to_csv(index=False).encode("utf-8"), file_name=f"{keys_by_kind[entsoe_kind]}_{chosen.replace(' ', '_')}.csv", mime="text/csv", key=f"download_{keys_by_kind[entsoe_kind]}")
        else:
            need_data_note("Nothing of this type has been fetched yet.")

    if tab_gasflows:
        st.subheader("ENTSOG pipeline / interconnection flow data")
        st.write("ENTSOG uses exact, case-sensitive point-direction identifiers and indicator names. Search the published point-direction catalogue, then request one point for a bounded period.")
        c1, c2 = st.columns([1, 2])
        if c1.button("Load point-direction catalogue", key="entsog_load_catalogue"):
            try:
                with st.spinner("Loading published point directions…"):
                    catalogue = fetch_entsog_point_catalogue(limit=1000)
                st.session_state.entsog_point_catalogue = catalogue
                st.success(f"Loaded {len(catalogue):,} catalogue records.")
            except Exception as exc:
                st.error(f"ENTSOG catalogue request failed: {exc}")
        catalogue = st.session_state.get("entsog_point_catalogue", pd.DataFrame())
        keyword = c2.text_input("Filter catalogue rows", placeholder="e.g. France, Germany, Fluxys, Zeebrugge", key="entsog_catalogue_search")
        if isinstance(catalogue, pd.DataFrame) and not catalogue.empty:
            shown = catalogue
            if keyword.strip():
                mask = shown.astype(str).apply(lambda col: col.str.contains(keyword.strip(), case=False, na=False)).any(axis=1)
                shown = shown.loc[mask]
            table(shown.head(100))
        direction_col = next((c for c in catalogue.columns if "".join(ch for ch in c.casefold() if ch.isalnum()) == "pointdirection"), None) if isinstance(catalogue, pd.DataFrame) else None
        direction_options = catalogue[direction_col].dropna().astype(str).unique().tolist() if direction_col and not catalogue.empty else []
        selected_direction = st.selectbox("Point direction from catalogue", [""] + direction_options, key="entsog_direction_selected") if direction_options else ""
        with st.form("entsog_flow_fetch_form"):
            manual_direction = st.text_input("Or paste pointDirection identifier", value=selected_direction, key="entsog_manual_direction", help="Use the exact pointDirection value from the ENTSOG operatorPointDirections catalogue; where needed, it is operatorKey + pointKey + directionKey.")
            f1, f2, f3 = st.columns(3)
            entsog_indicator = f1.selectbox("Indicator", ["Physical Flow", "Nomination", "Allocation"], key="entsog_indicator_choice")
            entsog_from = f2.date_input("From", value=date.today() - timedelta(days=30), key="entsog_from_date")
            entsog_to = f3.date_input("To (inclusive)", value=date.today() - timedelta(days=1), key="entsog_to_date")
            entsog_submit = st.form_submit_button("Fetch ENTSOG data", type="primary")
        if entsog_submit:
            try:
                with st.spinner("Requesting ENTSOG observations…"):
                    frame = fetch_entsog_flow(manual_direction, entsog_from, entsog_to, entsog_indicator)
                _stored_live_item("entsog_flow", f"ENTSOG {entsog_indicator}", frame, {"point_direction": manual_direction, "indicator": entsog_indicator})
                st.success(f"Loaded {len(frame):,} ENTSOG observations.")
            except Exception as exc:
                st.error(f"ENTSOG request failed: {exc}")
        entsog_frame = _live_dataset_frame("entsog_flow")
        if not entsog_frame.empty:
            st.caption(f"{fetched_text(st.session_state.live_data['entsog_flow']['fetched_at'])} · Units are supplied by ENTSOG; confirm the unit before interpreting volumes.")
            if "timestamp" in entsog_frame and "value_numeric" in entsog_frame:
                chart_data = entsog_frame.dropna(subset=["timestamp", "value_numeric"]).copy()
                if not chart_data.empty:
                    fig = px.line(chart_data, x="timestamp", y="value_numeric", markers=True, title="ENTSOG daily flow / indicator")
                    fig.update_layout(height=350)
                    chart(fig)
            table(entsog_frame)
            st.download_button("Download ENTSOG data (CSV)", entsog_frame.to_csv(index=False).encode("utf-8"), file_name="entsog_flows.csv", mime="text/csv", key="download_entsog_flow")
        st.markdown("[Open the ENTSOG Transparency Platform](https://transparency.entsog.eu/) for the official catalogue and point details.")

    if tab_weather:
        st.subheader("Weather-driven demand and renewable-resource proxies")
        st.write("Weather is relevant to European energy markets because temperature influences heating/cooling demand, while wind and solar conditions influence potential renewable output. These are contextual indicators—not forecasts of actual generation, system demand or power prices.")
        default_weather_locations = [
            "Amsterdam / TTF proxy", "London / NBP proxy", "Paris / French power proxy",
            "Berlin / German power proxy", "Brussels / Belgian power proxy", "Madrid / Iberian power proxy",
        ]
        with st.form("open_meteo_live_fetch_form"):
            weather_locations = st.multiselect("Energy-market reference locations", list(WEATHER_LOCATIONS.keys()),
                default=default_weather_locations, key="weather_location_choices",
                help="These are representative city points around key European energy markets, not country-weighted weather indices.")
            w1, w2 = st.columns(2)
            weather_past = w1.selectbox("Recent days for context", [0, 1, 3, 5, 7], index=2, key="weather_past_days")
            weather_forecast = w2.selectbox("Forecast horizon (days, including today)", [3, 5, 7, 10, 14, 16], index=3, key="weather_forecast_days")
            weather_commercial = st.checkbox("Use Open-Meteo commercial endpoint", value=False, key="weather_commercial_endpoint")
            weather_submit = st.form_submit_button("Fetch regional weather outlook", type="primary")
        if weather_submit:
            try:
                with st.spinner(f"Requesting weather for {len(weather_locations)} selected reference location(s)…"):
                    hourly, daily, failures = fetch_open_meteo_market_pack(
                        weather_locations, weather_past, weather_forecast, openmeteo_api_key, weather_commercial)
                if daily.empty:
                    raise RuntimeError("No weather data returned. " + " · ".join(failures))
                _stored_live_item("weather_hourly", "Open-Meteo hourly · European energy-market locations", hourly,
                    {"locations": weather_locations, "location": "Multiple energy-market reference locations", "forecast_days": weather_forecast})
                _stored_live_item("weather_daily", "Open-Meteo daily · European energy-market locations", daily,
                    {"locations": weather_locations, "location": "Multiple energy-market reference locations", "forecast_days": weather_forecast})
                st.session_state["weather_fetch_failures"] = failures
                st.success(f"Loaded {len(hourly):,} hourly rows and {len(daily):,} location-days for {daily['location'].nunique()} location(s).")
                if failures:
                    st.warning("Partial weather coverage — " + " · ".join(failures))
            except Exception as exc:
                st.error(f"Open-Meteo request failed: {exc}")
        weather_daily = _live_dataset_frame("weather_daily")
        if not weather_daily.empty:
            st.caption(f"{fetched_text(st.session_state.live_data['weather_daily']['fetched_at'])} · Source: [Open-Meteo](https://open-meteo.com/) · Data licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Public endpoint requires no API key for eligible non-commercial use; commercial use requires the appropriate plan.")
            weather_summary = weather_forecast_summary(weather_daily, horizon_days=7,
                fallback_location=str(_live_item_meta("weather_daily").get("location", "Selected location")))
            if not weather_summary.empty:
                st.subheader("Next 7 future local days — commercial context")
                table(weather_summary)
            if "temperature_mean_c" in weather_daily.columns:
                chart_data = weather_daily.copy()
                chart_data["date"] = pd.to_datetime(chart_data["date"], errors="coerce")
                fig = px.line(chart_data, x="date", y="temperature_mean_c", color="location" if "location" in chart_data.columns else None,
                    markers=True, title="Daily mean temperature — recent context and forecast",
                    labels={"date": "Local date", "temperature_mean_c": "Temperature (°C)", "location": "Reference location"})
                fig.update_layout(height=360)
                chart(fig)
            table(weather_daily)
            hourly_weather = _live_dataset_frame("weather_hourly")
            if not hourly_weather.empty:
                st.download_button("Download hourly weather (CSV)", hourly_weather.to_csv(index=False).encode("utf-8"), file_name="open_meteo_hourly_weather.csv", mime="text/csv", key="download_weather_hourly")
            st.caption("Commercial interpretation: HDD proxy (base 18°C) helps contextualise heating demand/gas burn; CDD proxy (base 22°C) flags cooling demand; wind at 100 m and shortwave radiation are weather-resource proxies. Compare them with ENTSO-E load/generation, and do not treat a weather proxy as actual output or a price forecast.")
            failures = st.session_state.get("weather_fetch_failures", [])
            if failures:
                st.warning("Some selected locations failed: " + " · ".join(failures))
        else:
            st.info("No weather series loaded yet. The public Open-Meteo endpoint requires no API key for eligible non-commercial use. Fetch a multi-region outlook above or from Market Overview.")

    if tab_forward:
        st.subheader("Licensed ICE Endex / EEX or other vendor forward curves")
        st.warning("ICE/EEX futures data is licensed. This adapter does not bypass exchange entitlements; use the endpoint and credentials supplied by your data agreement, or upload a permitted export.")
        st.markdown("Supported input schema after mapping: one row per contract month with a parseable delivery date and price. The selected product sets the canonical currency/unit; no hidden FX or unit conversion is performed.")
        st.subheader("Option A — Fetch from your licensed REST endpoint")
        p1, p2 = st.columns([2, 1])
        provider_url = p1.text_input("Licensed provider HTTPS endpoint", placeholder="https://your-provider.example/api/forward-curves", key="provider_curve_url")
        provider_auth = p2.selectbox("Authentication", ["Bearer token", "API-key header", "No authentication"], key="provider_auth_mode")
        p1, p2, p3 = st.columns(3)
        provider_token = p1.text_input("Provider token / key", type="password", key="provider_curve_token")
        provider_header = p2.text_input("Auth header name", value="Authorization" if provider_auth == "Bearer token" else "X-API-Key", key="provider_auth_header")
        provider_format = p3.selectbox("Response format", ["Auto", "JSON", "CSV"], key="provider_response_format")
        provider_query_json = st.text_area("Provider query parameters (JSON object, optional)", value="{}", height=70, key="provider_query_parameters", help="Use only the parameter names and values specified by your market-data vendor. Keep credentials in the authentication field/header where possible.")
        if st.button("Fetch licensed provider table", type="primary", key="fetch_provider_curve_button"):
            try:
                query_parameters = json.loads(provider_query_json or "{}")
                if not isinstance(query_parameters, dict):
                    raise ValueError("Provider query parameters must be a JSON object.")
                with st.spinner("Requesting licensed provider data…"):
                    raw_table = fetch_provider_table(provider_url, provider_token, provider_auth, provider_header, provider_format, query_parameters)
                st.session_state.provider_curve_raw_table = raw_table
                st.success(f"Received {len(raw_table):,} row(s) and {len(raw_table.columns):,} column(s) from the configured provider endpoint.")
            except Exception as exc:
                st.error(f"Provider request failed: {exc}")
        st.subheader("Option B — Upload an authorised vendor export")
        provider_upload = st.file_uploader("Upload provider CSV or JSON", type=["csv", "json"], key="provider_curve_upload")
        if provider_upload is not None:
            try:
                if provider_upload.name.lower().endswith(".csv"):
                    st.session_state.provider_curve_raw_table = pd.read_csv(provider_upload)
                else:
                    payload = json.loads(provider_upload.getvalue().decode("utf-8-sig"))
                    rows = _payload_rows(payload)
                    st.session_state.provider_curve_raw_table = _flatten_records(rows) if rows else pd.json_normalize(payload)
            except Exception as exc:
                st.error(f"Provider file could not be read: {exc}")
        raw_table = st.session_state.get("provider_curve_raw_table", pd.DataFrame())
        if isinstance(raw_table, pd.DataFrame) and not raw_table.empty:
            st.caption("Raw provider response. Confirm contract mapping and delivery units before applying this curve.")
            table(raw_table.head(100))
            columns = [c for c in raw_table.columns if not str(c).startswith("_")]
            date_guess = next((c for c in columns if any(t in str(c).lower() for t in ("delivery_month", "deliverydate", "delivery_date", "maturity", "tenor_date", "contract_date", "month"))), columns[0])
            price_guess = next((c for c in columns if any(t in str(c).lower() for t in ("settlement", "settle", "price", "close", "last"))), columns[-1])
            m1, m2, m3 = st.columns(3)
            mapped_product = m1.selectbox("Product to update", list(CURVE_PRODUCTS.keys()), key="provider_curve_product")
            mapped_date_col = m2.selectbox("Delivery date column", columns, index=columns.index(date_guess) if date_guess in columns else 0, key="provider_curve_date_col")
            mapped_price_col = m3.selectbox("Price column", columns, index=columns.index(price_guess) if price_guess in columns else len(columns)-1, key="provider_curve_price_col")
            if st.button("Validate and apply monthly forward curve", key="apply_provider_curve_button"):
                try:
                    candidate_curve = normalise_provider_curve(raw_table, mapped_product, mapped_date_col, mapped_price_col)
                    commit_custom_curve(mapped_product, candidate_curve, {"source_label": "Licensed provider", "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source_type": "REST endpoint"})
                    st.session_state["flash_message"] = f"Applied {len(candidate_curve):,} marks to {mapped_product}."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Curve rejected: {exc}")
        else:
            st.info("No licensed provider table loaded yet. Use the endpoint connector or upload a permitted CSV/JSON export.")
        st.caption("Provider APIs have vendor-specific URLs, response schemas, authentication and redistribution terms. If your licence uses a different authentication method or schema, adapt the connector to that official specification; do not paste vendor secrets into source code.")


def page_data_assumptions():
    title_block("Data and assumptions", "Products, curve coverage, conversions, saved data and the limits of this model.")
    st.subheader("Products")
    registry = pd.DataFrame([{"Product": p, "Commodity": s["commodity"], "Zone": s["zone"], "Currency": s["currency"], "Unit": s["unit"],
                              "Profile": s.get("profile", "n/a").title(), "Timezone": s["timezone"], "Tradable": p in PRODUCTS} for p, s in CURVE_PRODUCTS.items()])
    table(registry)
    st.subheader("Curves in use")
    rows = []
    for product_name, product_spec in CURVE_PRODUCTS.items():
        product_curve = curves[product_name]
        imported = product_name in st.session_state.get("custom_curves", {})
        dates = pd.to_datetime(product_curve.get("delivery_month", pd.Series(dtype="datetime64[ns]")), errors="coerce")
        rows.append({"Product": product_name, "Prices": product_price_state(product_name), "Source": ", ".join(sorted(set(curve_sources().get(product_name, {}).values()))), "Monthly marks": int(len(product_curve)),
                     "First delivery": dates.min().date() if len(dates.dropna()) else None, "Last delivery": dates.max().date() if len(dates.dropna()) else None,
                     "Imported (UTC)": st.session_state.get("curve_metadata", {}).get(product_name, {}).get("imported_at_utc", "") if imported else ""})
    table(pd.DataFrame(rows))
    st.subheader("Saved data")
    store = desk_store()
    history = saved_snapshots()
    days = int(history["snapshot_date"].nunique()) if not history.empty else 0
    live_rows = []
    for dataset_key, item in st.session_state.get("live_data", {}).items():
        frame = item.get("data") if isinstance(item, dict) else None
        is_loaded = isinstance(frame, pd.DataFrame) and not frame.empty
        live_rows.append({"Dataset": item.get("label", dataset_key), "Rows": len(frame) if isinstance(frame, pd.DataFrame) else 0,
                          "Latest observation": _latest_observation_date(frame) if is_loaded else None, "Fetched (UTC)": item.get("fetched_at", "")})
    if store is not None:
        st.write(f"Your book, curves, {days} day{'s' if days != 1 else ''} of curve history and fetched data are saved in `{store.path}`. API keys are never written there. Copy this file to back up the desk.")
    else:
        st.warning(f"Saving is unavailable, so changes last until this tab closes. {st.session_state.get('desk_store_error', '')}")
    if live_rows:
        table(pd.DataFrame(live_rows))
    st.subheader("Unit converter")
    a, b, c = st.columns(3)
    mwh = a.number_input("Energy (MWh)", min_value=0.0, value=1000.0, step=100.0)
    eur_mwh = b.number_input("Price (EUR/MWh)", min_value=-500.0, max_value=10000.0, value=35.0, step=1.0)
    fx = c.number_input("EUR per 1 GBP", min_value=0.01, value=round(fx_rate("eur_per_gbp"), 4), step=0.01, format="%.4f", key="data_fx")
    st.write(f"{mwh:,.2f} MWh is {mwh_to_mmbtu(mwh):,.2f} MMBtu. {eur_mwh:.2f} EUR/MWh is {price_eur_mwh_to_gbp_therm(eur_mwh, fx) * 100:.2f} p/therm at this rate.")
    with st.expander("Model conventions and limits", expanded=True):
        st.markdown("""
- **End dates:** the engine treats delivery ends as exclusive, so 1 January to 1 February is January only. Trade forms ask for the last delivery day and convert it.
- **Volume:** MWh is MW times delivery hours. Base uses actual local hours, including 23 and 25-hour days. Peak is weekdays 08:00 to 20:00 local time.
- **Calendar:** public holidays, exchange-specific peak rules and the 06:00 gas day aren't modelled.
- **Profiles:** each power product has one delivery shape, so a base product can't be traded as peak, or the reverse.
- **MtM:** signed MWh times (forward mark minus fixed price). Buy fixed is long, sell fixed is short. Values are undiscounted.
- **Curves:** every delivery month needs a mark. Missing marks leave a trade unpriced. Prices are never interpolated or invented.
- **Currency:** EUR and GBP stay separate until a page shows a total in EUR, at the rate shown there.
- **NBP:** trades in GBP/MWh. Convert p/therm quotes before importing.
- **Daily P&L:** compares saved curve days. Months leaving the curve count as realised at their last mark, which isn't a settlement price.
- **VaR:** first-order historical simulation of today's positions. It depends entirely on the days saved.
- **Not modelled:** credit, collateral, bid/ask, discounting, shaping premiums, imbalance, fees and exchange settlement rules.
""")
    st.subheader("Reset")
    st.caption("These can't be undone. Fetched market data stays either way.")
    c1, c2 = st.columns(2)
    if c1.button("Reset the demo book and imported curves", key="reset_book", width="stretch"):
        st.session_state.custom_curves = {}
        st.session_state.curve_metadata = {}
        bump("curves")
        bump("snapshots")
        st.session_state.pop("last_quote", None)
        st.session_state.pop("provider_curve_raw_table", None)
        st.session_state.pop("snapshot_day", None)
        _save(lambda s: s.clear_custom_curves())
        commit_trades(build_sample_trades())
        commit_ops(build_sample_physical_ops(st.session_state.trades))
        _save(lambda s: [s.save_snapshot(valuation_date, make_sample_curve(p, valuation_date, DEMO_CURVE_MONTHS), "Demo marks", replace=True) for p in CURVE_PRODUCTS])
        st.session_state["flash_message"] = "The demo book and demo curves are back."
        st.rerun()
    if c2.button("Delete all saved curve history", key="reset_history", width="stretch"):
        _save(lambda s: s.delete_snapshots())
        bump("snapshots")
        st.session_state.pop("snapshot_day", None)
        st.session_state["flash_message"] = "Curve history deleted. Today's curves will be saved again."
        st.rerun()


def page_model_validation():
    title_block("Model checks", "Automatic tests of the formulas and data handling. They test this software, not any trading strategy.")
    c1, _ = st.columns([1, 3])
    if c1.button("Run checks again", key="run_validation", width="stretch") or "validation_report" not in st.session_state:
        try:
            with st.spinner("Running checks"):
                st.session_state.validation_report = run_model_validation(st.session_state.trades.copy(), curves)
        except Exception as exc:
            st.error(f"The checks could not finish: {type(exc).__name__}: {exc}")
            return
    validation = st.session_state.validation_report
    passed = int(validation["Status"].eq("PASS").sum())
    failed = int(validation["Status"].eq("FAIL").sum())
    k1, k2, k3 = st.columns(3)
    kpi(k1, "Checks run", str(len(validation)))
    kpi(k2, "Passed", str(passed), tone="gain" if passed else "")
    kpi(k3, "Failed", str(failed), tone="loss" if failed else "")
    if failed:
        st.error(f"{failed} check(s) failed. Read the details below before relying on the affected numbers.")
    table(validation)
    st.download_button("Download the report as CSV", data=validation.to_csv(index=False).encode("utf-8"), file_name="energy_model_validation_report.csv", mime="text/csv")
    st.caption("Run the same suite outside the app with: pytest etttr.py. These checks don't validate real exchange data, holidays, full swap payment schedules, discount curves, accounting policy, collateral, credit adjustments, or provider access.")
    current_failures = validation.loc[validation["Status"].eq("FAIL") & validation["Area"].eq("Current user data")]
    if not current_failures.empty:
        st.warning("At least one failure concerns your own data rather than the formulas. Check that trade's price in Trading Book.")


# =============================================================================
# SECTION 9 · APP ENTRY POINT
# =============================================================================
PAGES = {
    "Market Overview": page_market_overview,
    "Morning Brief": page_morning_brief,
    "Market Read": page_market_read,
    "Market Fundamentals": page_market_fundamentals,
    "Power Spot": page_power_spot,
    "Spreads & Margins": page_spreads,
    "Forward Curves": page_forward_curves,
    "OTC Quote Builder": page_otc_quote_builder,
    "Trading Book & MtM": page_trading_book_mtm,
    "Daily P&L": page_daily_pnl,
    "Risk & Exposure": page_risk_exposure,
    "Hedge Advisor": page_hedge_advisor,
    "Physical Operations": page_physical_operations,
    "Live Data Integrations": page_live_data_integrations,
    "Data & Assumptions": page_data_assumptions,
    "Model Validation": page_model_validation,
}


def _configure_theme() -> None:
    """Set the dark theme from code, so the app needs no config file beside it."""
    wanted = {"theme.base": "dark", "theme.primaryColor": SODIUM, "theme.backgroundColor": BG,
              "theme.secondaryBackgroundColor": PANEL, "theme.textColor": TEXT, "theme.borderColor": BORDER,
              "theme.font": "Archivo, sans-serif", "client.toolbarMode": "minimal"}
    changed = False
    for option, value in wanted.items():
        try:
            if st.get_option(option) != value:
                st.config.set_option(option, value)
                changed = True
        except Exception:
            continue
    # Streamlit sends the theme when a run starts, so the very first run needs one repeat.
    if changed and not st.session_state.get("theme_rerun_done"):
        st.session_state.theme_rerun_done = True
        st.rerun()


def main() -> None:
    global valuation_date, curves
    version = tuple(int(part) for part in st.__version__.split(".")[:2] if part.isdigit())
    if version < MIN_STREAMLIT_VERSION:
        st.error(f"This app needs Streamlit {MIN_STREAMLIT_VERSION[0]}.{MIN_STREAMLIT_VERSION[1]} or newer, and {st.__version__} is installed. Run: pip install -U streamlit")
        st.stop()
    _configure_theme()
    st.set_page_config(page_title="Meridian Energy Desk", page_icon=":material/bolt:", layout="wide", initial_sidebar_state="auto")
    st.markdown(_CSS, unsafe_allow_html=True)
    init_session()
    collect_update()
    valuation_date = date.today()
    if not st.session_state.get("curve_folder_checked"):
        st.session_state.curve_folder_checked = True
        import_curve_folder()
    curves = build_curves()
    ensure_daily_snapshot()
    maybe_auto_update()
    try:
        record_market_reads()
    except Exception as exc:
        st.session_state.desk_store_error = f"Read record: {type(exc).__name__}: {exc}"
    # Apply queued navigation before the sidebar widgets exist.
    pending_nav_page = st.session_state.pop("pending_nav_page", None)
    if pending_nav_page:
        set_nav_page(str(pending_nav_page))
    page = render_sidebar()
    st.markdown(page_accent_css(st.session_state.nav_page), unsafe_allow_html=True)
    flash_message = st.session_state.pop("flash_message", None)
    if flash_message:
        st.toast(str(flash_message), icon=":material/check_circle:")
    PAGES[page]()
    render_next_step(page)


def test_model_validation_suite() -> None:
    """Lets `pytest etttr.py` run the whole check suite without opening the app."""
    day = date.today()
    report = run_model_validation(build_sample_trades(day), {p: make_sample_curve(p, day, DEMO_CURVE_MONTHS) for p in CURVE_PRODUCTS})
    failures = report.loc[report["Status"].eq("FAIL")]
    assert failures.empty, failures.to_string()


if __name__ == "__main__":
    main()
