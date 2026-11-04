"""Meridian Energy Desk: European gas and power trading and risk.

Run with:   streamlit run etttr.py          (needs Python 3.10+ and: pip install -U streamlit pandas plotly numpy)
Test with:  pytest etttr.py

Everything is in this one file, in order:
  1 Calculation engine      units, calendars, curves, trades, valuation
  2 Analytics engine        tenors, curve building, P&L explain, risk, spreads, spot, seasonal
  3 Saved data              SQLite file beside this script (meridian_desk.db)
  4 Market data connectors  ENTSO-E, GIE, ENTSOG, Open-Meteo, ECB, licensed providers
  5 Design system           colours, styles, chart template, shared components
  6 Navigation and session
  7 Model checks
  8 Pages
  9 App entry point

The sections above the pages import without opening the app. Built-in curves and trades
are synthetic demo data, not market quotes. This is an educational prototype, not a
production ETRM or valuation system. API keys are embedded in plain text: keep this file private.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, time, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo
import gzip
import hashlib
import html
import io
import json
import math
import os
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

MIN_STREAMLIT_VERSION = (1, 50)

# =============================================================================
# SECTION 1 · CALCULATION ENGINE
# Units, delivery calendars, curves, trades, mark-to-market and operations.
# =============================================================================
MWH_PER_MMBTU = 1.0 / 3.412141633
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
    for day, product, month, price in zip(history["snapshot_date"], history["product"],
                                          history["delivery_month"], history["price"]):
        result.setdefault(_to_date(day), {}).setdefault(product, {})[_to_date(month)] = float(price)
    return result


def curves_lookup(curves: dict[str, pd.DataFrame]) -> dict:
    return {product: _curve_lookup(validate_curve(curve)) for product, curve in curves.items()}


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
    """Daily P&L and open MtM in EUR for every consecutive pair of saved curve days."""
    columns = ["date", "pnl_eur", "mtm_eur", "cumulative_pnl_eur"]
    days = sorted(lookups)
    rows = []
    for previous_day, day in zip(days, days[1:]):
        step = pnl_explain(buckets, lookups[previous_day], lookups[day], previous_day, day)
        if step.empty:
            rows.append({"date": day, "pnl_eur": 0.0, "mtm_eur": 0.0})
            continue
        factor = step["currency"].map({"EUR": 1.0, "GBP": float(eur_per_gbp)})
        rows.append({"date": day, "pnl_eur": float((step["total_pnl"] * factor).sum()),
                     "mtm_eur": float((step["current_mtm"] * factor).sum())})
    out = pd.DataFrame(rows, columns=["date", "pnl_eur", "mtm_eur"])
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
                   confidence_levels: Sequence[float] = (0.95, 0.99)) -> dict:
    """One-day historical-simulation VaR and expected shortfall of today's positions.

    Each saved day's move in a contract is mapped to its distance from delivery, then
    applied to the bucket at the same distance today. Only same-contract moves are used,
    so a contract rolling to the next month never registers as a price move.
    """
    empty = {"pnl": pd.DataFrame(columns=["date", "pnl_eur"]), "var": {}, "es": {}, "uncovered": [], "observations": 0}
    if buckets is None or buckets.empty or history is None or history.empty:
        return empty
    data = history[["snapshot_date", "product", "delivery_month", "price"]].copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])
    data["delivery_month"] = pd.to_datetime(data["delivery_month"])
    data = data.sort_values(["product", "delivery_month", "snapshot_date"])
    grouped = data.groupby(["product", "delivery_month"])
    data["move"] = grouped["price"].diff()
    data["gap_days"] = grouped["snapshot_date"].diff().dt.days
    data = data.dropna(subset=["move"]).loc[lambda f: f["gap_days"].le(5)]
    if data.empty:
        return empty
    data["tenor"] = ((data["delivery_month"].dt.year - data["snapshot_date"].dt.year) * 12
                     + data["delivery_month"].dt.month - data["snapshot_date"].dt.month)
    moves = {product: frame.pivot_table(index="snapshot_date", columns="tenor", values="move")
             for product, frame in data.groupby("product")}
    today = date.today()
    position = buckets.groupby(["product", "currency", "delivery_month"], as_index=False)["net_mwh"].sum()
    all_days = sorted({day for frame in moves.values() for day in frame.index})
    pnl = pd.Series(0.0, index=all_days)
    uncovered = []
    for row in position.to_dict("records"):
        frame = moves.get(row["product"])
        if frame is None or frame.empty:
            uncovered.append(row["product"])
            continue
        tenor = (row["delivery_month"].year - today.year) * 12 + row["delivery_month"].month - today.month
        # A day missing this exact distance from delivery borrows the closest one it has.
        filled = frame.sort_index(axis=1).ffill(axis=1).bfill(axis=1)
        nearest = min(filled.columns, key=lambda t: abs(int(t) - tenor))
        series = filled[nearest].reindex(all_days).fillna(0.0)
        pnl = pnl + series * float(row["net_mwh"]) * (float(eur_per_gbp) if row["currency"] == "GBP" else 1.0)
    result = pd.DataFrame({"date": [d.date() for d in pnl.index], "pnl_eur": pnl.values})
    var, es = {}, {}
    for level in confidence_levels:
        threshold = float(np.quantile(result["pnl_eur"], 1.0 - float(level)))
        var[float(level)] = max(-threshold, 0.0)
        tail = result.loc[result["pnl_eur"].le(threshold), "pnl_eur"]
        es[float(level)] = max(-float(tail.mean()), 0.0) if not tail.empty else var[float(level)]
    return {"pnl": result, "var": var, "es": es, "uncovered": sorted(set(uncovered)), "observations": len(result)}


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
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out["value"] = _number_series(out["value"])
    out = out.dropna(subset=["timestamp", "value"]).sort_values("timestamp").drop_duplicates("timestamp")
    if out.empty:
        return pd.DataFrame(columns=columns)
    if "resolution" in out.columns:
        out["hours"] = out["resolution"].map(lambda r: pd.to_timedelta(r).total_seconds() / 3600.0 if isinstance(r, str) and r else float("nan"))
    else:
        out["hours"] = float("nan")
    fallback = out["timestamp"].diff().dt.total_seconds().div(3600).median()
    out["hours"] = out["hours"].fillna(fallback if pd.notna(fallback) and fallback > 0 else 1.0)
    out["local"] = out["timestamp"].dt.tz_convert(timezone_name)
    return out[columns].reset_index(drop=True)


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
            """)

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
        payload = gzip.compress(frame.to_json(orient="split", date_format="iso", date_unit="s", index=False).encode("utf-8"), compresslevel=5)
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO live_data VALUES (?,?,?,?,?)",
                         (key, str(item.get("label", key)), json.dumps(item.get("metadata", {}), default=str),
                          str(item.get("fetched_at", "")), payload))
        return True

    def load_live_items(self) -> dict:
        items = {}
        with self._connect() as conn:
            for key, label, metadata, fetched_at, payload in conn.execute("SELECT * FROM live_data"):
                try:
                    frame = pd.read_json(io.StringIO(gzip.decompress(payload).decode("utf-8")), orient="split", convert_dates=False)
                except Exception:
                    continue
                if "timestamp" in frame.columns:
                    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
                for column in ("date", "local_today"):
                    if column in frame.columns:
                        frame[column] = pd.to_datetime(frame[column], errors="coerce")
                items[key] = {"label": label, "data": frame, "fetched_at": fetched_at, "metadata": json.loads(metadata)}
        return items

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


def commit_trades(trades: pd.DataFrame) -> None:
    st.session_state.trades = trades
    st.session_state.physical_ops = sync_physical_ops_with_trades(st.session_state.get("physical_ops"), trades)
    _save(lambda s: (s.save_table("trades", trades), s.save_table("physical_ops", st.session_state.physical_ops)))


def commit_ops(operations: pd.DataFrame) -> None:
    st.session_state.physical_ops = operations
    _save(lambda s: s.save_table("physical_ops", operations))


def commit_custom_curve(product: str, curve: pd.DataFrame, metadata: dict) -> None:
    """Apply an imported curve for pricing and record it as today's saved marks."""
    st.session_state.custom_curves[product] = curve
    st.session_state.curve_metadata[product] = metadata
    st.session_state.pop("last_quote", None)
    _save(lambda s: (s.save_custom_curve(product, curve, metadata),
                     s.save_snapshot(date.today(), curve, str(metadata.get("source_label", "Imported curve")), replace=True)))


def fx_rate(key: str) -> float:
    return float(st.session_state.get("settings", {}).get(key, DEFAULT_SETTINGS[key]))


def set_fx_rate(key: str, value: float) -> None:
    st.session_state.setdefault("settings", {})[key] = float(value)
    _save(lambda s: s.set_setting(key, float(value)))


DEFAULT_SETTINGS = {"eur_per_gbp": 1.17, "eur_per_usd": 0.92}


# =============================================================================
# SECTION 4 · MARKET DATA CONNECTORS
# Requests run only on a user action. Secrets and environment variables override
# the credentials embedded below. Keep this file private; they are in plain text.
# =============================================================================
HTTP_TIMEOUT_SECONDS = 25
MAX_API_RESPONSE_BYTES = 20 * 1024 * 1024
ENTSOE_MAX_DAYS = 1096
ECB_FX_BASE = "https://data-api.ecb.europa.eu/service/data/EXR"
OPEN_METEO_ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
GIE_AGSI_API_KEY_DEFAULT = "76556429239ab949472fb6f5714c9017"
ENTSOE_API_TOKEN_DEFAULT = "65be7143-4662-4381-a527-45492d59fb9b"
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
    "B01": "Biomass", "B02": "Lignite", "B03": "Hard coal", "B04": "Natural gas",
    "B05": "Hard coal", "B06": "Oil", "B07": "Oil shale", "B08": "Peat",
    "B09": "Geothermal", "B10": "Hydro pumped storage", "B11": "Hydro run-of-river",
    "B12": "Hydro reservoir", "B13": "Marine", "B14": "Nuclear", "B15": "Other renewable",
    "B16": "Solar", "B17": "Waste", "B18": "Wind offshore", "B19": "Wind onshore",
    "B20": "Other", "B21": "AC link", "B22": "DC link", "B23": "Substation", "B24": "Energy storage",
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
    return series.map(one)


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


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_text(parent, names: Sequence[str]) -> Optional[str]:
    wanted = set(names)
    for node in parent.iter():
        if _xml_local_name(node.tag) in wanted and node.text and node.text.strip():
            return node.text.strip()
    return None


def _xml_direct_children(parent, name: str):
    return [child for child in list(parent) if _xml_local_name(child.tag) == name]


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
    out, col = _frame_with_time(frame, date_candidates)
    if out.empty:
        return None
    if col is None:
        return None
    try:
        value = out.iloc[-1][col]
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC")
        return stamp.strftime("%Y-%m-%d %H:%M UTC" if (stamp.hour or stamp.minute or stamp.second) else "%Y-%m-%d")
    except Exception:
        return str(out.iloc[-1][col])


def _series_value_at_latest_timestamp(frame: pd.DataFrame, series_names: Sequence[str]) -> Optional[float]:
    """Aggregate selected ENTSO-E series at the latest timestamp common to those rows."""
    if frame is None or frame.empty or "value" not in frame.columns:
        return None
    time_col = _find_column(frame, ("timestamp",))
    series_col = _find_column(frame, ("series", "name"))
    if not time_col or not series_col:
        return None
    out = frame.copy()
    out["_time"] = pd.to_datetime(out[time_col], errors="coerce", utc=True).dt.floor("h")
    out["_value"] = _number_series(out["value"])
    wanted = out[series_col].astype(str).str.casefold().apply(
        lambda text: any(name.casefold() in text for name in series_names)
    )
    out = out.loc[wanted & out["_time"].notna() & out["_value"].notna()]
    if out.empty:
        return None
    latest_time = out["_time"].max()
    latest = out.loc[out["_time"].eq(latest_time)]
    return float(latest.groupby(series_col)["_value"].last().sum())


def build_live_snapshot() -> dict[str, dict]:
    """Summarise current session feeds for dashboards, without fabricating missing values."""
    snapshot: dict[str, dict] = {}
    # GIE AGSI+ gas storage. EU type requests contain an aggregate; country requests stay country-level.
    agsi = _live_dataset_frame("gie_agsi")
    if not agsi.empty:
        full, row, _ = _latest_metric(agsi, ("full", "fillLevel", "fill_level"), prefer_aggregate=True)
        storage, storage_row, _ = _latest_metric(agsi, ("gasInStorage", "gas_in_storage"), prefer_aggregate=True)
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
        sendout, _, sendout_col = _latest_metric(alsi, ("sendOut", "sendout", "send_out", "sendout_gwh"), prefer_aggregate=True)
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
    """Small dependency-free HTTP GET. Error messages never echo query tokens or headers."""
    final_url = _build_url(url, params)
    req_headers = {"User-Agent": "MERIDIAN-Energy-Desk/2.0", "Accept": "application/json,*/*"}
    if headers:
        req_headers.update({str(k): str(v) for k, v in headers.items() if v is not None})
    request = Request(final_url, headers=req_headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read(MAX_API_RESPONSE_BYTES + 1)
            if len(body) > MAX_API_RESPONSE_BYTES:
                raise RuntimeError("API response exceeded the 20 MB safety limit. Narrow the requested dates or filters.")
            return body, content_type
    except HTTPError as exc:
        # Do not include exc.url because some vendors require tokens in URL query strings.
        try:
            detail = exc.read(8192)
        except Exception:
            detail = b""
        raise ProviderHTTPError(f"API returned HTTP {exc.code}. Check access rights, credentials, dates and query parameters.", exc.code, detail) from None
    except URLError as exc:
        reason = getattr(exc, "reason", "network error")
        raise RuntimeError(f"Could not reach the data provider ({reason}). Check your internet connection and provider availability.") from None
    except TimeoutError:
        raise RuntimeError("The data provider timed out. Try a shorter date range.") from None


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
    if any(_xml_local_name(n.tag) == "Acknowledgement_MarketDocument" for n in root.iter()):
        reason = _xml_text(root, ["text"]) or _xml_text(root, ["code"]) or "No matching market document was returned."
        raise RuntimeError(f"ENTSO-E acknowledgement: {reason}")
    rows = []
    kind = series_kind.lower()
    for ts in [n for n in root.iter() if _xml_local_name(n.tag) == "TimeSeries"]:
        psr_code = _xml_text(ts, ["psrType"]) or ""
        curve_type = _xml_text(ts, ["curveType"]) or ""
        series_name = ENTSOE_PSR_NAMES.get(psr_code, psr_code or ("Day-ahead price" if kind == "price" else kind.title()))
        children = {_xml_local_name(child.tag) for child in ts}
        if kind == "generation" and "outBiddingZone_Domain.mRID" in children:
            series_name = f"{series_name} (consumption)"
        periods = [n for n in ts.iter() if _xml_local_name(n.tag) == "Period"]
        for period in periods:
            interval_node = next((n for n in period.iter() if _xml_local_name(n.tag) == "timeInterval"), None)
            start_txt = _xml_text(interval_node, ["start"]) if interval_node is not None else _xml_text(period, ["start"])
            end_txt = _xml_text(interval_node, ["end"]) if interval_node is not None else None
            resolution = _xml_text(period, ["resolution"])
            if not start_txt or not resolution:
                continue
            try:
                period_start = pd.Timestamp(start_txt)
                period_start = period_start.tz_localize("UTC") if period_start.tzinfo is None else period_start.tz_convert("UTC")
                delta = _parse_duration(resolution)
            except Exception:
                continue
            values: dict[int, float] = {}
            for point in [n for n in period.iter() if _xml_local_name(n.tag) == "Point"]:
                pos_txt = _xml_text(point, ["position"])
                value_txt = _xml_text(point, ["price.amount", "quantity"] if kind == "price" else ["quantity", "price.amount"])
                if not pos_txt or value_txt is None:
                    continue
                try:
                    values[int(pos_txt)] = float(value_txt.replace(",", "."))
                except ValueError:
                    continue
            if not values:
                continue
            if curve_type == "A03":
                # A missing position repeats the previous value until the period ends.
                last_position = max(values)
                if end_txt:
                    try:
                        period_end = pd.Timestamp(end_txt)
                        period_end = period_end.tz_localize("UTC") if period_end.tzinfo is None else period_end.tz_convert("UTC")
                        last_position = max(last_position, int((period_end - period_start) / delta))
                    except Exception:
                        pass
                filled, current = {}, None
                for position in range(min(values), last_position + 1):
                    current = values.get(position, current)
                    filled[position] = current
                values = filled
            for position, value in values.items():
                rows.append({
                    "timestamp": period_start + (position - 1) * delta,
                    "value": value,
                    "series": series_name,
                    "psr_code": psr_code,
                    "resolution": resolution,
                    "unit": "EUR/MWh" if kind == "price" else "MW",
                    "_step": delta,
                })
    if not rows:
        msg = _xml_text(root, ["text"]) or "No observations found in ENTSO-E XML response."
        raise RuntimeError(f"ENTSO-E returned no usable observations. {msg}")
    # Where a document repeats a slot (for example 15- and 60-minute auctions), keep the finest.
    result = (pd.DataFrame(rows).sort_values(["timestamp", "series", "_step"])
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

    Long periods are split into 30-day requests and joined, so up to three years can be
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
        query_label = f"{source_zone} to {target_zone} physical flow"
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
        else:
            raise ValueError("Unsupported ENTSO-E series type.")
        query_label = f"{zone_name} {series_kind.lower()}"
    # ENTSO-E expects UTC period boundaries in yyyyMMddHHmm format.
    zone = ZoneInfo(ENTSOE_ZONES.get(zone_name, {}).get("timezone", "Europe/Paris"))
    request_http = _http_fn or _http_get_bytes
    windows = []
    cursor = start_date
    while cursor < end_date_exclusive:
        following = min(cursor + timedelta(days=30), end_date_exclusive)
        windows.append((cursor, following))
        cursor = following
    frames, no_data = [], ""
    for index, (window_start, window_end) in enumerate(windows):
        params = dict(base_params)
        params.update({
            "documentType": document_type,
            "periodStart": datetime.combine(window_start, time.min, tzinfo=zone).astimezone(timezone.utc).strftime("%Y%m%d%H%M"),
            "periodEnd": datetime.combine(window_end, time.min, tzinfo=zone).astimezone(timezone.utc).strftime("%Y%m%d%H%M"),
            "securityToken": token.strip(),
        })
        try:
            body, _ = request_http(ENTSOE_BASE, params=params, headers={"Accept": "application/xml, application/zip, */*"})
            frames.append(parse_entsoe_xml(body, parser_kind))
        except ProviderHTTPError as exc:
            message = _entsoe_error_message(exc)
            if "no matching data" not in message.casefold():
                raise RuntimeError(message) from None
            no_data = message
        except RuntimeError as exc:
            if "no matching data" not in str(exc).casefold():
                raise
            no_data = str(exc)
        if progress is not None:
            progress((index + 1) / len(windows))
    if not frames:
        raise RuntimeError(no_data or "ENTSO-E returned no usable observations for this selection.")
    frame = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["timestamp", "series"]).sort_values(["timestamp", "series"]).reset_index(drop=True))
    frame["_source"] = "ENTSO-E Transparency Platform"
    frame["_query"] = query_label
    frame["_fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame["_area"] = query_label
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


def fetch_temperature_normals(location_names: Sequence[str], years: int = 10, _json_get_fn=None) -> tuple[pd.DataFrame, list[str]]:
    """Calendar-day temperature and degree-day normals from Open-Meteo's historical archive."""
    request_json = _json_get_fn or _json_get
    end = date.today() - timedelta(days=7)
    start = date(end.year - int(years), end.month, 1)

    def fetch_one(name: str) -> pd.DataFrame:
        loc = WEATHER_LOCATIONS[name]
        payload = request_json(OPEN_METEO_ARCHIVE_BASE, params={
            "latitude": loc["latitude"], "longitude": loc["longitude"], "start_date": start.isoformat(),
            "end_date": end.isoformat(), "daily": "temperature_2m_mean", "timezone": loc["timezone"]})
        daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
        if not daily.get("time"):
            raise RuntimeError("no archive data returned")
        normals = temperature_normals_from_daily(pd.DataFrame({"date": daily["time"], "temperature": daily.get("temperature_2m_mean", [])}))
        normals["location"] = name
        return normals

    frames, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(fetch_one, name): name for name in location_names}
        for future in as_completed(future_map):
            try:
                frames.append(future.result())
            except Exception as exc:
                failures.append(f"{future_map[future]}: {exc}")
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), failures


def _stored_live_item(source_key: str, label: str, data, metadata: Optional[dict] = None) -> None:
    item = {
        "label": label,
        "data": data,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": metadata or {},
    }
    st.session_state.setdefault("live_data", {})[source_key] = item
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
    return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()


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
BG = "#102A33"          # page
PANEL = "#16353F"       # sidebar, raised surfaces
PANEL_SOFT = "#1B3E49"  # hover, table headers
BORDER = "#2F515C"      # hairlines
TEXT = "#EDF3F2"        # primary ink
GRAY = "#A7BBBE"        # secondary ink
SODIUM = "#F5B83D"      # actions and the current page
GAIN = "#62D9A5"        # profit, long
LOSS = "#FF9483"        # loss, short

# Chart series, validated for colour-blind separation on the page colour.
SERIES = ["#4C97E6", "#C2870C", "#17A891", "#E0703C", "#9180EE", "#6EA62E", "#DA6694"]
BLUE, AMBER, TEAL, ORANGE, PURPLE, GREEN, PINK = SERIES
RED = LOSS

ENTITY_COLORS = {
    "TTF Natural Gas": BLUE, "NBP Natural Gas": TEAL,
    "French Power Base": AMBER, "French Power Peak": ORANGE,
    "German Power Base": PURPLE, "German Power Peak": PINK,
    "EUA Carbon": GREEN, "API2 Coal": "#8FA3A8",
    "France": AMBER, "Germany-Luxembourg": PURPLE, "Netherlands": TEAL, "Belgium": ORANGE,
    "Spain": PINK, "Italy North": GREEN, "Austria": BLUE, "Great Britain": "#8FA3A8",
    "EUR": BLUE, "GBP": TEAL,
}

UI_FONT = "Archivo, 'Helvetica Neue', Arial, sans-serif"
NOTE_FONT = "Newsreader, Georgia, 'Times New Roman', serif"

pio.templates["meridian"] = go.layout.Template(layout=dict(
    colorway=SERIES,
    font=dict(family=UI_FONT, size=13, color=TEXT),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    title=dict(font=dict(size=14, color=TEXT), x=0.004, xanchor="left", xref="container", y=0.985, yanchor="top"),
    xaxis=dict(showgrid=False, linecolor=BORDER, tickcolor=BORDER, ticks="outside", ticklen=4, automargin=True,
               zeroline=False, separatethousands=True, title=dict(font=dict(size=12, color=GRAY)), tickfont=dict(color=GRAY)),
    yaxis=dict(gridcolor="rgba(167,187,190,0.13)", zeroline=True, zerolinecolor=BORDER, linecolor="rgba(0,0,0,0)",
               automargin=True, separatethousands=True, title=dict(font=dict(size=12, color=GRAY)), tickfont=dict(color=GRAY)),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)",
                title=dict(text=""), font=dict(size=12, color=GRAY)),
    hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(family=UI_FONT, size=13, color=TEXT)),
    bargap=0.25,
))
px.defaults.template = "meridian"
px.defaults.color_discrete_sequence = SERIES

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100,400;100,500;100,600;100,700&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
:root {{ color-scheme: dark; }}
html, body, .stApp, button, input, textarea, select, .stMarkdown, [data-testid="stWidgetLabel"] {{ font-family:{UI_FONT}; }}
.stApp {{ background:{BG}; color:{TEXT}; font-size:15px; font-feature-settings:"tnum" 1; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stDeployButton"], [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"] {{ display:none; }}
[data-testid="stMainBlockContainer"] {{ max-width:1360px; padding:2rem 2.5rem 3.5rem; }}
:focus-visible {{ outline:2px solid {SODIUM} !important; outline-offset:2px; }}

h1, h2, h3, h4 {{ color:{TEXT}; font-weight:600; letter-spacing:-0.005em; font-family:{UI_FONT}; }}
.stApp h2, .stApp h3 {{ font-size:1.14rem; font-weight:600; padding:1.3rem 0 0.35rem; line-height:1.3; }}
p, li {{ line-height:1.55; }}
a {{ color:{SODIUM}; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color:{GRAY}; font-size:0.86rem; line-height:1.5; max-width:88ch; }}
hr {{ border-color:{BORDER}; margin:1.2rem 0; }}

/* Page title */
.stApp h1.page-title {{ font-family:{NOTE_FONT}; font-weight:500; font-size:2.25rem; line-height:1.15; color:{TEXT}; margin:0; padding:0; letter-spacing:-0.012em; }}
.page-sub {{ color:{GRAY}; font-size:0.95rem; margin:0.3rem 0 0; max-width:78ch; }}
.pill {{ display:inline-flex; align-items:center; gap:7px; padding:3px 11px; border-radius:999px; font-size:0.8rem; color:{TEXT}; border:1px solid {BORDER}; white-space:nowrap; }}
.pill i {{ width:7px; height:7px; border-radius:50%; display:inline-block; background:{GRAY}; }}
.pill.good i {{ background:{GAIN}; }} .pill.warn i {{ background:{SODIUM}; }}

/* The desk note, the one voice-led element in the app */
.desk-note {{ font-family:{NOTE_FONT}; font-size:1.42rem; line-height:1.42; color:{TEXT}; max-width:34em; margin:0.2rem 0 1rem; }}
.desk-note p {{ margin:0 0 0.7rem; font-size:inherit; line-height:inherit; }}
.desk-note b {{ font-weight:500; font-family:{UI_FONT}; font-size:0.92em; white-space:nowrap; }}
.desk-note .soft {{ color:{GRAY}; font-size:0.86em; }}
.desk-note b.gain {{ color:{GAIN}; }} .desk-note b.loss {{ color:{LOSS}; }}

/* Figures: a label, a number and a thin rule, never a box */
.fig, [data-testid="stMetric"] {{ border-left:1px solid {BORDER}; padding:2px 0 4px 14px; background:none; margin-bottom:0.7rem; }}
.fig .l, [data-testid="stMetricLabel"] p {{ color:{GRAY}; font-size:0.84rem; line-height:1.3; font-weight:400; }}
.fig .v, [data-testid="stMetricValue"] {{ color:{TEXT}; font-size:1.42rem; font-weight:500; line-height:1.3; }}
[data-testid="stMetricValue"] > div {{ font-size:1.42rem; }}
.fig .n {{ color:{GRAY}; font-size:0.8rem; margin-top:1px; line-height:1.35; }}
.fig .v.gain {{ color:{GAIN}; }} .fig .v.loss {{ color:{LOSS}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {BORDER}; width:252px !important; }}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding:0 !important; }}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding:0 12px 24px !important; margin:0 !important; }}
[data-testid="stSidebarHeader"] {{ padding:12px 12px 4px; }}
.brand {{ font-family:{NOTE_FONT}; font-size:1.42rem; font-weight:500; color:{TEXT}; line-height:1.1; display:flex; align-items:center; gap:9px; padding:0 6px; }}
.brand small {{ display:block; font-family:{UI_FONT}; font-size:0.78rem; font-weight:400; color:{GRAY}; margin-top:3px; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] {{ margin-bottom:-4px; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] p {{ color:{GRAY}; font-size:0.8rem; padding:0 8px; margin-top:4px; }}
section[data-testid="stSidebar"] [data-testid="stRadio"], section[data-testid="stSidebar"] [data-testid="stRadio"] > div, section[data-testid="stSidebar"] [role="radiogroup"] {{ width:100% !important; }}
section[data-testid="stSidebar"] [role="radiogroup"] {{ gap:1px; flex-direction:column; }}
section[data-testid="stSidebar"] [role="radiogroup"] > div {{ width:100%; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {{ width:100%; margin:0; padding:5px 8px; border-radius:6px; cursor:pointer; display:block; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{ display:none; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ font-size:0.93rem; color:{TEXT}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] [role="img"] {{ color:{GRAY}; margin-right:4px; font-size:1.05rem !important; vertical-align:-3px; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background:{PANEL_SOFT}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background:rgba(245,184,61,0.13); box-shadow:inset 2px 0 0 {SODIUM}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] [role="img"] {{ color:{SODIUM}; }}
.side-foot {{ color:{GRAY}; font-size:0.8rem; line-height:1.5; padding:0 8px; }}
.side-foot b {{ color:{TEXT}; font-weight:500; }}

/* Tabs: an underline, not a pill bar */
.stTabs [data-baseweb="tab-list"] {{ gap:22px; border-bottom:1px solid {BORDER}; background:none; }}
.stTabs [data-baseweb="tab"] {{ background:none; padding:7px 1px; height:auto; color:{GRAY}; }}
.stTabs [data-baseweb="tab"] p {{ font-size:0.95rem; }}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p {{ color:{TEXT}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{SODIUM}; height:2px; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top:1rem; }}

/* Controls */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{ border-radius:6px; border:1px solid {BORDER}; background:transparent; color:{TEXT}; font-weight:500; min-height:2.45rem; }}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{ border-color:{SODIUM}; color:{SODIUM}; background:transparent; }}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind^="primary"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{ background:{SODIUM}; border-color:{SODIUM}; color:#14262C; }}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button[kind^="primary"]:hover, [data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {{ background:#FFCB5E; border-color:#FFCB5E; color:#14262C; }}
.stButton > button[kind="primary"] p, [data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p {{ color:#14262C; }}
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], .stTextArea textarea {{ background:{PANEL} !important; border-color:{BORDER} !important; border-radius:6px; }}
[data-testid="stWidgetLabel"] p {{ color:{GRAY}; font-size:0.86rem; }}
[data-baseweb="tag"] {{ background:{PANEL_SOFT} !important; border:1px solid {BORDER}; }}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg {{ color:{TEXT} !important; fill:{TEXT}; }}

/* Expanders and notes */
[data-testid="stExpander"] details {{ border:1px solid {BORDER}; border-radius:8px; background:none; }}
[data-testid="stExpander"] summary {{ padding:0.55rem 0.8rem; }}
[data-testid="stExpander"] summary:hover {{ color:{SODIUM}; }}
[data-testid="stAlert"] > div {{ border-radius:8px; }}
[data-testid="stAlertContainer"] {{ background:{PANEL} !important; border:1px solid {BORDER}; color:{TEXT} !important; }}
[data-testid="stAlertContainer"] p {{ color:{TEXT}; }}
[data-testid="stAlertContentWarning"] p::before {{ content:""; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{BORDER}; }}

/* Quote ticket */
.ticket-price {{ font-family:{NOTE_FONT}; font-size:2.5rem; line-height:1.1; font-weight:500; margin:0.15rem 0 0.1rem; }}
.ticket-unit {{ color:{GRAY}; font-size:0.9rem; margin-bottom:0.9rem; }}
.ticket-row {{ display:flex; justify-content:space-between; gap:18px; padding:6px 0; border-top:1px solid {BORDER}; font-size:0.94rem; }}
.ticket-row span:first-child {{ color:{GRAY}; }}

@media (max-width: 720px) {{
  [data-testid="stMainBlockContainer"] {{ padding:3rem 1rem 2.5rem; }}
  .stApp h1.page-title {{ font-size:1.7rem; }}
  .desk-note {{ font-size:1.2rem; }}
}}
</style>
"""


def _styled(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    """Apply the house chart style. Entity colours are fixed, so a filter never repaints a series."""
    resolved_height = height or fig.layout.height or 360
    has_title = bool(fig.layout.title and fig.layout.title.text)
    fig.update_layout(template="meridian", height=resolved_height,
                      margin=dict(l=8, r=12, t=62 if has_title else 34, b=8),
                      hovermode=fig.layout.hovermode or ("x unified" if any(t.type == "scatter" and "lines" in str(t.mode) for t in fig.data) else "closest"))
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title=dict(x=0.004, xanchor="left", xref="container", y=0.985, yanchor="top", font=dict(size=14, color=TEXT)))
    fig.update_xaxes(title_font=dict(size=12, color=GRAY))
    fig.update_yaxes(title_font=dict(size=12, color=GRAY))
    for trace in fig.data:
        name = str(getattr(trace, "name", "") or "")
        color = ENTITY_COLORS.get(name)
        if trace.type == "scatter":
            points = len(trace.x) if trace.x is not None else 0
            if "lines" in str(trace.mode):
                if points > 48 and "markers" in str(trace.mode):
                    trace.mode = "lines"
                if trace.line.width is None:
                    trace.line.width = 2
            if "markers" in str(trace.mode):
                trace.marker.size = trace.marker.size or 8
                trace.marker.line = dict(width=2, color=BG)
            if color:
                if "lines" in str(trace.mode):
                    trace.line.color = color
                trace.marker.color = color
        elif trace.type == "bar":
            if color:
                trace.marker.color = color
            trace.marker.line = dict(width=0)
    return fig


def chart(fig: go.Figure, height: Optional[int] = None) -> None:
    st.plotly_chart(_styled(fig, height), width="stretch", config={"displayModeBar": False, "responsive": True})


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


def money(value: float, currency: str = "EUR") -> str:
    return f"{value:,.2f} {currency}"


def tone_of(value: float) -> str:
    return "gain" if value > 0 else "loss" if value < 0 else ""


def kpi(col, label: str, value: str, sub: str = "", accent: str = "", tone: str = "") -> None:
    """One figure: label, number and optional note, separated by a thin rule."""
    note = f'<div class="n">{html.escape(sub)}</div>' if sub else ""
    col.markdown(f'<div class="fig"><div class="l">{html.escape(label)}</div>'
                 f'<div class="v {tone}">{html.escape(value)}</div>{note}</div>', unsafe_allow_html=True)


def curve_status_pill() -> str:
    user_curves = st.session_state.get("custom_curves", {})
    if user_curves and len(user_curves) >= len(CURVE_PRODUCTS):
        return '<span class="pill good"><i></i>Your imported marks</span>'
    if user_curves:
        return f'<span class="pill warn"><i></i>{len(user_curves)} of {len(CURVE_PRODUCTS)} curves imported</span>'
    return '<span class="pill warn"><i></i>Demo marks, not market prices</span>'


def render_header(title: str, subtitle: str) -> None:
    left, right = st.columns([5, 2], vertical_alignment="top")
    left.markdown(f'<h1 class="page-title">{html.escape(title)}</h1><p class="page-sub">{html.escape(subtitle)}</p>', unsafe_allow_html=True)
    right.markdown(f'<div style="text-align:right;padding-top:8px">{curve_status_pill()}</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:0.7rem"></div>', unsafe_allow_html=True)


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


def need_data_note(message: str) -> None:
    st.info(message, icon=":material/info:")


# =============================================================================
# SECTION 6 · NAVIGATION AND SESSION
# =============================================================================
NAV_SECTIONS = {
    "Market": [":material/dashboard:  Market Overview", ":material/public:  Market Fundamentals",
               ":material/bolt:  Power Spot", ":material/compare_arrows:  Spreads & Margins"],
    "Pricing": [":material/show_chart:  Forward Curves", ":material/request_quote:  OTC Quote Builder"],
    "Book and risk": [":material/menu_book:  Trading Book & MtM", ":material/calendar_today:  Daily P&L",
                      ":material/shield:  Risk & Exposure", ":material/local_shipping:  Physical Operations"],
    "Data and controls": [":material/cable:  Live Data Integrations", ":material/tune:  Data & Assumptions",
                          ":material/rule:  Model Validation"],
}
ALL_NAV_ITEMS = [label for labels in NAV_SECTIONS.values() for label in labels]
DEFAULT_NAV_ITEM = ALL_NAV_ITEMS[0]
NAV_TO_PAGE = {label: label.split("  ", 1)[1] for label in ALL_NAV_ITEMS}

WORKFLOW_NEXT = {
    "Market Overview": "Market Fundamentals",
    "Market Fundamentals": "Power Spot",
    "Power Spot": "Spreads & Margins",
    "Spreads & Margins": "Forward Curves",
    "Forward Curves": "OTC Quote Builder",
    "OTC Quote Builder": "Trading Book & MtM",
    "Trading Book & MtM": "Daily P&L",
    "Daily P&L": "Risk & Exposure",
    "Risk & Exposure": "Physical Operations",
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
        st.markdown('<div class="brand"><span>Meridian<small>European gas and power desk</small></span></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        for section, items in NAV_SECTIONS.items():
            key = f"navsec_{section}"
            if key not in st.session_state:
                st.session_state[key] = current if current in items else None
            st.radio(section, items, key=key, on_change=_on_nav, args=(section,))
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
        loaded = sum(1 for value in st.session_state.get("live_data", {}).values()
                     if isinstance(value, dict) and isinstance(value.get("data"), pd.DataFrame) and not value["data"].empty)
        saved = ("Your work saves automatically." if st.session_state.get("desk_store") is not None and not st.session_state.get("desk_store_error")
                 else "Saving is unavailable, so changes last until you close this tab.")
        feeds = f"{loaded} market dataset{'s' if loaded != 1 else ''} loaded." if loaded else "No market data loaded yet."
        st.markdown(f'<div class="side-foot"><b>{feeds}</b><br>{saved}<br>For analysis only, not for trade execution.</div>', unsafe_allow_html=True)
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
    if saved_trades is None:
        commit_trades(build_sample_trades())
    else:
        st.session_state.trades = saved_trades
        st.session_state.physical_ops = sync_physical_ops_with_trades(saved_ops, saved_trades)
    st.session_state.session_ready = True


def build_curves() -> dict[str, pd.DataFrame]:
    custom = st.session_state.get("custom_curves", {})
    return {product: custom[product] if product in custom else make_sample_curve(product, valuation_date, months=DEMO_CURVE_MONTHS)
            for product in CURVE_PRODUCTS}


def ensure_daily_snapshot() -> None:
    """Record today's marks the first time the desk opens each day. History builds by itself."""
    if st.session_state.get("snapshot_day") == valuation_date:
        return
    for product, curve in curves.items():
        source = str(st.session_state.get("curve_metadata", {}).get(product, {}).get("source_label", "Demo marks")) if product in st.session_state.get("custom_curves", {}) else "Demo marks"
        _save(lambda s, c=curve, src=source: s.save_snapshot(valuation_date, c, src, replace=False))
    st.session_state.snapshot_day = valuation_date


def saved_snapshots() -> pd.DataFrame:
    store = desk_store()
    if store is None:
        return pd.DataFrame(columns=["snapshot_date", "product", "delivery_month", "price", "source"])
    return store.load_snapshots()


def reference_mark(product: str, start: date, end: date, profile: str) -> float:
    return swap_fixed_price(curves[product], start, end, profile)


def month_start(offset: int = 1) -> date:
    return _add_months(valuation_date, offset)


def month_end_exclusive(start: date, months: int) -> date:
    """Delivery end for a tenor counted in months: 15 Dec plus 3 months ends on 15 Mar."""
    return (pd.Timestamp(start) + pd.DateOffset(months=int(months))).date()


def get_marked_trades() -> pd.DataFrame:
    """Mark the book, reusing the result while trades and curves are unchanged."""
    trades = st.session_state.trades
    digest = hashlib.sha1(trades.to_csv(index=False).encode("utf-8"))
    for product in sorted(curves):
        digest.update(product.encode("utf-8"))
        digest.update(curves[product][["delivery_month", "price"]].to_csv(index=False).encode("utf-8"))
    key = digest.hexdigest()
    memo = st.session_state.get("marked_memo")
    if memo and memo[0] == key:
        return memo[1].copy()
    marked = mark_trades_resilient(trades.copy(), curves)
    st.session_state.marked_memo = (key, marked)
    return marked.copy()


def curve_source_label(product: str) -> str:
    if product not in st.session_state.get("custom_curves", {}):
        return "Demo marks"
    metadata = st.session_state.get("curve_metadata", {}).get(product, {})
    return str(metadata.get("source_label", "Imported marks")).strip() or "Imported marks"


def resolved_credentials() -> dict[str, str]:
    """Credentials in priority order: replaced this session, secrets, environment, this file."""
    overrides = st.session_state.get("credential_overrides", {})
    def pick(name: str, secret: str, default: str) -> str:
        return overrides.get(name) or _secret_or_env(secret, secret) or default
    return {"gie": pick("gie", "GIE_API_KEY", GIE_AGSI_API_KEY_DEFAULT),
            "entsoe": pick("entsoe", "ENTSOE_API_TOKEN", ENTSOE_API_TOKEN_DEFAULT),
            "open_meteo": pick("open_meteo", "OPEN_METEO_API_KEY", "")}


def refresh_market_data() -> list[tuple[str, bool, str]]:
    """Fetch the morning data set in one action. One failed source never blocks the others."""
    keys = resolved_credentials()
    today = date.today()
    tasks = {
        "France day-ahead": lambda: fetch_entsoe_series(keys["entsoe"], "Day-ahead price", "France", today - timedelta(days=14), today + timedelta(days=2)),
        "Germany day-ahead": lambda: fetch_entsoe_series(keys["entsoe"], "Day-ahead price", "Germany-Luxembourg", today - timedelta(days=14), today + timedelta(days=2)),
        "EU gas storage": lambda: fetch_gie_history("AGSI", keys["gie"], "Europe / EU datasets", today - timedelta(days=30), today),
        "EU LNG terminals": lambda: fetch_gie_history("ALSI", keys["gie"], "Europe / EU datasets", today - timedelta(days=30), today),
        "Weather forecast": lambda: fetch_open_meteo_market_pack(list(WEATHER_LOCATIONS.keys()), past_days=3, forecast_days=10),
        "ECB FX rates": lambda: (fetch_ecb_rate("GBP"), fetch_ecb_rate("USD")),
    }
    outcomes: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(task): name for name, task in tasks.items()}
        for future in as_completed(future_map):
            try:
                outcomes[future_map[future]] = future.result()
            except Exception as exc:
                outcomes[future_map[future]] = exc
    results = []
    for name in tasks:
        outcome = outcomes[name]
        if isinstance(outcome, Exception):
            results.append((name, False, str(outcome).split(" Check your internet")[0]))
            continue
        if name == "France day-ahead" or name == "Germany day-ahead":
            zone = "France" if name.startswith("France") else "Germany-Luxembourg"
            _stored_live_item(f"entsoe_price|{zone}", f"Day-ahead price, {zone}", outcome, {"zone": zone})
            results.append((name, True, f"{len(outcome):,} prices"))
        elif name == "EU gas storage":
            _stored_live_item("gie_agsi", "AGSI gas storage", outcome, {"scope": "Europe / EU datasets", "country": ""})
            results.append((name, True, f"{len(outcome):,} days"))
        elif name == "EU LNG terminals":
            _stored_live_item("gie_alsi", "ALSI LNG terminals", outcome, {"scope": "Europe / EU datasets", "country": ""})
            results.append((name, True, f"{len(outcome):,} days"))
        elif name == "Weather forecast":
            hourly, daily, failures = outcome
            if daily.empty:
                first = failures[0].split(": ", 2)[-1].split(" Check your internet")[0] if failures else "No rows returned."
                results.append((name, False, f"All {len(failures) or len(WEATHER_LOCATIONS)} locations failed. {first}"))
                continue
            meta = {"locations": list(WEATHER_LOCATIONS.keys()), "location": "Multiple locations", "forecast_days": 10}
            _stored_live_item("weather_hourly", "Open-Meteo hourly weather", hourly, meta)
            _stored_live_item("weather_daily", "Open-Meteo daily weather", daily, meta)
            st.session_state["weather_fetch_failures"] = failures
            results.append((name, True, f"{daily['location'].nunique()} locations"))
        else:
            (gbp_per_eur, _), (usd_per_eur, _) = outcome
            set_fx_rate("eur_per_gbp", 1.0 / gbp_per_eur)
            set_fx_rate("eur_per_usd", 1.0 / usd_per_eur)
            results.append((name, True, f"EUR/GBP {gbp_per_eur:.4f}, EUR/USD {usd_per_eur:.4f}"))
    return results


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
                    "Physical Operations", "Market Fundamentals", "Power Spot", "Spreads & Margins", "Live Data Integrations", "Model Validation", "Data & Assumptions"}
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
        def fake_http(url, params=None, headers=None):
            windows.append(params["periodStart"])
            return (f'<Publication_MarketDocument xmlns="urn:x"><TimeSeries><Period><timeInterval><start>{params["periodStart"][:4]}-{params["periodStart"][4:6]}-{params["periodStart"][6:8]}T{params["periodStart"][8:10]}:00Z</start></timeInterval>'
                    f'<resolution>PT60M</resolution><Point><position>1</position><price.amount>{len(windows)}</price.amount></Point></Period></TimeSeries></Publication_MarketDocument>').encode(), "text/xml"
        frame = fetch_entsoe_series("test", "Day-ahead price", "France", date(2026, 1, 1), date(2026, 4, 1), _http_fn=fake_http)
        if len(windows) != 3 or len(frame) != 3:
            raise AssertionError(f"Expected three 30-day requests, got {len(windows)}.")
        return "A 90-day request is split into three windows and joined into one series."

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
            set_fx_rate("eur_per_gbp", 1.0 / gbp_per_eur)
            set_fx_rate("eur_per_usd", 1.0 / usd_per_eur)
            for suffix in ("gbp", "usd"):
                st.session_state.pop(f"{prefix}_{suffix}", None)
            st.session_state["flash_message"] = f"Using ECB reference rates from {day}."
            st.rerun()
        except Exception as exc:
            st.error(f"Could not get ECB rates: {exc}")
    if abs(gbp - fx_rate("eur_per_gbp")) > 1e-9:
        set_fx_rate("eur_per_gbp", gbp)
    if abs(usd - fx_rate("eur_per_usd")) > 1e-9:
        set_fx_rate("eur_per_usd", usd)
    return gbp, usd


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


def desk_note_html(metrics: dict, snapshot: dict, pnl_today: Optional[float]) -> str:
    """The morning note, written from whatever is actually loaded. Missing data stays unsaid."""
    parts = []
    total = metrics["eur_total"]
    if metrics["executed"]:
        trades = "trade" if metrics["executed"] == 1 else "trades"
        sentence = f'Your book is marked at <b class="{tone_of(total)}">{fmt_money(total)}</b> across {metrics["executed"]} executed {trades}'
        if pnl_today is not None:
            sentence += f', <b class="{tone_of(pnl_today)}">{fmt_money(pnl_today)}</b> on the day'
        parts.append(f"<p>{sentence}.</p>")
    else:
        parts.append("<p>Your book has no executed trades yet.</p>")
    market = []
    storage = snapshot.get("storage", {})
    if storage.get("fill_pct") is not None:
        text = f'EU gas storage is <b>{storage["fill_pct"]:.1f}%</b> full'
        change = storage.get("change_7_reports_twh")
        if change is not None:
            text += f', {"down" if change < 0 else "up"} {abs(change):.1f} TWh in a week'
        market.append(text)
    for zone, label in (("France", "French"), ("Germany-Luxembourg", "German")):
        series = spot_series(_live_dataset_frame(f"entsoe_price|{zone}"), ENTSOE_ZONES[zone]["timezone"])
        if not series.empty:
            today_local = datetime.now(ZoneInfo(ENTSOE_ZONES[zone]["timezone"])).date()
            current = series.loc[series["local"].dt.date.eq(today_local)]
            if not current.empty:
                market.append(f'{label} day-ahead averages <b>{spot_kpis(current)["base"]:.1f} €/MWh</b> today')
    if market:
        parts.append("<p>" + "; ".join(market) + ".</p>")
    else:
        parts.append('<p class="soft">Refresh market data to add storage, day-ahead power and weather to this note.</p>')
    if metrics["unmarked"]:
        word = "trade needs" if metrics["unmarked"] == 1 else "trades need"
        parts.append(f'<p class="soft">{metrics["unmarked"]} {word} a price before it can be valued. See Trading Book.</p>')
    return f'<div class="desk-note">{"".join(parts)}</div>'


def page_market_overview():
    title_block("Today at the desk", f"{valuation_date:%A %d %B %Y}. Your book and the European gas and power markets in one view.")
    marked = get_marked_trades()
    executed = marked.loc[marked["status"].eq("EXECUTED") & marked["mark_status"].eq("MARKED")] if not marked.empty else pd.DataFrame()
    unmarked = int(marked.loc[~marked["status"].isin(["CANCELLED", "EXPIRED"]) & ~marked["mark_status"].eq("MARKED")].shape[0]) if not marked.empty else 0
    eur_mtm = float(executed.loc[executed["currency"].eq("EUR"), "mtm"].sum()) if not executed.empty else 0.0
    gbp_mtm = float(executed.loc[executed["currency"].eq("GBP"), "mtm"].sum()) if not executed.empty else 0.0
    eur_per_gbp = fx_rate("eur_per_gbp")
    metrics = {"eur_total": eur_mtm + gbp_mtm * eur_per_gbp, "executed": int(len(executed)), "unmarked": unmarked}

    pnl_today = None
    lookups = snapshot_lookup(saved_snapshots())
    days = sorted(lookups)
    if len(days) >= 2:
        step = pnl_explain(book_buckets(st.session_state.trades), lookups[days[-2]], lookups[days[-1]], days[-2], days[-1])
        if not step.empty:
            pnl_today = float((step["total_pnl"] * step["currency"].map({"EUR": 1.0, "GBP": eur_per_gbp})).sum())

    snapshot = build_live_snapshot()
    note_col, chart_col = st.columns([1, 1.05], gap="large")
    with note_col:
        st.markdown(desk_note_html(metrics, snapshot, pnl_today), unsafe_allow_html=True)
        if st.button("Refresh market data", type="primary", key="overview_refresh"):
            with st.status("Fetching prices, storage, weather and FX", expanded=True) as status:
                results = refresh_market_data()
                for name, ok, detail in results:
                    st.write(f"{'Done' if ok else 'Failed'}: {name}. {detail}")
                status.update(label="Market data refreshed" if all(ok for _, ok, _ in results) else "Some sources could not be refreshed",
                              state="complete" if all(ok for _, ok, _ in results) else "error", expanded=not all(ok for _, ok, _ in results))
            st.session_state["overview_refresh_results"] = results
            if all(ok for _, ok, _ in results):
                st.rerun()
    with chart_col:
        names = ["TTF Natural Gas", "French Power Base", "German Power Base"]
        frame = pd.concat([curves[n].assign(delivery_month=lambda f: pd.to_datetime(f["delivery_month"])) for n in names], ignore_index=True)
        frame = frame.loc[frame["delivery_month"].lt(pd.Timestamp(_add_months(valuation_date, 25)))]
        fig = px.line(frame, x="delivery_month", y="price", color="product", title="Forward curves, EUR/MWh",
                      labels={"delivery_month": "", "price": "", "product": ""})
        chart(fig, 290)

    st.subheader("Book")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Open MtM, EUR trades", fmt_money(eur_mtm), tone=tone_of(eur_mtm))
    kpi(c2, "Open MtM, GBP trades", fmt_money(gbp_mtm, "GBP"), tone=tone_of(gbp_mtm))
    kpi(c3, "Total in EUR", fmt_money(metrics["eur_total"]), f"At {eur_per_gbp:.4f} EUR per GBP", tone=tone_of(metrics["eur_total"]))
    kpi(c4, "Trades needing a price", f"{unmarked}", "All marked" if not unmarked else "See Trading Book")

    st.subheader("Market")
    storage, lng, load = snapshot.get("storage", {}), snapshot.get("lng", {}), snapshot.get("power_load", {})
    m1, m2, m3, m4 = st.columns(4)
    kpi(m1, "EU gas storage", f"{storage['fill_pct']:.1f}%" if storage.get("fill_pct") is not None else "Not loaded", str(storage.get("date") or ""))
    kpi(m2, "Gas in storage", f"{storage['inventory_twh']:.0f} TWh" if storage.get("inventory_twh") is not None else "Not loaded")
    kpi(m3, "LNG send-out", f"{lng['sendout']:,.0f} GWh/d" if lng.get("sendout") is not None else "Not loaded", str(lng.get("date") or ""))
    kpi(m4, "Actual power load", f"{load['value'] / 1000:.1f} GW" if load.get("value") is not None else "Not loaded", str(load.get("zone") or ""))

    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
    weather_daily = _live_dataset_frame("weather_daily")
    normals = _live_dataset_frame("weather_normals")
    anomaly = hdd_vs_normal(weather_daily, normals)
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
    with st.container(border=True):
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
                _stored_live_item(f"entsoe_price|{zone}", f"Day-ahead price, {zone}", frame, {"zone": zone})
            except Exception as exc:
                failures.append(f"{zone}: {exc}")
            progress.progress((index + 1) / len(selected), text=f"Loaded {zone}")
        progress.empty()
        for failure in failures:
            st.error(failure)
    loaded = {z: spot_series(item["data"], ENTSOE_ZONES[z]["timezone"]) for z, item in _live_variants("entsoe_price").items() if z in ENTSOE_ZONES}
    active = {z: s for z, s in loaded.items() if z in selected and not s.empty}
    if not active:
        need_data_note("Choose your zones and select Fetch prices to load day-ahead data from ENTSO-E.")
        return

    rows = []
    for zone, series in active.items():
        k = spot_kpis(series)
        rows.append({"Zone": zone, "Base avg": k["base"], "Peak avg": k["peak"], "Peak minus base": k["peak_minus_base"], "Low": k["low"], "High": k["high"],
                     "Negative hours": k["negative_hours"], "Hourly volatility": k["volatility"], "Hours": k["hours"]})
    st.caption(f"EUR/MWh. Peak is weekdays 08:00 to 20:00 local time. Averages are weighted by time, so 15-minute and hourly data compare fairly. Data through {max(s['local'].max() for s in active.values()):%d %b %Y}.")
    table(pd.DataFrame(rows))

    tabs = st.tabs(["Prices", "Daily shape", "Heatmap", "Duration curve", "Zone spread", "Wind and solar capture"])
    with tabs[0]:
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
    with tabs[1]:
        frames = []
        for zone, series in active.items():
            frames.append(series.assign(hour=series["local"].dt.hour).groupby("hour", as_index=False)["value"].mean().assign(zone=zone))
        fig = px.line(pd.concat(frames, ignore_index=True), x="hour", y="value", color="zone", title="Average price by local hour of day",
                      labels={"hour": "Local hour", "value": "EUR/MWh", "zone": ""})
        fig.update_xaxes(dtick=2)
        chart(fig, 400)
    with tabs[2]:
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
    with tabs[3]:
        frames = [price_duration_curve(series).assign(zone=zone) for zone, series in active.items()]
        fig = px.line(pd.concat(frames, ignore_index=True), x="share_of_hours", y="price", color="zone", title="Share of hours at or above each price",
                      labels={"share_of_hours": "Share of hours (%)", "price": "EUR/MWh", "zone": ""})
        chart(fig, 400)
    with tabs[4]:
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
    with tabs[5]:
        zone = st.columns([1, 2])[0].selectbox("Zone", list(active), key="spot_capture_zone")
        generation = _live_dataset_frame(f"entsoe_generation|{zone}")
        if st.button(f"Fetch generation for {zone}", key="spot_capture_fetch"):
            series = active[zone]
            start_day = series["local"].min().date()
            end_day = min(series["local"].max().date() + timedelta(days=1), date.today() + timedelta(days=1))
            try:
                with st.spinner("Requesting actual generation"):
                    generation = fetch_entsoe_series(resolved_credentials()["entsoe"], "Actual generation by type", zone, start_day, end_day)
                _stored_live_item(f"entsoe_generation|{zone}", f"Actual generation, {zone}", generation, {"zone": zone})
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
    missing_imports = [p for p in ("EUA Carbon", "API2 Coal") if p not in st.session_state.get("custom_curves", {})]
    if missing_imports:
        st.caption(f"{' and '.join(missing_imports)} use demo marks until you import real curves in Forward Curves.")
    tab_margin, tab_other, tab_history = st.tabs(["Clean spark and dark", "Location, shape and hub", "History"])
    with tab_margin:
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
    with tab_other:
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
    with tab_history:
        history = saved_snapshots()
        lookups = snapshot_lookup(history)
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
    first = curve.iloc[0]
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

    if selected in st.session_state.get("custom_curves", {}):
        meta = st.session_state.get("curve_metadata", {}).get(selected, {})
        st.caption(f"Source: {curve_source_label(selected)}, imported {str(meta.get('imported_at_utc', 'at an unrecorded time'))[:16].replace('T', ' ')} UTC. Check the valuation date and your data licence.")
    else:
        st.caption("These are demo marks, not a market quote. Import real marks below to use this curve for valuation.")
    if spec["commodity"] == "Power":
        zone = "France" if spec["zone"] == "France" else "Germany-Luxembourg"
        series = spot_series(_live_dataset_frame(f"entsoe_price|{zone}"), ENTSOE_ZONES[zone]["timezone"])
        if not series.empty:
            st.caption(f"For context, {zone} day-ahead averaged {spot_kpis(series)['base']:.2f} €/MWh over the loaded period. Day-ahead prices never replace forward marks.")

    tab_table, tab_build, tab_import, tab_history = st.tabs(["Monthly marks", "Build from contract quotes", "Import a CSV", "Import past curves"])
    with tab_table:
        display_curve = curve.copy()
        display_curve["delivery_month"] = display_curve["delivery_month"].dt.date
        if previous:
            earlier_lookup = dict(zip(pd.to_datetime(earlier["delivery_month"]).dt.date, earlier["price"]))
            display_curve["change"] = [p - earlier_lookup[m] if m in earlier_lookup else float("nan") for m, p in zip(display_curve["delivery_month"], display_curve["price"])]
        table(display_curve[[c for c in display_curve.columns if c in ("delivery_month", "price", "change", "unit")]], height=320)
        st.download_button("Download this curve as CSV", data=display_curve.to_csv(index=False).encode("utf-8"),
                           file_name=f"{selected.lower().replace(' ', '_')}_forward_curve.csv", mime="text/csv")
    with tab_build:
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
    with tab_import:
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
    with tab_history:
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
        with st.container(border=True):
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
        tab_blotter, tab_positions, tab_add, tab_status, tab_import = st.tabs(["Blotter", "Positions by month", "Add a trade", "Change status", "Import or export"])
        with tab_blotter:
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
                table(view[[c for c in columns if c in view.columns]])
        with tab_positions:
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
        with tab_add:
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
        with tab_status:
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
        with tab_import:
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
    lookups = snapshot_lookup(history)
    days = sorted(lookups)
    trades = st.session_state.trades
    buckets = book_buckets(trades)
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
    series = pnl_history(buckets, {d: lookups[d] for d in days}, eur_per_gbp)
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
    with st.expander("P&L by trade and delivery month"):
        table(step.drop(columns=["previous_mtm"]))
    st.caption("New trade: today's value of trades done since the first date. Price move: volume times the change in mark. To delivery: a month that left the curve; its value moves from open MtM to realised, so it doesn't change P&L.")


def seed_demo_history() -> None:
    """Fill past days with labelled synthetic marks for products still using demo curves."""
    products = [p for p in CURVE_PRODUCTS if p not in st.session_state.get("custom_curves", {})]
    rows = synthetic_curve_history(valuation_date, products)
    _save(lambda s: s.add_snapshot_rows(rows, replace=False))
    st.session_state["flash_message"] = f"Created {rows['snapshot_date'].nunique()} days of labelled synthetic history."


def page_risk_exposure():
    title_block("Risk and exposure", "Where the book is exposed by delivery month, what named scenarios would do to it, and how much it could lose in a day.")
    trades = st.session_state.trades
    buckets = book_buckets(trades)
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

    tab_ladder, tab_scenarios, tab_var = st.tabs(["Delivery ladder", "Scenarios", "Value at risk"])
    with tab_ladder:
        ladder = buckets.groupby(["product", "delivery_month"], as_index=False)["net_mwh"].sum()
        pivot = ladder.pivot(index="product", columns="delivery_month", values="net_mwh").sort_index()
        limit = float(pivot.abs().max().max()) or 1.0
        fig = go.Figure(go.Heatmap(z=pivot.values, x=[month_contract_label(m) for m in pivot.columns], y=list(pivot.index), xgap=2, ygap=2,
                                   colorscale=[[0, LOSS], [0.5, "#23434E"], [1, BLUE]], zmid=0, zmin=-limit, zmax=limit,
                                   colorbar=dict(title="MWh", thickness=12, outlinewidth=0),
                                   hovertemplate="%{y}, %{x}: %{z:,.0f} MWh<extra></extra>"))
        fig.update_layout(title="Net position by product and delivery month. Blue is long, red is short.", hovermode="closest")
        fig.update_xaxes(type="category")
        chart(fig, max(260, 90 + 44 * len(pivot.index)))
        st.caption("A move of 1 currency unit per MWh in a delivery month changes the value of the book by the volume shown for that month. Volumes count delivery hours, including daylight-saving days.")
        ladder_table = pivot.fillna(0.0)
        ladder_table.columns = [month_contract_label(m) for m in ladder_table.columns]
        table(ladder_table.reset_index().rename(columns={"product": "Product"}))
    with tab_scenarios:
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
    with tab_var:
        history = saved_snapshots()
        outcome = historical_var(buckets, history, eur_per_gbp)
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
    if top_c.button("Manage data sources"):
        st.session_state["pending_nav_page"] = "Live Data Integrations"
        st.rerun()

    if not data_items:
        st.info("No external dataset has been fetched yet. Open Live Data Integrations to connect GIE AGSI+/ALSI, ENTSO-E, ENTSOG and Open-Meteo.")
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
    d4.metric("Latest day-ahead", f"{power['value']:.2f} {power.get('unit', 'EUR/MWh')}" if power.get("value") is not None else "Not loaded",
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
        st.info("Fetch one or more feeds to generate derived observations. Missing data are left blank rather than filled with fabricated values.")

    fund_tabs = st.tabs(["Gas storage", "LNG", "Power", "Gas flows", "Weather", "Cross-market analysis"])
    with fund_tabs[0]:
        st.subheader("EU gas storage")
        history_item = _live_dataset_frame("gie_agsi_history")
        c1, c2 = st.columns([1, 3], vertical_alignment="center")
        if c1.button("Load 6 years of storage", key="load_storage_history", width="stretch"):
            try:
                with st.spinner("Requesting six years of AGSI+ data"):
                    history_item = fetch_gie_history("AGSI", resolved_credentials()["gie"], "Europe / EU datasets", date(date.today().year - 5, 1, 1), date.today())
                _stored_live_item("gie_agsi_history", "AGSI gas storage, 6 years", history_item, {"scope": "Europe / EU datasets", "country": ""})
            except Exception as exc:
                st.error(f"AGSI+ request failed: {exc}")
        c2.caption("Compares this year's fill with last year and the range of the five years before.")
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
            st.info("Fetch AGSI+ in Live Data Integrations. Choose the EU aggregate if you want an aggregate-level series; a country/facility dataset must not be summed blindly.")

    with fund_tabs[1]:
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
            st.info("Fetch ALSI+ in Live Data Integrations. Inventory and send-out use different physical units and must not be combined as if they were comparable.")

    with fund_tabs[2]:
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
            need_data_note("Fetch a price, load, generation or flow series in Live Data, or load prices in Power Spot.")

    with fund_tabs[3]:
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

    with fund_tabs[4]:
        data = _live_dataset_frame("weather_daily")
        st.subheader("Weather")
        weather_for_normals = _live_dataset_frame("weather_daily")
        c1, c2 = st.columns([1, 3], vertical_alignment="center")
        if c1.button("Load 10-year normals", key="load_weather_normals", width="stretch"):
            with st.spinner("Requesting ten years of temperature history"):
                normals_frame, normal_failures = fetch_temperature_normals(list(WEATHER_LOCATIONS))
            if normals_frame.empty:
                st.error("Could not load normals: " + "; ".join(normal_failures))
            else:
                _stored_live_item("weather_normals", "Open-Meteo temperature normals", normals_frame, {"years": 10})
                for failure in normal_failures:
                    st.warning(failure)
        c2.caption("Compares the forecast with what is normal for these dates, using ten years of Open-Meteo history.")
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
            st.info("Fetch a multi-region weather outlook from Live Data Integrations or use Refresh regional weather outlook in Market Overview.")

    with fund_tabs[5]:
        st.subheader("Residual load against day-ahead price")
        ready = residual_ready_zones()
        if ready:
            residual_zone = st.selectbox("Zone", ready, key="residual_zone")
            residual = _build_residual_load_proxy(residual_zone)
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
            need_data_note("Load day-ahead prices, actual load and generation for the same zone and dates. Inputs from different zones are never combined.")

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
    title_block("Live data", "Connect the public transparency APIs and your licensed curve source. Nothing is requested until you ask for it.")
    saved = resolved_credentials()
    with st.expander("Access keys", expanded=False):
        st.caption("Saved keys are never sent to your browser, so these fields stay empty. Paste a key only to replace the saved one for this session. To change a key permanently, set a Streamlit secret or environment variable, which takes priority over the value in the code.")
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

    tab_gie, tab_power, tab_gasflows, tab_weather, tab_forward = st.tabs(["Gas storage and LNG", "Power", "Gas flows", "Weather", "Licensed forward curves"])

    with tab_gie:
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
            st.caption(f"Last fetch: {st.session_state.live_data[gie_show_dataset]['fetched_at']} · Source: {st.session_state.live_data[gie_show_dataset]['label']}")
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

    with tab_power:
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
                _stored_live_item(f"{keys_by_kind[entsoe_kind]}|{variant}", f"{entsoe_kind}, {variant}", frame,
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

    with tab_gasflows:
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
            st.caption(f"Last fetch: {st.session_state.live_data['entsog_flow']['fetched_at']} · Units are supplied by ENTSOG; confirm the unit before interpreting volumes.")
            if "timestamp" in entsog_frame and "value_numeric" in entsog_frame:
                chart_data = entsog_frame.dropna(subset=["timestamp", "value_numeric"]).copy()
                if not chart_data.empty:
                    fig = px.line(chart_data, x="timestamp", y="value_numeric", markers=True, title="ENTSOG daily flow / indicator")
                    fig.update_layout(height=350)
                    chart(fig)
            table(entsog_frame)
            st.download_button("Download ENTSOG data (CSV)", entsog_frame.to_csv(index=False).encode("utf-8"), file_name="entsog_flows.csv", mime="text/csv", key="download_entsog_flow")
        st.markdown("[Open the ENTSOG Transparency Platform](https://transparency.entsog.eu/) for the official catalogue and point details.")

    with tab_weather:
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
            st.caption(f"Last fetch: {st.session_state.live_data['weather_daily']['fetched_at']} · Source: [Open-Meteo](https://open-meteo.com/) · Data licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Public endpoint requires no API key for eligible non-commercial use; commercial use requires the appropriate plan.")
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

    with tab_forward:
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
        rows.append({"Product": product_name, "Source": curve_source_label(product_name), "Monthly marks": int(len(product_curve)),
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
    "Market Fundamentals": page_market_fundamentals,
    "Power Spot": page_power_spot,
    "Spreads & Margins": page_spreads,
    "Forward Curves": page_forward_curves,
    "OTC Quote Builder": page_otc_quote_builder,
    "Trading Book & MtM": page_trading_book_mtm,
    "Daily P&L": page_daily_pnl,
    "Risk & Exposure": page_risk_exposure,
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
    valuation_date = date.today()
    curves = build_curves()
    ensure_daily_snapshot()
    # Apply queued navigation before the sidebar widgets exist.
    pending_nav_page = st.session_state.pop("pending_nav_page", None)
    if pending_nav_page:
        set_nav_page(str(pending_nav_page))
    page = render_sidebar()
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
