"""MERIDIAN Energy Desk — European energy trading and risk management prototype.

Run with: streamlit run meridian_energy_desk_live.py

IMPORTANT: Built-in curves and trades are synthetic demo data, not live market quotes.
This is an educational prototype, not a production ETRM or valuation system.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo
import math
import io
import os
import json
import zipfile
import gzip
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Iterable, Sequence, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Calculation engine: units, curves, trade validation, and mark-to-market
# -----------------------------------------------------------------------------
MWH_PER_MMBTU = 1.0 / 3.412141633
MMBTU_PER_MWH = 3.412141633
THERM_PER_MWH = 34.12141633

PRODUCTS = {
    "TTF Natural Gas": {"commodity": "Gas", "zone": "TTF", "currency": "EUR", "unit": "EUR/MWh", "default_price": 35.0, "timezone": "Europe/Amsterdam"},
    "NBP Natural Gas": {"commodity": "Gas", "zone": "NBP", "currency": "GBP", "unit": "GBP/MWh", "default_price": 29.0, "timezone": "Europe/London"},
    "German Power Base": {"commodity": "Power", "zone": "Germany", "currency": "EUR", "unit": "EUR/MWh", "default_price": 85.0, "timezone": "Europe/Berlin"},
    "French Power Base": {"commodity": "Power", "zone": "France", "currency": "EUR", "unit": "EUR/MWh", "default_price": 78.0, "timezone": "Europe/Paris"},
    "French Power Peak": {"commodity": "Power", "zone": "France", "currency": "EUR", "unit": "EUR/MWh", "default_price": 92.0, "timezone": "Europe/Paris"},
}

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


def delivery_hours(start: date, end_exclusive: date, profile: str = "BASE",
                    timezone_name: str = "Europe/Paris") -> int:
    """Delivery hours for [start, end_exclusive) in the product's market timezone.

    BASE counts actual elapsed hours between consecutive local midnights, so 23/25-hour
    daylight-saving days are handled. PEAK is weekdays 08:00-20:00 local time (12 hours);
    public holidays and exchange-specific exceptions are not included in this prototype.
    """
    if end_exclusive <= start:
        raise ValueError("Delivery end must be after delivery start (end is exclusive).")
    p = str(profile).strip().upper()
    try:
        zone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise ValueError(f"Unknown IANA timezone: {timezone_name}") from exc
    days = pd.date_range(start=start, end=end_exclusive, inclusive="left", freq="D")
    if p == "BASE":
        total = 0
        for ts in days:
            day = ts.date()
            local_start = datetime.combine(day, time.min, tzinfo=zone)
            local_end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=zone)
            total += int((local_end.astimezone(timezone.utc) - local_start.astimezone(timezone.utc)).total_seconds() // 3600)
        return total
    if p == "PEAK":
        return int((days.weekday < 5).sum()) * 12
    raise ValueError("Profile must be BASE or PEAK.")


def make_sample_curve(product: str, valuation_date: date | None = None, months: int = 18) -> pd.DataFrame:
    """Create transparent illustrative monthly forward prices; not live market data."""
    if product not in PRODUCTS:
        raise ValueError(f"Unsupported product: {product}")
    if months < 1 or months > 120:
        raise ValueError("months must be between 1 and 120")
    valuation_date = valuation_date or date.today()
    spec = PRODUCTS[product]
    base = float(spec["default_price"])
    rows = []
    first = pd.Timestamp(valuation_date).to_period("M") + 1
    for i in range(months):
        period = first + i
        month = period.month
        # Smooth, deterministic seasonal shape purely for demonstration.
        season = 0.07 * math.cos(2 * math.pi * (month - 1) / 12)
        contango = 0.0015 * i
        price = base * (1 + season + contango)
        rows.append({
            "delivery_month": period.to_timestamp().date(),
            "product": product,
            "price": round(price, 4),
            "currency": spec["currency"],
            "unit": spec["unit"],
            "data_status": "ILLUSTRATIVE — NOT A LIVE QUOTE",
        })
    return pd.DataFrame(rows)


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
    unknown = sorted(set(out["product"]) - set(PRODUCTS))
    if unknown:
        raise ValueError(f"Unsupported curve product(s): {unknown}")
    if out.duplicated(["product", "delivery_month"]).any():
        raise ValueError("Curve contains duplicate product/delivery-month marks.")
    for product, group in out.groupby("product"):
        expected = PRODUCTS[product]
        if not group["currency"].eq(expected["currency"]).all():
            raise ValueError(f"Currency mismatch for {product}: expected {expected['currency']}.")
        if not group["unit"].eq(expected["unit"]).all():
            raise ValueError(f"Price unit mismatch for {product}: expected {expected['unit']}.")
    return out.sort_values(["product", "delivery_month"]).reset_index(drop=True)


def forward_month_price(curve: pd.DataFrame, product: str, delivery_start: date,
                        delivery_end_exclusive: date) -> float:
    """Return the single monthly mark for one full calendar-month forward."""
    if product not in PRODUCTS:
        raise ValueError(f"Unsupported product: {product}")
    start_ts = pd.Timestamp(delivery_start)
    expected_start = start_ts.to_period("M").start_time.date()
    expected_end = (start_ts.to_period("M") + 1).start_time.date()
    if delivery_start != expected_start or delivery_end_exclusive != expected_end:
        raise ValueError("FORWARD valuation requires exactly one full calendar-month delivery period.")
    validated = validate_curve(curve)
    if not validated["product"].eq(product).all():
        raise ValueError(f"Curve product mismatch for {product}.")
    target = start_ts.to_period("M").start_time.date()
    row = validated.loc[validated["delivery_month"].eq(target)]
    if len(row) != 1:
        raise ValueError(f"Missing or ambiguous forward mark for {target:%Y-%m}.")
    return float(row.iloc[0]["price"])


def swap_fixed_price(curve: pd.DataFrame, delivery_start: date, delivery_end_exclusive: date,
                     profile: str = "BASE") -> float:
    """Weighted fixed price from monthly marks; weighting by delivery hours in each month.

    The curve must contain one mark for every delivery month touched by the contract.
    """
    c = validate_curve(curve)
    start_ts = pd.Timestamp(delivery_start)
    end_ts = pd.Timestamp(delivery_end_exclusive)
    if end_ts <= start_ts:
        raise ValueError("Delivery end must be after delivery start.")
    products = c["product"].unique().tolist()
    if len(products) != 1:
        raise ValueError("swap_fixed_price expects a curve for exactly one product.")
    timezone_name = PRODUCTS[products[0]]["timezone"]
    months = pd.period_range(start=start_ts, end=end_ts - pd.Timedelta(days=1), freq="M")
    marks = c.copy()
    marks["period"] = pd.to_datetime(marks["delivery_month"]).dt.to_period("M")
    prices = []
    weights = []
    for period in months:
        month_start = max(start_ts, period.start_time)
        month_end = min(end_ts, (period + 1).start_time)
        start_d = month_start.date()
        end_d = month_end.date()
        hours = delivery_hours(start_d, end_d, profile, timezone_name)
        if hours == 0:
            continue
        row = marks.loc[marks["period"] == period]
        if len(row) != 1:
            raise ValueError(f"Missing or ambiguous curve mark for {period}.")
        prices.append(float(row.iloc[0]["price"]))
        weights.append(hours)
    if not weights or sum(weights) <= 0:
        raise ValueError("Contract has no delivery hours for the selected profile.")
    return float(sum(p * w for p, w in zip(prices, weights)) / sum(weights))


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
    out["product"] = out["product"].astype(str).str.strip()
    out["instrument"] = out["instrument"].astype(str).str.upper().str.strip()
    out["settlement_type"] = out["settlement_type"].astype(str).str.upper().str.strip()
    out["side"] = out["side"].astype(str).str.upper().str.strip()
    out["status"] = out["status"].astype(str).str.upper().str.strip()
    out["currency"] = out["currency"].astype(str).str.upper().str.strip()
    out["profile"] = out["profile"].astype(str).str.upper().str.strip()
    if out["trade_id"].isna().any():
        raise ValueError("Trade IDs must be non-empty and unique.")
    out["trade_id"] = out["trade_id"].astype(str).str.strip()
    invalid_ids = out["trade_id"].str.lower().isin(["", "nan", "none", "null", "nat"])
    if invalid_ids.any() or out["trade_id"].duplicated().any():
        raise ValueError("Trade IDs must be non-empty and unique.")
    if not out["product"].isin(PRODUCTS).all():
        bad = out.loc[~out["product"].isin(PRODUCTS), "product"].unique().tolist()
        raise ValueError(f"Unsupported product(s): {bad}")
    for _, row in out.iterrows():
        expected_currency = PRODUCTS[row["product"]]["currency"]
        if row["currency"] != expected_currency:
            raise ValueError(f"Currency mismatch for {row['product']}: expected {expected_currency}, got {row['currency']}.")
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
    for _, row in out.iterrows():
        if row["instrument"] == "FORWARD":
            start_ts = pd.Timestamp(row["delivery_start"])
            expected_start = start_ts.to_period("M").start_time.date()
            expected_end = (start_ts.to_period("M") + 1).start_time.date()
            if row["delivery_start"] != expected_start or row["delivery_end"] != expected_end:
                raise ValueError(
                    f"FORWARD trade {row['trade_id']} must cover exactly one full calendar month; "
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
    if not out["profile"].astype(str).str.upper().isin(["BASE", "PEAK"]).all():
        raise ValueError("profile must be BASE or PEAK.")
    gas_peak = out.apply(lambda r: PRODUCTS[r["product"]]["commodity"] == "Gas" and r["profile"] == "PEAK", axis=1)
    if gas_peak.any():
        raise ValueError("Gas products must use the BASE delivery profile; PEAK is only valid for power products.")
    if not out["status"].isin(["PROPOSED", "CONFIRMED", "EXECUTED", "CANCELLED", "EXPIRED"]).all():
        raise ValueError("Unsupported trade status.")
    out["profile"] = out["profile"].str.upper()
    out["quantity_mwh"] = [
        mw_to_mwh(float(mw), delivery_hours(s, e, p, PRODUCTS[product]["timezone"]))
        for mw, s, e, p, product in zip(
            out["quantity_mw"], out["delivery_start"], out["delivery_end"], out["profile"], out["product"]
        )
    ]
    out["signed_quantity_mwh"] = [q if side == "BUY_FIXED" else -q for q, side in zip(out["quantity_mwh"], out["side"])]
    return out


def mark_trades(trades: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mark each trade against a weighted market forward for its delivery window.

    Requires a monthly curve mark for every delivery month. Marks must share the
    trade's price unit/currency; this function intentionally does not silently FX-convert.
    """
    t = normalise_trades(trades)
    rows = []
    for _, trade in t.iterrows():
        curve = curves.get(trade["product"])
        if curve is None:
            raise ValueError(f"No curve supplied for {trade['product']}.")
        curve = validate_curve(curve)
        spec = PRODUCTS[trade["product"]]
        if not curve["product"].eq(trade["product"]).all():
            raise ValueError(f"Curve product mismatch for {trade['product']}.")
        if not curve["currency"].eq(spec["currency"]).all() or not curve["unit"].eq(spec["unit"]).all():
            raise ValueError(f"Curve currency/unit mismatch for {trade['product']}: expected {spec['currency']} and {spec['unit']}.")
        # Price weighting by delivery hours; for monthly products with a partial window,
        # swap_fixed_price applies the profile hours to each touched month.
        if trade["instrument"] == "FORWARD":
            # A monthly forward maps directly to its single monthly delivery tenor.
            market_price = forward_month_price(curve, trade["product"], trade["delivery_start"], trade["delivery_end"])
            valuation_method = f"Single-month {str(trade['settlement_type']).lower()} forward proxy"
        else:
            # A swap strip uses each monthly forward weighted by delivery hours.
            market_price = swap_fixed_price(curve, trade["delivery_start"], trade["delivery_end"], trade["profile"])
            valuation_method = f"{str(trade['settlement_type']).title()} delivery-hour-weighted swap-strip proxy"
        signed_q = float(trade["signed_quantity_mwh"])
        mtm = mtm_linear(signed_q, float(trade["trade_price"]), market_price)
        rows.append({**trade.to_dict(), "market_price": market_price, "mtm": mtm,
                     "mtm_unit": f"{trade['currency']}", "valuation_method": valuation_method,
                     "mark_status": "MARKED", "mark_error": ""})
    return pd.DataFrame(rows)


def mark_trades_resilient(trades: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mark trades independently so one missing curve tenor does not hide the entire book.

    Invalid/unavailable trades remain visible with mark_status='UNMARKED' and a reason.
    Cancelled and expired trades remain visible in the blotter but are excluded from MtM.
    """
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for _, raw in trades.iterrows():
        source = raw.to_dict()
        lifecycle = str(source.get("status", "")).upper().strip()
        if lifecycle in {"CANCELLED", "EXPIRED"}:
            source.update({"market_price": float("nan"), "mtm": float("nan"),
                           "valuation_method": "Not valued", "mark_status": lifecycle,
                           "mark_error": f"Excluded from valuation: trade status is {lifecycle}."})
            rows.append(source)
            continue
        try:
            normalized = normalise_trades(pd.DataFrame([source]))
            marked = mark_trades(normalized, curves)
            record = marked.iloc[0].to_dict()
            record["mark_status"] = "MARKED"
            record["mark_error"] = ""
            rows.append(record)
        except Exception as exc:
            source.setdefault("quantity_mwh", float("nan"))
            source.setdefault("signed_quantity_mwh", float("nan"))
            source.update({"market_price": float("nan"), "mtm": float("nan"),
                           "valuation_method": "Unavailable", "mark_status": "UNMARKED",
                           "mark_error": f"{type(exc).__name__}: {exc}"})
            rows.append(source)
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
    executed = trades.loc[trades["status"].astype(str).str.upper().eq("EXECUTED")].copy()
    rows = []
    for _, raw in executed.iterrows():
        try:
            trade = normalise_trades(pd.DataFrame([raw.to_dict()])).iloc[0]
            product = trade["product"]
            zone = PRODUCTS[product]["timezone"]
            curve = curves.get(product)
            validated_curve = validate_curve(curve) if curve is not None else pd.DataFrame()
            if not validated_curve.empty:
                if not validated_curve["product"].eq(product).all():
                    raise ValueError(f"Curve product mismatch for {product}.")
                validated_curve = validated_curve.copy()
                validated_curve["period"] = pd.to_datetime(validated_curve["delivery_month"]).dt.to_period("M")
            start_ts = pd.Timestamp(trade["delivery_start"])
            end_ts = pd.Timestamp(trade["delivery_end"])
            periods = pd.period_range(start=start_ts, end=end_ts - pd.Timedelta(days=1), freq="M")
            sign = 1.0 if trade["side"] == "BUY_FIXED" else -1.0
            for period in periods:
                segment_start = max(start_ts, period.start_time)
                segment_end = min(end_ts, (period + 1).start_time)
                hours = delivery_hours(segment_start.date(), segment_end.date(), trade["profile"], zone)
                if hours <= 0:
                    continue
                signed_mwh = float(trade["quantity_mw"]) * hours * sign
                market_price = float("nan")
                mtm_value = float("nan")
                status = "UNMARKED"
                error = "No curve mark for this delivery bucket."
                if not validated_curve.empty:
                    mark = validated_curve.loc[validated_curve["period"].eq(period)]
                    if len(mark) == 1:
                        market_price = float(mark.iloc[0]["price"])
                        mtm_value = mtm_linear(signed_mwh, float(trade["trade_price"]), market_price)
                        status = "MARKED"
                        error = ""
                    elif len(mark) > 1:
                        error = "Ambiguous duplicate curve marks for this delivery bucket."
                    else:
                        error = f"Missing curve mark for {period}."
                rows.append({"trade_id": trade["trade_id"], "product": product,
                             "delivery_month": period.start_time.date(), "profile": trade["profile"],
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



# -----------------------------------------------------------------------------
# Visual style and Streamlit interface
# -----------------------------------------------------------------------------


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



# -----------------------------------------------------------------------------
# Streamlit user interface — MERIDIAN Energy Desk brand; original dark visual language retained
# -----------------------------------------------------------------------------
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
        <div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)


def render_header(title: str, subtitle: str) -> None:
    user_curves = st.session_state.get("custom_curves", {})
    if user_curves:
        badge = '<span class="badge badge-amber">USER CURVE INPUTS · DEMO BOOK</span>'
    else:
        badge = '<span class="badge badge-amber">ILLUSTRATIVE CURVES · DEMO BOOK</span>'
    st.markdown(
        f"""<div style="display:flex;justify-content:space-between;align-items:baseline">
        <div><h2 style="margin-bottom:0">{title}</h2>
        <span style="color:{GRAY};font-size:0.8rem">{subtitle}</span></div>
        <div>{badge}</div></div>""", unsafe_allow_html=True)
    help_text = PAGE_HELP.get(title)
    if help_text:
        st.markdown(f'<div class="page-help">{help_text}</div>', unsafe_allow_html=True)
    st.markdown("")


def title_block(title: str, subtitle: str) -> None:
    render_header(title, subtitle)


# -----------------------------------------------------------------------------
# Live market-data integrations (public APIs + licensed market-data adapter)
# -----------------------------------------------------------------------------
# AGSI+ credential was explicitly supplied by the user for this local copy.
# Keep this file private: the key is embedded as a convenience for local use.
# Prefer environment variables / Streamlit secrets for shared or deployed copies.
# API calls are made only when the user clicks a Fetch/Refresh button.

HTTP_TIMEOUT_SECONDS = 25
MAX_API_RESPONSE_BYTES = 20 * 1024 * 1024
GIE_AGSI_API_KEY_DEFAULT = "76556429239ab949472fb6f5714c9017"
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


def _http_get_bytes(url: str, params: Optional[dict] = None,
                    headers: Optional[dict] = None,
                    timeout: int = HTTP_TIMEOUT_SECONDS) -> tuple[bytes, str]:
    """Small dependency-free HTTP GET. Error messages never echo query tokens or headers."""
    final_url = _build_url(url, params)
    req_headers = {"User-Agent": "MERIDIAN-Energy-Desk/1.0", "Accept": "application/json,*/*"}
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
        raise RuntimeError(f"API returned HTTP {exc.code}. Check access rights, credentials, dates and query parameters.") from None
    except URLError as exc:
        reason = getattr(exc, "reason", "network error")
        raise RuntimeError(f"Could not reach the data provider ({reason}). Check your internet connection and provider availability.") from None
    except TimeoutError:
        raise RuntimeError("The data provider timed out. Try a shorter date range.") from None


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
    if (end_date - start_date).days > 730:
        raise ValueError("Select a period of no more than 730 days to limit API calls.")
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


def parse_entsoe_xml(content: bytes, series_kind: str = "price") -> pd.DataFrame:
    """Parse standard ENTSO-E Publication/GL MarketDocument time series into rows."""
    if content[:2] == b"PK":
        import zipfile
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
        reason = _xml_text(root, ["text", "code"]) or "No matching market document was returned."
        raise RuntimeError(f"ENTSO-E acknowledgement: {reason}")
    rows = []
    kind = series_kind.lower()
    for ts in [n for n in root.iter() if _xml_local_name(n.tag) == "TimeSeries"]:
        psr_code = _xml_text(ts, ["psrType"]) or ""
        series_name = {
            "B01": "Biomass", "B02": "Lignite", "B03": "Hard coal", "B04": "Natural gas",
            "B05": "Hard coal", "B06": "Oil", "B07": "Oil shale", "B08": "Peat",
            "B09": "Geothermal", "B10": "Hydro pumped storage", "B11": "Hydro run-of-river",
            "B12": "Hydro reservoir", "B13": "Marine", "B14": "Nuclear", "B15": "Other renewable",
            "B16": "Solar", "B17": "Waste", "B18": "Wind offshore", "B19": "Wind onshore",
            "B20": "Other", "B21": "AC link", "B22": "DC link", "B23": "Substation", "B24": "Energy storage",
        }.get(psr_code, psr_code or ("Day-ahead price" if kind == "price" else kind.title()))
        # Period elements are direct children of TimeSeries in standard ENTSO-E XML.
        periods = [n for n in ts.iter() if _xml_local_name(n.tag) == "Period"]
        for period in periods:
            interval_node = next((n for n in period.iter() if _xml_local_name(n.tag) == "timeInterval"), None)
            start_txt = _xml_text(interval_node, ["start"]) if interval_node is not None else _xml_text(period, ["start"])
            resolution = _xml_text(period, ["resolution"])
            if not start_txt or not resolution:
                continue
            try:
                period_start = pd.Timestamp(start_txt)
                if period_start.tzinfo is None:
                    period_start = period_start.tz_localize("UTC")
                else:
                    period_start = period_start.tz_convert("UTC")
                delta = _parse_duration(resolution)
            except Exception:
                continue
            for point in [n for n in period.iter() if _xml_local_name(n.tag) == "Point"]:
                pos_txt = _xml_text(point, ["position"])
                if not pos_txt:
                    continue
                value_txt = _xml_text(point, ["price.amount", "quantity", "outBiddingZone_Domain.mRID"] if kind == "price" else ["quantity", "price.amount"])
                if value_txt is None:
                    continue
                try:
                    value = float(value_txt.replace(",", "."))
                    pos = int(pos_txt)
                except Exception:
                    continue
                rows.append({
                    "timestamp": period_start + (pos - 1) * delta,
                    "value": value,
                    "series": series_name,
                    "psr_code": psr_code,
                    "resolution": resolution,
                    "unit": "EUR/MWh" if kind == "price" else "MW",
                })
    if not rows:
        msg = _xml_text(root, ["text"]) or "No observations found in ENTSO-E XML response."
        raise RuntimeError(f"ENTSO-E returned no usable observations. {msg}")
    result = pd.DataFrame(rows).drop_duplicates(subset=["timestamp", "series", "value"]).sort_values(["timestamp", "series"])
    return result.reset_index(drop=True)


def fetch_entsoe_series(token: str, series_kind: str, zone_name: str,
                        start_date: date, end_date_exclusive: date,
                        source_zone: Optional[str] = None,
                        target_zone: Optional[str] = None) -> pd.DataFrame:
    """Fetch ENTSO-E day-ahead prices, actual load, generation or cross-border flow."""
    if not token.strip():
        raise ValueError("Enter your ENTSO-E Web API token. REST access must be enabled on your account.")
    if end_date_exclusive <= start_date:
        raise ValueError("End date must be later than start date (end date is exclusive).")
    if (end_date_exclusive - start_date).days > 31:
        raise ValueError("Select at most 31 delivery days per request to stay within API limits.")
    if series_kind == "Physical flow":
        if source_zone not in ENTSOE_ZONES or target_zone not in ENTSOE_ZONES:
            raise ValueError("Select both a source and destination bidding zone.")
        if source_zone == target_zone:
            raise ValueError("Physical flow requires two different bidding zones.")
        document_type = "A11"
        base_params = {"in_Domain": ENTSOE_ZONES[source_zone]["eic"], "out_Domain": ENTSOE_ZONES[target_zone]["eic"]}
        query_label = f"{source_zone} → {target_zone} physical flow"
        parser_kind = "flow"
    else:
        if zone_name not in ENTSOE_ZONES:
            raise ValueError("Select a supported ENTSO-E bidding zone.")
        eic = ENTSOE_ZONES[zone_name]["eic"]
        if series_kind == "Day-ahead price":
            document_type, parser_kind = "A44", "price"
            base_params = {"in_Domain": eic, "out_Domain": eic}
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
    tz_name = ENTSOE_ZONES.get(zone_name, {}).get("timezone", "Europe/Paris")
    zone = ZoneInfo(tz_name)
    start_local = datetime.combine(start_date, time.min, tzinfo=zone)
    end_local = datetime.combine(end_date_exclusive, time.min, tzinfo=zone)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    params = dict(base_params)
    params.update({
        "documentType": document_type,
        "periodStart": start_utc.strftime("%Y%m%d%H%M"),
        "periodEnd": end_utc.strftime("%Y%m%d%H%M"),
        "securityToken": token.strip(),
    })
    body, _ = _http_get_bytes(ENTSOE_BASE, params=params, headers={"Accept": "application/xml, application/zip, */*"})
    frame = parse_entsoe_xml(body, parser_kind)
    frame["_source"] = "ENTSO-E Transparency Platform"
    frame["_query"] = query_label
    frame["_fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame["_area"] = query_label
    return frame


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
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
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
    for col in ("temperature_2m", "wind_speed_10m", "shortwave_radiation"):
        if col in hourly:
            hourly[col] = pd.to_numeric(hourly[col], errors="coerce")
    agg = {c: "mean" for c in ("temperature_2m", "wind_speed_10m", "shortwave_radiation") if c in hourly.columns}
    daily = hourly.groupby("date", as_index=False).agg(agg) if agg else pd.DataFrame()
    if not daily.empty:
        if "temperature_2m" in daily: daily = daily.rename(columns={"temperature_2m": "temperature_mean_c"})
        if "wind_speed_10m" in daily: daily = daily.rename(columns={"wind_speed_10m": "wind_speed_mean_kmh"})
        if "shortwave_radiation" in daily: daily = daily.rename(columns={"shortwave_radiation": "solar_radiation_mean_wm2"})
        daily["date"] = pd.to_datetime(daily["date"])
        daily["timezone"] = timezone_name
    hourly["_source"] = "Open-Meteo"
    hourly["_fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not daily.empty:
        daily["_source"] = "Open-Meteo"
    return hourly, daily


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
    if product not in PRODUCTS:
        raise ValueError(f"Unsupported canonical product: {product}")
    if delivery_col not in raw.columns or price_col not in raw.columns:
        raise ValueError("Choose valid delivery date and price columns from the provider response.")
    out = pd.DataFrame({
        "delivery_month": pd.to_datetime(raw[delivery_col], errors="coerce"),
        "product": product,
        "price": _number_series(raw[price_col]),
        "currency": PRODUCTS[product]["currency"],
        "unit": PRODUCTS[product]["unit"],
    })
    if out["delivery_month"].isna().any():
        raise ValueError("Some delivery dates could not be parsed. Map a column containing ISO dates such as YYYY-MM-01, and convert contract labels before import.")
    if out["price"].isna().any():
        raise ValueError("Some provider prices are missing or non-numeric. No implicit conversion was applied.")
    return validate_curve(out)


def _stored_live_item(source_key: str, label: str, data, metadata: Optional[dict] = None) -> None:
    st.session_state.setdefault("live_data", {})[source_key] = {
        "label": label,
        "data": data,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": metadata or {},
    }


def _live_dataset_frame(key: str) -> pd.DataFrame:
    item = st.session_state.get("live_data", {}).get(key, {})
    data = item.get("data") if isinstance(item, dict) else None
    return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()


def _display_live_frame(frame: pd.DataFrame, title: str, chart_x: Optional[str] = None,
                        chart_y: Optional[str] = None, color: Optional[str] = None) -> None:
    if frame.empty:
        st.info(f"No {title.lower()} data loaded yet. Open Live Data Integrations and fetch this dataset.")
        return
    if chart_x and chart_y and chart_x in frame.columns and chart_y in frame.columns:
        plot_frame = frame.copy()
        fig = px.line(plot_frame, x=chart_x, y=chart_y, color=color if color in frame.columns else None,
                      markers=True, title=title)
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=350)
        st.plotly_chart(_styled(fig), use_container_width=True)
    st.dataframe(frame, use_container_width=True, hide_index=True)


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


def _live_item_fetched_at(key: str) -> str:
    item = st.session_state.get("live_data", {}).get(key, {})
    return str(item.get("fetched_at", "")) if isinstance(item, dict) else ""


def _live_item_meta(key: str) -> dict:
    item = st.session_state.get("live_data", {}).get(key, {})
    return item.get("metadata", {}) if isinstance(item, dict) and isinstance(item.get("metadata", {}), dict) else {}


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


def _build_residual_load_proxy() -> pd.DataFrame:
    """Join ENTSO-E load, wind/solar output and day-ahead price into hourly context.

    This proxy is only computed for overlapping timestamps: residual load = actual load
    minus actual wind and solar output. It is not a full system balance or a price model.
    """
    prices = _live_dataset_frame("entsoe_price")
    load = _live_dataset_frame("entsoe_load")
    generation = _live_dataset_frame("entsoe_generation")
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
    renewable_mask = labels.str.contains("wind onshore|wind offshore|solar", regex=True)
    gen = gen.loc[renewable_mask].copy()
    if gen.empty:
        return pd.DataFrame()
    gen["hour"] = pd.to_datetime(gen["timestamp"], errors="coerce", utc=True).dt.floor("h")
    gen["generation_mw"] = _number_series(gen["value"])
    gen = gen.dropna(subset=["hour", "generation_mw"])
    # Sum onshore wind + offshore wind + solar at each source timestamp, then average
    # these sub-hourly/ hourly totals into a common UTC-hour bucket.
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
            renewable = latest_gen.loc[labels.str.contains("wind onshore|wind offshore|solar", regex=True), "value"].sum()
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
        row = _latest_row(weather, ("date", "timestamp"))
        if row is not None:
            def row_num(aliases):
                col = _find_column(weather, aliases)
                if not col: return None
                x = _number_series(pd.Series([row.get(col)])).iloc[0]
                return float(x) if pd.notna(x) and math.isfinite(float(x)) else None
            snapshot["weather"] = {"date": str(row.get("date", "")),
                "temperature_c": row_num(("temperature_mean_c", "temperature_2m")),
                "wind_kmh": row_num(("wind_speed_mean_kmh", "wind_speed_10m")),
                "solar_wm2": row_num(("solar_radiation_mean_wm2", "shortwave_radiation")),
                "location": str(_live_item_meta("weather_daily").get("location", "selected location")),
                "fetched_at": _live_item_fetched_at("weather_daily")}
    return snapshot


NAV_SECTIONS = {
    "Markets": ["📊  Market Overview", "📈  Forward Curves", "🧭  Market Fundamentals", "🔌  Live Data Integrations"],
    "Trading & Risk": ["💬  OTC Quote Builder", "📒  Trading Book & MtM", "⚠️  P&L & Exposure"],
    "Physical Operations": ["🚢  Physical Operations"],
    "Analytics": ["🧪  Model Validation"],
    "Reference": ["🔧  Data & Assumptions"],
}
ALL_NAV_ITEMS = [label for labels in NAV_SECTIONS.values() for label in labels]
DEFAULT_NAV_ITEM = ALL_NAV_ITEMS[0]
NAV_TO_PAGE = {label: label.split("  ", 1)[1] for label in ALL_NAV_ITEMS}

PAGE_HELP = {
    "MERIDIAN Energy Trading Desk": "European gas and power OTC quote support, trade capture, mark-to-market and delivery monitoring. Built-in prices and trades are illustrative.",
    "OTC Quote Builder": "Indicative price = reference forward plus/minus commercial margin. The quote is not executable and must be checked against approved market data, credit limits and current conventions.",
    "Trading Book & Mark-to-Market": "FORWARD uses one monthly mark; SWAP uses a delivery-hour-weighted strip. MtM is simplified and undiscounted. Positions are split by delivery month and profile, with net and gross MWh shown separately.",
    "Forward Curves": "Monthly forward marks are illustrative unless replaced by validated user data. Missing delivery tenors are rejected rather than silently interpolated.",
    "P&L Sensitivity & Exposure": "A parallel curve shock estimates a simplified change in MtM. It is not VaR, realised daily P&L or a full risk measure.",
    "Physical Operations & Delivery Tracking": "Nominations, delivered volumes and invoice status persist in the session and are reconciled to linked confirmed/executed physical trades. External operational feeds are not connected.",
    "European Energy Fundamentals": "Charts are illustrative until connected to timestamped, licensed market data for European gas and power fundamentals.",
    "Model Validation & System Checks": "Runs deterministic unit and integration checks on conversions, delivery calendars, curve validation, trade capture, MtM, portfolio aggregation and scenario calculations.",
    "Data, Conventions & Validation": "Sources, units, delivery calendars and validation rules should be explicit before marks or trades are used for commercial decisions.",
    "Live Data Integrations": "Connect GIE storage/LNG, ENTSO-E electricity data, ENTSOG gas flows, Open-Meteo weather and a licensed forward-curve feed. Credentials are entered locally and are never embedded in the source file.",
}


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
        st.markdown(f"""
        <div style="padding:6px 0 10px 0">
          <span style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;color:{AMBER}">MERIDIAN ENERGY DESK</span><br>
          <span style="font-size:0.68rem;color:{GRAY}">EUROPEAN ENERGY &nbsp;·&nbsp; OTC &nbsp;·&nbsp; MTM</span>
        </div>""", unsafe_allow_html=True)
        live_items = st.session_state.get("live_data", {})
        loaded_count = sum(1 for value in live_items.values() if isinstance(value, dict) and value.get("data") is not None)
        status_label = f"{loaded_count} LIVE DATASET(S) LOADED" if loaded_count else "DEMO CURVES · NO LIVE DATA LOADED"
        status_class = "badge" if loaded_count else "badge badge-amber"
        st.markdown(f'<span class="badge {"" if loaded_count else "badge-amber"}">{status_label}</span>', unsafe_allow_html=True)
        for section, items in NAV_SECTIONS.items():
            key = f"navsec_{section}"
            if key not in st.session_state:
                st.session_state[key] = current if current in items else None
            st.radio(section, items, key=key, on_change=_on_nav, args=(section,))
        if st.button("↻ Reset demo book & curves", use_container_width=True):
            st.session_state.trades = build_sample_trades()
            st.session_state.custom_curves = {}
            st.session_state.curve_metadata = {}
            st.session_state.physical_ops = build_sample_physical_ops(st.session_state.trades)
            st.session_state.pop("last_quote", None)
            st.rerun()
        st.markdown("---")
        custom_count = len(st.session_state.get("custom_curves", {}))
        st.markdown(
            f'<span class="badge">PRODUCTS {len(PRODUCTS)}</span> '
            f'<span class="badge {"badge-amber" if custom_count else ""}">USER CURVES {custom_count}</span>',
            unsafe_allow_html=True)
        with st.expander("Feed / methodology status"):
            if loaded_count:
                st.caption(f"{loaded_count} dataset(s) were loaded from external APIs during this session.")
            else:
                st.caption("No external data has been fetched in this session. Use Live Data Integrations to connect provider APIs.")
            st.caption("Built-in forward curves remain illustrative unless replaced with a validated licensed curve or CSV.")
            st.caption("Day-ahead prices are not automatically substituted for monthly forward marks.")
            st.caption("Model Validation checks deterministic formulas and parser fixtures; external API access still depends on your credentials and provider availability.")
        st.markdown("---")
        st.markdown(f'<div style="font-size:0.62rem;color:{GRAY};line-height:1.5">Single-file educational ETRM prototype<br>European gas & power · not for trade execution</div>', unsafe_allow_html=True)
    return NAV_TO_PAGE[st.session_state.nav_page]


# Page setup must precede all other Streamlit commands.
st.set_page_config(page_title="MERIDIAN — European Energy Trading & Risk Desk", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(_CSS, unsafe_allow_html=True)

if "trades" not in st.session_state:
    st.session_state.trades = build_sample_trades()
if "custom_curves" not in st.session_state:
    st.session_state.custom_curves = {}
if "curve_metadata" not in st.session_state:
    st.session_state.curve_metadata = {}
if "physical_ops" not in st.session_state:
    st.session_state.physical_ops = build_sample_physical_ops(st.session_state.trades)
else:
    st.session_state.physical_ops = sync_physical_ops_with_trades(
        st.session_state.physical_ops, st.session_state.trades
    )
if "live_data" not in st.session_state:
    st.session_state.live_data = {}

valuation_date = date.today()
curves = {
    product: st.session_state.custom_curves.get(product,
              make_sample_curve(product, valuation_date, months=24))
    for product in PRODUCTS
}


def reference_mark(product: str, start: date, end: date, profile: str) -> float:
    return swap_fixed_price(curves[product], start, end, profile)


def money(value: float, currency: str = "EUR") -> str:
    return f"{value:,.2f} {currency}"


def month_start(offset: int = 1) -> date:
    return (pd.Timestamp(valuation_date).to_period("M") + offset).to_timestamp().date()


def month_end_exclusive(start: date, months: int) -> date:
    return (pd.Timestamp(start).to_period("M") + months).to_timestamp().date()


def get_marked_trades() -> pd.DataFrame:
    # Preserve cancelled/expired rows in the blotter for auditability while excluding
    # them from MtM and risk metrics. An unavailable tenor only marks that trade unmarked.
    return mark_trades_resilient(st.session_state.trades.copy(), curves)


def curve_source_label(product: str) -> str:
    if product not in st.session_state.get("custom_curves", {}):
        return "ILLUSTRATIVE DEMO MARKS"
    metadata = st.session_state.get("curve_metadata", {}).get(product, {})
    origin = str(metadata.get("source_label", "USER / PROVIDER-IMPORTED MARKS")).strip()
    return origin or "USER / PROVIDER-IMPORTED MARKS"


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
        required = {"Market Overview", "Forward Curves", "OTC Quote Builder", "Trading Book & MtM",
                    "P&L & Exposure", "Physical Operations", "Market Fundamentals", "Live Data Integrations", "Model Validation", "Data & Assumptions"}
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
        ("Licensed REST-provider CSV connector", "Live API integration", check_generic_provider_rest_csv),
        ("Live dashboard indicator integration", "Dashboard integration", check_live_snapshot_indicators),
        ("Cross-market hourly load/generation/price join", "Fundamentals analytics", check_residual_load_proxy_alignment),
    ]
    for name, area, fn in cases:
        run_case(name, area, fn)
    return pd.DataFrame(rows, columns=["Status", "Area", "Check", "Details"])


page = render_sidebar()

if page == "Market Overview":
    title_block("MERIDIAN Energy Trading Desk", "A compact ETRM prototype focused on European gas, power, OTC pricing and risk.")
    marked = get_marked_trades()
    executed = marked.loc[marked["status"].eq("EXECUTED") & marked["mark_status"].eq("MARKED")] if not marked.empty else pd.DataFrame()
    unmarked_count = int(marked.loc[~marked["status"].isin(["CANCELLED", "EXPIRED"]) & ~marked["mark_status"].eq("MARKED")].shape[0]) if not marked.empty else 0
    total_mtm = float(executed["mtm"].sum()) if not executed.empty else 0.0
    total_trades = int(len(executed))
    total_products = int(executed["product"].nunique()) if not executed.empty else 0
    eur_mtm = float(executed.loc[executed["currency"].eq("EUR"), "mtm"].sum()) if not executed.empty else 0.0
    gbp_mtm = float(executed.loc[executed["currency"].eq("GBP"), "mtm"].sum()) if not executed.empty else 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Executed trades", f"{total_trades}")
    c2.metric("Products traded", f"{total_products}")
    c3.metric("Session MtM · EUR", money(eur_mtm, "EUR"))
    c4.metric("Session MtM · GBP", money(gbp_mtm, "GBP"))
    c5.metric("Unmarked active trades", f"{unmarked_count}")
    if unmarked_count:
        st.warning(f"{unmarked_count} active trade(s) could not be marked. Review the trade-level diagnostic in Trading Book & MtM.")
    imported_products = sorted(st.session_state.get("custom_curves", {}).keys())
    fallback_products = sorted(set(PRODUCTS) - set(imported_products))
    source_parts = []
    if imported_products: source_parts.append("Imported marks: " + ", ".join(imported_products))
    if fallback_products: source_parts.append("Illustrative fallback marks: " + ", ".join(fallback_products))
    st.caption("MtM is reported separately by quote currency. " + "; ".join(source_parts) + ". Not an executable or accounting valuation.")

    # Live feeds are consumed here as market context; fundamentals are not used as substitute forward marks.
    snapshot = build_live_snapshot()
    st.subheader("Live market snapshot")
    if not snapshot:
        st.info("No provider data is loaded yet. Fetch GIE storage/LNG, ENTSO-E prices/load, ENTSOG flows or weather in Live Data Integrations. The dashboard will populate automatically when those datasets are loaded.")
        if st.button("Open Live Data Integrations", key="overview_open_integrations"):
            st.session_state.nav_page = next(label for label in ALL_NAV_ITEMS if label.endswith("Live Data Integrations"))
            st.rerun()
    else:
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        storage = snapshot.get("storage", {})
        lng = snapshot.get("lng", {})
        power = snapshot.get("power_price", {})
        load = snapshot.get("power_load", {})
        weather = snapshot.get("weather", {})
        gas_flow = snapshot.get("gas_flow", {})
        s1.metric("Gas storage fill", f"{storage['fill_pct']:.1f}%" if storage.get("fill_pct") is not None else "—",
                  help=f"GIE AGSI+ · {storage.get('scope', 'selected scope')} · observation {storage.get('date', 'n/a')}")
        change = storage.get("change_7_reports_twh")
        s2.metric("Gas in storage", f"{storage['inventory_twh']:.1f} TWh" if storage.get("inventory_twh") is not None else "—",
                  delta=f"{change:+.1f} TWh vs 7 reports ago" if change is not None else None,
                  help=f"GIE AGSI+ · {storage.get('date', 'n/a')}")
        s3.metric("LNG send-out", f"{lng['sendout']:.1f} GWh/d" if lng.get("sendout") is not None else "—",
                  help=f"GIE ALSI+ · {lng.get('scope', 'selected scope')} · {lng.get('date', 'n/a')}")
        s4.metric("Day-ahead power", f"{power['value']:.2f} {power.get('unit', 'EUR/MWh')}" if power.get("value") is not None else "—",
                  help=f"ENTSO-E · {power.get('zone', 'zone')} · {power.get('timestamp', 'n/a')}")
        s5.metric("Actual electricity load", f"{load['value']/1000:.2f} GW" if load.get("value") is not None else "—",
                  help=f"ENTSO-E · {load.get('zone', 'zone')} · {load.get('timestamp', 'n/a')} · source unit {load.get('unit', 'MW')}")
        s6.metric("Weather temperature", f"{weather['temperature_c']:.1f} °C" if weather.get("temperature_c") is not None else "—",
                  help=f"Open-Meteo daily mean · {weather.get('location', 'selected location')} · {weather.get('date', 'n/a')}")
        st.caption("Data sources: GIE AGSI+ (gas storage), GIE ALSI+ (LNG), ENTSO-E Transparency Platform (power prices/load), ENTSOG (selected gas point-direction) and Open-Meteo (weather). Observation timestamps and selected scopes appear in tooltips and the Data freshness expander.")
        if gas_flow.get("value") is not None:
            st.caption(f"Latest ENTSOG {gas_flow.get('label', 'indicator')}: {gas_flow['value']:,.2f} {gas_flow.get('unit', 'unit unverified')} · point direction {gas_flow.get('point_direction', 'selected point')} · observation {gas_flow.get('timestamp', 'n/a')}. Confirm the exact unit and direction before using operationally.")
        if weather.get("wind_kmh") is not None or weather.get("solar_wm2") is not None:
            bits = []
            if weather.get("wind_kmh") is not None: bits.append(f"daily mean wind {weather['wind_kmh']:.1f} km/h")
            if weather.get("solar_wm2") is not None: bits.append(f"daily mean solar radiation {weather['solar_wm2']:.1f} W/m²")
            st.caption(f"Weather context — {weather.get('location', 'selected location')}: " + "; ".join(bits) + ". These are weather proxies, not power-price forecasts.")
        st.subheader("Fundamental observations")
        observations = []
        if storage.get("change_7_reports_twh") is not None:
            direction = "inventory draw" if storage["change_7_reports_twh"] < 0 else "inventory build" if storage["change_7_reports_twh"] > 0 else "flat inventory"
            observations.append({"Observation": "Gas storage", "Latest signal": f"{direction}: {storage['change_7_reports_twh']:+.2f} TWh across the latest seven reporting observations", "Interpretation boundary": "Inventory change is context; it does not alone determine the TTF price direction."})
        if power:
            observations.append({"Observation": "Day-ahead power", "Latest signal": f"Latest {power.get('zone', 'zone')} observation {power['value']:.2f} {power.get('unit', 'EUR/MWh')}; fetched sample includes {power.get('negative_observations', 0)} negative observations", "Interpretation boundary": "Count is observations, not necessarily hours; market resolution may vary."})
        if weather:
            weather_text = ", ".join([x for x in [f"temperature {weather['temperature_c']:.1f} °C" if weather.get('temperature_c') is not None else "", f"wind {weather['wind_kmh']:.1f} km/h" if weather.get('wind_kmh') is not None else ""] if x])
            if weather_text:
                observations.append({"Observation": "Weather", "Latest signal": f"{weather.get('location', 'Selected location')}: {weather_text}", "Interpretation boundary": "Local weather is a proxy; it is not a system-wide demand or renewable forecast."})
        if observations:
            st.dataframe(pd.DataFrame(observations), use_container_width=True, hide_index=True)
        with st.expander("Data freshness and provenance", expanded=False):
            freshness_rows = []
            for key, item in st.session_state.get("live_data", {}).items():
                frame = item.get("data") if isinstance(item, dict) else None
                freshness_rows.append({"Dataset": item.get("label", key), "Observations": len(frame) if isinstance(frame, pd.DataFrame) else 0,
                    "Latest observation": _latest_observation_date(frame) if isinstance(frame, pd.DataFrame) else None,
                    "Fetched at (UTC)": item.get("fetched_at", ""),
                    "Source": (frame["_source"].iloc[0] if isinstance(frame, pd.DataFrame) and not frame.empty and "_source" in frame.columns else item.get("label", key))})
            st.dataframe(pd.DataFrame(freshness_rows), use_container_width=True, hide_index=True)
            st.caption("GIE AGSI+/ALSI attribution is mandatory. Market data may be delayed or revised. The last fetch timestamp is not the same as the observation timestamp.")

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Forward curves")
        selected = st.selectbox("Product", list(PRODUCTS.keys()), key="overview_product")
        curve = curves[selected].copy()
        fig = px.line(curve, x="delivery_month", y="price", markers=True,
                      title=f"{selected} — {curve_source_label(selected).title()}",
                      labels={"delivery_month": "Delivery month", "price": PRODUCTS[selected]["unit"]})
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=370)
        st.plotly_chart(_styled(fig), use_container_width=True)
    with right:
        st.subheader("Executed book by product")
        if not executed.empty:
            summary = position_summary(executed)
            fig = px.bar(summary, x="product", y="signed_mwh", color="currency", title="Signed delivery volume (MWh)")
            fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=370, xaxis_title="", yaxis_title="Signed MWh")
            st.plotly_chart(_styled(fig), use_container_width=True)
        else:
            st.caption("No executed trades in the current book.")
    st.subheader("Desk watchlist")
    watch = pd.DataFrame([{"Product": p, "Currency": spec["currency"], "Quote unit": spec["unit"], "Curve status": curve_source_label(p)} for p, spec in PRODUCTS.items()])
    st.dataframe(watch, use_container_width=True, hide_index=True)

elif page == "Forward Curves":
    title_block("Forward Curves", "Inspect monthly delivery marks by product. Curves are illustrative unless you import and validate your own market data.")
    selected = st.selectbox("Product", list(PRODUCTS.keys()), key="forward_curve_product")
    curve = curves[selected].copy().sort_values("delivery_month")
    curve["delivery_month"] = pd.to_datetime(curve["delivery_month"])
    c1, c2, c3, c4 = st.columns(4)
    first = curve.iloc[0]
    kpi(c1, "First delivery mark", f"{float(first['price']):,.2f}", first["delivery_month"].strftime("%b %Y"), AMBER)
    kpi(c2, "Curve average", f"{curve['price'].mean():,.2f}", PRODUCTS[selected]["unit"], BLUE)
    kpi(c3, "Curve low", f"{curve['price'].min():,.2f}", PRODUCTS[selected]["unit"], GREEN)
    kpi(c4, "Curve high", f"{curve['price'].max():,.2f}", PRODUCTS[selected]["unit"], RED)
    curve_meta = st.session_state.get("curve_metadata", {}).get(selected, {})
    if selected in st.session_state.get("custom_curves", {}):
        st.caption(f"Curve provenance: {curve_source_label(selected)} · imported at {curve_meta.get('imported_at_utc', 'timestamp not recorded')}. This is an imported mark set; verify valuation date, contract definition and vendor entitlement.")
    else:
        st.warning("This curve is illustrative demo data. Do not treat it as a market quote or use it as evidence of current TTF/NBP/power prices.")
    # Display day-ahead prices as contextual spot evidence only; they must not replace monthly forward marks.
    if PRODUCTS[selected]["commodity"] == "Power":
        context = build_live_snapshot().get("power_price", {})
        expected_zone = "France" if PRODUCTS[selected]["zone"] == "France" else "Germany-Luxembourg" if PRODUCTS[selected]["zone"] == "Germany" else None
        if context and (expected_zone is None or context.get("zone") == expected_zone):
            st.info(f"Day-ahead context only — latest ENTSO-E {context.get('zone', 'selected zone')} observation: {context.get('value', float('nan')):.2f} {context.get('unit', 'EUR/MWh')} at {context.get('timestamp', 'n/a')}. This observation is not used as the monthly forward mark.")
        elif context:
            st.caption(f"An ENTSO-E day-ahead price is loaded for {context.get('zone', 'another zone')}, which does not match the selected product's bidding zone; no cross-zone substitution was made.")
        else:
            st.caption("Connect ENTSO-E day-ahead prices in Live Data Integrations to display a spot-market context. This will not replace the forward curve.")
    elif PRODUCTS[selected]["commodity"] == "Gas":
        st.caption("GIE AGSI+/ALSI provide gas-storage and LNG fundamentals, not TTF/NBP traded forwards. A licensed or explicitly authorised futures curve is still required for market-referenced OTC MtM.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["delivery_month"], y=curve["price"], mode="lines+markers", name="Monthly forward", line=dict(color=AMBER, width=2)))
    fig.add_hline(y=float(curve["price"].mean()), line=dict(color=BLUE, dash="dot"), annotation_text="Curve average")
    fig.update_layout(title=f"{selected} — {curve_source_label(selected).title()}", xaxis_title="Delivery month", yaxis_title=PRODUCTS[selected]["unit"])
    st.plotly_chart(_styled(fig, 440), use_container_width=True)
    st.subheader("Monthly delivery marks")
    display_curve = curve.copy()
    display_curve["delivery_month"] = display_curve["delivery_month"].dt.date
    st.dataframe(display_curve, use_container_width=True, hide_index=True)
    st.download_button("Download displayed curve (CSV)", data=display_curve.to_csv(index=False).encode("utf-8"), file_name=f"{selected.lower().replace(' ', '_')}_forward_curve.csv", mime="text/csv")
    st.caption(f"Source status: {curve_source_label(selected)}. Confirm the provider, valuation timestamp, delivery definitions and unit before using imported marks. Curves are not executable quotes.")

elif page == "OTC Quote Builder":
    title_block("OTC Quote Builder", "Build an indicative forward or swap quote from the selected delivery curve. No quote is executable.")
    with st.form("quote_form"):
        a, b, c = st.columns(3)
        product = a.selectbox("Product", list(PRODUCTS.keys()))
        instrument = b.selectbox("Instrument structure", ["SWAP", "FORWARD"],
                                 help="FORWARD = one full calendar month. SWAP = fixed-price strip across one or more months.")
        settlement_type = c.selectbox("Settlement", ["FINANCIAL", "PHYSICAL"])
        profile = st.selectbox("Delivery profile", ["BASE", "PEAK"], disabled=PRODUCTS[product]["commodity"] == "Gas")
        if PRODUCTS[product]["commodity"] == "Gas":
            profile = "BASE"
        a, b, c = st.columns(3)
        start = a.date_input("Delivery start", value=month_start(1))
        tenor_options = [1] if instrument == "FORWARD" else [1, 3, 6, 12]
        months = b.selectbox("Delivery tenor", tenor_options, index=0 if instrument == "FORWARD" else 1,
                             format_func=lambda x: f"{x} month(s)")
        end = month_end_exclusive(start, months)
        mw = c.number_input("Contract volume (MW)", min_value=0.1, max_value=100000.0, value=10.0, step=0.5)
        a, b, c = st.columns(3)
        side = a.selectbox("Desk quote side", ["SELL_TO_CLIENT", "BUY_FROM_CLIENT"], format_func=lambda x: "Sell to client" if x == "SELL_TO_CLIENT" else "Buy from client")
        margin = b.number_input("Commercial margin per MWh", min_value=0.0, max_value=100.0, value=0.35, step=0.05)
        counterparty = c.text_input("Counterparty", value="Prospective client (Demo)")
        submitted = st.form_submit_button("Calculate indicative quote", type="primary")
    if submitted:
        try:
            if instrument == "FORWARD" and (start.day != 1 or months != 1):
                raise ValueError("A FORWARD quote must cover exactly one full calendar month. Choose SWAP for delivery strips.")
            ref = (forward_month_price(curves[product], product, start, end)
                   if instrument == "FORWARD" else reference_mark(product, start, end, profile))
            quoted = quote_price(ref, margin, side)
            hours = delivery_hours(start, end, profile, PRODUCTS[product]["timezone"])
            total_mwh = mw * hours
            st.session_state.last_quote = {"product": product, "instrument": instrument, "settlement_type": settlement_type, "profile": profile,
                "start": start, "end": end, "mw": mw, "mwh": total_mwh, "side": side,
                "reference": ref, "margin": margin, "quote": quoted, "counterparty": counterparty,
                "currency": PRODUCTS[product]["currency"], "unit": PRODUCTS[product]["unit"], "curve_source": curve_source_label(product)}
        except Exception as exc:
            st.error(f"Quote could not be calculated: {exc}")
    q = st.session_state.get("last_quote")
    if q:
        st.subheader("Quote summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reference price", f"{q['reference']:.4f} {q['unit']}")
        c2.metric("Indicative client price", f"{q['quote']:.4f} {q['unit']}")
        c3.metric("Delivery hours", f"{delivery_hours(q['start'], q['end'], q['profile'], PRODUCTS[q['product']]['timezone']):,}")
        c4.metric("Contract energy", f"{q['mwh']:,.0f} MWh")
        st.write(f"**Counterparty:** {q['counterparty']}  ·  **Delivery:** {q['start']} to {q['end']} (end date exclusive)  ·  **Curve source:** {q.get('curve_source', 'unverified')}  ·  **Status:** Indicative only")
        st.caption("Quote convention: sell-to-client = reference + margin; buy-from-client = reference − margin. This is a configurable illustrative convention, not a universal market convention.")
        if st.button("Add quote to blotter as PROPOSED"):
            new_id = f"Q-{pd.Timestamp.now().strftime('%y%m%d%H%M%S')}"
            new = pd.DataFrame([{"trade_id": new_id, "trade_date": valuation_date.isoformat(), "counterparty": q["counterparty"] or "Unspecified",
                "product": q["product"], "instrument": q["instrument"], "settlement_type": q.get("settlement_type", "FINANCIAL"),
                "side": "BUY_FIXED" if q["side"] == "BUY_FROM_CLIENT" else "SELL_FIXED",
                "delivery_start": q["start"].isoformat(), "delivery_end": q["end"].isoformat(),
                "profile": q["profile"], "quantity_mw": q["mw"], "trade_price": q["quote"],
                "currency": q["currency"], "status": "PROPOSED", "notes": "Generated by OTC Quote Builder; indicative quote"}])
            try:
                st.session_state.trades = add_trade_to_book(st.session_state.trades, new)
                st.session_state.physical_ops = sync_physical_ops_with_trades(st.session_state.physical_ops, st.session_state.trades)
                st.success(f"Quote {new_id} added to the trade book as PROPOSED. It is excluded from executed-book position totals.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not add quote: {exc}")

elif page == "Trading Book & MtM":
    title_block("Trading Book & Mark-to-Market", "Trade capture, curve marking and signed position monitoring. All built-in marks are illustrative.")
    with st.expander("Add a trade manually", expanded=False):
        with st.form("add_trade"):
            c1, c2, c3 = st.columns(3)
            trade_id = c1.text_input("Trade ID", value=f"TR-{pd.Timestamp.now().strftime('%H%M%S')}")
            product = c2.selectbox("Product", list(PRODUCTS.keys()), key="add_product")
            side = c3.selectbox("Side", ["BUY_FIXED", "SELL_FIXED"])
            c1, c2, c3 = st.columns(3)
            instrument = c1.selectbox("Instrument", ["SWAP", "FORWARD"], key="add_instrument")
            settlement_type = c2.selectbox("Settlement", ["FINANCIAL", "PHYSICAL"], key="add_settlement")
            tenor_options = [1] if instrument == "FORWARD" else [1, 3, 6, 12]
            tenor = c3.selectbox("Delivery tenor", tenor_options, index=0 if instrument == "FORWARD" else 1, key="add_tenor")
            c1, c2, c3 = st.columns(3)
            start = c1.date_input("Delivery start", value=month_start(1), key="add_start")
            mw = c2.number_input("Quantity (MW)", min_value=0.1, max_value=100000.0, value=5.0, step=0.5, key="add_mw")
            end = month_end_exclusive(start, tenor)
            c1, c2, c3 = st.columns(3)
            price = c1.number_input(f"Fixed price ({PRODUCTS[product]['unit']})", min_value=-500.0, max_value=10000.0, value=float(PRODUCTS[product]["default_price"]), step=0.5, key="add_price", help="Negative power prices are possible in European electricity markets.")
            profile = c2.selectbox("Profile", ["BASE", "PEAK"], disabled=PRODUCTS[product]["commodity"] == "Gas", key="add_profile")
            counterparty = c3.text_input("Counterparty", value="Demo counterparty", key="add_cp")
            status = st.selectbox("Status", ["PROPOSED", "CONFIRMED", "EXECUTED"], index=2)
            add = st.form_submit_button("Add trade")
        if add:
            if PRODUCTS[product]["commodity"] == "Gas":
                profile = "BASE"
            row = pd.DataFrame([{"trade_id": trade_id, "trade_date": valuation_date.isoformat(), "counterparty": counterparty,
                "product": product, "instrument": instrument, "settlement_type": settlement_type, "side": side, "delivery_start": start.isoformat(),
                "delivery_end": end.isoformat(), "profile": profile, "quantity_mw": mw, "trade_price": price,
                "currency": PRODUCTS[product]["currency"], "status": status, "notes": "Manually entered demo trade"}])
            try:
                st.session_state.trades = add_trade_to_book(st.session_state.trades, row)
                st.session_state.physical_ops = sync_physical_ops_with_trades(st.session_state.physical_ops, st.session_state.trades)
                st.success("Trade added.")
                st.rerun()
            except Exception as exc:
                st.error(f"Trade rejected: {exc}")
    try:
        marked = get_marked_trades()
        if marked.empty:
            st.info("The trade book is empty.")
        else:
            products_for_filter = ["All"] + sorted(marked["product"].dropna().unique().tolist())
            statuses_for_filter = ["All"] + sorted(marked["status"].dropna().astype(str).str.upper().unique().tolist())
            fc1, fc2 = st.columns(2)
            product_filter = fc1.selectbox("Filter product", products_for_filter, key="book_product_filter")
            status_filter = fc2.selectbox("Filter lifecycle status", statuses_for_filter, key="book_status_filter")
            view = filter_trade_book(marked, product_filter, status_filter)
            metrics = calculate_book_metrics(view)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Trades in view", metrics["trade_count"])
            c2.metric("Executed / marked", metrics["executed_count"])
            c3.metric("Unmarked active", metrics["unmarked_count"])
            c4.metric("Executed MtM · EUR", money(metrics["eur_mtm"], "EUR"))
            c5.metric("Executed MtM · GBP", money(metrics["gbp_mtm"], "GBP"))
            if metrics["unmarked_count"]:
                st.warning("Some active trades are unmarked. The reason is shown in the Mark error column; those rows are excluded from MtM sums.")
            st.caption("All KPI figures above are calculated from the same filtered rows shown below. EUR and GBP MtM are never added together.")
            display_cols = ["trade_id", "counterparty", "product", "instrument", "settlement_type", "side", "delivery_start", "delivery_end", "profile", "quantity_mw", "quantity_mwh", "trade_price", "market_price", "currency", "mtm", "status", "valuation_method", "mark_status", "mark_error"]
            display_cols = [col for col in display_cols if col in view.columns]
            st.dataframe(view[display_cols].round(4), use_container_width=True, hide_index=True)
            st.subheader("Position summary by delivery month")
            executed_view = view.loc[view["status"].eq("EXECUTED")].copy()
            month_positions = position_summary_by_delivery(executed_view, curves)
            if month_positions.empty:
                st.caption("No executed positions in the current filters.")
            else:
                month_positions["delivery_month"] = pd.to_datetime(month_positions["delivery_month"]).dt.date
                st.dataframe(month_positions.round(4), use_container_width=True, hide_index=True)
                st.caption("Net MWh is signed; gross MWh is absolute exposure before netting. If any mark is missing in a delivery bucket, bucket MtM is left blank rather than assumed.")
                fig = px.bar(month_positions, x="delivery_month", y="net_mwh", color="product", barmode="group", title="Net exposure by delivery month")
                fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, xaxis_title="Delivery month", yaxis_title="Signed MWh")
                st.plotly_chart(_styled(fig), use_container_width=True)
            st.download_button("Download filtered trade book (CSV)", view.to_csv(index=False).encode("utf-8"), "filtered_trade_book.csv", "text/csv")

            with st.expander("Trade lifecycle management", expanded=False):
                cancellable = marked.loc[~marked["status"].astype(str).str.upper().isin(["CANCELLED", "EXPIRED"])]
                if cancellable.empty:
                    st.caption("No cancellable trades in the current book.")
                else:
                    selected_trade_id = st.selectbox("Trade", cancellable["trade_id"].astype(str).tolist(), key="cancel_trade_id")
                    selected_status = str(cancellable.loc[cancellable["trade_id"].astype(str).eq(selected_trade_id), "status"].iloc[0])
                    allowed_transitions = {
                        "PROPOSED": ["CONFIRMED", "EXECUTED", "CANCELLED", "EXPIRED"],
                        "CONFIRMED": ["EXECUTED", "CANCELLED", "EXPIRED"],
                        "EXECUTED": ["CANCELLED", "EXPIRED"],
                    }.get(selected_status, [])
                    st.caption(f"Selected status: {selected_status}. Terminal statuses are retained in the blotter but excluded from exposure and MtM.")
                    if allowed_transitions:
                        target_status = st.selectbox("New status", allowed_transitions, key="target_trade_status")
                        if st.button("Apply trade status update", type="secondary", key="update_trade_status_button"):
                            try:
                                st.session_state.trades = set_trade_status(st.session_state.trades, selected_trade_id, target_status)
                                st.session_state.physical_ops = sync_physical_ops_with_trades(st.session_state.physical_ops, st.session_state.trades)
                                st.success(f"Trade {selected_trade_id} status updated to {target_status}.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Trade status was not changed: {exc}")
        with st.expander("Import trade book CSV"):
            st.caption("Required columns include trade_id, product, side, delivery_start, delivery_end, profile, quantity_mw, trade_price, currency and status. instrument defaults to SWAP and settlement_type defaults to FINANCIAL for legacy files.")
            uploaded = st.file_uploader("Upload trades CSV", type=["csv"], key="trade_upload")
            if uploaded is not None and st.button("Validate and replace trade book", key="replace_trade_book"):
                try:
                    imported = load_trade_book_csv(uploaded)
                    st.session_state.trades = imported
                    st.session_state.physical_ops = sync_physical_ops_with_trades(st.session_state.physical_ops, st.session_state.trades)
                    st.success(f"Imported {len(imported)} validated trades.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import rejected; existing book was not changed: {exc}")
    except Exception as exc:
        st.error(f"Marking failed safely: {exc}")
        st.caption("No fallback marks are invented. Check curve coverage, product, delivery dates and price units.")

elif page == "P&L & Exposure":
    title_block("P&L Sensitivity & Exposure", "Scenario-based sensitivity to an absolute parallel price move in each trade’s quote currency per MWh. This is not historical daily P&L.")
    try:
        marked = get_marked_trades()
        executed = marked.loc[marked["status"].eq("EXECUTED") & marked["mark_status"].eq("MARKED")].copy() if not marked.empty else pd.DataFrame()
        unmarked_exec = marked.loc[marked["status"].eq("EXECUTED") & marked["mark_status"].ne("MARKED")].copy() if not marked.empty else pd.DataFrame()
        if not unmarked_exec.empty:
            st.warning(f"{len(unmarked_exec)} executed trade(s) are not marked and are excluded from this scenario. Review the trade-level Mark error.")
        if executed.empty:
            st.info("No successfully marked executed trades to stress.")
        else:
            shock_per_mwh = st.slider("Absolute parallel price shock (quote currency/MWh)", min_value=-50.0, max_value=50.0, value=5.0, step=1.0)
            rows = []
            for _, tr in executed.iterrows():
                shocked_price = float(tr["market_price"]) + float(shock_per_mwh)
                delta_mtm = parallel_shock_pnl(float(tr["signed_quantity_mwh"]), float(shock_per_mwh))
                rows.append({"trade_id": tr["trade_id"], "product": tr["product"], "currency": tr["currency"],
                    "signed_quantity_mwh": tr["signed_quantity_mwh"], "current_market_price": tr["market_price"],
                    "shocked_market_price": shocked_price, "current_mtm": tr["mtm"], "scenario_pnl_change": delta_mtm})
            scenario = pd.DataFrame(rows)
            c1, c2, c3 = st.columns(3)
            c1.metric("Absolute price shock", f"{shock_per_mwh:+.2f} per MWh")
            eur_change = float(scenario.loc[scenario["currency"].eq("EUR"), "scenario_pnl_change"].sum())
            gbp_change = float(scenario.loc[scenario["currency"].eq("GBP"), "scenario_pnl_change"].sum())
            c2.metric("Scenario P&L change · EUR", money(eur_change, "EUR"))
            c3.metric("Scenario P&L change · GBP", money(gbp_change, "GBP"))
            st.dataframe(scenario.round(4), use_container_width=True, hide_index=True)
            by_product = scenario.groupby(["product", "currency"], as_index=False)["scenario_pnl_change"].sum()
            fig = px.bar(by_product, x="product", y="scenario_pnl_change", color="currency", title="Scenario P&L change by product")
            fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, yaxis_title="P&L change in each product currency", xaxis_title="")
            st.plotly_chart(_styled(fig), use_container_width=True)
            st.warning("This is a parallel absolute price shock applied in each trade’s quote currency per MWh. EUR and GBP results remain separate. It ignores curve-shape changes, basis risk, liquidity, optionality, discounting, FX movements and physical optionality. It is not VaR and not historical daily P&L.")
    except Exception as exc:
        st.error(f"Risk scenario could not be calculated: {exc}")
        st.caption("No missing marks are filled in. Review curve coverage and product/unit consistency.")

elif page == "Physical Operations":
    title_block("Physical Operations & Delivery Tracking", "Persistent nomination, delivery and invoice tracking linked to physical-settlement trades.")
    st.info("Operations edits persist in this Streamlit session and reconcile against linked CONFIRMED/EXECUTED trades flagged PHYSICAL. Records and prices remain illustrative; this is not a live nomination or logistics connection.")
    st.session_state.physical_ops = sync_physical_ops_with_trades(st.session_state.physical_ops, st.session_state.trades)
    ops = reconcile_physical_ops(st.session_state.physical_ops, st.session_state.trades)
    if ops.empty:
        st.warning("No trades are flagged for physical settlement. Add a trade with Settlement = PHYSICAL to create an operations record.")
    else:
        open_mask = ~ops["operational_status"].isin(["DELIVERED", "CANCELLED"])
        exceptions = ~ops["risk_flag"].eq("RECONCILED")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open contracts", int(open_mask.sum()))
        c2.metric("Outstanding volume", f"{ops['outstanding_mwh'].clip(lower=0).sum():,.0f} MWh")
        c3.metric("Operational exceptions", int(exceptions.sum()))
        c4.metric("Trade-link mismatches", int((ops["reconciliation"] != "MATCH").sum()))
        show_cols = ["contract_id", "trade_id", "product", "counterparty", "delivery_start", "delivery_end",
                     "profile", "contracted_mwh", "nominated_mwh", "delivered_mwh", "outstanding_mwh",
                     "nomination_gap_mwh", "delivery_vs_nomination_mwh", "operational_status", "invoice_status",
                     "trade_status", "reconciliation", "risk_flag", "updated_at"]
        st.dataframe(ops[show_cols].round(3), use_container_width=True, hide_index=True)
        st.download_button("Download physical operations (CSV)", ops.to_csv(index=False).encode("utf-8"),
                            "physical_operations_reconciliation.csv", "text/csv")
        st.subheader("Update nominations, delivery and invoice status")
        selected_contract = st.selectbox("Operations contract", ops["contract_id"].astype(str).tolist(), key="ops_selected_contract")
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
                st.session_state.physical_ops = update_physical_operation(
                    st.session_state.physical_ops, selected_contract, nominated, delivered,
                    operational_status, invoice_status
                )
                st.success(f"Saved operational updates for {selected_contract}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Operations update rejected: {exc}")
        if (ops["reconciliation"] != "MATCH").any():
            st.warning("At least one operations record does not match a linked physical-settlement trade. Correct the trade reference, product, settlement convention or contracted volume.")
        st.caption("Outstanding = contracted minus delivered. Nomination gap = contracted minus nominated. Delivery vs nomination = delivered minus nominated. Overdelivery and missing nominations are flagged rather than silently clipped.")

elif page == "Model Validation":
    title_block("Model Validation & System Checks", "Functional, numerical and integration checks for the current energy trading model — not a strategy performance backtest.")
    st.caption("This suite tests whether calculations and data controls behave as expected using deterministic, independent checks. It does not establish that any trading strategy is profitable or that this educational prototype is production-ready.")
    try:
        validation = run_model_validation(st.session_state.trades.copy(), curves)
        passed = int(validation["Status"].eq("PASS").sum())
        failed = int(validation["Status"].eq("FAIL").sum())
        total = int(len(validation))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Checks executed", str(total))
        c2.metric("Passed", str(passed))
        c3.metric("Failed", str(failed))
        c4.metric("Pass rate", f"{100.0 * passed / total:.1f}%" if total else "n/a")
        if failed:
            st.error(f"{failed} check(s) failed. Review the details below before relying on the affected calculation or input data.")
        else:
            st.success("All automated checks passed for the calculation engine and current data shown in this session.")
        st.dataframe(validation, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download validation report (CSV)", data=validation.to_csv(index=False).encode("utf-8"), file_name="energy_model_validation_report.csv", mime="text/csv")
        with st.expander("What is tested and what is not", expanded=True):
            st.markdown("""
            **Covered by these checks:** product configuration, energy and gas price conversions, European daylight-saving delivery hours, peak-hour counting, forward-curve validation, trade schema and lifecycle filtering, forward-versus-swap contract conventions, OTC quote margins, MtM signs and formula, daily P&L arithmetic, monthly delivery position splitting, per-trade marking failure isolation, CSV import and cancellation workflow, persistent physical-operation updates, operations-to-trade reconciliation, filtered-KPI consistency, negative power prices, invalid-input rejection, navigation registration, and marking the current trade book.

            **Not equivalent to a production certification:** the checks do not validate real exchange data, public holidays and exchange-specific calendars, full swap payment schedules and discount curves, accounting policy, collateral, credit valuation adjustments, licensed data APIs, concurrent-user behavior, or real-browser rendering. Built-in prices and trades remain illustrative.
            """)
        current_failures = validation.loc[validation["Status"].eq("FAIL") & validation["Area"].eq("Current user data")]
        if not current_failures.empty:
            st.warning("At least one check concerns the currently loaded user data rather than the core formulas. Review the specific trade-level mark error before relying on its valuation.")
    except Exception as exc:
        st.error(f"The validation suite could not complete: {type(exc).__name__}: {exc}")


elif page == "Live Data Integrations":
    title_block("Live Data Integrations", "Connect public transparency APIs and your licensed market-data source. API calls run only when you explicitly fetch data.")
    st.caption("Your AGSI+ key is preconfigured in this local source file. Keep the file private; for shared/deployed copies use Streamlit secrets or environment variables. The same GIE field is used for ALSI only if your key is entitled for that dataset. Do not commit credentials to a public repository.")

    # Streamlit secrets/environment override the locally embedded AGSI+ default.
    # This means the supplied AGSI+ key is available without manual re-entry in this file.
    gie_default = _secret_or_env("GIE_API_KEY", "GIE_API_KEY") or GIE_AGSI_API_KEY_DEFAULT
    entsoe_default = _secret_or_env("ENTSOE_API_TOKEN", "ENTSOE_API_TOKEN")
    openmeteo_default = _secret_or_env("OPEN_METEO_API_KEY", "OPEN_METEO_API_KEY")
    with st.expander("Credentials & access setup", expanded=True):
        c1, c2, c3 = st.columns(3)
        gie_api_key = c1.text_input("GIE API key (AGSI+ / ALSI)", value=gie_default, type="password", key="live_gie_api_key", help="The AGSI+ key you supplied is prefilled automatically. ALSI requests require your key to be entitled for that dataset.")
        entsoe_api_token = c2.text_input("ENTSO-E Web API token", value=entsoe_default, type="password", key="live_entsoe_token", help="Register on transparency.entsoe.eu and request RESTful API access by email.")
        openmeteo_api_key = c3.text_input("Open-Meteo commercial API key (optional)", value=openmeteo_default, type="password", key="live_openmeteo_key", help="Leave blank to use the public evaluation endpoint. Follow provider licensing and attribution terms.")
        st.markdown("**API access:** [AGSI+ API](https://agsi.gie.eu/#/api) · [ALSI API](https://alsi.gie.eu/#/api) · [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) · [ENTSOG API manual](https://transparency.entsog.eu/api/archiveDirectories/8/api-manual/ENTSOG_TP_API_UserManual_v3.0.pdf) · [Open-Meteo API](https://open-meteo.com/en/docs) · [ICE Endex data](https://developer.ice.com/fixed-income-data-services/catalog/ice-endex)")
        st.caption("GIE requires a registered personal key and selected access to AGSI+, ALSI, or both. ENTSO-E requires a registered account and activation of RESTful API access. ENTSOG is public. Open-Meteo's public endpoint is subject to non-commercial-use terms; check its commercial plan before commercial deployment.")
        with st.expander("How to obtain your ENTSO-E API token", expanded=False):
            st.markdown("""1. Create an account on the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).
2. From the email address used for registration, send a request to **transparency@entsoe.eu** with the subject **Restful API access**. Include that same registered email address in the message body.
3. Wait for ENTSO-E to enable REST API access. Their process/documentation describes this as an access request, not an instant self-service key.
4. Sign in again and open **Account Settings**. If the **Web API Security Token** field is available, choose the option to generate a token and copy it somewhere secure. The token is not displayed again after the dialog is closed.
5. Paste the token into the masked **ENTSO-E Web API token** field above or set `ENTSOE_API_TOKEN` in your local environment. Never commit the token to Git or send it in chat.

Official [token-management instructions](https://transparency.entsoe.eu/content/static_content/download?path=%2FStatic+content%2FAPI-Token-Management.pdf).""")
            st.info("A token can be unavailable until ENTSO-E activates REST access for the registered email. If the token field does not appear, contact transparency@entsoe.eu rather than repeatedly generating a new account.")

    tab_gie, tab_power, tab_gasflows, tab_weather, tab_forward = st.tabs([
        "Gas Storage & LNG", "Power Market · ENTSO-E", "Gas Flows · ENTSOG", "Weather", "Licensed Forward Curves"
    ])

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
                    frame = fetch_gie_history(dataset_code, gie_api_key, gie_scope, gie_from, gie_to, gie_country)
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
                chart = gie_frame.copy()
                chart[date_axis] = pd.to_datetime(chart[date_axis], errors="coerce")
                fig = px.line(chart, x=date_axis, y=value_axis, color=color_axis, markers=True, title=f"{gie_dataset} · {value_axis}")
                fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=350)
                st.plotly_chart(_styled(fig), use_container_width=True)
            st.dataframe(gie_frame, use_container_width=True, hide_index=True)
            st.download_button("Download GIE data (CSV)", gie_frame.to_csv(index=False).encode("utf-8"), file_name=f"{gie_show_dataset}.csv", mime="text/csv", key=f"download_{gie_show_dataset}")
        else:
            st.info("No GIE dataset has been fetched in this session yet.")

    with tab_power:
        st.subheader("ENTSO-E electricity transparency data")
        st.write("Fetch day-ahead prices, actual total load, actual generation by production type, or cross-border physical flows. Day-ahead prices are kept separate from forward curves and are not used to mark monthly OTC positions.")
        with st.form("entsoe_live_fetch_form"):
            e1, e2, e3 = st.columns(3)
            entsoe_kind = e1.selectbox("Dataset", ["Day-ahead price", "Actual load", "Actual generation by type", "Physical flow"], key="entsoe_kind_choice")
            entsoe_zone = e2.selectbox("Bidding zone", list(ENTSOE_ZONES.keys()), index=list(ENTSOE_ZONES.keys()).index("France"), key="entsoe_zone_choice")
            entsoe_source = e3.selectbox("Flow source zone", list(ENTSOE_ZONES.keys()), index=list(ENTSOE_ZONES.keys()).index("France"), key="entsoe_flow_source")
            e1, e2, e3 = st.columns(3)
            entsoe_target = e1.selectbox("Flow destination zone", list(ENTSOE_ZONES.keys()), index=list(ENTSOE_ZONES.keys()).index("Germany-Luxembourg"), key="entsoe_flow_target")
            entsoe_from = e2.date_input("Delivery start date", value=date.today() - timedelta(days=7), key="entsoe_start_date")
            entsoe_to = e3.date_input("Delivery end date (exclusive)", value=date.today(), key="entsoe_end_date")
            entsoe_submit = st.form_submit_button("Fetch ENTSO-E series", type="primary")
        if entsoe_submit:
            try:
                selected_zone = entsoe_source if entsoe_kind == "Physical flow" else entsoe_zone
                with st.spinner("Requesting ENTSO-E time series…"):
                    frame = fetch_entsoe_series(entsoe_api_token, entsoe_kind, selected_zone, entsoe_from, entsoe_to,
                                                source_zone=entsoe_source, target_zone=entsoe_target)
                key = {"Day-ahead price": "entsoe_price", "Actual load": "entsoe_load", "Actual generation by type": "entsoe_generation", "Physical flow": "entsoe_flow"}[entsoe_kind]
                _stored_live_item(key, entsoe_kind, frame, {"zone": selected_zone, "source_zone": entsoe_source, "target_zone": entsoe_target})
                st.success(f"Loaded {len(frame):,} ENTSO-E observations.")
            except Exception as exc:
                st.error(f"ENTSO-E request failed: {exc}")
        entsoe_data_key = {"Day-ahead price": "entsoe_price", "Actual load": "entsoe_load", "Actual generation by type": "entsoe_generation", "Physical flow": "entsoe_flow"}[entsoe_kind]
        entsoe_frame = _live_dataset_frame(entsoe_data_key)
        if not entsoe_frame.empty:
            st.caption(f"Last fetch: {st.session_state.live_data[entsoe_data_key]['fetched_at']} · Data unit: {entsoe_frame['unit'].iloc[0] if 'unit' in entsoe_frame.columns else 'provider-defined'}")
            entsoe_display = entsoe_frame.copy()
            entsoe_display["timestamp"] = pd.to_datetime(entsoe_display["timestamp"], errors="coerce", utc=True)
            fig = px.line(entsoe_display, x="timestamp", y="value", color="series" if entsoe_display["series"].nunique() > 1 else None,
                          markers=False, title=str(st.session_state.live_data[entsoe_data_key]["label"]))
            fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=350, xaxis_title="Delivery timestamp (UTC)", yaxis_title=str(entsoe_display["unit"].iloc[0]))
            st.plotly_chart(_styled(fig), use_container_width=True)
            st.dataframe(entsoe_display, use_container_width=True, hide_index=True)
            st.download_button("Download ENTSO-E data (CSV)", entsoe_display.to_csv(index=False).encode("utf-8"), file_name=f"{entsoe_data_key}.csv", mime="text/csv", key=f"download_{entsoe_data_key}")
        else:
            st.info("No ENTSO-E series of this type has been fetched in this session yet.")

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
            st.dataframe(shown.head(100), use_container_width=True, hide_index=True)
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
                chart = entsog_frame.dropna(subset=["timestamp", "value_numeric"]).copy()
                if not chart.empty:
                    fig = px.line(chart, x="timestamp", y="value_numeric", markers=True, title="ENTSOG daily flow / indicator")
                    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=350)
                    st.plotly_chart(_styled(fig), use_container_width=True)
            st.dataframe(entsog_frame, use_container_width=True, hide_index=True)
            st.download_button("Download ENTSOG data (CSV)", entsog_frame.to_csv(index=False).encode("utf-8"), file_name="entsog_flows.csv", mime="text/csv", key="download_entsog_flow")
        st.markdown("[Open the ENTSOG Transparency Platform](https://transparency.entsog.eu/) for the official catalogue and point details.")

    with tab_weather:
        st.subheader("Weather-driven demand and renewable proxies")
        st.write("Open-Meteo provides hourly temperature, wind speed and solar radiation. These are contextual proxies—not forecasts of power prices or exact system demand.")
        with st.form("open_meteo_live_fetch_form"):
            w1, w2, w3 = st.columns(3)
            weather_location = w1.selectbox("Reference location", list(WEATHER_LOCATIONS.keys()), key="weather_location_choice")
            weather_past = w2.selectbox("Past days", [0, 1, 3, 5, 7], index=2, key="weather_past_days")
            weather_forecast = w3.selectbox("Forecast days", [3, 5, 7, 10, 14, 16], index=2, key="weather_forecast_days")
            weather_commercial = st.checkbox("Use Open-Meteo commercial endpoint", value=False, key="weather_commercial_endpoint")
            weather_submit = st.form_submit_button("Fetch weather data", type="primary")
        if weather_submit:
            try:
                loc = WEATHER_LOCATIONS[weather_location]
                with st.spinner("Requesting weather data…"):
                    hourly, daily = fetch_open_meteo(loc["latitude"], loc["longitude"], loc["timezone"], weather_past,
                                                    weather_forecast, openmeteo_api_key, weather_commercial)
                _stored_live_item("weather_hourly", f"Open-Meteo hourly · {weather_location}", hourly, {"location": weather_location})
                _stored_live_item("weather_daily", f"Open-Meteo daily · {weather_location}", daily, {"location": weather_location})
                st.success(f"Loaded {len(hourly):,} hourly and {len(daily):,} daily weather observations.")
            except Exception as exc:
                st.error(f"Open-Meteo request failed: {exc}")
        weather_daily = _live_dataset_frame("weather_daily")
        if not weather_daily.empty:
            st.caption(f"Last fetch: {st.session_state.live_data['weather_daily']['fetched_at']} · Attribution: Open-Meteo / CC BY 4.0; check applicable usage terms.")
            if "temperature_mean_c" in weather_daily.columns:
                fig = px.line(weather_daily, x="date", y="temperature_mean_c", markers=True, title="Daily mean temperature")
                fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=330, yaxis_title="°C")
                st.plotly_chart(_styled(fig), use_container_width=True)
            st.dataframe(weather_daily, use_container_width=True, hide_index=True)
            hourly_weather = _live_dataset_frame("weather_hourly")
            st.download_button("Download hourly weather (CSV)", hourly_weather.to_csv(index=False).encode("utf-8"), file_name="open_meteo_hourly.csv", mime="text/csv", key="download_weather_hourly")
        else:
            st.info("No weather series has been fetched in this session yet. The public endpoint does not require an API key for evaluation.")

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
            st.dataframe(raw_table.head(100), use_container_width=True, hide_index=True)
            columns = [c for c in raw_table.columns if not str(c).startswith("_")]
            date_guess = next((c for c in columns if any(t in str(c).lower() for t in ("delivery_month", "deliverydate", "delivery_date", "maturity", "tenor_date", "contract_date", "month"))), columns[0])
            price_guess = next((c for c in columns if any(t in str(c).lower() for t in ("settlement", "settle", "price", "close", "last"))), columns[-1])
            m1, m2, m3 = st.columns(3)
            mapped_product = m1.selectbox("Canonical product to update", list(PRODUCTS.keys()), key="provider_curve_product")
            mapped_date_col = m2.selectbox("Delivery date column", columns, index=columns.index(date_guess) if date_guess in columns else 0, key="provider_curve_date_col")
            mapped_price_col = m3.selectbox("Price column", columns, index=columns.index(price_guess) if price_guess in columns else len(columns)-1, key="provider_curve_price_col")
            if st.button("Validate and apply monthly forward curve", key="apply_provider_curve_button"):
                try:
                    candidate_curve = normalise_provider_curve(raw_table, mapped_product, mapped_date_col, mapped_price_col)
                    st.session_state.custom_curves[mapped_product] = candidate_curve
                    st.session_state.curve_metadata[mapped_product] = {"source_label": "LICENSED PROVIDER IMPORT", "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source_type": "REST endpoint"}
                    st.success(f"Applied {len(candidate_curve):,} validated monthly marks to {mapped_product} for this session. Revaluation uses these marks; save an authorised export separately if you need persistence.")
                    st.dataframe(candidate_curve, use_container_width=True, hide_index=True)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Curve rejected: {exc}")
        else:
            st.info("No licensed provider table loaded yet. Use the endpoint connector or upload a permitted CSV/JSON export.")
        st.caption("Provider APIs have vendor-specific URLs, response schemas, authentication and redistribution terms. If your licence uses a different authentication method or schema, adapt the connector to that official specification; do not paste vendor secrets into source code.")

elif page == "Market Fundamentals":
    title_block("European Energy Fundamentals", "Translate fetched storage, LNG, power, gas-flow and weather observations into transparent market context.")
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
    if top_c.button("Manage / refresh data sources", type="primary", use_container_width=True):
        st.session_state.nav_page = next(label for label in ALL_NAV_ITEMS if label.endswith("Live Data Integrations"))
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
            st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
            st.caption("The observation timestamp describes the market/fundamental period; Last fetched describes when this app retrieved it. These are not interchangeable. GIE AGSI+/ALSI attribution is mandatory.")

    st.subheader("Derived market indicators")
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
    if weather:
        w = []
        if weather.get("temperature_c") is not None: w.append(f"mean temperature {weather['temperature_c']:.1f} °C")
        if weather.get("wind_kmh") is not None: w.append(f"mean wind {weather['wind_kmh']:.1f} km/h")
        if weather.get("solar_wm2") is not None: w.append(f"mean solar radiation {weather['solar_wm2']:.1f} W/m²")
        if w:
            insights.append({"Topic": "Weather", "Observation": f"{weather.get('location', 'Selected location')}: " + "; ".join(w), "Commercial reading": "Use as a local demand/renewables covariate. One city is not a weighted national forecast and does not directly predict hub prices."})
    if insights:
        st.dataframe(pd.DataFrame(insights), use_container_width=True, hide_index=True)
    else:
        st.info("Fetch one or more feeds to generate derived observations. Missing data are left blank rather than filled with fabricated values.")

    fund_tabs = st.tabs(["Gas storage", "LNG", "Power", "Gas flows", "Weather", "Cross-market analysis"])
    with fund_tabs[0]:
        data = _live_dataset_frame("gie_agsi")
        st.subheader("European gas storage — GIE AGSI+")
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
                st.plotly_chart(_styled(fig, 360), use_container_width=True)
                if fill_col:
                    fig_fill = px.line(frame, x=date_col, y=fill_col, markers=True, title="Reported storage fill level", labels={fill_col: "Fill (%)", date_col: "Gas day"})
                    fig_fill.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=300)
                    st.plotly_chart(_styled(fig_fill), use_container_width=True)
                elif cap_col:
                    frame["derived_fill_pct"] = frame[value_col] / frame[cap_col] * 100.0
                    st.caption("Fill percentage is derived as inventory / working gas volume because the feed did not provide a usable full field.")
                    fig_fill = px.line(frame, x=date_col, y="derived_fill_pct", markers=True, title="Derived storage fill level", labels={"derived_fill_pct": "Fill (%)", date_col: "Gas day"})
                    fig_fill.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=300)
                    st.plotly_chart(_styled(fig_fill), use_container_width=True)
            st.dataframe(data, use_container_width=True, hide_index=True)
        else:
            st.info("Fetch AGSI+ in Live Data Integrations. Choose the EU aggregate if you want an aggregate-level series; a country/facility dataset must not be summed blindly.")

    with fund_tabs[1]:
        data = _live_dataset_frame("gie_alsi")
        st.subheader("LNG terminals — GIE ALSI+")
        if not data.empty:
            frame, date_col = _frame_with_time(data, ("observation_date", "gasDayStart", "gas_day"))
            inv_col = _find_column(frame, ("inventory", "inventory_lng", "inventory_gwh", "inventory_twh"))
            send_col = _find_column(frame, ("sendOut", "sendout", "send_out", "sendout_gwh"))
            st.caption(f"Source: GIE ALSI+ · scope: {_live_item_meta('gie_alsi').get('scope', 'selected scope')} · inventory units follow the field returned by GIE (often 10³ m³ LNG); send-out is commonly GWh/d.")
            if date_col and inv_col:
                frame[inv_col] = _number_series(frame[inv_col])
                fig = px.line(frame, x=date_col, y=inv_col, color="name" if "name" in frame.columns and frame["name"].nunique() > 1 else None, markers=True, title=f"LNG inventory · {inv_col}")
                fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=330, xaxis_title="Gas day", yaxis_title="Provider field unit")
                st.plotly_chart(_styled(fig), use_container_width=True)
            if date_col and send_col:
                frame[send_col] = _number_series(frame[send_col])
                fig = px.line(frame, x=date_col, y=send_col, color="name" if "name" in frame.columns and frame["name"].nunique() > 1 else None, markers=True, title="LNG send-out")
                fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=330, xaxis_title="Gas day", yaxis_title="GWh/d (confirm selected field/unit)")
                st.plotly_chart(_styled(fig), use_container_width=True)
            st.dataframe(data, use_container_width=True, hide_index=True)
        else:
            st.info("Fetch ALSI+ in Live Data Integrations. Inventory and send-out use different physical units and must not be combined as if they were comparable.")

    with fund_tabs[2]:
        st.subheader("Power prices, load and generation — ENTSO-E")
        power_keys = [("entsoe_price", "Day-ahead price"), ("entsoe_load", "Actual load"), ("entsoe_generation", "Actual generation by type"), ("entsoe_flow", "Physical interconnector flow")]
        available = [(key, label) for key, label in power_keys if not _live_dataset_frame(key).empty]
        if available:
            selected_power_key = st.selectbox("Loaded ENTSO-E dataset", [x[0] for x in available], format_func=lambda x: dict(available)[x], key="fundamentals_entsoe_display_key")
            data = _live_dataset_frame(selected_power_key).copy()
            meta = _live_item_meta(selected_power_key)
            st.caption(f"Source: ENTSO-E Transparency Platform · {dict(available)[selected_power_key]} · zone: {meta.get('zone', 'not stored')} · last fetched {_live_item_fetched_at(selected_power_key)}")
            if {"timestamp", "value"}.issubset(data.columns):
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
                data["value"] = _number_series(data["value"])
                data = data.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
                chart = px.line(data, x="timestamp", y="value", color="series" if "series" in data.columns and data["series"].nunique() > 1 else None,
                                title=dict(available)[selected_power_key], labels={"timestamp": "Delivery timestamp (UTC)", "value": str(data["unit"].iloc[0]) if "unit" in data.columns and not data.empty else "Value"})
                chart.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=360)
                st.plotly_chart(_styled(chart), use_container_width=True)
                if selected_power_key == "entsoe_price":
                    neg_count = int((data["value"] < 0).sum())
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Last price", f"{data['value'].iloc[-1]:.2f} EUR/MWh")
                    p2.metric("Sample mean", f"{data['value'].mean():.2f} EUR/MWh")
                    p3.metric("Sample low", f"{data['value'].min():.2f} EUR/MWh")
                    p4.metric("Negative observations", str(neg_count))
                    st.caption("The count is observations, not necessarily hours. Use the data's resolution and bidding-zone conventions when interpreting negative-price periods.")
                if selected_power_key == "entsoe_generation" and "series" in data:
                    mix = data.groupby("series", as_index=False)["value"].mean().sort_values("value", ascending=False)
                    mix_fig = px.bar(mix, x="series", y="value", title="Average output by reported production type", labels={"series": "Production type", "value": str(data["unit"].iloc[0]) if "unit" in data.columns else "Value"})
                    mix_fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=350, xaxis_title="")
                    st.plotly_chart(_styled(mix_fig), use_container_width=True)
                st.dataframe(data, use_container_width=True, hide_index=True)
        else:
            st.info("Fetch a price, load, generation or flow series from ENTSO-E. The API requires approved REST access and a security token.")

    with fund_tabs[3]:
        data = _live_dataset_frame("entsog_flow")
        st.subheader("Gas infrastructure flows — ENTSOG")
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
                    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=350, xaxis_title="Provider observation period", yaxis_title="Provider value — verify unit")
                    st.plotly_chart(_styled(fig), use_container_width=True)
            st.dataframe(data, use_container_width=True, hide_index=True)
            st.caption("A network flow at a specific point is not automatically a delivery to a contractual counterparty; direction and physical units must be reconciled.")
        else:
            st.info("Fetch the ENTSOG point-direction catalogue, select an exact identifier, and fetch the chosen indicator in Live Data Integrations.")

    with fund_tabs[4]:
        data = _live_dataset_frame("weather_daily")
        st.subheader("Weather proxies — Open-Meteo")
        if not data.empty:
            meta = _live_item_meta("weather_daily")
            st.caption(f"Source: Open-Meteo · location: {meta.get('location', 'selected location')} · last fetched {_live_item_fetched_at('weather_daily')}")
            y_cols = [c for c in ("temperature_mean_c", "wind_speed_mean_kmh", "solar_radiation_mean_wm2") if c in data.columns]
            if "date" in data.columns and y_cols:
                chart = data.copy()
                chart["date"] = pd.to_datetime(chart["date"], errors="coerce")
                fig = go.Figure()
                for col in y_cols:
                    fig.add_trace(go.Scatter(x=chart["date"], y=chart[col], mode="lines+markers", name=col.replace("_", " ")))
                fig.update_layout(title="Daily weather context", xaxis_title="Local date", yaxis_title="Each series retains its own native unit")
                st.plotly_chart(_styled(fig, 360), use_container_width=True)
            st.dataframe(data, use_container_width=True, hide_index=True)
            st.caption("Weather data support scenario context but are not a direct power-price forecast. One city should not be treated as a country-wide weighted weather index.")
        else:
            st.info("Fetch a location-specific forecast/history from Open-Meteo in Live Data Integrations.")

    with fund_tabs[5]:
        st.subheader("Cross-market context: residual load proxy vs day-ahead price")
        residual = _build_residual_load_proxy()
        if not residual.empty:
            st.caption("Proxy definition: actual system load minus reported actual wind-onshore, wind-offshore and solar output. Data are aligned to UTC hourly buckets using averages. This omits other generation, imports/exports, storage, curtailment and losses; it is not a full residual-load model.")
            fig = px.line(residual, x="hour", y="residual_load_proxy_mw", title="Hourly residual-load proxy", labels={"hour": "UTC hour", "residual_load_proxy_mw": "MW"})
            fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=330)
            st.plotly_chart(_styled(fig), use_container_width=True)
            scatter = px.scatter(residual, x="residual_load_proxy_mw", y="day_ahead_eur_mwh",
                                 title="Day-ahead price vs residual-load proxy", labels={"residual_load_proxy_mw": "Residual-load proxy (MW)", "day_ahead_eur_mwh": "Day-ahead price (EUR/MWh)"})
            scatter.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=350)
            st.plotly_chart(_styled(scatter), use_container_width=True)
            corr = residual[["residual_load_proxy_mw", "day_ahead_eur_mwh"]].corr().iloc[0, 1] if len(residual) >= 3 else float("nan")
            st.metric("Sample Pearson correlation", f"{corr:.3f}" if pd.notna(corr) else "Insufficient overlap", help="Descriptive correlation only. It is not causation and is not a trading signal.")
            st.download_button("Download aligned power context (CSV)", residual.to_csv(index=False).encode("utf-8"), file_name="meridian_power_context.csv", mime="text/csv", key="download_residual_load_context")
        else:
            st.info("To create this comparison, fetch overlapping ENTSO-E day-ahead prices, actual load and actual generation by type for the same bidding zone and dates. The chart will remain unavailable rather than fabricating a proxy.")

    st.markdown("### Interpretation and limitations")
    st.markdown("- **GIE AGSI+/ALSI:** storage and LNG inventories, injections/withdrawals and send-out inform gas supply context. GIE attribution is mandatory.")
    st.markdown("- **ENTSO-E:** day-ahead prices, actual load and generation are spot/fundamental observations. They do not replace monthly OTC forward marks.")
    st.markdown("- **ENTSOG:** a point-direction transport observation must retain its direction, period and provider unit; do not equate flows with contract deliveries.")
    st.markdown("- **Open-Meteo:** weather covariates can contextualise demand and renewable production but do not constitute a price forecast.")
    st.markdown("- **ICE Endex/EEX:** a licensed futures curve or authorised vendor export is still required to make TTF/NBP/power OTC MtM market-referenced.")
    st.caption("All derived indicators are descriptive. APIs may publish delayed observations or retroactive corrections; user-provided data are not independently certified by this application.")

elif page == "Data & Assumptions":
    title_block("Data, Conventions & Validation", "Explicit assumptions and import validation are part of the risk controls, not an afterthought.")
    st.subheader("Product registry")
    registry = pd.DataFrame([{"Product": p, **spec} for p, spec in PRODUCTS.items()])
    st.dataframe(registry, use_container_width=True, hide_index=True)
    st.subheader("Upload a forward curve")
    st.write("Required CSV columns: `delivery_month`, `product`, `price`, `currency`, `unit`. One row per product and delivery month. Prices must use a consistent unit for the selected product. Negative prices are allowed because electricity prices can be negative.")
    curve_file = st.file_uploader("Upload monthly curve CSV", type=["csv"], key="curve_upload")
    if curve_file is not None:
        try:
            candidate = validate_curve(pd.read_csv(curve_file))
            chosen = st.selectbox("Product to replace", list(PRODUCTS.keys()), key="replace_curve_product")
            subset = candidate.loc[candidate["product"].eq(chosen)].copy()
            if subset.empty:
                st.error(f"No rows for {chosen} in uploaded file.")
            else:
                expected = PRODUCTS[chosen]
                if not subset["currency"].eq(expected["currency"]).all() or not subset["unit"].eq(expected["unit"]).all():
                    st.error(f"Unit/currency mismatch. Expected {expected['currency']} and {expected['unit']} for {chosen}.")
                else:
                    st.dataframe(subset, use_container_width=True, hide_index=True)
                    if st.button("Use uploaded curve for this session"):
                        st.session_state.custom_curves[chosen] = subset
                        st.success("Curve applied for this session. The app does not persist it to disk.")
                        st.rerun()
        except Exception as exc:
            st.error(f"Curve rejected: {exc}")
    st.subheader("Unit conversion utilities")
    a, b = st.columns(2)
    mwh = a.number_input("Energy (MWh)", min_value=0.0, value=1000.0, step=100.0)
    eur_mwh = b.number_input("Gas price (EUR/MWh)", min_value=0.0, value=35.0, step=1.0)
    fx = st.number_input("FX: EUR per GBP", min_value=0.01, value=1.17, step=0.01, help="Illustrative FX input. 1 GBP = this many EUR.")
    st.write(f"**Energy conversion:** {mwh:,.2f} MWh = {mwh_to_mmbtu(mwh):,.2f} MMBtu")
    st.write(f"**Price conversion:** {eur_mwh:.4f} EUR/MWh = {price_eur_mwh_to_gbp_therm(eur_mwh, fx):.6f} GBP/therm at the entered FX rate")
    st.subheader("Model conventions and limits")
    st.markdown("""
    - **End dates are exclusive:** a delivery from 1 January to 1 February includes January only.
    - **Power volume:** contracted MWh = MW × delivery hours. BASE uses actual local clock hours (including DST 23/25-hour days); PEAK counts weekdays 08:00–20:00 local time.
    - **Calendar limitation:** public holidays and exchange-specific peak / gas-day exceptions are not included.
    - **MtM:** signed MWh × (market forward − fixed trade price). BUY_FIXED is long; SELL_FIXED is short.
    - **Curve matching:** every touched delivery month must have exactly one price mark. Missing marks raise an error; the app does not invent or interpolate prices.
    - **Currency:** trades are marked in their quote currency. Do not sum EUR and GBP MtM without explicit FX conversion.
    - **NBP unit convention:** the simplified trade book normalises NBP to GBP/MWh. Convert market quotations published in p/therm or GBP/therm before importing; no silent conversion is performed.
    - **P&L:** the current implementation calculates indicative unrealised MtM; it does not yet produce a production-grade realised/unrealised daily P&L ledger.
    - **Valuation:** no credit valuation adjustment, collateral, bid/ask liquidity, discount curve, shaping premium, imbalance, fees, or exchange settlement rules are modelled.
    """)
    st.subheader("Reset demo data")
    if st.button("Reset trade book and uploaded curves"):
        st.session_state.trades = build_sample_trades()
        st.session_state.custom_curves = {}
        st.session_state.physical_ops = build_sample_physical_ops(st.session_state.trades)
        st.success("Demo book, custom curves and physical operations reset.")
        st.rerun()


