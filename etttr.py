"""MERIDIAN Energy Desk — European energy trading and risk management prototype.

Run with: streamlit run meridian_energy_desk.py

IMPORTANT: Built-in curves and trades are synthetic demo data, not live market quotes.
This is an educational prototype, not a production ETRM or valuation system.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo
import math
import io
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


NAV_SECTIONS = {
    "Markets": ["📊  Market Overview", "📈  Forward Curves", "🧭  Market Fundamentals"],
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
        st.markdown('<span class="badge badge-amber">DEMO MODE · NO LIVE MARKET FEED</span>', unsafe_allow_html=True)
        for section, items in NAV_SECTIONS.items():
            key = f"navsec_{section}"
            if key not in st.session_state:
                st.session_state[key] = current if current in items else None
            st.radio(section, items, key=key, on_change=_on_nav, args=(section,))
        if st.button("↻ Reset demo book & curves", use_container_width=True):
            st.session_state.trades = build_sample_trades()
            st.session_state.custom_curves = {}
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
            st.caption("No live exchange, ENTSO-E, ENTSOG or GIE API is connected in this version.")
            st.caption("Built-in forward curves and sample trades are illustrative. Uploaded curves are session-only.")
            st.caption("Model Validation runs deterministic functional checks against the calculation engine and current demo/user data.")
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
if "physical_ops" not in st.session_state:
    st.session_state.physical_ops = build_sample_physical_ops(st.session_state.trades)
else:
    st.session_state.physical_ops = sync_physical_ops_with_trades(
        st.session_state.physical_ops, st.session_state.trades
    )

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
                    "P&L & Exposure", "Physical Operations", "Market Fundamentals", "Model Validation", "Data & Assumptions"}
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
    c3.metric("Illustrative MtM · EUR", money(eur_mtm, "EUR"))
    c4.metric("Illustrative MtM · GBP", money(gbp_mtm, "GBP"))
    c5.metric("Unmarked active trades", f"{unmarked_count}")
    if unmarked_count:
        st.warning(f"{unmarked_count} active trade(s) could not be marked. Review the trade-level diagnostic in Trading Book & MtM.")
    st.caption("MtM is reported separately by quote currency. Values are based on illustrative curves and are not live valuations.")
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Forward curves")
        selected = st.selectbox("Product", list(PRODUCTS.keys()), key="overview_product")
        curve = curves[selected].copy()
        fig = px.line(curve, x="delivery_month", y="price", markers=True,
                      title=f"{selected} — illustrative monthly curve",
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
    watch = pd.DataFrame([{"Product": p, "Currency": s["currency"], "Quote unit": s["unit"], "Curve status": "Illustrative demo curve"} for p, s in PRODUCTS.items()])
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["delivery_month"], y=curve["price"], mode="lines+markers", name="Monthly forward", line=dict(color=AMBER, width=2)))
    fig.add_hline(y=float(curve["price"].mean()), line=dict(color=BLUE, dash="dot"), annotation_text="Curve average")
    fig.update_layout(title=f"{selected} — monthly forward curve", xaxis_title="Delivery month", yaxis_title=PRODUCTS[selected]["unit"])
    st.plotly_chart(_styled(fig, 440), use_container_width=True)
    st.subheader("Monthly delivery marks")
    display_curve = curve.copy()
    display_curve["delivery_month"] = display_curve["delivery_month"].dt.date
    st.dataframe(display_curve, use_container_width=True, hide_index=True)
    st.download_button("Download displayed curve (CSV)", data=display_curve.to_csv(index=False).encode("utf-8"), file_name=f"{selected.lower().replace(' ', '_')}_forward_curve.csv", mime="text/csv")
    st.caption("Curve shape and average are descriptive only. These marks are not a bid, offer, exchange settlement, or executable quote.")

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
                "currency": PRODUCTS[product]["currency"], "unit": PRODUCTS[product]["unit"]}
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
        st.write(f"**Counterparty:** {q['counterparty']}  ·  **Delivery:** {q['start']} to {q['end']} (end date exclusive)  ·  **Status:** Indicative only")
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


elif page == "Market Fundamentals":
    title_block("European Energy Fundamentals", "A place for market context: demand, generation, storage, flows and price spreads.")
    st.warning("No live ENTSO-E, ENTSOG, GIE or exchange feed is connected. The charts below are explicitly synthetic demonstrations; do not use them as market analysis.")
    product = st.selectbox("Illustrative price series", list(PRODUCTS.keys()))
    curve = curves[product].copy()
    curve["price_change_pct"] = curve["price"].pct_change() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["delivery_month"], y=curve["price"], mode="lines+markers", name="Illustrative forward"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, title=f"{product}: monthly forward curve", xaxis_title="Delivery month", yaxis_title=PRODUCTS[product]["unit"], height=400)
    st.plotly_chart(_styled(fig), use_container_width=True)
    st.subheader("Recommended real-data integrations")
    st.markdown("""
    - **ENTSO-E Transparency Platform:** power load, generation by type, cross-border flows and outages.
    - **ENTSOG Transparency Platform:** gas flows, capacity and infrastructure information.
    - **Gas Infrastructure Europe (AGSI/ALSI):** gas storage and LNG inventory information.
    - **Exchange or licensed market-data provider:** tradable spot and forward prices, with appropriate licensing.
    """)
    st.caption("For a defensible fundamental view, join market data on timestamps, bidding zones and delivery periods; record publication time and source, and never mix currencies or units implicitly.")

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


