"""European Energy Trading & Risk Management — enhanced single-file Streamlit app.

Run with: streamlit run energy_trading_desk_enhanced.py

IMPORTANT: Built-in curves and trades are synthetic demo data, not live market quotes.
This is an educational prototype, not a production ETRM or valuation system.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo
import math
from typing import Iterable, Sequence, Optional, Tuple

import numpy as np
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
    "trade_price", "currency", "status", "notes",
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
    s = side.upper()
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
         "product": "TTF Natural Gas", "instrument": "SWAP", "side": "BUY_FIXED", "delivery_start": _month_start(1, d0).isoformat(),
         "delivery_end": _month_start(4, d0).isoformat(), "profile": "BASE", "quantity_mw": 10.0,
         "trade_price": 35.5, "currency": "EUR", "status": "EXECUTED", "notes": "Illustrative long fixed-price gas swap"},
        {"trade_id": "TR-1002", "trade_date": valuation_date.isoformat(), "counterparty": "Alpine Power (Demo)",
         "product": "French Power Peak", "instrument": "FORWARD", "side": "SELL_FIXED", "delivery_start": _month_start(1, d0).isoformat(),
         "delivery_end": _month_start(2, d0).isoformat(), "profile": "PEAK", "quantity_mw": 5.0,
         "trade_price": 90.0, "currency": "EUR", "status": "EXECUTED", "notes": "Illustrative short fixed-price peak power"},
        {"trade_id": "TR-1003", "trade_date": valuation_date.isoformat(), "counterparty": "BlueGrid Energy (Demo)",
         "product": "German Power Base", "instrument": "SWAP", "side": "BUY_FIXED", "delivery_start": _month_start(2, d0).isoformat(),
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
        "notes": "",
    }
    for col, default in optional_defaults.items():
        if col not in out.columns:
            out[col] = default
    out["product"] = out["product"].astype(str).str.strip()
    out["side"] = out["side"].astype(str).str.upper().str.strip()
    out["status"] = out["status"].astype(str).str.upper().str.strip()
    out["currency"] = out["currency"].astype(str).str.upper().str.strip()
    out["profile"] = out["profile"].astype(str).str.upper().str.strip()
    out["trade_id"] = out["trade_id"].astype(str).str.strip()
    if out["trade_id"].eq("").any() or out["trade_id"].astype(str).duplicated().any():
        raise ValueError("Trade IDs must be non-empty and unique.")
    if not out["product"].isin(PRODUCTS).all():
        bad = out.loc[~out["product"].isin(PRODUCTS), "product"].unique().tolist()
        raise ValueError(f"Unsupported product(s): {bad}")
    for _, row in out.iterrows():
        expected_currency = PRODUCTS[row["product"]]["currency"]
        if row["currency"] != expected_currency:
            raise ValueError(f"Currency mismatch for {row['product']}: expected {expected_currency}, got {row['currency']}.")
    out["delivery_start"] = pd.to_datetime(out["delivery_start"], errors="raise").dt.date
    out["delivery_end"] = pd.to_datetime(out["delivery_end"], errors="raise").dt.date
    if any(e <= s for s, e in zip(out["delivery_start"], out["delivery_end"])):
        raise ValueError("Every delivery_end must be after delivery_start (exclusive end date).")
    out["quantity_mw"] = pd.to_numeric(out["quantity_mw"], errors="raise")
    out["trade_price"] = pd.to_numeric(out["trade_price"], errors="raise")
    if not out["quantity_mw"].map(math.isfinite).all() or (out["quantity_mw"] < 0).any():
        raise ValueError("quantity_mw must be finite and non-negative; direction belongs in side.")
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
    out["quantity_mwh"] = [mw_to_mwh(float(mw), delivery_hours(s, e, p)) for mw, s, e, p in zip(out["quantity_mw"], out["delivery_start"], out["delivery_end"], out["profile"])]
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
        market_price = swap_fixed_price(curve, trade["delivery_start"], trade["delivery_end"], trade["profile"])
        signed_q = float(trade["signed_quantity_mwh"])
        mtm = mtm_linear(signed_q, float(trade["trade_price"]), market_price)
        rows.append({**trade.to_dict(), "market_price": market_price, "mtm": mtm,
                     "mtm_unit": f"{trade['currency']}", "mark_status": "ILLUSTRATIVE CURVE"})
    return pd.DataFrame(rows)


def position_summary(marked_trades: pd.DataFrame) -> pd.DataFrame:
    if marked_trades.empty:
        return pd.DataFrame(columns=["product", "signed_mwh", "mtm", "currency"])
    return (marked_trades.loc[marked_trades["status"].eq("EXECUTED")]
            .groupby(["product", "currency"], as_index=False)
            .agg(signed_mwh=("signed_quantity_mwh", "sum"), mtm=("mtm", "sum")))


def _check_sign_convention() -> None:
    # Runtime sanity checks for the most important financial sign conventions.
    assert mtm_linear(100.0, 50.0, 55.0) == 500.0
    assert mtm_linear(-100.0, 50.0, 55.0) == -500.0
    assert mtm_linear(100.0, 55.0, 50.0) == -500.0


_check_sign_convention()



# -----------------------------------------------------------------------------
# Historical backtesting helpers — work with daily settlement/mark CSV imports.
# ----------------------------------------------------------------------------
BACKTEST_STRATEGIES = ["SMA Crossover", "Time-Series Momentum", "Mean Reversion (Z-Score)", "Buy & Hold"]


def normalise_price_history(raw: pd.DataFrame) -> pd.Series:
    """Read a daily history with Date/Price (or Date/Close/Settle) columns.

    Actual historical observations must be supplied by the user. No synthetic price
    history is substituted if a file is absent. Negative power prices are supported
    because the P&L engine uses price differences, not percentage returns.
    """
    if raw is None or raw.empty:
        raise ValueError("The uploaded price history is empty.")
    cols = {str(c).strip().lower(): c for c in raw.columns}
    date_col = next((cols[c] for c in ("date", "datetime", "timestamp", "time") if c in cols), None)
    price_col = next((cols[c] for c in ("price", "close", "settle", "settlement", "last") if c in cols), None)
    if date_col is None or price_col is None:
        raise ValueError("CSV must contain a date column and a price/close/settle column.")
    dt = pd.to_datetime(raw[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    pxs = pd.to_numeric(raw[price_col], errors="coerce")
    out = pd.DataFrame({"date": dt, "price": pxs}).dropna()
    if len(out) != len(raw):
        raise ValueError("CSV contains invalid or missing dates/prices; correct those rows before backtesting.")
    if not np.isfinite(out["price"].to_numpy(dtype=float)).all():
        raise ValueError("Historical prices must all be finite.")
    if out["date"].duplicated().any():
        raise ValueError("Historical price CSV has duplicate timestamps.")
    if out["date"].dt.normalize().duplicated().any():
        raise ValueError("Use at most one settlement/mark per calendar date. Aggregate intraday data to a documented daily close or settlement before backtesting.")
    out = out.sort_values("date")
    if len(out) < 40:
        raise ValueError("At least 40 historical daily observations are required.")
    return pd.Series(out["price"].to_numpy(dtype=float), index=pd.DatetimeIndex(out["date"]), name="price")


def _energy_backtest_metrics(daily: pd.DataFrame, pnl_col: str = "net_pnl",
                             capital: float = 100_000.0) -> dict:
    """P&L-based metrics appropriate for energy prices that can be negative.

    Position quantity is fixed in MWh-equivalent. Total return is cumulative P&L divided
    by stated capital; annualised return is simple/linear rather than a compounded CAGR.
    """
    if daily.empty:
        return {"Observations": 0, "Net P&L": 0.0, "Total Return %": 0.0,
                "Annualised Return %": np.nan, "Annualised Vol %": np.nan,
                "Sharpe (rf=0)": np.nan, "Max Drawdown %": np.nan,
                "Hit Rate %": np.nan, "Turnover (MWh)": 0.0, "Trading Costs": 0.0}
    pnl = pd.to_numeric(daily[pnl_col], errors="coerce").fillna(0.0)
    valid = daily["price_change"].notna() if "price_change" in daily else pd.Series(True, index=daily.index)
    r = pnl / float(capital)
    r_valid = r[valid]
    n = int(valid.sum())
    total_pnl = float(pnl.sum())
    total_return = total_pnl / capital * 100.0
    ann_return = total_pnl / capital * 252.0 / n * 100.0 if n > 0 else np.nan
    ann_vol = float(r_valid.std(ddof=1) * math.sqrt(252) * 100) if len(r_valid) > 1 else np.nan
    std = float(r_valid.std(ddof=1)) if len(r_valid) > 1 else np.nan
    sharpe = float(r_valid.mean() / std * math.sqrt(252)) if std and np.isfinite(std) and std > 0 else np.nan
    equity = float(capital) + pnl.cumsum()
    peak = equity.cummax()
    dd = equity / peak.where(peak.abs() > 1e-12) - 1.0
    hit_rate = float((pnl[valid] > 0).mean() * 100) if n else np.nan
    turnover = float(daily.get("turnover_mwh", pd.Series(0.0, index=daily.index)).sum())
    costs = float(daily.get("trading_cost", pd.Series(0.0, index=daily.index)).sum())
    return {"Observations": n, "Net P&L": total_pnl, "Total Return %": total_return,
            "Annualised Return %": ann_return, "Annualised Vol %": ann_vol,
            "Sharpe (rf=0)": sharpe,
            "Max Drawdown %": float(dd.min() * 100) if not dd.empty else np.nan,
            "Hit Rate %": hit_rate, "Turnover (MWh)": turnover,
            "Trading Costs": costs}


def energy_backtest(prices: pd.Series, strategy: str = "SMA Crossover", *,
                    fast_window: int = 20, slow_window: int = 100,
                    momentum_window: int = 60, z_window: int = 60,
                    z_entry: float = 1.5, volume_mwh: float = 1000.0,
                    capital: float = 100_000.0,
                    transaction_cost_per_mwh: float = 0.05,
                    long_only: bool = False) -> dict:
    """Backtest a causal directional rule on a user-provided energy price history.

    P&L = prior-bar position × daily price change × fixed MWh exposure. This handles
    negative electricity prices. Costs are charged on absolute position turnover in
    quote-currency/MWh. Signals are shifted by one observation (no look-ahead).
    """
    if strategy not in BACKTEST_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")
    s = prices.copy() if isinstance(prices, pd.Series) else pd.Series(prices)
    s = pd.to_numeric(s, errors="coerce").dropna().astype(float).sort_index()
    if len(s) < 40 or not np.isfinite(s.to_numpy()).all():
        raise ValueError("Provide at least 40 finite price observations.")
    if not s.index.is_unique:
        raise ValueError("Price history must have unique timestamps.")
    if isinstance(s.index, pd.DatetimeIndex) and s.index.normalize().duplicated().any():
        raise ValueError("The backtest expects at most one daily settlement/mark per calendar date.")
    if not (5 <= int(fast_window) < int(slow_window) <= 500):
        raise ValueError("SMA windows must satisfy 5 <= fast < slow <= 500.")
    if not (5 <= int(momentum_window) <= 500 and 10 <= int(z_window) <= 500):
        raise ValueError("Momentum and z-score windows must be between 5/10 and 500 observations.")
    if strategy == "SMA Crossover" and len(s) <= int(slow_window):
        raise ValueError(f"SMA Crossover needs more observations than the slow window ({slow_window}); upload a longer history or reduce the SMA windows.")
    if strategy == "Time-Series Momentum" and len(s) <= int(momentum_window):
        raise ValueError(f"Time-Series Momentum needs more observations than its lookback ({momentum_window}).")
    if strategy == "Mean Reversion (Z-Score)" and len(s) <= int(z_window):
        raise ValueError(f"Mean Reversion needs more observations than its z-score window ({z_window}).")
    if not (math.isfinite(float(z_entry)) and 0.25 <= float(z_entry) <= 5.0):
        raise ValueError("Z-score threshold must be between 0.25 and 5.0.")
    if not (math.isfinite(float(volume_mwh)) and volume_mwh > 0):
        raise ValueError("MWh exposure must be positive and finite.")
    if not (math.isfinite(float(capital)) and capital > 0):
        raise ValueError("Starting capital must be positive and finite.")
    if not (math.isfinite(float(transaction_cost_per_mwh)) and transaction_cost_per_mwh >= 0):
        raise ValueError("Transaction cost per MWh must be non-negative and finite.")

    fast = s.rolling(int(fast_window), min_periods=int(fast_window)).mean()
    slow = s.rolling(int(slow_window), min_periods=int(slow_window)).mean()
    if strategy == "SMA Crossover":
        signal = pd.Series(np.where(fast > slow, 1.0, np.where(fast < slow, -1.0, 0.0)), index=s.index)
        signal[fast.isna() | slow.isna()] = 0.0
    elif strategy == "Time-Series Momentum":
        signal = np.sign(s.diff(int(momentum_window))).fillna(0.0).astype(float)
    elif strategy == "Mean Reversion (Z-Score)":
        roll_mean = s.rolling(int(z_window), min_periods=int(z_window)).mean()
        roll_std = s.rolling(int(z_window), min_periods=int(z_window)).std(ddof=1).replace(0, np.nan)
        z = (s - roll_mean) / roll_std
        signal = pd.Series(np.where(z > z_entry, -1.0, np.where(z < -z_entry, 1.0, 0.0)), index=s.index)
        signal[z.isna()] = 0.0
    else:
        signal = pd.Series(1.0, index=s.index)
    if long_only and strategy != "Buy & Hold":
        signal = signal.clip(lower=0.0, upper=1.0)
    position = signal.shift(1).fillna(0.0)
    price_change = s.diff()
    turnover = position.diff().abs().fillna(position.abs())
    turnover_mwh = turnover * float(volume_mwh)
    costs = turnover_mwh * float(transaction_cost_per_mwh)
    gross_pnl = position * price_change.fillna(0.0) * float(volume_mwh)
    net_pnl = gross_pnl - costs
    equity = float(capital) + net_pnl.cumsum()
    drawdown = equity / equity.cummax().where(equity.cummax().abs() > 1e-12) - 1.0

    bh_position = pd.Series(1.0, index=s.index).shift(1).fillna(0.0)
    bh_turnover = bh_position.diff().abs().fillna(bh_position.abs()) * float(volume_mwh)
    bh_cost = bh_turnover * float(transaction_cost_per_mwh)
    bh_gross = bh_position * price_change.fillna(0.0) * float(volume_mwh)
    bh_net = bh_gross - bh_cost
    bh_equity = float(capital) + bh_net.cumsum()
    bh_dd = bh_equity / bh_equity.cummax().where(bh_equity.cummax().abs() > 1e-12) - 1.0
    daily = pd.DataFrame({
        "price": s, "signal_at_close": signal, "position": position,
        "price_change": price_change, "turnover_mwh": turnover_mwh,
        "gross_pnl": gross_pnl, "trading_cost": costs, "net_pnl": net_pnl,
        "equity": equity, "drawdown": drawdown,
        "benchmark_position": bh_position, "benchmark_cost": bh_cost,
        "benchmark_net_pnl": bh_net, "benchmark_equity": bh_equity,
        "benchmark_drawdown": bh_dd,
    })
    warnings = [
        "Use a consistent daily settlement or same-contract mark history. Unadjusted continuous front-month roll gaps can distort results.",
        "PnL is linear price-change PnL on a fixed MWh exposure. Contract-specific settlement, shaping, collateral, funding, margin calls, bid/ask, exchange fees, market impact and FX are not modelled.",
        "The holdout is chronological with fixed parameters, not an optimised walk-forward test. Avoid tuning parameters against the out-of-sample segment.",
    ]
    if equity.iloc[-1] <= 0 or bh_equity.iloc[-1] <= 0:
        warnings.append("A tested equity curve falls to zero or below. This is a warning condition; futures losses and margin requirements are not capped or simulated.")
    benchmark_daily = pd.DataFrame({
        "net_pnl": bh_net,
        "price_change": price_change,
        "turnover_mwh": bh_turnover,
        "trading_cost": bh_cost,
    }, index=s.index)
    return {"daily": daily, "metrics": _energy_backtest_metrics(daily, "net_pnl", capital),
            "benchmark_metrics": _energy_backtest_metrics(benchmark_daily, "net_pnl", capital),
            "warnings": warnings}



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
# Streamlit user interface — S&D Desk visual language retained
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
    "Analytics": ["📉  Backtesting Lab"],
    "Reference": ["🔧  Data & Assumptions"],
}
ALL_NAV_ITEMS = [label for labels in NAV_SECTIONS.values() for label in labels]
DEFAULT_NAV_ITEM = ALL_NAV_ITEMS[0]
NAV_TO_PAGE = {label: label.split("  ", 1)[1] for label in ALL_NAV_ITEMS}

PAGE_HELP = {
    "European Energy Trading Desk": "European gas and power OTC quote support, trade capture, mark-to-market and delivery monitoring. Built-in prices and trades are illustrative.",
    "OTC Quote Builder": "Indicative price = reference forward plus/minus commercial margin. The quote is not executable and must be checked against approved market data, credit limits and current conventions.",
    "Trading Book & Mark-to-Market": "MtM = signed delivery MWh × (market forward − contracted fixed price). BUY_FIXED is long fixed price; SELL_FIXED is short fixed price. EUR and GBP are reported separately.",
    "Forward Curves": "Monthly forward marks are illustrative unless replaced by validated user data. Missing delivery tenors are rejected rather than silently interpolated.",
    "P&L Sensitivity & Exposure": "A parallel curve shock estimates a simplified change in MtM. It is not VaR, realised daily P&L or a full risk measure.",
    "Physical Operations & Delivery Tracking": "Operations records are illustrative. Nominations, actual allocated energy, imbalances and invoices require a connected operational data source.",
    "European Energy Fundamentals": "Charts are illustrative until connected to timestamped, licensed market data for European gas and power fundamentals.",
    "Historical Strategy Backtesting": "Backtests run on user-provided historical marks and price changes, with one-bar-lagged signals and explicit transaction costs. They are research tools, not live-trading evidence.",
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
          <span style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;color:{AMBER}">S&D DESK</span><br>
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
            st.caption("The Backtesting Lab requires user-provided historical marks. It never synthesises a backtest history.")
        st.markdown("---")
        st.markdown(f'<div style="font-size:0.62rem;color:{GRAY};line-height:1.5">Single-file educational ETRM prototype<br>European gas & power · not for trade execution</div>', unsafe_allow_html=True)
    return NAV_TO_PAGE[st.session_state.nav_page]


# Page setup must precede all other Streamlit commands.
st.set_page_config(page_title="S&D — European Energy Trading Desk", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(_CSS, unsafe_allow_html=True)

if "trades" not in st.session_state:
    st.session_state.trades = build_sample_trades()
if "custom_curves" not in st.session_state:
    st.session_state.custom_curves = {}

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
    active = st.session_state.trades.copy()
    active = active.loc[~active["status"].isin(["CANCELLED", "EXPIRED"])].copy()
    if active.empty:
        return pd.DataFrame()
    return mark_trades(active, curves)


page = render_sidebar()

if page == "Market Overview":
    title_block("European Energy Trading Desk", "A compact ETRM prototype focused on European gas, power, OTC pricing and risk.")
    marked = get_marked_trades()
    executed = marked.loc[marked["status"].eq("EXECUTED")] if not marked.empty else pd.DataFrame()
    total_mtm = float(executed["mtm"].sum()) if not executed.empty else 0.0
    total_trades = int(len(executed))
    total_products = int(executed["product"].nunique()) if not executed.empty else 0
    eur_mtm = float(executed.loc[executed["currency"].eq("EUR"), "mtm"].sum()) if not executed.empty else 0.0
    gbp_mtm = float(executed.loc[executed["currency"].eq("GBP"), "mtm"].sum()) if not executed.empty else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Executed trades", f"{total_trades}")
    c2.metric("Products traded", f"{total_products}")
    c3.metric("Illustrative MtM · EUR", money(eur_mtm, "EUR"))
    c4.metric("Illustrative MtM · GBP", money(gbp_mtm, "GBP"))
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
        instrument = b.selectbox("Instrument", ["SWAP", "FORWARD"])
        profile = c.selectbox("Delivery profile", ["BASE", "PEAK"], disabled=PRODUCTS[product]["commodity"] == "Gas")
        if PRODUCTS[product]["commodity"] == "Gas":
            profile = "BASE"
        a, b, c = st.columns(3)
        start = a.date_input("Delivery start", value=month_start(1))
        months = b.selectbox("Delivery tenor", [1, 3, 6, 12], index=1, format_func=lambda x: f"{x} month(s)")
        end = month_end_exclusive(start, months)
        mw = c.number_input("Contract volume (MW)", min_value=0.1, max_value=100000.0, value=10.0, step=0.5)
        a, b, c = st.columns(3)
        side = a.selectbox("Desk quote side", ["SELL_TO_CLIENT", "BUY_FROM_CLIENT"], format_func=lambda x: "Sell to client" if x == "SELL_TO_CLIENT" else "Buy from client")
        margin = b.number_input("Commercial margin per MWh", min_value=0.0, max_value=100.0, value=0.35, step=0.05)
        counterparty = c.text_input("Counterparty", value="Prospective client (Demo)")
        submitted = st.form_submit_button("Calculate indicative quote", type="primary")
    if submitted:
        try:
            if start.day != 1:
                st.warning("The selected start date is not the first of a month. This quote includes a partial first delivery month; monthly curve marks are used for each touched month.")
            ref = reference_mark(product, start, end, profile)
            quoted = quote_price(ref, margin, side)
            hours = delivery_hours(start, end, profile)
            total_mwh = mw * hours
            st.session_state.last_quote = {"product": product, "instrument": instrument, "profile": profile,
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
        c3.metric("Delivery hours", f"{delivery_hours(q['start'], q['end'], q['profile']):,}")
        c4.metric("Contract energy", f"{q['mwh']:,.0f} MWh")
        st.write(f"**Counterparty:** {q['counterparty']}  ·  **Delivery:** {q['start']} to {q['end']} (end date exclusive)  ·  **Status:** Indicative only")
        st.caption("Quote convention: sell-to-client = reference + margin; buy-from-client = reference − margin. This is a configurable illustrative convention, not a universal market convention.")
        if st.button("Add quote to blotter as PROPOSED"):
            new_id = f"Q-{pd.Timestamp.now().strftime('%y%m%d%H%M%S')}"
            new = pd.DataFrame([{"trade_id": new_id, "trade_date": valuation_date.isoformat(), "counterparty": q["counterparty"] or "Unspecified",
                "product": q["product"], "instrument": q["instrument"],
                "side": "BUY_FIXED" if q["side"] == "BUY_FROM_CLIENT" else "SELL_FIXED",
                "delivery_start": q["start"].isoformat(), "delivery_end": q["end"].isoformat(),
                "profile": q["profile"], "quantity_mw": q["mw"], "trade_price": q["quote"],
                "currency": q["currency"], "status": "PROPOSED", "notes": "Generated by OTC Quote Builder; indicative quote"}])
            try:
                new = normalise_trades(new)
                st.session_state.trades = pd.concat([st.session_state.trades, new], ignore_index=True)
                st.success(f"Quote {new_id} added to the trade book as PROPOSED. It is excluded from executed-book position totals.")
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
            start = c1.date_input("Delivery start", value=month_start(1), key="add_start")
            tenor = c2.selectbox("Tenor", [1, 3, 6, 12], index=1, key="add_tenor")
            mw = c3.number_input("Quantity (MW)", min_value=0.1, max_value=100000.0, value=5.0, step=0.5, key="add_mw")
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
                "product": product, "instrument": "SWAP", "side": side, "delivery_start": start.isoformat(),
                "delivery_end": end.isoformat(), "profile": profile, "quantity_mw": mw, "trade_price": price,
                "currency": PRODUCTS[product]["currency"], "status": status, "notes": "Manually entered demo trade"}])
            try:
                row = normalise_trades(row)
                if row["trade_id"].iloc[0] in st.session_state.trades["trade_id"].astype(str).values:
                    raise ValueError("Trade ID already exists.")
                st.session_state.trades = pd.concat([st.session_state.trades, row], ignore_index=True)
                st.success("Trade added.")
                st.rerun()
            except Exception as exc:
                st.error(f"Trade rejected: {exc}")
    try:
        marked = get_marked_trades()
        if marked.empty:
            st.info("The trade book is empty.")
        else:
            executed = marked.loc[marked["status"].eq("EXECUTED")].copy()
            c1, c2, c3 = st.columns(3)
            c1.metric("Trades in book", len(marked))
            c2.metric("Executed trades", len(executed))
            eur_mtm = float(executed.loc[executed["currency"].eq("EUR"), "mtm"].sum()) if not executed.empty else 0.0
            gbp_mtm = float(executed.loc[executed["currency"].eq("GBP"), "mtm"].sum()) if not executed.empty else 0.0
            c3.metric("Executed MtM · EUR", money(eur_mtm, "EUR"))
            st.caption(f"Executed MtM by quote currency: EUR {eur_mtm:,.2f} · GBP {gbp_mtm:,.2f}. Do not add these figures without FX conversion.")
            display_cols = ["trade_id", "counterparty", "product", "instrument", "side", "delivery_start", "delivery_end", "profile", "quantity_mw", "quantity_mwh", "trade_price", "market_price", "currency", "mtm", "status", "mark_status"]
            st.dataframe(marked[display_cols].round(4), use_container_width=True, hide_index=True)
            st.subheader("Position summary by product and currency")
            summary = position_summary(executed)
            st.dataframe(summary.round(4), use_container_width=True, hide_index=True)
            st.download_button("Download marked trade book (CSV)", marked.to_csv(index=False).encode("utf-8"), "marked_trade_book.csv", "text/csv")
        with st.expander("Import trade book CSV"):
            st.caption("CSV must contain the trade columns documented in README.md. Existing trade IDs must remain unique after import.")
            uploaded = st.file_uploader("Upload trades CSV", type=["csv"], key="trade_upload")
            if uploaded is not None and st.button("Validate and replace trade book"):
                try:
                    imported = normalise_trades(pd.read_csv(uploaded))
                    st.session_state.trades = imported
                    st.success(f"Imported {len(imported)} validated trades.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import rejected: {exc}")
    except Exception as exc:
        st.error(f"Marking failed safely: {exc}")
        st.caption("No fallback marks are invented. Check curve coverage, product, delivery dates and price units.")

elif page == "P&L & Exposure":
    title_block("P&L Sensitivity & Exposure", "Scenario-based sensitivity of the executed book to parallel forward-price moves. This is not historical daily P&L.")
    try:
        marked = get_marked_trades()
        executed = marked.loc[marked["status"].eq("EXECUTED")].copy() if not marked.empty else pd.DataFrame()
        if executed.empty:
            st.info("No executed trades to stress.")
        else:
            shock_pct = st.slider("Parallel market-price shock", min_value=-30, max_value=30, value=5, step=1, format="%d%%")
            rows = []
            for _, tr in executed.iterrows():
                shocked_price = float(tr["market_price"]) * (1 + shock_pct / 100.0)
                delta_mtm = float(tr["signed_quantity_mwh"]) * (shocked_price - float(tr["market_price"]))
                rows.append({"trade_id": tr["trade_id"], "product": tr["product"], "currency": tr["currency"],
                    "signed_quantity_mwh": tr["signed_quantity_mwh"], "current_market_price": tr["market_price"],
                    "shocked_market_price": shocked_price, "current_mtm": tr["mtm"], "scenario_pnl_change": delta_mtm})
            scenario = pd.DataFrame(rows)
            c1, c2, c3 = st.columns(3)
            c1.metric("Scenario shock", f"{shock_pct:+d}%")
            eur_change = float(scenario.loc[scenario["currency"].eq("EUR"), "scenario_pnl_change"].sum())
            gbp_change = float(scenario.loc[scenario["currency"].eq("GBP"), "scenario_pnl_change"].sum())
            c2.metric("Scenario P&L change · EUR", money(eur_change, "EUR"))
            c3.metric("Scenario P&L change · GBP", money(gbp_change, "GBP"))
            st.dataframe(scenario.round(4), use_container_width=True, hide_index=True)
            by_product = scenario.groupby(["product", "currency"], as_index=False)["scenario_pnl_change"].sum()
            fig = px.bar(by_product, x="product", y="scenario_pnl_change", color="currency", title="Scenario P&L change by product")
            fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, yaxis_title="P&L change in each product currency", xaxis_title="")
            st.plotly_chart(_styled(fig), use_container_width=True)
            st.warning("This is a parallel percentage shock on each trade's current mark. It ignores curve-shape changes, basis risk, liquidity, optionality, discounting, FX movements and physical optionality. It is not VaR and not historical daily P&L.")
    except Exception as exc:
        st.error(f"Risk scenario could not be calculated: {exc}")
        st.caption("No missing marks are filled in. Review curve coverage and product/unit consistency.")

elif page == "Physical Operations":
    title_block("Physical Operations & Delivery Tracking", "A lightweight cargo/contract operations view linked conceptually to the trading book.")
    ops = pd.DataFrame([
        {"Contract ID": "PHY-2001", "Product": "TTF Natural Gas", "Counterparty": "Northstar Utilities (Demo)", "Delivery start": month_start(1), "Delivery end (exclusive)": month_end_exclusive(month_start(1), 1), "Contracted volume (MWh)": 7200, "Delivered volume (MWh)": 0, "Status": "Scheduled", "Risk flag": "Monitor nominations"},
        {"Contract ID": "PHY-2002", "Product": "French Power Base", "Counterparty": "Alpine Power (Demo)", "Delivery start": month_start(1), "Delivery end (exclusive)": month_end_exclusive(month_start(1), 1), "Contracted volume (MWh)": 3600, "Delivered volume (MWh)": 0, "Status": "Scheduled", "Risk flag": "Check settlement calendar"},
        {"Contract ID": "PHY-2003", "Product": "German Power Base", "Counterparty": "BlueGrid Energy (Demo)", "Delivery start": month_start(2), "Delivery end (exclusive)": month_end_exclusive(month_start(2), 1), "Contracted volume (MWh)": 5760, "Delivered volume (MWh)": 0, "Status": "Confirmed", "Risk flag": "No current exception"},
    ])
    ops["Outstanding volume (MWh)"] = ops["Contracted volume (MWh)"] - ops["Delivered volume (MWh)"]
    st.info("These are illustrative operational records, not linked to a live logistics or nomination system.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Open contracts", int((ops["Status"] != "Delivered").sum()))
    c2.metric("Outstanding volume", f"{ops['Outstanding volume (MWh)'].sum():,.0f} MWh")
    c3.metric("Exceptions", int(ops["Risk flag"].str.contains("Monitor|Check", case=False).sum()))
    st.dataframe(ops, use_container_width=True, hide_index=True)
    st.subheader("Update a demo delivery")
    selected_contract = st.selectbox("Contract", ops["Contract ID"])
    selected_row = ops.loc[ops["Contract ID"].eq(selected_contract)].iloc[0]
    delivered = st.number_input("Delivered volume (MWh)", min_value=0.0, max_value=float(selected_row["Contracted volume (MWh)"]), value=float(selected_row["Delivered volume (MWh)"]), step=100.0)
    if st.button("Preview delivery reconciliation"):
        outstanding = float(selected_row["Contracted volume (MWh)"]) - delivered
        st.metric("Outstanding after update", f"{outstanding:,.0f} MWh")
        if outstanding < 0:
            st.error("Delivered volume exceeds contracted volume.")
        elif outstanding == 0:
            st.success("Volume fully reconciled. This preview does not persist changes to the demo operations table.")
        else:
            st.warning("Partial delivery remains outstanding. This preview does not persist changes to the demo operations table.")

elif page == "Backtesting Lab":
    title_block("Historical Strategy Backtesting", "Causal energy-price signals · P&L on fixed MWh exposure · explicit transaction costs")
    st.warning("Upload a real daily settlement or same-contract forward-mark history. No synthetic price history is created or used. For a valid test, use a consistent instrument / roll methodology.")
    u1, u2 = st.columns([1.8, 1])
    with u1:
        history_file = st.file_uploader("Historical price CSV", type=["csv"], key="bt_price_upload",
                                        help="Required columns: date and price (or close / settle). One row per date or timestamp.")
    with u2:
        st.download_button("Download CSV header template", data="date,price\n", file_name="energy_price_history_template.csv", mime="text/csv")
    if history_file is None:
        st.info("CSV schema: **date, price**. Use one product and one consistent price series; do not combine TTF, NBP or different delivery tenors in the same column.")
    else:
        try:
            history = normalise_price_history(pd.read_csv(history_file))
            chosen_product = st.selectbox("Product / quote currency", list(PRODUCTS.keys()), key="bt_product")
            currency = PRODUCTS[chosen_product]["currency"]
            unit = PRODUCTS[chosen_product]["unit"]
            st.caption(f"Loaded {len(history):,} observations · {history.index.min().date()} to {history.index.max().date()} · expected P&L currency: {currency}. Confirm that this file belongs to {chosen_product} and is quoted in {unit}. Strategies require more observations than their respective lookback windows.")
            a, b, c, d, e = st.columns(5)
            strategy = a.selectbox("Strategy", BACKTEST_STRATEGIES, key="bt_strategy")
            fast = b.slider("Fast SMA", 5, 100, 20, 1, key="bt_fast")
            slow = c.slider("Slow SMA", 20, 250, 100, 5, key="bt_slow")
            momentum = d.slider("Momentum window", 5, 252, 60, 5, key="bt_momentum")
            zwin = e.slider("Z-score window", 10, 252, 60, 5, key="bt_zwin")
            a, b, c, d = st.columns(4)
            zentry = a.slider("Mean-reversion entry (σ)", 0.5, 3.0, 1.5, 0.1, key="bt_zentry")
            volume = b.number_input("Exposure (MWh-equivalent)", min_value=1.0, max_value=10_000_000.0, value=1000.0, step=100.0, key="bt_volume")
            capital = c.number_input(f"Starting capital ({currency})", min_value=1000.0, max_value=100_000_000.0, value=100_000.0, step=5000.0, key="bt_capital")
            cost = d.number_input(f"Trading cost ({currency}/MWh per turnover unit)", min_value=0.0, max_value=100.0, value=0.05, step=0.01, key="bt_cost")
            long_only = st.checkbox("Long-only mode (do not open short positions)", value=False, key="bt_long_only")
            if strategy == "SMA Crossover" and fast >= slow:
                st.warning("For SMA Crossover, the fast SMA must be strictly shorter than the slow SMA. Other selected strategies do not use these two windows.")
            else:
                result = energy_backtest(history, strategy, fast_window=fast, slow_window=slow,
                                         momentum_window=momentum, z_window=zwin, z_entry=zentry,
                                         volume_mwh=volume, capital=capital,
                                         transaction_cost_per_mwh=cost, long_only=long_only)
                daily = result["daily"]
                m = result["metrics"]
                bm = result["benchmark_metrics"]
                st.markdown("### Strategy performance")
                km = st.columns(5)
                kpi(km[0], "Net P&L", money(m["Net P&L"], currency), "after turnover costs", GREEN if m["Net P&L"] >= 0 else RED)
                kpi(km[1], "Total return", f"{m['Total Return %']:+.2f}%", "vs stated starting capital", GREEN if m["Total Return %"] >= 0 else RED)
                kpi(km[2], "Annualised P&L return", f"{m['Annualised Return %']:+.2f}%" if np.isfinite(m["Annualised Return %"]) else "n/a", "linear annualisation")
                kpi(km[3], "Sharpe ratio", f"{m['Sharpe (rf=0)']:.2f}" if np.isfinite(m["Sharpe (rf=0)"]) else "n/a", "risk-free rate assumed 0")
                kpi(km[4], "Max drawdown", f"{m['Max Drawdown %']:.2f}%" if np.isfinite(m["Max Drawdown %"]) else "n/a", "peak-to-trough equity", RED)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily.index, y=daily["equity"], name=strategy, line=dict(color=AMBER, width=2)))
                fig.add_trace(go.Scatter(x=daily.index, y=daily["benchmark_equity"], name="Buy & Hold benchmark", line=dict(color=BLUE, width=1.5)))
                fig.add_hline(y=capital, line=dict(color=GRAY, dash="dot", width=1), annotation_text="Starting capital")
                fig.update_layout(title=f"Equity / cumulative P&L — {chosen_product} ({currency})", xaxis_title="Date", yaxis_title=f"Equity ({currency})")
                st.plotly_chart(_styled(fig, 420), use_container_width=True)
                dd_fig = go.Figure()
                dd_fig.add_trace(go.Scatter(x=daily.index, y=daily["drawdown"] * 100, name="Strategy drawdown", line=dict(color=RED, width=1.3), fill="tozeroy", fillcolor="rgba(255,123,114,0.12)"))
                dd_fig.update_layout(title="Strategy drawdown", xaxis_title="Date", yaxis_title="Drawdown (%)")
                st.plotly_chart(_styled(dd_fig, 260), use_container_width=True)

                st.markdown("### Strategy comparison")
                comp_rows = []
                for candidate in BACKTEST_STRATEGIES:
                    try:
                        r = energy_backtest(history, candidate, fast_window=fast, slow_window=slow,
                                            momentum_window=momentum, z_window=zwin, z_entry=zentry,
                                            volume_mwh=volume, capital=capital,
                                            transaction_cost_per_mwh=cost, long_only=long_only)
                        comp_rows.append({"Strategy": candidate, "Status": "OK", **r["metrics"]})
                    except ValueError as exc:
                        comp_rows.append({"Strategy": candidate, "Status": f"Not run: {exc}"})
                st.dataframe(pd.DataFrame(comp_rows).round(3), use_container_width=True, hide_index=True)

                st.markdown("### Transaction-cost sensitivity")
                sens = []
                for trial_cost in [0.0, 0.02, 0.05, 0.10, 0.25]:
                    r = energy_backtest(history, strategy, fast_window=fast, slow_window=slow,
                                        momentum_window=momentum, z_window=zwin, z_entry=zentry,
                                        volume_mwh=volume, capital=capital,
                                        transaction_cost_per_mwh=trial_cost, long_only=long_only)
                    sens.append({"Cost / MWh": trial_cost, "Net P&L": r["metrics"]["Net P&L"],
                                 "Total Return %": r["metrics"]["Total Return %"],
                                 "Sharpe (rf=0)": r["metrics"]["Sharpe (rf=0)"],
                                 "Max Drawdown %": r["metrics"]["Max Drawdown %"],
                                 "Cost Paid": r["metrics"]["Trading Costs"]})
                st.dataframe(pd.DataFrame(sens).round(3), use_container_width=True, hide_index=True)

                st.markdown("### Chronological 70/30 holdout")
                split = int(len(daily) * 0.70)
                if split >= 30 and len(daily) - split >= 20:
                    out_rows = []
                    for label, part in [("In-sample (first 70%)", daily.iloc[:split]), ("Out-of-sample (last 30%)", daily.iloc[split:])]:
                        out_rows.append({"Period": label, "From": part.index.min().date().isoformat(),
                                         "To": part.index.max().date().isoformat(),
                                         **_energy_backtest_metrics(part, "net_pnl", capital)})
                    st.dataframe(pd.DataFrame(out_rows).round(3), use_container_width=True, hide_index=True)
                    st.caption(f"Chronological split date: {daily.index[split].date().isoformat()}. Parameters remain fixed; do not tune them against the holdout.")
                else:
                    st.info("There are not enough observations for a 70/30 holdout.")
                st.markdown("### Daily backtest ledger")
                st.dataframe(daily.tail(250).round(5), use_container_width=True, height=320)
                st.download_button("⬇️ Download full daily ledger (CSV)", data=daily.to_csv(index_label="Date").encode("utf-8"), file_name=f"energy_backtest_{chosen_product.lower().replace(' ', '_')}_{strategy.lower().replace(' ', '_')}.csv", mime="text/csv")
                with st.expander("Methodology and limits", expanded=True):
                    for warning in result["warnings"]:
                        st.warning(warning)
                    st.markdown("Signals are computed at close *t* and positions apply from the next observation. P&L uses price differences so negative power prices are supported. Transaction costs are applied to absolute exposure changes; a reversal from -1 to +1 has turnover 2 times the MWh exposure. Annualised statistics assume 252 observations per year and a zero risk-free rate; use daily settlement/mark data, not unaggregated intraday observations.")
        except Exception as exc:
            st.error(f"Backtest input rejected: {exc}")
            st.caption("Check that the CSV contains unique timestamps and finite prices, and that the selected parameters are valid.")

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
        st.success("Demo book and custom curves reset.")
        st.rerun()


