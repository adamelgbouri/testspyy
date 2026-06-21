"""Recurring high-impact event calendar for commodity desks."""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict


def get_market_events() -> List[Dict]:
    """Next ~6 weeks of EIA / WASDE / OPEC / IEA / central bank releases."""
    today = datetime.now()
    events: List[Dict] = []

    def add(label: str, date: datetime, tags: List[str], freq: str) -> None:
        if date >= today:
            events.append({
                "date": date.strftime("%Y-%m-%d"),
                "event": label, "tags": tags, "frequency": freq,
            })

    # EIA Weekly Petroleum Status — Wednesdays for 6 weeks
    d = today + timedelta(days=(2 - today.weekday()) % 7)
    for k in range(6):
        add("EIA Weekly Petroleum Status", d + timedelta(weeks=k),
            ["Energy"], "Weekly")

    # Monthly mid-month releases
    monthly_events = [
        ("USDA WASDE Report", 11, ["Ags"]),
        ("OPEC Monthly Oil Report", 12, ["Energy"]),
        ("IEA Oil Market Report", 14, ["Energy"]),
        ("US CPI Release", 13, ["Macro"]),
    ]
    for label, day, tags in monthly_events:
        for k in range(3):
            month = today.month + k
            year = today.year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            try:
                add(label, datetime(year, month, day), tags, "Monthly")
            except ValueError:
                continue

    # First Friday of month — NFP
    for k in range(3):
        month = today.month + k
        year = today.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        d = datetime(year, month, 1)
        d = d + timedelta(days=(4 - d.weekday()) % 7)
        add("US Non-Farm Payrolls", d, ["Macro"], "Monthly")

    # 6-week central bank cadence
    base = today + timedelta(days=14)
    for k in range(3):
        add("FOMC Decision", base + timedelta(weeks=6 * k), ["Macro"], "6-weekly")
        add("ECB Rate Decision",
            base + timedelta(weeks=6 * k, days=3), ["Macro"], "6-weekly")

    events.sort(key=lambda e: e["date"])
    return events[:30]
