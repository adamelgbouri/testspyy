"""
Regional balance module - per-region supply/demand and trade flows.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def regional_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute net trade balance and surplus/deficit signal per region."""
    out = df.copy()
    out["balance"] = out["supply"] - out["demand"]
    out["status"] = np.where(out["balance"] > 0, "Exporter",
                             np.where(out["balance"] < 0, "Importer", "Balanced"))
    return out


def build_trade_flows(df: pd.DataFrame) -> Tuple[List[str], List[int], List[int], List[float]]:
    """
    Construct simple Sankey source/target/value arrays from a regional balance.

    Net exporters supply net importers in proportion to the deficit.  The
    output suits plotly.graph_objects.Sankey.
    """
    rs = regional_summary(df)
    exporters = rs[rs["balance"] > 0].copy()
    importers = rs[rs["balance"] < 0].copy()
    if exporters.empty or importers.empty:
        return [], [], [], []

    # normalise importer demand for share allocation
    importers["deficit"] = -importers["balance"]
    importers["share"] = importers["deficit"] / importers["deficit"].sum()

    nodes = list(exporters["region"]) + list(importers["region"])
    sources: List[int] = []
    targets: List[int] = []
    values: List[float] = []

    for i, ex_row in enumerate(exporters.itertuples(index=False)):
        ex_idx = i
        ex_surplus = ex_row.balance
        for j, im_row in enumerate(importers.itertuples(index=False)):
            im_idx = len(exporters) + j
            flow = ex_surplus * im_row.share
            if flow > 0:
                sources.append(ex_idx)
                targets.append(im_idx)
                values.append(float(flow))
    return nodes, sources, targets, values


def arbitrage_signals(df: pd.DataFrame, price_diff_threshold: float = 0.0) -> pd.DataFrame:
    """
    Identify regions where supply surplus suggests export arbitrage.

    Adds an `arb_signal` column: "Export Arb", "Import Need", "Neutral".
    """
    rs = regional_summary(df)
    rs["arb_signal"] = np.where(
        rs["balance"] > price_diff_threshold,
        "Export Arb",
        np.where(rs["balance"] < -price_diff_threshold, "Import Need", "Neutral"),
    )
    return rs
