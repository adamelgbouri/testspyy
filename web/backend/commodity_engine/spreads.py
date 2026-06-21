"""Crack and location spread definitions + margin computation."""
from __future__ import annotations
from typing import Dict, Optional

# Each preset describes the legs and a typical margin level for context.
CRACK_SPREADS: Dict[str, Dict] = {
    "3-2-1 Crack (WTI → RBOB + ULSD)": {
        "F1": "wti_crude", "F1_mult": 3.0,
        "F2": "rbob_gasoline", "F2_mult": 2.0,
        "F3": "ulsd_heating_oil", "F3_mult": 1.0,
        "description": "3 bbl WTI → 2 bbl RBOB + 1 bbl Heating Oil. "
                       "Classic US refiner margin.",
        "typical": 22.0, "unit": "$/bbl",
    },
    "Simple Crack (WTI → RBOB)": {
        "F1": "wti_crude", "F1_mult": 1.0,
        "F2": "rbob_gasoline", "F2_mult": 1.0,
        "F3": None, "F3_mult": 0.0,
        "description": "1 bbl WTI → 1 bbl RBOB (heating oil ignored).",
        "typical": 14.0, "unit": "$/bbl",
    },
    "Location: Brent − WTI": {
        "F1": "wti_crude", "F1_mult": 1.0,
        "F2": "brent_crude", "F2_mult": 1.0,
        "F3": None, "F3_mult": 0.0,
        "description": "Atlantic basin location spread.",
        "typical": 4.0, "unit": "$/bbl",
    },
    "Gold / Silver ratio": {
        "F1": "silver", "F1_mult": 1.0,
        "F2": "gold", "F2_mult": 1.0,
        "F3": None, "F3_mult": 0.0,
        "description": "Gold/Silver ratio (precious metals relative value).",
        "typical": 85.0, "unit": "ratio",
    },
}


def crack_margin(spec: Dict, prices: Dict[str, float]) -> float:
    """Compute the crack/spread margin in product-equivalent units."""
    p1 = prices.get(spec["F1"], 0)
    p2 = prices.get(spec["F2"], 0)
    p3 = prices.get(spec["F3"], 0) if spec["F3"] else 0
    # Energy: convert $/gal product legs to $/bbl using 42 gal/bbl.
    if spec["F1"] in ("wti_crude", "brent_crude") and \
       spec["F2"] in ("rbob_gasoline", "ulsd_heating_oil"):
        p2_bbl = p2 * 42
        p3_bbl = p3 * 42 if spec["F3"] else 0
        return (spec["F2_mult"] * p2_bbl
                + spec["F3_mult"] * p3_bbl
                - spec["F1_mult"] * p1) / spec["F1_mult"]
    # Gold / Silver ratio: F2 / F1
    if spec["F1"] == "silver" and spec["F2"] == "gold":
        return p2 / p1 if p1 else 0
    # Location / generic: same-unit difference
    return (spec["F2_mult"] * p2 + spec["F3_mult"] * p3
            - spec["F1_mult"] * p1) / max(spec["F1_mult"], 1)
