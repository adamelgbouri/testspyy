"""IO helpers - CSV/Excel export, parameter save/load."""

from __future__ import annotations

import io
import json
from typing import Any, Dict

import pandas as pd


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to CSV bytes (for st.download_button)."""
    return df.to_csv(index=True).encode("utf-8")


def df_to_excel_bytes(frames: Dict[str, pd.DataFrame]) -> bytes:
    """Write one or more DataFrames to a single in-memory Excel workbook."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, df in frames.items():
            # Excel sheet names cap at 31 chars
            df.to_excel(writer, sheet_name=sheet[:31])
    return buf.getvalue()


def params_to_json(params: Dict[str, Any]) -> str:
    """Serialise a parameter dict to a pretty JSON string."""
    return json.dumps(params, indent=2, default=str)


def params_from_json(blob: str) -> Dict[str, Any]:
    """Deserialise a JSON parameter blob (raises ValueError on failure)."""
    try:
        return json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid parameter JSON: {exc}") from exc
