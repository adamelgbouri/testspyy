"""
FastAPI gateway for the commodity analytics engine.

Run locally with:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from commodity_engine import (
    COMMODITY_TEMPLATES,
    BalanceAssumptions,
    Black76,
    estimate_fair_value,
    get_futures_curve,
    get_live_spot,
    get_regional_dataset,
    get_sd_dataset,
    run_balance,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(
    title="Commodity Trading Desk API",
    version="0.1.0",
    description="Analytical backend for the commodity trading platform.",
)

# Wide-open CORS in dev — restrict to your domain in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Pydantic models
# =============================================================================

class CommoditySummary(BaseModel):
    key: str
    name: str
    sector: str
    ticker: str
    price_unit: str
    unit: str
    inventory_unit: str
    base_price: float
    days_cover_target: float
    ideal_utilization_pct: float


class SpotResponse(BaseModel):
    key: str
    name: str
    price_unit: str
    price: float
    change_pct: float
    asof: str
    source: str = Field(description="'yahoo' for live, 'reference' for synthetic")


class BalancePoint(BaseModel):
    date: str
    supply: float
    demand: float
    stocks: float
    days_cover: float
    price: float
    fair_value: float
    is_forecast: bool


class BalanceResponse(BaseModel):
    key: str
    name: str
    price_unit: str
    unit: str
    inventory_unit: str
    end_stocks: float
    end_days_cover: float
    end_utilization_pct: float
    end_fair_value: float
    points: List[BalancePoint]


class RegionalRow(BaseModel):
    region: str
    supply: float
    demand: float
    net_trade: float
    supply_share_pct: float
    demand_share_pct: float
    status: str


class RegionalResponse(BaseModel):
    key: str
    name: str
    unit: str
    world_supply: float
    world_demand: float
    world_balance: float
    rows: List[RegionalRow]


class CurvePoint(BaseModel):
    tenor_month: int
    label: str
    price: float
    source: str


class CurveResponse(BaseModel):
    key: str
    name: str
    price_unit: str
    structure: str
    points: List[CurvePoint]


class OptionsRequest(BaseModel):
    forward: float = Field(gt=0, description="Forward / futures price")
    strike: float = Field(gt=0)
    days_to_expiry: int = Field(ge=1, le=365 * 3)
    sigma: float = Field(gt=0, lt=5)
    rate: float = Field(ge=0, lt=0.5)
    option_type: str = Field("call", pattern="^(call|put)$")


class OptionsResponse(BaseModel):
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/commodities", response_model=List[CommoditySummary])
def list_commodities() -> List[CommoditySummary]:
    return [
        CommoditySummary(
            key=t.key, name=t.name, sector=t.sector, ticker=t.ticker,
            price_unit=t.price_unit, unit=t.unit, inventory_unit=t.inventory_unit,
            base_price=t.base_price, days_cover_target=t.days_cover_target,
            ideal_utilization_pct=t.ideal_utilization_pct,
        )
        for t in COMMODITY_TEMPLATES.values()
    ]


@app.get("/api/spot/{key}", response_model=SpotResponse)
def spot(key: str) -> SpotResponse:
    if key not in COMMODITY_TEMPLATES:
        raise HTTPException(404, f"Unknown commodity '{key}'")
    tpl = COMMODITY_TEMPLATES[key]
    live = get_live_spot(key)
    if live is not None:
        return SpotResponse(key=key, name=tpl.name, price_unit=tpl.price_unit,
                            price=live["price"], change_pct=live["change_pct"],
                            asof=live["asof"], source=live["source"])
    return SpotResponse(key=key, name=tpl.name, price_unit=tpl.price_unit,
                        price=tpl.base_price, change_pct=0.0,
                        asof="reference", source="reference")


@app.get("/api/balance/{key}", response_model=BalanceResponse)
def balance(
    key: str,
    forecast_months: int = Query(24, ge=6, le=36),
    supply_adj_pct: float = Query(0.0, ge=-10, le=10),
    demand_adj_pct: float = Query(0.0, ge=-10, le=10),
    gdp_growth_pct: float = Query(2.5, ge=-2, le=6),
) -> BalanceResponse:
    if key not in COMMODITY_TEMPLATES:
        raise HTTPException(404, f"Unknown commodity '{key}'")
    tpl = COMMODITY_TEMPLATES[key]
    df = get_sd_dataset(key, forecast_months=forecast_months)
    a = BalanceAssumptions(
        supply_adj_pct=supply_adj_pct,
        demand_adj_pct=demand_adj_pct,
        gdp_growth_pct=gdp_growth_pct,
        forecast_months=forecast_months,
    )
    bal = run_balance(df, key, a)
    fv = estimate_fair_value(bal)
    last = bal.iloc[-1]
    points = [
        BalancePoint(
            date=str(idx.date()),
            supply=float(row["supply"]),
            demand=float(row["demand"]),
            stocks=float(row["stocks_model"]),
            days_cover=float(row["days_cover_model"]),
            price=float(row["price"]),
            fair_value=float(fv.loc[idx, "fair_value_price"]),
            is_forecast=bool(row["is_forecast"]),
        )
        for idx, row in bal.iterrows()
    ]
    return BalanceResponse(
        key=key, name=tpl.name,
        price_unit=tpl.price_unit, unit=tpl.unit,
        inventory_unit=tpl.inventory_unit,
        end_stocks=float(last["stocks_model"]),
        end_days_cover=float(last["days_cover_model"]),
        end_utilization_pct=float(last["capacity_pct"]),
        end_fair_value=float(fv.iloc[-1]["fair_value_price"]),
        points=points,
    )


@app.get("/api/regional/{key}", response_model=RegionalResponse)
def regional(key: str) -> RegionalResponse:
    if key not in COMMODITY_TEMPLATES:
        raise HTTPException(404, f"Unknown commodity '{key}'")
    tpl = COMMODITY_TEMPLATES[key]
    reg = get_regional_dataset(key)
    rows = []
    for _, r in reg.iterrows():
        status = ("exporter" if r["net_trade"] > 0
                  else "importer" if r["net_trade"] < 0 else "balanced")
        rows.append(RegionalRow(
            region=r["region"], supply=float(r["supply"]),
            demand=float(r["demand"]), net_trade=float(r["net_trade"]),
            supply_share_pct=float(r["supply_share_pct"]),
            demand_share_pct=float(r["demand_share_pct"]),
            status=status,
        ))
    return RegionalResponse(
        key=key, name=tpl.name, unit=tpl.unit,
        world_supply=float(reg["supply"].sum()),
        world_demand=float(reg["demand"].sum()),
        world_balance=float(reg["supply"].sum() - reg["demand"].sum()),
        rows=rows,
    )


@app.get("/api/curve/{key}", response_model=CurveResponse)
def curve(key: str, n_max: int = Query(12, ge=2, le=24)) -> CurveResponse:
    if key not in COMMODITY_TEMPLATES:
        raise HTTPException(404, f"Unknown commodity '{key}'")
    tpl = COMMODITY_TEMPLATES[key]
    df = get_futures_curve(key, n_max=n_max)
    prices = df["price"].to_numpy()
    if len(prices) >= 2:
        structure = ("contango" if prices[-1] > prices[0]
                     else "backwardation" if prices[-1] < prices[0] else "flat")
    else:
        structure = "flat"
    points = [
        CurvePoint(
            tenor_month=int(r["tenor_month"]), label=str(r["label"]),
            price=float(r["price"]), source=str(r.get("source", "synthetic")),
        )
        for _, r in df.iterrows()
    ]
    return CurveResponse(key=key, name=tpl.name, price_unit=tpl.price_unit,
                         structure=structure, points=points)


@app.post("/api/options/price", response_model=OptionsResponse)
def options_price(req: OptionsRequest) -> OptionsResponse:
    opt = Black76(F=req.forward, K=req.strike, T=req.days_to_expiry / 365.0,
                  r=req.rate, sigma=req.sigma, option_type=req.option_type)
    g = opt.greeks()
    return OptionsResponse(**g)
