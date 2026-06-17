"""
FastAPI gateway for the commodity analytics engine.

Run locally with:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations
import logging
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from commodity_engine import (
    BalanceAssumptions, Black76, COMMODITY_TEMPLATES, CRACK_SPREADS,
    MCConfig, crack_margin, estimate_fair_value, get_futures_curve,
    get_live_spot, get_market_events, get_regional_dataset, get_sd_dataset,
    parametric_var, portfolio_var, run_balance, run_monte_carlo,
    stress_scenarios,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(
    title="Commodity Trading Desk API",
    version="0.2.0",
    description="Analytical backend for the commodity trading platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# =============================================================================
# Pydantic models
# =============================================================================

class CommoditySummary(BaseModel):
    key: str; name: str; sector: str; ticker: str; price_unit: str
    unit: str; inventory_unit: str
    base_price: float; days_cover_target: float; ideal_utilization_pct: float


class SpotResponse(BaseModel):
    key: str; name: str; price_unit: str
    price: float; change_pct: float; asof: str; source: str


class BalancePoint(BaseModel):
    date: str; supply: float; demand: float
    stocks: float; days_cover: float
    price: float; fair_value: float; is_forecast: bool


class BalanceResponse(BaseModel):
    key: str; name: str; price_unit: str; unit: str; inventory_unit: str
    end_stocks: float; end_days_cover: float
    end_utilization_pct: float; end_fair_value: float
    points: List[BalancePoint]


class RegionalRow(BaseModel):
    region: str; supply: float; demand: float; net_trade: float
    supply_share_pct: float; demand_share_pct: float; status: str


class RegionalResponse(BaseModel):
    key: str; name: str; unit: str
    world_supply: float; world_demand: float; world_balance: float
    rows: List[RegionalRow]


class CurvePoint(BaseModel):
    tenor_month: int; label: str; price: float; source: str


class CurveResponse(BaseModel):
    key: str; name: str; price_unit: str; structure: str
    points: List[CurvePoint]


class OptionsRequest(BaseModel):
    forward: float = Field(gt=0); strike: float = Field(gt=0)
    days_to_expiry: int = Field(ge=1, le=365 * 3)
    sigma: float = Field(gt=0, lt=5); rate: float = Field(ge=0, lt=0.5)
    option_type: str = Field("call", pattern="^(call|put)$")


class OptionsResponse(BaseModel):
    price: float; delta: float; gamma: float
    vega: float; theta: float; rho: float


class SpreadResponse(BaseModel):
    name: str; description: str; unit: str; typical: float; margin: float
    legs: List[dict]


class PositionIn(BaseModel):
    commodity_key: str; direction: str; quantity: float
    entry_price: float; entry_date: str = ""; notes: str = ""


class RiskRequest(BaseModel):
    positions: List[PositionIn]
    confidence: float = 0.95; horizon_days: int = 1


class RiskRow(BaseModel):
    commodity: str; sector: str; direction: str; quantity: float
    vol_pct: float; var: float; cvar: float


class StressRow(BaseModel):
    scenario: str; shock_pct: float
    new_price: float; pnl_impact: float


class RiskResponse(BaseModel):
    rows: List[RiskRow]
    total_var: float; total_cvar: float
    confidence: float; horizon_days: int
    stress: List[StressRow]


class MCRequest(BaseModel):
    n_paths: int = Field(500, ge=50, le=2000)
    supply_sigma_pct: float = 1.5; demand_sigma_pct: float = 1.2
    weather_sigma_pct: float = 1.0
    outage_prob: float = 0.05; outage_size_pct: float = 4.0
    forecast_months: int = 18


class HistBin(BaseModel):
    x: float; count: int


class FanPoint(BaseModel):
    date: str; p5: float; p50: float; p95: float


class MCResponse(BaseModel):
    key: str; name: str; price_unit: str; inventory_unit: str
    n_paths: int
    median_price: float; p5_price_avg: float; p95_price_avg: float
    median_end_stocks: float; var_95_price_drop: float
    histogram_price: List[HistBin]
    histogram_stocks: List[HistBin]
    fan_chart: List[FanPoint]


class EventRow(BaseModel):
    date: str; event: str; tags: List[str]; frequency: str


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
        ) for t in COMMODITY_TEMPLATES.values()
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


class SpotWithSector(SpotResponse):
    sector: str


@app.get("/api/spots", response_model=List[SpotWithSector])
def all_spots() -> List[SpotWithSector]:
    """Batch endpoint returning the latest spot for every supported commodity."""
    out: List[SpotWithSector] = []
    for key, tpl in COMMODITY_TEMPLATES.items():
        live = get_live_spot(key)
        if live is not None:
            out.append(SpotWithSector(
                key=key, name=tpl.name, price_unit=tpl.price_unit,
                sector=tpl.sector,
                price=live["price"], change_pct=live["change_pct"],
                asof=live["asof"], source=live["source"],
            ))
        else:
            # Use a small deterministic synthetic change so the heatmap has variety
            import random
            rng = random.Random(hash(key))
            out.append(SpotWithSector(
                key=key, name=tpl.name, price_unit=tpl.price_unit,
                sector=tpl.sector,
                price=tpl.base_price,
                change_pct=rng.uniform(-2.5, 2.5),
                asof="reference", source="reference",
            ))
    return out


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
        supply_adj_pct=supply_adj_pct, demand_adj_pct=demand_adj_pct,
        gdp_growth_pct=gdp_growth_pct, forecast_months=forecast_months,
    )
    bal = run_balance(df, key, a)
    fv = estimate_fair_value(bal)
    last = bal.iloc[-1]
    points = [
        BalancePoint(
            date=str(idx.date()), supply=float(row["supply"]),
            demand=float(row["demand"]), stocks=float(row["stocks_model"]),
            days_cover=float(row["days_cover_model"]),
            price=float(row["price"]),
            fair_value=float(fv.loc[idx, "fair_value_price"]),
            is_forecast=bool(row["is_forecast"]),
        ) for idx, row in bal.iterrows()
    ]
    return BalanceResponse(
        key=key, name=tpl.name, price_unit=tpl.price_unit,
        unit=tpl.unit, inventory_unit=tpl.inventory_unit,
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
        CurvePoint(tenor_month=int(r["tenor_month"]), label=str(r["label"]),
                   price=float(r["price"]), source=str(r.get("source", "synthetic")))
        for _, r in df.iterrows()
    ]
    return CurveResponse(key=key, name=tpl.name, price_unit=tpl.price_unit,
                         structure=structure, points=points)


@app.post("/api/options/price", response_model=OptionsResponse)
def options_price(req: OptionsRequest) -> OptionsResponse:
    opt = Black76(F=req.forward, K=req.strike, T=req.days_to_expiry / 365.0,
                  r=req.rate, sigma=req.sigma, option_type=req.option_type)
    return OptionsResponse(**opt.greeks())


class VolSurfaceRequest(BaseModel):
    forward: float = Field(gt=0)
    base_sigma: float = Field(0.30, gt=0, lt=3)
    rate: float = Field(0.045, ge=0, lt=0.5)
    n_strikes: int = Field(15, ge=5, le=30)
    n_maturities: int = Field(10, ge=4, le=20)


class VolSurfaceResponse(BaseModel):
    strikes: List[float]
    maturities: List[int]   # in days
    iv_grid: List[List[float]]   # iv_grid[m][k]
    forward: float


@app.post("/api/options/vol-surface", response_model=VolSurfaceResponse)
def vol_surface(req: VolSurfaceRequest) -> VolSurfaceResponse:
    """
    Build a synthetic implied vol surface vs (strike, maturity).
    SVI-style smile + maturity term structure.
    """
    import numpy as np

    F = req.forward
    sig0 = req.base_sigma
    strikes = list(np.linspace(F * 0.6, F * 1.4, req.n_strikes))
    # log-spaced maturities 15d → 2y
    maturities_d = [int(d) for d in np.geomspace(15, 730, req.n_maturities)]

    iv_grid: List[List[float]] = []
    for d in maturities_d:
        T = d / 365.0
        # term-structure: short-dated higher vol of vol, ATM vol slightly inverted
        atm = sig0 * (1.0 + 0.10 * np.exp(-T * 2))
        # SVI-like smile in log-moneyness space
        row: List[float] = []
        for K in strikes:
            k = np.log(K / F)
            # vol = atm + a*k + b*sqrt(k^2 + c)
            wing = 0.55 / np.sqrt(T + 0.05)        # wings rise as T → 0
            skew = -0.18                            # negative skew
            smile = atm + skew * k + wing * np.sqrt(k * k + 0.04) - wing * 0.20
            row.append(float(max(0.05, smile)))
        iv_grid.append(row)

    return VolSurfaceResponse(
        strikes=strikes, maturities=maturities_d,
        iv_grid=iv_grid, forward=F,
    )


@app.get("/api/spreads", response_model=List[SpreadResponse])
def spreads() -> List[SpreadResponse]:
    """Compute current margin for every preset spread."""
    out: List[SpreadResponse] = []
    for name, spec in CRACK_SPREADS.items():
        prices = {}
        legs = []
        for slot in ("F1", "F2", "F3"):
            ck = spec.get(slot)
            if not ck:
                continue
            tpl = COMMODITY_TEMPLATES[ck]
            live = get_live_spot(ck)
            p = float(live["price"]) if live else float(tpl.base_price)
            prices[ck] = p
            legs.append({
                "key": ck, "name": tpl.name, "price": p,
                "price_unit": tpl.price_unit,
                "multiplier": spec.get(f"{slot}_mult", 0),
                "source": (live["source"] if live else "reference"),
            })
        margin = crack_margin(spec, prices)
        out.append(SpreadResponse(
            name=name, description=spec["description"], unit=spec["unit"],
            typical=spec["typical"], margin=margin, legs=legs,
        ))
    return out


@app.post("/api/risk", response_model=RiskResponse)
def risk(req: RiskRequest) -> RiskResponse:
    if not req.positions:
        raise HTTPException(400, "At least one position required.")
    raw = portfolio_var(
        [p.dict() for p in req.positions],
        confidence=req.confidence, horizon_days=req.horizon_days,
    )
    # Stress on the first position
    p0 = req.positions[0]
    live = get_live_spot(p0.commodity_key)
    tpl = COMMODITY_TEMPLATES[p0.commodity_key]
    spot = float(live["price"]) if live else float(tpl.base_price)
    stress = stress_scenarios(spot, p0.quantity, p0.direction)
    return RiskResponse(
        rows=[RiskRow(**r) for r in raw["rows"]],
        total_var=raw["total_var"], total_cvar=raw["total_cvar"],
        confidence=raw["confidence"], horizon_days=raw["horizon_days"],
        stress=[StressRow(**s) for s in stress],
    )


@app.post("/api/montecarlo/{key}", response_model=MCResponse)
def montecarlo(key: str, req: MCRequest) -> MCResponse:
    if key not in COMMODITY_TEMPLATES:
        raise HTTPException(404, f"Unknown commodity '{key}'")
    cfg = MCConfig(
        n_paths=req.n_paths, supply_sigma_pct=req.supply_sigma_pct,
        demand_sigma_pct=req.demand_sigma_pct, weather_sigma_pct=req.weather_sigma_pct,
        outage_prob=req.outage_prob, outage_size_pct=req.outage_size_pct,
        forecast_months=req.forecast_months,
    )
    out = run_monte_carlo(key, cfg)
    return MCResponse(**out)


@app.get("/api/events", response_model=List[EventRow])
def events() -> List[EventRow]:
    return [EventRow(**e) for e in get_market_events()]
