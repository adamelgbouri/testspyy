/**
 * Typed client for the FastAPI commodity backend.
 *
 * Set `NEXT_PUBLIC_API_URL` in `.env.local` to point at the API
 * (defaults to http://localhost:8000).
 */
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function fetcher<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    next: { revalidate: 60 },           // ISR-style cache for GETs
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

// ── Types
export type Commodity = {
  key: string;
  name: string;
  sector: string;
  ticker: string;
  price_unit: string;
  unit: string;
  inventory_unit: string;
  base_price: number;
  days_cover_target: number;
  ideal_utilization_pct: number;
};

export type Spot = {
  key: string;
  name: string;
  price_unit: string;
  price: number;
  change_pct: number;
  asof: string;
  source: "yahoo" | "reference";
};

export type BalancePoint = {
  date: string;
  supply: number;
  demand: number;
  stocks: number;
  days_cover: number;
  price: number;
  fair_value: number;
  is_forecast: boolean;
};

export type BalanceResponse = {
  key: string;
  name: string;
  price_unit: string;
  unit: string;
  inventory_unit: string;
  end_stocks: number;
  end_days_cover: number;
  end_utilization_pct: number;
  end_fair_value: number;
  points: BalancePoint[];
};

export type RegionalRow = {
  region: string;
  supply: number;
  demand: number;
  net_trade: number;
  supply_share_pct: number;
  demand_share_pct: number;
  status: "exporter" | "importer" | "balanced";
};

export type RegionalResponse = {
  key: string;
  name: string;
  unit: string;
  world_supply: number;
  world_demand: number;
  world_balance: number;
  rows: RegionalRow[];
};

export type CurvePoint = {
  tenor_month: number;
  label: string;
  price: number;
  source: "yahoo" | "synthetic";
};

export type CurveResponse = {
  key: string;
  name: string;
  price_unit: string;
  structure: "contango" | "backwardation" | "flat";
  points: CurvePoint[];
};

export type OptionsRequest = {
  forward: number;
  strike: number;
  days_to_expiry: number;
  sigma: number;
  rate: number;
  option_type: "call" | "put";
};

export type OptionsResponse = {
  price: number;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
};

export type SpreadLeg = {
  key: string; name: string; price: number;
  price_unit: string; multiplier: number; source: string;
};

export type SpreadRow = {
  name: string; description: string; unit: string;
  typical: number; margin: number; legs: SpreadLeg[];
};

export type Position = {
  commodity_key: string; direction: "Long" | "Short";
  quantity: number; entry_price: number;
  entry_date?: string; notes?: string;
};

export type RiskRow = {
  commodity: string; sector: string; direction: string;
  quantity: number; vol_pct: number; var: number; cvar: number;
};

export type StressRow = {
  scenario: string; shock_pct: number;
  new_price: number; pnl_impact: number;
};

export type RiskResponse = {
  rows: RiskRow[]; total_var: number; total_cvar: number;
  confidence: number; horizon_days: number;
  stress: StressRow[];
};

export type MCRequest = {
  n_paths: number;
  supply_sigma_pct?: number; demand_sigma_pct?: number;
  weather_sigma_pct?: number;
  outage_prob?: number; outage_size_pct?: number;
  forecast_months?: number;
};

export type HistBin = { x: number; count: number };
export type FanPoint = { date: string; p5: number; p50: number; p95: number };

export type MCResponse = {
  key: string; name: string; price_unit: string; inventory_unit: string;
  n_paths: number;
  median_price: number; p5_price_avg: number; p95_price_avg: number;
  median_end_stocks: number; var_95_price_drop: number;
  histogram_price: HistBin[];
  histogram_stocks: HistBin[];
  fan_chart: FanPoint[];
};

export type EventRow = {
  date: string; event: string; tags: string[]; frequency: string;
};

// ── API methods
export const api = {
  commodities:  () => fetcher<Commodity[]>("/api/commodities"),
  spot:         (key: string) => fetcher<Spot>(`/api/spot/${key}`),
  allSpots:     () => fetcher<(Spot & { sector: string })[]>("/api/spots"),
  balance:      (key: string, q?: Record<string, number>) =>
    fetcher<BalanceResponse>(
      `/api/balance/${key}` +
      (q ? `?${new URLSearchParams(Object.fromEntries(Object.entries(q).map(([k, v]) => [k, String(v)])))}` : ""),
    ),
  regional:     (key: string) => fetcher<RegionalResponse>(`/api/regional/${key}`),
  curve:        (key: string) => fetcher<CurveResponse>(`/api/curve/${key}`),
  optionsPrice: (body: OptionsRequest) =>
    fetcher<OptionsResponse>("/api/options/price", {
      method: "POST", body: JSON.stringify(body),
    }),
  spreads: () => fetcher<SpreadRow[]>("/api/spreads"),
  risk: (positions: Position[], confidence = 0.95, horizon_days = 1) =>
    fetcher<RiskResponse>("/api/risk", {
      method: "POST",
      body: JSON.stringify({ positions, confidence, horizon_days }),
    }),
  montecarlo: (key: string, body: MCRequest) =>
    fetcher<MCResponse>(`/api/montecarlo/${key}`, {
      method: "POST", body: JSON.stringify(body),
    }),
  events: () => fetcher<EventRow[]>("/api/events"),
};
