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

// ── API methods
export const api = {
  commodities:  () => fetcher<Commodity[]>("/api/commodities"),
  spot:         (key: string) => fetcher<Spot>(`/api/spot/${key}`),
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
};
