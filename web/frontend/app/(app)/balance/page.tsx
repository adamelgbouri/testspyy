"use client";
import { useEffect, useState } from "react";
import { api, type BalanceResponse, type Commodity } from "@/lib/api";
import { BalanceChart } from "@/components/BalanceChart";
import { KPICard } from "@/components/KPICard";
import { fmtNum, fmtPrice } from "@/lib/utils";
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer,
  Tooltip, XAxis, YAxis, Cell,
} from "recharts";

export default function BalancePage() {
  const [commodities, setCommodities] = useState<Commodity[]>([]);
  const [key, setKey] = useState<string>("wti_crude");
  const [supplyAdj, setSupplyAdj] = useState(0);
  const [demandAdj, setDemandAdj] = useState(0);
  const [gdp, setGdp] = useState(2.5);
  const [horizon, setHorizon] = useState(18);
  const [data, setData] = useState<BalanceResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.commodities().then(setCommodities);
  }, []);

  useEffect(() => {
    const id = setTimeout(async () => {
      setLoading(true);
      try {
        const d = await api.balance(key, {
          forecast_months: horizon,
          supply_adj_pct: supplyAdj,
          demand_adj_pct: demandAdj,
          gdp_growth_pct: gdp,
        });
        setData(d);
      } finally {
        setLoading(false);
      }
    }, 200);
    return () => clearTimeout(id);
  }, [key, supplyAdj, demandAdj, gdp, horizon]);

  const commodity = commodities.find((c) => c.key === key);
  const buildDraws = data
    ? data.points
        .slice(-24)
        .map((p, i, arr) => ({
          date: p.date.slice(0, 7),
          delta: i === 0 ? 0 : p.stocks - arr[i - 1].stocks,
        }))
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Supply &amp; Demand Balance</h1>
        <p className="text-sm text-ink-200 mt-1">
          Drive the balance with live assumption sliders. The forecast portion
          recomputes instantly.
        </p>
      </div>

      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Assumptions</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4">
          <div>
            <label className="metric-label">Commodity</label>
            <select
              value={key}
              onChange={(e) => setKey(e.target.value)}
              className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm
                         text-ink-50 focus:outline-none focus:ring-2 focus:ring-accent"
            >
              {commodities.map((c) => (
                <option key={c.key} value={c.key}>
                  [{c.sector}] {c.name}
                </option>
              ))}
            </select>
          </div>
          <Slider label="Supply Δ %" min={-10} max={10} step={0.5}
                  value={supplyAdj} onChange={setSupplyAdj} />
          <Slider label="Demand Δ %" min={-10} max={10} step={0.5}
                  value={demandAdj} onChange={setDemandAdj} />
          <Slider label="GDP growth %" min={-2} max={6} step={0.1}
                  value={gdp} onChange={setGdp} />
          <Slider label="Forecast months" min={6} max={36} step={3} integer
                  value={horizon} onChange={setHorizon} />
        </div>
      </div>

      {data && commodity && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <KPICard
              label={`End stocks (${commodity.inventory_unit})`}
              value={fmtNum(data.end_stocks)}
            />
            <KPICard
              label="Days of cover"
              value={data.end_days_cover.toFixed(1)}
              delta={`target ${commodity.days_cover_target.toFixed(0)}`}
            />
            <KPICard
              label="Storage util %"
              value={`${data.end_utilization_pct.toFixed(1)} %`}
              delta={`ideal ${commodity.ideal_utilization_pct.toFixed(0)} %`}
            />
            <KPICard
              label="Fair value"
              value={fmtPrice(data.end_fair_value, commodity.price_unit)}
            />
          </div>

          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold">Supply, demand &amp; stocks</h2>
              <span className="text-[11px] text-ink-200 italic">
                Forecast on the right of the dashed line.
              </span>
            </div>
            <BalanceChart points={data.points} unit={commodity.unit}
                          inventoryUnit={commodity.inventory_unit} />
          </div>

          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold">Monthly builds &amp; draws</h2>
              <span className="text-[11px] text-ink-200 italic">
                Last 24 months — green = stocks rising, red = falling.
              </span>
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={buildDraws} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="date" stroke="#6b7280" tick={{ fontSize: 11 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    background: "#111827", border: "1px solid #1f2937",
                    borderRadius: 8, fontSize: 12,
                  }}
                  formatter={(v: number) => v.toFixed(1)}
                />
                <Bar dataKey="delta">
                  {buildDraws.map((d, i) => (
                    <Cell key={i} fill={d.delta >= 0 ? "#22c55e" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {loading && !data && <p className="text-sm text-ink-200">Loading…</p>}
    </div>
  );
}

function Slider({
  label, min, max, step, value, onChange, integer,
}: {
  label: string; min: number; max: number; step: number;
  value: number; onChange: (v: number) => void; integer?: boolean;
}) {
  return (
    <div>
      <label className="metric-label">{label}</label>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(integer ? parseInt(e.target.value) : parseFloat(e.target.value))}
        className="w-full mt-2 accent-accent"
      />
      <div className="text-xs text-ink-100 font-mono mt-1">{value}</div>
    </div>
  );
}
