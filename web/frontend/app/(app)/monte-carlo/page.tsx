"use client";
import { useEffect, useState } from "react";
import { api, type Commodity, type MCResponse } from "@/lib/api";
import { KPICard } from "@/components/KPICard";
import { fmtNum } from "@/lib/utils";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Play } from "lucide-react";

export default function MonteCarloPage() {
  const [commodities, setCommodities] = useState<Commodity[]>([]);
  const [key, setKey] = useState("wti_crude");
  const [nPaths, setNPaths] = useState(500);
  const [supplySig, setSupplySig] = useState(1.5);
  const [demandSig, setDemandSig] = useState(1.2);
  const [outageProb, setOutageProb] = useState(0.05);
  const [horizon, setHorizon] = useState(18);
  const [data, setData] = useState<MCResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { api.commodities().then(setCommodities); }, []);

  const run = async () => {
    try {
      setLoading(true);
      setError(null);
      const d = await api.montecarlo(key, {
        n_paths: nPaths, supply_sigma_pct: supplySig,
        demand_sigma_pct: demandSig, outage_prob: outageProb,
        forecast_months: horizon,
      });
      setData(d);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const commodity = commodities.find((c) => c.key === key);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Monte Carlo Engine</h1>
        <p className="text-sm text-ink-200 mt-1">
          Generate a probabilistic range of prices and stocks by drawing random
          supply, demand, weather and outage shocks.
        </p>
      </div>

      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Parameters</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="metric-label">Commodity</label>
            <select value={key} onChange={(e) => setKey(e.target.value)}
              className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm text-ink-50">
              {commodities.map((c) => (
                <option key={c.key} value={c.key}>[{c.sector}] {c.name}</option>
              ))}
            </select>
          </div>
          <SliderField label="Number of paths" min={50} max={2000} step={50}
                       value={nPaths} onChange={setNPaths} integer />
          <SliderField label="Forecast months" min={6} max={36} step={3}
                       value={horizon} onChange={setHorizon} integer />
          <SliderField label="Supply σ %" min={0.5} max={5} step={0.1}
                       value={supplySig} onChange={setSupplySig} />
          <SliderField label="Demand σ %" min={0.5} max={5} step={0.1}
                       value={demandSig} onChange={setDemandSig} />
          <SliderField label="Outage prob / month" min={0} max={0.20} step={0.01}
                       value={outageProb} onChange={setOutageProb} />
        </div>
        <button onClick={run} disabled={loading}
          className="mt-4 bg-accent text-ink-900 font-medium rounded-md px-5 py-2 text-sm hover:bg-accent/90 inline-flex items-center gap-2 disabled:opacity-50">
          <Play size={14} fill="#0e1117" /> {loading ? "Running…" : "Run simulation"}
        </button>
      </div>

      {error && <div className="card p-4 border-neg text-neg text-sm">{error}</div>}

      {data && commodity && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <KPICard label="Median price"
                     value={`${data.median_price.toFixed(2)} ${data.price_unit}`} />
            <KPICard label="P5 - P95"
                     value={`${data.p5_price_avg.toFixed(2)} - ${data.p95_price_avg.toFixed(2)}`}
                     delta={`${(((data.p95_price_avg - data.p5_price_avg) / data.median_price) * 100).toFixed(0)} % width`} />
            <KPICard label={`Median end stocks (${data.inventory_unit})`}
                     value={fmtNum(data.median_end_stocks, 0)} />
            <KPICard label="VaR 95 % price drop"
                     value={`${data.var_95_price_drop.toFixed(2)} ${data.price_unit}`}
                     deltaTone="neg" />
          </div>

          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold">Probabilistic fan chart</h2>
              <span className="text-[11px] text-ink-200 italic">
                P5-P95 band around the median path.
              </span>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={data.fan_chart}>
                <defs>
                  <linearGradient id="fan-fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00d4ff" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#00d4ff" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="date" stroke="#6b7280" tick={{ fontSize: 11 }}
                  tickFormatter={(d: string) => d.slice(0, 7)} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{
                  background: "#111827", border: "1px solid #1f2937",
                  borderRadius: 8, fontSize: 12,
                }} formatter={(v: number) => v.toFixed(2)} />
                <Area dataKey="p95" stroke="none" fill="url(#fan-fill)" />
                <Area dataKey="p5"  stroke="none" fill="#0e1117" />
                <Area dataKey="p50" stroke="#00d4ff" strokeWidth={2}
                      fill="none" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            <div className="card p-5">
              <h2 className="text-sm font-semibold mb-2">Price distribution</h2>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={data.histogram_price}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="x" stroke="#6b7280" tick={{ fontSize: 11 }}
                    tickFormatter={(v: number) => v.toFixed(0)} />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{
                    background: "#111827", border: "1px solid #1f2937",
                    borderRadius: 8, fontSize: 12,
                  }} />
                  <Bar dataKey="count" fill="#f59e0b" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="card p-5">
              <h2 className="text-sm font-semibold mb-2">End-stocks distribution</h2>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={data.histogram_stocks}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="x" stroke="#6b7280" tick={{ fontSize: 11 }}
                    tickFormatter={(v: number) => v.toFixed(0)} />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{
                    background: "#111827", border: "1px solid #1f2937",
                    borderRadius: 8, fontSize: 12,
                  }} />
                  <Bar dataKey="count" fill="#00d4ff" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {!data && !loading && (
        <p className="text-sm text-ink-200">
          Configure the parameters then click <b>Run simulation</b>. A 500-path
          run takes ~1-2 seconds.
        </p>
      )}
    </div>
  );
}

function SliderField({
  label, min, max, step, value, onChange, integer,
}: {
  label: string; min: number; max: number; step: number;
  value: number; onChange: (v: number) => void; integer?: boolean;
}) {
  return (
    <div>
      <label className="metric-label">{label}</label>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(integer ? parseInt(e.target.value) : parseFloat(e.target.value))}
        className="w-full mt-2 accent-accent" />
      <div className="text-xs text-ink-100 font-mono mt-1">{value}</div>
    </div>
  );
}
