"use client";
import { useEffect, useMemo, useState } from "react";
import { api, type OptionsResponse } from "@/lib/api";
import { KPICard } from "@/components/KPICard";
import { VolSurface } from "@/components/VolSurface";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis, ComposedChart,
} from "recharts";
import { Activity, Zap, Layers } from "lucide-react";

type Inputs = {
  forward: number;
  strike: number;
  days: number;
  sigma: number;
  rate: number;
  type: "call" | "put";
};

const defaults: Inputs = {
  forward: 70, strike: 72, days: 90, sigma: 0.30, rate: 0.045, type: "call",
};

export default function OptionsPage() {
  const [inputs, setInputs] = useState<Inputs>(defaults);
  const [result, setResult] = useState<OptionsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const id = setTimeout(async () => {
      try {
        setLoading(true);
        const r = await api.optionsPrice({
          forward: inputs.forward, strike: inputs.strike,
          days_to_expiry: inputs.days, sigma: inputs.sigma,
          rate: inputs.rate, option_type: inputs.type,
        });
        setResult(r);
        setError(null);
      } catch (e: any) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    }, 200);
    return () => clearTimeout(id);
  }, [inputs]);

  // Greeks across strike for visualisation
  const [greeksGrid, setGreeksGrid] = useState<any[]>([]);
  useEffect(() => {
    const fetchGrid = async () => {
      const strikes: number[] = [];
      const lo = inputs.forward * 0.85;
      const hi = inputs.forward * 1.15;
      for (let i = 0; i < 15; i++) {
        strikes.push(lo + (i / 14) * (hi - lo));
      }
      const results = await Promise.all(strikes.map(async (K) => {
        try {
          const r = await api.optionsPrice({
            forward: inputs.forward, strike: K,
            days_to_expiry: inputs.days, sigma: inputs.sigma,
            rate: inputs.rate, option_type: inputs.type,
          });
          return { strike: K, delta: r.delta, gamma: r.gamma, vega: r.vega };
        } catch {
          return { strike: K, delta: 0, gamma: 0, vega: 0 };
        }
      }));
      setGreeksGrid(results);
    };
    fetchGrid();
  }, [inputs.forward, inputs.days, inputs.sigma, inputs.rate, inputs.type]);

  const payoff = useMemo(() => {
    if (!result) return [];
    const lo = inputs.forward * 0.6;
    const hi = inputs.forward * 1.4;
    const n = 80;
    const pts: { price: number; pnl: number; intrinsic: number }[] = [];
    for (let i = 0; i < n; i++) {
      const F = lo + ((hi - lo) * i) / (n - 1);
      const intrinsic = inputs.type === "call"
        ? Math.max(F - inputs.strike, 0)
        : Math.max(inputs.strike - F, 0);
      pts.push({ price: F, pnl: intrinsic - result.price, intrinsic });
    }
    return pts;
  }, [result, inputs]);

  const moneyness = inputs.strike > inputs.forward ? "OTM" : inputs.strike < inputs.forward ? "ITM" : "ATM";

  return (
    <div className="space-y-6 animate-slide-up">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Zap size={14} className="text-accent" />
          <span className="badge">BLACK-76</span>
          <span className="badge">{moneyness}</span>
          <span className="badge border-violet/40 text-violet">{inputs.type.toUpperCase()}</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight">Options & Greeks</h1>
        <p className="text-sm text-ink-200 mt-1">
          European option pricer for commodity futures. All Greeks recompute instantly as you tweak inputs.
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        {/* Inputs */}
        <div className="card p-5 lg:col-span-1">
          <h2 className="text-sm font-semibold mb-3">Inputs</h2>
          <div className="space-y-3">
            <NumberInput label="Forward F" value={inputs.forward} step={0.5}
              onChange={(v) => setInputs({ ...inputs, forward: v })} />
            <NumberInput label="Strike K" value={inputs.strike} step={0.5}
              onChange={(v) => setInputs({ ...inputs, strike: v })} />
            <SliderInput label="Days to expiry" min={7} max={365} step={1}
              value={inputs.days} onChange={(v) => setInputs({ ...inputs, days: v })} suffix="d" />
            <SliderInput label="Volatility σ" min={0.05} max={1.5} step={0.01}
              value={inputs.sigma} onChange={(v) => setInputs({ ...inputs, sigma: v })}
              suffix={` (${(inputs.sigma * 100).toFixed(0)}%)`} />
            <SliderInput label="Risk-free rate" min={0} max={0.10} step={0.001}
              value={inputs.rate} onChange={(v) => setInputs({ ...inputs, rate: v })}
              suffix={` (${(inputs.rate * 100).toFixed(2)}%)`} />
            <div>
              <label className="metric-label">Option type</label>
              <div className="flex gap-2 mt-1.5">
                {(["call", "put"] as const).map((t) => (
                  <button key={t} onClick={() => setInputs({ ...inputs, type: t })}
                    className={`flex-1 py-2 rounded-md text-sm border transition font-semibold ${
                      inputs.type === t ? "bg-accent text-ink-900 border-accent" :
                      "border-ink-500 text-ink-100 hover:bg-ink-600"
                    }`}>
                    {t.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Output card - hero number */}
        <div className="card-glow p-5 lg:col-span-2 relative overflow-hidden">
          <div className="metric-label flex items-center gap-2">
            <Activity size={11} className="text-accent" /> Option premium
          </div>
          <div className="font-mono text-5xl font-bold text-accent mt-1 mb-3 tabular-nums">
            {result ? result.price.toFixed(4) : "—"}
            {loading && <span className="text-xs text-ink-200 ml-3 font-normal">recalculating…</span>}
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mt-4">
            {result && (
              <>
                <GreekTile label="Delta" value={result.delta.toFixed(4)} hint="hedge ratio" />
                <GreekTile label="Gamma" value={result.gamma.toFixed(5)} hint="Δ sensitivity" />
                <GreekTile label="Vega" value={result.vega.toFixed(4)} hint="per 1 vol pt" />
                <GreekTile label="Theta" value={result.theta.toFixed(4)} hint="per day" />
                <GreekTile label="Rho" value={result.rho.toFixed(4)} hint="rate sensitivity" />
              </>
            )}
          </div>
          {error && <div className="mt-3 text-neg text-sm">{error}</div>}
        </div>
      </div>

      {/* Payoff */}
      <div>
        <div className="card p-5">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold">Payoff at expiry</h2>
            <span className="text-[11px] text-ink-200 italic">
              Long {inputs.type} — intrinsic minus premium
            </span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={payoff}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="price" stroke="#6b7280" tick={{ fontSize: 11 }}
                tickFormatter={(v: number) => v.toFixed(0)} />
              <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, fontSize: 12 }}
                formatter={(v: number) => v.toFixed(3)} />
              <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="3 3" />
              <ReferenceLine x={inputs.strike} stroke="#a78bfa" strokeDasharray="4 4"
                label={{ value: "K", fill: "#a78bfa", fontSize: 11, position: "top" }} />
              <ReferenceLine x={inputs.forward} stroke="#00d4ff" strokeDasharray="2 2"
                label={{ value: "F", fill: "#00d4ff", fontSize: 11, position: "top" }} />
              <defs>
                <linearGradient id="payoff-fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22c55e" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
                </linearGradient>
              </defs>
              <Line dataKey="intrinsic" stroke="#9ca3af" strokeWidth={1}
                strokeDasharray="3 3" dot={false} />
              <Line dataKey="pnl" stroke="#f59e0b" dot={false} strokeWidth={2.5} />
            </ComposedChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-4 mt-2 text-[10px] text-ink-200">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-warn" /> P&L (after premium)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-ink-200 border-dashed" /> Intrinsic value
            </span>
          </div>
        </div>

      </div>

      {/* Volatility surface (3D) */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Layers size={15} className="text-accent" />
            <h2 className="text-sm font-semibold">Volatility surface</h2>
          </div>
          <span className="text-[11px] text-ink-200 italic">
            3D plot: X = strike · Y = implied vol · Z = maturity
          </span>
        </div>
        <VolSurface forward={inputs.forward} baseSigma={inputs.sigma} rate={inputs.rate} />
      </div>

      {/* Greeks across strike */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-2">Greeks across strike</h2>
        <p className="text-[11px] text-ink-200 italic mb-3">
          How delta, gamma and vega evolve as we move the strike around the
          forward (F = {inputs.forward.toFixed(2)}).
        </p>
        <div className="grid lg:grid-cols-3 gap-4">
          <GreekChart data={greeksGrid} dataKey="delta" colour="#22c55e" label="Delta" />
          <GreekChart data={greeksGrid} dataKey="gamma" colour="#00d4ff" label="Gamma" />
          <GreekChart data={greeksGrid} dataKey="vega" colour="#f59e0b" label="Vega" />
        </div>
      </div>
    </div>
  );
}

function GreekChart({ data, dataKey, colour, label }: any) {
  return (
    <div>
      <div className="text-xs text-ink-100 font-semibold mb-1">{label}</div>
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="strike" stroke="#6b7280" tick={{ fontSize: 10 }}
            tickFormatter={(v: number) => v.toFixed(0)} />
          <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, fontSize: 12 }}
            formatter={(v: number) => v.toFixed(4)} />
          <Line dataKey={dataKey} stroke={colour} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function GreekTile({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-lg bg-ink-700/60 border border-ink-600 px-3 py-2 hover:border-accent/40 transition">
      <div className="text-[9px] uppercase tracking-wider text-ink-300 font-semibold">{label}</div>
      <div className="font-mono text-sm text-ink-50 font-bold mt-0.5">{value}</div>
      <div className="text-[9px] text-ink-300 mt-0.5 italic">{hint}</div>
    </div>
  );
}

function NumberInput({ label, value, step, onChange }: { label: string; value: number; step: number; onChange: (v: number) => void }) {
  return (
    <div>
      <label className="metric-label">{label}</label>
      <input type="number" step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm font-mono text-ink-50 focus:outline-none focus:ring-2 focus:ring-accent" />
    </div>
  );
}

function SliderInput({ label, min, max, step, value, onChange, suffix }: { label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void; suffix?: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label className="metric-label">{label}</label>
        <span className="text-xs font-mono text-ink-100">{value}{suffix}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full mt-1.5 accent-accent" />
    </div>
  );
}
