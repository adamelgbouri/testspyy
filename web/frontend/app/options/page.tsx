"use client";
import { useEffect, useMemo, useState } from "react";
import { api, type OptionsResponse } from "@/lib/api";
import { KPICard } from "@/components/KPICard";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";

type Inputs = {
  forward: number;
  strike: number;
  days: number;
  sigma: number;
  rate: number;
  type: "call" | "put";
};

const defaults: Inputs = {
  forward: 70,
  strike: 72,
  days: 90,
  sigma: 0.30,
  rate: 0.045,
  type: "call",
};

export default function OptionsPage() {
  const [inputs, setInputs] = useState<Inputs>(defaults);
  const [result, setResult] = useState<OptionsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Debounced re-pricing on input change
  useEffect(() => {
    const id = setTimeout(async () => {
      try {
        setLoading(true);
        const r = await api.optionsPrice({
          forward: inputs.forward,
          strike: inputs.strike,
          days_to_expiry: inputs.days,
          sigma: inputs.sigma,
          rate: inputs.rate,
          option_type: inputs.type,
        });
        setResult(r);
        setError(null);
      } catch (e: any) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => clearTimeout(id);
  }, [inputs]);

  const payoff = useMemo(() => {
    if (!result) return [];
    const lo = inputs.forward * 0.6;
    const hi = inputs.forward * 1.4;
    const n = 80;
    const pts: { price: number; pnl: number }[] = [];
    for (let i = 0; i < n; i++) {
      const F = lo + ((hi - lo) * i) / (n - 1);
      const intrinsic =
        inputs.type === "call"
          ? Math.max(F - inputs.strike, 0)
          : Math.max(inputs.strike - F, 0);
      pts.push({ price: F, pnl: intrinsic - result.price });
    }
    return pts;
  }, [result, inputs]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Options &amp; Greeks</h1>
        <p className="text-sm text-ink-200 mt-1">
          Black-76 pricer for European commodity options.
        </p>
      </div>

      {/* Inputs */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Inputs</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <Field label="Forward F" value={inputs.forward}
                 onChange={(v) => setInputs({ ...inputs, forward: v })} step={0.5} />
          <Field label="Strike K" value={inputs.strike}
                 onChange={(v) => setInputs({ ...inputs, strike: v })} step={0.5} />
          <Field label="Days to expiry" value={inputs.days}
                 onChange={(v) => setInputs({ ...inputs, days: v })} step={1} integer />
          <Field label="Volatility σ (annual)" value={inputs.sigma}
                 onChange={(v) => setInputs({ ...inputs, sigma: v })} step={0.01} />
          <Field label="Risk-free rate" value={inputs.rate}
                 onChange={(v) => setInputs({ ...inputs, rate: v })} step={0.005} />
          <div>
            <label className="metric-label">Option type</label>
            <div className="flex gap-2 mt-1">
              {(["call", "put"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setInputs({ ...inputs, type: t })}
                  className={`px-4 py-1.5 rounded-md text-sm border transition ${
                    inputs.type === t
                      ? "bg-accent text-ink-900 border-accent font-medium"
                      : "border-ink-500 text-ink-100 hover:bg-ink-600"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Results */}
      {error && <div className="card p-4 border-neg text-neg text-sm">{error}</div>}
      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
            <KPICard label="Premium" value={result.price.toFixed(4)} />
            <KPICard label="Delta" value={result.delta.toFixed(4)} />
            <KPICard label="Gamma" value={result.gamma.toFixed(5)} />
            <KPICard label="Vega" value={result.vega.toFixed(4)} delta="per 1 vol pt" />
            <KPICard label="Theta" value={result.theta.toFixed(4)} delta="per day" />
            <KPICard label="Rho" value={result.rho.toFixed(4)} />
          </div>

          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold">Payoff at expiry</h2>
              <span className="text-[11px] text-ink-200 italic">
                Long {inputs.type} payoff across a price range, minus premium paid.
              </span>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={payoff} margin={{ top: 12, right: 24, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis
                  dataKey="price" stroke="#6b7280" tick={{ fontSize: 11 }}
                  tickFormatter={(v: number) => v.toFixed(0)}
                />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    background: "#111827", border: "1px solid #1f2937",
                    borderRadius: 8, fontSize: 12,
                  }}
                  formatter={(v: number) => v.toFixed(3)}
                />
                <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="3 3" />
                <ReferenceLine
                  x={inputs.strike} stroke="#a78bfa" strokeDasharray="4 4"
                  label={{ value: "K", fill: "#a78bfa", fontSize: 11, position: "top" }}
                />
                <Line dataKey="pnl" stroke="#f59e0b" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {loading && <p className="text-xs text-ink-200">Recalculating…</p>}
    </div>
  );
}

function Field({
  label, value, onChange, step, integer,
}: {
  label: string; value: number; step: number; integer?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="metric-label">{label}</label>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(integer ? parseInt(e.target.value) : parseFloat(e.target.value))}
        className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm
                   font-mono text-ink-50 focus:outline-none focus:ring-2 focus:ring-accent"
      />
    </div>
  );
}
