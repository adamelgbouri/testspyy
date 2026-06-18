"use client";
import { useEffect, useState } from "react";
import { api, type MacroResponse } from "@/lib/api";
import { KPICard } from "@/components/KPICard";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { Globe, TrendingUp, TrendingDown, Percent } from "lucide-react";

type Series = Record<string, MacroResponse>;

const PALETTE = ["#00d4ff", "#a78bfa", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4", "#f97316", "#9ca3af"];

export default function MacroPage() {
  const [countries, setCountries] = useState<string[]>([]);
  const [primary, setPrimary] = useState<string>("");
  const [compare, setCompare] = useState<string[]>([]);
  const [series, setSeries] = useState<Series>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.macroCountries().then((cs) => {
      setCountries(cs);
      setPrimary(cs[0]);
      setCompare([cs[2], cs[3]]);
    });
  }, []);

  useEffect(() => {
    if (!primary) return;
    const all = [primary, ...compare].filter((c, i, a) => a.indexOf(c) === i);
    setLoading(true);
    Promise.all(all.map((c) => api.macroCountry(c).then((d) => [c, d] as const)))
      .then((arr) => {
        const map: Series = {};
        for (const [c, d] of arr) map[c] = d;
        setSeries(map);
      })
      .finally(() => setLoading(false));
  }, [primary, compare]);

  const primaryData = series[primary];
  const latestRows = (primary ? [primary, ...compare] : [])
    .filter((c) => series[c])
    .map((c) => {
      const last = series[c].points[series[c].points.length - 1];
      const yoyIdx = Math.max(0, series[c].points.length - 13);
      const yoyGdp = (last.gdp_index / series[c].points[yoyIdx].gdp_index - 1) * 100;
      return { country: c, ...last, yoyGdp };
    });

  // Rebased GDP series (100 at start)
  const rebased = primaryData
    ? primaryData.points.map((p, i) => {
        const r: Record<string, number | string> = { date: p.date.slice(0, 7) };
        for (const c of [primary, ...compare]) {
          const ser = series[c];
          if (!ser) continue;
          const v = ser.points[i]?.gdp_index;
          if (v !== undefined) {
            r[c] = (v / ser.points[0].gdp_index) * 100;
          }
        }
        return r;
      })
    : [];

  const policyRates = primaryData
    ? primaryData.points.map((p, i) => {
        const r: Record<string, number | string> = { date: p.date.slice(0, 7) };
        for (const c of [primary, ...compare]) {
          const ser = series[c];
          if (!ser) continue;
          r[c] = ser.points[i]?.policy_rate;
        }
        return r;
      })
    : [];

  return (
    <div className="space-y-6 animate-slide-up">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Globe size={14} className="text-accent" />
          <span className="badge">{countries.length} COUNTRIES</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight">Macro Overlay</h1>
        <p className="text-sm text-ink-200 mt-1">
          Cross-country macro panels: GDP, PMI, FX vs USD, policy rate, CPI YoY.
          Pick a primary country and add others to compare side-by-side.
        </p>
      </div>

      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Country selection</h2>
        <div className="grid sm:grid-cols-2 gap-4">
          <div>
            <label className="metric-label">Primary country</label>
            <select value={primary}
              onChange={(e) => setPrimary(e.target.value)}
              className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm text-ink-50">
              {countries.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="metric-label">Compare with</label>
            <div className="mt-1 flex flex-wrap gap-2">
              {countries.filter((c) => c !== primary).map((c) => {
                const on = compare.includes(c);
                return (
                  <button key={c}
                    onClick={() => setCompare(on ? compare.filter((x) => x !== c) : [...compare, c])}
                    className={`px-3 py-1.5 rounded-full text-xs border transition ${
                      on ? "bg-accent text-ink-900 border-accent font-semibold"
                         : "border-ink-500 text-ink-100 hover:bg-ink-600"
                    }`}>
                    {c}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {loading && <p className="text-sm text-ink-200">Loading macro data…</p>}

      {primaryData && (
        <>
          {/* Primary snapshot */}
          <div>
            <h2 className="text-sm font-semibold mb-2 flex items-center gap-2">
              <Percent size={14} className="text-accent" />
              Snapshot — {primary}
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 stagger">
              {(() => {
                const last = primaryData.points[primaryData.points.length - 1];
                const yoyIdx = Math.max(0, primaryData.points.length - 13);
                const yoyGdp = (last.gdp_index / primaryData.points[yoyIdx].gdp_index - 1) * 100;
                const gdpSpark = primaryData.points.slice(-12).map((p) => p.gdp_index);
                const fxSpark = primaryData.points.slice(-12).map((p) => p.fx_vs_usd);
                const rateSpark = primaryData.points.slice(-12).map((p) => p.policy_rate);
                const cpiSpark = primaryData.points.slice(-12).map((p) => p.cpi_yoy);
                const pmiSpark = primaryData.points.slice(-12).map((p) => p.pmi);
                return (
                  <>
                    <KPICard label="GDP index"
                      value={last.gdp_index.toFixed(1)}
                      delta={`${yoyGdp >= 0 ? "+" : ""}${yoyGdp.toFixed(1)} % YoY`}
                      deltaTone={yoyGdp >= 0 ? "pos" : "neg"}
                      sparkline={gdpSpark} live />
                    <KPICard label="PMI"
                      value={last.pmi.toFixed(1)}
                      delta={last.pmi >= 50 ? "expansion" : "contraction"}
                      deltaTone={last.pmi >= 50 ? "pos" : "neg"}
                      sparkline={pmiSpark} />
                    <KPICard label="FX vs USD"
                      value={last.fx_vs_usd.toFixed(2)}
                      sparkline={fxSpark} />
                    <KPICard label="Policy rate"
                      value={`${last.policy_rate.toFixed(2)} %`}
                      sparkline={rateSpark} />
                    <KPICard label="CPI YoY"
                      value={`${last.cpi_yoy.toFixed(2)} %`}
                      deltaTone={last.cpi_yoy > 4 ? "neg" : last.cpi_yoy < 1 ? "neutral" : "pos"}
                      sparkline={cpiSpark} />
                  </>
                );
              })()}
            </div>
          </div>

          {/* Cross-country snapshot table */}
          {compare.length > 0 && (
            <div className="card p-5">
              <h2 className="text-sm font-semibold mb-2">Cross-country snapshot</h2>
              <p className="text-[11px] text-ink-200 italic mb-3">
                Latest reading of each indicator.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-ink-200 border-b border-ink-600">
                      <th className="text-left pb-2">Country</th>
                      <th className="text-right pb-2">GDP idx</th>
                      <th className="text-right pb-2">GDP YoY %</th>
                      <th className="text-right pb-2">PMI</th>
                      <th className="text-right pb-2">FX vs USD</th>
                      <th className="text-right pb-2">Policy rate %</th>
                      <th className="text-right pb-2">CPI YoY %</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    {latestRows.map((r, i) => (
                      <tr key={r.country} className="border-b border-ink-700/60">
                        <td className="py-1.5 text-ink-50">{r.country}</td>
                        <td className="text-right">{r.gdp_index.toFixed(1)}</td>
                        <td className={`text-right ${r.yoyGdp >= 0 ? "text-pos" : "text-neg"}`}>
                          {r.yoyGdp >= 0 ? "+" : ""}{r.yoyGdp.toFixed(1)}
                        </td>
                        <td className={`text-right ${r.pmi >= 50 ? "text-pos" : "text-neg"}`}>
                          {r.pmi.toFixed(1)}
                        </td>
                        <td className="text-right">{r.fx_vs_usd.toFixed(2)}</td>
                        <td className="text-right text-warn">{r.policy_rate.toFixed(2)}</td>
                        <td className={`text-right ${
                          r.cpi_yoy > 4 ? "text-neg" : r.cpi_yoy < 1 ? "text-ink-300" : "text-pos"
                        }`}>
                          {r.cpi_yoy.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* GDP chart */}
          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold flex items-center gap-2">
                <TrendingUp size={14} className="text-accent" /> GDP index (rebased to 100)
              </h2>
              <span className="text-[11px] text-ink-200 italic">
                Relative growth from the start of the panel
              </span>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rebased}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="date" stroke="#6b7280" tick={{ fontSize: 11 }} minTickGap={40} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, fontSize: 12 }}
                  formatter={(v: number) => v.toFixed(1)} />
                {[primary, ...compare].map((c, i) => (
                  <Line key={c} dataKey={c} stroke={PALETTE[i % PALETTE.length]}
                    strokeWidth={2} dot={false} name={c} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Policy rate chart */}
          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold flex items-center gap-2">
                <TrendingDown size={14} className="text-warn" /> Policy rate divergence
              </h2>
              <span className="text-[11px] text-ink-200 italic">
                Central-bank divergence drives FX and commodity carry
              </span>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={policyRates}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="date" stroke="#6b7280" tick={{ fontSize: 11 }} minTickGap={40} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }}
                  tickFormatter={(v: number) => `${v.toFixed(1)} %`} />
                <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8, fontSize: 12 }}
                  formatter={(v: number) => `${v.toFixed(2)} %`} />
                {[primary, ...compare].map((c, i) => (
                  <Line key={c} dataKey={c} stroke={PALETTE[i % PALETTE.length]}
                    strokeWidth={2} dot={false} name={c} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
