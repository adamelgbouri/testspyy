"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type Position, type RiskResponse } from "@/lib/api";
import { KPICard } from "@/components/KPICard";
import { KPICardSkeleton, TableSkeleton } from "@/components/Skeleton";
import { fmtNum } from "@/lib/utils";

const STORAGE_KEY = "trading_desk_positions";

export default function RiskPage() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [conf, setConf] = useState(0.95);
  const [horizon, setHorizon] = useState(1);
  const [data, setData] = useState<RiskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      try { setPositions(JSON.parse(raw)); } catch {}
    }
  }, []);

  useEffect(() => {
    if (positions.length === 0) { setData(null); return; }
    const id = setTimeout(async () => {
      try {
        setLoading(true);
        const d = await api.risk(positions, conf, horizon);
        setData(d);
        setError(null);
      } catch (e: any) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    }, 200);
    return () => clearTimeout(id);
  }, [positions, conf, horizon]);

  if (positions.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Risk Dashboard</h1>
          <p className="text-sm text-ink-200 mt-1">
            VaR, CVaR and stress tests on the loaded portfolio.
          </p>
        </div>
        <div className="card p-6 text-center">
          <p className="text-ink-100 mb-3">
            No positions to analyse. Add some positions first.
          </p>
          <Link href="/positions"
            className="bg-accent text-ink-900 font-medium rounded-md px-4 py-2 text-sm inline-block">
            Go to Positions →
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Risk Dashboard</h1>
        <p className="text-sm text-ink-200 mt-1">
          Parametric VaR / CVaR and stress tests on{" "}
          <span className="text-ink-100 font-mono">{positions.length}</span> position(s).
        </p>
      </div>

      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Parameters</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="metric-label">Confidence</label>
            <select value={conf}
              onChange={(e) => setConf(parseFloat(e.target.value))}
              className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm text-ink-50">
              <option value="0.90">90 %</option>
              <option value="0.95">95 %</option>
              <option value="0.99">99 %</option>
            </select>
          </div>
          <div>
            <label className="metric-label">Horizon (days)</label>
            <input type="number" min={1} max={30} value={horizon}
              onChange={(e) => setHorizon(parseInt(e.target.value))}
              className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm font-mono text-ink-50" />
          </div>
        </div>
      </div>

      {error && <div className="card p-4 border-neg text-neg text-sm">{error}</div>}
      {loading && !data && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Array.from({ length: 4 }).map((_, i) => <KPICardSkeleton key={i} />)}
          </div>
          <TableSkeleton rows={Math.min(positions.length, 6)} />
        </>
      )}

      {data && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <KPICard label={`VaR ${Math.round(data.confidence * 100)} %`}
                     value={`$${fmtNum(Math.abs(data.total_var), 0)}`}
                     delta={`${data.horizon_days}d horizon`} />
            <KPICard label={`CVaR ${Math.round(data.confidence * 100)} %`}
                     value={`$${fmtNum(Math.abs(data.total_cvar), 0)}`}
                     delta="Expected shortfall" />
            <KPICard label="Positions"
                     value={data.rows.length.toString()} />
            <KPICard label="Sectors"
                     value={Array.from(new Set(data.rows.map((r) => r.sector))).length.toString()} />
          </div>

          <div className="card p-5">
            <h2 className="text-sm font-semibold mb-2">Per-position risk decomposition</h2>
            <p className="text-[11px] text-ink-200 italic mb-3">
              Volatility-based parametric VaR/CVaR. Higher vol or larger positions
              push the numbers up.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-ink-200 border-b border-ink-600">
                    <th className="text-left pb-2">Commodity</th>
                    <th className="text-left pb-2">Sector</th>
                    <th className="text-left pb-2">Side</th>
                    <th className="text-right pb-2">Qty</th>
                    <th className="text-right pb-2">Vol (ann.)</th>
                    <th className="text-right pb-2">VaR</th>
                    <th className="text-right pb-2">CVaR</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {data.rows.map((r, i) => (
                    <tr key={i} className="border-b border-ink-700/60">
                      <td className="py-1.5 text-ink-50">{r.commodity}</td>
                      <td className="text-ink-100">{r.sector}</td>
                      <td>
                        <span className={`badge ${
                          r.direction === "Long" ? "border-pos/40 text-pos" : "border-neg/40 text-neg"
                        }`}>{r.direction}</span>
                      </td>
                      <td className="text-right">{fmtNum(r.quantity, 0)}</td>
                      <td className="text-right">{r.vol_pct.toFixed(1)} %</td>
                      <td className="text-right text-warn">${fmtNum(Math.abs(r.var), 0)}</td>
                      <td className="text-right text-neg">${fmtNum(Math.abs(r.cvar), 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card p-5">
            <h2 className="text-sm font-semibold mb-2">Stress scenarios</h2>
            <p className="text-[11px] text-ink-200 italic mb-3">
              Standard shock grid applied to the first position. Wider scenarios
              follow when more positions are added.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-ink-200 border-b border-ink-600">
                    <th className="text-left pb-2">Scenario</th>
                    <th className="text-right pb-2">Shock %</th>
                    <th className="text-right pb-2">New price</th>
                    <th className="text-right pb-2">P&L impact</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {data.stress.map((s, i) => (
                    <tr key={i} className="border-b border-ink-700/60">
                      <td className="py-1.5 text-ink-50">{s.scenario}</td>
                      <td className={`text-right ${s.shock_pct >= 0 ? "text-pos" : "text-neg"}`}>
                        {s.shock_pct >= 0 ? "+" : ""}{s.shock_pct.toFixed(1)} %
                      </td>
                      <td className="text-right">{s.new_price.toFixed(2)}</td>
                      <td className={`text-right ${s.pnl_impact >= 0 ? "text-pos" : "text-neg"}`}>
                        {s.pnl_impact >= 0 ? "+" : ""}${fmtNum(s.pnl_impact, 0)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
