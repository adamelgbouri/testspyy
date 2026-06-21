import { api } from "@/lib/api";
import { fmtNum } from "@/lib/utils";
import { ArrowLeftRight, ArrowDownRight, ArrowUpRight } from "lucide-react";

export const dynamic = "force-dynamic";

export default async function SpreadsPage() {
  const spreads = await api.spreads();

  return (
    <div className="space-y-6 animate-slide-up">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <ArrowLeftRight size={14} className="text-accent" />
          <span className="badge">{spreads.length} STRUCTURES</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight">Spreads & Cracks</h1>
        <p className="text-sm text-ink-200 mt-1">
          Multi-leg structures with live margin computed from current spots: refiner
          margins, location spreads, gold/silver ratio.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-4 stagger">
        {spreads.map((s) => {
          const margin = s.margin;
          const vsTypical = margin - s.typical;
          const vsTypicalPct = s.typical !== 0 ? (vsTypical / Math.abs(s.typical)) * 100 : 0;
          const positive = margin >= 0;
          const above = vsTypical >= 0;

          // Visual proportion: where is margin within [typical/2, typical*2]?
          const rangeLo = s.typical * 0.5;
          const rangeHi = s.typical * 1.8;
          const pct = Math.max(0, Math.min(1, (margin - rangeLo) / (rangeHi - rangeLo)));

          return (
            <div key={s.name} className="card-glow p-5">
              <div className="flex items-start justify-between mb-2">
                <h2 className="font-semibold text-ink-50 text-base">{s.name}</h2>
                <span className="badge">{s.unit}</span>
              </div>
              <p className="text-xs text-ink-200 italic mb-4 leading-relaxed">{s.description}</p>

              {/* Margin display */}
              <div className="flex items-end justify-between gap-3 mb-3">
                <div>
                  <div className="metric-label">Current margin</div>
                  <div className={`font-mono text-3xl font-bold mt-1 tabular-nums ${
                    positive ? "text-pos" : "text-neg"
                  }`}>
                    {positive ? "+" : ""}{fmtNum(margin, 2)}
                  </div>
                </div>
                <div className="text-right">
                  <div className="metric-label">vs typical {s.typical.toFixed(1)}</div>
                  <div className={`font-mono text-lg font-semibold mt-1 flex items-center justify-end gap-1 ${
                    above ? "text-pos" : "text-neg"
                  }`}>
                    {above ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                    {above ? "+" : ""}{vsTypicalPct.toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Range bar */}
              <div className="relative h-3 bg-ink-700 rounded-full mb-1">
                <div className="absolute top-0 bottom-0 w-0.5 bg-violet"
                  style={{ left: `${((s.typical - rangeLo) / (rangeHi - rangeLo)) * 100}%` }} />
                <div className={`absolute top-0 bottom-0 rounded-full transition-all ${
                  above ? "bg-pos/60" : "bg-neg/60"
                }`}
                  style={{ width: `${pct * 100}%` }} />
                <div className={`absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2 border-ink-50 ${
                  above ? "bg-pos" : "bg-neg"
                }`}
                  style={{ left: `${pct * 100}%`, transform: "translate(-50%, -50%)" }} />
              </div>
              <div className="flex justify-between text-[9px] text-ink-300 font-mono mb-4">
                <span>{rangeLo.toFixed(0)}</span>
                <span className="text-violet">↑ typical</span>
                <span>{rangeHi.toFixed(0)}</span>
              </div>

              {/* Legs */}
              <h3 className="section-title">Legs</h3>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-ink-200 border-b border-ink-600">
                    <th className="text-left pb-1.5">Commodity</th>
                    <th className="text-right pb-1.5">×</th>
                    <th className="text-right pb-1.5">Spot</th>
                    <th className="text-center pb-1.5">Source</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {s.legs.map((l) => (
                    <tr key={l.key} className="border-b border-ink-700/60">
                      <td className="py-1.5 text-ink-50">{l.name}</td>
                      <td className="text-right text-ink-200">{l.multiplier}</td>
                      <td className="text-right">
                        <span className="text-ink-50">{l.price.toFixed(2)}</span>
                        <span className="text-[9px] text-ink-300 ml-1">{l.price_unit}</span>
                      </td>
                      <td className="text-center">
                        <span className={`badge ${
                          l.source === "yahoo" ? "border-pos/40 text-pos" : "text-ink-300"
                        }`}>
                          {l.source}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>

      <div className="card p-5 border-l-2 border-accent">
        <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
          <ArrowLeftRight size={14} className="text-accent" /> How crack spreads work
        </h3>
        <p className="text-xs text-ink-100 leading-relaxed">
          The <b className="text-accent">3-2-1 Crack</b> represents the rough profit
          of refining 3 barrels of crude into 2 barrels of gasoline + 1 barrel of
          heating oil. Refiner margins widen when product prices rise faster than
          crude — that's the &quot;crack&quot;. The platform converts $/gal to $/bbl
          using 42 gal/bbl so all legs are comparable.
        </p>
      </div>
    </div>
  );
}
