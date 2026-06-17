import { api } from "@/lib/api";
import { fmtNum } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function SpreadsPage() {
  const spreads = await api.spreads();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Spreads &amp; Cracks</h1>
        <p className="text-sm text-ink-200 mt-1">
          Multi-leg structures with live margin computed from current spots:
          refiner crack spreads, location spreads, gold/silver ratio.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        {spreads.map((s) => {
          const margin = s.margin;
          const vsTypical = margin - s.typical;
          const vsTypicalPct = s.typical !== 0 ? (vsTypical / Math.abs(s.typical)) * 100 : 0;
          const positive = margin >= 0;
          return (
            <div key={s.name} className="card p-5">
              <div className="flex items-start justify-between mb-2">
                <h2 className="font-semibold text-ink-50">{s.name}</h2>
                <span className="badge">{s.unit}</span>
              </div>
              <p className="text-xs text-ink-200 italic mb-4">{s.description}</p>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <div>
                  <div className="metric-label">Current margin</div>
                  <div className={`metric-value text-2xl mt-1 ${
                    positive ? "text-pos" : "text-neg"
                  }`}>
                    {positive ? "+" : ""}{fmtNum(margin, 2)}
                  </div>
                </div>
                <div>
                  <div className="metric-label">vs typical ({s.typical.toFixed(1)})</div>
                  <div className={`metric-value text-2xl mt-1 ${
                    vsTypical >= 0 ? "text-pos" : "text-neg"
                  }`}>
                    {vsTypical >= 0 ? "+" : ""}{fmtNum(vsTypical, 2)}
                  </div>
                  <div className="text-[10px] font-mono text-ink-200 mt-0.5">
                    {vsTypicalPct >= 0 ? "+" : ""}{vsTypicalPct.toFixed(1)} %
                  </div>
                </div>
              </div>

              <h3 className="section-title">Legs</h3>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-ink-200 border-b border-ink-600">
                    <th className="text-left pb-1.5">Commodity</th>
                    <th className="text-right pb-1.5">Mult</th>
                    <th className="text-right pb-1.5">Spot</th>
                    <th className="text-left pb-1.5 pl-2">Source</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {s.legs.map((l) => (
                    <tr key={l.key} className="border-b border-ink-700/60">
                      <td className="py-1.5 text-ink-50">{l.name}</td>
                      <td className="text-right">{l.multiplier}</td>
                      <td className="text-right">
                        {l.price.toFixed(2)} {l.price_unit}
                      </td>
                      <td className="pl-2">
                        <span className={`badge ${
                          l.source === "yahoo" ? "border-pos/40 text-pos" : ""
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
        <h3 className="text-sm font-semibold mb-2">How crack spreads work</h3>
        <p className="text-xs text-ink-100 leading-relaxed">
          The <b>3-2-1 Crack</b> represents the rough profit of refining 3
          barrels of crude into 2 barrels of gasoline + 1 barrel of heating
          oil. Refiner margins widen when product prices rise faster than
          crude — that's the &quot;crack&quot;. The platform converts $/gal
          to $/bbl using 42 gal/bbl so all legs are comparable.
        </p>
      </div>
    </div>
  );
}
