import { api } from "@/lib/api";
import { CommoditySelector } from "@/components/CommoditySelector";
import { KPICard } from "@/components/KPICard";

export const dynamic = "force-dynamic";

type Props = { searchParams: { c?: string } };

export default async function CurvePage({ searchParams }: Props) {
  const commodities = await api.commodities();
  const key = searchParams.c ?? "wti_crude";
  const curve = await api.curve(key);
  const commodity = commodities.find((c) => c.key === key)!;

  const prices = curve.points.map((p) => p.price);
  const front = prices[0] ?? 0;
  const back = prices[prices.length - 1] ?? 0;
  const spread = back - front;
  const spreadPct = front > 0 ? (spread / front) * 100 : 0;

  // Min/max for the SVG path
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;
  const W = 1000;
  const H = 280;
  const PAD = 30;
  const xScale = (i: number) =>
    PAD + (i * (W - 2 * PAD)) / Math.max(curve.points.length - 1, 1);
  const yScale = (p: number) =>
    H - PAD - ((p - min) / range) * (H - 2 * PAD);
  const path = curve.points
    .map((p, i) => `${i === 0 ? "M" : "L"}${xScale(i)},${yScale(p.price)}`)
    .join(" ");
  const areaPath = path + ` L${xScale(curve.points.length - 1)},${H - PAD} L${PAD},${H - PAD} Z`;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Futures Curve — {commodity.name}</h1>
          <p className="text-sm text-ink-200 mt-1">
            Forward curve from CME contracts when available, synthetic shape
            otherwise. Quote: {commodity.price_unit}.
          </p>
        </div>
        <CommoditySelector commodities={commodities} current={key} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KPICard label="Structure"
                 value={curve.structure.charAt(0).toUpperCase() + curve.structure.slice(1)}
                 deltaTone={curve.structure === "contango" ? "neutral"
                            : curve.structure === "backwardation" ? "pos" : "neutral"} />
        <KPICard label={`Front (${curve.points[0]?.label ?? "M1"})`}
                 value={`${front.toFixed(2)} ${commodity.price_unit}`} />
        <KPICard label={`Back (${curve.points[curve.points.length - 1]?.label ?? "M12"})`}
                 value={`${back.toFixed(2)} ${commodity.price_unit}`} />
        <KPICard label="Spread (back − front)"
                 value={`${spread >= 0 ? "+" : ""}${spread.toFixed(2)}`}
                 delta={`${spreadPct >= 0 ? "+" : ""}${spreadPct.toFixed(1)} %`}
                 deltaTone={spread > 0 ? "neutral" : "pos"} />
      </div>

      <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold">Forward curve</h2>
          <span className="text-[11px] text-ink-200 italic">
            Data source: {curve.points[0]?.source ?? "—"}
          </span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          {/* gridlines */}
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <line key={f} x1={PAD} x2={W - PAD}
                  y1={H - PAD - f * (H - 2 * PAD)}
                  y2={H - PAD - f * (H - 2 * PAD)}
                  stroke="#1f2937" strokeDasharray="3 3" />
          ))}
          {/* area */}
          <defs>
            <linearGradient id="curve-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity={0} />
            </linearGradient>
          </defs>
          <path d={areaPath} fill="url(#curve-fill)" />
          {/* line */}
          <path d={path} stroke="#f59e0b" strokeWidth={2} fill="none" />
          {/* points + labels */}
          {curve.points.map((p, i) => (
            <g key={i}>
              <circle cx={xScale(i)} cy={yScale(p.price)} r={4}
                      fill={i === 0 ? "#00d4ff" : "#f59e0b"}
                      stroke="#0e1117" strokeWidth={1.5} />
              <text x={xScale(i)} y={H - 8}
                    fill="#9ca3af" fontSize={10} textAnchor="middle"
                    fontFamily="JetBrains Mono">
                {p.label.split("-")[0]}
              </text>
            </g>
          ))}
          {/* min/max labels */}
          <text x={PAD - 6} y={yScale(max) + 4} fill="#9ca3af" fontSize={10} textAnchor="end"
                fontFamily="JetBrains Mono">{max.toFixed(2)}</text>
          <text x={PAD - 6} y={yScale(min) + 4} fill="#9ca3af" fontSize={10} textAnchor="end"
                fontFamily="JetBrains Mono">{min.toFixed(2)}</text>
        </svg>
      </div>

      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Contract details</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-ink-200 border-b border-ink-600">
                <th className="text-left pb-2">Tenor</th>
                <th className="text-left pb-2">Contract</th>
                <th className="text-right pb-2">Price ({curve.price_unit})</th>
                <th className="text-right pb-2">Δ vs front</th>
                <th className="text-left pb-2 pl-3">Source</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {curve.points.map((p, i) => {
                const diff = p.price - front;
                return (
                  <tr key={i} className="border-b border-ink-700/60">
                    <td className="py-1.5 text-ink-100">M{p.tenor_month}</td>
                    <td className="text-ink-50">{p.label}</td>
                    <td className="text-right">{p.price.toFixed(3)}</td>
                    <td className={`text-right ${
                      diff > 0 ? "text-warn" : diff < 0 ? "text-pos" : ""
                    }`}>
                      {diff >= 0 ? "+" : ""}{diff.toFixed(3)}
                    </td>
                    <td className="pl-3">
                      <span className={`badge ${
                        p.source === "yahoo" ? "border-pos/40 text-pos" : ""
                      }`}>
                        {p.source}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
