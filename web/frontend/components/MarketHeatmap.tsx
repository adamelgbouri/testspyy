"use client";
import Link from "next/link";
import type { Spot, Commodity } from "@/lib/api";

type Props = { spots: (Spot & { sector: string })[] };

export function MarketHeatmap({ spots }: Props) {
  // Group by sector
  const sectors: Record<string, (Spot & { sector: string })[]> = {};
  spots.forEach((s) => {
    if (!sectors[s.sector]) sectors[s.sector] = [];
    sectors[s.sector].push(s);
  });

  const colourFor = (chg: number) => {
    const abs = Math.min(Math.abs(chg), 5);
    const intensity = abs / 5;
    if (chg >= 0) {
      return `rgba(34, 197, 94, ${0.10 + intensity * 0.55})`;
    }
    return `rgba(239, 68, 68, ${0.10 + intensity * 0.55})`;
  };

  return (
    <div className="space-y-3">
      {Object.entries(sectors).map(([sector, items]) => (
        <div key={sector}>
          <div className="text-[10px] uppercase tracking-[0.12em] text-ink-300 font-semibold mb-1.5 px-1">
            {sector}
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-1.5">
            {items.map((s) => (
              <Link
                key={s.key}
                href={`/dashboard?c=${s.key}` as any}
                className="rounded-lg border border-ink-600 p-2.5 transition-all duration-200 hover:scale-[1.02] hover:border-accent/40 hover:shadow-lg overflow-hidden relative group"
                style={{ background: colourFor(s.change_pct) }}
              >
                <div className="text-[11px] font-semibold text-ink-50 truncate">
                  {s.name}
                </div>
                <div className="font-mono text-base text-ink-50 font-bold mt-0.5">
                  {s.price.toFixed(2)}
                  <span className="text-[10px] text-ink-100 ml-1 font-normal">
                    {s.price_unit}
                  </span>
                </div>
                <div className={`text-[10px] font-mono font-bold mt-0.5 ${
                  s.change_pct >= 0 ? "text-pos" : "text-neg"
                }`}>
                  {s.change_pct >= 0 ? "▲" : "▼"} {s.change_pct >= 0 ? "+" : ""}
                  {s.change_pct.toFixed(2)}%
                </div>
                {/* shimmer on hover */}
                <div className="absolute inset-0 shimmer opacity-0 group-hover:opacity-100 pointer-events-none" />
              </Link>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
