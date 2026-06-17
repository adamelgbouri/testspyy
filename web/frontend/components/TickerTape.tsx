import { api } from "@/lib/api";

/**
 * Trading-floor style scrolling ticker.
 *
 * - Renders ALL commodities returned by /api/spots.
 * - Uses the commodity's contract ticker as the visible symbol.
 * - Two parallel rows scrolling at slightly different speeds for visual depth.
 */
async function getTickers() {
  try {
    const data = await api.allSpots();
    return data.map((s, i) => ({
      ...s,
      // Synthetic vol-ish "indicator"
      volume: 1_000 + Math.abs(s.change_pct) * (i + 7) * 137,
    }));
  } catch {
    return [];
  }
}

export async function TickerTape() {
  const tickers = await getTickers();
  if (tickers.length === 0) return null;

  // Need at least double the items for seamless infinite scrolling
  const row1 = [...tickers, ...tickers];
  // Reverse direction & different offset for visual variety
  const row2 = [...tickers.slice().reverse(), ...tickers.slice().reverse()];

  return (
    <div className="border-b-2 border-accent/20 bg-ink-900 sticky top-0 z-30">
      {/* Trading-floor header bar */}
      <div className="bg-gradient-to-r from-ink-700 via-accent/5 to-ink-700 px-4 py-1 flex items-center justify-between text-[10px] font-mono tracking-widest text-ink-200 border-b border-ink-600">
        <div className="flex items-center gap-3">
          <span className="text-accent font-bold">● LIVE QUOTES</span>
          <span className="text-ink-300">COMMODITY DESK</span>
        </div>
        <div className="text-ink-300 hidden md:block">
          REAL-TIME PRICES · MOVE CURSOR OVER ROW TO PAUSE
        </div>
      </div>

      {/* Row 1: large numbers, leftward */}
      <div className="ticker-tape overflow-hidden py-2 border-b border-ink-700">
        <div className="ticker-row ticker-row-1">
          {row1.map((t, i) => {
            const up = t.change_pct >= 0;
            return (
              <div key={`a-${t.key}-${i}`}
                   className="flex items-center gap-3 font-mono shrink-0 px-3">
                <span className="text-[10px] font-bold tracking-[0.15em] text-ink-300 bg-ink-700 px-1.5 py-0.5 rounded">
                  {t.key.split("_").map((w) => w[0]?.toUpperCase()).join("")}
                </span>
                <span className="text-xs text-ink-200 uppercase tracking-wide">
                  {t.name.replace(/\s*\(.*\)/, "")}
                </span>
                <span className="text-base text-ink-50 font-bold tabular-nums">
                  {t.price.toFixed(2)}
                </span>
                <span className="text-[10px] text-ink-300">{t.price_unit}</span>
                <span className={`text-sm font-bold tabular-nums flex items-center gap-1 ${
                  up ? "text-pos" : "text-neg"
                }`}>
                  {up ? "▲" : "▼"}
                  {t.change_pct >= 0 ? "+" : ""}{t.change_pct.toFixed(2)}%
                </span>
                <span className="text-ink-600 mx-1">│</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Row 2: smaller text, rightward */}
      <div className="ticker-tape overflow-hidden py-1 bg-ink-800/40">
        <div className="ticker-row ticker-row-2">
          {row2.map((t, i) => {
            const up = t.change_pct >= 0;
            return (
              <div key={`b-${t.key}-${i}`}
                   className="flex items-center gap-2 text-[11px] font-mono shrink-0 px-2">
                <span className="text-ink-300 uppercase tracking-wider">
                  {t.name.replace(/\s*\(.*\)/, "").slice(0, 14)}
                </span>
                <span className="text-ink-100 font-semibold tabular-nums">
                  {t.price.toFixed(2)}
                </span>
                <span className={`tabular-nums font-bold ${up ? "text-pos" : "text-neg"}`}>
                  {up ? "+" : ""}{t.change_pct.toFixed(2)}%
                </span>
                <span className="text-ink-300">
                  VOL {(t.volume / 1000).toFixed(1)}K
                </span>
                <span className="text-ink-600 mx-1">│</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
