import { api, type Spot } from "@/lib/api";

const KEYS = [
  "wti_crude", "brent_crude", "henry_hub_gas",
  "rbob_gasoline", "ulsd_heating_oil",
  "gold", "silver", "comex_copper",
  "cbot_wheat", "corn", "soybeans",
];

async function getTickers(): Promise<Spot[]> {
  try {
    return await Promise.all(KEYS.map((k) => api.spot(k)));
  } catch {
    return [];
  }
}

export async function TickerTape() {
  const tickers = await getTickers();
  if (tickers.length === 0) return null;

  // Duplicate for seamless scrolling
  const items = [...tickers, ...tickers];

  return (
    <div className="border-b border-ink-600 bg-ink-900/95 backdrop-blur sticky top-0 z-20 overflow-hidden">
      <style>{`
        @keyframes ticker-scroll {
          from { transform: translateX(0); }
          to { transform: translateX(-50%); }
        }
        .ticker-track {
          animation: ticker-scroll 80s linear infinite;
        }
        .ticker-track:hover {
          animation-play-state: paused;
        }
      `}</style>
      <div className="ticker-track flex items-center gap-8 py-1.5 whitespace-nowrap">
        {items.map((t, i) => {
          const up = t.change_pct >= 0;
          return (
            <div key={`${t.key}-${i}`} className="flex items-center gap-2.5 text-xs font-mono shrink-0">
              <span className="text-ink-300 text-[10px] uppercase tracking-wider">
                {t.name.replace(/\s*\(.*\)/, "")}
              </span>
              <span className="text-ink-50 font-bold">
                {t.price.toFixed(2)}
              </span>
              <span className="text-ink-300 text-[10px]">{t.price_unit}</span>
              <span className={`font-bold ${up ? "text-pos" : "text-neg"}`}>
                {up ? "▲" : "▼"}{t.change_pct >= 0 ? "+" : ""}{t.change_pct.toFixed(2)}%
              </span>
              <span className="text-ink-600 mx-2">│</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
