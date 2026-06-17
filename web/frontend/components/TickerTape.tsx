import { api, type Spot } from "@/lib/api";
import { ChevronDown, ChevronUp } from "lucide-react";

const KEYS = [
  "wti_crude", "brent_crude", "henry_hub_gas",
  "gold", "silver", "comex_copper", "cbot_wheat",
];

async function getTickers(): Promise<Spot[]> {
  try {
    return await Promise.all(KEYS.map((k) => api.spot(k)));
  } catch {
    return [];
  }
}

/** Server component — fetches once per request. */
export async function TickerTape() {
  const tickers = await getTickers();
  if (tickers.length === 0) return null;
  return (
    <div className="border-b border-ink-600 bg-ink-700/60 backdrop-blur">
      <div className="flex items-center gap-6 overflow-x-auto px-6 py-2 text-xs font-mono">
        {tickers.map((t) => {
          const up = t.change_pct >= 0;
          return (
            <div key={t.key} className="flex items-center gap-2 whitespace-nowrap">
              <span className="text-ink-200">{t.name}</span>
              <span className="text-ink-50 font-semibold">
                {t.price.toFixed(2)} {t.price_unit}
              </span>
              <span className={up ? "text-pos" : "text-neg"}>
                {up ? <ChevronUp className="inline" size={12} /> : <ChevronDown className="inline" size={12} />}
                {t.change_pct >= 0 ? "+" : ""}
                {t.change_pct.toFixed(2)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
