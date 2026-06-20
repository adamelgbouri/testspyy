"use client";
import { useEffect, useState } from "react";
import { Clock } from "lucide-react";

/**
 * Shows the elapsed time since a given ISO date string ("loaded 2s ago"),
 * refreshes every second so it always reflects reality.
 */
export function Freshness({ since, label = "data fetched" }: { since: string; label?: string }) {
  const [text, setText] = useState("just now");

  useEffect(() => {
    const start = new Date(since).getTime();
    const tick = () => {
      const secs = Math.max(0, Math.round((Date.now() - start) / 1000));
      if (secs < 5) setText("just now");
      else if (secs < 60) setText(`${secs}s ago`);
      else if (secs < 3600) setText(`${Math.round(secs / 60)}m ago`);
      else setText(`${Math.round(secs / 3600)}h ago`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [since]);

  return (
    <span className="inline-flex items-center gap-1 text-[11px] text-ink-300 font-mono">
      <Clock size={10} /> {label} {text}
    </span>
  );
}
