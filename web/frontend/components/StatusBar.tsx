"use client";
import { useEffect, useState } from "react";
import { Activity, Clock, Wifi, WifiOff } from "lucide-react";
import { PulseDot } from "./PulseDot";

export function StatusBar() {
  const [now, setNow] = useState("");
  const [api, setApi] = useState<"ok" | "down" | "checking">("checking");
  const [latency, setLatency] = useState<number | null>(null);

  useEffect(() => {
    const tick = () => {
      const d = new Date();
      setNow(d.toLocaleTimeString("en-US", { hour12: false }));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const check = async () => {
      const t0 = performance.now();
      try {
        const r = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/health`,
          { cache: "no-store" }
        );
        setApi(r.ok ? "ok" : "down");
        setLatency(Math.round(performance.now() - t0));
      } catch {
        setApi("down");
        setLatency(null);
      }
    };
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="fixed bottom-0 left-0 right-0 z-30 border-t border-ink-600 bg-ink-900/95 backdrop-blur px-4 py-1.5 text-[10px] font-mono text-ink-200 flex items-center gap-5 lg:pl-[280px]">
      <div className="flex items-center gap-2">
        {api === "ok" ? <PulseDot tone="pos" size={6} /> : api === "down" ? <PulseDot tone="neg" size={6} /> : null}
        <span className={
          api === "ok" ? "text-pos" :
          api === "down" ? "text-neg" :
          "text-ink-200"
        }>
          {api === "ok" && "API LIVE"}
          {api === "down" && "API OFFLINE"}
          {api === "checking" && "CONNECTING…"}
        </span>
        {latency !== null && <span className="text-ink-300">· {latency}ms</span>}
      </div>
      <div className="flex items-center gap-1.5">
        <Activity size={11} />
        <span>YAHOO · SYNTHETIC FALLBACK</span>
      </div>
      <div className="flex items-center gap-1.5 ml-auto">
        <Clock size={11} />
        <span className="text-ink-100">{now}</span>
        <span className="text-ink-300">UTC{new Date().getTimezoneOffset() === 0 ? "" : new Date().toString().match(/GMT[-+]\d{4}/)?.[0]?.replace("GMT", "")}</span>
      </div>
      <div className="hidden md:flex items-center gap-1.5">
        <span className="text-ink-300">PRESS</span>
        <kbd>⌘</kbd><kbd>K</kbd>
        <span className="text-ink-300">SEARCH</span>
      </div>
    </div>
  );
}
