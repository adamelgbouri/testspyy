"use client";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowRight, LayoutDashboard, LineChart, Globe2, ArrowLeftRight,
  Target, Briefcase, Shield, FlaskConical, Calendar, Settings, Search,
} from "lucide-react";

type Item = {
  href: string;
  label: string;
  hint: string;
  icon: React.ReactNode;
  keywords: string;
};

const ITEMS: Item[] = [
  { href: "/dashboard",  label: "Dashboard",       hint: "Market overview", icon: <LayoutDashboard size={15} />, keywords: "home overview market" },
  { href: "/balance",    label: "Supply & Demand", hint: "Balance engine", icon: <LineChart size={15} />, keywords: "s&d stock supply demand inventory" },
  { href: "/regional",   label: "Regional Flows",  hint: "Trade flows", icon: <Globe2 size={15} />, keywords: "regions world country export import" },
  { href: "/curve",      label: "Futures Curve",   hint: "Forward curve", icon: <LineChart size={15} />, keywords: "forward term structure contango backwardation" },
  { href: "/spreads",    label: "Spreads & Cracks", hint: "Crack spreads", icon: <ArrowLeftRight size={15} />, keywords: "crack 3-2-1 ratio location" },
  { href: "/options",    label: "Options & Greeks", hint: "Black-76 pricer", icon: <Target size={15} />, keywords: "delta gamma vega black scholes call put" },
  { href: "/positions",  label: "Positions & P&L", hint: "Trade blotter", icon: <Briefcase size={15} />, keywords: "trade pnl mark to market blotter" },
  { href: "/risk",       label: "Risk Dashboard",  hint: "VaR / CVaR", icon: <Shield size={15} />, keywords: "var cvar stress shortfall expected" },
  { href: "/monte-carlo", label: "Monte Carlo",    hint: "Simulation", icon: <FlaskConical size={15} />, keywords: "simulation probabilistic shock path fan chart" },
  { href: "/events",     label: "Events",          hint: "Releases calendar", icon: <Calendar size={15} />, keywords: "fomc eia opec wasde nfp cpi" },
  { href: "/settings",   label: "Settings",        hint: "Preferences", icon: <Settings size={15} />, keywords: "preferences config theme account" },
];

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [hoverIdx, setHoverIdx] = useState(0);
  const router = useRouter();

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((o) => !o);
        setQ("");
        setHoverIdx(0);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const filtered = useMemo(() => {
    if (!q.trim()) return ITEMS;
    const t = q.toLowerCase();
    return ITEMS.filter(
      (it) =>
        it.label.toLowerCase().includes(t) ||
        it.hint.toLowerCase().includes(t) ||
        it.keywords.includes(t)
    );
  }, [q]);

  useEffect(() => {
    if (hoverIdx >= filtered.length) setHoverIdx(0);
  }, [filtered, hoverIdx]);

  const go = (href: string) => {
    router.push(href as any);
    setOpen(false);
  };

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHoverIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHoverIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && filtered[hoverIdx]) {
      go(filtered[hoverIdx].href);
    }
  };

  if (!open) return null;

  return (
    <div
      onClick={() => setOpen(false)}
      className="fixed inset-0 z-50 flex items-start justify-center pt-[18vh] bg-ink-900/70 backdrop-blur-sm animate-slide-up"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="glass w-[560px] max-w-[90vw] rounded-xl overflow-hidden shadow-2xl"
      >
        <div className="flex items-center gap-3 px-4 py-3 border-b border-ink-600">
          <Search size={16} className="text-ink-200" />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={onKey}
            placeholder="Jump to page or search…"
            className="flex-1 bg-transparent outline-none text-sm text-ink-50 placeholder:text-ink-300"
          />
          <kbd>esc</kbd>
        </div>
        <ul className="max-h-[50vh] overflow-y-auto py-1">
          {filtered.length === 0 ? (
            <li className="px-4 py-6 text-center text-sm text-ink-300">No match.</li>
          ) : (
            filtered.map((it, i) => (
              <li
                key={it.href}
                onMouseEnter={() => setHoverIdx(i)}
                onClick={() => go(it.href)}
                className={`flex items-center gap-3 px-4 py-2.5 cursor-pointer transition ${
                  hoverIdx === i ? "bg-ink-600/70" : "hover:bg-ink-600/40"
                }`}
              >
                <span className="text-ink-200">{it.icon}</span>
                <span className="flex-1 text-sm text-ink-50">{it.label}</span>
                <span className="text-[11px] text-ink-300">{it.hint}</span>
                {hoverIdx === i && <ArrowRight size={13} className="text-accent" />}
              </li>
            ))
          )}
        </ul>
        <div className="flex items-center gap-3 px-4 py-2 border-t border-ink-600 text-[10px] text-ink-300">
          <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>
          <span><kbd>⏎</kbd> open</span>
          <span className="ml-auto"><kbd>⌘</kbd><kbd>K</kbd> toggle</span>
        </div>
      </div>
    </div>
  );
}
