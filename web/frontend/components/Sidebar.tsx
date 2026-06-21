"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, LineChart, Globe2, ArrowLeftRight,
  Target, Briefcase, Shield, FlaskConical, Calendar, Settings, Landmark,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Item = { href: string; label: string; icon: React.ReactNode; live?: boolean };
type Group = { title: string; items: Item[] };

// `live: true` marks pages that are fully implemented; everything else
// renders a Coming-Soon placeholder.
const GROUPS: Group[] = [
  {
    title: "Market Analytics",
    items: [
      { href: "/dashboard", label: "Dashboard", icon: <LayoutDashboard size={16} />, live: true },
      { href: "/balance",   label: "Supply & Demand", icon: <LineChart size={16} />, live: true },
      { href: "/regional",  label: "Regional Flows",  icon: <Globe2 size={16} />, live: true },
      { href: "/macro",     label: "Macro Overlay",   icon: <Landmark size={16} />, live: true },
    ],
  },
  {
    title: "Trading Desk",
    items: [
      { href: "/curve",    label: "Futures Curve",    icon: <LineChart size={16} />, live: true },
      { href: "/spreads",  label: "Spreads & Cracks", icon: <ArrowLeftRight size={16} />, live: true },
      { href: "/options",  label: "Options & Greeks", icon: <Target size={16} />, live: true },
      { href: "/positions", label: "Positions & P&L", icon: <Briefcase size={16} />, live: true },
    ],
  },
  {
    title: "Risk",
    items: [
      { href: "/risk",         label: "Risk Dashboard", icon: <Shield size={16} />, live: true },
      { href: "/monte-carlo",  label: "Monte Carlo",    icon: <FlaskConical size={16} />, live: true },
    ],
  },
  {
    title: "Tools",
    items: [
      { href: "/events",   label: "Events",   icon: <Calendar size={16} />, live: true },
      { href: "/settings", label: "Settings", icon: <Settings size={16} />, live: true },
    ],
  },
];

export function Sidebar({ mobile = false }: { mobile?: boolean } = {}) {
  const pathname = usePathname();
  const wrapperClass = mobile
    ? "flex flex-col w-64 h-full border-r border-ink-600 bg-ink-900 px-4 pb-6"
    : "hidden lg:flex flex-col w-64 shrink-0 border-r border-ink-600 bg-ink-900 px-4 pb-6";
  return (
    <aside className={wrapperClass}>
      <Link href="/" className="flex items-center gap-2 py-5 hover:opacity-80">
        <div className="w-7 h-7 rounded-md bg-gradient-to-br from-accent to-violet flex items-center justify-center font-bold text-ink-900">
          C
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-ink-50">Commodity Desk</div>
          <div className="text-[10px] text-ink-200 uppercase tracking-widest">
            trading platform
          </div>
        </div>
      </Link>
      <nav className="flex-1 overflow-y-auto">
        {GROUPS.map((g) => (
          <div key={g.title} className="mb-1">
            <div className="section-title">{g.title}</div>
            <ul className="space-y-0.5">
              {g.items.map((it) => {
                const active = pathname === it.href;
                return (
                  <li key={it.href}>
                    <Link
                      href={it.href as any}
                      className={cn(
                        "nav-link group justify-between",
                        active && "nav-link-active",
                      )}
                    >
                      <span className="flex items-center gap-2">
                        {it.icon}
                        <span>{it.label}</span>
                      </span>
                      {it.live ? (
                        <span className="w-1.5 h-1.5 rounded-full bg-pos" title="Live" />
                      ) : (
                        <span className="text-[9px] uppercase tracking-wider text-ink-300 group-hover:text-ink-200">
                          soon
                        </span>
                      )}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>
      <div className="pt-4 border-t border-ink-600 space-y-1">
        <div className="text-[10px] text-ink-300">
          v0.2 · data: yahoo + synthetic fallback
        </div>
        <div className="text-[10px] text-ink-200 font-mono tracking-wider">
          by <span className="font-semibold text-ink-50">Adam EL GBOURI</span>
        </div>
      </div>
    </aside>
  );
}
