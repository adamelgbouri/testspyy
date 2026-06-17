"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, LineChart, Globe2, ArrowLeftRight,
  Target, Briefcase, Shield, FlaskConical, Calendar, Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Item = { href: string; label: string; icon: React.ReactNode };
type Group = { title: string; items: Item[] };

const GROUPS: Group[] = [
  {
    title: "Market Analytics",
    items: [
      { href: "/dashboard", label: "Dashboard", icon: <LayoutDashboard size={16} /> },
      { href: "/balance",   label: "Supply & Demand", icon: <LineChart size={16} /> },
      { href: "/regional",  label: "Regional Flows",  icon: <Globe2 size={16} /> },
    ],
  },
  {
    title: "Trading Desk",
    items: [
      { href: "/curve",    label: "Futures Curve",    icon: <LineChart size={16} /> },
      { href: "/spreads",  label: "Spreads & Cracks", icon: <ArrowLeftRight size={16} /> },
      { href: "/options",  label: "Options & Greeks", icon: <Target size={16} /> },
      { href: "/positions", label: "Positions & P&L", icon: <Briefcase size={16} /> },
    ],
  },
  {
    title: "Risk",
    items: [
      { href: "/risk",         label: "Risk Dashboard", icon: <Shield size={16} /> },
      { href: "/monte-carlo",  label: "Monte Carlo",    icon: <FlaskConical size={16} /> },
    ],
  },
  {
    title: "Tools",
    items: [
      { href: "/events",   label: "Events",   icon: <Calendar size={16} /> },
      { href: "/settings", label: "Settings", icon: <Settings size={16} /> },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="hidden lg:flex flex-col w-64 shrink-0 border-r border-ink-600 bg-ink-900 px-4 pb-6">
      <div className="flex items-center gap-2 py-5">
        <div className="w-7 h-7 rounded-md bg-gradient-to-br from-accent to-violet flex items-center justify-center font-bold text-ink-900">
          C
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-ink-50">Commodity Desk</div>
          <div className="text-[10px] text-ink-200 uppercase tracking-widest">trading platform</div>
        </div>
      </div>
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
                      className={cn("nav-link", active && "nav-link-active")}
                    >
                      {it.icon}
                      <span>{it.label}</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>
      <div className="text-[10px] text-ink-300 pt-4 border-t border-ink-600">
        v0.1 · data: yahoo + synthetic fallback
      </div>
    </aside>
  );
}
