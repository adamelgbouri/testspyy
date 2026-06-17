import Link from "next/link";
import { ArrowRight, BarChart3, Calculator, Globe2, Shield } from "lucide-react";

const FEATURES = [
  {
    icon: <BarChart3 size={20} />,
    title: "Supply & Demand Engine",
    desc: "Build commodity balances, project inventory, surface fair value.",
  },
  {
    icon: <Globe2 size={20} />,
    title: "Regional Flows",
    desc: "World production vs consumption with implied trade flows and shares.",
  },
  {
    icon: <Calculator size={20} />,
    title: "Options & Pricing",
    desc: "Black-76 Greeks, implied vol back-solve, multi-leg strategy payoffs.",
  },
  {
    icon: <Shield size={20} />,
    title: "Risk & Positioning",
    desc: "VaR, CVaR, stress tests and a live mark-to-market P&L blotter.",
  },
];

export default function Landing() {
  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="border-b border-ink-600 bg-ink-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-gradient-to-br from-accent to-violet flex items-center justify-center font-bold text-ink-900">
              C
            </div>
            <span className="font-semibold tracking-tight">Commodity Trading Desk</span>
          </div>
          <Link
            href="/dashboard"
            className="text-sm bg-accent text-ink-900 font-medium rounded-md px-4 py-1.5 hover:bg-accent/90 transition"
          >
            Launch app →
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-24 pb-20">
        <div className="text-center max-w-3xl mx-auto">
          <span className="badge mb-6">v0.1 · early access</span>
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6 leading-[1.05]">
            The analytics platform for{" "}
            <span className="bg-gradient-to-r from-accent to-violet bg-clip-text text-transparent">
              commodity trading desks
            </span>
          </h1>
          <p className="text-ink-200 text-lg leading-relaxed mb-10">
            Supply &amp; demand balances, term structure, regional flows, options pricing,
            VaR and a live P&amp;L blotter — all in one professional dashboard.
          </p>
          <div className="flex items-center justify-center gap-3">
            <Link
              href="/dashboard"
              className="bg-accent text-ink-900 font-medium rounded-md px-6 py-3 hover:bg-accent/90 transition inline-flex items-center gap-2"
            >
              Open the dashboard <ArrowRight size={16} />
            </Link>
            <Link
              href="/options"
              className="border border-ink-500 text-ink-100 rounded-md px-6 py-3 hover:bg-ink-600 transition"
            >
              Try the options pricer
            </Link>
          </div>
        </div>
      </section>

      {/* Feature grid */}
      <section className="max-w-6xl mx-auto px-6 pb-24">
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURES.map((f) => (
            <div key={f.title} className="card p-5">
              <div className="w-9 h-9 rounded-lg bg-ink-600 flex items-center justify-center text-accent mb-3">
                {f.icon}
              </div>
              <h3 className="font-semibold text-ink-50 mb-1">{f.title}</h3>
              <p className="text-sm text-ink-200 leading-snug">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="border-t border-ink-600 py-8 text-center">
        <div className="text-xs text-ink-300 mb-2">
          Built with Next.js + FastAPI · live data via Yahoo Finance · synthetic fallback when offline
        </div>
        <div className="text-sm text-ink-100 font-mono">
          designed &amp; built by{" "}
          <span className="font-bold bg-gradient-to-r from-accent to-violet bg-clip-text text-transparent">
            Adam EL GBOURI
          </span>
        </div>
      </footer>
    </main>
  );
}
