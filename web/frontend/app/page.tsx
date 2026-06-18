import Link from "next/link";
import {
  ArrowRight, BarChart3, Calculator, Globe2, Shield,
  Github, Linkedin, Mail, ExternalLink, Sparkles, Activity,
} from "lucide-react";

const FEATURES = [
  {
    icon: <BarChart3 size={20} />,
    title: "Supply & Demand Engine",
    desc: "Build commodity balances with editable assumptions, surface fair value via regression on inventory days-of-cover.",
  },
  {
    icon: <Globe2 size={20} />,
    title: "Regional Trade Flows",
    desc: "World production vs consumption with implied trade flows, top exporter/importer call-outs.",
  },
  {
    icon: <Calculator size={20} />,
    title: "Options & Greeks",
    desc: "Black-76 European pricer with full Greeks, implied vol back-solve, payoff visualiser, 3D volatility surface.",
  },
  {
    icon: <Shield size={20} />,
    title: "Risk & Positioning",
    desc: "Parametric and historical VaR / CVaR, stress tests, live MTM P&L blotter, multi-position aggregation.",
  },
];

const STACK = [
  { label: "Frontend",  value: "Next.js 14 · TypeScript · Tailwind · Recharts" },
  { label: "Backend",   value: "FastAPI · Pydantic v2 · Uvicorn" },
  { label: "Analytics", value: "NumPy · Pandas · SciPy · scikit-learn" },
  { label: "Data",      value: "Yahoo Finance (yfinance) · synthetic fallback" },
  { label: "Design",    value: "Custom dark trading-floor system · live-data flashes" },
];

const PAGES = [
  "Market Dashboard", "Supply & Demand", "Regional Flows", "Macro Overlay",
  "Futures Curve", "Spreads & Cracks", "Options & Greeks", "Positions & P&L",
  "Risk Dashboard", "Monte Carlo", "Events Calendar",
];

export default function Landing() {
  return (
    <main className="min-h-screen">
      {/* Top bar */}
      <header className="border-b border-ink-600 bg-ink-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-md bg-gradient-to-br from-accent via-orange-500 to-cyan flex items-center justify-center font-bold text-ink-900 text-sm shadow-glow">
              C
            </div>
            <div className="leading-tight">
              <div className="text-sm font-bold text-ink-50">Commodity Trading Desk</div>
              <div className="text-[10px] text-ink-300 uppercase tracking-widest font-mono">
                by Adam EL GBOURI
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <a
              href="https://github.com/adamelgbouri/testspyy"
              target="_blank" rel="noopener noreferrer"
              className="text-ink-200 hover:text-ink-50 transition p-2"
              aria-label="GitHub"
            >
              <Github size={18} />
            </a>
            <Link
              href="/dashboard"
              className="text-sm bg-accent text-ink-900 font-semibold rounded-md px-4 py-1.5 hover:bg-accent/90 transition shadow-glow"
            >
              Launch app →
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-20 pb-16">
        <div className="max-w-3xl mx-auto text-center">
          <div className="flex items-center justify-center gap-2 mb-6">
            <span className="pulse-dot pulse-accent" />
            <span className="badge border-accent/40 text-accent">PORTFOLIO PROJECT · 2025</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6 leading-[1.05]">
            A commodity trading desk{" "}
            <span className="bg-gradient-to-r from-accent via-orange-500 to-cyan bg-clip-text text-transparent">
              built from scratch
            </span>
          </h1>
          <p className="text-ink-200 text-lg leading-relaxed mb-10">
            11 interactive pages — supply &amp; demand balances, regional flows,
            Black-76 options, VaR risk, Monte Carlo, live mark-to-market — wired
            to a Python analytics backend and rendered in a Bloomberg-style dark UI.
          </p>
          <div className="flex items-center justify-center gap-3 flex-wrap">
            <Link
              href="/dashboard"
              className="bg-accent text-ink-900 font-semibold rounded-md px-6 py-3 hover:bg-accent/90 transition inline-flex items-center gap-2 shadow-glow"
            >
              Open the dashboard <ArrowRight size={16} />
            </Link>
            <Link
              href="/options"
              className="border border-ink-500 text-ink-100 rounded-md px-6 py-3 hover:bg-ink-700 transition"
            >
              Try the options pricer
            </Link>
            <a
              href="https://github.com/adamelgbouri/testspyy"
              target="_blank" rel="noopener noreferrer"
              className="text-ink-300 hover:text-ink-100 transition text-sm inline-flex items-center gap-1.5 px-2 py-3"
            >
              <Github size={14} /> source code
            </a>
          </div>
        </div>
      </section>

      {/* Live tickers preview */}
      <section className="border-y border-ink-600 bg-ink-800/40 py-3">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex items-center gap-2 mb-2 text-[10px] font-mono tracking-widest text-ink-300">
            <Activity size={11} className="text-accent" /> LIVE BENCHMARKS (synthetic preview)
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 text-xs font-mono">
            {[
              { sym: "WTI",  px: "70.42", chg: "+1.2%", up: true },
              { sym: "BRENT", px: "74.18", chg: "+0.9%", up: true },
              { sym: "NG",   px: "3.21",  chg: "-2.1%", up: false },
              { sym: "GOLD", px: "2,648", chg: "+0.4%", up: true },
              { sym: "CU",   px: "4.42",  chg: "-0.3%", up: false },
              { sym: "ZW",   px: "562",   chg: "+1.8%", up: true },
            ].map((t) => (
              <div key={t.sym} className="flex items-center justify-between gap-1.5 bg-ink-900/50 border border-ink-600 rounded px-2 py-1.5">
                <span className="text-ink-300 text-[10px]">{t.sym}</span>
                <span className="text-ink-50 font-bold">{t.px}</span>
                <span className={`${t.up ? "text-pos" : "text-neg"} font-bold`}>
                  {t.up ? "▲" : "▼"} {t.chg}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Feature grid */}
      <section className="max-w-6xl mx-auto px-6 py-20">
        <h2 className="text-2xl font-bold mb-2 text-center">What's inside</h2>
        <p className="text-ink-200 text-center mb-10 text-sm">
          Real analytics — not a static mockup.
        </p>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
          {FEATURES.map((f) => (
            <div key={f.title} className="card p-5 group">
              <div className="w-10 h-10 rounded-lg bg-ink-700 flex items-center justify-center text-accent mb-3 group-hover:bg-accent/10 transition">
                {f.icon}
              </div>
              <h3 className="font-semibold text-ink-50 mb-1">{f.title}</h3>
              <p className="text-sm text-ink-200 leading-snug">{f.desc}</p>
            </div>
          ))}
        </div>

        {/* Pages chip */}
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles size={14} className="text-accent" />
            <h3 className="text-sm font-semibold">Pages you can explore</h3>
            <span className="badge ml-auto">{PAGES.length} TOTAL</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {PAGES.map((p) => (
              <span key={p} className="badge">{p}</span>
            ))}
          </div>
        </div>
      </section>

      {/* Tech stack */}
      <section className="max-w-6xl mx-auto px-6 pb-20">
        <h2 className="text-2xl font-bold mb-6">Tech stack</h2>
        <div className="card p-6">
          <table className="w-full text-sm">
            <tbody>
              {STACK.map((s) => (
                <tr key={s.label} className="border-b border-ink-600 last:border-0">
                  <td className="py-2.5 pr-6 metric-label whitespace-nowrap">{s.label}</td>
                  <td className="py-2.5 font-mono text-ink-50">{s.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* About / contact */}
      <section className="max-w-6xl mx-auto px-6 pb-24">
        <div className="card-glow p-8 text-center">
          <div className="inline-block bg-gradient-to-br from-accent via-orange-500 to-cyan p-[1px] rounded-full mb-4">
            <div className="bg-ink-900 rounded-full w-14 h-14 flex items-center justify-center text-2xl font-bold text-ink-50">
              AE
            </div>
          </div>
          <h2 className="text-2xl font-bold mb-2">Adam EL GBOURI</h2>
          <p className="text-ink-200 text-sm max-w-xl mx-auto mb-6">
            Solo-built portfolio project showcasing modern React/Next.js, Python
            backend architecture, financial analytics and trading-desk UX. Reach
            out if you'd like to discuss the work.
          </p>
          <div className="flex items-center justify-center gap-3 flex-wrap">
            <a
              href="https://github.com/adamelgbouri"
              target="_blank" rel="noopener noreferrer"
              className="border border-ink-500 text-ink-100 rounded-md px-4 py-2 text-sm hover:bg-ink-700 transition inline-flex items-center gap-2"
            >
              <Github size={14} /> GitHub
              <ExternalLink size={11} className="opacity-50" />
            </a>
            <a
              href="https://linkedin.com/"
              target="_blank" rel="noopener noreferrer"
              className="border border-ink-500 text-ink-100 rounded-md px-4 py-2 text-sm hover:bg-ink-700 transition inline-flex items-center gap-2"
            >
              <Linkedin size={14} /> LinkedIn
              <ExternalLink size={11} className="opacity-50" />
            </a>
            <a
              href="mailto:contact@example.com"
              className="border border-ink-500 text-ink-100 rounded-md px-4 py-2 text-sm hover:bg-ink-700 transition inline-flex items-center gap-2"
            >
              <Mail size={14} /> Email
            </a>
          </div>
        </div>
      </section>

      <footer className="border-t border-ink-600 py-6 text-center">
        <div className="text-[10px] text-ink-300 font-mono tracking-wider">
          BUILT WITH NEXT.JS · FASTAPI · YAHOO FINANCE · 2025
        </div>
      </footer>
    </main>
  );
}
