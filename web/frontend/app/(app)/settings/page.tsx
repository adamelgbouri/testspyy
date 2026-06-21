"use client";
import { useEffect, useState } from "react";
import { api, type Commodity } from "@/lib/api";
import { useToast } from "@/components/Toast";
import { Check, Database, Globe, Trash2 } from "lucide-react";

type Prefs = {
  defaultCommodity: string;
  forecastHorizon: number;
  showTickerTape: boolean;
  liveData: boolean;
};

const PREFS_KEY = "trading_desk_prefs";
const POSITIONS_KEY = "trading_desk_positions";

const DEFAULTS: Prefs = {
  defaultCommodity: "wti_crude",
  forecastHorizon: 18,
  showTickerTape: true,
  liveData: true,
};

export default function SettingsPage() {
  const toast = useToast();
  const [commodities, setCommodities] = useState<Commodity[]>([]);
  const [prefs, setPrefs] = useState<Prefs>(DEFAULTS);
  const [apiStatus, setApiStatus] = useState<"ok" | "down" | "checking">("checking");
  const [positionsCount, setPositionsCount] = useState(0);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.commodities().then(setCommodities);
    fetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/health`)
      .then((r) => setApiStatus(r.ok ? "ok" : "down"))
      .catch(() => setApiStatus("down"));
    const raw = localStorage.getItem(PREFS_KEY);
    if (raw) {
      try { setPrefs({ ...DEFAULTS, ...JSON.parse(raw) }); } catch {}
    }
    const positions = localStorage.getItem(POSITIONS_KEY);
    if (positions) {
      try { setPositionsCount(JSON.parse(positions).length); } catch {}
    }
  }, []);

  const save = () => {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
    toast.push({
      tone: "success",
      title: "Preferences saved",
      message: "Your settings now persist in this browser.",
    });
  };

  const clearAllData = () => {
    if (confirm("This will erase all positions and reset preferences. Continue?")) {
      const n = positionsCount;
      localStorage.removeItem(POSITIONS_KEY);
      localStorage.removeItem(PREFS_KEY);
      setPrefs(DEFAULTS);
      setPositionsCount(0);
      toast.push({
        tone: "warning",
        title: "Local data cleared",
        message: `${n} position(s) and preferences removed.`,
      });
    }
  };

  const exportPositions = () => {
    const raw = localStorage.getItem(POSITIONS_KEY) ?? "[]";
    const blob = new Blob([raw], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "positions.json";
    a.click();
    URL.revokeObjectURL(url);
    toast.push({
      tone: "success",
      title: "Export ready",
      message: `${positionsCount} position(s) downloaded as positions.json`,
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-sm text-ink-200 mt-1">
          Configure your preferences, inspect the backend connection, manage
          local data.
        </p>
      </div>

      {/* Preferences */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Preferences</h2>
        <div className="grid sm:grid-cols-2 gap-4">
          <div>
            <label className="metric-label">Default commodity</label>
            <select value={prefs.defaultCommodity}
              onChange={(e) => setPrefs({ ...prefs, defaultCommodity: e.target.value })}
              className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm text-ink-50">
              {commodities.map((c) => (
                <option key={c.key} value={c.key}>[{c.sector}] {c.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="metric-label">Default forecast horizon (months)</label>
            <input type="number" min={6} max={36} step={3}
              value={prefs.forecastHorizon}
              onChange={(e) => setPrefs({ ...prefs, forecastHorizon: parseInt(e.target.value) })}
              className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm font-mono text-ink-50" />
          </div>
          <Toggle label="Show ticker tape" value={prefs.showTickerTape}
                  onChange={(v) => setPrefs({ ...prefs, showTickerTape: v })} />
          <Toggle label="Prefer live data (Yahoo)" value={prefs.liveData}
                  onChange={(v) => setPrefs({ ...prefs, liveData: v })}
                  hint="Off = reference values only (offline mode)" />
        </div>
        <div className="flex items-center gap-3 mt-5">
          <button onClick={save}
            className="bg-accent text-ink-900 font-medium rounded-md px-4 py-2 text-sm hover:bg-accent/90">
            Save preferences
          </button>
          {saved && (
            <span className="text-pos text-xs inline-flex items-center gap-1">
              <Check size={14} /> Saved
            </span>
          )}
        </div>
      </div>

      {/* Backend status */}
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-3">
          <Database size={14} className="text-accent" />
          <h2 className="text-sm font-semibold">Backend connection</h2>
        </div>
        <div className="space-y-2 text-sm">
          <Row label="API URL"
               value={process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"} />
          <Row label="Status"
               value={apiStatus === "ok" ? "online" : apiStatus === "down" ? "offline" : "checking…"}
               tone={apiStatus === "ok" ? "pos" : apiStatus === "down" ? "neg" : "neutral"} />
          <Row label="Commodities loaded" value={`${commodities.length}`} />
        </div>
      </div>

      {/* Data source */}
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-3">
          <Globe size={14} className="text-accent" />
          <h2 className="text-sm font-semibold">Data sources</h2>
        </div>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-ink-200 border-b border-ink-600">
              <th className="text-left pb-2">Source</th>
              <th className="text-left pb-2">Used for</th>
              <th className="text-left pb-2">Status</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-ink-700/60">
              <td className="py-2 text-ink-50">Yahoo Finance (yfinance)</td>
              <td className="text-ink-100">Spot prices, forward curves</td>
              <td>
                <span className="badge border-pos/40 text-pos">live · free</span>
              </td>
            </tr>
            <tr className="border-b border-ink-700/60">
              <td className="py-2 text-ink-50">Synthetic generator</td>
              <td className="text-ink-100">Balance series, fallback for offline</td>
              <td><span className="badge">always-on</span></td>
            </tr>
            <tr className="border-b border-ink-700/60">
              <td className="py-2 text-ink-50">Refinitiv / Bloomberg</td>
              <td className="text-ink-100">Production-grade real-time data</td>
              <td><span className="badge text-ink-300">not connected</span></td>
            </tr>
            <tr className="border-b border-ink-700/60">
              <td className="py-2 text-ink-50">Polygon.io / TradingView</td>
              <td className="text-ink-100">Affordable real-time alternative</td>
              <td><span className="badge text-ink-300">not connected</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Local data */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Local data</h2>
        <div className="text-sm text-ink-100 mb-4">
          You currently have <b className="font-mono">{positionsCount}</b>{" "}
          position(s) saved in your browser.
        </div>
        <div className="flex flex-wrap gap-2">
          <button onClick={exportPositions} disabled={positionsCount === 0}
            className="border border-ink-500 text-ink-100 rounded-md px-4 py-2 text-sm hover:bg-ink-600 disabled:opacity-50">
            Export positions (JSON)
          </button>
          <button onClick={clearAllData}
            className="border border-neg/40 text-neg rounded-md px-4 py-2 text-sm hover:bg-neg/10 inline-flex items-center gap-1.5">
            <Trash2 size={13} /> Clear all local data
          </button>
        </div>
      </div>

      <div className="card p-5 border-l-2 border-accent">
        <h3 className="text-sm font-semibold mb-2">Roadmap</h3>
        <ul className="text-xs text-ink-100 space-y-1.5 list-disc list-inside">
          <li>User accounts (Clerk / Supabase Auth)</li>
          <li>Postgres persistence for positions and preferences</li>
          <li>Refinitiv / Bloomberg / Polygon real-time feeds</li>
          <li>WebSocket live price streaming</li>
          <li>PDF desk reports + email scheduling</li>
        </ul>
      </div>
    </div>
  );
}

function Toggle({
  label, value, onChange, hint,
}: { label: string; value: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <div>
      <label className="metric-label">{label}</label>
      <button onClick={() => onChange(!value)}
        className={`mt-2 flex items-center gap-3 px-3 py-2 rounded-md border transition w-full ${
          value ? "bg-accent/10 border-accent/40 text-ink-50" : "border-ink-500 text-ink-200"
        }`}>
        <div className={`w-9 h-5 rounded-full p-0.5 transition ${
          value ? "bg-accent" : "bg-ink-500"
        }`}>
          <div className={`w-4 h-4 rounded-full bg-ink-50 transition ${
            value ? "ml-4" : ""
          }`} />
        </div>
        <span className="text-sm">{value ? "Enabled" : "Disabled"}</span>
      </button>
      {hint && <div className="text-[10px] text-ink-300 mt-1 italic">{hint}</div>}
    </div>
  );
}

function Row({ label, value, tone }: { label: string; value: string; tone?: "pos" | "neg" | "neutral" }) {
  return (
    <div className="flex justify-between">
      <span className="text-ink-200">{label}</span>
      <span className={`font-mono ${
        tone === "pos" ? "text-pos" : tone === "neg" ? "text-neg" : "text-ink-50"
      }`}>
        {value}
      </span>
    </div>
  );
}
