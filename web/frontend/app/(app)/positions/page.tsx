"use client";
import { useEffect, useState } from "react";
import { api, type Commodity, type Position } from "@/lib/api";
import { KPICard } from "@/components/KPICard";
import { useToast } from "@/components/Toast";
import { fmtNum, fmtPrice } from "@/lib/utils";
import { Plus, Trash2 } from "lucide-react";

const STORAGE_KEY = "trading_desk_positions";

type Mark = { price: number; source: string };

export default function PositionsPage() {
  const toast = useToast();
  const [commodities, setCommodities] = useState<Commodity[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [marks, setMarks] = useState<Record<string, Mark>>({});

  // Form state
  const [form, setForm] = useState<Position>({
    commodity_key: "wti_crude",
    direction: "Long",
    quantity: 100,
    entry_price: 70,
    entry_date: new Date().toISOString().slice(0, 10),
    notes: "",
  });

  // Load positions from localStorage on mount
  useEffect(() => {
    api.commodities().then(setCommodities);
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      try {
        setPositions(JSON.parse(raw));
      } catch {}
    }
  }, []);

  // Persist positions on change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(positions));
  }, [positions]);

  // Refresh marks whenever positions change
  useEffect(() => {
    const keys = Array.from(new Set(positions.map((p) => p.commodity_key)));
    Promise.all(keys.map((k) => api.spot(k).then((s) => [k, s] as const)))
      .then((arr) => {
        const m: Record<string, Mark> = {};
        for (const [k, s] of arr) m[k] = { price: s.price, source: s.source };
        setMarks(m);
      })
      .catch(() => {});
  }, [positions]);

  const handleAdd = () => {
    setPositions([...positions, { ...form }]);
    const c = commodities.find((x) => x.key === form.commodity_key);
    toast.push({
      tone: "success",
      title: "Position added",
      message: `${form.direction} ${form.quantity} × ${c?.name ?? form.commodity_key} @ ${form.entry_price}`,
    });
  };
  const handleRemove = (idx: number) => {
    const removed = positions[idx];
    setPositions(positions.filter((_, i) => i !== idx));
    if (removed) {
      const c = commodities.find((x) => x.key === removed.commodity_key);
      toast.push({
        tone: "info",
        title: "Position removed",
        message: `${c?.name ?? removed.commodity_key}`,
      });
    }
  };
  const handleClear = () => {
    if (confirm("Clear all positions?")) {
      const n = positions.length;
      setPositions([]);
      toast.push({ tone: "warning", title: "Blotter cleared", message: `${n} positions removed` });
    }
  };

  // Compute P&L
  const rows = positions.map((p) => {
    const mark = marks[p.commodity_key]?.price ?? p.entry_price;
    const sign = p.direction === "Long" ? 1 : -1;
    const pnlUnit = sign * (mark - p.entry_price);
    const pnlTotal = pnlUnit * p.quantity;
    const retPct = p.entry_price ? (pnlUnit / p.entry_price) * 100 : 0;
    const commodity = commodities.find((c) => c.key === p.commodity_key);
    return { ...p, mark, pnlUnit, pnlTotal, retPct, commodity };
  });

  const totalLong = rows.filter((r) => r.direction === "Long")
    .reduce((s, r) => s + r.mark * r.quantity, 0);
  const totalShort = rows.filter((r) => r.direction === "Short")
    .reduce((s, r) => s + r.mark * r.quantity, 0);
  const totalPnL = rows.reduce((s, r) => s + r.pnlTotal, 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Positions &amp; P&amp;L</h1>
        <p className="text-sm text-ink-200 mt-1">
          Trade blotter with live mark-to-market. Positions persist in your
          browser (localStorage) — no server-side storage yet.
        </p>
      </div>

      {/* Add form */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Add position</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-6 gap-3 items-end">
          <div className="lg:col-span-2">
            <label className="metric-label">Commodity</label>
            <select value={form.commodity_key}
              onChange={(e) => setForm({ ...form, commodity_key: e.target.value })}
              className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm text-ink-50">
              {commodities.map((c) => (
                <option key={c.key} value={c.key}>
                  [{c.sector}] {c.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="metric-label">Side</label>
            <select value={form.direction}
              onChange={(e) => setForm({ ...form, direction: e.target.value as "Long" | "Short" })}
              className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm text-ink-50">
              <option>Long</option><option>Short</option>
            </select>
          </div>
          <NumberField label="Quantity" value={form.quantity}
                       onChange={(v) => setForm({ ...form, quantity: v })} step={10} />
          <NumberField label="Entry price" value={form.entry_price}
                       onChange={(v) => setForm({ ...form, entry_price: v })} step={0.5} />
          <button onClick={handleAdd}
            className="bg-accent text-ink-900 font-medium rounded-md px-4 py-2 text-sm hover:bg-accent/90 inline-flex items-center justify-center gap-2">
            <Plus size={14} /> Add
          </button>
        </div>
      </div>

      {/* Summary */}
      {rows.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <KPICard label="Gross long ($)" value={fmtNum(totalLong, 0)} />
          <KPICard label="Gross short ($)" value={fmtNum(totalShort, 0)} />
          <KPICard label="Net exposure ($)"
                   value={`${totalLong - totalShort >= 0 ? "+" : ""}${fmtNum(totalLong - totalShort, 0)}`}
                   deltaTone={totalLong - totalShort >= 0 ? "pos" : "neg"} />
          <KPICard label="Total P&L ($)"
                   value={`${totalPnL >= 0 ? "+" : ""}${fmtNum(totalPnL, 0)}`}
                   deltaTone={totalPnL >= 0 ? "pos" : "neg"}
                   delta={`${rows.length} positions`} />
        </div>
      )}

      {/* Blotter */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold">Trade blotter</h2>
          {rows.length > 0 && (
            <button onClick={handleClear}
              className="text-xs text-neg hover:underline">Clear all</button>
          )}
        </div>
        {rows.length === 0 ? (
          <p className="text-sm text-ink-200 italic">
            No positions yet. Add one above to start.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-ink-200 border-b border-ink-600">
                  <th className="text-left pb-2">Commodity</th>
                  <th className="text-left pb-2">Side</th>
                  <th className="text-right pb-2">Qty</th>
                  <th className="text-right pb-2">Entry</th>
                  <th className="text-right pb-2">Mark</th>
                  <th className="text-right pb-2">P&L/unit</th>
                  <th className="text-right pb-2">P&L total</th>
                  <th className="text-right pb-2">Return %</th>
                  <th className="text-center pb-2"></th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {rows.map((r, i) => (
                  <tr key={i} className="border-b border-ink-700/60">
                    <td className="py-2 text-ink-50">{r.commodity?.name ?? r.commodity_key}</td>
                    <td>
                      <span className={`badge ${
                        r.direction === "Long" ? "border-pos/40 text-pos" : "border-neg/40 text-neg"
                      }`}>
                        {r.direction}
                      </span>
                    </td>
                    <td className="text-right">{fmtNum(r.quantity, 0)}</td>
                    <td className="text-right">{r.entry_price.toFixed(2)}</td>
                    <td className="text-right">{r.mark.toFixed(2)}</td>
                    <td className={`text-right ${r.pnlUnit >= 0 ? "text-pos" : "text-neg"}`}>
                      {r.pnlUnit >= 0 ? "+" : ""}{r.pnlUnit.toFixed(3)}
                    </td>
                    <td className={`text-right ${r.pnlTotal >= 0 ? "text-pos" : "text-neg"}`}>
                      {r.pnlTotal >= 0 ? "+" : ""}{fmtNum(r.pnlTotal, 0)}
                    </td>
                    <td className={`text-right ${r.retPct >= 0 ? "text-pos" : "text-neg"}`}>
                      {r.retPct >= 0 ? "+" : ""}{r.retPct.toFixed(1)} %
                    </td>
                    <td className="text-center">
                      <button onClick={() => handleRemove(i)}
                        className="text-ink-300 hover:text-neg">
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function NumberField({
  label, value, onChange, step,
}: { label: string; value: number; onChange: (v: number) => void; step: number }) {
  return (
    <div>
      <label className="metric-label">{label}</label>
      <input type="number" step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="mt-1 w-full bg-ink-700 border border-ink-500 rounded-md px-3 py-2 text-sm font-mono text-ink-50 focus:outline-none focus:ring-2 focus:ring-accent" />
    </div>
  );
}
