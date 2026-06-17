"use client";
import { useEffect, useMemo, useState } from "react";
import { api, type EventRow } from "@/lib/api";
import { Calendar, Filter } from "lucide-react";

const TAG_COLORS: Record<string, string> = {
  Energy: "border-warn/40 text-warn",
  Ags:    "border-pos/40 text-pos",
  Metals: "border-violet/40 text-violet",
  Macro:  "border-accent/40 text-accent",
};

export default function EventsPage() {
  const [events, setEvents] = useState<EventRow[]>([]);
  const [tagFilter, setTagFilter] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.events()
      .then(setEvents)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const allTags = useMemo(() => {
    const s = new Set<string>();
    events.forEach((e) => e.tags.forEach((t) => s.add(t)));
    return Array.from(s).sort();
  }, [events]);

  const filtered = useMemo(() => {
    if (tagFilter.length === 0) return events;
    return events.filter((e) => e.tags.some((t) => tagFilter.includes(t)));
  }, [events, tagFilter]);

  // Group by week
  const grouped = useMemo(() => {
    const m = new Map<string, EventRow[]>();
    filtered.forEach((e) => {
      const d = new Date(e.date);
      const monday = new Date(d);
      monday.setDate(d.getDate() - ((d.getDay() + 6) % 7));
      const key = monday.toISOString().slice(0, 10);
      const arr = m.get(key) ?? [];
      arr.push(e);
      m.set(key, arr);
    });
    return Array.from(m.entries()).sort();
  }, [filtered]);

  const toggle = (tag: string) =>
    setTagFilter((f) => f.includes(tag) ? f.filter((x) => x !== tag) : [...f, tag]);

  const today = new Date().toISOString().slice(0, 10);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Events &amp; Reports</h1>
        <p className="text-sm text-ink-200 mt-1">
          High-impact data releases and central-bank meetings for the next ~6 weeks.
        </p>
      </div>

      {error && <div className="card p-4 border-neg text-neg text-sm">{error}</div>}

      <div className="card p-5">
        <div className="flex items-center gap-2 mb-3">
          <Filter size={14} className="text-accent" />
          <h2 className="text-sm font-semibold">Filter by tag</h2>
          {tagFilter.length > 0 && (
            <button onClick={() => setTagFilter([])}
              className="text-xs text-ink-200 hover:text-ink-50 ml-2">Clear</button>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          {allTags.map((t) => {
            const active = tagFilter.includes(t);
            return (
              <button key={t} onClick={() => toggle(t)}
                className={`badge cursor-pointer ${
                  active ? TAG_COLORS[t] ?? "border-accent/40 text-accent"
                         : "opacity-50 hover:opacity-100"
                }`}>
                {t}
              </button>
            );
          })}
        </div>
      </div>

      {loading && <p className="text-sm text-ink-200">Loading…</p>}

      {grouped.map(([weekStart, evs]) => (
        <div key={weekStart} className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <Calendar size={14} className="text-ink-200" />
            <h3 className="text-sm font-semibold text-ink-100">
              Week of {formatWeek(weekStart)}
            </h3>
            <span className="text-xs text-ink-300">{evs.length} event(s)</span>
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-ink-200 border-b border-ink-600">
                <th className="text-left pb-2 w-32">Date</th>
                <th className="text-left pb-2">Event</th>
                <th className="text-left pb-2 w-32">Tags</th>
                <th className="text-left pb-2 w-28">Frequency</th>
              </tr>
            </thead>
            <tbody>
              {evs.map((e, i) => (
                <tr key={i} className={`border-b border-ink-700/60 ${
                  e.date === today ? "bg-ink-600/30" : ""
                }`}>
                  <td className="py-2 font-mono text-ink-100">
                    {formatDate(e.date)}
                    {e.date === today && (
                      <span className="ml-2 text-[9px] text-accent uppercase">today</span>
                    )}
                  </td>
                  <td className="text-ink-50">{e.event}</td>
                  <td>
                    <div className="flex gap-1 flex-wrap">
                      {e.tags.map((t) => (
                        <span key={t} className={`badge ${TAG_COLORS[t] ?? ""}`}>{t}</span>
                      ))}
                    </div>
                  </td>
                  <td className="text-ink-200">{e.frequency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}

      {!loading && grouped.length === 0 && (
        <p className="text-sm text-ink-200 italic">No events matching the filter.</p>
      )}
    </div>
  );
}

function formatDate(s: string): string {
  const d = new Date(s);
  return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
}

function formatWeek(s: string): string {
  const d = new Date(s);
  return d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
}
