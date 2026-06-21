"use client";
import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useToast } from "./Toast";

/**
 * Vim-style two-key navigation:
 *   g d → /dashboard
 *   g b → /balance
 *   g r → /regional
 *   g c → /curve
 *   g s → /spreads
 *   g o → /options
 *   g p → /positions
 *   g m → /macro
 *   g k → /risk
 *   g e → /events
 *   g t → /settings
 *   g h → /
 *   ?   → keyboard cheatsheet
 *
 * Triggered only outside of inputs / textareas / selects.
 */
const SHORTCUTS: Record<string, { path: string; label: string }> = {
  d: { path: "/dashboard",   label: "Dashboard" },
  b: { path: "/balance",     label: "Supply & Demand" },
  r: { path: "/regional",    label: "Regional Flows" },
  c: { path: "/curve",       label: "Futures Curve" },
  s: { path: "/spreads",     label: "Spreads & Cracks" },
  o: { path: "/options",     label: "Options & Greeks" },
  p: { path: "/positions",   label: "Positions & P&L" },
  m: { path: "/macro",       label: "Macro Overlay" },
  k: { path: "/risk",        label: "Risk Dashboard" },
  e: { path: "/events",      label: "Events" },
  t: { path: "/settings",    label: "Settings" },
  h: { path: "/",            label: "Home" },
};

export function KeyboardShortcuts() {
  const router = useRouter();
  const { push } = useToast();
  const lastG = useRef<number>(0);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (!t) return;
      const tag = t.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      // Help
      if (e.key === "?") {
        push({
          tone: "info",
          title: "Keyboard shortcuts",
          message: "g+d Dashboard · g+o Options · g+p Positions · g+m Macro · g+k Risk · ⌘K search",
        });
        return;
      }

      // g + letter combo
      const now = performance.now();
      if (e.key === "g") {
        lastG.current = now;
        return;
      }
      if (now - lastG.current < 800) {
        const sc = SHORTCUTS[e.key.toLowerCase()];
        if (sc) {
          e.preventDefault();
          router.push(sc.path as any);
          push({ tone: "info", title: `→ ${sc.label}` });
          lastG.current = 0;
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [router, push]);

  return null;
}
