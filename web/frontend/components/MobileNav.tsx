"use client";
import { useEffect, useState } from "react";
import { Menu, X } from "lucide-react";
import { Sidebar } from "@/components/Sidebar";

/**
 * Mobile hamburger that slides out the same Sidebar as a drawer.
 * Hidden on lg+ where the Sidebar is permanently visible.
 */
export function MobileNav() {
  const [open, setOpen] = useState(false);

  // Close on route change (detect via scroll trick) — and on escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <>
      <button
        aria-label="Open menu"
        onClick={() => setOpen(true)}
        className="lg:hidden fixed top-3 left-3 z-30 bg-ink-700 border border-ink-500 rounded-md p-2 text-ink-100 hover:bg-ink-600 transition"
      >
        <Menu size={18} />
      </button>

      {open && (
        <div
          onClick={() => setOpen(false)}
          className="lg:hidden fixed inset-0 z-40 bg-ink-900/80 backdrop-blur-sm animate-slide-up"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="h-full w-64 bg-ink-900 border-r border-ink-600 shadow-xl"
            style={{ animation: "slide-in-left 0.25s ease both" }}
          >
            <button
              aria-label="Close"
              onClick={() => setOpen(false)}
              className="absolute top-3 right-3 text-ink-200 hover:text-ink-50 p-1.5"
            >
              <X size={18} />
            </button>
            <Sidebar mobile />
          </div>
        </div>
      )}
    </>
  );
}
