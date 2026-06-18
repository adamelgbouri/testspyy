import type { Config } from "tailwindcss";

/**
 * Trading-floor palette — Bloomberg-Terminal-meets-modern-web.
 *   - Deep midnight-blue background (almost black, never pure black)
 *   - Subtle navy layers for depth
 *   - Amber accent for highlights & live ticks
 *   - Cyan accent for interactive elements
 *   - Green / red for P&L
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          50:  "#f0f4fa",
          100: "#dbe4f0",
          200: "#97a8be",
          300: "#6c809b",
          400: "#4a5f7e",
          500: "#2c4564",     // border bright
          600: "#1f3553",     // border default
          700: "#152641",     // panels elevated
          800: "#0e1f33",     // cards
          900: "#0a1628",     // canvas
        },
        // Bloomberg-style accents
        accent: { DEFAULT: "#ffb800", muted: "#cc9400" },    // amber for highlights
        cyan:   { DEFAULT: "#00d4ff", muted: "#0891b2" },
        pos:    "#00d18c",
        neg:    "#ff4757",
        warn:   "#ffb800",
        violet: "#a78bfa",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        card: "0 1px 0 rgba(0,212,255,0.04) inset, 0 1px 8px rgba(0,0,0,0.4)",
        glow: "0 0 24px rgba(255,184,0,0.15)",
      },
      backgroundImage: {
        "grid-pattern":
          "linear-gradient(to right, rgba(31,53,83,0.25) 1px, transparent 1px), linear-gradient(to bottom, rgba(31,53,83,0.25) 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
};

export default config;
