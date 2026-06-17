import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Trading-desk dark palette
        ink: {
          50:  "#f8fafc",
          100: "#e5e7eb",
          200: "#9ca3af",
          300: "#6b7280",
          400: "#4b5563",
          500: "#374151",
          600: "#1f2937",
          700: "#161b22",
          800: "#0e1117",
          900: "#0b0f14",
        },
        accent: { DEFAULT: "#00d4ff", muted: "#0891b2" },
        pos: "#22c55e",
        neg: "#ef4444",
        warn: "#f59e0b",
        violet: "#a78bfa",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        card: "0 1px 3px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.02)",
      },
    },
  },
  plugins: [],
};

export default config;
