import { ComingSoon } from "@/components/ComingSoon";
export default function CurvePage() {
  return (
    <ComingSoon
      title="Futures Curve"
      description="Forward curve from live Yahoo contracts (or synthetic fallback), calendar spreads and storage economics."
      features={[
        "Live forward curve from CME contracts (CL, NG, GC, SI, ZW, ZC, ZS, KC, SB...)",
        "Synthetic shape (contango / backwardation / flat) when offline",
        "Contract labels on the X axis (Jun-2026, Aug-2026...)",
        "Calendar spread board (m1-m2, m1-m6, m1-m12, m6-m12)",
        "Storage economics: contango premium vs monthly carry cost",
        "Stocks ↔ curve tightness score",
      ]}
      hint="The /api/curve/{key} endpoint returns the full curve already — Recharts line + table components remain."
    />
  );
}
