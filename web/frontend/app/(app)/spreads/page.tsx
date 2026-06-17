import { ComingSoon } from "@/components/ComingSoon";
export default function SpreadsPage() {
  return (
    <ComingSoon
      title="Spreads & Cracks"
      description="Multi-leg structures: refiner margins, location spreads, calendar spreads."
      features={[
        "3-2-1 Crack (WTI → RBOB + ULSD) with live margin",
        "Simple crack (WTI → RBOB) and Brent → diesel",
        "Brent-WTI location spread",
        "Soybean crush (proxy)",
        "Calendar spread board from the loaded forward curve",
        "Spread P&L grid with size sliders",
      ]}
      hint="Needs a new /api/spreads/* endpoint built from the same crack-spread math already in the Streamlit prototype."
    />
  );
}
