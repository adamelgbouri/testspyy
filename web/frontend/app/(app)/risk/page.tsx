import { ComingSoon } from "@/components/ComingSoon";
export default function RiskPage() {
  return (
    <ComingSoon
      title="Risk Dashboard"
      description="Portfolio VaR, CVaR, stress tests and risk-budget pie."
      features={[
        "Parametric and historical VaR at 95% / 99%",
        "CVaR (Expected Shortfall)",
        "Per-position risk decomposition",
        "Standard stress shocks (-5% / -10% / -20% / -35% / +10% / +25%)",
        "Risk-budget pie by sector and by commodity",
        "Risk-limit breach alerts",
      ]}
      hint="Will read positions from /api/positions once the Positions page is wired."
    />
  );
}
