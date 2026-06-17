import { ComingSoon } from "@/components/ComingSoon";
export default function BalancePage() {
  return (
    <ComingSoon
      title="Supply & Demand Balance"
      description="Build a commodity balance, apply assumptions and watch how stocks evolve."
      features={[
        "Monthly / quarterly / yearly balance with editable assumptions",
        "Build / draw waterfall and days-of-cover trajectory",
        "Seasonality decomposition (trend / seasonal / residual)",
        "Price elasticity curves with analytical equilibrium",
        "Lagged supply response simulator",
        "CSV / Excel export of the full balance table",
      ]}
      hint="The /api/balance/{key} endpoint already returns the full series — this page only needs the chart components ported from Streamlit."
    />
  );
}
