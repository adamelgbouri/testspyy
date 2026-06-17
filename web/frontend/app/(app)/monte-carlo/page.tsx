import { ComingSoon } from "@/components/ComingSoon";
export default function MonteCarloPage() {
  return (
    <ComingSoon
      title="Monte Carlo"
      description="Probabilistic forecast under random supply, demand, weather and outage shocks."
      features={[
        "Configurable shock σ for supply, demand and weather",
        "Outage probability and size",
        "100 - 2 000 simulated paths",
        "Distribution histograms of average price, end stocks, build/draw",
        "Probabilistic fan charts (P5 / P50 / P95)",
        "VaR 95% on price drop from median",
      ]}
      hint="Heavy compute - will be run as a /api/montecarlo/{key} POST job, with a small job queue (Celery or RQ) for >1 000 paths."
    />
  );
}
