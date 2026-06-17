import { ComingSoon } from "@/components/ComingSoon";
export default function EventsPage() {
  return (
    <ComingSoon
      title="Events & Reports"
      description="High-impact event calendar and downloadable daily desk report."
      features={[
        "EIA Weekly Petroleum Status (Wednesdays)",
        "USDA WASDE Report (mid-month)",
        "OPEC Monthly Oil Report",
        "IEA Oil Market Report",
        "FOMC, ECB, BoE rate decisions",
        "US Non-Farm Payrolls, CPI, PCE",
        "Daily desk report export (PDF / CSV / Excel)",
      ]}
      hint="Calendar will be sourced from a real feed (Trading Economics / Investing.com / MarketWatch) instead of the simplified recurring schedule used in the Streamlit prototype."
    />
  );
}
