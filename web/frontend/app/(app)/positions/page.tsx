import { ComingSoon } from "@/components/ComingSoon";
export default function PositionsPage() {
  return (
    <ComingSoon
      title="Positions & P&L"
      description="Interactive trade blotter with live mark-to-market P&L."
      features={[
        "Add positions (commodity, side, quantity, entry price, notes)",
        "Mark-to-market refreshed from live spots",
        "Per-line P&L per unit, total P&L, return %",
        "Aggregated gross long, gross short, net exposure",
        "P&L attribution by sector and by commodity",
        "Persistence in Postgres (Supabase) once auth is wired",
      ]}
      hint="Will need a Postgres table + auth before going live. Today's blotter is session-bound in the Streamlit prototype."
    />
  );
}
