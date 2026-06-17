import { ComingSoon } from "@/components/ComingSoon";
export default function SettingsPage() {
  return (
    <ComingSoon
      title="Settings"
      description="Save / load assumptions, manage commodity templates, configure data sources."
      features={[
        "Export / import assumptions as JSON",
        "Browse and edit commodity templates",
        "Toggle between live (Yahoo / Refinitiv) and synthetic data",
        "User profile & preferences",
        "API key management for premium data feeds",
        "Theme: dark / light / auto",
      ]}
      hint="User-scoped settings will land in Postgres alongside positions once Clerk / Supabase auth is wired."
    />
  );
}
