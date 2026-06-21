/**
 * Per-route template — re-runs on every navigation, used to fire the
 * fade-in animation cleanly across the App Router.
 */
export default function PageTemplate({ children }: { children: React.ReactNode }) {
  return <div className="page-transition">{children}</div>;
}
