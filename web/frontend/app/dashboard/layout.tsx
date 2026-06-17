import { Sidebar } from "@/components/Sidebar";
import { TickerTape } from "@/components/TickerTape";

/**
 * Shared layout for every app route (dashboard, balance, options…).
 * Sidebar on the left, ticker tape on top, page content on the right.
 */
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        {/* @ts-expect-error – async server component */}
        <TickerTape />
        <main className="flex-1 p-6 max-w-[1400px] w-full mx-auto">{children}</main>
      </div>
    </div>
  );
}
