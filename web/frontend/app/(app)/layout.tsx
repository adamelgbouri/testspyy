import { Sidebar } from "@/components/Sidebar";
import { TickerTape } from "@/components/TickerTape";
import { StatusBar } from "@/components/StatusBar";
import { CommandPalette } from "@/components/CommandPalette";

/** Shared app shell: sidebar + ticker + status bar + global Cmd+K. */
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen pb-7">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        {/* @ts-expect-error – async server component */}
        <TickerTape />
        <main className="flex-1 p-6 max-w-[1400px] w-full mx-auto">{children}</main>
      </div>
      <CommandPalette />
      <StatusBar />
    </div>
  );
}
