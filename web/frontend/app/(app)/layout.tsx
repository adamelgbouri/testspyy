import { Sidebar } from "@/components/Sidebar";
import { TickerTape } from "@/components/TickerTape";
import { StatusBar } from "@/components/StatusBar";
import { CommandPalette } from "@/components/CommandPalette";
import { ToastProvider } from "@/components/Toast";
import { NavigationProgress } from "@/components/NavigationProgress";
import { MobileNav } from "@/components/MobileNav";
import { KeyboardShortcuts } from "@/components/KeyboardShortcuts";

/** Shared app shell — sidebar (or mobile drawer), ticker, status bar, toasts. */
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <ToastProvider>
      <NavigationProgress />
      <KeyboardShortcuts />
      <div className="flex min-h-screen pb-7 overflow-x-hidden">
        <Sidebar />
        <MobileNav />
        <div className="flex-1 flex flex-col min-w-0">
          {/* @ts-expect-error – async server component */}
          <TickerTape />
          <main className="flex-1 p-4 sm:p-6 max-w-[1400px] w-full mx-auto">
            {children}
          </main>
        </div>
        <CommandPalette />
        <StatusBar />
      </div>
    </ToastProvider>
  );
}
