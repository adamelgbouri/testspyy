import Link from "next/link";
import { Compass } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="card p-8 max-w-md w-full text-center">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-accent/10 text-accent mb-4">
          <Compass size={22} />
        </div>
        <div className="text-5xl font-bold text-ink-50 font-mono mb-2">404</div>
        <h1 className="text-lg font-semibold text-ink-50 mb-1">Off the chart</h1>
        <p className="text-sm text-ink-200 mb-5">
          That page is not in the desk&apos;s blotter. Maybe try one of these:
        </p>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <Link href="/dashboard" className="border border-ink-500 rounded-md py-2 hover:bg-ink-700">Dashboard</Link>
          <Link href="/balance" className="border border-ink-500 rounded-md py-2 hover:bg-ink-700">Supply &amp; Demand</Link>
          <Link href="/positions" className="border border-ink-500 rounded-md py-2 hover:bg-ink-700">Positions</Link>
          <Link href="/risk" className="border border-ink-500 rounded-md py-2 hover:bg-ink-700">Risk</Link>
        </div>
        <Link href="/" className="text-[11px] text-ink-300 hover:text-accent mt-5 inline-block">
          ← Back to home
        </Link>
      </div>
    </div>
  );
}
