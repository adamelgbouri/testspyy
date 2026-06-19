"use client";
import { useEffect } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

export default function AppError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("App error boundary caught:", error);
  }, [error]);

  return (
    <div className="min-h-[60vh] flex items-center justify-center px-4">
      <div className="card p-8 max-w-md w-full text-center">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-neg/10 text-neg mb-4">
          <AlertTriangle size={22} />
        </div>
        <h1 className="text-lg font-bold text-ink-50 mb-2">Something went wrong</h1>
        <p className="text-sm text-ink-200 mb-1">
          The dashboard hit an unexpected error. The backend may be cold-starting
          (Render free tier sleeps after 15 min of inactivity).
        </p>
        {error.digest && (
          <p className="text-[10px] text-ink-300 font-mono mb-4">
            ref: {error.digest}
          </p>
        )}
        <div className="flex flex-wrap gap-2 justify-center mt-4">
          <button
            onClick={reset}
            className="bg-accent text-ink-900 font-medium rounded-md px-4 py-2 text-sm hover:bg-accent/90 inline-flex items-center gap-2"
          >
            <RefreshCw size={14} /> Try again
          </button>
          <a
            href="/"
            className="border border-ink-500 text-ink-100 rounded-md px-4 py-2 text-sm hover:bg-ink-600"
          >
            Go home
          </a>
        </div>
      </div>
    </div>
  );
}
