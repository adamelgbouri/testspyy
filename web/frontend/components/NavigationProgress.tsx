"use client";
import { usePathname, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

function ProgressBar() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [animating, setAnimating] = useState(false);

  useEffect(() => {
    setAnimating(true);
    const t = window.setTimeout(() => setAnimating(false), 700);
    return () => window.clearTimeout(t);
  }, [pathname, searchParams]);

  if (!animating) return null;
  return (
    <div
      className="fixed top-0 left-0 right-0 z-[60] h-[2px] bg-gradient-to-r from-cyan via-accent to-cyan animate-nav-progress shadow-glow pointer-events-none"
    />
  );
}

/**
 * Vercel-style top progress bar that fires whenever the route changes.
 * Pure CSS keyframes, no external dependency. Wrapped in Suspense because
 * useSearchParams requires it during static rendering in Next 14.
 */
export function NavigationProgress() {
  return (
    <Suspense fallback={null}>
      <ProgressBar />
    </Suspense>
  );
}
