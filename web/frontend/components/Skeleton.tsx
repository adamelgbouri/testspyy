import { cn } from "@/lib/utils";

/** Animated shimmer placeholder. */
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton", className)} />;
}

export function KPICardSkeleton() {
  return (
    <div className="card px-4 py-3">
      <Skeleton className="h-2.5 w-20 mb-3" />
      <Skeleton className="h-5 w-28 mb-2" />
      <Skeleton className="h-2 w-16" />
    </div>
  );
}

export function ChartSkeleton({ height = 320 }: { height?: number }) {
  return (
    <div className="card p-5">
      <Skeleton className="h-3 w-40 mb-4" />
      <div className="relative" style={{ height }}>
        <Skeleton className="absolute inset-0" />
        {/* fake gridlines */}
        <div className="absolute inset-0 flex flex-col justify-between opacity-30 pointer-events-none">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-px bg-ink-500" />
          ))}
        </div>
      </div>
    </div>
  );
}

export function TableSkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <div className="card p-5">
      <Skeleton className="h-3 w-32 mb-4" />
      <div className="space-y-2.5">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex gap-3">
            <Skeleton className="h-3 w-1/4" />
            <Skeleton className="h-3 w-1/6" />
            <Skeleton className="h-3 w-1/6" />
            <Skeleton className="h-3 w-1/6" />
            <Skeleton className="h-3 w-1/6" />
          </div>
        ))}
      </div>
    </div>
  );
}
