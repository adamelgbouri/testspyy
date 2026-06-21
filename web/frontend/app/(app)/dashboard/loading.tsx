import { KPICardSkeleton, ChartSkeleton } from "@/components/Skeleton";

export default function DashboardLoading() {
  return (
    <div className="space-y-6">
      <div>
        <div className="skeleton h-7 w-64 mb-2" />
        <div className="skeleton h-3 w-96" />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {Array.from({ length: 4 }).map((_, i) => <KPICardSkeleton key={i} />)}
      </div>
      <div className="grid lg:grid-cols-3 gap-3">
        <ChartSkeleton height={300} />
        <ChartSkeleton height={300} />
        <ChartSkeleton height={300} />
      </div>
      <ChartSkeleton height={360} />
    </div>
  );
}
