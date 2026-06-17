import { cn } from "@/lib/utils";
import { Sparkline } from "./Sparkline";
import { PulseDot } from "./PulseDot";

type Props = {
  label: string;
  value: string | number;
  delta?: string;
  deltaTone?: "pos" | "neg" | "neutral";
  sparkline?: number[];
  live?: boolean;
  icon?: React.ReactNode;
  className?: string;
};

/** Pro-grade KPI card with optional sparkline + live indicator. */
export function KPICard({
  label, value, delta, deltaTone = "neutral",
  sparkline, live, icon, className,
}: Props) {
  return (
    <div className={cn(
      "card px-4 py-3 group relative overflow-hidden animate-slide-up",
      className,
    )}>
      <div className="flex items-start justify-between gap-2">
        <div className="metric-label flex items-center gap-2">
          {icon && <span className="text-ink-300 group-hover:text-accent transition-colors">{icon}</span>}
          {label}
        </div>
        {live && <PulseDot tone={deltaTone === "neg" ? "neg" : "pos"} />}
      </div>
      <div className="flex items-end justify-between gap-2 mt-1">
        <div className="metric-value text-[1.05rem] leading-tight">{value}</div>
        {sparkline && sparkline.length > 1 && (
          <Sparkline data={sparkline} width={56} height={20} />
        )}
      </div>
      {delta && (
        <div className={cn(
          "text-[10px] mt-1 font-mono flex items-center gap-1",
          deltaTone === "pos" && "text-pos",
          deltaTone === "neg" && "text-neg",
          deltaTone === "neutral" && "text-ink-200",
        )}>
          {delta}
        </div>
      )}
      {/* Subtle hover glow */}
      <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-accent/0 to-violet/0 opacity-0 group-hover:opacity-10 transition-opacity pointer-events-none" />
    </div>
  );
}
