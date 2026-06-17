import { cn } from "@/lib/utils";

type Props = {
  label: string;
  value: string | number;
  delta?: string;
  deltaTone?: "pos" | "neg" | "neutral";
  className?: string;
};

export function KPICard({ label, value, delta, deltaTone = "neutral", className }: Props) {
  return (
    <div className={cn("card px-4 py-3", className)}>
      <div className="metric-label">{label}</div>
      <div className="metric-value text-lg mt-1">{value}</div>
      {delta && (
        <div
          className={cn(
            "text-[11px] mt-1 font-mono",
            deltaTone === "pos" && "text-pos",
            deltaTone === "neg" && "text-neg",
            deltaTone === "neutral" && "text-ink-200",
          )}
        >
          {delta}
        </div>
      )}
    </div>
  );
}
