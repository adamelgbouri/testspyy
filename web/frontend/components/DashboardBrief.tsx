import { Sparkles, AlertTriangle, TrendingUp, TrendingDown } from "lucide-react";

type Props = {
  commodityName: string;
  price: number;
  priceUnit: string;
  changePct: number;
  fairValue: number;
  fvDeviation: number;
  daysCover: number;
  daysCoverTarget: number;
  utilizationPct: number;
  idealUtilization: number;
};

/**
 * Auto-generated "trader brief" card — turns the same numbers shown on the
 * KPI strip into a short narrative paragraph so non-experts get the story.
 */
export function DashboardBrief({
  commodityName, price, priceUnit, changePct, fairValue,
  fvDeviation, daysCover, daysCoverTarget, utilizationPct, idealUtilization,
}: Props) {
  const direction = changePct > 0 ? "up" : changePct < 0 ? "down" : "flat";
  const tone = changePct > 0 ? "text-pos" : changePct < 0 ? "text-neg" : "text-ink-100";

  // Valuation flag
  const fvFlag = fvDeviation > 12 ? "rich" : fvDeviation < -12 ? "cheap" : "fairly priced";
  const fvTone = fvDeviation > 12 ? "text-neg" : fvDeviation < -12 ? "text-pos" : "text-accent";

  // Inventory flag
  const dcDelta = daysCover - daysCoverTarget;
  const dcFlag = dcDelta < -daysCoverTarget * 0.15 ? "tight" :
                 dcDelta > daysCoverTarget * 0.15 ? "well-supplied" : "balanced";
  const dcTone = dcDelta < -daysCoverTarget * 0.15 ? "text-neg" :
                 dcDelta > daysCoverTarget * 0.15 ? "text-pos" : "text-accent";

  // Risk callouts
  const alerts: string[] = [];
  if (Math.abs(fvDeviation) > 20) alerts.push(`Spot trades ${Math.abs(fvDeviation).toFixed(0)}% off fair value`);
  if (utilizationPct > idealUtilization + 12) alerts.push(`Storage filling fast (${utilizationPct.toFixed(0)}%)`);
  if (utilizationPct < idealUtilization - 12) alerts.push(`Storage draining fast (${utilizationPct.toFixed(0)}%)`);
  if (Math.abs(changePct) > 3) alerts.push(`High intraday volatility (${changePct >= 0 ? "+" : ""}${changePct.toFixed(1)}%)`);

  return (
    <div className="card p-5 relative overflow-hidden">
      <div className="absolute -top-12 -right-12 w-32 h-32 rounded-full bg-accent/5 blur-2xl pointer-events-none" />
      <div className="flex items-center gap-2 mb-2">
        <Sparkles size={14} className="text-accent" />
        <h2 className="text-sm font-semibold">Trader brief</h2>
        <span className="badge ml-auto text-[9px]">AUTO-GENERATED</span>
      </div>
      <p className="text-sm text-ink-100 leading-relaxed">
        <span className="font-semibold text-ink-50">{commodityName}</span> is{" "}
        <span className={tone}>
          {direction === "up" ? <TrendingUp size={12} className="inline -mt-0.5" /> :
           direction === "down" ? <TrendingDown size={12} className="inline -mt-0.5" /> : null}{" "}
          {direction === "flat" ? "unchanged" : `${changePct >= 0 ? "+" : ""}${changePct.toFixed(2)}% on the session`}
        </span>{" "}
        at <span className="font-mono text-ink-50">{price.toFixed(2)} {priceUnit}</span>.
        The fair-value model puts it at{" "}
        <span className="font-mono text-ink-50">{fairValue.toFixed(2)}</span> —{" "}
        <span className={`font-semibold ${fvTone}`}>{fvFlag}</span>{" "}
        ({fvDeviation >= 0 ? "+" : ""}{fvDeviation.toFixed(1)}% vs spot).
        Inventory at <span className="font-mono text-ink-50">{daysCover.toFixed(1)}d cover</span>{" "}
        vs <span className="text-ink-200">{daysCoverTarget.toFixed(0)}d</span> target reads as{" "}
        <span className={`font-semibold ${dcTone}`}>{dcFlag}</span>.
      </p>
      {alerts.length > 0 && (
        <div className="mt-3 pt-3 border-t border-ink-600/60 space-y-1">
          {alerts.map((a, i) => (
            <div key={i} className="flex items-center gap-2 text-[11px] text-warn">
              <AlertTriangle size={11} /> {a}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
