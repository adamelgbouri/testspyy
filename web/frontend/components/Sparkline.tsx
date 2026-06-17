"use client";

type Props = {
  data: number[];
  width?: number;
  height?: number;
  positive?: boolean;
  showArea?: boolean;
};

/** Minimal SVG sparkline for KPI cards. */
export function Sparkline({
  data,
  width = 80,
  height = 24,
  positive,
  showArea = true,
}: Props) {
  if (!data || data.length < 2) {
    return <div style={{ width, height }} className="opacity-20" />;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const stepX = width / (data.length - 1);

  // auto-detect direction if not provided
  const dir = positive ?? data[data.length - 1] >= data[0];
  const color = dir ? "#22c55e" : "#ef4444";

  const points = data.map((v, i) => ({
    x: i * stepX,
    y: height - ((v - min) / range) * height,
  }));
  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
  const areaPath = linePath + ` L${points[points.length - 1].x},${height} L0,${height} Z`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      {showArea && (
        <>
          <defs>
            <linearGradient id={`spark-${color}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.4} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <path d={areaPath} fill={`url(#spark-${color})`} />
        </>
      )}
      <path d={linePath} stroke={color} strokeWidth={1.5} fill="none"
            strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={points[points.length - 1].x} cy={points[points.length - 1].y}
              r={2} fill={color} />
    </svg>
  );
}
