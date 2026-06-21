"use client";

type Props = { score: number; label?: string };

/** Semi-circular gauge (0-100). */
export function SentimentGauge({ score, label = "Market sentiment" }: Props) {
  const clamped = Math.max(0, Math.min(100, score));
  // Half-circle: 180deg = 0, 0deg = 100
  const angle = 180 - (clamped / 100) * 180;
  const RAD = (a: number) => (a * Math.PI) / 180;

  // Arc endpoints
  const r = 80;
  const cx = 100;
  const cy = 100;
  const start = { x: cx - r, y: cy };
  const end = { x: cx + r, y: cy };

  // Needle endpoint
  const needleR = r - 8;
  const nx = cx + needleR * Math.cos(RAD(angle));
  const ny = cy - needleR * Math.sin(RAD(angle));

  let tone: "neg" | "pos" | "accent" = "accent";
  let txt = "Neutral";
  if (clamped < 25) { tone = "neg"; txt = "Bearish"; }
  else if (clamped < 45) { tone = "neg"; txt = "Mildly Bearish"; }
  else if (clamped < 55) { tone = "accent"; txt = "Neutral"; }
  else if (clamped < 75) { tone = "pos"; txt = "Mildly Bullish"; }
  else { tone = "pos"; txt = "Bullish"; }

  return (
    <div className="card p-5 flex flex-col items-center">
      <div className="metric-label mb-2">{label}</div>
      <svg viewBox="0 0 200 120" className="w-full max-w-[240px]">
        <defs>
          <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#ef4444" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#22c55e" />
          </linearGradient>
        </defs>
        {/* Background arc */}
        <path
          d={`M ${start.x},${start.y} A ${r},${r} 0 0 1 ${end.x},${end.y}`}
          fill="none" stroke="#1f2937" strokeWidth={14} strokeLinecap="round"
        />
        {/* Coloured arc */}
        <path
          d={`M ${start.x},${start.y} A ${r},${r} 0 0 1 ${end.x},${end.y}`}
          fill="none" stroke="url(#gaugeGrad)" strokeWidth={14}
          strokeLinecap="round" strokeOpacity={0.85}
        />
        {/* Tick marks */}
        {[0, 25, 50, 75, 100].map((v) => {
          const a = RAD(180 - (v / 100) * 180);
          const x1 = cx + (r - 18) * Math.cos(a);
          const y1 = cy - (r - 18) * Math.sin(a);
          const x2 = cx + (r - 4) * Math.cos(a);
          const y2 = cy - (r - 4) * Math.sin(a);
          return <line key={v} x1={x1} y1={y1} x2={x2} y2={y2}
                       stroke="#6b7280" strokeWidth={1} />;
        })}
        {/* Needle */}
        <line x1={cx} y1={cy} x2={nx} y2={ny}
              stroke="#f3f4f6" strokeWidth={2.5} strokeLinecap="round" />
        <circle cx={cx} cy={cy} r={6} fill="#0e1117" stroke="#f3f4f6" strokeWidth={1.5} />
        {/* Score */}
        <text x={cx} y={cy - 25} textAnchor="middle"
              fontSize="22" fontWeight="700" fill="#f3f4f6"
              fontFamily="JetBrains Mono">
          {clamped.toFixed(0)}
        </text>
      </svg>
      <div className={`text-sm font-semibold mt-2 ${
        tone === "pos" ? "text-pos" : tone === "neg" ? "text-neg" : "text-accent"
      }`}>
        {txt}
      </div>
    </div>
  );
}
