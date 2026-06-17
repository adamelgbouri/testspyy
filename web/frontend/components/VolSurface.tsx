"use client";
import { useEffect, useState } from "react";
import { api, type VolSurfaceResponse } from "@/lib/api";

type Props = {
  forward: number;
  baseSigma: number;
  rate: number;
};

/**
 * Pseudo-3D implied vol surface (no extra dependency).
 *
 * The grid is projected with an isometric transform:
 *     x_screen = strike − maturity·cos(30°)
 *     y_screen = iv − (strike + maturity)·sin(30°)
 *
 * Quads are shaded by IV value with a viridis-like ramp, sorted back-to-front
 * so the depth feels correct.  Rotation can be tweaked via state.
 */
export function VolSurface({ forward, baseSigma, rate }: Props) {
  const [data, setData] = useState<VolSurfaceResponse | null>(null);
  const [tilt, setTilt] = useState(28);
  const [rot, setRot] = useState(35);
  const [hover, setHover] = useState<{
    strike: number; maturity: number; iv: number; x: number; y: number
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetchSurface = async () => {
      try {
        const d = await api.volSurface({
          forward, base_sigma: baseSigma, rate,
          n_strikes: 14, n_maturities: 10,
        });
        if (!cancelled) setData(d);
      } catch (e) {
        console.error(e);
      }
    };
    fetchSurface();
    return () => { cancelled = true; };
  }, [forward, baseSigma, rate]);

  if (!data) {
    return (
      <div className="h-[420px] flex items-center justify-center text-sm text-ink-200">
        Loading surface…
      </div>
    );
  }

  const W = 760;
  const H = 420;
  const nS = data.strikes.length;
  const nT = data.maturities.length;

  // Normalise iv range for colours
  const flat = data.iv_grid.flat();
  const ivMin = Math.min(...flat);
  const ivMax = Math.max(...flat);
  const ivRange = ivMax - ivMin || 1;

  const Kmin = Math.min(...data.strikes);
  const Kmax = Math.max(...data.strikes);
  const Tmin = Math.min(...data.maturities);
  const Tmax = Math.max(...data.maturities);

  // Projection helpers
  const tiltRad = (tilt * Math.PI) / 180;
  const rotRad = (rot * Math.PI) / 180;

  const cellW = (W * 0.85) / Math.max(nS - 1, 1);
  const cellD = (H * 0.55) / Math.max(nT - 1, 1);
  const ivH = H * 0.55;        // amplitude vertical des IV
  const ox = W * 0.08;
  const oy = H * 0.78;

  const project = (i: number, j: number, iv: number) => {
    // i = strike index, j = maturity index
    const x0 = i * cellW;
    const z0 = j * cellD;
    const y0 = ((iv - ivMin) / ivRange) * ivH;
    // Rotate around z-axis (yaw) for left-right rotation
    const xr = x0 * Math.cos(rotRad) - z0 * Math.sin(rotRad);
    const zr = x0 * Math.sin(rotRad) + z0 * Math.cos(rotRad);
    // Isometric tilt
    return {
      x: ox + xr - zr * Math.cos(tiltRad) * 0.5,
      y: oy - y0 + zr * Math.sin(tiltRad),
    };
  };

  // Colour ramp - viridis-ish
  const colour = (iv: number) => {
    const t = (iv - ivMin) / ivRange;
    // 0 → indigo, 0.5 → teal, 1 → yellow
    const stops = [
      { t: 0,   r: 68,  g: 1,   b: 84 },
      { t: 0.3, r: 59,  g: 82,  b: 139 },
      { t: 0.55, r: 33, g: 145, b: 140 },
      { t: 0.85, r: 94, g: 201, b: 98 },
      { t: 1.0, r: 253, g: 231, b: 37 },
    ];
    let from = stops[0], to = stops[stops.length - 1];
    for (let k = 0; k < stops.length - 1; k++) {
      if (t >= stops[k].t && t <= stops[k + 1].t) {
        from = stops[k]; to = stops[k + 1];
        break;
      }
    }
    const r = (to.t - from.t) || 1;
    const f = (t - from.t) / r;
    return `rgb(${Math.round(from.r + (to.r - from.r) * f)},${
      Math.round(from.g + (to.g - from.g) * f)},${
      Math.round(from.b + (to.b - from.b) * f)})`;
  };

  // Build quads
  type Quad = {
    pts: [number, number][];
    fill: string;
    avgZ: number;
    iv: number;
    i: number; j: number;
  };
  const quads: Quad[] = [];
  for (let j = 0; j < nT - 1; j++) {
    for (let i = 0; i < nS - 1; i++) {
      const ivAvg = (
        data.iv_grid[j][i] + data.iv_grid[j][i + 1] +
        data.iv_grid[j + 1][i] + data.iv_grid[j + 1][i + 1]
      ) / 4;
      const p1 = project(i,     j,     data.iv_grid[j][i]);
      const p2 = project(i + 1, j,     data.iv_grid[j][i + 1]);
      const p3 = project(i + 1, j + 1, data.iv_grid[j + 1][i + 1]);
      const p4 = project(i,     j + 1, data.iv_grid[j + 1][i]);
      // depth for sorting
      const avgZ = j * cellD * Math.cos(rotRad) + i * cellW * Math.sin(rotRad);
      quads.push({
        pts: [[p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y], [p4.x, p4.y]],
        fill: colour(ivAvg), avgZ, iv: ivAvg,
        i, j,
      });
    }
  }
  // Sort back-to-front
  quads.sort((a, b) => b.avgZ - a.avgZ);

  // Axes
  const axisOrigin = project(0, 0, ivMin);
  const axisStrike = project(nS - 1, 0, ivMin);
  const axisMat = project(0, nT - 1, ivMin);
  const axisIV = project(0, 0, ivMax);

  return (
    <div className="relative">
      {/* Controls */}
      <div className="flex items-center gap-4 mb-2 text-[11px] text-ink-200">
        <div className="flex items-center gap-2">
          <span>Rotate</span>
          <input type="range" min={-60} max={70} value={rot}
            onChange={(e) => setRot(parseInt(e.target.value))}
            className="accent-accent w-32" />
        </div>
        <div className="flex items-center gap-2">
          <span>Tilt</span>
          <input type="range" min={10} max={50} value={tilt}
            onChange={(e) => setTilt(parseInt(e.target.value))}
            className="accent-accent w-32" />
        </div>
        <span className="ml-auto font-mono">
          IV {(ivMin * 100).toFixed(0)}% – {(ivMax * 100).toFixed(0)}%
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {/* Grid floor (sortie de jeu de profondeur) */}
        <g opacity={0.15}>
          {Array.from({ length: nS }).map((_, i) => {
            const a = project(i, 0, ivMin);
            const b = project(i, nT - 1, ivMin);
            return <line key={`v${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                         stroke="#9ca3af" strokeWidth={0.5} />;
          })}
          {Array.from({ length: nT }).map((_, j) => {
            const a = project(0, j, ivMin);
            const b = project(nS - 1, j, ivMin);
            return <line key={`h${j}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                         stroke="#9ca3af" strokeWidth={0.5} />;
          })}
        </g>

        {/* Surface quads */}
        {quads.map((q, k) => (
          <polygon key={k}
            points={q.pts.map((p) => p.join(",")).join(" ")}
            fill={q.fill}
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={0.5}
            onMouseEnter={() => setHover({
              strike: data.strikes[q.i],
              maturity: data.maturities[q.j],
              iv: q.iv,
              x: q.pts[0][0], y: q.pts[0][1],
            })}
            onMouseLeave={() => setHover(null)}
            style={{ cursor: "crosshair" }}
          />
        ))}

        {/* Axes */}
        <line x1={axisOrigin.x} y1={axisOrigin.y} x2={axisStrike.x} y2={axisStrike.y}
              stroke="#00d4ff" strokeWidth={1.5} />
        <line x1={axisOrigin.x} y1={axisOrigin.y} x2={axisMat.x} y2={axisMat.y}
              stroke="#a78bfa" strokeWidth={1.5} />
        <line x1={axisOrigin.x} y1={axisOrigin.y} x2={axisIV.x} y2={axisIV.y}
              stroke="#f59e0b" strokeWidth={1.5} />

        {/* Axis labels */}
        <text x={axisStrike.x + 8} y={axisStrike.y + 4}
              fill="#00d4ff" fontSize={11} fontFamily="JetBrains Mono">
          STRIKE  ({Kmin.toFixed(0)} → {Kmax.toFixed(0)})
        </text>
        <text x={axisMat.x - 4} y={axisMat.y + 16}
              fill="#a78bfa" fontSize={11} fontFamily="JetBrains Mono">
          MATURITY  ({Tmin}d → {Tmax}d)
        </text>
        <text x={axisIV.x - 6} y={axisIV.y - 6}
              fill="#f59e0b" fontSize={11} fontFamily="JetBrains Mono">
          IV
        </text>

        {/* Hover tooltip */}
        {hover && (
          <g pointerEvents="none">
            <rect x={hover.x + 8} y={hover.y - 38} width={160} height={50}
                  fill="#0e1117" stroke="#374151" rx={6} />
            <text x={hover.x + 16} y={hover.y - 23}
                  fill="#f3f4f6" fontSize={11} fontFamily="JetBrains Mono" fontWeight={600}>
              K = {hover.strike.toFixed(1)}  ·  T = {hover.maturity}d
            </text>
            <text x={hover.x + 16} y={hover.y - 8}
                  fill="#00d4ff" fontSize={13} fontFamily="JetBrains Mono" fontWeight={700}>
              IV = {(hover.iv * 100).toFixed(1)}%
            </text>
          </g>
        )}
      </svg>

      {/* Colour legend */}
      <div className="flex items-center gap-2 mt-2">
        <span className="text-[10px] text-ink-200 font-mono">{(ivMin * 100).toFixed(0)}%</span>
        <div className="flex-1 h-2 rounded-full"
             style={{
               background: "linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37))",
             }} />
        <span className="text-[10px] text-ink-200 font-mono">{(ivMax * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}
