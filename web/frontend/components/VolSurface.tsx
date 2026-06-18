"use client";
import { useEffect, useMemo, useState } from "react";
import { api, type VolSurfaceResponse } from "@/lib/api";

type Props = {
  forward: number;
  baseSigma: number;
  rate: number;
};

/**
 * Professional pseudo-3D implied vol surface.
 *
 * Renders a wireframe surface inside a 3D bounding box with floor + back walls
 * (à la matplotlib/Plotly), high-resolution viridis colouring, contour
 * projection on the floor, and a highlighted ATM cross-section (K = F).
 */
export function VolSurface({ forward, baseSigma, rate }: Props) {
  const [data, setData] = useState<VolSurfaceResponse | null>(null);
  const [tilt, setTilt] = useState(32);
  const [rot, setRot] = useState(38);
  const [hover, setHover] = useState<{
    K: number; T: number; iv: number; x: number; y: number;
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .volSurface({
        forward, base_sigma: baseSigma, rate,
        n_strikes: 20, n_maturities: 14,
      })
      .then((d) => { if (!cancelled) setData(d); })
      .catch(console.error);
    return () => { cancelled = true; };
  }, [forward, baseSigma, rate]);

  // ===== Stable layout dimensions =====
  const W = 820;
  const H = 460;
  const ox = 95;       // origin x
  const oy = 380;      // origin y
  const cellW = 32;    // strike axis pixel step (rotated)
  const cellD = 22;    // maturity axis pixel step (rotated/depth)
  const ivH = 220;     // vertical scale for IV
  // ===================================

  const tiltRad = (tilt * Math.PI) / 180;
  const rotRad = (rot * Math.PI) / 180;

  // 3D → 2D isometric-style projection
  const project = (i: number, j: number, ivNorm: number) => {
    const x0 = i * cellW;
    const z0 = j * cellD;
    const y0 = ivNorm * ivH;
    const xr = x0 * Math.cos(rotRad) - z0 * Math.sin(rotRad);
    const zr = x0 * Math.sin(rotRad) + z0 * Math.cos(rotRad);
    return {
      x: ox + xr - zr * Math.cos(tiltRad) * 0.55,
      y: oy - y0 + zr * Math.sin(tiltRad),
    };
  };

  // Viridis colour ramp
  const viridis = (t: number) => {
    t = Math.max(0, Math.min(1, t));
    const stops = [
      [68,  1,   84],
      [72,  35,  116],
      [64,  67,  135],
      [52,  94,  141],
      [41,  120, 142],
      [32,  144, 140],
      [34,  167, 132],
      [68,  190, 112],
      [121, 209, 81],
      [189, 222, 38],
      [253, 231, 36],
    ];
    const s = t * (stops.length - 1);
    const k = Math.floor(s);
    const f = s - k;
    const a = stops[k];
    const b = stops[Math.min(k + 1, stops.length - 1)];
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${
      Math.round(a[1] + (b[1] - a[1]) * f)},${
      Math.round(a[2] + (b[2] - a[2]) * f)})`;
  };

  // Build the rendering structures
  const view = useMemo(() => {
    if (!data) return null;
    const nS = data.strikes.length;
    const nT = data.maturities.length;
    const flat = data.iv_grid.flat();
    const ivMin = Math.min(...flat);
    const ivMax = Math.max(...flat);
    const ivRange = ivMax - ivMin || 1;
    const ivN = (v: number) => (v - ivMin) / ivRange;

    // Quads
    type Quad = {
      pts: [number, number][];
      fill: string;
      avgDepth: number;
      iv: number; i: number; j: number;
      pts3: { x: number; y: number; iv: number; K: number; T: number }[];
    };
    const quads: Quad[] = [];
    for (let j = 0; j < nT - 1; j++) {
      for (let i = 0; i < nS - 1; i++) {
        const iv00 = data.iv_grid[j][i];
        const iv10 = data.iv_grid[j][i + 1];
        const iv11 = data.iv_grid[j + 1][i + 1];
        const iv01 = data.iv_grid[j + 1][i];
        const avg = (iv00 + iv10 + iv11 + iv01) / 4;
        const p1 = project(i,     j,     ivN(iv00));
        const p2 = project(i + 1, j,     ivN(iv10));
        const p3 = project(i + 1, j + 1, ivN(iv11));
        const p4 = project(i,     j + 1, ivN(iv01));
        const avgDepth = (i + j) * (Math.sin(rotRad) + Math.cos(rotRad));
        quads.push({
          pts: [[p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y], [p4.x, p4.y]],
          fill: viridis(ivN(avg)),
          avgDepth, iv: avg, i, j,
          pts3: [
            { x: p1.x, y: p1.y, iv: iv00, K: data.strikes[i],     T: data.maturities[j] },
            { x: p2.x, y: p2.y, iv: iv10, K: data.strikes[i + 1], T: data.maturities[j] },
            { x: p3.x, y: p3.y, iv: iv11, K: data.strikes[i + 1], T: data.maturities[j + 1] },
            { x: p4.x, y: p4.y, iv: iv01, K: data.strikes[i],     T: data.maturities[j + 1] },
          ],
        });
      }
    }
    quads.sort((a, b) => a.avgDepth - b.avgDepth);

    // Bounding box corners (in normalised iv space, 0 at floor / 1 at ceiling)
    const corners = {
      o00: project(0,       0,       0),
      o10: project(nS - 1,  0,       0),
      o11: project(nS - 1,  nT - 1,  0),
      o01: project(0,       nT - 1,  0),
      c00: project(0,       0,       1),
      c10: project(nS - 1,  0,       1),
      c11: project(nS - 1,  nT - 1,  1),
      c01: project(0,       nT - 1,  1),
    };

    // ATM line (where strike ≈ forward)
    const atmIdx = data.strikes.findIndex((K, idx) =>
      idx === data.strikes.length - 1 ||
      Math.abs(K - data.forward) <= Math.abs(data.strikes[idx + 1] - data.forward));
    const atmLine: { x: number; y: number }[] = [];
    if (atmIdx >= 0) {
      // Interpolate strike where K = F (more precise than nearest index)
      for (let j = 0; j < nT; j++) {
        // linear interp at forward
        const K0 = data.strikes[atmIdx];
        const K1 = data.strikes[Math.min(atmIdx + 1, nS - 1)];
        const iv0 = data.iv_grid[j][atmIdx];
        const iv1 = data.iv_grid[j][Math.min(atmIdx + 1, nS - 1)];
        const t = (data.forward - K0) / (K1 - K0 || 1);
        const ivAtm = iv0 + t * (iv1 - iv0);
        const pos = atmIdx + t;
        const p = project(pos, j, ivN(ivAtm));
        atmLine.push({ x: p.x, y: p.y });
      }
    }

    // Contour projection on the floor (3 iso-IV lines)
    const contourLevels = [0.33, 0.66];
    type ContourSeg = { x1: number; y1: number; x2: number; y2: number };
    const contours: { level: number; segs: ContourSeg[] }[] = [];
    for (const lvl of contourLevels) {
      const segs: ContourSeg[] = [];
      // Marching squares on a normalised grid -> project to floor
      for (let j = 0; j < nT - 1; j++) {
        for (let i = 0; i < nS - 1; i++) {
          const v00 = ivN(data.iv_grid[j][i]);
          const v10 = ivN(data.iv_grid[j][i + 1]);
          const v11 = ivN(data.iv_grid[j + 1][i + 1]);
          const v01 = ivN(data.iv_grid[j + 1][i]);
          const corners = [
            { v: v00, i, j },
            { v: v10, i: i + 1, j },
            { v: v11, i: i + 1, j: j + 1 },
            { v: v01, i, j: j + 1 },
          ];
          const above = corners.map((c) => c.v >= lvl);
          // For each edge crossing, compute intersection projected onto floor.
          const edges: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 0]];
          const pts: { x: number; y: number }[] = [];
          for (const [a, b] of edges) {
            if (above[a] !== above[b]) {
              const t = (lvl - corners[a].v) / (corners[b].v - corners[a].v || 1);
              const ai = corners[a].i + (corners[b].i - corners[a].i) * t;
              const aj = corners[a].j + (corners[b].j - corners[a].j) * t;
              pts.push(project(ai, aj, 0));   // floor (iv=0)
            }
          }
          if (pts.length === 2) {
            segs.push({ x1: pts[0].x, y1: pts[0].y, x2: pts[1].x, y2: pts[1].y });
          }
        }
      }
      contours.push({ level: lvl, segs });
    }

    return { quads, corners, atmLine, contours, ivMin, ivMax, nS, nT };
  }, [data, rot, tilt]);

  if (!data || !view) {
    return (
      <div className="h-[460px] flex items-center justify-center text-sm text-ink-200">
        Loading surface…
      </div>
    );
  }

  const { quads, corners, atmLine, contours, ivMin, ivMax, nS, nT } = view;
  const Kmin = Math.min(...data.strikes);
  const Kmax = Math.max(...data.strikes);
  const Tmin = Math.min(...data.maturities);
  const Tmax = Math.max(...data.maturities);

  return (
    <div className="relative">
      {/* Controls */}
      <div className="flex items-center gap-5 mb-3 text-[11px] text-ink-200">
        <div className="flex items-center gap-2">
          <span className="font-mono">YAW</span>
          <input type="range" min={-60} max={70} value={rot}
            onChange={(e) => setRot(parseInt(e.target.value))}
            className="accent-accent w-32" />
          <span className="font-mono text-ink-100 w-7">{rot}°</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono">PITCH</span>
          <input type="range" min={10} max={50} value={tilt}
            onChange={(e) => setTilt(parseInt(e.target.value))}
            className="accent-accent w-32" />
          <span className="font-mono text-ink-100 w-7">{tilt}°</span>
        </div>
        <span className="ml-auto font-mono">
          σ_imp ∈ [{(ivMin * 100).toFixed(1)}, {(ivMax * 100).toFixed(1)}] %
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ background: "#0a0d12" }}>
        <defs>
          {/* Subtle vignette */}
          <radialGradient id="vsBg" cx="50%" cy="50%" r="70%">
            <stop offset="0%" stopColor="#10141b" />
            <stop offset="100%" stopColor="#0a0d12" />
          </radialGradient>
        </defs>
        <rect width={W} height={H} fill="url(#vsBg)" />

        {/* BACK WALLS (depth) */}
        {/* Back face (j = nT − 1 plane: K × IV at max maturity) */}
        <polygon
          points={`${corners.o01.x},${corners.o01.y} ${corners.o11.x},${corners.o11.y} ${corners.c11.x},${corners.c11.y} ${corners.c01.x},${corners.c01.y}`}
          fill="rgba(20,28,40,0.6)" stroke="#1f2937" strokeWidth={0.5}
        />
        {/* Left face (i = 0 plane: T × IV at min strike) */}
        <polygon
          points={`${corners.o00.x},${corners.o00.y} ${corners.o01.x},${corners.o01.y} ${corners.c01.x},${corners.c01.y} ${corners.c00.x},${corners.c00.y}`}
          fill="rgba(15,22,32,0.55)" stroke="#1f2937" strokeWidth={0.5}
        />

        {/* Grid lines on back walls */}
        <g stroke="#243349" strokeWidth={0.5} strokeDasharray="2 3">
          {/* horizontal IV lines on back face */}
          {[0.25, 0.5, 0.75, 1].map((t) => {
            const left = project(0, nT - 1, t);
            const right = project(nS - 1, nT - 1, t);
            return <line key={`hb-${t}`} x1={left.x} y1={left.y} x2={right.x} y2={right.y} />;
          })}
          {/* horizontal IV lines on left face */}
          {[0.25, 0.5, 0.75, 1].map((t) => {
            const a = project(0, 0, t);
            const b = project(0, nT - 1, t);
            return <line key={`hl-${t}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} />;
          })}
          {/* vertical strikes on back face */}
          {Array.from({ length: nS }).filter((_, i) => i % 4 === 0).map((_, idx) => {
            const i = idx * 4;
            const a = project(i, nT - 1, 0);
            const b = project(i, nT - 1, 1);
            return <line key={`vs-${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} />;
          })}
          {/* vertical maturities on left face */}
          {Array.from({ length: nT }).filter((_, j) => j % 3 === 0).map((_, idx) => {
            const j = idx * 3;
            const a = project(0, j, 0);
            const b = project(0, j, 1);
            return <line key={`vm-${j}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} />;
          })}
        </g>

        {/* FLOOR grid */}
        <polygon
          points={`${corners.o00.x},${corners.o00.y} ${corners.o10.x},${corners.o10.y} ${corners.o11.x},${corners.o11.y} ${corners.o01.x},${corners.o01.y}`}
          fill="rgba(11,18,26,0.85)" stroke="#1f2937" strokeWidth={0.6}
        />
        <g stroke="#1c2838" strokeWidth={0.4}>
          {Array.from({ length: nS }).map((_, i) => {
            const a = project(i, 0, 0);
            const b = project(i, nT - 1, 0);
            return <line key={`fl-i-${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} />;
          })}
          {Array.from({ length: nT }).map((_, j) => {
            const a = project(0, j, 0);
            const b = project(nS - 1, j, 0);
            return <line key={`fl-j-${j}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} />;
          })}
        </g>

        {/* Contour projections on the floor */}
        {contours.map((c, idx) => (
          <g key={`ct-${idx}`} stroke={viridis(c.level)} strokeWidth={1.2}
             strokeLinecap="round" opacity={0.85}>
            {c.segs.map((s, k) => (
              <line key={k} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2} />
            ))}
          </g>
        ))}

        {/* SURFACE quads */}
        {quads.map((q, k) => (
          <polygon key={k}
            points={q.pts.map((p) => p.join(",")).join(" ")}
            fill={q.fill}
            stroke="rgba(255,255,255,0.10)"
            strokeWidth={0.5}
            onMouseEnter={() => setHover({
              K: data.strikes[q.i] + (data.strikes[q.i + 1] - data.strikes[q.i]) / 2,
              T: data.maturities[q.j],
              iv: q.iv,
              x: q.pts[0][0], y: q.pts[0][1],
            })}
            onMouseLeave={() => setHover(null)}
            style={{ cursor: "crosshair" }}
          />
        ))}

        {/* ATM line on the surface */}
        {atmLine.length > 1 && (
          <polyline
            points={atmLine.map((p) => `${p.x},${p.y}`).join(" ")}
            fill="none" stroke="#fbbf24" strokeWidth={2}
            strokeDasharray="4 3"
            style={{ filter: "drop-shadow(0 0 4px rgba(251,191,36,0.6))" }}
          />
        )}

        {/* Bounding-box edges (frame) */}
        <g stroke="#445" strokeWidth={0.8} fill="none">
          <line x1={corners.o00.x} y1={corners.o00.y} x2={corners.o10.x} y2={corners.o10.y} />
          <line x1={corners.o00.x} y1={corners.o00.y} x2={corners.o01.x} y2={corners.o01.y} />
          <line x1={corners.o00.x} y1={corners.o00.y} x2={corners.c00.x} y2={corners.c00.y} />
          <line x1={corners.o10.x} y1={corners.o10.y} x2={corners.o11.x} y2={corners.o11.y} />
          <line x1={corners.o10.x} y1={corners.o10.y} x2={corners.c10.x} y2={corners.c10.y} />
          <line x1={corners.o01.x} y1={corners.o01.y} x2={corners.o11.x} y2={corners.o11.y} />
          <line x1={corners.o01.x} y1={corners.o01.y} x2={corners.c01.x} y2={corners.c01.y} />
        </g>

        {/* Axis labels with mathematical notation */}
        <g fontFamily="JetBrains Mono">
          {/* X axis ticks - strikes */}
          {[0, Math.floor((nS - 1) / 2), nS - 1].map((i) => {
            const p = project(i, 0, 0);
            return (
              <g key={`tk-${i}`}>
                <line x1={p.x} y1={p.y} x2={p.x} y2={p.y + 5}
                      stroke="#5a6d85" strokeWidth={0.8} />
                <text x={p.x} y={p.y + 18} textAnchor="middle"
                      fontSize={10} fill="#9ca3af">
                  {data.strikes[i].toFixed(0)}
                </text>
              </g>
            );
          })}
          {/* Z axis ticks - maturities */}
          {[0, Math.floor((nT - 1) / 2), nT - 1].map((j) => {
            const p = project(0, j, 0);
            return (
              <g key={`tt-${j}`}>
                <line x1={p.x} y1={p.y} x2={p.x - 8} y2={p.y + 2}
                      stroke="#5a6d85" strokeWidth={0.8} />
                <text x={p.x - 12} y={p.y + 4} textAnchor="end"
                      fontSize={10} fill="#9ca3af">
                  {data.maturities[j]}d
                </text>
              </g>
            );
          })}
          {/* Y axis ticks - IV */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const p = project(0, 0, t);
            const iv = ivMin + t * (ivMax - ivMin);
            return (
              <g key={`tv-${t}`}>
                <line x1={p.x} y1={p.y} x2={p.x - 5} y2={p.y}
                      stroke="#5a6d85" strokeWidth={0.8} />
                <text x={p.x - 9} y={p.y + 3} textAnchor="end"
                      fontSize={10} fill="#9ca3af">
                  {(iv * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}

          {/* Axis main labels with italic math styling */}
          <text x={(corners.o00.x + corners.o10.x) / 2}
                y={Math.max(corners.o00.y, corners.o10.y) + 36}
                textAnchor="middle"
                fontSize={12} fill="#e5e7eb" fontStyle="italic" fontWeight={600}>
            K  <tspan fontSize={10} fill="#9ca3af" fontStyle="normal">(strike)</tspan>
          </text>
          <text x={corners.o01.x - 50} y={corners.o01.y + 24}
                fontSize={12} fill="#e5e7eb" fontStyle="italic" fontWeight={600}>
            T  <tspan fontSize={10} fill="#9ca3af" fontStyle="normal">(maturity)</tspan>
          </text>
          <text x={corners.c00.x - 28} y={corners.c00.y - 6}
                fontSize={12} fill="#e5e7eb" fontStyle="italic" fontWeight={600}>
            σ <tspan fontSize={9} fill="#9ca3af" baselineShift="sub">imp</tspan>
            <tspan fontSize={10} fill="#9ca3af" fontStyle="normal">  (vol)</tspan>
          </text>
        </g>

        {/* ATM legend (top-right) */}
        <g transform="translate(610, 30)">
          <rect x={-4} y={-12} width={180} height={50} rx={6}
                fill="rgba(11,15,20,0.85)" stroke="#1f2937" />
          <line x1={4} y1={2} x2={28} y2={2}
                stroke="#fbbf24" strokeWidth={2} strokeDasharray="4 3" />
          <text x={32} y={6} fontSize={11} fill="#e5e7eb" fontFamily="JetBrains Mono">
            ATM (K = F = {data.forward.toFixed(2)})
          </text>
          <line x1={4} y1={20} x2={28} y2={20} stroke={viridis(0.5)} strokeWidth={1.2} />
          <text x={32} y={24} fontSize={11} fill="#9ca3af" fontFamily="JetBrains Mono">
            iso-IV contours
          </text>
        </g>

        {/* Hover tooltip */}
        {hover && (
          <g pointerEvents="none">
            <rect x={hover.x + 10} y={hover.y - 42} width={190} height={56}
                  fill="#0e1117" stroke="#374151" rx={6} />
            <text x={hover.x + 22} y={hover.y - 24}
                  fill="#9ca3af" fontSize={10} fontFamily="JetBrains Mono">
              K = {hover.K.toFixed(2)}  ·  T = {hover.T}d
            </text>
            <text x={hover.x + 22} y={hover.y - 8}
                  fill="#fbbf24" fontSize={13} fontFamily="JetBrains Mono" fontWeight={700}>
              σ_imp = {(hover.iv * 100).toFixed(2)}%
            </text>
            <text x={hover.x + 22} y={hover.y + 8}
                  fill="#9ca3af" fontSize={9} fontFamily="JetBrains Mono">
              moneyness {(Math.log(hover.K / data.forward)).toFixed(3)}
            </text>
          </g>
        )}
      </svg>

      {/* Viridis colour ramp legend */}
      <div className="flex items-center gap-3 mt-3 text-[10px] text-ink-200 font-mono">
        <span>{(ivMin * 100).toFixed(0)}%</span>
        <div className="flex-1 h-2 rounded-full"
          style={{
            background:
              "linear-gradient(to right, rgb(68,1,84), rgb(72,35,116), rgb(64,67,135), rgb(52,94,141), rgb(41,120,142), rgb(32,144,140), rgb(34,167,132), rgb(68,190,112), rgb(121,209,81), rgb(189,222,38), rgb(253,231,36))",
          }} />
        <span>{(ivMax * 100).toFixed(0)}%</span>
        <span className="text-ink-300">σ_imp</span>
      </div>
    </div>
  );
}
