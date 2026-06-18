"use client";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { api, type VolSurfaceResponse } from "@/lib/api";

const Plot = dynamic(() => import("./PlotlyChart"), {
  ssr: false,
  loading: () => (
    <div className="h-[520px] flex items-center justify-center text-sm text-ink-200">
      Loading 3D engine…
    </div>
  ),
});

type Props = { forward: number; baseSigma: number; rate: number };

/**
 * True 3D implied vol surface powered by Plotly.
 *
 * Full interactive controls out of the box:
 *   • drag = rotate camera
 *   • scroll wheel = zoom
 *   • shift + drag = pan
 *   • double-click = reset
 *   • hover = tooltip with K / T / σ_imp
 */
export function VolSurface({ forward, baseSigma, rate }: Props) {
  const [data, setData] = useState<VolSurfaceResponse | null>(null);
  const [showContours, setShowContours] = useState(true);
  const [colorscale, setColorscale] = useState<string>("Viridis");

  useEffect(() => {
    let cancelled = false;
    api
      .volSurface({
        forward, base_sigma: baseSigma, rate,
        n_strikes: 25, n_maturities: 18,
      })
      .then((d) => { if (!cancelled) setData(d); })
      .catch(console.error);
    return () => { cancelled = true; };
  }, [forward, baseSigma, rate]);

  // ATM ridge - vol along K = F for each maturity
  const atmRidge = useMemo(() => {
    if (!data) return null;
    const K = data.strikes;
    const idxLo = K.findIndex((v, i) =>
      i === K.length - 1 || (v <= forward && K[i + 1] > forward));
    const idx = Math.max(0, idxLo);
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    for (let j = 0; j < data.maturities.length; j++) {
      const K0 = K[idx];
      const K1 = K[Math.min(idx + 1, K.length - 1)];
      const t = (forward - K0) / (K1 - K0 || 1);
      const ivLo = data.iv_grid[j][idx];
      const ivHi = data.iv_grid[j][Math.min(idx + 1, K.length - 1)];
      xs.push(forward);
      ys.push(data.maturities[j]);
      zs.push(ivLo + t * (ivHi - ivLo));
    }
    return { xs, ys, zs };
  }, [data, forward]);

  if (!data || !atmRidge) {
    return (
      <div className="h-[520px] flex items-center justify-center text-sm text-ink-200">
        Loading surface…
      </div>
    );
  }

  const ivAsPct = data.iv_grid.map((row) => row.map((v) => v * 100));

  const traces: any[] = [
    {
      type: "surface",
      x: data.strikes,
      y: data.maturities,
      z: ivAsPct,
      colorscale,
      contours: showContours
        ? {
            z: {
              show: true,
              project: { z: true },
              usecolormap: true,
              highlightcolor: "#fbbf24",
              width: 2,
            },
          }
        : undefined,
      lighting: {
        ambient: 0.55, diffuse: 0.85, roughness: 0.4,
        specular: 0.25, fresnel: 0.15,
      },
      lightposition: { x: -1500, y: 1500, z: 1500 },
      opacity: 0.92,
      colorbar: {
        title: { text: "σ<sub>imp</sub> (%)", font: { color: "#e5e7eb" } },
        tickfont: { color: "#9ca3af", family: "JetBrains Mono", size: 10 },
        thickness: 12,
        len: 0.75,
        x: 1.02,
        outlinecolor: "#1f2937",
      },
      hovertemplate:
        "<b>K = %{x:.2f}</b><br>" +
        "T = %{y}d<br>" +
        "<b>σ<sub>imp</sub> = %{z:.2f}%</b>" +
        "<extra></extra>",
      name: "IV surface",
    },
    // ATM ridge - K = F
    {
      type: "scatter3d",
      mode: "lines",
      x: atmRidge.xs,
      y: atmRidge.ys,
      z: atmRidge.zs.map((v) => v * 100),
      line: { color: "#fbbf24", width: 6 },
      name: "ATM (K = F)",
      hovertemplate:
        "<b>ATM</b><br>T = %{y}d<br>σ = %{z:.2f}%<extra></extra>",
    },
  ];

  const layout: any = {
    scene: {
      xaxis: {
        title: { text: "<i>K</i> — strike", font: { color: "#e5e7eb", family: "JetBrains Mono" } },
        gridcolor: "#1f2937",
        zerolinecolor: "#374151",
        showbackground: true,
        backgroundcolor: "rgba(20,28,40,0.5)",
        color: "#9ca3af",
        tickfont: { color: "#9ca3af", family: "JetBrains Mono", size: 10 },
      },
      yaxis: {
        title: { text: "<i>T</i> — maturity (days)", font: { color: "#e5e7eb", family: "JetBrains Mono" } },
        gridcolor: "#1f2937",
        zerolinecolor: "#374151",
        showbackground: true,
        backgroundcolor: "rgba(15,22,32,0.5)",
        color: "#9ca3af",
        tickfont: { color: "#9ca3af", family: "JetBrains Mono", size: 10 },
        type: "log",
      },
      zaxis: {
        title: { text: "σ<sub>imp</sub> (%)", font: { color: "#e5e7eb", family: "JetBrains Mono" } },
        gridcolor: "#1f2937",
        zerolinecolor: "#374151",
        showbackground: true,
        backgroundcolor: "rgba(11,18,26,0.55)",
        color: "#9ca3af",
        tickfont: { color: "#9ca3af", family: "JetBrains Mono", size: 10 },
        ticksuffix: "%",
      },
      camera: { eye: { x: 1.6, y: 1.6, z: 0.9 } },
      aspectmode: "cube",
    },
    paper_bgcolor: "rgba(10,13,18,0)",
    plot_bgcolor: "rgba(10,13,18,0)",
    margin: { l: 0, r: 30, t: 0, b: 0 },
    height: 540,
    showlegend: true,
    legend: {
      x: 0.02, y: 0.98,
      bgcolor: "rgba(15,22,32,0.85)",
      bordercolor: "#1f2937",
      borderwidth: 1,
      font: { color: "#e5e7eb", family: "JetBrains Mono", size: 10 },
    },
    hoverlabel: {
      bgcolor: "#0e1117",
      bordercolor: "#374151",
      font: { color: "#f3f4f6", family: "JetBrains Mono" },
    },
  };

  const config: any = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: [
      "sendDataToCloud", "lasso2d", "select2d", "autoScale2d",
    ],
    toImageButtonOptions: { format: "png", filename: "vol-surface" },
  };

  const COLORSCALES = [
    "Viridis", "Inferno", "Plasma", "Magma", "Cividis",
    "Turbo", "RdYlBu", "Portland",
  ];

  return (
    <div className="relative">
      <div className="flex items-center gap-5 mb-3 text-[11px] text-ink-200">
        <div className="flex items-center gap-2">
          <span className="metric-label">COLOURMAP</span>
          <select
            value={colorscale}
            onChange={(e) => setColorscale(e.target.value)}
            className="bg-ink-700 border border-ink-500 rounded px-2 py-1 text-xs text-ink-50 font-mono"
          >
            {COLORSCALES.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showContours}
            onChange={(e) => setShowContours(e.target.checked)}
            className="accent-accent"
          />
          <span className="text-xs">Iso-vol contours on Z-plane</span>
        </label>
        <span className="ml-auto text-[10px] text-ink-300 font-mono">
          DRAG: rotate · SCROLL: zoom · SHIFT+DRAG: pan · DBL-CLICK: reset
        </span>
      </div>

      <div className="rounded-lg overflow-hidden border border-ink-600 bg-ink-900">
        <Plot
          data={traces}
          layout={layout}
          config={config}
          style={{ width: "100%", height: "540px" }}
          useResizeHandler
        />
      </div>

      <div className="grid grid-cols-3 gap-3 mt-3 text-xs">
        <div className="card p-3">
          <div className="metric-label">ATM SECTION</div>
          <div className="font-mono text-ink-50 text-sm mt-1">
            K = F = <span className="text-warn">{forward.toFixed(2)}</span>
          </div>
          <div className="text-[10px] text-ink-300 italic mt-1">
            Yellow ridge across the surface
          </div>
        </div>
        <div className="card p-3">
          <div className="metric-label">MESH RESOLUTION</div>
          <div className="font-mono text-ink-50 text-sm mt-1">
            {data.strikes.length} × {data.maturities.length} = {data.strikes.length * data.maturities.length} pts
          </div>
          <div className="text-[10px] text-ink-300 italic mt-1">
            Strike × maturity grid
          </div>
        </div>
        <div className="card p-3">
          <div className="metric-label">IV RANGE</div>
          <div className="font-mono text-ink-50 text-sm mt-1">
            {(Math.min(...data.iv_grid.flat()) * 100).toFixed(1)}%
            {" "}—{" "}
            {(Math.max(...data.iv_grid.flat()) * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-ink-300 italic mt-1">
            σ<sub>imp</sub> across the surface
          </div>
        </div>
      </div>
    </div>
  );
}
