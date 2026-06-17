"use client";
import { useState } from "react";
import type { RegionalRow } from "@/lib/api";

type Props = { rows: RegionalRow[]; unit: string };

/**
 * Approximate longitude/latitude per known region or country group.
 * Picked at country centroids for major commodity producers/consumers.
 */
const REGION_COORDS: Record<string, [number, number]> = {
  // Lon, Lat
  "US": [-98, 39],
  "US Northeast": [-77, 41],
  "Canada": [-100, 56],
  "Mexico": [-102, 23],
  "Brazil": [-52, -14],
  "Argentina": [-65, -38],
  "Europe": [10, 50],
  "EU": [10, 50],
  "Russia & CIS": [60, 56],
  "Russia": [55, 60],
  "Middle East": [45, 25],
  "China": [105, 35],
  "China (mine)": [110, 32],
  "India": [78, 22],
  "India (consumer)": [78, 22],
  "Other Asia": [110, 5],
  "Asia": [110, 5],
  "Japan/Korea": [135, 36],
  "Taiwan": [121, 24],
  "Australia": [134, -25],
  "Chile & Peru": [-72, -20],
  "DRC & Zambia": [25, -10],
  "Ukraine": [32, 49],
  "Africa": [20, 5],
  "Indonesia": [120, -2],
  "LNG export": [-90, 30],            // US Gulf LNG hub
  "Power": [-95, 38],
  "Industrial": [-95, 38],
  "OECD ETFs": [0, 50],
  "OECD ETFs / investment": [0, 50],
  "OECD": [0, 50],
  "Central Banks": [-20, 35],
  "Black Sea": [37, 44],
  "Rest of World": [40, 0],
};

const MAP_W = 1000;
const MAP_H = 480;

/** Equirectangular projection. */
function project(lon: number, lat: number): [number, number] {
  const x = ((lon + 180) / 360) * MAP_W;
  const y = ((90 - lat) / 180) * MAP_H;
  return [x, y];
}

export function WorldMap({ rows, unit }: Props) {
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);

  const exporters = rows.filter((r) => r.net_trade > 0).sort((a, b) => b.net_trade - a.net_trade);
  const importers = rows.filter((r) => r.net_trade < 0).sort((a, b) => a.net_trade - b.net_trade);

  const totalExports = exporters.reduce((s, r) => s + r.net_trade, 0);
  const totalImports = importers.reduce((s, r) => s + Math.abs(r.net_trade), 0);

  // Build flows: each exporter ships to importers proportional to import deficit
  const flows: { from: string; to: string; volume: number }[] = [];
  for (const e of exporters) {
    for (const im of importers) {
      const share = Math.abs(im.net_trade) / Math.max(totalImports, 1);
      const vol = e.net_trade * share;
      if (vol > 0) flows.push({ from: e.region, to: im.region, volume: vol });
    }
  }
  const maxFlow = Math.max(...flows.map((f) => f.volume), 0.1);

  const getCoord = (region: string): [number, number] => {
    const c = REGION_COORDS[region];
    if (c) return project(c[0], c[1]);
    return [MAP_W / 2, MAP_H / 2];
  };

  const maxNet = Math.max(...rows.map((r) => Math.abs(r.net_trade)), 0.1);

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${MAP_W} ${MAP_H}`} className="w-full bg-ink-900/60 rounded-lg border border-ink-600">
        <defs>
          <marker id="arrow-pos" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#22c55e" />
          </marker>
          <marker id="arrow-neg" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#ef4444" />
          </marker>
          <radialGradient id="globe-glow">
            <stop offset="0%" stopColor="rgba(0,212,255,0.04)" />
            <stop offset="100%" stopColor="rgba(0,212,255,0)" />
          </radialGradient>
        </defs>

        <rect width={MAP_W} height={MAP_H} fill="url(#globe-glow)" />
        {/* Simplified land masses */}
        <g fill="#1f2937" stroke="#374151" strokeWidth={0.5} opacity={0.75}>
          {/* North America */}
          <path d="M 80,90 Q 130,80 180,100 L 220,140 L 240,200 L 230,260 L 200,290 L 150,300 L 100,260 L 70,200 L 60,140 Z" />
          {/* Central America */}
          <path d="M 200,290 L 230,320 L 250,360 L 240,380 L 220,360 L 210,330 Z" />
          {/* South America */}
          <path d="M 250,340 Q 280,330 310,360 L 330,420 L 320,460 L 290,480 L 270,460 L 260,420 L 250,380 Z" />
          {/* Europe */}
          <path d="M 460,110 L 510,100 L 560,120 L 580,160 L 560,200 L 520,220 L 490,210 L 460,180 L 450,150 Z" />
          {/* Africa */}
          <path d="M 470,230 Q 510,225 540,250 L 560,320 L 550,400 L 530,440 L 500,440 L 470,400 L 460,340 L 460,280 Z" />
          {/* Middle East */}
          <path d="M 570,210 L 620,210 L 640,240 L 640,280 L 610,290 L 580,280 L 570,250 Z" />
          {/* Russia & Northern Asia */}
          <path d="M 580,80 L 720,70 L 850,75 L 920,90 L 940,130 L 900,150 L 800,140 L 700,130 L 600,135 L 570,110 Z" />
          {/* China & SE Asia */}
          <path d="M 720,160 L 800,160 L 870,180 L 880,220 L 850,250 L 800,260 L 770,250 L 740,220 L 720,190 Z" />
          {/* India */}
          <path d="M 700,240 L 740,240 L 760,280 L 740,320 L 710,320 L 690,280 Z" />
          {/* Indonesia & Philippines */}
          <path d="M 820,290 L 870,290 L 900,310 L 880,330 L 840,330 L 820,310 Z" />
          {/* Australia */}
          <path d="M 830,370 L 900,360 L 940,380 L 940,430 L 900,440 L 850,430 L 830,400 Z" />
          {/* Greenland */}
          <path d="M 380,40 L 410,30 L 440,50 L 430,80 L 400,90 L 380,70 Z" />
          {/* UK */}
          <path d="M 460,140 L 480,135 L 490,160 L 470,170 L 460,160 Z" />
          {/* Japan */}
          <path d="M 900,180 L 920,170 L 925,200 L 910,210 Z" />
        </g>

        {/* Grid latitude lines */}
        <g stroke="#1f2937" strokeWidth={0.5} opacity={0.4}>
          {[-60, -30, 0, 30, 60].map((lat) => {
            const [, y] = project(0, lat);
            return <line key={lat} x1={0} y1={y} x2={MAP_W} y2={y} strokeDasharray="2 4" />;
          })}
          {[-120, -60, 0, 60, 120].map((lon) => {
            const [x] = project(lon, 0);
            return <line key={lon} x1={x} y1={0} x2={x} y2={MAP_H} strokeDasharray="2 4" />;
          })}
        </g>

        {/* Flow arrows (curved bezier from exporter to importer) */}
        {flows.map((f, idx) => {
          const [x1, y1] = getCoord(f.from);
          const [x2, y2] = getCoord(f.to);
          // Control point for curvature (curve upward)
          const mx = (x1 + x2) / 2;
          const my = Math.min(y1, y2) - Math.abs(x2 - x1) * 0.25 - 30;
          const opacity = 0.18 + (f.volume / maxFlow) * 0.6;
          const width = 0.5 + (f.volume / maxFlow) * 4;
          const visible = hoveredRegion === null
            || hoveredRegion === f.from
            || hoveredRegion === f.to;
          return (
            <path
              key={idx}
              d={`M ${x1},${y1} Q ${mx},${my} ${x2},${y2}`}
              fill="none"
              stroke="#22c55e"
              strokeWidth={width}
              opacity={visible ? opacity : 0.05}
              markerEnd="url(#arrow-pos)"
              style={{ transition: "opacity 0.2s" }}
            />
          );
        })}

        {/* Region markers */}
        {rows.map((r) => {
          const [x, y] = getCoord(r.region);
          const radius = 6 + (Math.abs(r.net_trade) / maxNet) * 16;
          const colour = r.status === "exporter" ? "#22c55e"
                       : r.status === "importer" ? "#ef4444" : "#9ca3af";
          const isHovered = hoveredRegion === r.region;
          return (
            <g key={r.region}
               onMouseEnter={() => setHoveredRegion(r.region)}
               onMouseLeave={() => setHoveredRegion(null)}
               style={{ cursor: "pointer" }}>
              <circle cx={x} cy={y} r={radius + 6} fill={colour} opacity={0.15} />
              <circle cx={x} cy={y} r={radius} fill={colour} opacity={0.7}
                      stroke="#0e1117" strokeWidth={1.5} />
              {/* Pulse for largest exporter and importer */}
              {(r === exporters[0] || r === importers[0]) && (
                <circle cx={x} cy={y} r={radius}
                        fill="none" stroke={colour} strokeWidth={1}>
                  <animate attributeName="r" from={radius} to={radius + 12}
                           dur="2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" from={0.7} to={0}
                           dur="2s" repeatCount="indefinite" />
                </circle>
              )}
              <text x={x} y={y - radius - 6} textAnchor="middle"
                    fontSize={isHovered ? 13 : 11}
                    fill={isHovered ? "#f3f4f6" : "#e5e7eb"}
                    fontWeight={isHovered ? 700 : 600}
                    fontFamily="JetBrains Mono"
                    style={{
                      paintOrder: "stroke",
                      stroke: "#0e1117",
                      strokeWidth: 3,
                    }}>
                {r.region}
              </text>
              {isHovered && (
                <text x={x} y={y + radius + 14} textAnchor="middle" fontSize={10}
                      fill={colour} fontFamily="JetBrains Mono" fontWeight={600}
                      style={{
                        paintOrder: "stroke",
                        stroke: "#0e1117",
                        strokeWidth: 3,
                      }}>
                  {r.net_trade > 0 ? "+" : ""}{r.net_trade.toFixed(1)} {unit}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="flex items-center justify-between mt-3 text-[11px] text-ink-200">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-pos" /> Net exporter
          </span>
          <span className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-neg" /> Net importer
          </span>
          <span className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-ink-300" /> Balanced
          </span>
        </div>
        <div className="text-ink-300 italic">
          Hover a region to highlight its flows · marker size ∝ |net trade|
        </div>
      </div>
    </div>
  );
}
