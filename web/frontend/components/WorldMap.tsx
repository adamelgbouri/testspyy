"use client";
import { useState } from "react";
import type { RegionalRow } from "@/lib/api";

type Props = { rows: RegionalRow[]; unit: string };

const MAP_W = 1000;
const MAP_H = 500;

/** Equirectangular projection (longitude → x, latitude → y). */
const proj = (lon: number, lat: number): [number, number] => [
  ((lon + 180) / 360) * MAP_W,
  ((90 - lat) / 180) * MAP_H,
];

/** Approximate centroids of each region used in the platform. */
const REGION_COORDS: Record<string, [number, number]> = {
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
  "Asia": [105, 25],
  "Japan/Korea": [135, 36],
  "Taiwan": [121, 24],
  "Australia": [134, -25],
  "Chile & Peru": [-72, -20],
  "DRC & Zambia": [25, -10],
  "Ukraine": [32, 49],
  "Africa": [20, 5],
  "Indonesia": [120, -2],
  "LNG export": [-90, 30],
  "Power": [-95, 38],
  "Industrial": [-95, 38],
  "OECD ETFs": [0, 50],
  "OECD ETFs / investment": [0, 50],
  "OECD": [0, 50],
  "Central Banks": [-20, 35],
  "Black Sea": [37, 44],
  "Rest of World": [0, 0],   // overridden dynamically
};

/**
 * "Rest of World" → spread across continents that AREN'T already covered
 * by another listed region in the data.
 */
const CONTINENT_HUBS: { name: string; lon: number; lat: number; keywords: string[] }[] = [
  { name: "N. America",   lon: -95,  lat: 38,  keywords: ["us", "canada", "mexico", "north america"] },
  { name: "S. America",   lon: -60,  lat: -15, keywords: ["brazil", "argentina", "chile", "south america", "peru"] },
  { name: "Europe",       lon: 15,   lat: 50,  keywords: ["europe", "eu", "russia", "ukraine", "black sea", "oecd"] },
  { name: "Africa",       lon: 20,   lat: 5,   keywords: ["africa", "drc", "zambia", "south africa"] },
  { name: "Mid. East",    lon: 45,   lat: 25,  keywords: ["middle east"] },
  { name: "Asia",         lon: 100,  lat: 30,  keywords: ["china", "india", "asia", "japan", "korea", "taiwan", "indonesia"] },
  { name: "Oceania",      lon: 135,  lat: -25, keywords: ["australia", "oceania", "new zealand"] },
];

function computeRoWHubs(regions: string[]): { lon: number; lat: number; name: string }[] {
  const lowered = regions.map((r) => r.toLowerCase());
  const uncovered = CONTINENT_HUBS.filter(
    (h) => !h.keywords.some((kw) => lowered.some((r) => r.includes(kw)))
  );
  return uncovered.length > 0
    ? uncovered.map(({ lon, lat, name }) => ({ lon, lat, name }))
    : [{ lon: 0, lat: 0, name: "RoW" }];
}

export function WorldMap({ rows, unit }: Props) {
  const [hovered, setHovered] = useState<string | null>(null);

  const exporters = rows.filter((r) => r.net_trade > 0).sort((a, b) => b.net_trade - a.net_trade);
  const importers = rows.filter((r) => r.net_trade < 0).sort((a, b) => a.net_trade - b.net_trade);
  const totalExports = exporters.reduce((s, r) => s + r.net_trade, 0);
  const totalImports = importers.reduce((s, r) => s + Math.abs(r.net_trade), 0);

  /** Region → array of (lon,lat). Most regions: a single point. RoW: many. */
  const regionPositions: Record<string, [number, number][]> = {};
  const otherRegions = rows.map((r) => r.region).filter((r) => r !== "Rest of World");
  for (const r of rows) {
    if (r.region === "Rest of World") {
      regionPositions[r.region] = computeRoWHubs(otherRegions).map((h) => [h.lon, h.lat]);
    } else {
      const c = REGION_COORDS[r.region];
      regionPositions[r.region] = c ? [c] : [[0, 0]];
    }
  }

  const flows: { from: string; to: string; volume: number;
                 fromPt: [number, number]; toPt: [number, number] }[] = [];
  for (const e of exporters) {
    for (const im of importers) {
      const share = Math.abs(im.net_trade) / Math.max(totalImports, 1);
      const vol = e.net_trade * share;
      if (vol <= 0) continue;
      const fromPts = regionPositions[e.region];
      const toPts = regionPositions[im.region];
      // Distribute equally across split points (e.g. RoW)
      const splits = fromPts.length * toPts.length;
      for (const f of fromPts) {
        for (const t of toPts) {
          flows.push({
            from: e.region, to: im.region, volume: vol / splits,
            fromPt: f, toPt: t,
          });
        }
      }
    }
  }
  const maxFlow = Math.max(...flows.map((f) => f.volume), 0.1);
  const maxNet = Math.max(...rows.map((r) => Math.abs(r.net_trade)), 0.1);

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${MAP_W} ${MAP_H}`} className="w-full bg-ink-900 rounded-lg border border-ink-600">
        <defs>
          <marker id="arrow-flow" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#22c55e" />
          </marker>
          <radialGradient id="ocean-glow">
            <stop offset="0%" stopColor="rgba(14,165,233,0.05)" />
            <stop offset="60%" stopColor="rgba(14,116,144,0.03)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0)" />
          </radialGradient>
        </defs>

        {/* Ocean background */}
        <rect width={MAP_W} height={MAP_H} fill="#0b1623" />
        <rect width={MAP_W} height={MAP_H} fill="url(#ocean-glow)" />

        {/* Lat/Lon grid */}
        <g stroke="#1a2837" strokeWidth={0.6}>
          {[-60, -30, 0, 30, 60].map((lat) => {
            const [, y] = proj(0, lat);
            return <line key={lat} x1={0} y1={y} x2={MAP_W} y2={y} strokeDasharray="2 6" />;
          })}
          {[-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150].map((lon) => {
            const [x] = proj(lon, 0);
            return <line key={lon} x1={x} y1={0} x2={x} y2={MAP_H} strokeDasharray="2 6" />;
          })}
          {/* Equator (darker) */}
          <line x1={0} y1={MAP_H / 2} x2={MAP_W} y2={MAP_H / 2}
                stroke="#2b3e54" strokeWidth={0.8} strokeDasharray="4 6" />
        </g>

        {/* CONTINENTS - simplified but proportional shapes */}
        <g fill="#1e2a3a" stroke="#2c3e52" strokeWidth={0.6}>
          {/* Greenland */}
          <path d="M 388,52 Q 405,40 430,46 Q 442,56 444,74 Q 438,86 422,90 Q 402,88 392,72 Q 386,62 388,52 Z" />

          {/* North America (Alaska + Canada + US + Mexico + Central America) */}
          <path d="M 110,75 Q 138,62 178,68 Q 215,72 248,80
                   Q 285,82 318,90 Q 354,92 374,105
                   Q 372,128 358,150 Q 348,175 342,202
                   Q 338,230 330,254 Q 316,275 300,290
                   Q 292,300 285,310
                   L 280,326 Q 274,332 268,320
                   Q 260,310 248,302 Q 230,290 216,280
                   Q 200,272 188,256 Q 176,238 168,222
                   Q 158,206 148,194 Q 140,180 132,166
                   Q 124,148 118,130 Q 110,108 110,90 Z" />

          {/* Central America bridge */}
          <path d="M 270,318 Q 280,326 290,336 Q 298,345 302,355 Q 305,365 305,376 Z" />

          {/* South America */}
          <path d="M 295,355 Q 318,348 342,360 Q 360,375 368,400
                   Q 372,430 366,460 Q 358,490 348,500
                   Q 332,500 322,488 Q 312,470 308,450
                   Q 300,432 296,415 Q 290,398 292,380 Z" />

          {/* Africa - LARGE, proper proportions (~30M km² in reality) */}
          <path d="M 478,180 Q 510,178 540,182 Q 565,188 580,200
                   Q 590,218 588,240 Q 586,262 582,284
                   Q 580,306 574,326 Q 570,348 562,370
                   Q 556,392 544,408 Q 530,420 514,422
                   Q 500,420 488,406 Q 478,388 472,366
                   Q 466,344 462,322 Q 458,300 456,278
                   Q 458,254 462,232 Q 466,210 472,194 Z" />

          {/* Europe - small but distinct */}
          <path d="M 460,108 Q 488,102 516,112 Q 540,124 548,142
                   Q 552,156 546,168 Q 532,176 510,178
                   Q 488,176 472,170 Q 458,158 456,142
                   Q 454,124 460,108 Z" />
          {/* UK & Ireland */}
          <path d="M 446,128 Q 456,126 462,138 Q 460,150 452,154 Q 442,152 440,142 Q 440,132 446,128 Z" />
          {/* Scandinavia */}
          <path d="M 504,84 Q 526,80 540,96 Q 544,114 538,128 Q 528,138 514,134 Q 502,124 500,108 Q 500,94 504,84 Z" />

          {/* Russia & northern Asia - HUGE */}
          <path d="M 546,84 Q 600,76 680,72 Q 760,72 832,80 Q 880,90 902,102
                   Q 904,118 894,134 Q 870,142 836,140 Q 790,138 740,134
                   Q 690,130 644,128 Q 608,128 580,130
                   Q 558,128 548,114 Q 542,98 546,84 Z" />

          {/* Middle East */}
          <path d="M 562,178 Q 598,176 624,184 Q 644,196 648,214
                   Q 644,228 626,232 Q 600,232 580,228
                   Q 564,222 556,210 Q 556,194 562,178 Z" />

          {/* China & East Asia */}
          <path d="M 728,148 Q 768,142 814,148 Q 858,158 874,178
                   Q 882,196 878,216 Q 870,234 854,244
                   Q 832,252 808,250 Q 778,246 758,238
                   Q 740,228 728,210 Q 724,192 728,176 Z" />

          {/* India */}
          <path d="M 704,224 Q 730,222 748,232 Q 758,250 756,270
                   Q 752,290 740,304 Q 726,310 718,302
                   Q 712,288 708,274 Q 704,256 700,240 Z" />

          {/* Southeast Asia / Indonesia / Philippines */}
          <path d="M 798,266 Q 832,264 866,272 Q 884,282 884,296
                   Q 876,308 854,310 Q 826,310 806,308
                   Q 790,302 786,290 Q 788,274 798,266 Z" />

          {/* Australia */}
          <path d="M 836,360 Q 868,354 902,358 Q 928,368 936,386
                   Q 938,404 924,418 Q 902,424 872,422
                   Q 848,418 834,406 Q 826,388 830,372 Q 832,362 836,360 Z" />

          {/* New Zealand */}
          <path d="M 942,420 Q 950,418 954,428 Q 952,438 942,436 Q 938,428 942,420 Z" />

          {/* Japan */}
          <path d="M 892,170 Q 904,166 914,178 Q 916,196 906,204 Q 894,202 890,188 Q 888,176 892,170 Z" />
        </g>

        {/* Flow arrows */}
        {flows.map((f, idx) => {
          const [x1, y1] = proj(f.fromPt[0], f.fromPt[1]);
          const [x2, y2] = proj(f.toPt[0], f.toPt[1]);
          // Bezier control point above midpoint - curvature ∝ horizontal distance
          const mx = (x1 + x2) / 2;
          const curvature = Math.abs(x2 - x1) * 0.3 + 25;
          const my = Math.min(y1, y2) - curvature;
          const opacity = 0.15 + (f.volume / maxFlow) * 0.5;
          const width = 0.6 + (f.volume / maxFlow) * 3.2;
          const visible = hovered === null || hovered === f.from || hovered === f.to;
          return (
            <path key={idx}
              d={`M ${x1},${y1} Q ${mx},${my} ${x2},${y2}`}
              fill="none"
              stroke="#22c55e"
              strokeWidth={width}
              opacity={visible ? opacity : 0.04}
              markerEnd="url(#arrow-flow)"
              style={{ transition: "opacity 0.2s" }}
            />
          );
        })}

        {/* Region markers */}
        {rows.flatMap((r) => {
          const positions = regionPositions[r.region];
          const ratio = Math.abs(r.net_trade) / maxNet;
          const radius = 5 + ratio * 14;
          const colour = r.status === "exporter" ? "#22c55e"
                       : r.status === "importer" ? "#ef4444" : "#9ca3af";
          const isHovered = hovered === r.region;
          const showLabelOnce = positions.length === 1 || r.region === "Rest of World";

          return positions.map(([lon, lat], idx) => {
            const [x, y] = proj(lon, lat);
            const radiusHere = positions.length > 1 ? Math.max(4, radius * 0.55) : radius;
            return (
              <g key={`${r.region}-${idx}`}
                 onMouseEnter={() => setHovered(r.region)}
                 onMouseLeave={() => setHovered(null)}
                 style={{ cursor: "pointer" }}>
                <circle cx={x} cy={y} r={radiusHere + 5}
                        fill={colour} opacity={0.18} />
                <circle cx={x} cy={y} r={radiusHere} fill={colour} opacity={0.78}
                        stroke="#0e1117" strokeWidth={1.5} />
                {(r === exporters[0] || r === importers[0]) && idx === 0 && (
                  <circle cx={x} cy={y} r={radiusHere}
                          fill="none" stroke={colour} strokeWidth={1}>
                    <animate attributeName="r" from={radiusHere} to={radiusHere + 12}
                             dur="2s" repeatCount="indefinite" />
                    <animate attributeName="opacity" from={0.7} to={0}
                             dur="2s" repeatCount="indefinite" />
                  </circle>
                )}
                {(showLabelOnce ? idx === 0 : true) && (
                  <text x={x} y={y - radiusHere - 5} textAnchor="middle"
                        fontSize={isHovered ? 12 : 10}
                        fill={isHovered ? "#f3f4f6" : "#d1d5db"}
                        fontWeight={isHovered ? 700 : 600}
                        fontFamily="JetBrains Mono"
                        style={{
                          paintOrder: "stroke",
                          stroke: "#0b1623",
                          strokeWidth: 3,
                        }}>
                    {r.region}
                  </text>
                )}
                {isHovered && idx === 0 && (
                  <text x={x} y={y + radiusHere + 14} textAnchor="middle" fontSize={10}
                        fill={colour} fontFamily="JetBrains Mono" fontWeight={700}
                        style={{
                          paintOrder: "stroke",
                          stroke: "#0b1623",
                          strokeWidth: 3,
                        }}>
                    {r.net_trade > 0 ? "+" : ""}{r.net_trade.toFixed(1)} {unit}
                  </text>
                )}
              </g>
            );
          });
        })}
      </svg>

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
          Hover a region · marker size ∝ |net trade| · &quot;Rest of World&quot; split across uncited continents
        </div>
      </div>
    </div>
  );
}
