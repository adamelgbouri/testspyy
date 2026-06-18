"use client";
import { useEffect, useMemo, useState } from "react";
import { geoCylindricalEqualArea, geoPath, type GeoProjection } from "d3-geo";
import { feature } from "topojson-client";
import type { FeatureCollection, Geometry } from "geojson";
import type { Topology } from "topojson-specification";
import type { RegionalRow } from "@/lib/api";

type Props = { rows: RegionalRow[]; unit: string };

/* ------------------------------------------------------------- */
/*  Geography                                                    */
/* ------------------------------------------------------------- */
const TOPOJSON_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

// ISO 3166-1 numeric codes (matching world-atlas) for the platform's regions
// and continent fallback hubs.
const ISO_BY_NAME: Record<string, string[]> = {
  "US":               ["840"],
  "Canada":           ["124"],
  "Mexico":           ["484"],
  "US Northeast":     ["840"],
  "Brazil":           ["076"],
  "Argentina":        ["032"],
  "Chile & Peru":     ["152", "604"],
  "Europe":           ["276", "250", "724", "380", "528", "208"],   // DE FR ES IT NL DK
  "EU":               ["276", "250", "724", "380", "528"],
  "Russia & CIS":     ["643", "398"],
  "Russia":           ["643"],
  "Middle East":      ["682", "364", "368", "784", "414", "634"],   // SA IR IQ AE KW QA
  "China":            ["156"],
  "China (mine)":     ["156"],
  "India":            ["356"],
  "India (consumer)": ["356"],
  "Other Asia":       ["704", "458", "608", "764"],                 // VN MY PH TH
  "Asia":             ["156", "356", "392"],
  "Japan/Korea":      ["392", "410"],
  "Taiwan":           ["158"],
  "Australia":        ["036"],
  "DRC & Zambia":     ["180", "894"],
  "Ukraine":          ["804"],
  "Africa":           ["818", "566", "012", "710"],                 // EG NG DZ ZA
  "Indonesia":        ["360"],
  "Black Sea":        ["804", "642", "100"],                        // UA RO BG
  "OECD ETFs":        ["840", "276", "826"],                        // US DE GB
  "OECD ETFs / investment": ["840", "276", "826"],
  "OECD":             ["840", "276", "826"],
  "Central Banks":    ["840", "276", "826", "392"],
  "LNG export":       ["840"],
  "Power":            ["840"],
  "Industrial":       ["840"],
  "Mexico (export)":  ["484"],
};

// Continent-level fallback hubs (lon, lat) used to spread "Rest of World"
// across continents that aren't already cited.
const CONTINENT_HUBS = [
  { name: "N. America",  lon: -98,  lat: 40,  keywords: ["us", "canada", "mexico", "north america"] },
  { name: "S. America",  lon: -60,  lat: -15, keywords: ["brazil", "argentina", "chile", "peru", "south america"] },
  { name: "Europe",      lon: 15,   lat: 50,  keywords: ["europe", "eu", "russia", "ukraine", "black sea", "oecd"] },
  { name: "Africa",      lon: 20,   lat: 5,   keywords: ["africa", "drc", "zambia", "south africa"] },
  { name: "Middle East", lon: 45,   lat: 25,  keywords: ["middle east"] },
  { name: "Asia",        lon: 100,  lat: 30,  keywords: ["china", "india", "asia", "japan", "korea", "taiwan", "indonesia"] },
  { name: "Oceania",     lon: 140,  lat: -25, keywords: ["australia", "oceania", "new zealand"] },
];

// Approx country/region centroid for marker placement (lon, lat)
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
  "Mexico (export)": [-102, 23],
};

function computeRoWHubs(otherRegions: string[]): { lon: number; lat: number; name: string }[] {
  const lowered = otherRegions.map((r) => r.toLowerCase());
  const uncovered = CONTINENT_HUBS.filter(
    (h) => !h.keywords.some((kw) => lowered.some((r) => r.includes(kw)))
  );
  return uncovered.length > 0
    ? uncovered.map(({ lon, lat, name }) => ({ lon, lat, name }))
    : [{ lon: 0, lat: 0, name: "RoW" }];
}

/* ------------------------------------------------------------- */
/*  Component                                                    */
/* ------------------------------------------------------------- */
export function WorldMap({ rows, unit }: Props) {
  const [topo, setTopo] = useState<Topology | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(TOPOJSON_URL)
      .then((r) => r.json())
      .then((j: Topology) => { if (!cancelled) setTopo(j); })
      .catch((e) => console.error("world-atlas fetch failed:", e));
    return () => { cancelled = true; };
  }, []);

  /** Gall-Peters projection: cylindrical equal-area at parallel 45°. */
  const W = 1000;
  const H = 500;
  const projection = useMemo<GeoProjection>(() => {
    return geoCylindricalEqualArea()
      .parallel(45)
      .scale(150)
      .translate([W / 2, H / 2]);
  }, []);

  const pathGen = useMemo(() => geoPath(projection as any), [projection]);

  const countries = useMemo(() => {
    if (!topo) return null;
    const fc = feature(topo, topo.objects.countries as any) as unknown as FeatureCollection<Geometry, { name: string }>;
    return fc.features;
  }, [topo]);

  /** Highlight set = numeric ISO codes for hovered region. */
  const highlightedISO = useMemo(() => {
    if (!hovered) return new Set<string>();
    return new Set(ISO_BY_NAME[hovered] ?? []);
  }, [hovered]);

  /* ----- Trade flow computation ----- */
  const exporters = rows.filter((r) => r.net_trade > 0).sort((a, b) => b.net_trade - a.net_trade);
  const importers = rows.filter((r) => r.net_trade < 0).sort((a, b) => a.net_trade - b.net_trade);
  const totalExports = exporters.reduce((s, r) => s + r.net_trade, 0);
  const totalImports = importers.reduce((s, r) => s + Math.abs(r.net_trade), 0);

  /** For each region: list of (lon,lat) markers. */
  const regionPositions = useMemo(() => {
    const map: Record<string, [number, number][]> = {};
    const others = rows.map((r) => r.region).filter((r) => r !== "Rest of World");
    for (const r of rows) {
      if (r.region === "Rest of World") {
        map[r.region] = computeRoWHubs(others).map((h) => [h.lon, h.lat]);
      } else {
        const c = REGION_COORDS[r.region];
        map[r.region] = c ? [c] : [[0, 0]];
      }
    }
    return map;
  }, [rows]);

  const flows = useMemo(() => {
    const out: {
      from: string; to: string; volume: number;
      fromXY: [number, number]; toXY: [number, number];
    }[] = [];
    for (const e of exporters) {
      for (const im of importers) {
        const share = Math.abs(im.net_trade) / Math.max(totalImports, 1);
        const vol = e.net_trade * share;
        if (vol <= 0) continue;
        const fps = regionPositions[e.region];
        const tps = regionPositions[im.region];
        const splits = fps.length * tps.length;
        for (const f of fps) {
          for (const t of tps) {
            const fxy = projection(f) as [number, number] | null;
            const txy = projection(t) as [number, number] | null;
            if (!fxy || !txy) continue;
            out.push({
              from: e.region, to: im.region,
              volume: vol / splits, fromXY: fxy, toXY: txy,
            });
          }
        }
      }
    }
    return out;
  }, [exporters, importers, regionPositions, projection, totalImports]);

  const maxFlow = Math.max(...flows.map((f) => f.volume), 0.1);
  const maxNet = Math.max(...rows.map((r) => Math.abs(r.net_trade)), 0.1);

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-ink-600"
           style={{ background: "#0a1320" }}>
        <defs>
          <marker id="arrow-flow" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#22c55e" />
          </marker>
          <linearGradient id="ocean-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0c1828" />
            <stop offset="100%" stopColor="#080f1a" />
          </linearGradient>
        </defs>

        <rect width={W} height={H} fill="url(#ocean-grad)" />

        {/* Lat/Lon graticule */}
        <g stroke="#16263a" strokeWidth={0.5} fill="none" opacity={0.8}>
          {[-60, -30, 0, 30, 60].map((lat) => {
            const path = pathGen({
              type: "LineString",
              coordinates: [[-180, lat], [-120, lat], [-60, lat], [0, lat],
                            [60, lat], [120, lat], [180, lat]],
            } as any);
            return path ? <path key={lat} d={path} strokeDasharray={lat === 0 ? "" : "2 6"}
                                stroke={lat === 0 ? "#243f60" : undefined} /> : null;
          })}
          {[-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150].map((lon) => {
            const path = pathGen({
              type: "LineString",
              coordinates: [[lon, -85], [lon, 85]],
            } as any);
            return path ? <path key={lon} d={path} strokeDasharray="2 6" /> : null;
          })}
        </g>

        {/* Countries */}
        {countries ? (
          <g>
            {countries.map((f) => {
              const d = pathGen(f as any);
              if (!d) return null;
              const id = String((f as any).id);
              const lit = highlightedISO.has(id);
              return (
                <path key={id} d={d}
                      fill={lit ? "#2a3f5e" : "#172638"}
                      stroke="#22354c" strokeWidth={0.4}
                      style={{ transition: "fill 0.15s" }}>
                  <title>{(f.properties as any).name}</title>
                </path>
              );
            })}
          </g>
        ) : (
          <text x={W / 2} y={H / 2} fill="#9ca3af" textAnchor="middle"
                fontSize={14} fontFamily="JetBrains Mono">
            Loading world atlas…
          </text>
        )}

        {/* Flow arrows */}
        {flows.map((f, idx) => {
          const [x1, y1] = f.fromXY;
          const [x2, y2] = f.toXY;
          const mx = (x1 + x2) / 2;
          const curvature = Math.abs(x2 - x1) * 0.32 + 20;
          const my = Math.min(y1, y2) - curvature;
          const opacity = 0.15 + (f.volume / maxFlow) * 0.55;
          const width = 0.6 + (f.volume / maxFlow) * 3.2;
          const visible = hovered === null || hovered === f.from || hovered === f.to;
          return (
            <path key={idx}
              d={`M ${x1},${y1} Q ${mx},${my} ${x2},${y2}`}
              fill="none"
              stroke="#22c55e"
              strokeWidth={width}
              opacity={visible ? opacity : 0.05}
              markerEnd="url(#arrow-flow)"
              style={{ transition: "opacity 0.2s" }}
            />
          );
        })}

        {/* Region markers */}
        {rows.flatMap((r) => {
          const positions = regionPositions[r.region];
          const ratio = Math.abs(r.net_trade) / maxNet;
          const baseRadius = 6 + ratio * 14;
          const colour = r.status === "exporter" ? "#22c55e"
                       : r.status === "importer" ? "#ef4444" : "#9ca3af";
          const isHovered = hovered === r.region;

          return positions.map((pt, idx) => {
            const xy = projection(pt) as [number, number] | null;
            if (!xy) return null;
            const [x, y] = xy;
            const radius = positions.length > 1 ? Math.max(5, baseRadius * 0.6) : baseRadius;
            return (
              <g key={`${r.region}-${idx}`}
                 onMouseEnter={() => setHovered(r.region)}
                 onMouseLeave={() => setHovered(null)}
                 style={{ cursor: "pointer" }}>
                <circle cx={x} cy={y} r={radius + 5}
                        fill={colour} opacity={0.18} />
                <circle cx={x} cy={y} r={radius} fill={colour} opacity={0.85}
                        stroke="#0a1320" strokeWidth={1.5} />
                {(r === exporters[0] || r === importers[0]) && idx === 0 && (
                  <circle cx={x} cy={y} r={radius}
                          fill="none" stroke={colour} strokeWidth={1}>
                    <animate attributeName="r" from={radius} to={radius + 14}
                             dur="2s" repeatCount="indefinite" />
                    <animate attributeName="opacity" from={0.7} to={0}
                             dur="2s" repeatCount="indefinite" />
                  </circle>
                )}
                {idx === 0 && (
                  <text x={x} y={y - radius - 5} textAnchor="middle"
                        fontSize={isHovered ? 12 : 10}
                        fill={isHovered ? "#f3f4f6" : "#d1d5db"}
                        fontWeight={isHovered ? 700 : 600}
                        fontFamily="JetBrains Mono"
                        style={{ paintOrder: "stroke", stroke: "#0a1320", strokeWidth: 3 }}>
                    {r.region}
                  </text>
                )}
                {isHovered && idx === 0 && (
                  <text x={x} y={y + radius + 14} textAnchor="middle" fontSize={10}
                        fill={colour} fontFamily="JetBrains Mono" fontWeight={700}
                        style={{ paintOrder: "stroke", stroke: "#0a1320", strokeWidth: 3 }}>
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
        <div className="text-ink-300 italic font-mono">
          Gall-Peters equal-area projection · world-atlas 110m · {countries?.length ?? "—"} countries
        </div>
      </div>
    </div>
  );
}
