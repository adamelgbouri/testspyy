import { api } from "@/lib/api";
import { CommoditySelector } from "@/components/CommoditySelector";
import { KPICard } from "@/components/KPICard";
import { fmtNum } from "@/lib/utils";
import { Globe, TrendingUp, TrendingDown } from "lucide-react";

export const dynamic = "force-dynamic";

type Props = { searchParams: { c?: string } };

export default async function RegionalPage({ searchParams }: Props) {
  const commodities = await api.commodities();
  const key = searchParams.c ?? commodities[0]?.key ?? "wti_crude";
  const regional = await api.regional(key);
  const commodity = commodities.find((c) => c.key === key)!;

  const exporters = regional.rows
    .filter((r) => r.net_trade > 0)
    .sort((a, b) => b.net_trade - a.net_trade);
  const importers = regional.rows
    .filter((r) => r.net_trade < 0)
    .sort((a, b) => a.net_trade - b.net_trade);

  const totalExports = exporters.reduce((s, r) => s + r.net_trade, 0);
  const totalImports = importers.reduce((s, r) => s + Math.abs(r.net_trade), 0);

  // For bar visualisation (max share)
  const maxShare = Math.max(...regional.rows.map((r) => Math.max(r.supply_share_pct, r.demand_share_pct)));

  return (
    <div className="space-y-6 animate-slide-up">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Regional Flows — {commodity.name}</h1>
          <p className="text-sm text-ink-200 mt-1">
            World production vs consumption, market shares, exporters and
            importers. Source: typical IEA / EIA / USDA / WGC / ICSG splits.
          </p>
        </div>
        <CommoditySelector commodities={commodities} current={key} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 stagger">
        <KPICard label={`World supply (${regional.unit})`}
          value={fmtNum(regional.world_supply, 1)} icon={<Globe size={11} />} />
        <KPICard label={`World demand (${regional.unit})`}
          value={fmtNum(regional.world_demand, 1)} icon={<Globe size={11} />} />
        <KPICard label={`Global balance (${regional.unit})`}
          value={`${regional.world_balance > 0 ? "+" : ""}${fmtNum(regional.world_balance, 1)}`}
          deltaTone={regional.world_balance >= 0 ? "pos" : "neg"}
          delta="+ surplus / − deficit" />
        <KPICard label={`Implied trade (${regional.unit})`}
          value={fmtNum(totalExports, 1)}
          delta={`${exporters.length} exporters → ${importers.length} importers`} />
      </div>

      {/* Share visualization */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Market share — supply vs demand</h2>
        <div className="space-y-2">
          {regional.rows.map((r) => (
            <div key={r.region} className="grid grid-cols-12 gap-3 items-center text-xs">
              <div className="col-span-3 text-ink-50 truncate">{r.region}</div>
              <div className="col-span-4">
                <div className="flex items-center gap-2">
                  <span className="text-pos font-mono w-12 text-right">{r.supply_share_pct.toFixed(1)}%</span>
                  <div className="flex-1 h-2.5 bg-ink-700 rounded-full overflow-hidden">
                    <div className="h-full bg-pos rounded-full transition-all"
                      style={{ width: `${(r.supply_share_pct / maxShare) * 100}%` }} />
                  </div>
                </div>
              </div>
              <div className="col-span-4">
                <div className="flex items-center gap-2">
                  <span className="text-neg font-mono w-12 text-right">{r.demand_share_pct.toFixed(1)}%</span>
                  <div className="flex-1 h-2.5 bg-ink-700 rounded-full overflow-hidden">
                    <div className="h-full bg-neg rounded-full transition-all"
                      style={{ width: `${(r.demand_share_pct / maxShare) * 100}%` }} />
                  </div>
                </div>
              </div>
              <div className="col-span-1">
                <span className={`badge ${
                  r.status === "exporter" ? "border-pos/40 text-pos" :
                  r.status === "importer" ? "border-neg/40 text-neg" : ""
                }`}>{r.status}</span>
              </div>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-4 mt-4 pt-3 border-t border-ink-600 text-[11px] text-ink-200">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2.5 bg-pos rounded-sm" /> Supply share
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2.5 bg-neg rounded-sm" /> Demand share
          </span>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Exporters */}
        <div className="card p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-pos">🟢 Net exporters</h2>
            {exporters[0] && (
              <span className="text-[11px] text-ink-200 italic">
                Top: {exporters[0].region}
              </span>
            )}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-ink-200 border-b border-ink-600">
                  <th className="text-left pb-2">Region</th>
                  <th className="text-right pb-2">Supply</th>
                  <th className="text-right pb-2">Demand</th>
                  <th className="text-right pb-2 text-pos">Net exports</th>
                  <th className="text-right pb-2">World share %</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {exporters.map((r) => (
                  <tr key={r.region} className="border-b border-ink-700/60">
                    <td className="py-1.5 text-ink-50">{r.region}</td>
                    <td className="text-right">{fmtNum(r.supply, 2)}</td>
                    <td className="text-right">{fmtNum(r.demand, 2)}</td>
                    <td className="text-right text-pos">+{fmtNum(r.net_trade, 2)}</td>
                    <td className="text-right">
                      {totalExports > 0 ? ((r.net_trade / totalExports) * 100).toFixed(1) : "—"}
                    </td>
                  </tr>
                ))}
                {exporters.length === 0 && (
                  <tr><td colSpan={5} className="text-center text-ink-200 py-3">No net exporter</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Importers */}
        <div className="card p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-neg">🔴 Net importers</h2>
            {importers[0] && (
              <span className="text-[11px] text-ink-200 italic">
                Top: {importers[0].region}
              </span>
            )}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-ink-200 border-b border-ink-600">
                  <th className="text-left pb-2">Region</th>
                  <th className="text-right pb-2">Supply</th>
                  <th className="text-right pb-2">Demand</th>
                  <th className="text-right pb-2 text-neg">Net imports</th>
                  <th className="text-right pb-2">World share %</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {importers.map((r) => (
                  <tr key={r.region} className="border-b border-ink-700/60">
                    <td className="py-1.5 text-ink-50">{r.region}</td>
                    <td className="text-right">{fmtNum(r.supply, 2)}</td>
                    <td className="text-right">{fmtNum(r.demand, 2)}</td>
                    <td className="text-right text-neg">
                      {fmtNum(Math.abs(r.net_trade), 2)}
                    </td>
                    <td className="text-right">
                      {totalImports > 0 ? ((Math.abs(r.net_trade) / totalImports) * 100).toFixed(1) : "—"}
                    </td>
                  </tr>
                ))}
                {importers.length === 0 && (
                  <tr><td colSpan={5} className="text-center text-ink-200 py-3">No net importer</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Market shares */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Market shares</h2>
        <p className="text-[11px] text-ink-200 italic mb-3">
          Each region's share of world production and consumption.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-ink-200 border-b border-ink-600">
                <th className="text-left pb-2">Region</th>
                <th className="text-right pb-2">Supply share %</th>
                <th className="text-right pb-2">Demand share %</th>
                <th className="text-right pb-2">Net trade</th>
                <th className="text-left pb-2 pl-3">Status</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {regional.rows.map((r) => (
                <tr key={r.region} className="border-b border-ink-700/60">
                  <td className="py-1.5 text-ink-50">{r.region}</td>
                  <td className="text-right">{r.supply_share_pct.toFixed(1)}</td>
                  <td className="text-right">{r.demand_share_pct.toFixed(1)}</td>
                  <td className={`text-right ${
                    r.net_trade > 0 ? "text-pos" : r.net_trade < 0 ? "text-neg" : ""
                  }`}>
                    {r.net_trade > 0 ? "+" : ""}{fmtNum(r.net_trade, 2)}
                  </td>
                  <td className="pl-3">
                    <span className={`badge ${
                      r.status === "exporter" ? "border-pos/40 text-pos"
                        : r.status === "importer" ? "border-neg/40 text-neg" : ""
                    }`}>
                      {r.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
