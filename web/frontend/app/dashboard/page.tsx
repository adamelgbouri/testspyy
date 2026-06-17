import { api } from "@/lib/api";
import { CommoditySelector } from "@/components/CommoditySelector";
import { KPICard } from "@/components/KPICard";
import { BalanceChart } from "@/components/BalanceChart";
import { fmtNum, fmtPct, fmtPrice } from "@/lib/utils";

export const dynamic = "force-dynamic";

type Props = { searchParams: { c?: string } };

export default async function DashboardPage({ searchParams }: Props) {
  const commodities = await api.commodities();
  const key = searchParams.c ?? commodities[0]?.key ?? "wti_crude";

  const [spot, balance, regional] = await Promise.all([
    api.spot(key),
    api.balance(key, { forecast_months: 18 }),
    api.regional(key),
  ]);

  const commodity = commodities.find((c) => c.key === key)!;
  const upDelta = spot.change_pct >= 0;

  // Last historic point for spot vs fair value
  const lastHist = [...balance.points].reverse().find((p) => !p.is_forecast);
  const fvNow = lastHist?.fair_value ?? balance.end_fair_value;
  const fvDeviation = ((spot.price - fvNow) / fvNow) * 100;

  return (
    <div className="space-y-6">
      {/* Header row */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">{commodity.name}</h1>
          <p className="text-sm text-ink-200">
            Sector: <span className="text-ink-100">{commodity.sector}</span> ·{" "}
            Futures contract:{" "}
            <span className="text-ink-100 font-mono">{commodity.ticker}</span> ·{" "}
            Quote: <span className="text-ink-100">{commodity.price_unit}</span>
          </p>
        </div>
        <CommoditySelector commodities={commodities} current={key} />
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <KPICard
          label="Spot price"
          value={fmtPrice(spot.price, spot.price_unit)}
          delta={`${fmtPct(spot.change_pct)} · ${
            spot.source === "yahoo" ? "live " + spot.asof : "reference"
          }`}
          deltaTone={upDelta ? "pos" : "neg"}
        />
        <KPICard
          label="Fair value"
          value={fmtPrice(fvNow, commodity.price_unit)}
          delta={`${fmtPct(fvDeviation)} vs spot`}
          deltaTone={fvDeviation > 10 ? "neg" : fvDeviation < -10 ? "pos" : "neutral"}
        />
        <KPICard
          label={`End stocks (${commodity.inventory_unit})`}
          value={fmtNum(balance.end_stocks)}
        />
        <KPICard
          label="Days of cover"
          value={balance.end_days_cover.toFixed(1)}
          delta={`target ${commodity.days_cover_target.toFixed(0)}`}
        />
        <KPICard
          label="Storage utilisation"
          value={`${balance.end_utilization_pct.toFixed(1)} %`}
          delta={`ideal ${commodity.ideal_utilization_pct.toFixed(0)} %`}
        />
      </div>

      {/* Main grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="card p-5 lg:col-span-2">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold">Supply, demand &amp; stocks</h2>
            <span className="text-[11px] text-ink-200 italic">
              Forecast period highlighted on the right of the dashed line.
            </span>
          </div>
          <BalanceChart
            points={balance.points}
            unit={commodity.unit}
            inventoryUnit={commodity.inventory_unit}
          />
        </div>

        {/* World metrics */}
        <div className="card p-5">
          <h2 className="text-sm font-semibold mb-3">Global balance</h2>
          <p className="text-[11px] text-ink-200 italic mb-3">
            World production vs consumption split with implied net trade.
          </p>
          <div className="space-y-2 font-mono text-sm">
            <Row label="World supply" value={`${fmtNum(regional.world_supply, 1)} ${regional.unit}`} />
            <Row label="World demand" value={`${fmtNum(regional.world_demand, 1)} ${regional.unit}`} />
            <Row
              label="Balance"
              value={`${regional.world_balance > 0 ? "+" : ""}${fmtNum(regional.world_balance, 1)} ${regional.unit}`}
              tone={regional.world_balance >= 0 ? "pos" : "neg"}
            />
          </div>

          <h3 className="section-title">Top exporters</h3>
          <ul className="text-xs space-y-1">
            {regional.rows
              .filter((r) => r.net_trade > 0)
              .sort((a, b) => b.net_trade - a.net_trade)
              .slice(0, 3)
              .map((r) => (
                <li key={r.region} className="flex justify-between">
                  <span className="text-ink-100">{r.region}</span>
                  <span className="text-pos font-mono">
                    +{fmtNum(r.net_trade, 1)} {regional.unit}
                  </span>
                </li>
              ))}
          </ul>

          <h3 className="section-title">Top importers</h3>
          <ul className="text-xs space-y-1">
            {regional.rows
              .filter((r) => r.net_trade < 0)
              .sort((a, b) => a.net_trade - b.net_trade)
              .slice(0, 3)
              .map((r) => (
                <li key={r.region} className="flex justify-between">
                  <span className="text-ink-100">{r.region}</span>
                  <span className="text-neg font-mono">
                    {fmtNum(r.net_trade, 1)} {regional.unit}
                  </span>
                </li>
              ))}
          </ul>
        </div>
      </div>

      {/* Regional table */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold mb-3">Regional snapshot</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-ink-200 border-b border-ink-600">
                <th className="text-left pb-2">Region</th>
                <th className="text-right pb-2">Supply</th>
                <th className="text-right pb-2">Demand</th>
                <th className="text-right pb-2">Net trade</th>
                <th className="text-right pb-2">Supply %</th>
                <th className="text-right pb-2">Demand %</th>
                <th className="text-left pb-2 pl-3">Status</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {regional.rows.map((r) => (
                <tr key={r.region} className="border-b border-ink-700/60">
                  <td className="py-1.5 text-ink-50">{r.region}</td>
                  <td className="text-right">{fmtNum(r.supply, 2)}</td>
                  <td className="text-right">{fmtNum(r.demand, 2)}</td>
                  <td
                    className={`text-right ${
                      r.net_trade > 0 ? "text-pos" : r.net_trade < 0 ? "text-neg" : ""
                    }`}
                  >
                    {r.net_trade > 0 ? "+" : ""}
                    {fmtNum(r.net_trade, 2)}
                  </td>
                  <td className="text-right">{r.supply_share_pct.toFixed(1)}</td>
                  <td className="text-right">{r.demand_share_pct.toFixed(1)}</td>
                  <td className="pl-3">
                    <span
                      className={`badge ${
                        r.status === "exporter"
                          ? "border-pos/40 text-pos"
                          : r.status === "importer"
                          ? "border-neg/40 text-neg"
                          : ""
                      }`}
                    >
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

function Row({ label, value, tone }: { label: string; value: string; tone?: "pos" | "neg" }) {
  return (
    <div className="flex justify-between">
      <span className="text-ink-200">{label}</span>
      <span
        className={
          tone === "pos" ? "text-pos" : tone === "neg" ? "text-neg" : "text-ink-50"
        }
      >
        {value}
      </span>
    </div>
  );
}
