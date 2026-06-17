import { api } from "@/lib/api";
import { CommoditySelector } from "@/components/CommoditySelector";
import { KPICard } from "@/components/KPICard";
import { BalanceChart } from "@/components/BalanceChart";
import { MarketHeatmap } from "@/components/MarketHeatmap";
import { SentimentGauge } from "@/components/SentimentGauge";
import { PulseDot } from "@/components/PulseDot";
import { fmtNum, fmtPct, fmtPrice } from "@/lib/utils";
import { TrendingUp, TrendingDown, BarChart3, Box, Droplets, Globe } from "lucide-react";

export const dynamic = "force-dynamic";

type Props = { searchParams: { c?: string } };

export default async function DashboardPage({ searchParams }: Props) {
  const commodities = await api.commodities();
  const key = searchParams.c ?? commodities[0]?.key ?? "wti_crude";

  const [spot, balance, regional, allSpots] = await Promise.all([
    api.spot(key),
    api.balance(key, { forecast_months: 18 }),
    api.regional(key),
    api.allSpots(),
  ]);

  const commodity = commodities.find((c) => c.key === key)!;
  const upDelta = spot.change_pct >= 0;
  const lastHist = [...balance.points].reverse().find((p) => !p.is_forecast);
  const fvNow = lastHist?.fair_value ?? balance.end_fair_value;
  const fvDeviation = ((spot.price - fvNow) / fvNow) * 100;

  // Sparklines: last 12 historic prices and stocks
  const histPoints = balance.points.filter((p) => !p.is_forecast).slice(-12);
  const sparkPrice = histPoints.map((p) => p.price);
  const sparkStocks = histPoints.map((p) => p.stocks);
  const sparkDC = histPoints.map((p) => p.days_cover);
  const sparkUtil = histPoints.map((p) => p.stocks / commodity.days_cover_target * 100);

  // Top movers
  const movers = [...allSpots].sort((a, b) => Math.abs(b.change_pct) - Math.abs(a.change_pct)).slice(0, 5);
  const gainers = [...allSpots].filter((s) => s.change_pct > 0).sort((a, b) => b.change_pct - a.change_pct).slice(0, 3);
  const losers = [...allSpots].filter((s) => s.change_pct < 0).sort((a, b) => a.change_pct - b.change_pct).slice(0, 3);

  // Market sentiment score (0-100): based on % of commodities up
  const upCount = allSpots.filter((s) => s.change_pct > 0).length;
  const sentiment = (upCount / allSpots.length) * 100;
  const avgChange = allSpots.reduce((s, x) => s + x.change_pct, 0) / allSpots.length;

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <PulseDot tone={upDelta ? "pos" : "neg"} />
            <span className="badge">LIVE</span>
            <span className="text-[11px] text-ink-200 font-mono">
              {spot.source === "yahoo" ? `Yahoo Finance · as of ${spot.asof}` : "Reference data"}
            </span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight">{commodity.name}</h1>
          <p className="text-sm text-ink-200 mt-1">
            <span className="text-ink-100">{commodity.sector}</span> ·{" "}
            <span className="font-mono text-ink-100">{commodity.ticker}</span> ·{" "}
            Flow {commodity.unit} · Stocks {commodity.inventory_unit} ·{" "}
            Quote {commodity.price_unit}
          </p>
        </div>
        <CommoditySelector commodities={commodities} current={key} />
      </div>

      {/* KPI strip with sparklines */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 stagger">
        <KPICard
          label="Spot price"
          icon={<TrendingUp size={11} />}
          value={fmtPrice(spot.price, spot.price_unit)}
          delta={fmtPct(spot.change_pct) + " 1D"}
          deltaTone={upDelta ? "pos" : "neg"}
          sparkline={sparkPrice}
          live
        />
        <KPICard
          label="Fair value"
          icon={<BarChart3 size={11} />}
          value={fmtPrice(fvNow, commodity.price_unit)}
          delta={`${fmtPct(fvDeviation)} vs spot`}
          deltaTone={fvDeviation > 10 ? "neg" : fvDeviation < -10 ? "pos" : "neutral"}
        />
        <KPICard
          label={`End stocks ${commodity.inventory_unit}`}
          icon={<Box size={11} />}
          value={fmtNum(balance.end_stocks)}
          sparkline={sparkStocks}
        />
        <KPICard
          label="Days of cover"
          icon={<Droplets size={11} />}
          value={balance.end_days_cover.toFixed(1)}
          delta={`target ${commodity.days_cover_target.toFixed(0)}`}
          deltaTone={
            balance.end_days_cover < commodity.days_cover_target * 0.85 ? "neg" :
            balance.end_days_cover > commodity.days_cover_target * 1.15 ? "pos" : "neutral"
          }
          sparkline={sparkDC}
        />
        <KPICard
          label="Storage util %"
          icon={<Box size={11} />}
          value={`${balance.end_utilization_pct.toFixed(1)}%`}
          delta={`ideal ${commodity.ideal_utilization_pct.toFixed(0)}%`}
          deltaTone={
            balance.end_utilization_pct > commodity.ideal_utilization_pct + 12 ? "neg" :
            balance.end_utilization_pct < commodity.ideal_utilization_pct - 12 ? "neg" : "pos"
          }
          sparkline={sparkUtil}
        />
      </div>

      {/* Hero row: chart left, sentiment + top movers right */}
      <div className="grid lg:grid-cols-5 gap-4">
        <div className="card p-5 lg:col-span-3">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold">Supply, demand & stocks</h2>
            <span className="text-[11px] text-ink-200 italic">
              Dashed line marks the start of the forecast
            </span>
          </div>
          <BalanceChart
            points={balance.points}
            unit={commodity.unit}
            inventoryUnit={commodity.inventory_unit}
          />
        </div>

        <div className="lg:col-span-2 space-y-4">
          <SentimentGauge score={sentiment}
            label={`Market breadth — avg ${fmtPct(avgChange)}`} />

          <div className="card p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <TrendingUp size={14} className="text-pos" /> Top gainers
            </h3>
            <ul className="space-y-1.5">
              {gainers.length > 0 ? gainers.map((g) => (
                <li key={g.key} className="flex justify-between items-center text-xs">
                  <span className="text-ink-100 truncate">{g.name}</span>
                  <span className="text-pos font-mono font-semibold">
                    +{g.change_pct.toFixed(2)}%
                  </span>
                </li>
              )) : <li className="text-xs text-ink-300">No gainer in the session.</li>}
            </ul>
            <div className="border-t border-ink-600 my-3" />
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <TrendingDown size={14} className="text-neg" /> Top losers
            </h3>
            <ul className="space-y-1.5">
              {losers.length > 0 ? losers.map((g) => (
                <li key={g.key} className="flex justify-between items-center text-xs">
                  <span className="text-ink-100 truncate">{g.name}</span>
                  <span className="text-neg font-mono font-semibold">
                    {g.change_pct.toFixed(2)}%
                  </span>
                </li>
              )) : <li className="text-xs text-ink-300">No loser in the session.</li>}
            </ul>
          </div>
        </div>
      </div>

      {/* Market Heatmap */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Globe size={15} className="text-accent" />
            <h2 className="text-sm font-semibold">Market heatmap</h2>
          </div>
          <span className="text-[11px] text-ink-200 italic">
            Click any tile to switch commodity · colour intensity ∝ 1d %
          </span>
        </div>
        <MarketHeatmap spots={allSpots} />
      </div>

      {/* Regional snapshot */}
      <div className="grid lg:grid-cols-3 gap-4">
        <div className="card p-5 lg:col-span-2">
          <h2 className="text-sm font-semibold mb-3">Regional snapshot — {commodity.name}</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-ink-200 border-b border-ink-600">
                  <th className="text-left pb-2">Region</th>
                  <th className="text-right pb-2">Supply</th>
                  <th className="text-right pb-2">Demand</th>
                  <th className="text-right pb-2">Net trade</th>
                  <th className="text-right pb-2">S %</th>
                  <th className="text-right pb-2">D %</th>
                  <th className="text-left pb-2 pl-3">Status</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {regional.rows.map((r) => (
                  <tr key={r.region} className="border-b border-ink-700/60 hover:bg-ink-600/20 transition">
                    <td className="py-2 text-ink-50">{r.region}</td>
                    <td className="text-right">{fmtNum(r.supply, 2)}</td>
                    <td className="text-right">{fmtNum(r.demand, 2)}</td>
                    <td className={`text-right font-semibold ${
                      r.net_trade > 0 ? "text-pos" : r.net_trade < 0 ? "text-neg" : ""
                    }`}>
                      {r.net_trade > 0 ? "+" : ""}{fmtNum(r.net_trade, 2)}
                    </td>
                    <td className="text-right text-ink-200">{r.supply_share_pct.toFixed(1)}</td>
                    <td className="text-right text-ink-200">{r.demand_share_pct.toFixed(1)}</td>
                    <td className="pl-3">
                      <span className={`badge ${
                        r.status === "exporter" ? "border-pos/40 text-pos" :
                        r.status === "importer" ? "border-neg/40 text-neg" : ""
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

        <div className="card p-5">
          <h2 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <Globe size={14} className="text-accent" /> Global balance
          </h2>
          <div className="space-y-3 font-mono">
            <Stat label="World supply" value={`${fmtNum(regional.world_supply, 1)}`} unit={regional.unit} />
            <Stat label="World demand" value={`${fmtNum(regional.world_demand, 1)}`} unit={regional.unit} />
            <div className="border-t border-ink-600 pt-3">
              <Stat
                label="Balance"
                value={`${regional.world_balance > 0 ? "+" : ""}${fmtNum(regional.world_balance, 1)}`}
                unit={regional.unit}
                tone={regional.world_balance >= 0 ? "pos" : "neg"}
                big
              />
            </div>
          </div>

          <h3 className="section-title text-pos">Top exporter</h3>
          {regional.rows.filter((r) => r.net_trade > 0).sort((a, b) => b.net_trade - a.net_trade)[0] && (
            <div className="rounded-lg bg-pos/10 border border-pos/20 px-3 py-2 mt-1">
              <div className="font-semibold text-ink-50 text-sm">
                {regional.rows.filter((r) => r.net_trade > 0).sort((a, b) => b.net_trade - a.net_trade)[0].region}
              </div>
              <div className="text-pos font-mono text-xs mt-0.5">
                +{fmtNum(regional.rows.filter((r) => r.net_trade > 0).sort((a, b) => b.net_trade - a.net_trade)[0].net_trade, 1)} {regional.unit}
              </div>
            </div>
          )}

          <h3 className="section-title text-neg">Top importer</h3>
          {regional.rows.filter((r) => r.net_trade < 0).sort((a, b) => a.net_trade - b.net_trade)[0] && (
            <div className="rounded-lg bg-neg/10 border border-neg/20 px-3 py-2 mt-1">
              <div className="font-semibold text-ink-50 text-sm">
                {regional.rows.filter((r) => r.net_trade < 0).sort((a, b) => a.net_trade - b.net_trade)[0].region}
              </div>
              <div className="text-neg font-mono text-xs mt-0.5">
                {fmtNum(regional.rows.filter((r) => r.net_trade < 0).sort((a, b) => a.net_trade - b.net_trade)[0].net_trade, 1)} {regional.unit}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, unit, tone, big }: {
  label: string; value: string; unit: string;
  tone?: "pos" | "neg"; big?: boolean;
}) {
  return (
    <div className="flex justify-between items-baseline">
      <span className="metric-label">{label}</span>
      <span className={`${big ? "text-lg" : "text-sm"} ${
        tone === "pos" ? "text-pos" : tone === "neg" ? "text-neg" : "text-ink-50"
      }`}>
        {value} <span className="text-[10px] text-ink-300">{unit}</span>
      </span>
    </div>
  );
}
