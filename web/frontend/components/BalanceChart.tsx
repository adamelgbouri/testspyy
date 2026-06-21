"use client";
import {
  Area, AreaChart, CartesianGrid, ComposedChart, Legend, Line,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { BalancePoint } from "@/lib/api";

type Props = { points: BalancePoint[]; unit: string; inventoryUnit: string };

export function BalanceChart({ points, unit, inventoryUnit }: Props) {
  const forecastStart = points.find((p) => p.is_forecast)?.date;
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ComposedChart data={points} margin={{ top: 12, right: 24, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="date"
          stroke="#6b7280"
          tick={{ fontSize: 11 }}
          tickFormatter={(d: string) => d.slice(0, 7)}
          minTickGap={40}
        />
        <YAxis yAxisId="flow" stroke="#6b7280" tick={{ fontSize: 11 }} />
        <YAxis yAxisId="stk" orientation="right" stroke="#00d4ff" tick={{ fontSize: 11 }} />
        <Tooltip
          contentStyle={{
            background: "#111827", border: "1px solid #1f2937",
            borderRadius: 8, fontSize: 12,
          }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <defs>
          <linearGradient id="stocks-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#00d4ff" stopOpacity={0.4} />
            <stop offset="100%" stopColor="#00d4ff" stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          yAxisId="stk"
          dataKey="stocks"
          name={`Stocks (${inventoryUnit})`}
          stroke="#00d4ff"
          fill="url(#stocks-fill)"
          strokeWidth={1.5}
        />
        <Line
          yAxisId="flow"
          dataKey="supply"
          name={`Supply (${unit})`}
          stroke="#22c55e"
          dot={false}
          strokeWidth={2}
        />
        <Line
          yAxisId="flow"
          dataKey="demand"
          name={`Demand (${unit})`}
          stroke="#ef4444"
          dot={false}
          strokeWidth={2}
        />
        {forecastStart && (
          <ReferenceLine
            x={forecastStart}
            yAxisId="flow"
            stroke="#6b7280"
            strokeDasharray="4 4"
            label={{ value: "Forecast", fill: "#9ca3af", fontSize: 10, position: "top" }}
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
