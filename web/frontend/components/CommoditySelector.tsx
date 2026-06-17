"use client";
import { useRouter, useSearchParams } from "next/navigation";
import type { Commodity } from "@/lib/api";

type Props = { commodities: Commodity[]; current: string };

export function CommoditySelector({ commodities, current }: Props) {
  const router = useRouter();
  const sp = useSearchParams();
  return (
    <select
      value={current}
      onChange={(e) => {
        const url = new URLSearchParams(sp.toString());
        url.set("c", e.target.value);
        router.push(`?${url.toString()}` as any);
      }}
      className="bg-ink-700 border border-ink-500 rounded-md px-3 py-1.5 text-sm
                 text-ink-100 font-mono focus:outline-none focus:ring-2 focus:ring-accent"
    >
      {commodities.map((c) => (
        <option key={c.key} value={c.key}>
          [{c.sector}] {c.name}
        </option>
      ))}
    </select>
  );
}
