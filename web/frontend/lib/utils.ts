import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Conditional class-name merger (Tailwind-aware). */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format a price with its unit. */
export function fmtPrice(value: number, unit: string, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  return `${value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })} ${unit}`;
}

/** Format a number with thousands separators. */
export function fmtNum(value: number, digits = 0): string {
  if (!Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** Format a percentage with sign. */
export function fmtPct(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)} %`;
}
