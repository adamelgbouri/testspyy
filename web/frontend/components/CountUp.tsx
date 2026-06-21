"use client";
import { useEffect, useRef, useState } from "react";

type Props = {
  value: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
  flash?: boolean;
};

/**
 * Smoothly counts from the previous value to the new one (200-700ms),
 * and flashes green / red on direction change.
 */
export function CountUp({
  value, duration = 500, decimals = 2,
  prefix = "", suffix = "", className = "", flash = true,
}: Props) {
  const [displayed, setDisplayed] = useState(value);
  const [flashClass, setFlashClass] = useState("");
  const prevValue = useRef(value);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const start = prevValue.current;
    const end = value;
    if (start === end) return;
    const startTime = performance.now();
    const ease = (t: number) => 1 - Math.pow(1 - t, 3);

    const step = (now: number) => {
      const t = Math.min(1, (now - startTime) / duration);
      const v = start + (end - start) * ease(t);
      setDisplayed(v);
      if (t < 1) rafRef.current = requestAnimationFrame(step);
      else setDisplayed(end);
    };
    rafRef.current = requestAnimationFrame(step);

    if (flash) {
      setFlashClass(end >= start ? "flash-pos" : "flash-neg");
      const id = window.setTimeout(() => setFlashClass(""), 700);
      return () => {
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        window.clearTimeout(id);
      };
    }
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [value, duration, flash]);

  useEffect(() => { prevValue.current = value; }, [value]);

  const formatted = displayed.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

  return (
    <span className={`${className} ${flashClass} rounded px-0.5 transition-colors tabular-nums`}>
      {prefix}{formatted}{suffix}
    </span>
  );
}
