"use client";
import { useState, type ReactNode } from "react";

type Props = {
  content: ReactNode;
  children: ReactNode;
  position?: "top" | "bottom" | "left" | "right";
  className?: string;
};

/** Lightweight tooltip - no Radix needed. Renders on hover/focus. */
export function Tooltip({ content, children, position = "top", className = "" }: Props) {
  const [visible, setVisible] = useState(false);

  const positions: Record<string, string> = {
    top:    "bottom-full left-1/2 -translate-x-1/2 mb-2",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
    left:   "right-full top-1/2 -translate-y-1/2 mr-2",
    right:  "left-full top-1/2 -translate-y-1/2 ml-2",
  };

  return (
    <span
      className={`relative inline-flex ${className}`}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      {children}
      {visible && (
        <span
          className={`absolute ${positions[position]} z-50 pointer-events-none whitespace-nowrap`}
        >
          <span className="block bg-ink-900 border border-ink-500 rounded-md px-2.5 py-1.5 text-[11px] text-ink-50 font-mono shadow-lg animate-slide-up">
            {content}
          </span>
        </span>
      )}
    </span>
  );
}
