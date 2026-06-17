type Props = { tone?: "pos" | "neg" | "accent"; size?: number };

/** Animated live-data indicator. */
export function PulseDot({ tone = "accent", size = 8 }: Props) {
  return (
    <span
      className={`pulse-dot pulse-${tone}`}
      style={{ width: size, height: size }}
    />
  );
}
