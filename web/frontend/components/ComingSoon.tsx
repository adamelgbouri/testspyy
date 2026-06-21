import Link from "next/link";
import { ArrowRight, Sparkles } from "lucide-react";

type Props = {
  title: string;
  description: string;
  features: string[];
  hint?: string;
};

/** Polished placeholder shown on pages that aren't implemented yet. */
export function ComingSoon({ title, description, features, hint }: Props) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{title}</h1>
        <p className="text-sm text-ink-200 mt-1">{description}</p>
      </div>

      <div className="card p-6">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles size={16} className="text-accent" />
          <h2 className="text-sm font-semibold">Coming soon</h2>
        </div>
        <p className="text-sm text-ink-200 mb-5">
          This page is part of the platform roadmap. The Python analytics already
          exist in the Streamlit prototype and only need to be ported to the
          web stack.
        </p>
        <h3 className="section-title">Planned features</h3>
        <ul className="space-y-2 text-sm">
          {features.map((f) => (
            <li key={f} className="flex items-start gap-2">
              <span className="text-accent mt-0.5">›</span>
              <span className="text-ink-100">{f}</span>
            </li>
          ))}
        </ul>
        {hint && (
          <p className="text-xs text-ink-300 italic mt-5 pt-4 border-t border-ink-600">
            {hint}
          </p>
        )}
      </div>

      <div className="flex gap-3">
        <Link
          href="/dashboard"
          className="text-sm bg-accent text-ink-900 font-medium rounded-md px-4 py-2 hover:bg-accent/90 transition inline-flex items-center gap-2"
        >
          Back to Dashboard <ArrowRight size={14} />
        </Link>
        <Link
          href="/options"
          className="text-sm border border-ink-500 text-ink-100 rounded-md px-4 py-2 hover:bg-ink-600 transition"
        >
          Try Options pricer
        </Link>
      </div>
    </div>
  );
}
