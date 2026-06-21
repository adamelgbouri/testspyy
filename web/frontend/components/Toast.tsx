"use client";
import {
  createContext, useCallback, useContext, useEffect,
  useRef, useState, type ReactNode,
} from "react";
import { CheckCircle2, XCircle, Info, AlertTriangle, X } from "lucide-react";

type Tone = "success" | "error" | "info" | "warning";
type Toast = { id: string; tone: Tone; title: string; message?: string };

type Ctx = { push: (t: Omit<Toast, "id">) => void };
const ToastCtx = createContext<Ctx>({ push: () => {} });

/** Hook used anywhere in the app to push a toast. */
export const useToast = () => useContext(ToastCtx);

const TONE_STYLES: Record<Tone, { icon: ReactNode; border: string; bg: string; iconColor: string }> = {
  success: {
    icon: <CheckCircle2 size={18} />,
    border: "border-pos/40", bg: "from-pos/10 to-transparent",
    iconColor: "text-pos",
  },
  error: {
    icon: <XCircle size={18} />,
    border: "border-neg/40", bg: "from-neg/10 to-transparent",
    iconColor: "text-neg",
  },
  info: {
    icon: <Info size={18} />,
    border: "border-cyan/40", bg: "from-cyan/10 to-transparent",
    iconColor: "text-cyan",
  },
  warning: {
    icon: <AlertTriangle size={18} />,
    border: "border-accent/40", bg: "from-accent/10 to-transparent",
    iconColor: "text-accent",
  },
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const counter = useRef(0);

  const push = useCallback((t: Omit<Toast, "id">) => {
    const id = `t-${++counter.current}`;
    setToasts((cur) => [...cur, { ...t, id }]);
    setTimeout(() => {
      setToasts((cur) => cur.filter((x) => x.id !== id));
    }, 4500);
  }, []);

  const dismiss = (id: string) =>
    setToasts((cur) => cur.filter((x) => x.id !== id));

  return (
    <ToastCtx.Provider value={{ push }}>
      {children}
      <div className="fixed top-24 right-4 z-50 flex flex-col gap-2 pointer-events-none">
        {toasts.map((t) => {
          const style = TONE_STYLES[t.tone];
          return (
            <div
              key={t.id}
              className={`pointer-events-auto card border ${style.border} bg-gradient-to-r ${style.bg} px-4 py-3 min-w-[280px] max-w-[420px] animate-toast-in`}
            >
              <div className="flex items-start gap-3">
                <div className={`mt-0.5 ${style.iconColor}`}>{style.icon}</div>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-sm text-ink-50">{t.title}</div>
                  {t.message && (
                    <div className="text-xs text-ink-200 mt-0.5">{t.message}</div>
                  )}
                </div>
                <button
                  onClick={() => dismiss(t.id)}
                  className="text-ink-300 hover:text-ink-50 transition shrink-0"
                  aria-label="Dismiss"
                >
                  <X size={14} />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </ToastCtx.Provider>
  );
}
