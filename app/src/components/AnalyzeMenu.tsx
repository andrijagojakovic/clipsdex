import { useEffect, useRef, useState } from "react";
import type { AnalyzerInfo } from "../api";

interface Props {
  analyzers: AnalyzerInfo[];
  onPick: (name: string) => void;
  label?: string;
  align?: "left" | "right"; // which edge of the button the menu lines up with
}

/** "Analyze" button with the analyzers this version can run; the others are shown, but
 * disabled, so users know they are coming. */
export function AnalyzeMenu({ analyzers, onPick, label = "Analyze", align = "right" }: Props) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !box.current?.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", esc);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", esc);
    };
  }, [open]);

  return (
    <div ref={box} className="relative">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="rounded-md bg-sky-600 px-3 py-1 text-sm text-white"
      >
        {label} ▾
      </button>
      {open && (
        <ul
          role="menu"
          className={`absolute ${align === "right" ? "right-0" : "left-0"} z-10 mt-1 w-56 overflow-hidden rounded-md border border-zinc-200 bg-white text-sm shadow-lg dark:border-zinc-700 dark:bg-zinc-800`}
        >
          {analyzers.map((a) => (
            <li key={a.name} role="none">
              <button
                type="button"
                role="menuitem"
                disabled={!a.available}
                onClick={() => {
                  setOpen(false);
                  onPick(a.name);
                }}
                className="flex w-full justify-between px-3 py-2 text-left hover:bg-zinc-100 disabled:cursor-default disabled:opacity-50 disabled:hover:bg-transparent dark:hover:bg-zinc-700"
              >
                <span>{a.title}</span>
                {!a.available && <span className="text-xs">coming later</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
