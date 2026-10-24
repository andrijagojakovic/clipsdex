import { useEffect, useState } from "react";
import { ApiError, request, type Connection, type Plan, type PlanRequest } from "../api";
import { analyzerTitle, humanize, jobTitle } from "../format";

interface Props {
  conn: Connection;
  ask: PlanRequest; // what the user asked for
  scope: string; // in words, e.g. 'category "business"' or "3 selected videos"
  titles?: Record<string, string>; // analyzer name -> what the user asked for ("Make brief")
  onClose: () => void;
  onStarted: (jobIds: number[]) => void;
}

/** Shows what would run and how long it would take, before anything starts. */
export function PlanDialog({ conn, ask, scope, titles = {}, onClose, onStarted }: Props) {
  const [redo, setRedo] = useState(Boolean(ask.redo));
  const [plan, setPlan] = useState<Plan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const body = JSON.stringify({ ...ask, redo });

  useEffect(() => {
    let cancelled = false;
    setPlan(null);
    setError(null);
    request<Plan>(conn, "/plan", { method: "POST", body })
      .then((p) => !cancelled && setPlan(p))
      .catch((e: unknown) => !cancelled && setError(e instanceof ApiError ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [conn.url, conn.token, body]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function start() {
    setStarting(true);
    try {
      const r = await request<{ queued: number[] }>(conn, "/run", { method: "POST", body });
      onStarted(r.queued);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setStarting(false);
    }
  }

  const title = ask.analyzers.map((a) => titles[a] ?? analyzerTitle(a)).join(", ");
  const runs = plan?.videos.filter((v) => !v.blocked && (v.analyzers.length > 0 || v.missing.length > 0)) ?? [];
  const skipped = plan?.videos.filter((v) => Object.keys(v.skipped).length > 0) ?? [];
  const done = plan?.videos.filter((v) => v.done.length > 0) ?? [];
  const blocked = plan?.videos.filter((v) => v.blocked) ?? [];

  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`${title}: plan`}
        className="flex max-h-[85vh] w-full max-w-xl flex-col gap-4 overflow-y-auto rounded-xl bg-white p-5 shadow-xl dark:bg-zinc-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <p className="text-sm text-zinc-500">for {scope}</p>
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}
        {!plan && !error && <p className="text-sm text-zinc-500">Working out the plan…</p>}

        {plan && (
          <>
            {plan.not_available.length > 0 && (
              <p className="text-sm text-amber-700 dark:text-amber-400">
                {plan.not_available.map(analyzerTitle).join(", ")} is not available in this version yet;
                only the steps it needs would run.
              </p>
            )}
            <p className="text-sm">
              {runs.length === 0
                ? "Nothing to do: everything is already done or doesn't apply."
                : `${runs.length} video${runs.length === 1 ? "" : "s"} will be processed.`}
            </p>

            {plan.jobs.length > 0 && (
              <table className="w-full text-sm">
                <tbody>
                  {plan.jobs.map((j) => (
                    <tr key={j.name} className="border-b border-zinc-200 dark:border-zinc-800">
                      <td className="py-1">{jobTitle(j.name)}</td>
                      <td className="py-1 text-right text-zinc-500">{j.videos}×</td>
                      <td
                        className="py-1 text-right tabular-nums"
                        title={j.measured_jobs ? `based on ${j.measured_jobs} jobs on this computer` : "first estimate"}
                      >
                        {humanize(j.seconds)}
                      </td>
                    </tr>
                  ))}
                  {plan.load_seconds > 0 && (
                    <tr className="text-zinc-500">
                      <td className="py-1">Loading models</td>
                      <td />
                      <td className="py-1 text-right tabular-nums">{humanize(plan.load_seconds)}</td>
                    </tr>
                  )}
                  <tr className="font-semibold">
                    <td className="pt-2">Total</td>
                    <td />
                    <td className="pt-2 text-right tabular-nums">{humanize(plan.total_seconds)}</td>
                  </tr>
                </tbody>
              </table>
            )}

            {skipped.length > 0 && (
              <details className="text-sm">
                <summary className="cursor-pointer">{skipped.length} skipped (doesn't apply)</summary>
                <ul className="mt-1 ml-4 list-disc text-zinc-500">
                  {skipped.map((v) => (
                    <li key={v.video_id}>
                      {v.name}: {Object.values(v.skipped).join("; ")}
                    </li>
                  ))}
                </ul>
              </details>
            )}
            {blocked.length > 0 && (
              <details className="text-sm">
                <summary className="cursor-pointer">{blocked.length} can't be processed</summary>
                <ul className="mt-1 ml-4 list-disc text-zinc-500">
                  {blocked.map((v) => (
                    <li key={v.video_id}>
                      {v.name}: {v.blocked}
                    </li>
                  ))}
                </ul>
              </details>
            )}
            {(done.length > 0 || redo) && (
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={redo} onChange={(e) => setRedo(e.target.checked)} />
                Run again on {redo ? "videos that already have a result" : `${done.length} already done`}
              </label>
            )}
          </>
        )}

        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="rounded-md px-3 py-1.5 text-sm">
            Cancel
          </button>
          <button
            type="button"
            disabled={!plan || plan.jobs.length === 0 || starting}
            onClick={start}
            className="rounded-md bg-sky-600 px-3 py-1.5 text-sm text-white disabled:opacity-40"
          >
            {starting ? "Starting…" : "Start"}
          </button>
        </div>
      </div>
    </div>
  );
}
