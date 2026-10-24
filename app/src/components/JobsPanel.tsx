import { useState } from "react";
import { request, type Connection } from "../api";
import { jobTitle, percent } from "../format";
import { isActive, type Job, type JobsState } from "../jobs";

interface Props {
  conn: Connection;
  jobs: JobsState;
  videoName: (id: number) => string;
  onClose: () => void;
}

function describeWorker(message: string | null): string | null {
  if (!message) return null;
  const load = message.match(/^load (.+)$/);
  if (load) return `Loading ${load[1].replace(/^ollama:/, "")}…`;
  if (message.startsWith("unload ")) return null;
  return null;
}

function JobRow({ job, name, onCancel }: { job: Job; name: string; onCancel?: () => void }) {
  return (
    <li className="flex flex-col gap-1 border-b border-zinc-200 py-2 dark:border-zinc-800">
      <div className="flex items-center justify-between gap-2 text-sm">
        <span className="min-w-0 truncate">
          {jobTitle(job.name)} <span className="text-zinc-500">· {name}</span>
        </span>
        {onCancel ? (
          <button type="button" onClick={onCancel} className="text-xs text-zinc-500 hover:text-red-600">
            Cancel
          </button>
        ) : (
          <span
            className={`text-xs ${
              job.status === "failed" ? "text-red-600" : job.status === "done" ? "text-emerald-600" : "text-zinc-500"
            }`}
          >
            {job.status}
          </span>
        )}
      </div>
      {job.status === "running" && (
        <div className="flex items-center gap-2" aria-label="Progress">
          <span className="h-1.5 flex-1 overflow-hidden rounded bg-zinc-200 dark:bg-zinc-700">
            <span className="block h-full bg-sky-500" style={{ width: percent(job.progress) }} />
          </span>
          <span className="text-xs tabular-nums text-zinc-500">{percent(job.progress)}</span>
        </div>
      )}
      {job.status === "queued" && <span className="text-xs text-zinc-500">waiting</span>}
      {job.status === "failed" && job.error && <p className="text-xs text-red-600">{job.error}</p>}
    </li>
  );
}

/** Live list of queued, running and recently finished jobs. */
export function JobsPanel({ conn, jobs, videoName, onClose }: Props) {
  const [error, setError] = useState<string | null>(null);
  const active = jobs.jobs.filter(isActive).sort((a, b) => a.id - b.id);
  const finished = jobs.jobs.filter((j) => !isActive(j)).slice(0, 30);
  const status = describeWorker(jobs.message);
  const name = (j: Job) => (j.video_id === null ? "library" : videoName(j.video_id));

  function cancel(ids: number[] | null) {
    request(conn, "/jobs/cancel", { method: "POST", body: JSON.stringify({ job_ids: ids }) }).catch(
      (e: unknown) => setError(String(e instanceof Error ? e.message : e)),
    );
  }

  return (
    <aside aria-label="Jobs" className="flex h-full flex-col gap-3 overflow-y-auto p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Jobs</h2>
        <button type="button" onClick={onClose} aria-label="Close jobs" className="px-2 text-lg">
          ×
        </button>
      </div>
      {status && <p className="text-sm text-sky-700 dark:text-sky-300">{status}</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      <section>
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
            In progress ({active.length})
          </h3>
          {active.length > 1 && (
            <button type="button" onClick={() => cancel(null)} className="text-xs text-zinc-500 hover:text-red-600">
              Cancel all
            </button>
          )}
        </div>
        {active.length === 0 ? (
          <p className="py-2 text-sm text-zinc-500">Nothing is running.</p>
        ) : (
          <ul>
            {active.map((j) => (
              <JobRow key={j.id} job={j} name={name(j)} onCancel={() => cancel([j.id])} />
            ))}
          </ul>
        )}
      </section>

      {finished.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Finished</h3>
          <ul>
            {finished.map((j) => (
              <JobRow key={j.id} job={j} name={name(j)} />
            ))}
          </ul>
        </section>
      )}
    </aside>
  );
}
