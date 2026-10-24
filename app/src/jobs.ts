// Live job state from the engine's event stream: one connection for the whole app.

import { useEffect, useReducer, useRef, useState } from "react";
import type { Connection } from "./api";
import { followEvents, type StreamEvent } from "./sse";

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Job {
  id: number;
  name: string;
  kind: string;
  video_id: number | null;
  status: JobStatus;
  progress: number;
  attempts: number;
  error: string | null;
  resource: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface JobsState {
  jobs: Job[]; // newest first, at most KEEP
  counts: Partial<Record<JobStatus, number>>;
  message: string | null; // the worker's latest message, e.g. "load ollama:gemma3:4b"
  finished: number; // grows whenever a job leaves queued/running
}

export const KEEP = 200;
export const EMPTY: JobsState = { jobs: [], counts: {}, message: null, finished: 0 };

export const isActive = (j: Pick<Job, "status">) => j.status === "queued" || j.status === "running";

function upsert(jobs: Job[], job: Job): Job[] {
  const rest = jobs.filter((j) => j.id !== job.id);
  return [job, ...rest].sort((a, b) => b.id - a.id).slice(0, KEEP);
}

/** Applies one stream event; pure, so it can be tested without a server. */
export function reduceJobs(state: JobsState, e: StreamEvent): JobsState {
  const data = e.data as Record<string, unknown>;
  switch (e.event) {
    case "snapshot": {
      const active = (data.jobs as Job[]) ?? [];
      // Jobs that were active before a reconnect but are not any more have finished.
      const kept = state.jobs.filter((j) => !isActive(j) || active.some((a) => a.id === j.id));
      const jobs = active.reduce(upsert, kept);
      return { ...state, jobs, counts: data.counts as JobsState["counts"] };
    }
    case "job": {
      const job = data as unknown as Job;
      const before = state.jobs.find((j) => j.id === job.id);
      const ended = !isActive(job) && (before === undefined || isActive(before));
      return { ...state, jobs: upsert(state.jobs, job), finished: state.finished + (ended ? 1 : 0) };
    }
    case "counts":
      return { ...state, counts: data as JobsState["counts"] };
    case "worker":
      return { ...state, message: typeof data.message === "string" ? data.message : state.message };
    default:
      return state;
  }
}

/** Job state from /events, and a `finished` counter that changes at most once a second
 * (the library reloads when it changes). */
export function useJobs(conn: Connection | null): JobsState {
  const [state, dispatch] = useReducer(reduceJobs, EMPTY);
  useEffect(() => {
    if (conn === null) return;
    const stop = new AbortController();
    followEvents(conn, dispatch, stop.signal);
    return () => stop.abort();
  }, [conn?.url, conn?.token]);

  const [finished, setFinished] = useState(0);
  const timer = useRef<number | null>(null);
  const latest = useRef(0);
  latest.current = state.finished;
  useEffect(() => {
    if (state.finished === 0 || timer.current !== null) return;
    timer.current = window.setTimeout(() => {
      timer.current = null;
      setFinished(latest.current);
    }, 1000);
  }, [state.finished]);
  useEffect(() => () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
  }, []);

  return { ...state, finished };
}
