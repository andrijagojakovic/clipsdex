import { describe, expect, it } from "vitest";
import { EMPTY, reduceJobs, type Job } from "./jobs";

const job = (id: number, status: Job["status"], progress = 0): Job => ({
  id,
  name: "transcript",
  kind: "artifact",
  video_id: 1,
  status,
  progress,
  attempts: 1,
  error: null,
  resource: "whisper:small",
  started_at: null,
  finished_at: null,
});

describe("reduceJobs", () => {
  it("follows a job from queued to done and counts it as finished once", () => {
    let s = reduceJobs(EMPTY, { event: "snapshot", data: { counts: { queued: 1 }, jobs: [job(1, "queued")] } });
    s = reduceJobs(s, { event: "job", data: job(1, "running", 0.5) });
    expect(s.jobs).toHaveLength(1);
    expect(s.jobs[0].progress).toBe(0.5);
    expect(s.finished).toBe(0);
    s = reduceJobs(s, { event: "job", data: job(1, "done", 1) });
    expect(s.finished).toBe(1);
    s = reduceJobs(s, { event: "job", data: job(1, "done", 1) }); // the same news again
    expect(s.finished).toBe(1);
    s = reduceJobs(s, { event: "counts", data: { done: 1 } });
    expect(s.counts).toEqual({ done: 1 });
  });

  it("keeps the newest first and remembers the worker's last message", () => {
    let s = reduceJobs(EMPTY, { event: "job", data: job(1, "queued") });
    s = reduceJobs(s, { event: "job", data: job(3, "queued") });
    s = reduceJobs(s, { event: "job", data: job(2, "queued") });
    expect(s.jobs.map((j) => j.id)).toEqual([3, 2, 1]);
    s = reduceJobs(s, { event: "worker", data: { message: "load ollama:gemma3:4b" } });
    expect(s.message).toBe("load ollama:gemma3:4b");
  });

  it("drops jobs a reconnect no longer reports as active", () => {
    let s = reduceJobs(EMPTY, { event: "job", data: job(1, "running") });
    s = reduceJobs(s, { event: "job", data: job(2, "failed") });
    s = reduceJobs(s, { event: "snapshot", data: { counts: {}, jobs: [] } });
    expect(s.jobs.map((j) => j.id)).toEqual([2]); // finished jobs stay in the list
  });
});
