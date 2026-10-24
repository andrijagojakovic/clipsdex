import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { Plan } from "../api";
import { AnalyzeMenu } from "./AnalyzeMenu";
import { JobsPanel } from "./JobsPanel";
import { PlanDialog } from "./PlanDialog";
import { EMPTY } from "../jobs";

const conn = { url: "http://127.0.0.1:5000", token: "t" };

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const plan = (over: Partial<Plan> = {}): Plan => ({
  analyzers: ["brief"],
  artifacts: [],
  videos: [
    { video_id: 1, name: "pricing.mp4", analyzers: ["brief"], skipped: {}, done: [], missing: ["transcript"], blocked: null },
    { video_id: 2, name: "song.mp4", analyzers: [], skipped: { brief: "no speech and no on-screen text" }, done: [], missing: [], blocked: null },
    { video_id: 3, name: "old.mp4", analyzers: [], skipped: {}, done: ["brief"], missing: [], blocked: null },
  ],
  jobs: [
    { name: "transcript", videos: 1, seconds: 30, measured_jobs: 0, resource: "whisper:small" },
    { name: "brief", videos: 1, seconds: 20, measured_jobs: 4, resource: "ollama:gemma3:4b" },
  ],
  load_seconds: 20,
  total_seconds: 70,
  not_available: [],
  ...over,
});

function engine(...answers: unknown[]) {
  const fetcher = vi.fn();
  for (const a of answers) fetcher.mockResolvedValueOnce(new Response(JSON.stringify(a), { status: 200 }));
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

describe("PlanDialog", () => {
  it("shows the estimate and what is skipped, then starts", async () => {
    const fetcher = engine(plan(), { plan: plan(), queued: [11, 12] });
    const started = vi.fn();
    render(
      <PlanDialog conn={conn} ask={{ analyzers: ["brief"], category: "business" }} scope='category "business"' titles={{ brief: "Make brief" }} onClose={() => undefined} onStarted={started} />,
    );
    expect(await screen.findByText("1 video will be processed.")).toBeTruthy();
    expect(screen.getByRole("dialog", { name: "Make brief: plan" })).toBeTruthy();
    expect(screen.getByText("Transcribing speech")).toBeTruthy();
    expect(screen.getByText("~1 min")).toBeTruthy(); // total 70 s
    expect(screen.getByText("1 skipped (doesn't apply)")).toBeTruthy();
    expect(screen.getByText(/song.mp4: no speech/)).toBeTruthy();

    fireEvent.click(screen.getByText("Start"));
    await waitFor(() => expect(started).toHaveBeenCalledWith([11, 12]));
    const [url, init] = fetcher.mock.calls[1] as [string, RequestInit];
    expect(url).toBe("http://127.0.0.1:5000/run");
    expect(JSON.parse(String(init.body))).toEqual({ analyzers: ["brief"], category: "business", redo: false });
  });

  it("asks again with redo when the user wants to run done videos again", async () => {
    const fetcher = engine(plan(), plan());
    render(<PlanDialog conn={conn} ask={{ analyzers: ["brief"] }} scope="x" onClose={() => undefined} onStarted={() => undefined} />);
    fireEvent.click(await screen.findByLabelText(/Run again on 1 already done/));
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
    const [, init] = fetcher.mock.calls[1] as [string, RequestInit];
    expect(JSON.parse(String(init.body)).redo).toBe(true);
  });

  it("can't start when there is nothing to do", async () => {
    engine(plan({ jobs: [], total_seconds: 0, load_seconds: 0, videos: [] }));
    render(<PlanDialog conn={conn} ask={{ analyzers: ["brief"] }} scope="x" onClose={() => undefined} onStarted={() => undefined} />);
    expect(await screen.findByText(/Nothing to do/)).toBeTruthy();
    expect((screen.getByText("Start") as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("JobsPanel and AnalyzeMenu", () => {
  it("lists running jobs with progress and cancels one", async () => {
    const fetcher = engine({ cancelled: 1 });
    const running = { id: 5, name: "frame_descriptions", kind: "artifact", video_id: 1, status: "running" as const, progress: 0.25, attempts: 1, error: null, resource: "ollama:gemma3:4b", started_at: null, finished_at: null };
    const failed = { ...running, id: 4, status: "failed" as const, error: "The model is not downloaded yet." };
    render(
      <JobsPanel conn={conn} jobs={{ ...EMPTY, jobs: [running, failed], message: "load ollama:gemma3:4b" }} videoName={() => "pricing.mp4"} onClose={() => undefined} />,
    );
    expect(screen.getByText("Loading gemma3:4b…")).toBeTruthy();
    expect(screen.getByText("25%")).toBeTruthy();
    expect(screen.getByText("The model is not downloaded yet.")).toBeTruthy();
    fireEvent.click(screen.getByText("Cancel"));
    await waitFor(() => expect(fetcher).toHaveBeenCalled());
    const [url, init] = fetcher.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://127.0.0.1:5000/jobs/cancel");
    expect(JSON.parse(String(init.body))).toEqual({ job_ids: [5] });
  });

  it("offers only what this version can run", () => {
    const pick = vi.fn();
    render(
      <AnalyzeMenu
        analyzers={[
          { name: "brief", title: "Make brief", available: true, default_categories: [], requires: [] },
          { name: "song_id", title: "Find songs", available: false, default_categories: [], requires: [] },
        ]}
        onPick={pick}
      />,
    );
    fireEvent.click(screen.getByText("Analyze ▾"));
    expect((screen.getByText("Find songs").closest("button") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByText("Make brief"));
    expect(pick).toHaveBeenCalledWith("brief");
  });
});
