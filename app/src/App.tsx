import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  ApiError,
  request,
  type CategoryItem,
  type Connection,
  type EngineStatus,
  type Health,
  type VideoItem,
} from "./api";
import "./App.css";

interface Overview {
  health: Health;
  videos: VideoItem[];
  categories: CategoryItem[];
}

function useEngine(): EngineStatus {
  const [status, setStatus] = useState<EngineStatus>({ state: "starting" });
  useEffect(() => {
    const unlisten = listen<EngineStatus>("engine-status", (e) => setStatus(e.payload));
    invoke<EngineStatus>("engine_status").then(setStatus);
    return () => {
      unlisten.then((stop) => stop());
    };
  }, []);
  return status;
}

function useOverview(conn: Connection | null) {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (conn === null) return;
    setError(null);
    Promise.all([
      request<Health>(conn, "/health"),
      request<{ videos: VideoItem[] }>(conn, "/videos"),
      request<{ categories: CategoryItem[] }>(conn, "/categories"),
    ])
      .then(([health, v, c]) =>
        setOverview({ health, videos: v.videos, categories: c.categories }),
      )
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : String(e)));
  }, [conn?.url, conn?.token]);
  return { overview, error };
}

export default function App() {
  const status = useEngine();
  const conn = status.state === "ready" ? { url: status.url, token: status.token } : null;
  const { overview, error } = useOverview(conn);

  return (
    <main className="app">
      <header>
        <h1>ClipsDex</h1>
        <EngineBadge status={status} worker={overview?.health.worker ?? null} />
      </header>

      {status.state === "starting" && <p className="muted">Starting the engine…</p>}

      {status.state === "failed" && (
        <section className="problem">
          <h2>The engine is not running</h2>
          <pre>{status.message}</pre>
          <button onClick={() => invoke("restart_engine")}>Try again</button>
        </section>
      )}

      {error && <p className="problem">{error}</p>}

      {overview && (
        <section>
          <h2>
            {overview.videos.length} video{overview.videos.length === 1 ? "" : "s"} in your
            library
          </h2>
          <ul className="categories">
            {overview.categories
              .filter((c) => c.videos > 0)
              .map((c) => (
                <li key={c.name} title={c.description ?? undefined}>
                  <span>{c.name}</span>
                  <span className="muted">{c.videos}</span>
                </li>
              ))}
          </ul>
        </section>
      )}
    </main>
  );
}

function EngineBadge({ status, worker }: { status: EngineStatus; worker: string | null }) {
  const text =
    status.state === "ready"
      ? `Engine ready${worker ? ` · worker ${worker}` : ""}`
      : status.state === "starting"
        ? "Engine starting…"
        : "Engine stopped";
  return <span className={`badge ${status.state}`}>{text}</span>;
}
