import { useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { EngineStatus } from "./api";
import { matchesText } from "./format";
import { useEngine, useLibrary, useVideo } from "./hooks";
import { Sidebar } from "./components/Sidebar";
import { VideoDetail } from "./components/VideoDetail";
import { VideoGrid, type Layout } from "./components/VideoGrid";
import "./App.css";

export default function App() {
  const status = useEngine();
  const conn = useMemo(
    () => (status.state === "ready" ? { url: status.url, token: status.token } : null),
    [status.state === "ready" ? status.url : null, status.state === "ready" ? status.token : null],
  );
  const library = useLibrary(conn);
  const [category, setCategory] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [layout, setLayout] = useState<Layout>("grid");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const detail = useVideo(conn, selectedId, library.loadedAt);

  const shown = useMemo(
    () =>
      library.videos.filter(
        (v) =>
          (category === null || (category === "" ? v.category === null : v.category === category)) &&
          matchesText(v, text),
      ),
    [library.videos, category, text],
  );
  const uncategorized = library.videos.filter((v) => v.category === null).length;

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between gap-4 border-b border-zinc-200 px-4 py-2 dark:border-zinc-800">
        <h1 className="text-lg font-semibold">ClipsDex</h1>
        <EngineBadge status={status} worker={library.health?.worker ?? null} />
      </header>

      {status.state !== "ready" ? (
        <EngineMessage status={status} />
      ) : (
        <div className="grid min-h-0 flex-1 grid-cols-[220px_1fr_auto]">
          <aside className="min-h-0 border-r border-zinc-200 dark:border-zinc-800">
            <Sidebar
              categories={library.categories}
              total={library.videos.length}
              uncategorized={uncategorized}
              selected={category}
              onSelect={setCategory}
            />
          </aside>

          <main className="flex min-h-0 flex-col">
            <div className="flex items-center gap-3 border-b border-zinc-200 px-4 py-2 dark:border-zinc-800">
              <input
                type="search"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Filter by name, author or caption"
                aria-label="Filter videos"
                className="w-72 rounded-md border border-zinc-300 bg-transparent px-3 py-1 text-sm dark:border-zinc-700"
              />
              <span className="text-sm text-zinc-500">
                {shown.length} of {library.videos.length}
              </span>
              <div className="ml-auto flex overflow-hidden rounded-md border border-zinc-300 text-sm dark:border-zinc-700" role="group" aria-label="Layout">
                {(["grid", "list"] as const).map((l) => (
                  <button
                    key={l}
                    type="button"
                    aria-pressed={layout === l}
                    onClick={() => setLayout(l)}
                    className={`px-3 py-1 capitalize ${layout === l ? "bg-zinc-200 dark:bg-zinc-700" : ""}`}
                  >
                    {l}
                  </button>
                ))}
              </div>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto p-4">
              {library.error && <p className="mb-3 text-sm text-red-600">{library.error}</p>}
              {library.loading ? (
                <p className="text-zinc-500">Loading your library…</p>
              ) : library.videos.length === 0 ? (
                <p className="p-8 text-center text-zinc-500">
                  Your library is empty. Importing videos comes with the next version of this screen.
                </p>
              ) : (
                <VideoGrid conn={conn} videos={shown} layout={layout} selectedId={selectedId} onOpen={setSelectedId} />
              )}
            </div>
          </main>

          {selectedId !== null && (
            <div className="min-h-0 w-96 border-l border-zinc-200 dark:border-zinc-800">
              {detail.error ? (
                <p className="p-4 text-sm text-red-600">{detail.error}</p>
              ) : detail.video ? (
                <VideoDetail video={detail.video} onClose={() => setSelectedId(null)} />
              ) : (
                <p className="p-4 text-sm text-zinc-500">Loading…</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function EngineMessage({ status }: { status: EngineStatus }) {
  if (status.state === "starting") {
    return <p className="p-6 text-zinc-500">Starting the engine…</p>;
  }
  if (status.state === "failed") {
    return (
      <section className="m-6 flex flex-col gap-3 border-l-4 border-red-600 pl-4">
        <h2 className="font-semibold">The engine is not running</h2>
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap text-xs">{status.message}</pre>
        <button type="button" className="self-start rounded-md border px-3 py-1" onClick={() => invoke("restart_engine")}>
          Try again
        </button>
      </section>
    );
  }
  return null;
}

function EngineBadge({ status, worker }: { status: EngineStatus; worker: string | null }) {
  const [text, style] =
    status.state === "ready"
      ? [`Engine ready${worker ? ` · worker ${worker}` : ""}`, "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-100"]
      : status.state === "starting"
        ? ["Engine starting…", "bg-zinc-200 text-zinc-700 dark:bg-zinc-700 dark:text-zinc-200"]
        : ["Engine stopped", "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100"];
  return <span className={`rounded-full px-3 py-0.5 text-xs ${style}`}>{text}</span>;
}
