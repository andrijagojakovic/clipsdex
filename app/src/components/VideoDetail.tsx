import { useState } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import type { Connection, ResultRecord, VideoRecord } from "../api";
import { Player } from "./Player";
import { analyzerTitle, duration, percent, resultLists, resultSummary } from "../format";

// Results that are only shown inside another section.
const SHOWN_ELSEWHERE = new Set(["category_guess"]);

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">{title}</h3>
      {children}
    </section>
  );
}

function Confidence({ value }: { value: number | null }) {
  if (value === null) return null;
  return (
    <span className="flex items-center gap-1.5 text-xs text-zinc-500" title="How sure the app is">
      <span className="h-1.5 w-16 overflow-hidden rounded bg-zinc-200 dark:bg-zinc-700">
        <span className="block h-full bg-sky-500" style={{ width: percent(value) }} />
      </span>
      {percent(value)}
    </span>
  );
}

/** A result with how it was found: the method and the evidence, always visible. */
export function ResultCard({ name, result }: { name: string; result: ResultRecord }) {
  return (
    <article className="flex flex-col gap-1.5 rounded-lg border border-zinc-200 p-3 dark:border-zinc-700">
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-sm font-semibold">{analyzerTitle(name)}</h4>
        <Confidence value={result.confidence} />
      </div>
      <p className="text-sm">{resultSummary(name, result.value)}</p>
      {resultLists(result.value).map((list) =>
        list.label === "Tags" ? (
          <ul key={list.label} aria-label="Tags" className="flex flex-wrap gap-1">
            {list.items.map((tag) => (
              <li key={tag} className="rounded-full bg-zinc-200 px-2 py-0.5 text-xs dark:bg-zinc-700">
                {tag}
              </li>
            ))}
          </ul>
        ) : (
          <div key={list.label}>
            <p className="text-xs text-zinc-500">{list.label}</p>
            <ul className="ml-4 list-disc text-sm">
              {list.items.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        ),
      )}
      <p className="text-xs text-zinc-500">
        How it was found: {result.method}
        {result.evidence ? ` — “${result.evidence}”` : ""}
      </p>
    </article>
  );
}

function Fold({ text, limit = 400 }: { text: string; limit?: number }) {
  const [open, setOpen] = useState(false);
  if (text.length <= limit) return <p className="whitespace-pre-wrap text-sm">{text}</p>;
  return (
    <div>
      <p className="whitespace-pre-wrap text-sm">{open ? text : `${text.slice(0, limit)}…`}</p>
      <button type="button" className="text-xs text-sky-600" onClick={() => setOpen(!open)}>
        {open ? "Show less" : "Show all"}
      </button>
    </div>
  );
}

export function VideoDetail({
  video,
  onClose,
  conn = null,
}: {
  video: VideoRecord;
  onClose: () => void;
  conn?: Connection | null;
}) {
  const p = video.profile;
  const results = Object.entries(video.results).filter(([name]) => !SHOWN_ELSEWHERE.has(name));
  // The reason comes from whichever result set the current category: the full
  // categorize step if it agrees, otherwise the first guess.
  const decidedBy = [video.results.categorize, video.results.category_guess].find(
    (r) => r && (r.value as { category?: string } | null)?.category === video.category,
  );
  const categoryHow =
    video.category_by === "user"
      ? "chosen by you"
      : video.category
        ? `guessed by the app, ${percent(video.category_confidence)} sure`
        : null;

  return (
    <aside aria-label="Video details" className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold" title={video.name}>
            {video.name}
          </h2>
          {video.author && <p className="text-sm text-zinc-500">{video.author}</p>}
        </div>
        <button type="button" onClick={onClose} aria-label="Close details" className="rounded px-2 text-lg">
          ×
        </button>
      </div>

      {conn && <Player conn={conn} video={video} />}

      {video.original_url && (
        <button
          type="button"
          className="self-start text-sm text-sky-600 hover:underline"
          onClick={() => openUrl(video.original_url ?? "")}
        >
          Open the original post
        </button>
      )}

      <Section title="Category">
        <p className="text-sm">
          {video.category ?? "Not sorted yet"}
          {categoryHow && <span className="text-zinc-500"> · {categoryHow}</span>}
        </p>
        {decidedBy && video.category_by !== "user" && decidedBy.evidence && (
          <p className="text-xs text-zinc-500">Why: {decidedBy.evidence}</p>
        )}
      </Section>

      {video.caption && (
        <Section title="Caption">
          <Fold text={video.caption} limit={300} />
        </Section>
      )}

      {p && (
        <Section title="Profile">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-sm">
            <dt className="text-zinc-500">Length</dt>
            <dd>{duration(p.duration_s)}</dd>
            <dt className="text-zinc-500">Cuts</dt>
            <dd>{p.cuts ?? "–"}</dd>
            <dt className="text-zinc-500">Speech / music</dt>
            <dd>
              {p.has_audio === false ? "no sound" : `${percent(p.speech)} / ${percent(p.music)}`}
            </dd>
            <dt className="text-zinc-500">Text on screen</dt>
            <dd>{p.has_text === null ? "–" : p.has_text ? "yes" : "no"}</dd>
          </dl>
        </Section>
      )}

      {results.length > 0 && (
        <Section title="Results">
          {results.map(([name, r]) => (
            <ResultCard key={name} name={name} result={r} />
          ))}
        </Section>
      )}

      {video.transcript?.text && (
        <Section title={`What is said${video.transcript.language ? ` (${video.transcript.language})` : ""}`}>
          <Fold text={video.transcript.text} />
        </Section>
      )}
      {video.on_screen_text && (
        <Section title="Text on screen">
          <Fold text={video.on_screen_text} limit={300} />
        </Section>
      )}
      {video.scenes.length > 0 && (
        <Section title="Scenes">
          <ul className="ml-4 list-disc text-sm">
            {video.scenes.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </Section>
      )}
    </aside>
  );
}
