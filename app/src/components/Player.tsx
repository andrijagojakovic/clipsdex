import { useEffect, useState } from "react";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { ApiError, request, type Connection, type VideoRecord } from "../api";

type Props = { conn: Connection; video: Pick<VideoRecord, "id" | "file_path" | "file_deleted" | "original_url"> };

/** Plays the local file through a signed link from the engine (a <video> element can't
 * send the token). An expired link is fetched again once; a format the web view can't
 * play offers to show the file in its folder instead. */
export function Player({ conn, video }: Props) {
  const [src, setSrc] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const hasFile = Boolean(video.file_path) && !video.file_deleted;

  useEffect(() => {
    setSrc(null);
    setProblem(null);
    if (!hasFile) return;
    let cancelled = false;
    request<{ url: string }>(conn, `/videos/${video.id}/play`)
      .then((r) => !cancelled && setSrc(`${conn.url}${r.url}`))
      .catch((e: unknown) => !cancelled && setProblem(e instanceof ApiError ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [conn.url, conn.token, video.id, hasFile, attempt]);

  if (!hasFile) {
    return (
      <p className="rounded-lg bg-zinc-100 p-3 text-sm text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
        The video file was removed. The results{video.original_url ? " and the link to the original post" : ""} are
        kept.
      </p>
    );
  }
  if (problem) {
    return (
      <div className="flex flex-col gap-2 rounded-lg bg-zinc-100 p-3 text-sm dark:bg-zinc-800">
        <p>{problem}</p>
        {video.file_path && (
          <button
            type="button"
            className="self-start text-sky-600 hover:underline"
            onClick={() => revealItemInDir(video.file_path ?? "")}
          >
            Show in folder
          </button>
        )}
      </div>
    );
  }
  if (!src) return <div className="aspect-[9/16] max-h-[55vh] rounded-lg bg-zinc-200 dark:bg-zinc-800" />;
  return (
    <video
      key={src}
      src={src}
      controls
      preload="metadata"
      className="max-h-[55vh] w-full rounded-lg bg-black"
      onError={() => {
        if (attempt === 0) {
          setAttempt(1); // the link may have expired: get a fresh one once
        } else {
          setProblem("This video can't be played inside ClipsDex (its format isn't supported here).");
        }
      }}
    />
  );
}
