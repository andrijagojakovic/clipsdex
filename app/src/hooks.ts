import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  ApiError,
  request,
  requestRaw,
  type CategoryItem,
  type Connection,
  type EngineStatus,
  type Health,
  type VideoItem,
  type VideoRecord,
} from "./api";

export function useEngine(): EngineStatus {
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

function message(e: unknown): string {
  return e instanceof ApiError || e instanceof Error ? e.message : String(e);
}

export interface Library {
  health: Health | null;
  videos: VideoItem[];
  categories: CategoryItem[];
  loading: boolean;
  loadedAt: number; // changes on every successful load
  error: string | null;
  reload: () => void;
}

/** Videos and categories; reloads when `changed` changes (a job finished). */
export function useLibrary(conn: Connection | null, changed = 0): Library {
  const [state, setState] = useState<Omit<Library, "reload">>({
    health: null,
    videos: [],
    categories: [],
    loading: true,
    loadedAt: 0,
    error: null,
  });
  const [version, setVersion] = useState(0);
  const reload = useCallback(() => setVersion((v) => v + 1), []);

  useEffect(() => {
    if (conn === null) return;
    let cancelled = false;
    Promise.all([
      request<Health>(conn, "/health"),
      request<{ videos: VideoItem[] }>(conn, "/videos?limit=10000"),
      request<{ categories: CategoryItem[] }>(conn, "/categories"),
    ])
      .then(([health, v, c]) => {
        if (!cancelled) {
          setState({
            health,
            videos: v.videos,
            categories: c.categories,
            loading: false,
            loadedAt: Date.now(),
            error: null,
          });
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) setState((s) => ({ ...s, loading: false, error: message(e) }));
      });
    return () => {
      cancelled = true;
    };
  }, [conn?.url, conn?.token, version, changed]);

  return { ...state, reload };
}

export function useVideo(conn: Connection | null, id: number | null, version: number) {
  const [video, setVideo] = useState<VideoRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (conn === null || id === null) {
      setVideo(null);
      return;
    }
    let cancelled = false;
    setError(null);
    request<VideoRecord>(conn, `/videos/${id}`)
      .then((v) => !cancelled && setVideo(v))
      .catch((e: unknown) => !cancelled && setError(message(e)));
    return () => {
      cancelled = true;
    };
  }, [conn?.url, conn?.token, id, version]);
  return { video, error };
}

/** A thumbnail as an object URL, fetched only once the element is on screen. */
export function useThumbnail(conn: Connection | null, id: number, enabled: boolean) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState(false);
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node || visible) return;
    if (typeof IntersectionObserver === "undefined") {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => entries.some((e) => e.isIntersecting) && setVisible(true),
      { rootMargin: "300px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [visible]);

  useEffect(() => {
    if (!visible || !enabled || conn === null) return;
    let url: string | null = null;
    let cancelled = false;
    requestRaw(conn, `/videos/${id}/thumbnail`)
      .then((r) => r.blob())
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setSrc(url);
      })
      .catch(() => undefined); // no picture: the card shows a placeholder
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [visible, enabled, conn?.url, conn?.token, id]);

  return { ref, src };
}
