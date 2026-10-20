// Talking to the engine's local API: every request carries the token, and every error
// comes back as an ApiError with the engine's user-facing message.

export type EngineStatus =
  | { state: "starting" }
  | { state: "ready"; url: string; token: string }
  | { state: "failed"; message: string };

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface Connection {
  url: string;
  token: string;
}

type Fetch = typeof fetch;

async function send(
  conn: Connection,
  path: string,
  init: RequestInit,
  fetcher: Fetch,
): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${conn.token}`);
  if (init.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  let response: Response;
  try {
    response = await fetcher(`${conn.url}${path}`, { ...init, headers });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(0, "The ClipsDex engine is not reachable.");
  }
  if (!response.ok) {
    let message = `The engine answered ${response.status}.`;
    try {
      const body = (await response.json()) as { error?: unknown };
      if (typeof body.error === "string" && body.error) message = body.error;
    } catch {
      // not JSON: keep the generic message
    }
    throw new ApiError(response.status, message);
  }
  return response;
}

export async function request<T>(
  conn: Connection,
  path: string,
  init: RequestInit = {},
  fetcher: Fetch = fetch,
): Promise<T> {
  return (await (await send(conn, path, init, fetcher)).json()) as T;
}

/** The response itself, for pictures and streams. */
export function requestRaw(
  conn: Connection,
  path: string,
  init: RequestInit = {},
  fetcher: Fetch = fetch,
): Promise<Response> {
  return send(conn, path, init, fetcher);
}

export interface Health {
  status: string;
  version: string;
  worker: string | null;
}

export type AssignedBy = "auto" | "user";

export interface VideoItem {
  id: number;
  name: string;
  author: string | null;
  caption: string | null;
  original_url: string | null;
  source: string | null;
  saved_at: string | null;
  imported_at: string;
  duration_s: number | null;
  profiled: boolean;
  category: string | null;
  category_by: AssignedBy | null;
  category_confidence: number | null;
  results: string[];
  has_file: boolean;
  file_deleted: boolean;
}

export interface CategoryItem {
  name: string;
  description: string | null;
  color: string | null;
  created_by: "system" | "user";
  videos: number;
  by_user: number;
}

export interface ResultRecord {
  value: unknown;
  confidence: number | null;
  method: string;
  evidence: string | null;
  created_at: string | null;
}

export interface VideoRecord {
  id: number;
  name: string;
  original_url: string | null;
  file_path: string | null;
  file_deleted: boolean;
  source: string;
  author: string | null;
  caption: string | null;
  collection: string | null;
  saved_at: string | null;
  imported_at: string | null;
  category: string | null;
  category_by: AssignedBy | null;
  category_confidence: number | null;
  profile: {
    duration_s: number | null;
    width: number | null;
    height: number | null;
    has_audio: boolean | null;
    cuts: number | null;
    speech: number | null;
    music: number | null;
    silence: number | null;
    has_vocals: boolean | null;
    has_text: boolean | null;
  } | null;
  results: Record<string, ResultRecord>;
  transcript: { language: string | null; text: string | null } | null;
  on_screen_text: string | null;
  scenes: string[];
}
