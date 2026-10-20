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

export async function request<T>(
  conn: Connection,
  path: string,
  init: RequestInit = {},
  fetcher: Fetch = fetch,
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${conn.token}`);
  if (init.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  let response: Response;
  try {
    response = await fetcher(`${conn.url}${path}`, { ...init, headers });
  } catch {
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
  return (await response.json()) as T;
}

export interface Health {
  status: string;
  version: string;
  worker: string | null;
}

export interface VideoItem {
  id: number;
  name: string;
  category: string | null;
}

export interface CategoryItem {
  name: string;
  description: string | null;
  videos: number;
}
