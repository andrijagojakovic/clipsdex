// Server-Sent Events over fetch. The browser's EventSource can't send the bearer token,
// so the stream is read by hand: events are separated by a blank line, each line is
// "field: value", and lines starting with ":" are comments (keep-alive pings).

import { requestRaw, type Connection } from "./api";

export interface StreamEvent {
  event: string;
  data: unknown;
}

/** Splits buffered text into complete events; returns them and the unfinished rest. */
export function parseEvents(buffer: string): { events: StreamEvent[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: StreamEvent[] = [];
  for (const block of blocks) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line === "" || line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    if (data.length === 0) continue;
    try {
      events.push({ event, data: JSON.parse(data.join("\n")) });
    } catch {
      // not JSON: ignore it rather than break the stream
    }
  }
  return { events, rest };
}

/** Calls ``onEvent`` for every event until ``signal`` aborts; reconnects after errors. */
export async function followEvents(
  conn: Connection,
  onEvent: (e: StreamEvent) => void,
  signal: AbortSignal,
  retryMs = 3000,
): Promise<void> {
  while (!signal.aborted) {
    try {
      const response = await requestRaw(conn, "/events", { signal });
      const reader = response.body?.getReader();
      if (!reader) throw new Error("no stream");
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseEvents(buffer);
        buffer = parsed.rest;
        parsed.events.forEach(onEvent);
      }
    } catch {
      if (signal.aborted) return;
    }
    await new Promise((resolve) => setTimeout(resolve, retryMs));
  }
}
