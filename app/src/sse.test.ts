import { describe, expect, it } from "vitest";
import { parseEvents } from "./sse";

describe("parseEvents", () => {
  it("reads complete events and keeps the unfinished rest", () => {
    const text =
      "retry: 3000\n\n" +
      'id: 0\nevent: snapshot\ndata: {"counts": {"queued": 1}}\n\n' +
      ": ping\n\n" +
      'id: 1\nevent: job\ndata: {"id": 7, "status": "running"}\n\n' +
      'id: 2\nevent: wor';
    const { events, rest } = parseEvents(text);
    expect(events).toEqual([
      { event: "snapshot", data: { counts: { queued: 1 } } },
      { event: "job", data: { id: 7, status: "running" } },
    ]);
    expect(rest).toBe("id: 2\nevent: wor");
  });

  it("handles Windows line endings and skips broken data", () => {
    const { events } = parseEvents(
      'event: worker\r\ndata: {"message": "load x"}\r\n\r\nevent: job\r\ndata: {broken\r\n\r\n',
    );
    expect(events).toEqual([{ event: "worker", data: { message: "load x" } }]);
  });
});
