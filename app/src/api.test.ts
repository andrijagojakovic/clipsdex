import { describe, expect, it, vi } from "vitest";
import { ApiError, request } from "./api";

const conn = { url: "http://127.0.0.1:5000", token: "secret-token" };

function answer(status: number, body: unknown): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
}

describe("request", () => {
  it("sends the token and JSON", async () => {
    const fetcher = vi.fn(async () => answer(200, { ok: true }));
    const result = await request<{ ok: boolean }>(
      conn,
      "/plan",
      { method: "POST", body: JSON.stringify({ analyzers: ["brief"] }) },
      fetcher,
    );
    expect(result).toEqual({ ok: true });
    const [url, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("http://127.0.0.1:5000/plan");
    const headers = new Headers(init.headers);
    expect(headers.get("Authorization")).toBe("Bearer secret-token");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("turns the engine's error message into an ApiError", async () => {
    const fetcher = vi.fn(async () => answer(400, { error: 'There is no category "boats".' }));
    await expect(request(conn, "/categories/boats", {}, fetcher)).rejects.toEqual(
      new ApiError(400, 'There is no category "boats".'),
    );
  });

  it("explains errors without a JSON body", async () => {
    const fetcher = vi.fn(async () => answer(502, "<html>bad gateway</html>"));
    await expect(request(conn, "/health", {}, fetcher)).rejects.toMatchObject({
      status: 502,
      message: "The engine answered 502.",
    });
  });

  it("says when the engine can't be reached", async () => {
    const fetcher = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    await expect(request(conn, "/health", {}, fetcher)).rejects.toMatchObject({
      status: 0,
      message: "The ClipsDex engine is not reachable.",
    });
  });
});
