import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Player } from "./Player";

vi.mock("@tauri-apps/plugin-opener", () => ({ revealItemInDir: vi.fn() }));

const conn = { url: "http://127.0.0.1:5000", token: "t" };
const video = { id: 1, file_path: "C:/v/clip.mp4", file_deleted: false, original_url: null };

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function engineAnswers(...bodies: [number, unknown][]) {
  const fetcher = vi.fn();
  for (const [status, body] of bodies) {
    fetcher.mockResolvedValueOnce(new Response(JSON.stringify(body), { status }));
  }
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

describe("Player", () => {
  it("plays through a signed link and gets a new one once if it fails", async () => {
    const fetcher = engineAnswers(
      [200, { url: "/media/1?exp=1&sig=a" }],
      [200, { url: "/media/1?exp=2&sig=b" }],
    );
    const { container } = render(<Player conn={conn} video={video} />);
    await waitFor(() => expect(container.querySelector("video")).not.toBeNull());
    const first = container.querySelector("video") as HTMLVideoElement;
    expect(first.getAttribute("src")).toBe("http://127.0.0.1:5000/media/1?exp=1&sig=a");
    const [, init] = fetcher.mock.calls[0] as [string, RequestInit];
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer t");

    fireEvent.error(first); // e.g. the link expired
    await waitFor(() =>
      expect(container.querySelector("video")?.getAttribute("src")).toContain("sig=b"),
    );
    fireEvent.error(container.querySelector("video") as HTMLVideoElement);
    expect(await screen.findByText(/can't be played inside ClipsDex/)).toBeTruthy();
    expect(screen.getByText("Show in folder")).toBeTruthy();
  });

  it("shows the engine's reason when the file is missing", async () => {
    engineAnswers([404, { error: "The video file is no longer at C:/v/clip.mp4." }]);
    render(<Player conn={conn} video={video} />);
    expect(await screen.findByText("The video file is no longer at C:/v/clip.mp4.")).toBeTruthy();
  });

  it("explains removed files without asking the engine", () => {
    const fetcher = engineAnswers();
    render(
      <Player
        conn={conn}
        video={{ ...video, file_path: null, file_deleted: true, original_url: "https://e.com/1" }}
      />,
    );
    expect(screen.getByText(/The video file was removed/)).toBeTruthy();
    expect(fetcher).not.toHaveBeenCalled();
  });
});
