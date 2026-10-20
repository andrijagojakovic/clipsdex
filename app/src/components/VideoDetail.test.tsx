import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import type { VideoRecord } from "../api";
import { VideoDetail } from "./VideoDetail";

vi.mock("@tauri-apps/plugin-opener", () => ({ openUrl: vi.fn() }));
afterEach(cleanup);

const video: VideoRecord = {
  id: 1,
  name: "pricing.mp4",
  original_url: "https://example.com/r/1",
  file_path: "C:/v/pricing.mp4",
  file_deleted: false,
  source: "manual",
  author: "@coach",
  caption: "3 pricing mistakes",
  collection: null,
  saved_at: null,
  imported_at: null,
  category: "business",
  category_by: "auto",
  category_confidence: 0.85,
  profile: {
    duration_s: 42,
    width: 1080,
    height: 1920,
    has_audio: true,
    cuts: 3,
    speech: 0.9,
    music: 0.1,
    silence: 0,
    has_vocals: false,
    has_text: true,
  },
  results: {
    category_guess: {
      value: { category: "business" },
      confidence: 0.85,
      method: "quick profile + gemma3:4b",
      evidence: "Talks about prices.",
      created_at: null,
    },
    brief: {
      value: { has_advice: true, main_message: "Sell packages.", advice: ["Charge per package."] },
      confidence: 0.9,
      method: "transcript read by gemma3:4b",
      evidence: "Sell packages.",
      created_at: null,
    },
  },
  transcript: { language: "en", text: "Stop charging by the hour." },
  on_screen_text: "3 PRICING MISTAKES",
  scenes: ["A man talks to the camera."],
};

describe("VideoDetail", () => {
  it("shows every result with how it was found", () => {
    render(<VideoDetail video={video} onClose={() => undefined} />);
    expect(screen.getByText("Brief notes")).toBeTruthy();
    expect(screen.getByText("Charge per package.")).toBeTruthy();
    expect(screen.getByText(/How it was found: transcript read by gemma3:4b/)).toBeTruthy();
    // The first guess is explained under Category, not repeated as a result.
    expect(screen.getByText(/guessed by the app, 85% sure/)).toBeTruthy();
    expect(screen.getByText("Why: Talks about prices.")).toBeTruthy();
    expect(screen.queryByText("First category guess")).toBeNull();
    expect(screen.getByText("Stop charging by the hour.")).toBeTruthy();
    expect(screen.getByText("Open the original post")).toBeTruthy();
  });

  it("explains the category with the result that set it", () => {
    const categorized = {
      ...video,
      results: {
        ...video.results,
        categorize: {
          value: { category: "business", summary: "Pricing advice.", tags: ["pricing"] },
          confidence: 0.9,
          method: "transcript read by gemma3:4b",
          evidence: "Advice about prices.",
          created_at: null,
        },
      },
    };
    render(<VideoDetail video={categorized} onClose={() => undefined} />);
    expect(screen.getByText("Why: Advice about prices.")).toBeTruthy();
    expect(screen.getByRole("list", { name: "Tags" }).textContent).toBe("pricing");
  });

  it("says when the user chose the category", () => {
    render(
      <VideoDetail video={{ ...video, category_by: "user", results: {} }} onClose={() => undefined} />,
    );
    expect(screen.getByText(/chosen by you/)).toBeTruthy();
    expect(screen.queryByText("Results")).toBeNull();
  });
});
