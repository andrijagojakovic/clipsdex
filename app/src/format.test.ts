import { describe, expect, it } from "vitest";
import { duration, matchesText, percent, resultLists, resultSummary } from "./format";

describe("format", () => {
  it("summarizes each kind of result", () => {
    expect(resultSummary("categorize", { category: "business", summary: "Pricing tips." })).toBe(
      "business: Pricing tips.",
    );
    expect(resultSummary("category_guess", { category: "edit" })).toBe("edit");
    expect(resultSummary("brief", { has_advice: false })).toBe("No advice found");
    expect(resultSummary("brief", { has_advice: true, main_message: "Sell packages." })).toBe(
      "Sell packages.",
    );
    expect(resultSummary("song_id", { found: true, title: "Song A", artist: "Artist A" })).toBe(
      "Song A – Artist A",
    );
    expect(resultSummary("movie_id", { found: true, title: "Heat", year: 1995 })).toBe(
      "Heat – 1995",
    );
    expect(resultSummary("movie_id", { found: false })).toBe("Not identified");
  });

  it("lists notes and tags", () => {
    expect(resultLists({ advice: ["A"], facts: [], tags: ["x", "y"] })).toEqual([
      { label: "Advice", items: ["A"] },
      { label: "Tags", items: ["x", "y"] },
    ]);
    expect(resultLists("text")).toEqual([]);
  });

  it("formats numbers for people", () => {
    expect(percent(0.856)).toBe("86%");
    expect(percent(null)).toBe("–");
    expect(duration(65.4)).toBe("1:05");
    expect(duration(null)).toBe("");
  });

  it("filters by name, author or caption", () => {
    const v = { name: "pricing.mp4", author: "@Coach", caption: "Цене за фриленсере" };
    expect(matchesText(v, "coach")).toBe(true);
    expect(matchesText(v, "цене")).toBe(true);
    expect(matchesText(v, "  ")).toBe(true);
    expect(matchesText(v, "dentist")).toBe(false);
  });
});
