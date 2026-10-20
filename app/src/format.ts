// Turning engine data into short text for people.

export const ANALYZER_TITLES: Record<string, string> = {
  category_guess: "First category guess",
  categorize: "Category",
  brief: "Brief notes",
  song_id: "Song",
  movie_id: "Movie",
};

export function analyzerTitle(name: string): string {
  return ANALYZER_TITLES[name] ?? name;
}

export function percent(ratio: number | null | undefined): string {
  return ratio === null || ratio === undefined ? "–" : `${Math.round(ratio * 100)}%`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "";
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

type Dict = Record<string, unknown>;

function isDict(v: unknown): v is Dict {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function str(v: unknown): string | null {
  return typeof v === "string" && v.trim() ? v : null;
}

/** One line saying what a result found. */
export function resultSummary(analyzer: string, value: unknown): string {
  if (!isDict(value)) return JSON.stringify(value);
  switch (analyzer) {
    case "category_guess":
    case "categorize": {
      const summary = str(value.summary);
      return summary ? `${value.category}: ${summary}` : String(value.category);
    }
    case "brief":
      return value.has_advice === false
        ? "No advice found"
        : (str(value.main_message) ?? "Notes taken");
    case "song_id":
    case "movie_id": {
      if (!value.found) return "Not identified";
      const title = str(value.title) ?? "Unknown";
      const extra = str(value.artist) ?? (value.year ? String(value.year) : null);
      return extra ? `${title} – ${extra}` : title;
    }
    default:
      return JSON.stringify(value);
  }
}

/** Lists inside a result worth showing as bullets (brief notes, tags). */
export function resultLists(value: unknown): { label: string; items: string[] }[] {
  if (!isDict(value)) return [];
  const lists: { label: string; items: string[] }[] = [];
  for (const [key, label] of [
    ["advice", "Advice"],
    ["facts", "Facts"],
    ["steps", "Steps"],
    ["tags", "Tags"],
  ] as const) {
    const items = value[key];
    if (Array.isArray(items) && items.length > 0) {
      lists.push({ label, items: items.map(String) });
    }
  }
  return lists;
}

export function matchesText(
  video: { name: string; author: string | null; caption: string | null },
  query: string,
): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return [video.name, video.author, video.caption].some((v) => v?.toLowerCase().includes(q));
}
