import type { Connection, VideoItem } from "../api";
import { analyzerTitle, duration, percent } from "../format";
import { useThumbnail } from "../hooks";

export type Layout = "grid" | "list";

interface Props {
  conn: Connection | null;
  videos: VideoItem[];
  layout: Layout;
  selectedId: number | null; // the video open in the detail panel
  checked: Set<number>; // videos picked for an action
  onOpen: (id: number) => void;
  onCheck: (id: number) => void;
}

export function CategoryChip({ video }: { video: Pick<VideoItem, "category" | "category_by" | "category_confidence"> }) {
  if (!video.category) {
    return <span className="text-xs text-zinc-500">not sorted yet</span>;
  }
  const how =
    video.category_by === "user"
      ? "chosen by you"
      : `guessed, ${percent(video.category_confidence)} sure`;
  return (
    <span
      title={how}
      className={`rounded-full px-2 py-0.5 text-xs ${
        video.category_by === "user"
          ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-100"
          : "bg-zinc-200 text-zinc-700 dark:bg-zinc-700 dark:text-zinc-200"
      }`}
    >
      {video.category}
    </span>
  );
}

function Thumbnail({ conn, video, className }: { conn: Connection | null; video: VideoItem; className: string }) {
  const { ref, src } = useThumbnail(conn, video.id, video.has_file || video.profiled);
  return (
    <div ref={ref} className={`overflow-hidden bg-zinc-200 dark:bg-zinc-800 ${className}`}>
      {src ? (
        <img src={src} alt="" className="h-full w-full object-cover" />
      ) : (
        <div className="flex h-full items-center justify-center text-xs text-zinc-500">
          {video.has_file ? "" : "link only"}
        </div>
      )}
    </div>
  );
}

interface ItemProps {
  conn: Connection | null;
  video: VideoItem;
  active: boolean;
  checked: boolean;
  onOpen: () => void;
  onCheck: () => void;
}

function Check({ video, checked, onCheck, className }: Pick<ItemProps, "video" | "checked" | "onCheck"> & { className: string }) {
  return (
    <input
      type="checkbox"
      checked={checked}
      onChange={onCheck}
      onClick={(e) => e.stopPropagation()}
      aria-label={`Select ${video.name}`}
      className={`h-4 w-4 cursor-pointer accent-sky-600 ${className}`}
    />
  );
}

function Card({ conn, video, active, checked, onOpen, onCheck }: ItemProps) {
  return (
    <div className="group relative">
      <button
        type="button"
        onClick={onOpen}
        aria-pressed={active}
        className={`flex w-full flex-col overflow-hidden rounded-lg border text-left ${
          active || checked
            ? "border-sky-500 ring-2 ring-sky-500/40"
            : "border-zinc-200 hover:border-zinc-400 dark:border-zinc-700 dark:hover:border-zinc-500"
        }`}
      >
        <div className="relative w-full">
          <Thumbnail conn={conn} video={video} className="aspect-[9/16] w-full" />
          {video.duration_s !== null && (
            <span className="absolute right-1.5 bottom-1.5 rounded bg-black/70 px-1.5 text-xs text-white">
              {duration(video.duration_s)}
            </span>
          )}
        </div>
        <div className="flex w-full flex-col gap-1 p-2">
          <span className="truncate text-sm font-medium" title={video.name}>
            {video.name}
          </span>
          <div className="flex items-center justify-between gap-2">
            <CategoryChip video={video} />
            {video.author && <span className="truncate text-xs text-zinc-500">{video.author}</span>}
          </div>
        </div>
      </button>
      <Check
        video={video}
        checked={checked}
        onCheck={onCheck}
        className={`absolute top-2 left-2 ${checked ? "" : "opacity-0 group-hover:opacity-100 focus:opacity-100"}`}
      />
    </div>
  );
}

function Row({ conn, video, active, checked, onOpen, onCheck }: ItemProps) {
  return (
    <tr
      onClick={onOpen}
      aria-selected={active}
      className={`cursor-pointer border-b border-zinc-200 dark:border-zinc-800 ${
        active || checked ? "bg-sky-50 dark:bg-sky-950" : "hover:bg-zinc-100 dark:hover:bg-zinc-800/60"
      }`}
    >
      <td className="w-8 pl-2">
        <Check video={video} checked={checked} onCheck={onCheck} className="" />
      </td>
      <td className="w-12 py-1.5">
        <Thumbnail conn={conn} video={video} className="h-12 w-9 rounded" />
      </td>
      <td className="max-w-64 truncate px-2 text-sm" title={video.name}>
        {video.name}
      </td>
      <td className="px-2"><CategoryChip video={video} /></td>
      <td className="px-2 text-sm text-zinc-500">{video.author ?? ""}</td>
      <td className="px-2 text-sm tabular-nums text-zinc-500">{duration(video.duration_s)}</td>
      <td className="px-2 text-xs text-zinc-500">
        {video.results.filter((r) => r !== "category_guess").map(analyzerTitle).join(", ")}
      </td>
    </tr>
  );
}

export function VideoGrid({ conn, videos, layout, selectedId, checked, onOpen, onCheck }: Props) {
  if (videos.length === 0) {
    return <p className="p-8 text-center text-zinc-500">No videos here.</p>;
  }
  const item = (v: VideoItem) => ({
    conn,
    video: v,
    active: v.id === selectedId,
    checked: checked.has(v.id),
    onOpen: () => onOpen(v.id),
    onCheck: () => onCheck(v.id),
  });
  if (layout === "list") {
    return (
      <table className="w-full border-collapse">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wide text-zinc-500">
            <th className="py-2 pl-2" />
            <th />
            <th className="px-2">Name</th>
            <th className="px-2">Category</th>
            <th className="px-2">Author</th>
            <th className="px-2">Length</th>
            <th className="px-2">Results</th>
          </tr>
        </thead>
        <tbody>
          {videos.map((v) => (
            <Row key={v.id} {...item(v)} />
          ))}
        </tbody>
      </table>
    );
  }
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-3">
      {videos.map((v) => (
        <Card key={v.id} {...item(v)} />
      ))}
    </div>
  );
}
