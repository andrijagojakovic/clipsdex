import type { CategoryItem } from "../api";

interface Props {
  categories: CategoryItem[];
  total: number;
  uncategorized: number;
  selected: string | null; // null = all videos, "" = no category yet
  onSelect: (category: string | null) => void;
}

function Row({
  label,
  count,
  active,
  title,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  title?: string;
  onClick: () => void;
}) {
  return (
    <li>
      <button
        type="button"
        title={title}
        onClick={onClick}
        aria-current={active ? "true" : undefined}
        className={`flex w-full items-center justify-between rounded-md px-3 py-1.5 text-left text-sm ${
          active
            ? "bg-zinc-200 font-semibold dark:bg-zinc-700"
            : "hover:bg-zinc-100 dark:hover:bg-zinc-800"
        }`}
      >
        <span className="truncate">{label}</span>
        <span className="ml-2 text-xs text-zinc-500">{count}</span>
      </button>
    </li>
  );
}

export function Sidebar({ categories, total, uncategorized, selected, onSelect }: Props) {
  const used = categories.filter((c) => c.videos > 0);
  const empty = categories.filter((c) => c.videos === 0);
  return (
    <nav aria-label="Categories" className="flex flex-col gap-4 overflow-y-auto p-3">
      <ul className="flex flex-col gap-0.5">
        <Row label="All videos" count={total} active={selected === null} onClick={() => onSelect(null)} />
        {uncategorized > 0 && (
          <Row
            label="Not sorted yet"
            count={uncategorized}
            active={selected === ""}
            onClick={() => onSelect("")}
          />
        )}
      </ul>
      <div>
        <h2 className="px-3 pb-1 text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Categories
        </h2>
        <ul className="flex flex-col gap-0.5">
          {[...used, ...empty].map((c) => (
            <Row
              key={c.name}
              label={c.name}
              count={c.videos}
              title={c.description ?? undefined}
              active={selected === c.name}
              onClick={() => onSelect(c.name)}
            />
          ))}
        </ul>
      </div>
    </nav>
  );
}
