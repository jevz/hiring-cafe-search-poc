import type { SearchMeta as SearchMetaType } from "../types";

interface Props {
  meta: SearchMetaType | null;
  resultCount: number;
}

export function SearchMeta({ meta, resultCount }: Props) {
  if (!meta) return null;

  const totalTime = meta.intent_time_ms + meta.embed_time_ms + meta.search_time_ms;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-5 py-3 bg-white border border-slate-200 rounded-xl shadow-sm text-xs text-slate-500">
      <span>
        Showing <span className="font-medium text-slate-700">{resultCount}</span> of{" "}
        <span className="font-medium text-slate-700">{meta.total_jobs.toLocaleString()}</span> jobs
        {meta.matched_filters < meta.total_jobs && (
          <>
            {" "}
            (<span className="font-medium text-slate-700">{meta.matched_filters.toLocaleString()}</span> passed
            filters)
          </>
        )}
      </span>
      <span className="hidden sm:inline text-slate-300">|</span>
      <span>
        Total: <span className="font-mono">{totalTime.toFixed(0)}ms</span>
      </span>
      <span className="hidden sm:inline text-slate-300">|</span>
      <span>
        Intent: <span className="font-mono">{meta.intent_time_ms.toFixed(0)}ms</span>
      </span>
      <span className="hidden sm:inline text-slate-300">|</span>
      <span>
        Embed: <span className="font-mono">{meta.embed_time_ms.toFixed(0)}ms</span>
      </span>
      <span className="hidden sm:inline text-slate-300">|</span>
      <span>
        Search: <span className="font-mono">{meta.search_time_ms.toFixed(0)}ms</span>
      </span>
    </div>
  );
}
