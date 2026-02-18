import type { JobResult } from "../types";
import { ResultCard } from "./ResultCard";

interface Props {
  results: JobResult[];
  isLoading: boolean;
}

function SkeletonCard({ index }: { index: number }) {
  return (
    <div
      className="animate-fade-in-up bg-white rounded-xl border border-slate-200 p-5 shadow-sm"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      <div className="flex items-start gap-3">
        <div className="skeleton w-8 h-8 rounded-lg" />
        <div className="flex-1 space-y-2">
          <div className="skeleton h-5 w-3/4" />
          <div className="skeleton h-4 w-1/2" />
        </div>
      </div>
      <div className="flex gap-1.5 mt-3">
        <div className="skeleton h-6 w-16 rounded-md" />
        <div className="skeleton h-6 w-20 rounded-md" />
        <div className="skeleton h-6 w-16 rounded-md" />
      </div>
      <div className="flex gap-1 mt-2.5">
        <div className="skeleton h-5 w-14 rounded" />
        <div className="skeleton h-5 w-18 rounded" />
        <div className="skeleton h-5 w-12 rounded" />
      </div>
    </div>
  );
}

export function ResultsList({ results, isLoading }: Props) {
  if (isLoading) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <SkeletonCard key={i} index={i} />
        ))}
      </div>
    );
  }

  if (results.length === 0) return null;

  return (
    <div className="space-y-3">
      {results.map((result, i) => (
        <ResultCard key={result.id} result={result} index={i} />
      ))}
    </div>
  );
}
