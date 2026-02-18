import type { JobResult } from "../types";

interface Props {
  result: JobResult;
  index: number;
}

function scoreColor(score: number): string {
  if (score >= 0.6) return "bg-emerald-500";
  if (score >= 0.45) return "bg-amber-500";
  return "bg-slate-400";
}

function tagStyle(type: string): string {
  switch (type) {
    case "remote":
      return "bg-emerald-50 text-emerald-700 border-emerald-200";
    case "seniority":
      return "bg-amber-50 text-amber-700 border-amber-200";
    case "employment":
      return "bg-blue-50 text-blue-700 border-blue-200";
    default:
      return "bg-slate-50 text-slate-600 border-slate-200";
  }
}

export function ResultCard({ result, index }: Props) {
  const maxScore = 1.0;
  const scorePercent = Math.min((result.score / maxScore) * 100, 100);

  return (
    <div
      className="animate-fade-in-up bg-white rounded-xl border border-slate-200 p-5 shadow-sm
                 hover:shadow-md hover:-translate-y-0.5 transition-all duration-200"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      {/* Header: rank + title + score */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 min-w-0">
          <span className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-lg bg-slate-100 text-slate-500 text-sm font-semibold">
            {result.rank}
          </span>
          <div className="min-w-0">
            <h3 className="text-base font-semibold text-slate-900 truncate">
              {result.title || "Untitled"}
            </h3>
            <p className="text-sm text-slate-500 truncate">
              {result.company_name || "Unknown Company"} &mdash;{" "}
              {result.location || "Location N/A"}
            </p>
          </div>
        </div>

        {/* Score */}
        <div className="flex-shrink-0 text-right">
          <span className="text-xs font-mono text-slate-400">{result.score.toFixed(4)}</span>
          <div className="mt-1 w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full animate-score-bar ${scoreColor(result.score)}`}
              style={{ width: `${scorePercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1.5 mt-3">
        {result.remote_type && (
          <span className={`px-2 py-0.5 rounded-md text-xs font-medium border ${tagStyle("remote")}`}>
            {result.remote_type}
          </span>
        )}
        {result.seniority_level && (
          <span className={`px-2 py-0.5 rounded-md text-xs font-medium border ${tagStyle("seniority")}`}>
            {result.seniority_level}
          </span>
        )}
        {result.employment_type && (
          <span className={`px-2 py-0.5 rounded-md text-xs font-medium border ${tagStyle("employment")}`}>
            {result.employment_type}
          </span>
        )}
        {result.company_type && (
          <span className={`px-2 py-0.5 rounded-md text-xs font-medium border ${tagStyle("default")}`}>
            {result.company_type}
          </span>
        )}
      </div>

      {/* Salary */}
      {result.salary_display && (
        <p className="mt-2.5 text-sm font-semibold text-emerald-700">
          {result.salary_display}
        </p>
      )}

      {/* Skills */}
      {result.required_skills.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2.5">
          {result.required_skills.slice(0, 6).map((skill) => (
            <span
              key={skill}
              className="px-2 py-0.5 bg-slate-50 text-slate-600 rounded text-xs border border-slate-100"
            >
              {skill}
            </span>
          ))}
          {result.required_skills.length > 6 && (
            <span className="px-2 py-0.5 text-slate-400 text-xs">
              +{result.required_skills.length - 6} more
            </span>
          )}
        </div>
      )}

      {/* Industries */}
      {result.industries.length > 0 && (
        <p className="mt-2 text-xs text-slate-400 truncate">
          {result.industries.join(" · ")}
        </p>
      )}

      {/* Apply link */}
      {result.apply_url && (
        <div className="mt-3 pt-3 border-t border-slate-100">
          <a
            href={result.apply_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm font-medium text-indigo-600 hover:text-indigo-700 transition-colors"
          >
            Apply
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </a>
        </div>
      )}
    </div>
  );
}
