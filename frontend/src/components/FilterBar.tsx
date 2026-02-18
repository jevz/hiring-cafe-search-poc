import type { ParsedIntent } from "../types";

interface Props {
  intent: ParsedIntent | null;
}

const FILTER_COLORS: Record<string, { bg: string; text: string }> = {
  remote_type: { bg: "bg-emerald-100", text: "text-emerald-700" },
  seniority_level: { bg: "bg-amber-100", text: "text-amber-700" },
  employment_type: { bg: "bg-blue-100", text: "text-blue-700" },
  company_type: { bg: "bg-purple-100", text: "text-purple-700" },
  min_salary: { bg: "bg-rose-100", text: "text-rose-700" },
  max_salary: { bg: "bg-rose-100", text: "text-rose-700" },
  industries: { bg: "bg-teal-100", text: "text-teal-700" },
};

const FILTER_LABELS: Record<string, string> = {
  remote_type: "Remote",
  seniority_level: "Seniority",
  employment_type: "Type",
  company_type: "Company",
  min_salary: "Salary",
  max_salary: "Salary",
  industries: "Industry",
};

function formatFilterValue(key: string, value: string | number | string[]): string {
  if (key === "min_salary" && typeof value === "number") {
    return `>= $${(value / 1000).toFixed(0)}k`;
  }
  if (key === "max_salary" && typeof value === "number") {
    return `<= $${(value / 1000).toFixed(0)}k`;
  }
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  return String(value);
}

function describeSearchFocus(weights: { explicit: number; inferred: number; company: number }): string {
  const { explicit, inferred, company } = weights;
  if (company >= 0.4) return "Company-focused search";
  if (inferred > explicit) return "Skill-focused search";
  if (explicit >= 0.55) return "Role-focused search";
  return "Balanced search";
}

export function FilterBar({ intent }: Props) {
  if (!intent) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {Object.entries(intent.filters).map(([key, value]) => {
        const colors = FILTER_COLORS[key] || { bg: "bg-slate-100", text: "text-slate-700" };
        const label = FILTER_LABELS[key] || key;

        return (
          <span
            key={key}
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${colors.bg} ${colors.text}`}
          >
            <span className="opacity-70">{label}:</span>
            {formatFilterValue(key, value)}
          </span>
        );
      })}

      {/* Search focus indicator */}
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium bg-slate-100 text-slate-500">
        {describeSearchFocus(intent.weights)}
      </span>

      {intent.exclusions.length > 0 && (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-700">
          <span className="opacity-70">Excluding:</span>
          {intent.exclusions.join(", ")}
        </span>
      )}
    </div>
  );
}
