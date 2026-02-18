import { useEffect, useRef, useState } from "react";

const PLACEHOLDER_EXAMPLES = [
  "python backend engineer at a startup",
  "remote machine learning jobs paying over 150k",
  "senior data scientist in healthcare",
  "entry level software engineer",
  "non-profit product manager",
];

interface Props {
  onSearch: (query: string) => void;
  onClear: () => void;
  isLoading: boolean;
  hasHistory: boolean;
}

export function SearchBar({ onSearch, onClear, isLoading, hasHistory }: Props) {
  const [query, setQuery] = useState("");
  const [placeholderIndex, setPlaceholderIndex] = useState(0);
  const [placeholderText, setPlaceholderText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Animated placeholder
  useEffect(() => {
    const target = PLACEHOLDER_EXAMPLES[placeholderIndex];
    let charIndex = 0;
    let deleting = false;
    let timeout: ReturnType<typeof setTimeout>;

    const tick = () => {
      if (!deleting) {
        setPlaceholderText(target.slice(0, charIndex + 1));
        charIndex++;
        if (charIndex >= target.length) {
          timeout = setTimeout(() => {
            deleting = true;
            tick();
          }, 2000);
          return;
        }
        timeout = setTimeout(tick, 50);
      } else {
        setPlaceholderText(target.slice(0, charIndex));
        charIndex--;
        if (charIndex < 0) {
          setPlaceholderIndex((i) => (i + 1) % PLACEHOLDER_EXAMPLES.length);
          return;
        }
        timeout = setTimeout(tick, 30);
      }
    };

    timeout = setTimeout(tick, 500);
    return () => clearTimeout(timeout);
  }, [placeholderIndex]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSearch(query.trim());
      setQuery("");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative flex items-center gap-3">
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 flex items-center pl-4 pointer-events-none">
            <svg
              className="w-5 h-5 text-slate-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={placeholderText}
            className="w-full pl-12 pr-4 py-4 text-lg bg-white border border-slate-200 rounded-2xl shadow-sm
                       placeholder:text-slate-400
                       focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-400
                       transition-all duration-200"
            disabled={isLoading}
          />
          {isLoading && (
            <div className="absolute inset-y-0 right-0 flex items-center pr-4">
              <div className="w-5 h-5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
            </div>
          )}
        </div>

        {hasHistory && (
          <button
            type="button"
            onClick={onClear}
            className="px-4 py-4 text-sm font-medium text-slate-600 bg-white border border-slate-200 rounded-2xl shadow-sm
                       hover:bg-slate-50 hover:border-slate-300 transition-all duration-200 whitespace-nowrap"
          >
            New Search
          </button>
        )}
      </div>
    </form>
  );
}
