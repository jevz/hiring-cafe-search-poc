import { ChatPanel } from "./components/ChatPanel";
import { FilterBar } from "./components/FilterBar";
import { ResultsList } from "./components/ResultsList";
import { SearchBar } from "./components/SearchBar";
import { SearchMeta } from "./components/SearchMeta";
import { useSearch } from "./hooks/useSearch";

export default function App() {
  const {
    results,
    meta,
    intent,
    messages,
    isLoading,
    error,
    hasSearched,
    search,
    clearSession,
  } = useSearch();

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-indigo-600">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-900">HiringCafe Search</h1>
              <p className="text-xs text-slate-400">Semantic job search over 71k US jobs</p>
            </div>
          </div>

          <SearchBar
            onSearch={search}
            onClear={clearSession}
            isLoading={isLoading}
            hasHistory={hasSearched}
          />

          {/* Filters */}
          <div className="mt-3">
            <FilterBar intent={intent} />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-6">
        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
            {error}
          </div>
        )}

        {!hasSearched && !isLoading && (
          <div className="flex items-center justify-center py-24">
            <div className="text-center max-w-md">
              <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-indigo-50 flex items-center justify-center">
                <svg className="w-10 h-10 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-slate-800 mb-2">
                Search 71,284 jobs with AI
              </h2>
              <p className="text-sm text-slate-500 leading-relaxed">
                Type a natural language query like "remote python engineer paying over 150k"
                and watch the LLM parse your intent, extract filters, and find semantically
                matching jobs.
              </p>
              <div className="mt-6 flex flex-wrap justify-center gap-2">
                {["Semantic search", "Multi-turn refinement", "LLM intent parsing", "3 embedding spaces"].map((tag) => (
                  <span key={tag} className="px-3 py-1 bg-indigo-50 text-indigo-600 rounded-full text-xs font-medium">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {(hasSearched || isLoading) && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Results (2/3) */}
            <div className="lg:col-span-2 space-y-4">
              <ResultsList results={results} isLoading={isLoading} />
            </div>

            {/* Chat panel (1/3) */}
            <div className="lg:col-span-1">
              <div className="sticky top-[180px] h-[calc(100vh-270px)]">
                <ChatPanel
                  messages={messages}
                  onRefine={search}
                  isLoading={isLoading}
                  hasSearched={hasSearched}
                />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer meta */}
      {meta && (
        <footer className="sticky bottom-0 z-20 px-4 sm:px-6 pb-4 pt-2 bg-slate-50">
          <div className="max-w-7xl mx-auto">
            <SearchMeta meta={meta} resultCount={results.length} />
          </div>
        </footer>
      )}
    </div>
  );
}
