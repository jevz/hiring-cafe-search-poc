import { useCallback, useRef, useState } from "react";
import { apiClient } from "../api/client";
import type { ChatMessage, JobResult, ParsedIntent, SearchMeta } from "../types";

interface SearchState {
  results: JobResult[];
  meta: SearchMeta | null;
  intent: ParsedIntent | null;
  messages: ChatMessage[];
  sessionId: string;
  isLoading: boolean;
  error: string | null;
  hasSearched: boolean;
}

function newSessionId(): string {
  return crypto.randomUUID();
}

export function useSearch() {
  const [state, setState] = useState<SearchState>({
    results: [],
    meta: null,
    intent: null,
    messages: [],
    sessionId: newSessionId(),
    isLoading: false,
    error: null,
    hasSearched: false,
  });

  const sessionIdRef = useRef(state.sessionId);

  const search = useCallback(async (query: string) => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    // Add user message immediately
    const userMessage: ChatMessage = { role: "user", content: query };
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, userMessage],
    }));

    try {
      const response = await apiClient.search(query, sessionIdRef.current);

      const filterCount = Object.keys(response.intent.filters).length;
      const systemMessage: ChatMessage = {
        role: "system",
        content: `Found ${response.meta.matched_filters.toLocaleString()} matching jobs for "${response.intent.semantic_query}"${
          filterCount > 0
            ? ` with ${filterCount} filter${filterCount > 1 ? "s" : ""} active`
            : ""
        }, showing top ${response.results.length}. Searched ${response.meta.total_jobs.toLocaleString()} jobs in ${response.meta.search_time_ms}ms.`,
        meta: {
          resultCount: response.results.length,
          filters: response.intent.filters,
          semanticQuery: response.intent.semantic_query,
        },
      };

      setState((prev) => ({
        ...prev,
        results: response.results,
        meta: response.meta,
        intent: response.intent,
        messages: [...prev.messages, systemMessage],
        sessionId: response.session_id,
        isLoading: false,
        hasSearched: true,
      }));

      sessionIdRef.current = response.session_id;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Search failed";
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, []);

  const clearSession = useCallback(async () => {
    await apiClient.clearSession(sessionIdRef.current).catch(() => {});
    const newId = newSessionId();
    sessionIdRef.current = newId;
    setState({
      results: [],
      meta: null,
      intent: null,
      messages: [],
      sessionId: newId,
      isLoading: false,
      error: null,
      hasSearched: false,
    });
  }, []);

  return { ...state, search, clearSession };
}
