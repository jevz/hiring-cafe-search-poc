import type { SearchResponse } from "../types";

const BASE_URL = "/api";

export const apiClient = {
  async search(query: string, sessionId: string): Promise<SearchResponse> {
    const res = await fetch(`${BASE_URL}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: sessionId }),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Search failed: ${res.status} ${text}`);
    }
    return res.json();
  },

  async clearSession(sessionId: string): Promise<void> {
    await fetch(`${BASE_URL}/session/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
  },

  async health(): Promise<{ status: string; jobs_loaded: number }> {
    const res = await fetch(`${BASE_URL}/health`);
    return res.json();
  },
};
