export interface JobResult {
  rank: number;
  score: number;
  id: string;
  title: string | null;
  company_name: string | null;
  location: string | null;
  remote_type: string | null;
  seniority_level: string | null;
  employment_type: string | null;
  salary_display: string | null;
  salary_min: number | null;
  salary_max: number | null;
  required_skills: string[];
  industries: string[];
  apply_url: string | null;
  company_type: string | null;
}

export interface ParsedIntent {
  semantic_query: string;
  filters: Record<string, string | number | string[]>;
  weights: { explicit: number; inferred: number; company: number };
  exclusions: string[];
}

export interface SearchMeta {
  total_jobs: number;
  matched_filters: number;
  search_time_ms: number;
  intent_time_ms: number;
  embed_time_ms: number;
}

export interface SearchResponse {
  results: JobResult[];
  meta: SearchMeta;
  intent: ParsedIntent;
  conversation_history: string[];
  session_id: string;
}

export interface ChatMessage {
  role: "user" | "system";
  content: string;
  meta?: {
    resultCount: number;
    filters: Record<string, string | number | string[]>;
    semanticQuery: string;
  };
}
