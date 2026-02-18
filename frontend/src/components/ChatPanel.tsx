import { useEffect, useRef, useState } from "react";
import type { ChatMessage as ChatMessageType } from "../types";
import { ChatMessage } from "./ChatMessage";

interface Props {
  messages: ChatMessageType[];
  onRefine: (query: string) => void;
  isLoading: boolean;
  hasSearched: boolean;
}

export function ChatPanel({ messages, onRefine, isLoading, hasSearched }: Props) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onRefine(input.trim());
      setInput("");
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-100 bg-slate-50/50">
        <h3 className="text-sm font-semibold text-slate-700">Refine Results</h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Narrow down with natural language
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 min-h-0">
        {!hasSearched && messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center px-4">
              <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-indigo-50 flex items-center justify-center">
                <svg className="w-6 h-6 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                  />
                </svg>
              </div>
              <p className="text-sm text-slate-500">
                Search for jobs above, then refine your results here.
              </p>
              <p className="text-xs text-slate-400 mt-1">
                Try "make it remote" or "paying over 150k"
              </p>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <ChatMessage key={i} message={msg} />
        ))}

        {isLoading && (
          <div className="animate-slide-up flex justify-start">
            <div className="bg-slate-100 text-slate-700 rounded-2xl rounded-bl-md px-4 py-3">
              <div className="flex items-center gap-1.5">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-slate-100">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={hasSearched ? "Refine: \"make it remote\", \"at startups\"..." : "Search first to start refining"}
            className="flex-1 px-3.5 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl
                       placeholder:text-slate-400
                       focus:outline-none focus:ring-2 focus:ring-indigo-500/30 focus:border-indigo-300
                       transition-all duration-200"
            disabled={isLoading || !hasSearched}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim() || !hasSearched}
            className="px-4 py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-xl
                       hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed
                       transition-all duration-200"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
