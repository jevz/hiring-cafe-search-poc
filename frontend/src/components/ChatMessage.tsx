import type { ChatMessage as ChatMessageType } from "../types";

interface Props {
  message: ChatMessageType;
}

export function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div
      className={`animate-slide-up flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[85%] px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed ${
          isUser
            ? "bg-indigo-600 text-white rounded-br-md"
            : "bg-slate-100 text-slate-700 rounded-bl-md"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}
