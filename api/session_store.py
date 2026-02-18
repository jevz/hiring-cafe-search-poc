"""In-memory session store for multi-turn conversation history."""

import time
import uuid
from dataclasses import dataclass, field

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes


@dataclass
class Session:
    id: str
    history: list[str] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str | None) -> Session:
        self._cleanup_expired()

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_accessed = time.time()
            return session

        new_id = session_id or str(uuid.uuid4())
        session = Session(id=new_id)
        self._sessions[new_id] = session
        return session

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_accessed > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]


store = SessionStore()
