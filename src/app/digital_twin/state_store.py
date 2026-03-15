from __future__ import annotations

from abc import ABC, abstractmethod

from app.ai.schemas import BusinessState


class StateStore(ABC):
    """
    State store abstraction for per-client "Business Brain".
    Can be backed by Neo4j (KG) or a relational schema (baseline).
    """

    @abstractmethod
    def ensure_client(self, client_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_event(self, client_id: str, *, event_type: str, payload: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_latest_state(self, client_id: str) -> BusinessState | None:
        raise NotImplementedError

    @abstractmethod
    def set_latest_state(self, client_id: str, state: BusinessState) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_recent_events(self, client_id: str, *, limit: int = 200) -> list[dict]:
        raise NotImplementedError

