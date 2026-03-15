from __future__ import annotations

import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.ai.schemas import BusinessState
from app.digital_twin.models import Base, Client, Event, StateSnapshot
from app.digital_twin.state_store import StateStore


class SQLiteStateStore(StateStore):
    def __init__(self, *, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(db_url, future=True)
        Base.metadata.create_all(self.engine)

    def ensure_client(self, client_id: str) -> None:
        with Session(self.engine) as s:
            existing = s.scalar(select(Client).where(Client.client_id == client_id))
            if existing is None:
                s.add(Client(client_id=client_id))
                s.commit()

    def _get_client_fk(self, s: Session, client_id: str) -> int:
        client = s.scalar(select(Client).where(Client.client_id == client_id))
        if client is None:
            client = Client(client_id=client_id)
            s.add(client)
            s.commit()
            s.refresh(client)
        return client.id

    def append_event(self, client_id: str, *, event_type: str, payload: dict) -> None:
        with Session(self.engine) as s:
            client_fk = self._get_client_fk(s, client_id)
            s.add(Event(client_fk=client_fk, event_type=event_type, payload_json=json.dumps(payload, ensure_ascii=False)))
            s.commit()

    def get_latest_state(self, client_id: str) -> BusinessState | None:
        with Session(self.engine) as s:
            client = s.scalar(select(Client).where(Client.client_id == client_id))
            if client is None:
                return None
            snap = s.scalar(
                select(StateSnapshot)
                .where(StateSnapshot.client_fk == client.id)
                .order_by(StateSnapshot.created_at.desc())
            )
            if snap is None:
                return None
            return BusinessState(**json.loads(snap.state_json))

    def set_latest_state(self, client_id: str, state: BusinessState) -> None:
        with Session(self.engine) as s:
            client_fk = self._get_client_fk(s, client_id)
            s.add(StateSnapshot(client_fk=client_fk, state_json=state.model_dump_json()))
            s.commit()

    def get_recent_events(self, client_id: str, *, limit: int = 200) -> list[dict]:
        with Session(self.engine) as s:
            client = s.scalar(select(Client).where(Client.client_id == client_id))
            if client is None:
                return []
            rows = s.scalars(
                select(Event)
                .where(Event.client_fk == client.id)
                .order_by(Event.created_at.desc())
                .limit(limit)
            ).all()
            out: list[dict] = []
            for r in rows:
                out.append(
                    {
                        "event_type": r.event_type,
                        "created_at": r.created_at.isoformat(),
                        "payload": json.loads(r.payload_json),
                    }
                )
            return out

