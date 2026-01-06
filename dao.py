# dao.py
from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, create_engine, Session, select

class User(SQLModel, table=True):
    user_id: Optional[int] = Field(default=None, primary_key=True)
    telegram_id: int = Field(index=True, unique=True)
    display_name: Optional[str] = None
    target_language: str = "en"
    cefr_level: str = "B2"
    accent_pref: str = "international"
    daily_minutes: int = 45
    feedback_mode: str = "end_of_turn"
    timezone: str = "UTC"
    consent_flags: Optional[str] = None  # JSON string
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class SessionRec(SQLModel, table=True):
    session_id: str = Field(primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.user_id")
    started_at: str
    ended_at: Optional[str] = None
    turns: int = 0
    notes: Optional[str] = None

class MistakeEvent(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.user_id")
    timestamp: str
    modality: str = "speaking"
    target_language: str
    cefr_tag: Optional[str] = None
    category: str
    subcategory: Optional[str] = None
    source_prompt: Optional[str] = None
    user_attempt: Optional[str] = None
    correction: Optional[str] = None
    explanation: Optional[str] = None
    ipa: Optional[str] = None
    severity: Optional[int] = None
    confidence: Optional[float] = None
    times_seen: int = 0
    times_correct: int = 0
    last_seen: Optional[str] = None
    next_due: Optional[str] = None
    interval: Optional[int] = None
    easiness: Optional[float] = None
    repetitions: Optional[int] = None

def engine_from_env(url: str):
    return create_engine(url, echo=False, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})

def init_db(engine):
    SQLModel.metadata.create_all(engine)

def upsert_user(engine, telegram_id: int, defaults: dict) -> User:
    with Session(engine) as s:
        u = s.exec(select(User).where(User.telegram_id == telegram_id)).first()
        if not u:
            u = User(telegram_id=telegram_id, **defaults)
            s.add(u)
        else:
            for k,v in defaults.items():
                setattr(u, k, v)
            u.updated_at = datetime.now().isoformat()
        s.commit(); s.refresh(u)
        return u

def get_user_by_tid(engine, telegram_id: int) -> Optional[User]:
    with Session(engine) as s:
        return s.exec(select(User).where(User.telegram_id == telegram_id)).first()

def add_mistake(engine, evt: MistakeEvent):
    with Session(engine) as s:
        s.add(evt); s.commit()

def due_items(engine, user_id: int, now_iso: str) -> List[MistakeEvent]:
    with Session(engine) as s:
        return list(s.exec(select(MistakeEvent).where(
            MistakeEvent.user_id == user_id,
            MistakeEvent.next_due <= now_iso  # type: ignore
        )))