from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class OutputMode(StrEnum):
    SOURCE = "source"
    TRANSLATED = "translated"
    BILINGUAL = "bilingual"
    ALL = "all"


class JobStatus(StrEnum):
    CREATED = "created"
    ACQUIRING = "acquiring"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    ALIGNING = "aligning"
    REVIEWING = "reviewing"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(slots=True)
class Cue:
    id: str
    start: float
    end: float
    source: str
    translated: str | None = None
    timing_confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Kept after ``metadata``: several construction sites pass every field
    # positionally, and a new field ahead of ``metadata`` would silently bind
    # their metadata argument to it.
    speaker: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid cue timing: {self.start}..{self.end}")

    def with_timing(self, start: float, end: float, confidence: float | None = None) -> Cue:
        return replace(self, start=start, end=end, timing_confidence=confidence)

@dataclass(slots=True)
class TranscriptEvent:
    cue: Cue
    final: bool = False


@dataclass(slots=True)
class AgentTraceEvent:
    """A backend-neutral, user-visible event from one Agent operation."""

    operation_id: str
    kind: str
    message: str
    state: str = "running"
    cue_ids: tuple[str, ...] = ()
    model: str | None = None
    attempt: int | None = None
    max_attempts: int | None = None
    event_id: str | None = None
    detail: str | None = None
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class JobEvent:
    kind: str
    message: str
    progress: float | None = None
    cue: Cue | None = None
    created_at: str = field(default_factory=utc_now)
    stage: str | None = None
    estimated_remaining: int | None = None
    agent_trace: AgentTraceEvent | None = None
    error_traceback: str | None = None


def cue_id(ordinal: int) -> str:
    """Return a simple one-based sequence number for a generated cue."""
    return str(ordinal + 1)
