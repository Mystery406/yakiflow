import asyncio
import importlib.util
import wave
from collections import deque
from pathlib import Path

import pytest

from conftest import make_settings
from yakiflow.database import JobDatabase
from yakiflow.elevenlabs import (
    ElevenLabsRealtimeTranscriber,
    _RealtimeSession,
    merge_streamed_words,
)
from yakiflow.models import Word


def _settings():
    return make_settings(
        source_language="en",
        target_language="zh-CN",
        transcription={"backend": "elevenlabs-stream"},
        agent={"backend": "codex"},
    ).resolved()


def _write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * int(16000 * seconds))


def _word_payload(text: str, start: float, end: float) -> dict:
    return {"type": "word", "text": text, "start": start, "end": end}


class FakeSession:
    """Scripted stand-in for the SDK transport seam."""

    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.previous_texts: list[str] = []
        self.committed = False
        self.closed = False
        self._events: deque = deque()

    def push(self, kind: str, data=None) -> None:
        self._events.append((kind, data))

    async def open(self, previous_text: str = "") -> None:
        self.previous_texts.append(previous_text)

    async def send_pcm(self, pcm: bytes) -> None:
        self.sent.append(pcm)
        self.on_send(len(self.sent))

    def on_send(self, count: int) -> None:  # overridden per test
        pass

    async def commit(self) -> None:
        self.committed = True
        self.on_commit()

    def on_commit(self) -> None:  # overridden per test
        pass

    def pending_events(self):
        events = list(self._events)
        self._events.clear()
        return events

    async def next_event(self, timeout: float):
        if self._events:
            return self._events.popleft()
        return None

    async def close(self) -> None:
        self.closed = True


def test_merge_streamed_words_deduplicates_the_reheard_overlap() -> None:
    existing = [
        Word(ordinal=0, start=0.0, end=0.5, text="one "),
        Word(ordinal=1, start=0.6, end=1.0, text="two "),
    ]
    incoming = [
        # Re-heard with slightly shifted timing: a duplicate.
        Word(ordinal=0, start=0.65, end=1.05, text="two "),
        Word(ordinal=1, start=1.2, end=1.6, text="three"),
    ]
    merged = merge_streamed_words(existing, incoming)
    assert [word.text for word in merged] == ["one ", "two ", "three"]
    assert [word.ordinal for word in merged] == [0, 1, 2]


def test_merge_streamed_words_never_renumbers_durable_words() -> None:
    existing = [
        Word(ordinal=0, start=0.0, end=0.5, text="one "),
        Word(ordinal=1, start=2.0, end=2.5, text="three"),
    ]
    # A differing hearing that would land between the durable words loses:
    # ordinals are the currency of dispatched batches and cue coverage.
    incoming = [Word(ordinal=0, start=1.0, end=1.5, text="two ")]
    merged = merge_streamed_words(existing, incoming)
    assert [word.text for word in merged] == ["one ", "three"]


def test_transcribe_feeds_frames_and_persists_words_incrementally(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "reference.wav"
    _write_wav(audio, 3.0)
    session = FakeSession()

    def on_send(count: int) -> None:
        if count == 2:
            session.push("words", {"words": [
                _word_payload("hello ", 0.2, 0.6),
                _word_payload("there", 0.7, 1.1),
            ]})

    def on_commit() -> None:
        session.push("words", {"words": [_word_payload("end", 2.0, 2.4)]})
        session.push("closed")

    session.on_send = on_send
    session.on_commit = on_commit
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsRealtimeTranscriber(
        _settings(), tmp_path, db, session_factory=lambda: session
    )
    progress: list[float] = []
    events: list[tuple[bool, str]] = []

    async def on_event(item) -> None:
        events.append((item.final, item.cue.source))

    async def on_progress(fraction: float) -> None:
        progress.append(fraction)

    preview = asyncio.run(transcriber.transcribe(audio, on_event, on_progress))

    # One second of PCM per frame, whole file fed, then committed and closed.
    assert [len(chunk) for chunk in session.sent] == [32000, 32000, 32000]
    assert session.committed and session.closed
    stored = db.list_transcript_words()
    assert [word.text for word in stored] == ["hello ", "there", "end"]
    assert [word.ordinal for word in stored] == [0, 1, 2]
    # Preview cues split at the ≥0.8 s pause and reach the caller as finals.
    assert [cue.source for cue in preview] == ["hello there", "end"]
    assert (False, "hello there") in events
    assert (True, "end") in events
    assert progress and max(progress) <= 1.0
    db.close()


def test_session_limit_reconnects_with_context_and_deduplicates(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "reference.wav"
    _write_wav(audio, 4.0)
    first = FakeSession()
    second = FakeSession()
    sessions = deque([first, second])

    def first_send(count: int) -> None:
        if count == 2:
            first.push("words", {"words": [
                _word_payload("alpha ", 0.1, 0.5),
                _word_payload("beta", 0.6, 1.0),
            ]})
            first.push("session_limit")

    def second_send(count: int) -> None:
        if count == 1:
            # The overlap window re-hears "beta" (relative time 0.6, and the
            # session restarted at 1.0 - 5.0 clamped to 0.0 → absolute 0.6)
            # and adds a genuinely new word.
            second.push("words", {"words": [
                _word_payload("beta", 0.62, 1.02),
                _word_payload("gamma", 1.4, 1.9),
            ]})

    def second_commit() -> None:
        second.push("closed")

    first.on_send = first_send
    second.on_send = second_send
    second.on_commit = second_commit
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsRealtimeTranscriber(
        _settings(), tmp_path, db, session_factory=lambda: sessions.popleft()
    )

    asyncio.run(transcriber.transcribe(audio))

    assert first.closed
    # The reconnect carried the durable tail as context.
    assert second.previous_texts == ["alpha beta"]
    stored = db.list_transcript_words()
    assert [word.text for word in stored] == ["alpha ", "beta", "gamma"]
    db.close()


def test_repeated_session_failures_preserve_words_and_give_up(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "reference.wav"
    _write_wav(audio, 2.0)
    made: list[FakeSession] = []

    def factory() -> FakeSession:
        session = FakeSession()

        def on_send(count: int) -> None:
            if len(made) == 1 and count == 1:
                session.push("words", {"words": [_word_payload("kept", 0.1, 0.5)]})
            session.push("session_limit")

        session.on_send = on_send
        made.append(session)
        return session

    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsRealtimeTranscriber(
        _settings(), tmp_path, db, session_factory=factory
    )

    with pytest.raises(RuntimeError, match="resume to continue"):
        asyncio.run(transcriber.transcribe(audio))

    # 1 initial session + 3 reconnects, then give up; the words survive.
    assert len(made) == 4
    assert [word.text for word in db.list_transcript_words()] == ["kept"]
    db.close()


def test_chunks_share_one_session_and_send_only_new_bytes(tmp_path: Path) -> None:
    chunk1 = tmp_path / "chunk1.wav"
    chunk2 = tmp_path / "chunk2.wav"
    # First chunk covers [0, 2); the second re-carries one second of context
    # and covers [1, 4).
    _write_wav(chunk1, 2.0)
    _write_wav(chunk2, 3.0)
    session = FakeSession()

    def on_send(count: int) -> None:
        if count == 2:
            session.push("words", {"words": [_word_payload("late", 2.2, 2.6)]})

    session.on_send = on_send
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = ElevenLabsRealtimeTranscriber(
        _settings(), tmp_path, db, session_factory=lambda: session
    )

    async def exercise():
        first = await transcriber.submit_chunk(chunk1, 0.0)
        second = await transcriber.submit_chunk(chunk2, 1.0)
        return first, second

    first, second = asyncio.run(exercise())

    assert len(session.previous_texts) == 1
    # 2 s, then only the 2 s past the already-fed end (offset 1 + 3 − fed 2).
    assert [len(chunk) for chunk in session.sent] == [64000, 64000]
    assert first == []
    assert [cue.source for cue in second] == ["late"]
    assert [cue.source for cue in transcriber.chunk_cues] == ["late"]
    # Preview words never touch the durable word table.
    assert db.list_transcript_words() == []
    asyncio.run(transcriber.close_chunks())
    assert session.committed and session.closed
    db.close()


@pytest.mark.skipif(
    importlib.util.find_spec("elevenlabs") is not None,
    reason="the elevenlabs SDK is installed",
)
def test_missing_sdk_is_reported_with_the_extra_name() -> None:
    session = _RealtimeSession(_settings())
    with pytest.raises(RuntimeError, match=r"yakiflow\[elevenlabs\]"):
        asyncio.run(session.open())
