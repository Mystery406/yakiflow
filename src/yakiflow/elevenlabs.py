"""ElevenLabs speech-to-text support.

This module hosts everything specific to the ElevenLabs backends: API-key
resolution, word-stream normalization, the mechanical preview segmentation,
word-batch splitting for the draft agent, and the transcribers themselves.

The authoritative output of these backends is word-level. Final cues are cut
by the draft agent (see ``TranslationPipeline.segment_and_translate``); the
mechanical ``cues_from_words`` segmentation exists only so previews have
something to show before the agent has spoken.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import Settings
from .database import JobDatabase
from .models import Cue, TranscriptEvent, Word, preview_cue_id
from .process import CommandRunner
from .transcription import (
    ElapsedProgressTicker,
    EventCallback,
    ProgressCallback,
    Transcriber,
    normalize_source_language,
)


KEYRING_SERVICE = "yakiflow"
KEYRING_ENTRY = "elevenlabs"
API_KEY_ENV_VAR = "ELEVENLABS_API_KEY"

# A cut between two cues (mechanical or agent-made) is natural where the whole
# conversation is silent at least this long. Shared by the preview segmenter,
# the batch splitter, and the junction-repair heuristic so all three agree on
# what a legitimate boundary looks like.
PAUSE_SPLIT_SECONDS = 0.8
SENTENCE_END_CHARS = ".?!。？！…"
_CLOSING_QUOTES = "\"'”’»›』」）)]"

# The documented ceilings are 10 hours of audio in a file of at most 3 GB.
# Duration is the binding one here: 10 hours of 16 kHz mono PCM is ~1.2 GB.
MAX_UPLOAD_SECONDS = 36000.0

_NO_KEY_ERROR = (
    "no ElevenLabs API key is configured; provide one via "
    "-c elevenlabs.api-key=…, the ELEVENLABS_API_KEY environment variable, "
    "elevenlabs.api-key / api-key-file / api-key-command in a configuration "
    "file, or 'yakiflow secret set elevenlabs' (needs yakiflow[keyring])"
)


def _configured_slot(settings: Settings) -> str | None:
    """Name which spelling of the config API-key slot is set, if any."""
    elevenlabs = settings.elevenlabs
    if elevenlabs.api_key is not None:
        return "api_key"
    if elevenlabs.api_key_file is not None:
        return "api_key_file"
    if elevenlabs.api_key_command is not None:
        return "api_key_command"
    return None


def _slot_key(settings: Settings) -> str | None:
    """Resolve the configured slot to its key.

    A configured source that fails to produce a key is an error, never a
    silent fall-through to a lower-priority source: the key that would be used
    instead is not the one the user pointed at.
    """
    elevenlabs = settings.elevenlabs
    slot = _configured_slot(settings)
    if slot is None:
        return None
    if slot == "api_key":
        key = (elevenlabs.api_key or "").strip()
        if not key:
            raise ValueError("elevenlabs.api-key is empty")
        return key
    if slot == "api_key_file":
        assert elevenlabs.api_key_file is not None
        try:
            text = elevenlabs.api_key_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"cannot read elevenlabs.api-key-file {elevenlabs.api_key_file}: {exc}"
            ) from exc
        key = text.rstrip()
        if not key:
            raise ValueError(
                f"elevenlabs.api-key-file {elevenlabs.api_key_file} is empty"
            )
        return key
    assert elevenlabs.api_key_command is not None
    result = subprocess.run(
        elevenlabs.api_key_command,
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(
            "elevenlabs.api-key-command exited with status "
            f"{result.returncode}{suffix}"
        )
    key = result.stdout.strip()
    if not key:
        raise ValueError("elevenlabs.api-key-command printed no API key")
    return key


_keyring_cached_key: str | None = None


def _keyring_key() -> str | None:
    """Fetch the keyring key at most once per process.

    A key that resolved when the job started must keep resolving for the rest
    of the run: a Secret Service hiccup hours in would otherwise resurface as
    "no API key is configured" and fail work the key already authorized.
    """
    global _keyring_cached_key
    if _keyring_cached_key is not None:
        return _keyring_cached_key
    try:
        import keyring
    except ImportError:
        return None
    try:
        key = keyring.get_password(KEYRING_SERVICE, KEYRING_ENTRY) or None
    except Exception:
        # An unavailable Secret Service backend means this layer has no key,
        # the same as when the optional dependency is not installed at all.
        key = None
    _keyring_cached_key = key
    return key


def elevenlabs_api_key(settings: Settings) -> str:
    """Resolve the API key at time of use; never stored in ``Settings``.

    Priority, highest first: the config slot when the command line set it, the
    environment variable, the config slot from configuration files, the OS
    keyring.
    """
    from_cli = settings.elevenlabs.api_key_from_cli
    if from_cli:
        key = _slot_key(settings)
        if key:
            return key
    env = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if env:
        return env
    if not from_cli:
        key = _slot_key(settings)
        if key:
            return key
    key = _keyring_key()
    if key:
        return key
    raise ValueError(_NO_KEY_ERROR)


def elevenlabs_api_key_source(settings: Settings) -> str:
    """Name where the key would come from, without touching the key itself."""
    slot_labels = {
        "api_key": "config",
        "api_key_file": "file",
        "api_key_command": "command",
    }
    slot = _configured_slot(settings)
    if slot is not None and settings.elevenlabs.api_key_from_cli:
        return slot_labels[slot]
    if os.environ.get(API_KEY_ENV_VAR, "").strip():
        return "env"
    if slot is not None:
        return slot_labels[slot]
    if _keyring_key():
        return "keyring"
    return "not set"


def normalize_speaker(speaker_id: object) -> str | None:
    """Reduce the API's ``speaker_N`` labels to their bare number.

    The numbering carries no meaning beyond identity, so the shortest stable
    spelling is used everywhere: database, ``Cue.speaker``, the ASS Name
    field, and the draft agent's input.
    """
    if speaker_id is None:
        return None
    label = str(speaker_id).strip()
    if not label:
        return None
    if label.startswith("speaker_"):
        return label[len("speaker_"):]
    return label


def _get(item: Any, key: str, default: Any = None) -> Any:
    """Read one field off an SDK typed object or a plain response dict."""
    if isinstance(item, dict):
        return item.get(key, default)
    value = getattr(item, key, default)
    return default if value is None else value


def words_from_response(items: Sequence[Any], *, offset: float = 0.0) -> list[Word]:
    """Normalize an API word list: fold spacing, drop audio events.

    Spacing entries become part of the preceding word's text, so joining the
    word texts reconstructs the transcript verbatim; a leading spacing with no
    word before it is dropped.
    """
    words: list[Word] = []
    for item in items:
        kind = str(_get(item, "type", "word"))
        if kind == "audio_event":
            continue
        text = str(_get(item, "text", ""))
        if kind == "spacing":
            if words:
                words[-1].text += text
            continue
        start = float(_get(item, "start", 0.0)) + offset
        end = max(start, float(_get(item, "end", start)) + offset)
        logprob = _get(item, "logprob")
        words.append(Word(
            ordinal=len(words),
            start=start,
            end=end,
            text=text,
            speaker=normalize_speaker(_get(item, "speaker_id")),
            logprob=float(logprob) if logprob is not None else None,
        ))
    return words


def qualified_silences(words: Sequence[Word]) -> list[tuple[int, float]]:
    """``(index, gap)`` pairs where a cut after ``words[index]`` is legal.

    A silence qualifies only when the whole conversation is quiet: the gap is
    measured from the furthest end of any word so far, so a speaker whose word
    spans the pause disqualifies it even when the next word starts late.
    """
    silences: list[tuple[int, float]] = []
    max_end = float("-inf")
    for index in range(len(words) - 1):
        max_end = max(max_end, words[index].end)
        gap = words[index + 1].start - max_end
        if gap >= PAUSE_SPLIT_SECONDS:
            silences.append((index, gap))
    return silences


@dataclass(frozen=True, slots=True)
class WordBatch:
    """One contiguous slice of the time-ordered word stream for one agent call."""

    words: tuple[Word, ...]
    # A forced end means no qualified silence existed anywhere in the search
    # window, so the boundary cuts through continuous speech and must be
    # revisited by junction repair.
    forced_end: bool = False


def split_word_batches(
    words: Sequence[Word], target_size: int, hard_limit: int | None = None
) -> list[WordBatch]:
    """Slice the full-track word stream at qualified silences.

    Words of everyone speaking at once must share a batch — the agent cannot
    resolve half of a conversation — so cuts land only on silences no word
    spans. The cut nearest ``target_size`` with the largest gap wins; when the
    window up to the target holds none, the search widens to ``hard_limit``
    (double the target) and takes the first qualified silence; only truly
    continuous speech forces a cut, marked for junction repair.
    """
    hard_limit = hard_limit or 2 * target_size
    silences = dict(qualified_silences(words))
    batches: list[WordBatch] = []
    start = 0
    total = len(words)
    while start < total:
        if total - start <= hard_limit:
            batches.append(WordBatch(tuple(words[start:total])))
            break
        window_low = start + max(0, target_size // 2 - 1)
        window_high = start + target_size - 1
        best: int | None = None
        for index in range(window_low, window_high + 1):
            if index in silences and (best is None or silences[index] > silences[best]):
                best = index
        if best is None:
            for index in range(window_high + 1, start + hard_limit - 1):
                if index in silences:
                    best = index
                    break
        if best is not None:
            batches.append(WordBatch(tuple(words[start:best + 1])))
            start = best + 1
        else:
            cut = start + hard_limit
            batches.append(WordBatch(tuple(words[start:cut]), forced_end=True))
            start = cut
    return batches


def dispatchable_word_count(words: Sequence[Word]) -> int:
    """How many leading words may be dispatched while the stream still runs.

    Everything after the last qualified silence is held back so a dispatched
    batch always ends on a real pause; the held-back tail joins a batch once
    more audio arrives or the stream ends.
    """
    silences = qualified_silences(words)
    return silences[-1][0] + 1 if silences else 0


def cues_from_words(
    words: Sequence[Word],
    *,
    max_cue_seconds: float,
    max_cue_chars: int,
) -> list[Cue]:
    """Mechanically segment words into provisional preview cues.

    Preview only: these cues stand in until the draft agent has segmented and
    translated the words, and never reach the final output. Each speaker is
    segmented on their own track, so the result legitimately contains
    overlapping cues when people talk over each other.
    """
    tracks: dict[str | None, list[Word]] = {}
    for word in words:
        tracks.setdefault(word.speaker, []).append(word)
    collected: list[tuple[float, float, str | None, str, list[int]]] = []
    for speaker, track in tracks.items():
        current: list[Word] = []

        def flush() -> None:
            if not current:
                return
            source = "".join(word.text for word in current).strip()
            if source:
                collected.append((
                    current[0].start,
                    max(current[0].start, max(word.end for word in current)),
                    speaker,
                    source,
                    [current[0].ordinal, current[-1].ordinal],
                ))
            current.clear()

        for word in track:
            if current:
                gap = word.start - current[-1].end
                previous = current[-1].text.rstrip().rstrip(_CLOSING_QUOTES)
                combined = "".join(w.text for w in current) + word.text
                if (
                    gap >= PAUSE_SPLIT_SECONDS
                    or (previous and previous[-1] in SENTENCE_END_CHARS)
                    or word.end - current[0].start > max_cue_seconds
                    or len(combined.strip()) > max_cue_chars
                ):
                    flush()
            current.append(word)
        flush()
    collected.sort(key=lambda item: (item[0], item[1], item[2] or ""))
    return [
        Cue(
            preview_cue_id(position),
            start,
            end,
            source,
            metadata={"word_range": word_range},
            speaker=speaker,
        )
        for position, (start, end, speaker, source, word_range) in enumerate(
            collected, 1
        )
    ]


def _error_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return int(status) if isinstance(status, int) else None


# --- batch request wall-clock estimate ---

# ElevenLabs splits a batch request into internally parallel segments:
# ``min(4, ceil(duration / 480))``, so wall clock stops shrinking once the
# audio is long enough to saturate all four.
_SEGMENT_SECONDS = 480.0
_MAX_SEGMENTS = 4

# Published measurements put Scribe v2 near 55x real time on a ten-minute
# file, which the split above runs as two segments; that is ~27x per segment,
# rounded down here for headroom against a busy queue.
_SEGMENT_SPEED = 20.0

# The upload is the other half of the wall clock: the audio handed to this
# backend is 16 kHz mono PCM, so an hour of it is ~115 MB. Assume a modest
# uplink rather than the speed of a well-connected server.
_UPLOAD_BYTES_PER_SECOND = 2_000_000.0

# Request setup plus the queue wait before a segment starts.
_REQUEST_OVERHEAD_SECONDS = 8.0


def estimate_convert_seconds(duration: float | None, size_bytes: int) -> float:
    """Guess how long one whole-file request takes, for synthetic progress.

    The endpoint reports nothing until it answers, so the progress bar rides
    this estimate; it only needs the right order of magnitude.
    """
    audio_seconds = duration if duration else 600.0
    segments = min(
        _MAX_SEGMENTS, max(1, ceil(audio_seconds / _SEGMENT_SECONDS))
    )
    transcribe_seconds = audio_seconds / (_SEGMENT_SPEED * segments)
    upload_seconds = max(0, size_bytes) / _UPLOAD_BYTES_PER_SECOND
    return max(
        15.0, _REQUEST_OVERHEAD_SECONDS + upload_seconds + transcribe_seconds
    )


class ElevenLabsTranscriber(Transcriber):
    """Batch Scribe transcription through the official SDK."""

    name = "elevenlabs"
    _RETRY_ATTEMPTS = 3

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        db: JobDatabase,
        runner: CommandRunner | None = None,
        client_factory: Callable[[], Any] | None = None,
    ):
        super().__init__()
        self.settings = settings
        self.work_dir = work_dir
        self.db = db
        self.runner = runner
        self._client_factory = client_factory
        self._client: Any = None

    def _make_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                try:
                    from elevenlabs.client import AsyncElevenLabs
                except ImportError as exc:
                    raise RuntimeError(
                        "the elevenlabs SDK is not installed; install "
                        "yakiflow[elevenlabs] to use the ElevenLabs backends"
                    ) from exc
                self._client = AsyncElevenLabs(
                    api_key=elevenlabs_api_key(self.settings)
                )
        return self._client

    def _convert_kwargs(self, *, diarize: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model_id": self.settings.elevenlabs.model,
            "diarize": diarize,
            "tag_audio_events": False,
        }
        language = normalize_source_language(self.settings.source_language)
        if language is not None:
            kwargs["language_code"] = self.settings.source_language
        if diarize and self.settings.elevenlabs.num_speakers is not None:
            kwargs["num_speakers"] = self.settings.elevenlabs.num_speakers
        if diarize and self.settings.elevenlabs.use_speaker_library:
            kwargs["use_speaker_library"] = True
        return kwargs

    async def _convert(self, audio: Path, *, diarize: bool) -> Any:
        client = self._make_client()
        delay = 5.0
        for attempt in range(1, self._RETRY_ATTEMPTS + 1):
            try:
                with audio.open("rb") as fh:
                    return await client.speech_to_text.convert(
                        file=fh, **self._convert_kwargs(diarize=diarize)
                    )
            except Exception as exc:
                status = _error_status(exc)
                if status in {401, 403}:
                    raise RuntimeError(
                        "ElevenLabs rejected the API key (HTTP "
                        f"{status}); check the configured key"
                    ) from exc
                if status == 413:
                    raise RuntimeError(
                        "ElevenLabs rejected the upload as too large "
                        "(HTTP 413)"
                    ) from exc
                retryable = status == 429 or (status is not None and status >= 500)
                if not retryable or attempt == self._RETRY_ATTEMPTS:
                    if status is not None:
                        body = getattr(exc, "body", None)
                        summary = f": {str(body)[:300]}" if body else ""
                        raise RuntimeError(
                            f"ElevenLabs request failed with HTTP {status}{summary}"
                        ) from exc
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable ElevenLabs retry state")

    async def transcribe(
        self,
        audio: Path,
        on_event: EventCallback | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        resume_from: Sequence[Cue] = (),
    ) -> list[Cue]:
        # ``resume_from`` is deliberately ignored: the whole-file request is
        # idempotent, and already-finished agent batches are skipped by their
        # word_range coverage rather than by a timeline offset.
        from .media import pcm_audio_duration

        duration = pcm_audio_duration(audio)
        if duration is not None and duration > MAX_UPLOAD_SECONDS:
            raise ValueError(
                f"audio runs {duration / 3600:.2f} h, above the ElevenLabs "
                f"limit of {MAX_UPLOAD_SECONDS / 3600:.1f} h; split the input "
                "or use a Whisper backend"
            )
        estimate = estimate_convert_seconds(duration, audio.stat().st_size)
        async with ElapsedProgressTicker(estimate, on_progress):
            response = await self._convert(
                audio, diarize=self.settings.elevenlabs.diarize
            )
        language = normalize_source_language(_get(response, "language_code"))
        if language:
            self.db.checkpoint("detected_source_language", language)
        words = words_from_response(_get(response, "words", []) or [])
        self.db.replace_transcript_words(words)
        preview = cues_from_words(
            words,
            max_cue_seconds=self.settings.subtitles.max_cue_seconds,
            max_cue_chars=self.settings.subtitles.max_cue_chars,
        )
        for cue in preview:
            if on_event:
                await on_event(TranscriptEvent(cue, final=True))
        return preview

    async def submit_chunk(self, wav: Path, offset: float) -> list[Cue]:
        # Diarization is forced off for chunks: speaker numbering is not
        # stable between independent requests, and the preview is provisional
        # anyway.
        response = await self._convert(wav, diarize=False)
        words = words_from_response(
            _get(response, "words", []) or [], offset=offset
        )
        incoming = cues_from_words(
            words,
            max_cue_seconds=self.settings.subtitles.max_cue_seconds,
            max_cue_chars=self.settings.subtitles.max_cue_chars,
        )
        return self._chunk_timeline.add(incoming)


# --- realtime (Scribe v2 Realtime WebSocket) ---

# How far before the last durable word a broken realtime session resumes, and
# how far apart two hearings of the same word may start and still be one word.
RECONNECT_OVERLAP_SECONDS = 5.0
_WORD_MERGE_TOLERANCE = 0.6
_MAX_RECONNECTS = 3
# How much fed-but-uncommitted audio may be in flight before the feeder waits
# for the server to catch up.
_FLOW_CONTROL_WINDOW_SECONDS = 30.0
_EVENT_QUIET_TIMEOUT_SECONDS = 30.0


def merge_streamed_words(
    existing: Sequence[Word],
    incoming: Sequence[Word],
    tolerance: float = _WORD_MERGE_TOLERANCE,
) -> list[Word]:
    """Merge words from a re-fed overlap without duplicating any of them.

    A reconnect re-feeds a few seconds the previous session already
    transcribed; the same speech renders as words with nearly the same start
    and the same text, and those duplicates are dropped. The merge is
    strictly append-only: a differing hearing that would land *between*
    already-durable words loses to them, because word ordinals are the
    currency of dispatched agent batches and finished cue coverage — a
    renumbering would silently corrupt both.
    """
    result = list(existing)
    last_start = result[-1].start if result else float("-inf")
    for word in incoming:
        normalized = word.text.strip().casefold()
        duplicate = any(
            abs(candidate.start - word.start) <= tolerance
            and candidate.text.strip().casefold() == normalized
            for candidate in result[-80:]
        )
        if duplicate or word.start < last_start:
            continue
        word.ordinal = len(result)
        result.append(word)
        last_start = word.start
    return result


class _SessionEnded(Exception):
    """The realtime session ended before the audio did (limit or disconnect)."""


class _RealtimeSession:
    """Thin transport adapter over the SDK's realtime connection.

    Owns only open/send/commit/close and a queue of received events; every
    piece of timeline math lives in the transcriber, so an SDK interface
    change touches exactly this class. Should the pinned SDK lose its
    realtime client, an equivalent hand-written ``websockets`` client can
    stand in behind this same seam.
    """

    _ERROR_EVENTS = (
        "error", "auth_error", "quota_exceeded", "transcriber_error",
        "input_error", "invalid_request", "queue_overflow",
        "resource_exhausted", "chunk_size_exceeded", "rate_limited",
        "unaccepted_terms",
    )

    def __init__(self, settings: Settings):
        self.settings = settings
        self._connection: Any = None
        self._events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._first_send = True
        self._previous_text = ""

    async def open(self, previous_text: str = "") -> None:
        try:
            from elevenlabs.client import AsyncElevenLabs
            from elevenlabs.realtime import AudioFormat, CommitStrategy
        except ImportError as exc:
            raise RuntimeError(
                "the elevenlabs SDK is not installed; install "
                "yakiflow[elevenlabs] to use the ElevenLabs backends"
            ) from exc
        client = AsyncElevenLabs(api_key=elevenlabs_api_key(self.settings))
        options: dict[str, Any] = {
            "model_id": self.settings.elevenlabs.realtime_model,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": 16000,
            "commit_strategy": CommitStrategy.VAD,
            "include_timestamps": True,
        }
        if normalize_source_language(self.settings.source_language) is not None:
            options["language_code"] = self.settings.source_language
        self._connection = await client.speech_to_text.realtime.connect(options)
        self._previous_text = previous_text
        self._first_send = True

        def enqueue(kind: str):
            def handler(*args: Any) -> None:
                self._events.put_nowait((kind, args[0] if args else None))
            return handler

        self._connection.on(
            "committed_transcript_with_timestamps", enqueue("words")
        )
        self._connection.on(
            "session_time_limit_exceeded", enqueue("session_limit")
        )
        self._connection.on("close", enqueue("closed"))
        for event in self._ERROR_EVENTS:
            self._connection.on(event, enqueue("error"))

    async def send_pcm(self, pcm: bytes) -> None:
        import base64

        from websockets.exceptions import ConnectionClosed

        payload: dict[str, Any] = {
            "audio_base_64": base64.b64encode(pcm).decode("ascii"),
        }
        if self._first_send and self._previous_text:
            payload["previous_text"] = self._previous_text
        self._first_send = False
        try:
            await self._connection.send(payload)
        except ConnectionClosed as exc:
            raise _SessionEnded(f"connection closed: {exc}") from exc

    async def commit(self) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            await self._connection.commit()
        except ConnectionClosed as exc:
            raise _SessionEnded(f"connection closed: {exc}") from exc

    def pending_events(self) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except asyncio.QueueEmpty:
                return events

    async def next_event(self, timeout: float) -> tuple[str, Any] | None:
        try:
            return await asyncio.wait_for(self._events.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                pass
            self._connection = None


class ElevenLabsRealtimeTranscriber(Transcriber):
    """Realtime Scribe transcription over one (or few) WebSocket sessions.

    The only remote backend with true incremental persistence: committed
    words land in ``transcript_words`` as they arrive, so an interrupted
    authoritative pass resumes from the last durable word instead of zero.
    """

    name = "elevenlabs-stream"

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        db: JobDatabase,
        runner: CommandRunner | None = None,
        session_factory: Callable[[], Any] | None = None,
    ):
        super().__init__()
        self.settings = settings
        self.work_dir = work_dir
        self.db = db
        self.runner = runner
        self._session_factory = session_factory or (
            lambda: _RealtimeSession(settings)
        )
        self._session: Any = None
        self._words: list[Word] = []
        self._persist = False
        self._session_base = 0.0
        self._fed_end: float | None = None
        self._emitted_preview_ids: set[str] = set()

    # --- shared event handling ---

    def _handle_events(self, events: Sequence[tuple[str, Any]]) -> bool:
        """Fold received events into the word stream; True when words changed."""
        changed = False
        for kind, data in events:
            if kind == "words":
                incoming = words_from_response(
                    _get(data, "words", []) or [], offset=self._session_base
                )
                if not incoming:
                    continue
                merged = merge_streamed_words(self._words, incoming)
                if len(merged) != len(self._words):
                    fresh = merged[len(self._words):]
                    self._words = merged
                    changed = True
                    if self._persist:
                        # Append-only merges make the delta exactly the tail.
                        self.db.append_transcript_words(fresh)
            elif kind in {"session_limit", "closed"}:
                raise _SessionEnded(kind)
            elif kind == "error":
                raise RuntimeError(f"ElevenLabs realtime error: {data}")
        return changed

    def _preview(self) -> list[Cue]:
        return cues_from_words(
            self._words,
            max_cue_seconds=self.settings.subtitles.max_cue_seconds,
            max_cue_chars=self.settings.subtitles.max_cue_chars,
        )

    async def _absorb_session_end(self) -> None:
        """Close the ended session, but read what the server said first.

        The close often trails an explicit error event (auth, quota, terms);
        replaying the drained events turns that into the RuntimeError the
        caller sees instead of a bare disconnect.
        """
        session, self._session = self._session, None
        self._fed_end = None
        events = [
            event for event in session.pending_events()
            if event[0] not in {"session_limit", "closed"}
        ]
        await session.close()
        self._handle_events(events)

    # --- chunk previews: one continuous session, new bytes only ---

    async def start_chunks(self) -> None:
        # The server closes a session that sits idle while the download spins
        # up, so the socket opens lazily with the first chunk instead.
        return

    async def submit_chunk(self, wav: Path, offset: float) -> list[Cue]:
        import wave

        if self._session is None:
            self._session = self._session_factory()
            await self._session.open()
        with wave.open(str(wav), "rb") as reader:
            parameters = reader.getparams()
            frames = reader.readframes(parameters.nframes)
        bytes_per_second = (
            parameters.framerate * parameters.sampwidth * parameters.nchannels
        )
        duration = parameters.nframes / parameters.framerate
        if self._fed_end is None:
            self._session_base = offset
            self._fed_end = offset
        # The excerpt re-carries a few context seconds already sent to this
        # continuous session; only the bytes past the fed end are new.
        skip_seconds = max(0.0, self._fed_end - offset)
        skip_bytes = min(len(frames), round(skip_seconds * bytes_per_second))
        fresh = frames[skip_bytes:]
        try:
            if fresh:
                await self._session.send_pcm(fresh)
                self._fed_end = max(self._fed_end, offset + duration)
            self._handle_events(self._session.pending_events())
        except _SessionEnded:
            # A capped preview session simply reconnects on the next chunk;
            # the authoritative pass owns durable delivery.
            await self._absorb_session_end()
        preview = self._preview()
        self._chunk_timeline.cues = preview
        fresh_cues = [
            cue for cue in preview if cue.id not in self._emitted_preview_ids
        ]
        self._emitted_preview_ids.update(cue.id for cue in fresh_cues)
        return fresh_cues

    async def close_chunks(self) -> None:
        if self._session is None:
            return
        try:
            await self._session.commit()
        except Exception:
            pass
        await self._session.close()
        self._session = None

    # --- the authoritative pass ---

    async def transcribe(
        self,
        audio: Path,
        on_event: EventCallback | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        resume_from: Sequence[Cue] = (),
    ) -> list[Cue]:
        from .media import pcm_audio_duration

        self._persist = True
        duration = pcm_audio_duration(audio) or 0.0
        stored = self.db.list_transcript_words()
        if stored:
            self._words = stored
        reconnects = 0
        while True:
            if self._words:
                start_at = max(
                    0.0, self._words[-1].end - RECONNECT_OVERLAP_SECONDS
                )
                previous_text = "".join(
                    word.text for word in self._words[-60:]
                ).strip()[-500:]
            else:
                start_at = 0.0
                previous_text = ""
            self._session = self._session_factory()
            self._session_base = start_at
            await self._session.open(previous_text)
            try:
                await self._feed(audio, start_at, duration, on_event, on_progress)
                break
            except _SessionEnded:
                await self._absorb_session_end()
                reconnects += 1
                if reconnects > _MAX_RECONNECTS:
                    raise RuntimeError(
                        "the ElevenLabs realtime session ended "
                        f"{reconnects} times before the audio did; the words "
                        "delivered so far are preserved, resume to continue"
                    ) from None
            finally:
                if self._session is not None:
                    await self._session.close()
                    self._session = None
        preview = self._preview()
        for cue in preview:
            if on_event:
                await on_event(TranscriptEvent(cue, final=True))
        return preview

    async def _feed(
        self,
        audio: Path,
        start_at: float,
        duration: float,
        on_event: EventCallback | None,
        on_progress: ProgressCallback | None,
    ) -> None:
        import wave

        async def note_progress(fed: float) -> None:
            if on_progress and duration:
                await on_progress(min(0.95, fed / duration))

        async def emit_new_preview() -> None:
            if on_event is None:
                return
            preview = self._preview()
            fresh = [
                cue
                for cue in preview
                if cue.id not in self._emitted_preview_ids
            ]
            self._emitted_preview_ids.update(cue.id for cue in fresh)
            for cue in fresh:
                await on_event(TranscriptEvent(cue, final=False))

        with wave.open(str(audio), "rb") as reader:
            parameters = reader.getparams()
            bytes_per_second = (
                parameters.framerate * parameters.sampwidth * parameters.nchannels
            )
            reader.setpos(
                min(parameters.nframes, round(start_at * parameters.framerate))
            )
            fed = start_at
            acked_floor = start_at
            while True:
                frames = reader.readframes(parameters.framerate)
                if not frames:
                    break
                await self._session.send_pcm(frames)
                fed += len(frames) / bytes_per_second
                self._fed_end = fed
                if self._handle_events(self._session.pending_events()):
                    await emit_new_preview()
                # Flow control: never run more than the window ahead of what
                # the server has demonstrably consumed. Committed words are
                # the only explicit acknowledgement, but silence commits
                # nothing, so waited wall clock also counts as progress: a
                # realtime server consumes at least in real time.
                while (
                    fed
                    - max(
                        self._words[-1].end if self._words else start_at,
                        acked_floor,
                    )
                    > _FLOW_CONTROL_WINDOW_SECONDS
                ):
                    event = await self._session.next_event(10.0)
                    if event is None:
                        acked_floor += 10.0
                        continue
                    if self._handle_events([event]):
                        await emit_new_preview()
                await note_progress(fed)
        await self._session.commit()
        while True:
            event = await self._session.next_event(_EVENT_QUIET_TIMEOUT_SECONDS)
            if event is None:
                break
            if event[0] == "closed":
                break
            if event[0] == "session_limit":
                # The audio was already fully fed; whatever was committed is
                # in, so the session ending now is completion, not failure.
                break
            if self._handle_events([event]):
                await emit_new_preview()
        await note_progress(duration or self._fed_end or 0.0)
