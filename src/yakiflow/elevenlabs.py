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
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import Settings
from .database import JobDatabase
from .models import Cue, TranscriptEvent, Word
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

# The batch API accepts files up to 5 GB, but 16 kHz mono PCM reaches that
# only after far more audio than a session should hold; 4.5 hours is the
# documented duration ceiling.
MAX_UPLOAD_SECONDS = 16200.0

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


def _keyring_key() -> str | None:
    try:
        import keyring
    except ImportError:
        return None
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_ENTRY) or None
    except Exception:
        # An unavailable Secret Service backend means this layer has no key,
        # the same as when the optional dependency is not installed at all.
        return None


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
            f"preview-{position}",
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
        estimate = max(30.0, (duration or 600.0) * 0.1)
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
