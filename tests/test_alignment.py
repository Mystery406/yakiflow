import asyncio
import math
import struct
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import yakiflow.alignment as alignment_module
from yakiflow.alignment import (
    AlignmentModelDecisionRequired,
    AlignmentResult,
    PcmVolumeStartRefiner,
    WhisperVadAlignmentBackend,
    WhisperXAlignmentBackend,
    adjust_cue_starts_for_long_vad_silences,
    extend_cue_ends,
)
from yakiflow.models import Cue


def _wav(path: Path) -> None:
    rate = 16000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        samples = []
        for i in range(rate * 3):
            t = i / rate
            value = (
                int(9000 * math.sin(2 * math.pi * 220 * t))
                if t < 0.8 or t >= 1.5
                else 0
            )
            samples.append(struct.pack("<h", value))
        wav.writeframes(b"".join(samples))


def _volume_wav(path: Path, amplitude_at, *, duration: float = 2.0) -> None:
    rate = 16_000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        samples = [
            struct.pack(
                "<h",
                round(
                    amplitude_at(index / rate)
                    * math.sin(2 * math.pi * 220 * index / rate)
                ),
            )
            for index in range(round(rate * duration))
        ]
        wav.writeframes(b"".join(samples))


class FakeVadBackend(WhisperVadAlignmentBackend):
    def __init__(self, timings: dict[str, tuple[float, float]] | None = None) -> None:
        self.calls = 0
        self.timings = timings or {}

    async def align(self, audio, cues, **kwargs):
        self.calls += 1
        output = [
            cue.with_timing(*self.timings.get(cue.id, (cue.start, cue.end)), 0.4)
            for cue in cues
        ]
        return AlignmentResult(output, "fake-vad")


class FakeWhisperX:
    PUNKT_LANGUAGES = {"en": "english"}

    def __init__(self, align_impl, *, duration: float = 12.0) -> None:
        self.align_impl = align_impl
        self.audio = [0.0] * round(duration * 16_000)
        self.load_audio_calls = 0
        self.load_model_calls = 0
        self.excerpt_durations: list[float] = []
        self.model_kwargs: dict = {}

    def load_audio(self, path: str):
        self.load_audio_calls += 1
        return self.audio

    def nltk_load(self, _resource: str):
        return object()

    def load_align_model(self, **kwargs):
        self.load_model_calls += 1
        self.model_kwargs = kwargs
        return object(), {"dictionary": "fake"}

    def align(self, segments, model, metadata, audio, device, **kwargs):
        self.excerpt_durations.append(len(audio) / 16_000)
        return self.align_impl(segments, len(self.excerpt_durations))


def test_whisperx_retries_with_exact_expanded_window_and_loads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def result(_segments, call):
        if call == 1:
            return {"segments": []}
        return {
            "segments": [{
                "text": "corrected source",
                "words": [{"word": "source", "start": 3.2, "end": 3.8, "score": 0.8}],
            }]
        }

    fake = FakeWhisperX(result)
    monkeypatch.setattr(alignment_module, "import_module", lambda name: fake)

    vad = FakeVadBackend()
    backend = WhisperXAlignmentBackend(
        language="en", device="cpu", model_name="custom-aligner", vad_backend=vad
    )
    progress: list[tuple[int, int]] = []

    aligned = asyncio.run(
        backend.align(
            tmp_path / "audio.wav",
            [Cue("old-7", 5, 6, "corrected source", "译文")],
            on_progress=lambda completed, total: progress.append((completed, total)),
        )
    )

    assert fake.excerpt_durations == pytest.approx([1.7, 5.0])
    assert fake.load_audio_calls == fake.load_model_calls == 1
    assert fake.model_kwargs == {
        "language_code": "en",
        "device": "cpu",
        "model_name": "custom-aligner",
    }
    assert vad.calls == 0
    assert progress == [(1, 1)]
    assert (aligned.cues[0].start, aligned.cues[0].translated) == (6.2, "译文")
    assert aligned.cues[0].metadata["parent_id"] == "old-7"
    assert aligned.cues[0].metadata["whisperx"]["window_start"] == 3.0
    assert aligned.cues[0].metadata["whisperx"]["window_end"] == 8.0
    assert aligned.cues[0].metadata["whisperx"]["attempts"] == [
        {
            "window_start": 4.8,
            "window_end": 6.5,
            "outcome": "failed",
        },
        {
            "window_start": 3.0,
            "window_end": 8.0,
            "outcome": "success",
        },
    ]


def test_whisperx_alignment_does_not_block_the_asyncio_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow_result(_segments, _call):
        time.sleep(0.15)
        return {
            "segments": [{
                "text": "source",
                "words": [{"word": "source", "start": 0.7, "end": 1.3}],
            }]
        }

    fake = FakeWhisperX(slow_result)
    monkeypatch.setattr(alignment_module, "import_module", lambda _name: fake)

    async def exercise() -> float:
        loop = asyncio.get_running_loop()
        started = loop.time()

        async def heartbeat() -> float:
            await asyncio.sleep(0.01)
            return loop.time() - started

        heartbeat_task = asyncio.create_task(heartbeat())
        await WhisperXAlignmentBackend(language="en", device="cpu").align(
            tmp_path / "audio.wav", [Cue("parent", 1, 2, "source")]
        )
        return await heartbeat_task

    assert asyncio.run(exercise()) < 0.08


def test_whisperx_model_failure_retries_only_after_explicit_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeWhisperX(
        lambda _segments, _call: {
            "segments": [{
                "text": "source",
                "words": [{"word": "source", "start": 0.7, "end": 1.3}],
            }]
        }
    )
    attempts = 0

    def load_align_model(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("model download timed out")
        return object(), {"dictionary": "fake"}

    fake.load_align_model = load_align_model
    monkeypatch.setattr(alignment_module, "import_module", lambda _name: fake)
    decisions: list[str] = []
    vad = FakeVadBackend()

    async def choose(message: str):
        decisions.append(message)
        return "retry"

    aligned = asyncio.run(
        WhisperXAlignmentBackend(
            language="en", device="cpu", vad_backend=vad
        ).align(
            tmp_path / "audio.wav",
            [Cue("parent", 1, 2, "source")],
            on_model_failure=choose,
        )
    )

    assert attempts == 2
    assert len(decisions) == 1
    assert "timed out" in decisions[0]
    assert aligned.backend == "whisperx"
    assert vad.calls == 0


def test_whisperx_model_failure_falls_back_only_after_explicit_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeWhisperX(lambda _segments, _call: {})

    def fail_model_load(**_kwargs):
        raise OSError("download failed")

    fake.load_align_model = fail_model_load
    monkeypatch.setattr(alignment_module, "import_module", lambda _name: fake)
    vad = FakeVadBackend()

    aligned = asyncio.run(
        WhisperXAlignmentBackend(
            language="en", device="cpu", vad_backend=vad
        ).align(
            tmp_path / "audio.wav",
            [Cue("parent", 1, 2, "source")],
            on_model_failure=lambda _message: "fallback",
        )
    )

    assert aligned.backend == "whisperx-vad-fallback"
    assert vad.calls == 1


def test_whisperx_model_failure_without_ui_does_not_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeWhisperX(lambda _segments, _call: {})

    def fail_model_load(**_kwargs):
        raise OSError("download failed")

    fake.load_align_model = fail_model_load
    monkeypatch.setattr(alignment_module, "import_module", lambda _name: fake)
    vad = FakeVadBackend()

    with pytest.raises(AlignmentModelDecisionRequired, match="interactive UI"):
        asyncio.run(
            WhisperXAlignmentBackend(
                language="en", device="cpu", vad_backend=vad
            ).align(
                tmp_path / "audio.wav",
                [Cue("parent", 1, 2, "source")],
            )
        )

    assert vad.calls == 0


def test_whisperx_missing_nltk_data_is_downloaded_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader_calls: list[str] = []
    installed = False

    def missing_loader(resource: str):
        loader_calls.append(resource)
        if not installed:
            raise LookupError("punkt_tab is not installed")
        return object()

    whisperx_alignment = SimpleNamespace(
        nltk_load=missing_loader,
        PUNKT_LANGUAGES={"en": "english"},
    )
    fake = FakeWhisperX(
        lambda _segments, _call: {
            "segments": [{
                "text": "works",
                "words": [{"word": "works", "start": 0.7, "end": 1.3}],
            }]
        }
    )

    def import_module(name: str):
        if name == "whisperx.alignment":
            return whisperx_alignment
        return fake

    monkeypatch.setattr(alignment_module, "import_module", import_module)
    backend = WhisperXAlignmentBackend(language="en", device="cpu")

    async def download_punkt():
        nonlocal installed
        installed = True
        return tmp_path / "nltk_data"

    monkeypatch.setattr(backend, "_download_nltk_punkt", download_punkt)

    aligned = asyncio.run(
        backend.align(
            tmp_path / "audio.wav",
            [Cue("parent", 1, 2, "works")],
        )
    )

    assert aligned.backend == "whisperx"
    assert loader_calls == [
        "tokenizers/punkt_tab/english.pickle",
        "tokenizers/punkt_tab/english.pickle",
    ]


def test_whisperx_nltk_download_failure_requires_retry_or_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = False

    def load_punkt(_resource: str):
        if not installed:
            raise LookupError("punkt_tab is not installed")
        return object()

    whisperx_alignment = SimpleNamespace(
        nltk_load=load_punkt,
        PUNKT_LANGUAGES={"en": "english"},
    )
    fake = FakeWhisperX(
        lambda _segments, _call: {
            "segments": [{
                "text": "works",
                "words": [{"word": "works", "start": 0.7, "end": 1.3}],
            }]
        }
    )

    def import_module(name: str):
        return whisperx_alignment if name == "whisperx.alignment" else fake

    monkeypatch.setattr(alignment_module, "import_module", import_module)
    backend = WhisperXAlignmentBackend(language="en", device="cpu")
    attempts = 0

    async def download_punkt():
        nonlocal attempts, installed
        attempts += 1
        if attempts == 1:
            raise TimeoutError("timed out after 60 seconds")
        installed = True
        return tmp_path / "nltk_data"

    monkeypatch.setattr(backend, "_download_nltk_punkt", download_punkt)
    messages: list[str] = []

    def choose(message: str):
        messages.append(message)
        return "retry"

    aligned = asyncio.run(
        backend.align(
            tmp_path / "audio.wav",
            [Cue("parent", 1, 2, "works")],
            on_model_failure=choose,
        )
    )

    assert attempts == 2
    assert len(messages) == 1
    assert "NLTK punkt_tab" in messages[0]
    assert aligned.backend == "whisperx"


def test_whisperx_splits_sentences_and_records_word_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeWhisperX(
        lambda _segments, _call: {
            "segments": [
                {
                    "text": "First sentence.",
                    "words": [
                        {"word": "First", "start": 0.7, "end": 1.0, "score": 0.8},
                        {"word": "sentence", "start": 1.0, "end": 1.4, "score": 0.6},
                    ],
                },
                {
                    "text": "Second sentence.",
                    "words": [
                        {"word": "Second", "start": 1.7, "end": 2.0, "score": 0.9},
                        {"word": "sentence", "start": 2.0, "end": 2.4, "score": 0.7},
                    ],
                },
            ]
        }
    )
    monkeypatch.setattr(alignment_module, "import_module", lambda name: fake)

    aligned = asyncio.run(
        WhisperXAlignmentBackend(language="en", device="cpu").align(
            tmp_path / "audio.wav",
            [Cue("parent", 1, 3, "Agent-corrected source", "existing translation")],
        )
    )

    assert [cue.id for cue in aligned.cues] == ["1", "2"]
    assert [cue.source for cue in aligned.cues] == [
        "First sentence.",
        "Second sentence.",
    ]
    assert [cue.translated for cue in aligned.cues] == [None, None]
    assert [cue.metadata["parent_id"] for cue in aligned.cues] == ["parent", "parent"]
    assert aligned.cues[0].metadata["whisperx"]["score"] == pytest.approx(0.7)
    assert aligned.cues[1].timing_confidence == pytest.approx(0.8)
    assert aligned.cues[0].metadata["whisperx"]["words"][0]["start"] == 1.5


def test_whisperx_aligns_japanese_source_and_preserves_unsplit_bilingual_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[str] = []

    def result(segments, _call):
        text = segments[0]["text"]
        received.append(text)
        return {
            "segments": [{
                "text": text,
                "words": [{
                    "word": text.removesuffix("。"),
                    "start": 0.7,
                    "end": 1.3,
                    "score": 0.9,
                }],
            }]
        }

    fake = FakeWhisperX(result)
    monkeypatch.setattr(alignment_module, "import_module", lambda _name: fake)
    original = Cue(
        "parent",
        1,
        2,
        "今日は晴れです。",
        "It is sunny today.",
    )

    aligned = asyncio.run(
        WhisperXAlignmentBackend(language="ja", device="cpu").align(
            tmp_path / "audio.wav", [original]
        )
    )

    assert received == ["今日は晴れです。"]
    assert (aligned.cues[0].source, aligned.cues[0].translated) == (
        "今日は晴れです。",
        "It is sunny today.",
    )


def test_whisperx_japanese_split_uses_source_fragments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[str] = []

    def result(segments, _call):
        received.append(segments[0]["text"])
        return {
            "segments": [
                {
                    "text": "今日は晴れ。",
                    "words": [{"word": "今日は晴れ", "start": 0.7, "end": 1.1}],
                },
                {
                    "text": "明日は雨。",
                    "words": [{"word": "明日は雨", "start": 1.3, "end": 1.7}],
                },
            ]
        }

    fake = FakeWhisperX(result)
    monkeypatch.setattr(alignment_module, "import_module", lambda _name: fake)
    parent = Cue(
        "parent",
        1,
        2,
        "今日は晴れ。明日は雨。",
        "Sunny today; rain tomorrow.",
    )

    aligned = asyncio.run(
        WhisperXAlignmentBackend(language="ja", device="cpu").align(
            tmp_path / "audio.wav", [parent]
        )
    )

    assert received == ["今日は晴れ。明日は雨。", "今日は晴れ。明日は雨。"]
    assert [cue.source for cue in aligned.cues] == ["今日は晴れ。", "明日は雨。"]
    assert [cue.translated for cue in aligned.cues] == [None, None]


def test_whisperx_only_falls_back_failed_parents_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def result(segments, _call):
        if segments[0]["text"] == "bad":
            return {"segments": []}
        return {
            "segments": [{
                "text": "good",
                "words": [{"word": "good", "start": 0.7, "end": 1.3, "score": 0.9}],
            }]
        }

    fake = FakeWhisperX(result)
    monkeypatch.setattr(alignment_module, "import_module", lambda name: fake)
    vad = FakeVadBackend({"good-parent": (0.1, 0.4), "bad-parent": (5.1, 6.1)})
    warnings: list[str] = []
    progress: list[tuple[int, int]] = []

    async def record_progress(completed: int, total: int) -> None:
        progress.append((completed, total))

    aligned = asyncio.run(
        WhisperXAlignmentBackend(
            language="en", device="cpu", vad_backend=vad
        ).align(
            tmp_path / "audio.wav",
            [
                Cue("good-parent", 1, 2, "good", "好"),
                Cue("bad-parent", 5, 6, "bad", "坏"),
            ],
            on_warning=lambda message: warnings.append(message),
            on_progress=record_progress,
        )
    )

    assert vad.calls == 1
    assert progress == [(1, 2), (2, 2)]
    assert len(fake.excerpt_durations) == 3
    assert aligned.cues[0].start == 1.5
    assert aligned.cues[0].metadata["alignment_backend"] == "whisperx"
    assert (aligned.cues[1].start, aligned.cues[1].end) == (5.1, 6.1)
    assert aligned.cues[1].metadata["alignment_backend"] == "vad-fallback"
    assert aligned.cues[1].metadata["whisperx"]["attempts"] == [
        {
            "window_start": 4.8,
            "window_end": 6.5,
            "outcome": "failed",
        },
        {
            "window_start": 3.0,
            "window_end": 8.0,
            "outcome": "failed",
        },
    ]
    assert aligned.low_confidence_ids == ["2"]
    assert "bad-parent" in warnings[0]


def test_whisperx_zero_first_token_score_retries_with_expanded_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def result(_segments, call):
        if call == 1:
            return {
                "segments": [{
                    "text": "source",
                    "words": [
                        {"word": "unmatched", "score": 0},
                        {"word": "source", "start": 0.7, "end": 1.3, "score": 0.9},
                    ],
                }]
            }
        return {
            "segments": [{
                "text": "source",
                "words": [{"word": "source", "start": 3.2, "end": 3.8, "score": 0.8}],
            }]
        }

    fake = FakeWhisperX(result)
    monkeypatch.setattr(alignment_module, "import_module", lambda _name: fake)
    vad = FakeVadBackend()
    warnings: list[str] = []

    aligned = asyncio.run(
        WhisperXAlignmentBackend(
            language="en", device="cpu", vad_backend=vad
        ).align(
            tmp_path / "audio.wav",
            [Cue("parent", 5, 6, "source", "translation")],
            on_warning=warnings.append,
        )
    )

    assert fake.excerpt_durations == pytest.approx([1.7, 5.0])
    assert vad.calls == 0
    assert warnings == []
    assert (aligned.cues[0].start, aligned.cues[0].end) == (6.2, 6.8)
    assert aligned.cues[0].metadata["alignment_backend"] == "whisperx"
    assert aligned.cues[0].metadata["whisperx"]["attempts"] == [
        {
            "window_start": 4.8,
            "window_end": 6.5,
            "outcome": "zero-first-token-score",
        },
        {
            "window_start": 3.0,
            "window_end": 8.0,
            "outcome": "success",
        },
    ]


def test_whisperx_zero_first_token_score_uses_vad_after_expanded_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeWhisperX(
        lambda _segments, _call: {
            "segments": [{
                "text": "source",
                "words": [
                    {"word": "unmatched", "score": 0},
                    {"word": "source", "start": 0.7, "end": 1.3, "score": 0.9},
                ],
            }]
        }
    )
    monkeypatch.setattr(alignment_module, "import_module", lambda _name: fake)
    vad = FakeVadBackend({"parent": (1.1, 1.9)})
    warnings: list[str] = []

    aligned = asyncio.run(
        WhisperXAlignmentBackend(
            language="en", device="cpu", vad_backend=vad
        ).align(
            tmp_path / "audio.wav",
            [Cue("parent", 1, 2, "source", "translation")],
            on_warning=warnings.append,
        )
    )

    assert fake.excerpt_durations == pytest.approx([1.7, 4.0])
    assert vad.calls == 1
    assert (aligned.cues[0].start, aligned.cues[0].end) == (1.1, 1.9)
    assert aligned.cues[0].metadata["alignment_backend"] == "vad-fallback"
    assert aligned.cues[0].metadata["whisperx"]["attempts"] == [
        {
            "window_start": 0.8,
            "window_end": 2.5,
            "outcome": "zero-first-token-score",
        },
        {
            "window_start": 0.0,
            "window_end": 4.0,
            "outcome": "zero-first-token-score",
        },
    ]
    assert aligned.cues[0].metadata["whisperx"]["words"] == [
        {"word": "unmatched", "score": 0.0},
        {
            "word": "source",
            "score": 0.9,
            "start": pytest.approx(0.7),
            "end": pytest.approx(1.3),
        },
    ]
    assert "first token score was 0" in warnings[0]


def test_volume_refiner_detects_quiet_speech_before_a_loud_peak(tmp_path: Path) -> None:
    audio = tmp_path / "quiet-then-loud.wav"
    _volume_wav(
        audio,
        lambda time: (
            12
            if 0.5 <= time < 0.9
            else 8_000
            if time < 0.4 or 1.2 <= time < 1.6
            else 2
        ),
    )

    refined = PcmVolumeStartRefiner._refine(
        audio, [Cue("cue", 0.4, 1.8, "quiet speech")]
    )[0]

    assert refined.start == pytest.approx(0.44, abs=0.03)
    assert refined.metadata["volume_start"]["detected_start"] == pytest.approx(
        0.5, abs=0.02
    )
    assert refined.metadata["volume_start"]["weak_threshold"] < 8


def test_volume_refiner_ignores_a_single_short_spike(tmp_path: Path) -> None:
    audio = tmp_path / "spike-then-quiet-speech.wav"
    _volume_wav(
        audio,
        lambda time: (
            8_000
            if 0.3 <= time < 0.32
            else 40
            if 0.7 <= time < 1.0
            else 2
        ),
    )

    refined = PcmVolumeStartRefiner._refine(
        audio, [Cue("cue", 0.1, 1.5, "quiet speech")]
    )[0]

    assert refined.start == pytest.approx(0.64, abs=0.03)
    assert refined.metadata["volume_start"]["detected_start"] == pytest.approx(
        0.7, abs=0.02
    )


def _token_cue(
    start: float,
    end: float,
    token_start_ms: int,
    token_end_ms: int,
    *,
    cue_id: str = "cue",
    token_text: str = " source",
) -> Cue:
    return Cue(
        cue_id,
        start,
        end,
        "source",
        "translated",
        metadata={
            "keep": {"nested": True},
            "whisper": {
                "tokens": [
                    {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}},
                    {
                        "text": token_text,
                        "offsets": {"from": token_start_ms, "to": token_end_ms},
                    },
                ]
            },
        },
    )


def _split_leading_token_cue() -> Cue:
    return Cue(
        "cue",
        0.5,
        2.5,
        "source",
        metadata={
            "whisper": {
                "tokens": [
                    {"text": "first", "offsets": {"from": 950, "to": 1050}},
                    {"text": "second", "offsets": {"from": 1050, "to": 1300}},
                ]
            }
        },
    )


def test_whisper_vad_moves_chained_start_and_preserves_mapping() -> None:
    cues = [Cue("a", 0, 1, "one", "一"), Cue("b", 1, 2.5, "two", "二")]
    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), cues, [(0, 0.8), (1.5, 2.3)]
    )

    assert result.backend == "whisper-vad"
    assert result.cues[1].start == 1.5
    assert [(c.id, c.source, c.translated) for c in result.cues] == [
        ("a", "one", "一"),
        ("b", "two", "二"),
    ]


def test_pcm_vad_is_used_without_whisper_intervals(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    _wav(audio)
    cues = [Cue("a", 0, 1, "one"), Cue("b", 1, 2.5, "two")]

    result = WhisperVadAlignmentBackend._align(audio, cues, [])

    assert result.backend == "pcm-vad"
    assert result.cues[1].start >= 1.45
    assert result.cues[0].end == pytest.approx(0.8)
    assert result.cues[1].end == pytest.approx(3.0)
    extended = extend_cue_ends(result.cues)
    assert extended[0].end == pytest.approx(extended[1].start)
    assert extended[1].end == pytest.approx(3.5)


def test_failed_pcm_vad_keeps_original_timeline(tmp_path: Path) -> None:
    cue = Cue("id", 0, 1, "hello")
    warnings: list[str] = []

    result = asyncio.run(
        WhisperVadAlignmentBackend().align(
            tmp_path / "missing.wav", [cue], on_warning=warnings.append
        )
    )

    assert result.cues == [cue]
    assert result.backend == "whisper-vad-unaligned"
    assert warnings and "timing adjustment skipped" in warnings[0]


def test_pre_alignment_start_uses_last_vad_silence_over_six_seconds() -> None:
    cue = Cue("id", 1, 50, "hello", timing_confidence=0.7)

    adjusted = adjust_cue_starts_for_long_vad_silences(
        [cue],
        [(0, 2), (13, 15), (30, 32), (40, 45)],
    )

    assert adjusted[0].start == 37
    assert adjusted[0].end == 50
    assert adjusted[0].timing_confidence == 0.7


def test_pre_alignment_start_ignores_exactly_six_seconds_and_trailing_silence() -> None:
    cues = [Cue("exact", 0, 20, "one"), Cue("trailing", 20, 50, "two")]

    adjusted = adjust_cue_starts_for_long_vad_silences(
        cues,
        [(0, 2), (8, 14), (20, 22)],
    )

    assert [cue.start for cue in adjusted] == [0, 20]


def test_first_token_crossing_long_processed_silence_moves_start() -> None:
    cue = _token_cue(0.5, 2.5, 1050, 1250)

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [cue], [(0, 1), (2, 3)]
    )

    assert result.cues[0].start == 2
    assert result.cues[0].timing_confidence == 0.9
    assert result.low_confidence_ids == []


def test_first_token_crossing_multiple_long_silences_uses_last_one() -> None:
    cue = _token_cue(0.5, 4.5, 1000, 2500)

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [cue], [(0, 1), (2, 3), (4, 5)]
    )

    assert result.cues[0].start == 4


@pytest.mark.parametrize(
    ("next_start", "expected"),
    [(1.319, 0.5), (1.32, 1.32)],
)
def test_multi_character_long_silence_threshold_is_inclusive(
    next_start: float, expected: float
) -> None:
    cue = _token_cue(0.5, 2.5, 1050, 1250)

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [cue], [(0, 1), (next_start, 3)]
    )

    assert result.cues[0].start == pytest.approx(expected)


def test_single_character_first_token_accepts_any_vad_silence() -> None:
    cue = _token_cue(0.5, 2.5, 1050, 1250, token_text="私")

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [cue], [(0, 1), (1.16, 3)]
    )

    assert result.cues[0].start == pytest.approx(1.16)


def test_short_first_token_can_cross_processed_silence() -> None:
    cue = _token_cue(0.5, 2.5, 1100, 1200)

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [cue], [(0, 1), (2, 3)]
    )

    assert result.cues[0].start == 2


def test_first_token_not_crossing_entire_processed_silence_uses_normal_start() -> None:
    cue = _token_cue(0.5, 2.5, 1110, 1200)

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [cue], [(0, 1), (2, 3)]
    )

    assert result.cues[0].start == 0.5


def test_split_leading_token_moves_when_first_token_has_no_pcm_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        WhisperVadAlignmentBackend,
        "_pcm_intervals",
        classmethod(lambda cls, audio: [(0, 0.8), (2, 3)]),
    )

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [_split_leading_token_cue()], [(0, 1), (2, 3)]
    )

    assert result.cues[0].start == 2


def test_split_leading_token_stays_when_first_token_has_pcm_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        WhisperVadAlignmentBackend,
        "_pcm_intervals",
        classmethod(lambda cls, audio: [(0.95, 1.05), (2, 3)]),
    )

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [_split_leading_token_cue()], [(0, 1), (2, 3)]
    )

    assert result.cues[0].start == 0.5


def test_split_leading_token_pcm_failure_keeps_normal_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_pcm(cls: type[WhisperVadAlignmentBackend], audio: Path):
        raise wave.Error("invalid PCM")

    monkeypatch.setattr(
        WhisperVadAlignmentBackend, "_pcm_intervals", classmethod(fail_pcm)
    )

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [_split_leading_token_cue()], [(0, 1), (2, 3)]
    )

    assert result.backend == "whisper-vad"
    assert result.cues[0].start == 0.5


def test_normal_start_alignment_still_uses_first_suitable_speech_span() -> None:
    cue = Cue("cue", 0.5, 2, "source")

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [cue], [(0, 0.2), (0.8, 1.2), (1.5, 1.8)]
    )

    assert result.cues[0].start == 0.8


def test_start_stays_when_current_vad_has_more_than_threshold_remaining() -> None:
    cues = [Cue("a", 0, 1, "one"), Cue("b", 1.1, 3, "two")]

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), cues, [(0, 2), (2.5, 3)]
    )

    assert result.cues[1].start == 1.1


def test_start_moves_when_current_vad_has_exactly_threshold_remaining() -> None:
    cues = [Cue("a", 0, 1, "one"), Cue("b", 1.2, 3, "two")]

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), cues, [(0, 2), (2.5, 3)]
    )

    assert result.cues[1].start == 2.5


def test_vad_end_uses_nearest_endpoint_within_one_second() -> None:
    cues = [Cue("a", 0, 2, "one"), Cue("b", 3.8, 4.6, "two")]

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), cues, [(0, 1.6), (2, 2.3), (3.8, 4.5)]
    )

    # 2.3 is closer to the original 2.0 end than 1.6. End extension is a
    # separate, backend-neutral pass.
    assert result.cues[0].end == pytest.approx(2.3)
    assert extend_cue_ends(result.cues)[0].end == pytest.approx(2.8)


def test_vad_end_more_than_one_second_away_keeps_original_end() -> None:
    cues = [Cue("a", 0, 2, "one"), Cue("b", 4, 4.8, "two")]

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), cues, [(0, 0.9), (4, 4.7)]
    )

    assert result.cues[0].end == 2.0
    assert extend_cue_ends(result.cues)[0].end == 2.5


@pytest.mark.parametrize(
    ("next_start", "expected_end"),
    [
        (1.7, 1.7),
        (1.8, 1.4),
        (2.0, 1.5),
        (2.2, 1.5),
    ],
)
def test_neighbor_gap_adjusts_end(next_start: float, expected_end: float) -> None:
    cues = [Cue("a", 0, 1, "one"), Cue("b", next_start, next_start + 0.5, "two")]

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), cues, [(0, 1), (next_start, next_start + 0.4)]
    )

    assert result.cues[0].end == pytest.approx(1.0)
    assert extend_cue_ends(result.cues)[0].end == pytest.approx(expected_end)


def test_last_cue_uses_refined_end_then_adds_half_second() -> None:
    cue = Cue("a", 0, 2, "one")

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [cue], [(0, 1.7)]
    )

    assert result.cues[0].end == 1.7
    assert extend_cue_ends(result.cues)[0].end == 2.2


@pytest.mark.parametrize(
    ("fallback_end", "forced_start", "expected_fallback_end"),
    [(1.0, 1.5, 1.5), (1.5, 1.0, 1.0)],
)
def test_shared_end_extension_uses_final_mixed_timeline_starts(
    fallback_end: float,
    forced_start: float,
    expected_fallback_end: float,
) -> None:
    extended = extend_cue_ends(
        [
            Cue(
                "fallback",
                0,
                fallback_end,
                "fallback",
                metadata={"alignment_backend": "vad-fallback"},
            ),
            Cue(
                "forced",
                forced_start,
                2,
                "forced",
                metadata={"alignment_backend": "whisperx"},
            ),
        ]
    )

    assert extended[0].end == expected_fallback_end
    assert extended[1].start == forced_start


def test_alignment_preserves_cue_content_and_metadata() -> None:
    cue = _token_cue(0.5, 2.5, 1050, 1250, cue_id="stable-id")
    metadata = cue.metadata

    result = WhisperVadAlignmentBackend._align(
        Path("unused.wav"), [cue], [(0, 1), (2, 3)]
    )

    aligned = result.cues[0]
    assert (aligned.id, aligned.source, aligned.translated) == (
        "stable-id",
        "source",
        "translated",
    )
    assert aligned.metadata is metadata
