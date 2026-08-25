import asyncio
import json
from pathlib import Path

import pytest

from conftest import make_settings
from yakiflow.database import JobDatabase
from yakiflow.elevenlabs import cues_from_words
from yakiflow.models import Cue, Word
from yakiflow.translation import AgentBackend, TranslationPipeline


def _word(ordinal: int, start: float, end: float, text: str, speaker=None) -> Word:
    return Word(ordinal=ordinal, start=start, end=end, text=text, speaker=speaker)


def _settings(**draft_overrides):
    draft = {"model": "draft", "retry_delay_seconds": 0, "word_batch_size": 50}
    draft.update(draft_overrides)
    return make_settings(
        source_language="en",
        target_language="zh-CN",
        transcription={"backend": "elevenlabs"},
        agent={"backend": "codex", "draft": draft},
    ).resolved()


class SegmentingBackend(AgentBackend):
    """Groups consecutive same-speaker words into cues, one call per batch."""

    name = "fake"

    def __init__(self, bad_first_responses: int = 0):
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.bad_first_responses = bad_first_responses

    async def invoke_with_trace(
        self, prompt, *, system="", model, effort, schema, on_event=None
    ):
        self.prompts.append(prompt)
        self.systems.append(system)
        payload = json.loads(
            prompt.split("INPUT:\n", 1)[1].split("\nYour previous", 1)[0]
        )
        words = payload["words"]
        if self.bad_first_responses > 0:
            self.bad_first_responses -= 1
            # Claim only the first word: mechanical validation must reject
            # the uncovered remainder and feed the ordinals back.
            return {"cues": [{
                "first_word": words[0]["i"],
                "last_word": words[0]["i"],
                "translated": "T:incomplete",
            }]}
        cues = []
        current: list[dict] = []
        for word in words:
            if current and current[-1].get("s") != word.get("s"):
                cues.append(self._cue(current))
                current = []
            current.append(word)
        if current:
            cues.append(self._cue(current))
        return {"cues": cues}

    @staticmethod
    def _cue(words: list[dict]) -> dict:
        text = "".join(word["w"] for word in words).strip()
        return {
            "first_word": words[0]["i"],
            "last_word": words[-1]["i"],
            "translated": f"T:{text}",
        }


def _pipeline(db, backend, settings=None) -> TranslationPipeline:
    return TranslationPipeline(settings or _settings(), backend, db, "")


def _store_words(db, words) -> None:
    db.replace_transcript_words(words)


def _conversation_words() -> list[Word]:
    return [
        _word(0, 0.0, 0.4, "How ", "1"),
        _word(1, 0.5, 0.9, "are ", "1"),
        _word(2, 1.0, 1.4, "you?", "1"),
        _word(3, 1.2, 1.6, "Fine ", "2"),
        _word(4, 1.7, 2.1, "thanks", "2"),
        # 1.0 s of full-track silence, then a last remark.
        _word(5, 3.2, 3.6, "Good", "1"),
    ]


def test_segment_and_translate_replaces_preview_cues_with_a_sorted_timeline(
    tmp_path: Path,
) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    words = _conversation_words()
    _store_words(db, words)
    preview = cues_from_words(words, max_cue_seconds=8.0, max_cue_chars=84)
    db.upsert_cues(preview, stable=False)
    backend = SegmentingBackend()
    pipeline = _pipeline(db, backend)
    snapshots: list[list[str]] = []

    async def on_batch(cues) -> None:
        snapshots.append([cue.id for cue in cues])

    final = asyncio.run(pipeline.segment_and_translate(on_batch))

    assert [cue.id for cue in final] == [str(i) for i in range(1, len(final) + 1)]
    assert [cue.speaker for cue in final] == ["1", "2", "1"]
    assert [cue.source for cue in final] == ["How are you?", "Fine thanks", "Good"]
    assert all(cue.translated and cue.translated.startswith("T:") for cue in final)
    # The crosstalk overlap survives, sorted by start.
    assert final[1].start < final[0].end
    assert final[0].metadata["word_range"] == [0, 2]
    assert final[1].metadata["word_range"] == [3, 4]
    # Word-level evidence rides along.
    assert [word["ordinal"] for word in final[0].metadata["words"]] == [0, 1, 2]
    # Preview cues never reach the stable timeline.
    assert not any(cue.id.startswith("preview-") for cue in final)
    assert final == db.list_cues(stable_only=True)
    db.close()


def test_speaker_labels_travel_into_the_agent_input(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    words = _conversation_words()
    _store_words(db, words)
    backend = SegmentingBackend()
    pipeline = _pipeline(db, backend)

    asyncio.run(pipeline.segment_and_translate())

    payload = json.loads(backend.prompts[0].split("INPUT:\n", 1)[1])
    assert payload["words"][0] == {"i": 0, "t": 0.0, "w": "How ", "s": "1"}
    assert "speaker" in backend.systems[0].lower()
    assert "never write timestamps" in backend.systems[0]
    db.close()


def test_invalid_response_is_fed_back_with_word_ordinals_and_retried(
    tmp_path: Path,
) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    words = _conversation_words()
    _store_words(db, words)
    backend = SegmentingBackend(bad_first_responses=1)
    pipeline = _pipeline(db, backend)

    final = asyncio.run(pipeline.segment_and_translate())

    assert len(final) == 3
    retry_prompt = backend.prompts[1]
    assert "mechanical validation" in retry_prompt
    # The uncovered word ordinals are quoted back.
    assert "not covered by any cue: [1, 2, 5]" in retry_prompt
    db.close()


def test_validation_failures_exhaust_attempts_and_fail(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    _store_words(db, _conversation_words())
    backend = SegmentingBackend(bad_first_responses=99)
    pipeline = _pipeline(db, backend, _settings(max_attempts=2))

    with pytest.raises(ValueError, match="not covered"):
        asyncio.run(pipeline.segment_and_translate())
    db.close()


def test_missing_word_batches_skip_covered_ranges(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    words = _conversation_words()
    _store_words(db, words)
    db.upsert_cues([
        Cue("w0-2", 0.0, 1.4, "How are you?", "已译", metadata={"word_range": [0, 2]}, speaker="1"),
    ], stable=True)
    pipeline = _pipeline(db, SegmentingBackend())

    batches = pipeline.missing_word_batches(words)

    covered = [word.ordinal for batch in batches for word in batch.words]
    assert covered == [3, 4, 5]
    db.close()


def test_resume_only_dispatches_uncovered_words(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    words = _conversation_words()
    _store_words(db, words)
    db.upsert_cues([
        Cue("w0-2", 0.0, 1.4, "How are you?", "已译", metadata={"word_range": [0, 2]}, speaker="1"),
    ], stable=True)
    backend = SegmentingBackend()
    pipeline = _pipeline(db, backend)

    final = asyncio.run(pipeline.segment_and_translate())

    # Only the uncovered words were sent to the agent.
    dispatched = [
        json.loads(prompt.split("INPUT:\n", 1)[1])["words"][0]["i"]
        for prompt in backend.prompts
    ]
    assert 0 not in dispatched
    # The already-finished translation survives into the final timeline.
    assert [cue.translated for cue in final][0] == "已译"
    assert len(final) == 3
    db.close()


def test_forced_batch_boundaries_get_junction_repair(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    # 12 continuous words, no qualified silence anywhere: with a target of 5
    # (hard limit 10) the first boundary is a forced cut through speech.
    words = [
        _word(index, index * 0.4, index * 0.4 + 0.35, f"w{index} ", "1")
        for index in range(12)
    ]
    _store_words(db, words)
    backend = SegmentingBackend()
    pipeline = _pipeline(db, backend, _settings(word_batch_size=5))

    final = asyncio.run(pipeline.segment_and_translate())

    # Two batch calls plus one junction-repair call over the combined span.
    assert len(backend.prompts) == 3
    repair_payload = json.loads(backend.prompts[-1].split("INPUT:\n", 1)[1])
    repair_ordinals = [word["i"] for word in repair_payload["words"]]
    assert repair_ordinals == list(range(12))
    # The repair's single re-cut cue replaced both halves.
    assert len(final) == 1
    assert final[0].metadata["word_range"] == [0, 11]
    assert final[0].source == " ".join(f"w{i}" for i in range(12))
    db.close()


def test_natural_boundaries_are_not_repaired(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    # A qualified 1.0 s silence separates the two halves, so the boundary is
    # a legitimate cue boundary and stays untouched.
    words = [
        _word(0, 0.0, 0.4, "one ", "1"),
        _word(1, 0.5, 0.9, "two", "1"),
        _word(2, 2.0, 2.4, "three ", "1"),
        _word(3, 2.5, 2.9, "four ", "1"),
        _word(4, 3.0, 3.4, "five ", "1"),
        _word(5, 3.5, 3.9, "six", "1"),
    ]
    _store_words(db, words)
    backend = SegmentingBackend()
    pipeline = _pipeline(db, backend, _settings(word_batch_size=2))

    final = asyncio.run(pipeline.segment_and_translate())

    assert len(backend.prompts) == 2
    assert [cue.metadata["word_range"] for cue in final] == [[0, 1], [2, 5]]
    db.close()


# --- the mechanical validation matrix ---


def _validate(raw, batch_words, tmp_path):
    db = JobDatabase(tmp_path / "job.sqlite3")
    try:
        pipeline = _pipeline(db, SegmentingBackend())
        return pipeline._cues_from_word_response(raw, batch_words)
    finally:
        db.close()


def test_validation_rejects_cross_speaker_intervals(tmp_path: Path) -> None:
    words = [_word(0, 0.0, 0.4, "a ", "1"), _word(1, 0.5, 0.9, "b", "2")]
    with pytest.raises(ValueError, match="cue 0–1.*one speaker"):
        _validate(
            {"cues": [{"first_word": 0, "last_word": 1, "translated": "t"}]},
            words,
            tmp_path,
        )


def test_validation_rejects_out_of_batch_words(tmp_path: Path) -> None:
    words = [_word(5, 0.0, 0.4, "a")]
    with pytest.raises(ValueError, match="outside this batch"):
        _validate(
            {"cues": [{"first_word": 5, "last_word": 9, "translated": "t"}]},
            words,
            tmp_path,
        )


def test_validation_rejects_double_and_missing_coverage(tmp_path: Path) -> None:
    words = [
        _word(0, 0.0, 0.4, "a ", "1"),
        _word(1, 0.5, 0.9, "b ", "1"),
        _word(2, 1.0, 1.4, "c", "1"),
    ]
    with pytest.raises(ValueError, match="already covered.*\\[1\\]"):
        _validate(
            {"cues": [
                {"first_word": 0, "last_word": 1, "translated": "t"},
                {"first_word": 1, "last_word": 2, "translated": "t"},
            ]},
            words,
            tmp_path,
        )
    with pytest.raises(ValueError, match="not covered by any cue: \\[2\\]"):
        _validate(
            {"cues": [{"first_word": 0, "last_word": 1, "translated": "t"}]},
            words,
            tmp_path,
        )


def test_validation_enforces_subtitle_limits_and_nonempty_translation(
    tmp_path: Path,
) -> None:
    slow = [_word(0, 0.0, 0.4, "a ", "1"), _word(1, 9.5, 9.9, "b", "1")]
    with pytest.raises(ValueError, match="above the 8s limit"):
        _validate(
            {"cues": [{"first_word": 0, "last_word": 1, "translated": "t"}]},
            slow,
            tmp_path,
        )
    words = [_word(0, 0.0, 0.4, "a", "1")]
    with pytest.raises(ValueError, match="empty translation"):
        _validate(
            {"cues": [{"first_word": 0, "last_word": 0, "translated": "  "}]},
            words,
            tmp_path,
        )
    with pytest.raises(ValueError, match="characters"):
        _validate(
            {"cues": [{"first_word": 0, "last_word": 0, "translated": "长" * 120}]},
            words,
            tmp_path,
        )


def test_cue_derivation_is_mechanical(tmp_path: Path) -> None:
    words = [
        _word(3, 10.0, 10.4, "there ", "2"),
        _word(4, 10.5, 10.9, "cat", "2"),
    ]
    cues = _validate(
        {"cues": [{
            "first_word": 3,
            "last_word": 4,
            "source": "their cat",
            "translated": "他们的猫",
        }]},
        words,
        tmp_path,
    )
    cue = cues[0]
    assert (cue.start, cue.end) == (10.0, 10.9)
    assert cue.speaker == "2"
    assert cue.id == "w3-4"
    assert cue.source == "their cat"
    assert cue.translated == "他们的猫"
    assert cue.metadata["word_range"] == [3, 4]

    default = _validate(
        {"cues": [{"first_word": 3, "last_word": 4, "translated": "他们的猫"}]},
        words,
        tmp_path,
    )
    assert default[0].source == "there cat"
