"""Opt-in real-model smoke test; normal test runs never download models."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from yakiflow.alignment import WhisperXAlignmentBackend
from yakiflow.models import Cue


pytestmark = pytest.mark.whisperx_smoke


def test_local_whisperx_alignment_smoke() -> None:
    audio_value = os.environ.get("YAKIFLOW_WHISPERX_SMOKE_AUDIO")
    if not audio_value:
        pytest.skip("set YAKIFLOW_WHISPERX_SMOKE_AUDIO to run the real-model smoke test")
    pytest.importorskip("whisperx")
    audio = Path(audio_value)
    language = os.environ.get("YAKIFLOW_WHISPERX_SMOKE_LANGUAGE", "en")
    text = os.environ.get("YAKIFLOW_WHISPERX_SMOKE_TEXT", "hello world")
    start = float(os.environ.get("YAKIFLOW_WHISPERX_SMOKE_START", "0.5"))
    end = float(os.environ.get("YAKIFLOW_WHISPERX_SMOKE_END", "5.0"))

    result = asyncio.run(
        WhisperXAlignmentBackend(language=language).align(
            audio,
            [Cue("parent", start, end, text)],
        )
    )

    assert result.backend == "whisperx"
    assert result.cues
    assert all(cue.metadata.get("alignment_backend") == "whisperx" for cue in result.cues)
    assert all(0 <= cue.start <= cue.end for cue in result.cues)
