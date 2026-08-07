from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

from .models import Cue, OutputMode


_LANGUAGE_COMPONENT_RE = re.compile(r"[^a-z0-9-]+")


def timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def cue_text(cue: Cue, mode: OutputMode | str) -> str:
    mode = OutputMode(mode)
    translated = (cue.translated or "").strip()
    source = cue.source.strip()
    if mode is OutputMode.SOURCE:
        return source
    if mode is OutputMode.TRANSLATED:
        return translated or source
    # Bilingual output deliberately puts translation first.
    return f"{translated}\n{source}" if translated else source


def render_srt(cues: Iterable[Cue], mode: OutputMode | str = OutputMode.BILINGUAL) -> str:
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(
            f"{index}\n{timestamp(cue.start)} --> {timestamp(cue.end)}\n{cue_text(cue, mode)}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt_atomic(path: Path, cues: Iterable[Cue], mode: OutputMode | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_srt(cues, mode))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _language_component(language: str) -> str:
    component = language.strip().lower().replace("_", "-")
    component = _LANGUAGE_COMPONENT_RE.sub("-", component).strip("-")
    if not component:
        raise ValueError("subtitle language cannot be empty")
    return component


def publish_outputs(
    base: Path,
    cues: Iterable[Cue],
    mode: OutputMode | str,
    *,
    source_language: str | None = None,
    target_language: str | None = None,
) -> list[Path]:
    cues = list(cues)
    mode = OutputMode(mode)
    modes = (
        [OutputMode.SOURCE, OutputMode.TRANSLATED, OutputMode.BILINGUAL]
        if mode is OutputMode.ALL
        else [mode]
    )
    suffixes = {
        OutputMode.SOURCE: ".source.srt",
        OutputMode.TRANSLATED: ".translated.srt",
    }
    if OutputMode.BILINGUAL in modes:
        if source_language is None or target_language is None:
            raise ValueError("bilingual output requires source and target languages")
        suffixes[OutputMode.BILINGUAL] = (
            f".{_language_component(source_language)}-"
            f"{_language_component(target_language)}.srt"
        )
    paths = []
    for selected in modes:
        path = base.parent / f"{base.name}{suffixes[selected]}"
        write_srt_atomic(path, cues, selected)
        paths.append(path)
    return paths
