from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .models import Cue, OutputMode


_LANGUAGE_COMPONENT_RE = re.compile(r"[^a-z0-9-]+")
_TIMING_RE = re.compile(
    r"^(?P<start>\d{2,}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2,}:\d{2}:\d{2}[,.]\d{3})\s*$"
)


def timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def seconds(value: str) -> float:
    """Read one ``HH:MM:SS,mmm`` subtitle timestamp."""
    hours, minutes, rest = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


@dataclass(frozen=True, slots=True)
class SrtBlock:
    """One parsed SRT block, keeping the numbering line exactly as written."""

    number: str
    start: float
    end: float
    text: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SrtProblem:
    """A structural defect found in a rendered subtitle file."""

    kind: str
    message: str


def parse_srt_blocks(text: str) -> tuple[list[SrtBlock], list[SrtProblem]]:
    """Parse an SRT into blocks, reporting the ones that do not parse."""
    blocks: list[SrtBlock] = []
    problems: list[SrtProblem] = []
    chunks = re.split(r"\n\s*\n", text.lstrip("﻿").replace("\r\n", "\n"))
    for position, chunk in enumerate(chunks, 1):
        lines = [line.rstrip("\r") for line in chunk.strip("\n").splitlines()]
        if not lines:
            continue
        if len(lines) < 3 or not lines[0].strip().isdigit():
            problems.append(SrtProblem(
                "unparsable", f"block {position} is not a numbered SRT cue"
            ))
            continue
        match = _TIMING_RE.match(lines[1].strip())
        if match is None:
            problems.append(SrtProblem(
                "unparsable", f"block {position} has no valid timing line"
            ))
            continue
        body = tuple(line for line in lines[2:] if line.strip())
        if not body:
            problems.append(SrtProblem(
                "empty_text", f"block {position} has no subtitle text"
            ))
            continue
        blocks.append(SrtBlock(
            lines[0].strip(),
            seconds(match.group("start")),
            seconds(match.group("end")),
            body,
        ))
    return blocks, problems


def srt_problems(text: str) -> list[SrtProblem]:
    """Report every structural defect in one rendered subtitle file.

    Numbering is deliberately excluded: it is mechanical enough to repair
    without asking anyone, and :func:`renumbered_srt` does exactly that.
    """
    blocks, problems = parse_srt_blocks(text)
    for index, block in enumerate(blocks):
        label = f"cue {block.number}"
        if block.end < block.start:
            problems.append(SrtProblem(
                "end_before_start", f"{label} ends before it starts"
            ))
        if index == 0:
            continue
        previous = blocks[index - 1]
        if block.start < previous.start:
            problems.append(SrtProblem(
                "out_of_order", f"{label} starts before cue {previous.number}"
            ))
        elif block.start < previous.end:
            problems.append(SrtProblem(
                "overlap", f"{label} overlaps cue {previous.number}"
            ))
    return problems


def alignment_problems(
    artifacts: Sequence[tuple[str, Sequence[SrtBlock]]],
) -> list[SrtProblem]:
    """Report cue drift between subtitle files that must stay equivalent."""
    if len(artifacts) < 2:
        return []
    problems: list[SrtProblem] = []
    reference_name, reference = artifacts[0]
    for name, blocks in artifacts[1:]:
        if len(blocks) != len(reference):
            problems.append(SrtProblem(
                "artifact_mismatch",
                f"{name} has {len(blocks)} cues but {reference_name} has "
                f"{len(reference)}",
            ))
            continue
        for index, (expected, block) in enumerate(zip(reference, blocks), 1):
            same_timing = (
                round(expected.start, 3) == round(block.start, 3)
                and round(expected.end, 3) == round(block.end, 3)
            )
            if not same_timing:
                problems.append(SrtProblem(
                    "artifact_mismatch",
                    f"{name} cue {index} does not share the timing of "
                    f"{reference_name}",
                ))
                break
    return problems


def renumbered_srt(text: str) -> str | None:
    """Return ``text`` with cue numbers 1..N, or ``None`` when already correct."""
    blocks, problems = parse_srt_blocks(text)
    if problems or not blocks:
        return None
    if [block.number for block in blocks] == [
        str(index) for index in range(1, len(blocks) + 1)
    ]:
        return None
    return "\n\n".join(
        f"{index}\n{timestamp(block.start)} --> {timestamp(block.end)}\n"
        + "\n".join(block.text)
        for index, block in enumerate(blocks, 1)
    ) + "\n"


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


def output_modes(mode: OutputMode | str) -> list[OutputMode]:
    """Expand one configured output mode into the modes actually written."""
    mode = OutputMode(mode)
    return (
        [OutputMode.SOURCE, OutputMode.TRANSLATED, OutputMode.BILINGUAL]
        if mode is OutputMode.ALL
        else [mode]
    )


def publish_outputs(
    base: Path,
    cues: Iterable[Cue],
    mode: OutputMode | str,
    *,
    source_language: str | None = None,
    target_language: str | None = None,
) -> list[Path]:
    cues = list(cues)
    modes = output_modes(mode)
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
