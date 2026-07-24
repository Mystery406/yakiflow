from __future__ import annotations

import asyncio
import json
import re
import socket
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, Sequence

from .config import Settings
from .database import JobDatabase
from .models import Cue, TranscriptEvent, cue_id
from .process import CommandRunner, ProcessError


EventCallback = Callable[[TranscriptEvent], Awaitable[None]]
ProgressCallback = Callable[[float], Awaitable[None]]
LINE_RE = re.compile(
    r"\[\s*(?P<start>\d\d:\d\d:\d\d[.,]\d{3})\s*-->\s*(?P<end>\d\d:\d\d:\d\d[.,]\d{3})\s*\]\s*(?P<text>.*)"
)
PROGRESS_RE = re.compile(r"progress\s*=\s*(?P<percent>\d{1,3})%")
VAD_SEGMENT_RE = re.compile(
    r"VAD segment \d+: start = (?P<start>\d+(?:\.\d+)?), "
    r"end = (?P<end>\d+(?:\.\d+)?)"
)


def normalize_source_language(language: object) -> str | None:
    """Normalize Whisper locale labels to WhisperX's language codes."""
    if not isinstance(language, str):
        return None
    normalized = language.strip().replace("_", "-").casefold()
    if not normalized or normalized == "auto":
        return None
    return normalized.split("-", 1)[0]


def detected_source_language(data: dict[str, Any]) -> str | None:
    result = data.get("result")
    language = result.get("language") if isinstance(result, dict) else None
    if language is None:
        language = data.get("language")
    return normalize_source_language(language)


def parse_timestamp(
    value: str | int | float, *, integer_milliseconds: bool = False
) -> float:
    if isinstance(value, (int, float)):
        return (
            float(value) / 1000
            if integer_milliseconds and isinstance(value, int)
            else float(value)
        )
    parts = value.replace(",", ".").split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(value)


def parse_vad_segment(value: str, *, offset: float = 0.0) -> tuple[float, float] | None:
    """Parse one original-timeline VAD interval from whisper.cpp stderr."""
    match = VAD_SEGMENT_RE.search(value)
    if match is None:
        return None
    start = float(match.group("start")) + offset
    end = float(match.group("end")) + offset
    return (round(start, 3), round(end, 3)) if end >= start else None


def merge_vad_intervals(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end < start:
            continue
        if merged and start <= merged[-1][1] + 0.01:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def parse_json_full(data: dict[str, Any], *, offset: float = 0.0, ordinal_offset: int = 0) -> list[Cue]:
    entries = data.get("transcription") or data.get("segments") or []
    cues: list[Cue] = []
    for index, entry in enumerate(entries):
        timestamps = entry.get("timestamps", {})
        offsets = entry.get("offsets", {})

        def entry_time(timestamp_key: str, segment_key: str, default: float) -> float:
            if timestamp_key in timestamps:
                return parse_timestamp(timestamps[timestamp_key])
            if segment_key in entry:
                # The segments JSON format measures start/end in seconds,
                # including when those values happen to be integral.
                return parse_timestamp(entry[segment_key])
            if timestamp_key in offsets:
                # whisper.cpp's integer offset fields are milliseconds.
                return parse_timestamp(
                    offsets[timestamp_key], integer_milliseconds=True
                )
            return default

        relative_start = entry_time("from", "start", 0.0)
        relative_end = entry_time("to", "end", relative_start)
        start = relative_start + offset
        end = relative_end + offset
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        ordinal = ordinal_offset + len(cues)
        cues.append(Cue(cue_id(ordinal), start, max(start, end), text, metadata={"whisper": entry}))
    return cues


def merge_overlap(existing: list[Cue], incoming: list[Cue]) -> list[Cue]:
    """Merge recovery/stream results without duplicating an overlap window."""
    result = list(existing)
    for cue in incoming:
        normalized = " ".join(cue.source.casefold().split())
        duplicate = next(
            (old for old in reversed(result[-8:]) if abs(old.start - cue.start) < 2.5 and
             " ".join(old.source.casefold().split()) == normalized),
            None,
        )
        if duplicate:
            continue
        ordinal = len(result)
        result.append(Cue(cue_id(ordinal), cue.start, cue.end, cue.source, metadata=cue.metadata))
    result.sort(key=lambda item: (item.start, item.end))
    return result


class Transcriber(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio: Path,
        on_event: EventCallback | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        resume_from: Sequence[Cue] = (),
    ) -> list[Cue]: ...


class WhisperCliTranscriber(Transcriber):
    def __init__(self, settings: Settings, work_dir: Path, db: JobDatabase, runner: CommandRunner | None = None):
        self.settings = settings
        self.work_dir = work_dir
        self.db = db
        self.runner = runner or CommandRunner()

    async def transcribe(
        self,
        audio: Path,
        on_event: EventCallback | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        resume_from: Sequence[Cue] = (),
    ) -> list[Cue]:
        preserved = list(resume_from)
        input_audio = audio
        input_offset = 0.0
        prefix = self.work_dir / "whisper-final"
        if preserved:
            # Whisper's stdout cues are complete segments. Restart at the end
            # of the last durable segment so their IDs, text, and translations
            # remain attached to the same timeline prefix.
            input_offset = preserved[-1].end
            input_audio = self.work_dir / "whisper-resume.wav"
            await self.runner.run([
                self.settings.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(input_offset), "-i", audio,
                "-ac", "1", "-ar", "16000", input_audio,
            ])
            prefix = self.work_dir / "whisper-resume"
        incremental: list[Cue] = []
        stored_intervals = self.db.get_checkpoint("whisper_vad_intervals", [])
        vad_intervals = (
            [(float(start), float(end)) for start, end in stored_intervals]
            if preserved and self.settings.vad_model
            else []
        )
        self.db.checkpoint("whisper_vad_intervals", vad_intervals)

        def record_vad(value: str, *, offset: float) -> None:
            interval = parse_vad_segment(value, offset=offset)
            if interval is None:
                return
            vad_intervals.append(interval)
            vad_intervals[:] = merge_vad_intervals(vad_intervals)
            # VAD is computed before decoding. Persist it incrementally so an
            # interrupted Whisper run can resume without losing the timeline.
            self.db.checkpoint("whisper_vad_intervals", vad_intervals)

        async def line(stream: str, value: str) -> None:
            self.db.log("whisper-cli", stream, value)
            if stream == "stderr" and self.settings.vad_model:
                record_vad(value, offset=input_offset)
            progress = PROGRESS_RE.search(value)
            if progress and on_progress:
                await on_progress(min(100, int(progress.group("percent"))) / 100)
            match = LINE_RE.search(value)
            if stream != "stdout" or not match:
                return
            start = parse_timestamp(match.group("start")) + input_offset
            end = parse_timestamp(match.group("end")) + input_offset
            cue = Cue(
                cue_id(len(preserved) + len(incremental)),
                start,
                end,
                match.group("text").strip(),
            )
            incremental.append(cue)
            if on_event:
                await on_event(TranscriptEvent(cue, stable=True, final=False))

        args = [
            self.settings.whisper_cli, "-m", self.settings.whisper_model,
            "-f", input_audio, "-mc", "0", "--print-progress",
            "--output-json-full", "--output-file", prefix,
        ]
        if self.settings.source_language:
            args += ["--language", self.settings.source_language]
        if self.settings.vad_model:
            args += ["--vad", "--vad-model", str(self.settings.vad_model)]
        try:
            await self.runner.run(args, on_line=line)
        except ProcessError:
            # Resume from five seconds before the last stable cue and merge the
            # overlap. This retry remains a single full-model invocation.
            if not incremental:
                raise
            durable = preserved + incremental
            resume_at = max(0.0, durable[-1].end - 5.0)
            recovered = self.work_dir / "recovery.wav"
            await self.runner.run([
                self.settings.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(resume_at), "-i", audio, "-ac", "1", "-ar", "16000", recovered,
            ])
            recovery_prefix = self.work_dir / "whisper-recovery"
            recovery_args = list(args)
            recovery_args[recovery_args.index("-f") + 1] = recovered
            recovery_args[recovery_args.index("--output-file") + 1] = recovery_prefix

            async def recovery_line(stream: str, value: str) -> None:
                self.db.log("whisper-cli-recovery", stream, value)
                if stream == "stderr" and self.settings.vad_model:
                    record_vad(value, offset=resume_at)

            await self.runner.run(
                recovery_args,
                on_line=recovery_line,
            )
            recovered_cues = self._load_json(
                recovery_prefix,
                offset=resume_at,
                ordinal_offset=len(durable),
            )
            cues = merge_overlap(durable, recovered_cues)
        else:
            completed = (
                self._load_json(
                    prefix,
                    offset=input_offset,
                    ordinal_offset=len(preserved),
                )
                if self._json_path(prefix).exists()
                else incremental
            )
            cues = preserved + completed
        self.db.checkpoint("whisper_vad_intervals", vad_intervals)
        for cue in cues:
            if on_event:
                await on_event(TranscriptEvent(cue, stable=True, final=True))
        return cues

    @staticmethod
    def _json_path(prefix: Path) -> Path:
        return prefix.with_suffix(".json")

    def _load_json(self, prefix: Path, *, offset: float = 0.0, ordinal_offset: int = 0) -> list[Cue]:
        with self._json_path(prefix).open(encoding="utf-8") as fh:
            data = json.load(fh)
        language = detected_source_language(data)
        if language:
            self.db.checkpoint("detected_source_language", language)
        return parse_json_full(data, offset=offset, ordinal_offset=ordinal_offset)


def _allocate_loopback_port() -> int:
    """Reserve an unused loopback port long enough to select its number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        return int(reservation.getsockname()[1])


class WhisperServerTranscriber:
    """Persistent whisper-server client used for low-latency stream chunks."""

    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        db: JobDatabase,
        port: int | None = None,
    ):
        self.settings = settings
        self.work_dir = work_dir
        self.db = db
        self.port = port
        self.request_path = f"/yakiflow-{uuid.uuid4().hex}"
        self.process: asyncio.subprocess.Process | None = None
        self.cues: list[Cue] = []
        self._log_tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if self.port is None:
            self.port = _allocate_loopback_port()
        args: list[str] = [
            self.settings.whisper_server, "-m", str(self.settings.whisper_model),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--request-path", self.request_path,
        ]
        if self.settings.source_language:
            args += ["--language", self.settings.source_language]
        if self.settings.vad_model:
            args += ["--vad", "--vad-model", str(self.settings.vad_model)]
        self.process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        async def drain(reader: asyncio.StreamReader, stream: str) -> None:
            while raw := await reader.readline():
                self.db.log("whisper-server", stream, raw.decode(errors="replace"))
        self._log_tasks = [
            asyncio.create_task(drain(self.process.stdout, "stdout")),
            asyncio.create_task(drain(self.process.stderr, "stderr")),
        ]
        try:
            await self._wait_until_ready()
        except BaseException:
            await self.close()
            raise

    async def _wait_until_ready(self, timeout: float = 180.0) -> None:
        """Wait until whisper-server's HTTP listener accepts connections."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if self.process is None or self.process.returncode is not None:
                raise RuntimeError("whisper-server exited during startup")
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", self.port),
                    timeout=0.5,
                )
            except (OSError, asyncio.TimeoutError):
                if loop.time() >= deadline:
                    raise RuntimeError(
                        f"whisper-server was not ready after {timeout:g} seconds"
                    )
                await asyncio.sleep(0.1)
                continue
            try:
                writer.write(
                    (
                        f"OPTIONS {self.request_path}/inference HTTP/1.1\r\n"
                        "Host: 127.0.0.1\r\nConnection: close\r\n\r\n"
                    ).encode()
                )
                await writer.drain()
                status_line = await asyncio.wait_for(reader.readline(), 0.5)
            finally:
                writer.close()
                await writer.wait_closed()
            if not status_line.startswith(b"HTTP/1.1 200"):
                raise RuntimeError(
                    "whisper-server port is owned by an unexpected listener"
                )
            if self.process is None or self.process.returncode is not None:
                raise RuntimeError("whisper-server exited during startup")
            return

    async def submit(self, wav_path: Path, offset: float = 0.0) -> list[Cue]:
        payload = await asyncio.to_thread(self._post_audio, wav_path)
        language = detected_source_language(payload)
        if language:
            self.db.checkpoint("detected_source_language", language)
        incoming = parse_json_full(payload, offset=offset, ordinal_offset=len(self.cues))
        self.cues = merge_overlap(self.cues, incoming)
        return incoming

    def _post_audio(self, wav_path: Path) -> dict[str, Any]:
        import http.client
        boundary = "----yakiflow-whisper-boundary"
        audio = wav_path.read_bytes()
        fields = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n".encode(),
            # Repeat `auto` for each independent chunk; never replace it with
            # the language detected from the first chunk.
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{self.settings.source_language}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"chunk.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode(),
            audio,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=180)
        connection.request(
            "POST",
            f"{self.request_path}/inference",
            body=b"".join(fields),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = connection.getresponse()
        body = response.read()
        connection.close()
        if response.status >= 400:
            raise RuntimeError(f"whisper-server HTTP {response.status}: {body[:500]!r}")
        return json.loads(body)

    async def close(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self._log_tasks:
            await asyncio.gather(*self._log_tasks, return_exceptions=True)
