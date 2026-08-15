from __future__ import annotations

import asyncio
import re
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Awaitable, Callable, Iterable
from urllib.parse import urlparse

from .config import Settings
from .database import JobDatabase
from .process import CommandRunner


@dataclass(frozen=True, slots=True)
class MediaSource:
    value: str
    is_url: bool

    @classmethod
    def parse(cls, value: str) -> MediaSource:
        parsed = urlparse(value)
        is_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        return cls(
            value if is_url else str(Path(value).expanduser().resolve()),
            is_url,
        )


@dataclass(slots=True)
class MediaArtifact:
    source: MediaSource
    audio_path: Path
    media_path: Path | None = None
    persistent: bool = False


ChunkCallback = Callable[[Path, float], Awaitable[None]]
ProgressCallback = Callable[[float], Awaitable[None]]
WarningCallback = Callable[[str], Awaitable[None]]
DOWNLOAD_PROGRESS_RE = re.compile(r"\[download\]\s+(?:~\s*)?(?P<percent>\d+(?:\.\d+)?)%")


def _sizes(paths: Iterable[Path]) -> list[tuple[Path, int]]:
    """Pair each existing non-empty file with its size, tolerating races.

    yt-dlp renames and removes its fragment files while the directory is being
    polled, so a path can vanish between the glob and the ``stat``.
    """
    sized: list[tuple[Path, int]] = []
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 0 and path.is_file():
            sized.append((path, size))
    return sized


def pcm_audio_duration(audio: Path) -> float | None:
    """Return the playable duration of a PCM WAV file, or ``None`` when unusable.

    Streaming snapshots and interrupted extractions can leave a truncated or
    zero-rate header behind, so every caller wants the guarded answer.
    """
    try:
        with wave.open(str(audio), "rb") as source:
            rate = source.getframerate()
            duration = source.getnframes() / rate if rate > 0 else 0.0
    except (EOFError, OSError, wave.Error):
        return None
    return duration if duration > 0 else None


async def _drain_stream_chunks(
    snapshot: Path,
    duration: float,
    emitted_duration: float,
    chunk_seconds: float,
    on_chunk: ChunkCallback,
) -> float:
    """Emit every complete, not-yet-processed chunk in one audio snapshot."""
    while duration >= emitted_duration + chunk_seconds:
        await on_chunk(snapshot, emitted_duration)
        emitted_duration += chunk_seconds
    return emitted_duration


class MediaAcquirer:
    def __init__(
        self,
        settings: Settings,
        work_dir: Path,
        db: JobDatabase,
        runner: CommandRunner | None = None,
        on_progress: ProgressCallback | None = None,
        on_warning: WarningCallback | None = None,
    ):
        self.settings = settings
        self.work_dir = work_dir
        self.db = db
        self.runner = runner or CommandRunner()
        self.on_progress = on_progress
        self.on_warning = on_warning

    async def _progress(self, fraction: float) -> None:
        if self.on_progress:
            await self.on_progress(max(0.0, min(1.0, fraction)))

    async def _download_log(self, stream: str, line: str) -> None:
        self.db.log("yt-dlp", stream, line)
        match = DOWNLOAD_PROGRESS_RE.search(line)
        if match:
            # Reserve the tail for audio extraction/finalization.
            await self._progress(float(match.group("percent")) / 100 * 0.85)

    async def acquire(self, source: MediaSource) -> MediaArtifact:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if source.is_url:
            media_path = await self._download(source)
            persistent = self.settings.download_dir is not None
        else:
            media_path = Path(source.value).expanduser().resolve()
            if not media_path.is_file():
                raise FileNotFoundError(media_path)
            persistent = True
            await self._progress(0.2)
        audio_path = self.work_dir / "reference.wav"
        await self.extract_audio(media_path, audio_path)
        await self._progress(0.95)
        artifact = MediaArtifact(source, audio_path, media_path, persistent)
        self.db.add_artifact("reference_audio", audio_path)
        self.db.add_artifact("source_media", media_path)
        return artifact

    async def _download(self, source: MediaSource) -> Path:
        target_dir = (self.settings.download_dir or self.work_dir / "download").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        args = [self.settings.yt_dlp, "--newline", "--no-playlist", "--print", "after_move:filepath", "-P", target_dir]
        if self.settings.download_dir is None:
            args += ["-f", "bestaudio/best", "-o", "source.%(ext)s"]
        else:
            # yt-dlp's normal best format and merge logic preserves the preferred
            # container and falls back to mkv when needed.
            args += ["--merge-output-format", "mkv"]
        args.extend(self.settings.yt_dlp_options)
        args.append(source.value)

        started = time()
        result = await self.runner.run(args, on_line=self._download_log)
        candidates = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        for candidate in reversed(candidates):
            if candidate.is_file():
                return candidate.resolve()
        # A configured download_dir is shared with the user's own downloads, so
        # only files this run wrote may stand in for the printed path.
        files = [
            p for p in target_dir.iterdir()
            if p.is_file()
            and not p.name.endswith((".part", ".ytdl"))
            and p.stat().st_mtime >= started - 1
        ]
        if not files:
            raise RuntimeError("yt-dlp completed without producing a media file")
        return max(files, key=lambda p: p.stat().st_mtime).resolve()

    async def extract_audio(self, media_path: Path, output: Path) -> None:
        args = [self.settings.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        args += ["-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output]
        await self.runner.run(args, on_line=lambda stream, line: self.db.log("ffmpeg", stream, line))

    async def acquire_stream(self, source: MediaSource, on_chunk: ChunkCallback) -> MediaArtifact:
        """Download once and expose growing audio snapshots to a streaming transcriber.

        Snapshot extraction is opportunistic: formats that ffmpeg cannot read while
        growing simply become available at the next interval or finalization.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if not source.is_url:
            return await self.acquire(source)
        target_dir = (self.settings.download_dir or self.work_dir / "download").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        download_stem = f"source-{uuid.uuid4().hex}"
        template = target_dir / f"{download_stem}.%(ext)s"
        args = [
            self.settings.yt_dlp, "--newline", "--no-playlist",
            "--print", "after_move:filepath", "-o", template,
        ]
        if self.settings.download_dir is None:
            args += ["-f", "bestaudio/best"]
        else:
            args += ["--merge-output-format", "mkv"]
        # A live service can expose a single fragmented MP4 format (``fmp4``).
        # ``--merge-output-format`` is ignored when no merge is needed, and
        # recent yt-dlp versions reject that unusual extension for safety.
        # Remuxing gives yt-dlp a safe, stable final extension in both cases.
        args += ["--remux-video", "mkv"]
        args.extend(self.settings.yt_dlp_options)
        args.append(source.value)
        task = asyncio.create_task(self.runner.run(args, on_line=self._download_log))
        result = None
        interrupted = False
        emitted_duration = 0.0
        snapshot_index = 0
        try:
            while not task.done():
                await asyncio.sleep(1)
                sized = _sizes(target_dir.glob(f"{download_stem}.*"))
                if not sized or snapshot_index and snapshot_index % self.settings.stream_chunk_seconds:
                    snapshot_index += 1
                    continue
                growing = max(sized, key=lambda item: item[1])[0]
                snapshot = self.work_dir / "stream-snapshot.wav"
                try:
                    await self.extract_audio(growing, snapshot)
                    duration = await asyncio.to_thread(pcm_audio_duration, snapshot)
                    if duration is None:
                        snapshot_index += 1
                        continue
                    emitted_duration = await _drain_stream_chunks(
                        snapshot,
                        duration,
                        emitted_duration,
                        self.settings.stream_chunk_seconds,
                        on_chunk,
                    )
                except Exception as exc:
                    # A snapshot that cannot be read yet is expected, but a
                    # failing transcriber or Agent would otherwise repeat
                    # silently for the whole download.
                    self.db.log("stream-extract", "stderr", str(exc))
                    if self.on_warning:
                        await self.on_warning(f"stream chunk skipped: {exc}")
                snapshot_index += 1
            result = await task
        except asyncio.CancelledError:
            interrupted = True
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if result is not None:
            printed = [
                Path(line.strip()) for line in result.stdout.splitlines()
                if line.strip()
            ]
            files = [
                path for path in printed
                if path.is_file() and path.parent.resolve() == target_dir
                and path.name.startswith(f"{download_stem}.")
            ]
        else:
            files = []
        if not files:
            files = [
                p for p in target_dir.glob(f"{download_stem}.*")
                if p.is_file() and not p.name.endswith((".part", ".ytdl"))
            ]
        sized = _sizes(files) if files else []
        if not sized:
            # A graceful first interrupt may leave a useful partial file.
            sized = _sizes(target_dir.glob(f"{download_stem}.*"))
        if not sized:
            if interrupted:
                # Nothing had been written when the interrupt arrived. Report
                # the interruption so the job is preserved as resumable rather
                # than recorded as a hard failure.
                raise asyncio.CancelledError
            raise RuntimeError("stream ended without usable media")
        media_path = max(sized, key=lambda item: item[1])[0]
        audio_path = self.work_dir / "reference.wav"
        await self.extract_audio(media_path, audio_path)
        self.db.add_artifact("reference_audio", audio_path)
        self.db.add_artifact("source_media", media_path)
        return MediaArtifact(source, audio_path, media_path, self.settings.download_dir is not None)


def default_output_base(artifact: MediaArtifact, settings: Settings, work_dir: Path) -> Path:
    if settings.output_dir:
        return settings.output_dir / (artifact.media_path.stem if artifact.media_path else "subtitle")
    if artifact.persistent and artifact.media_path:
        return artifact.media_path.parent / artifact.media_path.stem
    return Path.cwd() / (artifact.media_path.stem if artifact.media_path else "subtitle")
