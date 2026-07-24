import asyncio
from pathlib import Path

from yakiflow.config import Settings
from yakiflow.database import JobDatabase
from yakiflow.media import MediaAcquirer, MediaSource, _drain_stream_chunks
from yakiflow.process import ProcessResult


class FakeRunner:
    def __init__(self, media: Path):
        self.media = media
        self.calls = []

    async def run(self, args, **kwargs):
        self.calls.append([str(value) for value in args])
        if args[0] == "yt-dlp":
            on_line = kwargs.get("on_line")
            if on_line:
                result = on_line("stderr", "[download]  50.0% of 10.00MiB")
                if asyncio.iscoroutine(result):
                    await result
            self.media.parent.mkdir(parents=True, exist_ok=True)
            self.media.write_bytes(b"media")
            return ProcessResult(tuple(map(str, args)), 0, str(self.media) + "\n", "")
        Path(args[-1]).write_bytes(b"wav")
        return ProcessResult(tuple(map(str, args)), 0, "", "")


def test_media_source_preserves_urls_and_canonicalizes_local_paths(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert MediaSource.parse("clip.mp4") == MediaSource(
        str(tmp_path / "clip.mp4"), False
    )
    assert MediaSource.parse("https://example.test/live") == MediaSource(
        "https://example.test/live", True
    )


def test_persistent_url_downloads_once_then_extracts(tmp_path: Path) -> None:
    media = tmp_path / "downloads" / "video.mkv"
    runner = FakeRunner(media)
    settings = Settings(download_dir=media.parent, ffmpeg="ffmpeg", yt_dlp="yt-dlp")
    db = JobDatabase(tmp_path / "db.sqlite3")
    progress: list[float] = []

    async def on_progress(fraction: float) -> None:
        progress.append(fraction)

    artifact = asyncio.run(
        MediaAcquirer(
            settings, tmp_path / "work", db, runner, on_progress=on_progress
        ).acquire(MediaSource.parse("https://example.test/v"))
    )
    assert artifact.media_path == media.resolve()
    assert sum(call[0] == "yt-dlp" for call in runner.calls) == 1
    assert sum(call[0] == "ffmpeg" for call in runner.calls) == 1
    assert progress == [0.425, 0.95]
    db.close()


def test_stream_drain_advances_only_by_processed_chunks(tmp_path: Path) -> None:
    starts: list[float] = []

    async def on_chunk(_snapshot: Path, start: float) -> None:
        starts.append(start)

    emitted = asyncio.run(
        _drain_stream_chunks(tmp_path / "snapshot.wav", 47, 0, 15, on_chunk)
    )

    assert starts == [0, 15, 30]
    assert emitted == 45


def test_stream_download_ignores_files_from_previous_jobs(
    tmp_path: Path, monkeypatch
) -> None:
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    stale = download_dir / "source.webm"
    stale.write_bytes(b"old" * 10_000)

    class StreamRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []
            self.downloaded: Path | None = None

        async def run(self, args, **kwargs):
            values = [str(value) for value in args]
            self.calls.append(values)
            if values[0] == "yt-dlp":
                template = values[values.index("-o") + 1]
                self.downloaded = Path(template.replace("%(ext)s", "mkv"))
                self.downloaded.write_bytes(b"current")
                return ProcessResult(tuple(values), 0, f"{self.downloaded}\n", "")
            Path(values[-1]).write_bytes(b"wav")
            return ProcessResult(tuple(values), 0, "", "")

    async def no_complete_chunk(_self, _audio: Path) -> float:
        return 0

    monkeypatch.setattr(MediaAcquirer, "_duration", no_complete_chunk)
    runner = StreamRunner()
    db = JobDatabase(tmp_path / "db.sqlite3")
    artifact = asyncio.run(
        MediaAcquirer(
            Settings(download_dir=download_dir, ffmpeg="ffmpeg", yt_dlp="yt-dlp"),
            tmp_path / "work",
            db,
            runner,
        ).acquire_stream(MediaSource.parse("https://example.test/current"), lambda *_: None)
    )

    assert artifact.media_path == runner.downloaded
    assert artifact.media_path != stale
    yt_dlp_call = next(call for call in runner.calls if call[0] == "yt-dlp")
    assert Path(yt_dlp_call[yt_dlp_call.index("-o") + 1]).name.startswith(
        "source-"
    )
    db.close()
