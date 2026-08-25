import subprocess
from pathlib import Path

import pytest

from yakiflow.config import Settings
import yakiflow.media_player as media_player
from yakiflow.media_player import media_open_argv


def test_default_mpv_command_forces_explicit_subtitle_visibility() -> None:
    media = Path("movie.mkv")
    subtitle = Path("movie.srt")

    expected = [
        "mpv",
        "--no-sub-auto",
        "--no-resume-playback",
        f"--sub-file={subtitle}",
        "--sid=auto",
        "--sub-visibility=yes",
        str(media),
    ]
    assert media_open_argv(None, media, subtitle) == expected
    assert media_open_argv(Settings().review.video_open_command, media, subtitle) == expected


def test_open_media_does_not_inherit_tui_output_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "movie.mkv"
    subtitle = tmp_path / "movie.srt"
    media.write_bytes(b"media")
    subtitle.write_text("", encoding="utf-8")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_popen(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(media_player.subprocess, "Popen", fake_popen)

    assert media_player.open_media(None, media, subtitle, cwd=tmp_path)
    assert len(calls) == 1
    _args, kwargs = calls[0]
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
