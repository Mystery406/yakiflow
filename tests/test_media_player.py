from pathlib import Path

from yakiflow.config import Settings
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
    assert media_open_argv(Settings().video_open_command, media, subtitle) == expected
