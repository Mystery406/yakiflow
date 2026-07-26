"""Launch the configured media player for a subtitle preview."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


DEFAULT_VIDEO_OPEN_COMMAND = (
    "mpv --no-sub-auto --no-resume-playback "
    "--sub-file={subtitle} --sid=auto --sub-visibility=yes {file}"
)


def media_open_argv(
    command: str | None,
    media_path: Path,
    subtitle_path: Path,
) -> list[str]:
    """Expand a media-player command without invoking a shell.

    ``{file}`` (or the equivalent ``{media}``) and ``{subtitle}`` are replaced by shell-quoted paths before
    splitting, which keeps paths containing spaces as one argv element.  As
    with the review-file command, omitted placeholders are filled in: the
    media path is appended and a standard mpv-compatible subtitle option is
    added when the command did not mention a subtitle placeholder.
    """
    text = command or DEFAULT_VIDEO_OPEN_COMMAND
    has_file = "{file}" in text or "{media}" in text
    has_subtitle = "{subtitle}" in text
    text = text.replace("{file}", shlex.quote(str(media_path)))
    text = text.replace("{media}", shlex.quote(str(media_path)))
    text = text.replace("{subtitle}", shlex.quote(str(subtitle_path)))
    argv = shlex.split(text)
    if not has_subtitle:
        argv.extend(("--sub-file", str(subtitle_path)))
    if not has_file:
        argv.append(str(media_path))
    return argv


def open_media(
    command: str | None,
    media_path: Path | None,
    subtitle_path: Path | None,
    *,
    cwd: Path | None = None,
) -> bool:
    """Start a player in the background, returning false for unavailable media/player."""
    if (
        media_path is None
        or subtitle_path is None
        or not media_path.is_file()
        or not subtitle_path.is_file()
    ):
        return False
    try:
        subprocess.Popen(
            media_open_argv(command, media_path, subtitle_path),
            cwd=cwd,
            # The player is a background convenience process.  Inheriting
            # the TUI's terminal streams lets mpv status/log output overwrite
            # Textual's rendering.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, IndexError, ValueError):
        return False
    return True
