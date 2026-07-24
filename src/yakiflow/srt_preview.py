from __future__ import annotations

import re
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding

from .media_player import open_media
from .models import Cue
from .tui import SubtitleTable


_TIMING = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


def _seconds(value: str) -> float:
    hours, minutes, rest = value.replace(",", ".").split(":")
    seconds, fraction = rest.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + float(f"0.{fraction}")


def read_srt(path: Path) -> list[Cue]:
    if not path.is_file():
        return []
    blocks = path.read_text(encoding="utf-8", errors="replace").split("\n\n")
    cues: list[Cue] = []
    for block in blocks:
        lines = [line.rstrip("\r") for line in block.splitlines()]
        if len(lines) < 3:
            continue
        match = _TIMING.match(lines[1])
        if not match:
            continue
        text = [line for line in lines[2:] if line]
        if len(text) > 1:
            translated, source = text[0], "\n".join(text[1:])
        else:
            translated, source = None, (text[0] if text else "")
        cues.append(
            Cue(
                lines[0],
                _seconds(match.group("start")),
                _seconds(match.group("end")),
                source,
                translated,
            )
        )
    return cues


class SrtPreviewApp(App[None]):
    CSS = """
    Screen { padding: 0; }
    #recent { width: 1fr; height: 1fr; }
    """

    BINDINGS = [
        Binding("o", "open_media", "Open media", priority=True),
    ]

    def __init__(
        self,
        path: Path,
        marker: Path,
        media_path: Path | None = None,
        video_open_command: str | None = None,
    ):
        super().__init__()
        self.path = path
        self.marker = marker
        self.media_path = media_path
        self.video_open_command = video_open_command
        self._signature: tuple[int, int] | None = None

    def compose(self) -> ComposeResult:
        table = SubtitleTable(id="recent")
        table.border_title = "Live subtitles"
        table.border_subtitle = "↑/↓ · PgUp/PgDn · ^Home/^End"
        yield table

    def action_open_media(self) -> None:
        if open_media(
            self.video_open_command,
            self.media_path,
            self.path,
            cwd=self.path.parent,
        ):
            self.notify("Opened media with subtitles.")
        else:
            self.notify("Media or subtitles are not ready (or mpv is unavailable).")

    def on_mount(self) -> None:
        self.set_interval(0.4, self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        if self.marker.exists():
            self.exit()
            return
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        if signature == self._signature:
            return
        self._signature = signature
        table = self.query_one("#recent", SubtitleTable)
        previous_y = table.scroll_y
        follow_tail = table.follow_tail
        table.clear()
        for cue in read_srt(self.path):
            table.add_subtitle(cue)
        # Rebuilding the DataTable resets its virtual scroll extent. Keep
        # the user's viewport stable across Agent edits, while preserving
        # the existing live-tail behavior for users already at the end.
        table.follow_tail = follow_tail
        table.restore_scroll_after_update(previous_y)
def main() -> int:
    if len(sys.argv) != 5:
        return 2
    media = Path(sys.argv[3]) if sys.argv[3] else None
    command = sys.argv[4] or None
    SrtPreviewApp(Path(sys.argv[1]), Path(sys.argv[2]), media, command).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
