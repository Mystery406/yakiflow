from __future__ import annotations

import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding

from .media_player import open_media
from .models import Cue
from .srt import parse_srt_blocks
from .tui import SubtitleTable


def read_srt(path: Path) -> list[Cue]:
    """Read a subtitle file for preview, skipping any block that cannot parse."""
    if not path.is_file():
        return []
    blocks, _problems = parse_srt_blocks(
        path.read_text(encoding="utf-8", errors="replace")
    )
    cues: list[Cue] = []
    for block in blocks:
        first, *rest = block.text
        translated = first if rest else None
        source = "\n".join(rest) if rest else first
        cues.append(Cue(block.number, block.start, block.end, source, translated))
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
