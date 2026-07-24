from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MemoryFileSnapshot:
    """The content state used to detect concurrent destination edits."""

    exists: bool
    content: str = ""

    @classmethod
    def read(cls, path: Path) -> MemoryFileSnapshot:
        try:
            return cls(True, path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(False)

    @classmethod
    def from_checkpoint(cls, value: object) -> MemoryFileSnapshot | None:
        if not isinstance(value, dict) or not isinstance(value.get("exists"), bool):
            return None
        content = value.get("content", "")
        if not isinstance(content, str):
            return None
        return cls(value["exists"], content)

    def checkpoint(self) -> dict[str, object]:
        return {"exists": self.exists, "content": self.content}


class MemoryDestinationConflict(RuntimeError):
    """The durable memory changed after it was staged in the work directory."""

    def __init__(
        self,
        destination: Path,
        previous: MemoryFileSnapshot,
        current: MemoryFileSnapshot,
    ) -> None:
        super().__init__(f"memory destination changed while the job was running: {destination}")
        self.destination = destination
        self.previous = previous
        self.current = current

    def diff(self) -> str:
        old = self.previous.content.splitlines(keepends=True) if self.previous.exists else []
        new = self.current.content.splitlines(keepends=True) if self.current.exists else []
        rendered = "".join(
            unified_diff(
                old,
                new,
                fromfile=f"{self.destination} (previous)",
                tofile=f"{self.destination} (current)",
            )
        )
        # An empty file being created or removed has no lines for difflib to
        # render, but it is still a meaningful destination state change.
        if rendered:
            return rendered
        state = "created as an empty file" if self.current.exists else "removed"
        return f"{self.destination}: {state}\n"


class MemoryStore:
    """The staged Markdown memory exposed to translation and review Agents."""

    def __init__(self, path: Path):
        self.path = path

    def read(self) -> str:
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")
