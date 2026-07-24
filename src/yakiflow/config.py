from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from platformdirs import user_cache_path, user_config_path


DEFAULT_MODEL_NAME = "ggml-large-v3-turbo-q5_0.bin"


def default_model_path() -> Path:
    return user_cache_path("yakiflow") / "models" / DEFAULT_MODEL_NAME


@dataclass(frozen=True, slots=True)
class Settings:
    source_language: str | None = None
    target_language: str | None = None
    whisper_model: Path | None = None
    vad_model: Path | None = None
    alignment_backend: str = "vad"
    alignment_device: str = "auto"
    alignment_model: str | None = None
    translation_backend: str | None = None
    stream: bool = False
    download_dir: Path | None = None
    output_dir: Path | None = None
    output_mode: str = "bilingual"
    memory: Path | None = None
    context_files: tuple[Path, ...] = ()
    work_dir: Path | None = None
    keep_workdir: bool = False
    draft_model: str | None = None
    draft_effort: str = "low"
    final_model: str | None = None
    final_effort: str = "high"
    agent_workers: int = 4
    translation_batch_size: int = 20
    translation_context: int = 10
    draft_agent_timeout_seconds: float = 600
    agent_max_attempts: int = 3
    agent_retry_delay_seconds: float = 1
    review_display_mode: str = "split"
    review_open_command: str | None = None
    video_open_command: str = (
        "mpv --no-sub-auto --no-resume-playback "
        "--sub-file={subtitle} --sid=auto --sub-visibility=yes {file}"
    )
    stream_chunk_seconds: int = 15
    stream_context_seconds: int = 5
    whisper_cli: str = "whisper-cli"
    whisper_server: str = "whisper-server"
    ffmpeg: str = "ffmpeg"
    yt_dlp: str = "yt-dlp"

    def resolved(self) -> Settings:
        def absolute(path: Path | None) -> Path | None:
            return path.expanduser().resolve() if path is not None else None

        memory = absolute(self.memory or user_config_path("yakiflow") / "memory.md")
        model = absolute(self.whisper_model or default_model_path())
        draft = self.draft_model
        final = self.final_model
        if self.translation_backend == "codex":
            draft = draft or "gpt-5.6-terra"
            final = final or "gpt-5.6-sol"
        elif self.translation_backend == "claude":
            draft = draft or "sonnet"
            final = final or "opus"
        return replace(
            self,
            memory=memory,
            whisper_model=model,
            vad_model=absolute(self.vad_model),
            download_dir=absolute(self.download_dir),
            output_dir=absolute(self.output_dir),
            context_files=tuple(
                path.expanduser().resolve() for path in self.context_files
            ),
            work_dir=absolute(self.work_dir),
            draft_model=draft,
            final_model=final,
        )


PATH_FIELDS = {
    "whisper_model",
    "vad_model",
    "download_dir",
    "output_dir",
    "memory",
    "work_dir",
}

PATH_SEQUENCE_FIELDS = {"context_files"}


def _read_toml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    section = data.get("yakiflow", data)
    if not isinstance(section, dict):
        raise ValueError(f"expected a table in {path}")
    return section


def _clean(values: Mapping[str, Any]) -> dict[str, Any]:
    valid = {f.name for f in fields(Settings)}
    result: dict[str, Any] = {}
    for key, value in values.items():
        key = key.replace("-", "_")
        if key not in valid or value is None:
            continue
        if key in PATH_FIELDS:
            result[key] = Path(value).expanduser()
        elif key in PATH_SEQUENCE_FIELDS:
            values = (
                value
                if isinstance(value, Sequence) and not isinstance(value, str)
                else (value,)
            )
            result[key] = tuple(Path(item).expanduser() for item in values)
        else:
            result[key] = value
    return result


def load_settings(
    cli: Mapping[str, Any] | None = None,
    *,
    project_file: Path | None = None,
    user_file: Path | None = None,
) -> Settings:
    """Merge defaults < user config < project config < explicit CLI values."""
    user_file = user_file or user_config_path("yakiflow") / "config.toml"
    if project_file is None:
        candidate = Path.cwd() / "yakiflow.toml"
        project_file = candidate if candidate.exists() else None
    merged = asdict(Settings())
    merged.update(_clean(_read_toml(user_file)))
    merged.update(_clean(_read_toml(project_file)))
    merged.update(_clean(cli or {}))
    return Settings(**merged).resolved()


def validate_run_settings(settings: Settings) -> None:
    if not settings.target_language:
        raise ValueError("--target-language is required")
    if not settings.source_language:
        raise ValueError("--source-language is required (use 'auto' for automatic detection)")
    if settings.translation_backend not in {"codex", "claude"}:
        raise ValueError("--translation-backend must be codex or claude")
    if settings.output_mode not in {"source", "translated", "bilingual", "all"}:
        raise ValueError("invalid --output-mode")
    if settings.alignment_backend not in {"vad", "whisperx"}:
        raise ValueError("--alignment-backend must be vad or whisperx")
    if settings.alignment_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("--alignment-device must be auto, cpu, or cuda")
    if settings.review_display_mode not in {"split", "open"}:
        raise ValueError("invalid --review-display-mode")
    if settings.review_display_mode == "open" and not settings.review_open_command:
        raise ValueError("--review-open-command is required when review display mode is open")
    if settings.agent_workers < 1 or settings.translation_batch_size < 1:
        raise ValueError("agent worker and batch counts must be positive")
    if settings.translation_context < 0:
        raise ValueError("translation context cannot be negative")
    if settings.draft_agent_timeout_seconds <= 0:
        raise ValueError("agent timeouts must be positive")
    if settings.agent_max_attempts < 1:
        raise ValueError("agent attempt count must be positive")
    if settings.agent_retry_delay_seconds < 0:
        raise ValueError("agent retry delay cannot be negative")
    if settings.stream_chunk_seconds <= 0:
        raise ValueError("stream chunk duration must be positive")
    if settings.stream_context_seconds < 0:
        raise ValueError("stream context duration cannot be negative")
