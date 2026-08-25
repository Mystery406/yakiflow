from __future__ import annotations

import shutil
import shlex
import subprocess
import importlib.util
from dataclasses import dataclass
from urllib.parse import urlparse

from .config import Settings


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def _which(checks: list[Check], label: str, command: str) -> None:
    path = shutil.which(command)
    checks.append(Check(label, bool(path), path or "not found"))


def _probe_server_url(url: str, timeout: float = 5.0) -> Check:
    import http.client

    parsed = urlparse(url)
    connection_class = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    try:
        connection = connection_class(parsed.netloc, timeout=timeout)
        try:
            connection.request("OPTIONS", parsed.path or "/")
            response = connection.getresponse()
            response.read()
        finally:
            connection.close()
    except OSError as exc:
        return Check("whisper-server URL", False, f"{url}: {exc}")
    return Check(
        "whisper-server URL", True, f"{url} responded with HTTP {response.status}"
    )


def run_doctor(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    _which(checks, "ffmpeg", settings.commands.ffmpeg)
    _which(checks, "yt-dlp", settings.commands.yt_dlp)
    backend = settings.transcription.backend
    if backend == "whisper-cli":
        _which(checks, "whisper-cli", settings.whisper.cli)
    elif backend == "whisper-server":
        if settings.whisper.server_url:
            checks.append(_probe_server_url(settings.whisper.server_url))
        else:
            _which(checks, "whisper-server", settings.whisper.server)
    if backend in {"whisper-cli", "whisper-server"}:
        model = settings.whisper.model
        checks.append(
            Check("whisper model", bool(model and model.is_file()), str(model))
        )
        if settings.whisper.vad_model:
            checks.append(
                Check(
                    "whisper VAD model",
                    settings.whisper.vad_model.is_file(),
                    str(settings.whisper.vad_model),
                )
            )
    else:
        sdk_installed = importlib.util.find_spec("elevenlabs") is not None
        checks.append(
            Check(
                "elevenlabs SDK",
                sdk_installed,
                "installed" if sdk_installed else
                "not installed; install yakiflow[elevenlabs]",
            )
        )
        from .elevenlabs import elevenlabs_api_key_source

        # Only where the key would come from is reported, never the key.
        source = elevenlabs_api_key_source(settings)
        checks.append(Check("elevenlabs API key", source != "not set", source))
    if settings.alignment.backend == "whisperx":
        whisperx_installed = importlib.util.find_spec("whisperx") is not None
        checks.append(
            Check(
                "WhisperX",
                whisperx_installed,
                (
                    "installed"
                    if whisperx_installed
                    else "not installed; install yakiflow[whisperx-cpu] or yakiflow[whisperx-cuda]"
                ),
            )
        )
        cuda_available = False
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            torch_detail = "CUDA available" if cuda_available else "CUDA unavailable"
        except (ImportError, RuntimeError) as exc:
            torch_detail = str(exc)
        if settings.alignment.device == "cuda":
            checks.append(Check("WhisperX CUDA", cuda_available, torch_detail))
        else:
            selected = "cuda" if settings.alignment.device == "auto" and cuda_available else "cpu"
            checks.append(Check("WhisperX device", True, f"{selected} ({torch_detail})"))
    agent_backends = {
        stage.backend
        for stage in (settings.agent.draft, settings.agent.final)
        if stage.backend
    }
    for name in sorted(agent_backends) or ["codex", "claude"]:
        executable = shutil.which(name)
        if not executable:
            checks.append(Check(f"{name} CLI", False, "not found"))
            continue
        auth_command = [name, "login", "status"] if name == "codex" else [name, "auth", "status"]
        try:
            result = subprocess.run(
                auth_command,
                capture_output=True,
                text=True,
                # The locale encoding would raise on a non-ASCII account name
                # under LC_ALL=C and abort every remaining check.
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            detail = (result.stdout or result.stderr).strip().splitlines()
            checks.append(Check(f"{name} auth", result.returncode == 0, detail[-1] if detail else f"exit {result.returncode}"))
        except (OSError, subprocess.SubprocessError) as exc:
            checks.append(Check(f"{name} auth", False, str(exc)))
    if settings.review.display_mode in {"split", "both"}:
        tmux = shutil.which("tmux")
        checks.append(Check("tmux", bool(tmux), tmux or "not found (split review will have no preview)"))
    if settings.review.display_mode in {"open", "both"}:
        command = settings.review.open_command or ""
        try:
            executable = shlex.split(command)[0]
        except IndexError:
            executable = ""
        path = shutil.which(executable) if executable else None
        checks.append(Check("review open command", bool(path), path or "not found"))
    return checks
