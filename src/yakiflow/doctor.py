from __future__ import annotations

import shutil
import shlex
import subprocess
import importlib.util
from dataclasses import dataclass

from .config import Settings


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    commands = [
        ("ffmpeg", settings.ffmpeg),
        ("yt-dlp", settings.yt_dlp),
        ("whisper-cli", settings.whisper_cli),
        ("whisper-server", settings.whisper_server),
    ]
    for label, command in commands:
        path = shutil.which(command)
        if label == "whisper-server" and not path:
            checks.append(
                Check(
                    label,
                    not settings.stream,
                    "not found (only required for streaming runs)",
                )
            )
        else:
            checks.append(Check(label, bool(path), path or "not found"))
    model = settings.whisper_model
    checks.append(Check("whisper model", bool(model and model.is_file()), str(model)))
    if settings.vad_model:
        checks.append(
            Check("whisper VAD model", settings.vad_model.is_file(), str(settings.vad_model))
        )
    if settings.alignment_backend == "whisperx":
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
        torch_detail = "PyTorch unavailable"
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            torch_detail = "CUDA available" if cuda_available else "CUDA unavailable"
        except (ImportError, RuntimeError) as exc:
            torch_detail = str(exc)
        if settings.alignment_device == "cuda":
            checks.append(Check("WhisperX CUDA", cuda_available, torch_detail))
        else:
            selected = "cuda" if settings.alignment_device == "auto" and cuda_available else "cpu"
            checks.append(Check("WhisperX device", True, f"{selected} ({torch_detail})"))
    backend = settings.translation_backend
    for name in ([backend] if backend else ["codex", "claude"]):
        if not name:
            continue
        executable = shutil.which(name)
        if not executable:
            checks.append(Check(f"{name} CLI", False, "not found"))
            continue
        auth_command = (
            [executable, "login", "status"]
            if name == "codex"
            else [executable, "auth", "status"]
        )
        try:
            result = subprocess.run(auth_command, capture_output=True, text=True, timeout=10)
            detail = (result.stdout or result.stderr).strip().splitlines()
            checks.append(Check(f"{name} auth", result.returncode == 0, detail[-1] if detail else f"exit {result.returncode}"))
        except (OSError, subprocess.SubprocessError) as exc:
            checks.append(Check(f"{name} auth", False, str(exc)))
    if settings.review_display_mode in {"split", "both"}:
        tmux = shutil.which("tmux")
        checks.append(Check("tmux", bool(tmux), tmux or "not found (split review will have no preview)"))
    if settings.review_display_mode in {"open", "both"}:
        command = settings.review_open_command or ""
        try:
            executable = shlex.split(command)[0]
        except IndexError:
            executable = ""
        path = shutil.which(executable) if executable else None
        checks.append(Check("review open command", bool(path), path or "not found"))
    return checks
