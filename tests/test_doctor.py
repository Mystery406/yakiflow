from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yakiflow.doctor as doctor_module
from yakiflow.config import Settings
from yakiflow.doctor import run_doctor


def test_doctor_checks_whisperx_package_and_explicit_cuda(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "whisper.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(doctor_module.shutil, "which", lambda command: f"/bin/{command}")
    monkeypatch.setattr(
        doctor_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="authenticated", stderr=""),
    )
    monkeypatch.setattr(doctor_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    checks = run_doctor(
        Settings(
            whisper_model=model,
            translation_backend="codex",
            alignment_backend="whisperx",
            alignment_device="cuda",
        )
    )
    mapped = {check.name: check for check in checks}

    assert mapped["WhisperX"].ok
    assert not mapped["WhisperX CUDA"].ok
    assert mapped["WhisperX CUDA"].detail == "CUDA unavailable"


def test_doctor_does_not_require_whisperx_for_default_vad(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "whisper.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(doctor_module.shutil, "which", lambda command: f"/bin/{command}")
    monkeypatch.setattr(
        doctor_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )

    checks = run_doctor(Settings(whisper_model=model, translation_backend="codex"))

    assert all(not check.name.startswith("WhisperX") for check in checks)


def test_doctor_names_both_whisperx_install_variants(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "whisper.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(doctor_module.shutil, "which", lambda command: f"/bin/{command}")
    monkeypatch.setattr(
        doctor_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    monkeypatch.setattr(doctor_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    checks = run_doctor(
        Settings(
            whisper_model=model,
            translation_backend="codex",
            alignment_backend="whisperx",
        )
    )
    whisperx = next(check for check in checks if check.name == "WhisperX")

    assert not whisperx.ok
    assert "yakiflow[whisperx-cpu]" in whisperx.detail
    assert "yakiflow[whisperx-cuda]" in whisperx.detail


def test_doctor_checks_both_review_display_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "whisper.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(doctor_module.shutil, "which", lambda command: f"/bin/{command}")
    monkeypatch.setattr(
        doctor_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )

    checks = run_doctor(
        Settings(
            whisper_model=model,
            translation_backend="codex",
            review_display_mode="both",
            review_open_command="editor {srt}",
        )
    )
    mapped = {check.name: check for check in checks}

    assert mapped["tmux"].ok
    assert mapped["review open command"].ok
