from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yakiflow.doctor as doctor_module
from conftest import make_settings
from yakiflow.doctor import run_doctor


def _stub_dependencies(tmp_path: Path, monkeypatch) -> Path:
    """Create a Whisper model and stub out CLI discovery and auth checks."""
    model = tmp_path / "whisper.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(doctor_module.shutil, "which", lambda command: f"/bin/{command}")
    monkeypatch.setattr(
        doctor_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    return model


def test_doctor_checks_whisperx_package_and_explicit_cuda(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(doctor_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    checks = run_doctor(
        make_settings(
            whisper={"model": model},
            agent={"backend": "codex"},
            alignment={"backend": "whisperx", "device": "cuda"},
        ).resolved()
    )
    mapped = {check.name: check for check in checks}

    assert mapped["WhisperX"].ok
    assert not mapped["WhisperX CUDA"].ok
    assert mapped["WhisperX CUDA"].detail == "CUDA unavailable"


def test_doctor_does_not_require_whisperx_for_default_vad(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)

    checks = run_doctor(
        make_settings(whisper={"model": model}, agent={"backend": "codex"}).resolved()
    )

    assert all(not check.name.startswith("WhisperX") for check in checks)


def test_doctor_names_both_whisperx_install_variants(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(doctor_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    checks = run_doctor(
        make_settings(
            whisper={"model": model},
            agent={"backend": "codex"},
            alignment={"backend": "whisperx"},
        ).resolved()
    )
    whisperx = next(check for check in checks if check.name == "WhisperX")

    assert not whisperx.ok
    assert "yakiflow[whisperx-cpu]" in whisperx.detail
    assert "yakiflow[whisperx-cuda]" in whisperx.detail


def test_doctor_checks_both_review_display_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    model = _stub_dependencies(tmp_path, monkeypatch)

    checks = run_doctor(
        make_settings(
            whisper={"model": model},
            agent={"backend": "codex"},
            review={"display_mode": "both", "open_command": "editor {srt}"},
        ).resolved()
    )
    mapped = {check.name: check for check in checks}

    assert mapped["tmux"].ok
    assert mapped["review open command"].ok
