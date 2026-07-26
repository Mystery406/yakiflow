from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import yakiflow.cli as cli
import yakiflow.tui as tui
from yakiflow.config import Settings
from yakiflow.job import YakiFlowJob
from yakiflow.process import ProcessResult
from yakiflow.translation import AgentBackend


class TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_alignment_cli_options_are_loaded(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    args = cli._parser().parse_args([
        "run",
        "input.mp4",
        "--alignment-backend",
        "whisperx",
        "--alignment-device",
        "cpu",
        "--alignment-model",
        "custom/model",
    ])

    settings = cli._settings(args)
    assert settings.alignment_backend == "whisperx"
    assert settings.alignment_device == "cpu"
    assert settings.alignment_model == "custom/model"


def test_auto_open_video_cli_option_is_loaded(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    args = cli._parser().parse_args([
        "run",
        "input.mp4",
        "--auto-open-video",
    ])

    settings = cli._settings(args)

    assert settings.auto_open_video is True


def test_context_file_cli_option_is_repeatable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    args = cli._parser().parse_args([
        "run",
        "input.mp4",
        "--context-file",
        "chat.xml",
        "--context-file",
        "notes.txt",
    ])

    settings = cli._settings(args)

    assert settings.context_files == (
        tmp_path / "chat.xml",
        tmp_path / "notes.txt",
    )


def test_profile_option_is_available_on_config_loading_commands(monkeypatch) -> None:
    selected: list[str | None] = []

    def fake_load_settings(_values, **kwargs):
        selected.append(kwargs.get("profile"))
        return Settings()

    monkeypatch.setattr(cli, "load_settings", fake_load_settings)
    parser = cli._parser()
    for argv in (
        ["run", "input.mp4", "--profile", "stream"],
        ["doctor", "--profile", "stream"],
        ["models", "fetch", "--profile", "stream"],
    ):
        cli._settings(parser.parse_args(argv))

    assert selected == ["stream", "stream", "stream"]


def test_tui_exit_prints_resume_guide(tmp_path: Path, monkeypatch, capsys) -> None:
    work_dir = tmp_path / "preserved job"

    class FakeJob:
        def __init__(self) -> None:
            self.work_dir = work_dir

    monkeypatch.setattr(
        cli.YakiFlowJob,
        "from_workdir",
        lambda *args, **kwargs: FakeJob(),
    )
    monkeypatch.setattr(tui, "run", lambda job: False)
    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(stdin=TTY(), stdout=TTY(), stderr=sys.stderr),
    )

    assert cli.main(["resume", str(work_dir)]) == 2
    assert capsys.readouterr().err == (
        f"Job preserved. Resume with: yakiflow resume '{work_dir}'\n"
    )


def test_memory_conflict_agent_retries_when_destination_changes_again(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "memory.md"
    work_dir = tmp_path / "work"
    destination.write_text("# Memory\n\n- original\n", encoding="utf-8")

    class UnusedBackend(AgentBackend):
        name = "unused"

        async def invoke_with_trace(self, prompt, *, model, effort, schema, on_event=None):
            raise AssertionError("structured agent should not run")

    class ConflictRunner:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run_interactive(self, args, *, cwd=None, check=True):
            assert cwd == work_dir.resolve()
            prompt = str(args[-1])
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                assert "+- destination edit 1" in prompt
                (cwd / "memory.md").write_text(
                    "# Memory\n\n- original\n- work edit\n- destination edit 1\n",
                    encoding="utf-8",
                )
                destination.write_text(
                    "# Memory\n\n- original\n- destination edit 1\n- destination edit 2\n",
                    encoding="utf-8",
                )
            else:
                assert "- destination edit 1" in prompt
                assert "+- destination edit 2" in prompt
                (cwd / "memory.md").write_text(
                    "# Memory\n\n- original\n- work edit\n"
                    "- destination edit 1\n- destination edit 2\n",
                    encoding="utf-8",
                )
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    runner = ConflictRunner()
    job = YakiFlowJob(
        "input.mp4",
        Settings(
            source_language="en",
            target_language="zh-CN",
            translation_backend="codex",
            memory=destination,
            work_dir=work_dir,
        ),
        runner=runner,
        backend=UnusedBackend(),
    )
    job.memory_path.write_text(
        "# Memory\n\n- original\n- work edit\n", encoding="utf-8"
    )
    destination.write_text(
        "# Memory\n\n- original\n- destination edit 1\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO()),
    )

    assert asyncio.run(cli._complete_job(job, [])) == 0

    assert len(runner.prompts) == 2
    assert destination.read_text(encoding="utf-8") == (
        "# Memory\n\n- original\n- work edit\n"
        "- destination edit 1\n- destination edit 2\n"
    )


def test_non_tty_run_continues_when_signal_handlers_are_unsupported(
    tmp_path: Path, monkeypatch
) -> None:
    ran = False

    class FakeJob:
        work_dir = tmp_path / "work"

        async def run(self):
            nonlocal ran
            ran = True
            return []

    async def complete(_job, outputs):
        assert outputs == []
        return 0

    async def exercise() -> int:
        loop = asyncio.get_running_loop()
        removed = False

        def unsupported(*_args) -> None:
            raise NotImplementedError

        def remove(*_args) -> None:
            nonlocal removed
            removed = True

        # Restore the live loop before asyncio.run() closes it. Keeping bound
        # loop methods in pytest's outer monkeypatch fixture can retain a
        # closed loop and its executor across later asyncio.run() calls.
        with monkeypatch.context() as context:
            context.setattr(loop, "add_signal_handler", unsupported)
            context.setattr(loop, "remove_signal_handler", remove)
            result = await cli._run_job(FakeJob())
        assert not removed
        return result

    monkeypatch.setattr(cli, "_complete_job", complete)
    assert asyncio.run(exercise()) == 0
    assert ran


def test_non_tty_run_prints_only_finalized_output_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    staged = tmp_path / "work" / "movie.srt"
    finalized = tmp_path / "out" / "movie.srt"

    class FakeJob:
        work_dir = tmp_path / "work"

        async def run(self):
            return [staged]

    async def complete(_job, outputs):
        assert outputs == [staged]
        assert capsys.readouterr().out == ""
        print(finalized)
        return 0

    monkeypatch.setattr(cli, "_complete_job", complete)

    assert asyncio.run(cli._run_job(FakeJob())) == 0
    assert capsys.readouterr().out == f"{finalized}\n"
