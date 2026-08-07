from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import signal
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import load_settings, validate_run_settings
from .doctor import run_doctor
from .job import YakiFlowJob
from .models import JobEvent
from .models_manager import fetch_model
from .interactive_agent import (
    run_interactive_agent,
    run_memory_conflict_agent,
    start_agent_file_display,
)
from .memory import MemoryDestinationConflict


def _resume_command(work_dir: Path) -> str:
    return shlex.join(("yakiflow", "resume", str(work_dir)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yakiflow")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a new subtitle job")
    run.add_argument("input")
    _settings_arguments(run)
    resume = sub.add_parser("resume", help="resume a preserved work directory")
    resume.add_argument("workdir", type=Path)
    doctor = sub.add_parser("doctor", help="check external dependencies and authentication")
    _settings_arguments(doctor)
    models = sub.add_parser("models", help="manage Whisper models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    fetch = models_sub.add_parser("fetch")
    fetch.add_argument("--config", type=Path)
    fetch.add_argument("--profile", metavar="NAME")
    fetch.add_argument("--whisper-model", type=Path)
    return parser


def _settings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile", metavar="NAME")
    parser.add_argument("--source-language")
    parser.add_argument("--target-language")
    parser.add_argument("--whisper-model", type=Path)
    parser.add_argument("--vad-model", type=Path)
    parser.add_argument("--alignment-backend", choices=["vad", "whisperx"])
    parser.add_argument("--alignment-device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--alignment-model")
    parser.add_argument("--translation-backend", choices=["codex", "claude"])
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-mode", choices=["source", "translated", "bilingual", "all"])
    parser.add_argument("--memory", type=Path)
    parser.add_argument(
        "--context-file",
        dest="context_files",
        action="append",
        type=Path,
        metavar="FILE",
        help=(
            "copy a reference file into the work directory for interactive "
            "review; repeatable"
        ),
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-workdir", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--draft-model")
    parser.add_argument("--draft-effort", choices=["minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--final-model")
    parser.add_argument("--final-effort", choices=["minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--agent-workers", type=int)
    parser.add_argument("--draft-agent-timeout-seconds", type=float)
    parser.add_argument("--agent-max-attempts", type=int)
    parser.add_argument("--agent-retry-delay-seconds", type=float)
    parser.add_argument("--review-display-mode", choices=["split", "open", "both"])
    parser.add_argument("--review-open-command")
    parser.add_argument(
        "--auto-open-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="open the source video automatically when interactive review starts",
    )
    parser.add_argument("--video-open-command")


def _settings(namespace: argparse.Namespace):
    values = vars(namespace).copy()
    config = values.pop("config", None)
    profile = values.pop("profile", None)
    for key in ("command", "input", "workdir", "models_command"):
        values.pop(key, None)
    return load_settings(values, project_file=config, profile=profile)


async def _run_job(job: YakiFlowJob) -> int:
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(job.run())
    interrupts = 0

    def interrupt() -> None:
        nonlocal interrupts
        interrupts += 1
        if interrupts == 1:
            print("\nStopping acquisition; finalizing available media. Press Ctrl-C again to force exit.", file=sys.stderr)
            task.cancel()
        else:
            os._exit(130)

    signal_handler_installed = False
    if hasattr(loop, "add_signal_handler"):
        try:
            loop.add_signal_handler(signal.SIGINT, interrupt)
        except NotImplementedError:
            pass
        else:
            signal_handler_installed = True
    try:
        outputs = await task
    except asyncio.CancelledError:
        print(f"Interrupted. Resume with: {_resume_command(job.work_dir)}", file=sys.stderr)
        return 130
    finally:
        if signal_handler_installed and hasattr(loop, "remove_signal_handler"):
            loop.remove_signal_handler(signal.SIGINT)
    return await _complete_job(job, outputs)


async def _complete_job(
    job: YakiFlowJob,
    outputs: Sequence[Path],
) -> int:
    if sys.stdin.isatty() and sys.stdout.isatty():
        display = await start_agent_file_display(job.settings, job.work_dir, outputs)
        try:
            await run_interactive_agent(
                job.settings,
                job.work_dir,
                outputs,
                context_files=job.context_files,
                runner=job.runner,
            )
        finally:
            await display.close()
    while True:
        try:
            outputs = job.finalize_artifacts()
            break
        except MemoryDestinationConflict as conflict:
            await run_memory_conflict_agent(
                job.settings,
                job.work_dir,
                conflict.destination,
                conflict.diff(),
                runner=job.runner,
            )
            job.accept_memory_destination_change(conflict)
    for path in outputs:
        print(path)
    job.finish_review()
    cleanup = (
        job.temporary_workdir
        and not job.settings.keep_workdir
    )
    work_dir = job.work_dir
    job.close(cleanup=cleanup)
    if not cleanup:
        print(f"Work directory: {work_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "models":
            settings = _settings(args)
            def progress(done: int, total: int | None) -> None:
                suffix = f"/{total}" if total else ""
                print(f"\rDownloading {done}{suffix} bytes", end="", file=sys.stderr)
            path = fetch_model(settings.whisper_model, progress)
            print(f"\n{path}")
            return 0
        if args.command == "doctor":
            settings = _settings(args)
            checks = run_doctor(settings)
            for check in checks:
                print(f"{'OK' if check.ok else 'FAIL':4} {check.name}: {check.detail}")
            return 0 if all(check.ok for check in checks) else 1
        if args.command == "resume":
            job = YakiFlowJob.from_workdir(args.workdir, listener=_print_event)
        else:
            settings = _settings(args)
            validate_run_settings(settings)
            job = YakiFlowJob(args.input, settings, listener=_print_event)
        if sys.stdin.isatty() and sys.stdout.isatty():
            from . import tui

            succeeded = tui.run(job)
            if not succeeded:
                print(
                    f"Job preserved. Resume with: {_resume_command(job.work_dir)}",
                    file=sys.stderr,
                )
                return 2
            return asyncio.run(_complete_job(job, job.outputs))
        return asyncio.run(_run_job(job))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"yakiflow: {exc}", file=sys.stderr)
        return 2


async def _print_event(event: JobEvent) -> None:
    if event.kind in {"stage", "warning", "failed", "published"}:
        print(f"[{event.kind}] {event.message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
