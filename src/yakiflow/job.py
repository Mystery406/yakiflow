from __future__ import annotations

import asyncio
import inspect
import json
import os
import stat
import traceback
import shutil
import subprocess
import tempfile
import uuid
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Sequence

from platformdirs import user_runtime_path

from .alignment import (
    AlignmentModelFailureListener,
    AlignmentResult,
    PcmVolumeStartRefiner,
    adjust_cue_starts_for_long_vad_silences,
    extend_cue_ends,
    make_alignment_backend,
)
from .config import Settings
from .database import JobDatabase
from .media import MediaAcquirer, MediaArtifact, MediaSource, default_output_base
from .memory import MemoryDestinationConflict, MemoryFileSnapshot, MemoryStore
from .models import AgentTraceEvent, Cue, JobEvent, JobStatus, TranscriptEvent
from .process import CommandRunner
from .progress import ProgressPlan, StageTimeEstimator, make_progress_plan
from .srt import (
    SrtBlock,
    alignment_problems,
    output_modes,
    parse_srt_blocks,
    publish_outputs,
    render_srt,
    renumbered_srt,
    srt_problems,
    write_srt_atomic,
)
from .transcription import (
    WhisperCliTranscriber,
    WhisperServerTranscriber,
    normalize_source_language,
)
from .translation import AgentBackend, TranslationPipeline, make_backend


EventListener = Callable[[JobEvent], Awaitable[None] | None]

_DRAFT_PARTIAL_SUFFIX = ".draft.incomplete.srt"


def _temporary_workdir() -> Path:
    """Create a job directory below a private per-user temporary repo."""
    root = user_runtime_path("yakiflow", appauthor=False)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root_stat = root.lstat()
    getuid = getattr(os, "getuid", None)
    unsafe = not stat.S_ISDIR(root_stat.st_mode)
    if callable(getuid):
        unsafe = unsafe or (
            root_stat.st_uid != getuid()
            or stat.S_IMODE(root_stat.st_mode) != 0o700
        )
    if unsafe:
        raise RuntimeError(
            f"unsafe temporary repository {root}: expected an owner-only "
            "directory owned by the current user"
        )
    if not (root / ".git").exists():
        try:
            subprocess.run(
                ["git", "init", "--quiet", str(root)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            # Keep a stable repository marker even when the git executable is
            # unavailable. Codex can then associate all jobs with this shared
            # workspace instead of probing a parent directory each time.
            (root / ".git").mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="job-", dir=root))


class YakiFlowJob:
    def __init__(
        self,
        input_value: str,
        settings: Settings,
        *,
        work_dir: Path | None = None,
        runner: CommandRunner | None = None,
        backend: AgentBackend | None = None,
        listener: EventListener | None = None,
        _resume: bool = False,
        _temporary: bool | None = None,
    ):
        settings = settings.resolved()
        source = MediaSource.parse(input_value)
        input_value = source.value
        requested = work_dir or settings.work_dir
        self.temporary_workdir = (
            requested is None if _temporary is None else _temporary
        )
        self.work_dir = _temporary_workdir() if requested is None else Path(requested).resolve()
        if not _resume and self.work_dir.exists() and any(self.work_dir.iterdir()):
            raise ValueError(
                f"work directory is not empty: {self.work_dir}; "
                "use 'yakiflow resume' for an existing job"
            )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.input_value = input_value
        self.settings = settings
        self.context_files = (
            self._staged_context_files()
            if _resume
            else self._copy_context_files(settings.context_files)
        )
        self.runner = runner or CommandRunner()
        self.db = JobDatabase(self.work_dir / "job.sqlite3")
        self.listener = listener
        draft_options = (
            settings.draft_codex_options
            if settings.translation_backend == "codex"
            else settings.draft_claude_options
        )
        self.backend = backend or make_backend(
            settings.translation_backend or "",
            self.work_dir,
            self.runner,
            options=draft_options,
        )
        self.memory_destination = settings.memory
        self.memory_path = self.work_dir / "memory.md"
        existing_job = self.db.job() is not None
        saved_memory_snapshot = MemoryFileSnapshot.from_checkpoint(
            self.db.get_checkpoint("memory_destination_snapshot")
        )
        if existing_job:
            if saved_memory_snapshot is None:
                raise ValueError("work directory is missing its memory destination snapshot")
            self._memory_destination_snapshot = saved_memory_snapshot
        else:
            current_memory = (
                MemoryFileSnapshot.read(self.memory_destination)
                if self.memory_destination
                else MemoryFileSnapshot(False)
            )
            self._memory_destination_snapshot = current_memory
            self.db.checkpoint(
                "memory_destination_snapshot",
                self._memory_destination_snapshot.checkpoint(),
            )
            if (
                current_memory.exists
                and self.memory_destination is not None
                and self.memory_path.resolve() != self.memory_destination.resolve()
            ):
                self.memory_path.write_text(current_memory.content, encoding="utf-8")
        self.memory_store = MemoryStore(self.memory_path)
        if not existing_job:
            self.db.create_job(uuid.uuid4().hex, input_value, asdict(settings))
            self.db.checkpoint("temporary_workdir", self.temporary_workdir)
        job_row = self.db.job()
        self._resume_status = (
            JobStatus(job_row["status"])
            if _resume and job_row is not None
            else None
        )
        self.outputs: list[Path] = []
        self._output_destination_base: Path | None = None
        self.alignment_result: AlignmentResult | None = None
        self.alignment_model_failure_listener: AlignmentModelFailureListener | None = None
        self._agent_operations: set[str] = set()
        self._progress_plan: ProgressPlan | None = None
        self._time_estimator: StageTimeEstimator | None = None
        self._progress_value = 0.0
        self._stage_progress: dict[str, float] = {}
        self._completed_progress_stages: set[str] = set()
        self._combined_translation_progress = False
        self._translation_batches_completed = 0
        self._translation_batches_total = 0
        self._artifact: MediaArtifact | None = None

    def _staged_context_files(self) -> tuple[Path, ...]:
        context_dir = self.work_dir / "context"
        if not context_dir.is_dir():
            return ()
        return tuple(path for path in sorted(context_dir.iterdir()) if path.is_file())

    def _copy_context_files(self, sources: Sequence[Path]) -> tuple[Path, ...]:
        """Snapshot interactive-review references inside the job directory."""
        if not sources:
            return ()
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(
                    f"context file does not exist or is not a file: {source}"
                )
        context_dir = self.work_dir / "context"
        context_dir.mkdir()
        copied: list[Path] = []
        for source in sources:
            destination = context_dir / source.name
            suffix = 2
            while destination.exists():
                destination = context_dir / f"{source.stem}-{suffix}{source.suffix}"
                suffix += 1
            shutil.copy2(source, destination)
            copied.append(destination)
        return tuple(copied)

    @property
    def media_path(self) -> Path | None:
        """The original or downloaded media file used by this job."""
        if self._artifact is not None:
            return self._artifact.media_path
        return self.db.artifact("source_media")

    @property
    def subtitle_path(self) -> Path | None:
        """The best subtitle file to attach while the job is being reviewed."""
        if self.outputs:
            return self.outputs[0]
        if self._artifact is not None:
            return self._draft_partial_path(self._artifact)
        candidates = sorted(
            self.work_dir.glob("*.srt"), key=lambda path: path.stat().st_mtime_ns
        )
        return candidates[-1] if candidates else None

    @classmethod
    def from_workdir(
        cls,
        work_dir: Path,
        *,
        listener: EventListener | None = None,
        runner: CommandRunner | None = None,
        backend: AgentBackend | None = None,
    ) -> YakiFlowJob:
        db = JobDatabase(Path(work_dir) / "job.sqlite3")
        row = db.job()
        if row is None:
            db.close()
            raise ValueError(f"not a YakiFlow work directory: {work_dir}")
        values = json.loads(row["config_json"])
        temporary = db.get_checkpoint("temporary_workdir")
        if temporary is None:
            # Jobs created before this status was persisted can still be
            # identified by the private runtime directory used by yakiflow.
            resolved = Path(work_dir).resolve()
            runtime_root = user_runtime_path("yakiflow", appauthor=False).resolve()
            temporary = (
                resolved.parent == runtime_root
                and resolved.name.startswith("job-")
            )
        db.close()
        from .config import load_settings
        settings = load_settings(values, project_file=Path("/__yakiflow_no_project_config__"), user_file=Path("/__yakiflow_no_user_config__"))
        return cls(
            row["input"], settings, work_dir=work_dir, runner=runner,
            backend=backend, listener=listener, _resume=True,
            _temporary=temporary is True,
        )

    async def emit(
        self,
        kind: str,
        message: str,
        progress: float | None = None,
        cue: Cue | None = None,
        *,
        stage: str | None = None,
        estimated_remaining: int | None = None,
        agent_trace: AgentTraceEvent | None = None,
        error_traceback: str | None = None,
    ) -> None:
        if not self.listener:
            return
        maybe = self.listener(
            JobEvent(
                kind,
                message,
                progress,
                cue,
                stage=stage,
                estimated_remaining=estimated_remaining,
                agent_trace=agent_trace,
                error_traceback=error_traceback,
            )
        )
        if inspect.isawaitable(maybe):
            await maybe

    def _agent_status(self) -> str:
        if not self._agent_operations:
            return "Agent · Idle"
        count = len(self._agent_operations)
        suffix = f" · {count} active" if count > 1 else ""
        return f"Agent · Running · draft translation{suffix}"

    async def agent_event(self, event: AgentTraceEvent) -> None:
        """Forward one Agent trace event and keep the compact summary honest."""
        if event.kind == "lifecycle":
            if event.state in {"running", "retrying"}:
                self._agent_operations.add(event.operation_id)
            else:
                self._agent_operations.discard(event.operation_id)
        await self.emit(
            "agent_trace", event.message, agent_trace=event
        )
        if event.kind == "lifecycle":
            await self.emit("agent", self._agent_status())

    async def _agent_retry(self, message: str) -> None:
        await self.emit("warning", message)

    def _translation_pipeline(self) -> TranslationPipeline:
        return TranslationPipeline(
            self.settings,
            self.backend,
            self.db,
            self.memory_store.read(),
            on_retry=self._agent_retry,
            on_agent_event=self.agent_event,
        )

    def _configure_progress(self) -> None:
        from .config import default_model_path

        source = MediaSource.parse(self.input_value)
        model = self.settings.whisper_model
        needs_model = bool(
            model
            and not model.is_file()
            and model.expanduser().resolve() == default_model_path().expanduser().resolve()
        )
        existing_audio = self.db.artifact("reference_audio")
        needs_acquire = not bool(existing_audio and existing_audio.exists())
        needs_transcription = not bool(self.db.get_checkpoint("transcribed"))
        resume_from = self._transcription_resume_cues() if needs_transcription else []
        existing_cues = self.db.list_cues(stable_only=True)
        alignment_complete = bool(self.db.get_checkpoint("alignment_complete", False))
        has_missing_translation = any(not cue.translated for cue in existing_cues)
        needs_translation = needs_transcription or (
            has_missing_translation and not alignment_complete
        )
        needs_post_alignment_translation = (
            self.settings.alignment_backend == "whisperx"
            and (not alignment_complete or has_missing_translation)
        )
        # Draft batches run alongside Whisper and are already represented by
        # the tail of the transcription stage. Keep a separate translation
        # range only when transcription was restored from a checkpoint.
        self._combined_translation_progress = needs_transcription and needs_translation
        self._translation_batches_completed = 0
        self._translation_batches_total = 0
        transcription_scale = self._remaining_transcription_fraction(
            existing_audio, resume_from
        )
        self._progress_plan = make_progress_plan(
            needs_model=needs_model,
            is_url=source.is_url,
            streaming=self.settings.stream,
            needs_acquire=needs_acquire,
            needs_transcription=needs_transcription,
            needs_translation=(
                needs_translation and not self._combined_translation_progress
            ),
            needs_post_alignment_translation=needs_post_alignment_translation,
            transcription_scale=transcription_scale,
        )
        self._time_estimator = StageTimeEstimator(self._progress_plan)

    def _transcription_resume_cues(self) -> list[Cue]:
        existing = self.db.list_cues(stable_only=True)
        if existing and self.db.get_checkpoint("transcription_started"):
            return existing
        return []

    @classmethod
    def _remaining_transcription_fraction(
        cls, audio: Path | None, resume_from: Sequence[Cue]
    ) -> float:
        """Return the unprocessed share of a resumable PCM WAV file."""
        if audio is None or not resume_from:
            return 1.0
        duration = cls._pcm_audio_duration(audio)
        if duration is None:
            return 1.0
        processed = max(cue.end for cue in resume_from)
        # Retain a small non-zero span for finalization when the last durable
        # cue reaches the apparent end of the file.
        return max(0.01, min(1.0, (duration - processed) / duration))

    @staticmethod
    def _pcm_audio_duration(audio: Path) -> float | None:
        try:
            with wave.open(str(audio), "rb") as source:
                rate = source.getframerate()
                duration = source.getnframes() / rate if rate > 0 else 0.0
        except (EOFError, OSError, wave.Error):
            return None
        return duration if duration > 0 else None

    async def _report_progress(
        self,
        stage: str,
        fraction: float,
        message: str,
        *,
        kind: str = "progress",
    ) -> None:
        if stage in self._completed_progress_stages:
            return
        fraction = max(self._stage_progress.get(stage, 0.0), min(1.0, max(0.0, fraction)))
        self._stage_progress[stage] = fraction
        if self._progress_plan and stage in self._progress_plan.ranges:
            self._progress_value = max(self._progress_value, self._progress_plan.value(stage, fraction))
        estimated_remaining = (
            self._time_estimator.update(stage, fraction)
            if self._time_estimator is not None
            else None
        )
        await self.emit(
            kind,
            message,
            self._progress_value,
            stage=stage,
            estimated_remaining=estimated_remaining,
        )

    async def _begin_stage(self, stage: str, message: str) -> None:
        await self._report_progress(stage, 0.0, message, kind="stage")

    async def _finish_stage(self, stage: str, message: str) -> None:
        await self._report_progress(stage, 1.0, message)
        self._completed_progress_stages.add(stage)

    async def _report_draft_batch_progress(self) -> None:
        await self._report_progress(
            "transcribe",
            0.85
            + 0.15
            * self._translation_batches_completed
            / self._translation_batches_total,
            f"draft Agent batches · {self._translation_batches_completed}/{self._translation_batches_total}",
        )

    async def run(self) -> list[Path]:
        try:
            if self._resume_status is JobStatus.REVIEWING:
                self.outputs = self._restore_review_outputs()
                await self.emit(
                    "published", "restored subtitles awaiting review", 1.0,
                    stage="publish", estimated_remaining=0,
                )
                return self.outputs
            self._configure_progress()
            await self._ensure_model()
            artifact = await self._media_stage()
            self._artifact = artifact
            cues = await self._transcription_stage(artifact)
            if not self.db.get_checkpoint("alignment_complete", False):
                cues = await self._translation_stage(artifact, cues)
            cues = await self._alignment_stage(artifact, cues)
            cues = await self._post_alignment_translation_stage(artifact, cues)
            await self._begin_stage("publish", "publishing subtitles")
            self.outputs = self._publish(artifact, cues)
            self.db.set_status(JobStatus.REVIEWING)
            await self._finish_stage("publish", "subtitles published")
            await self.emit(
                "published", "final subtitles published", 1.0,
                stage="publish", estimated_remaining=0,
            )
            return self.outputs
        except asyncio.CancelledError:
            self.db.set_status(JobStatus.INTERRUPTED, "interrupted")
            await self.emit("interrupted", f"job preserved at {self.work_dir}")
            raise
        except Exception as exc:
            self.db.set_status(JobStatus.FAILED, str(exc))
            await self.emit("failed", str(exc), error_traceback=traceback.format_exc())
            raise

    async def _ensure_model(self) -> None:
        vad_model = self.settings.vad_model
        if vad_model is not None and not vad_model.is_file():
            raise FileNotFoundError(
                f"configured Whisper VAD model does not exist: {vad_model}"
            )
        model = self.settings.whisper_model
        if model is None or model.is_file():
            return
        from .config import default_model_path
        if model.expanduser().resolve() != default_model_path().expanduser().resolve():
            raise FileNotFoundError(f"configured Whisper model does not exist: {model}")
        await self._begin_stage("model", f"fetching default Whisper model to {model}")
        from .models_manager import fetch_model
        loop = asyncio.get_running_loop()
        last_percent = -1
        last_unknown_report = 0

        def progress(received: int, total: int | None) -> None:
            nonlocal last_percent, last_unknown_report
            if total:
                percent = min(100, int(received * 100 / total))
                if percent <= last_percent:
                    return
                last_percent = percent
                fraction = received / total
                message = f"downloading Whisper model · {percent}%"
            else:
                report_interval = 64 * 1024 * 1024
                if received - last_unknown_report < report_interval:
                    return
                last_unknown_report = received
                fraction = 0.05
                message = f"downloading Whisper model · {received // (1024 * 1024)} MiB"
            future = asyncio.run_coroutine_threadsafe(
                self._report_progress("model", fraction, message), loop
            )
            future.result()

        await asyncio.to_thread(fetch_model, model, progress)
        await self._finish_stage("model", "Whisper model ready")

    async def _media_stage(self) -> MediaArtifact:
        existing_audio = self.db.artifact("reference_audio")
        existing_media = self.db.artifact("source_media")
        source = MediaSource.parse(self.input_value)
        if existing_audio and existing_audio.exists():
            persistent = bool(self.settings.download_dir) or not source.is_url
            await self._begin_stage("acquire", "using cached media")
            artifact = MediaArtifact(source, existing_audio, existing_media, persistent)
            await self._finish_stage("acquire", "media ready")
            return artifact
        self.db.set_status(JobStatus.ACQUIRING)
        await self._begin_stage("acquire", "acquiring media")

        async def media_progress(fraction: float) -> None:
            await self._report_progress(
                "acquire", fraction, f"acquiring media · {round(fraction * 100)}%"
            )

        acquirer = MediaAcquirer(
            self.settings, self.work_dir, self.db, self.runner, on_progress=media_progress
        )
        if not self.settings.stream:
            artifact = await acquirer.acquire(source)
            await self._finish_stage("acquire", "media ready")
            return artifact

        server = WhisperServerTranscriber(self.settings, self.work_dir, self.db)
        await server.start()
        pipeline = self._translation_pipeline()
        partial_path = self._live_partial_path(source)
        completed_chunks = 0

        async def chunk(snapshot: Path, emitted: float) -> None:
            nonlocal completed_chunks
            start = max(0.0, emitted - self.settings.stream_context_seconds)
            excerpt = self.work_dir / "stream-chunk.wav"
            await self.runner.run([
                self.settings.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(start), "-t", str(self.settings.stream_chunk_seconds + self.settings.stream_context_seconds),
                "-i", snapshot, "-ac", "1", "-ar", "16000", excerpt,
            ])
            existing_ids = {cue.id for cue in server.cues}
            await server.submit(excerpt, start)
            new_cues = [cue for cue in server.cues if cue.id not in existing_ids]
            self.db.replace_transcript(server.cues)
            timeline = self.db.list_cues(stable_only=True)
            await self._write_partial(partial_path, timeline)
            for cue in new_cues:
                await self.emit("transcript", "Whisper subtitle", cue=cue)
            pending = [cue for cue in timeline if not cue.translated]
            if pending:
                updated = await pipeline.translate_draft(pending)
                translated_ids = {cue.id for cue in pending}
                await self._write_partial(partial_path, updated)
                await self._emit_agent_cues(cue for cue in updated if cue.id in translated_ids)
            completed_chunks += 1
            await self._report_progress(
                "acquire",
                min(0.9, 0.15 + completed_chunks * 0.04),
                f"streaming media · {completed_chunks} chunks analyzed",
            )
            await self.emit("stream", f"updated {partial_path}")

        try:
            artifact = await acquirer.acquire_stream(source, chunk)
        finally:
            await server.close()
        await self._finish_stage("acquire", "stream acquisition complete")
        return artifact

    async def _transcription_stage(self, artifact: MediaArtifact) -> list[Cue]:
        partial_path = self._draft_partial_path(artifact)
        if self.db.get_checkpoint("transcribed"):
            cues = self.db.list_cues(stable_only=True)
            if cues:
                await self._write_partial(partial_path, cues)
                await self._begin_stage("transcribe", "using cached transcription")
                await self._finish_stage("transcribe", "transcription ready")
                return cues
        resume_from = self._transcription_resume_cues()
        if self.settings.stream and not self.db.get_checkpoint("transcription_started"):
            # Streaming cues use positional IDs that the authoritative
            # full-audio pass will reuse. Their translations belong to the
            # provisional source text and must not survive that ID collision.
            self.db.discard_provisional_transcript()
        self.db.checkpoint("transcription_started", True)
        self.db.set_status(JobStatus.TRANSCRIBING)
        if resume_from:
            await self._begin_stage(
                "transcribe",
                f"resuming Whisper after {resume_from[-1].end:.2f}s",
            )
        else:
            await self._begin_stage(
                "transcribe", "running authoritative full-audio transcription"
            )
        transcriber = WhisperCliTranscriber(self.settings, self.work_dir, self.db, self.runner)
        incremental: list[Cue] = []
        translation_tasks: list[asyncio.Task[None]] = []
        scheduled_translation_ids: set[str] = set()
        whisper_done = False
        pipeline = self._translation_pipeline()

        async def translate_ready(
            batch: list[Cue],
            preceding_context: Sequence[Cue] = (),
            following_context: Sequence[Cue] = (),
        ) -> None:
            updated = await pipeline.translate_draft(
                batch,
                lambda all_cues: self._write_partial(partial_path, all_cues),
                preceding_context=preceding_context,
                following_context=following_context,
            )
            translated_ids = {cue.id for cue in batch}
            await self._emit_agent_cues(cue for cue in updated if cue.id in translated_ids)
            if self._combined_translation_progress:
                self._translation_batches_completed += 1
            if (
                whisper_done
                and self._combined_translation_progress
                and self._translation_batches_total
            ):
                await self._report_draft_batch_progress()

        async def event(item: TranscriptEvent) -> None:
            if item.final:
                return
            incremental.append(item.cue)
            self.db.upsert_cues([item.cue])
            await self._write_partial(
                partial_path, self.db.list_cues(stable_only=True)
            )
            await self.emit("transcript", "Whisper subtitle", cue=item.cue)
            if len(incremental) % self.settings.translation_batch_size == 0:
                batch = incremental[-self.settings.translation_batch_size:]
                timeline = self.db.list_cues(stable_only=True)
                batch_start = next(
                    index for index, cue in enumerate(timeline)
                    if cue.id == batch[0].id
                )
                context_start = max(
                    0, batch_start - self.settings.translation_context
                )
                preceding_context = timeline[context_start:batch_start]
                scheduled_translation_ids.update(cue.id for cue in batch)
                translation_tasks.append(
                    asyncio.create_task(translate_ready(batch, preceding_context))
                )

        async def whisper_progress(fraction: float) -> None:
            await self._report_progress(
                "transcribe",
                fraction * 0.85,
                f"Whisper transcription · {round(fraction * 100)}%",
            )

        # A previous run may have persisted Whisper cues before their draft
        # Agent batch completed. Catch those translations up immediately while
        # Whisper resumes, instead of waiting until the whole remaining audio
        # has been transcribed.
        for batch, context, following in self._missing_translation_batches(resume_from):
            scheduled_translation_ids.update(cue.id for cue in batch)
            translation_tasks.append(
                asyncio.create_task(translate_ready(batch, context, following))
            )

        transcription_succeeded = False
        try:
            final = await transcriber.transcribe(
                artifact.audio_path,
                event,
                whisper_progress,
                resume_from=resume_from,
            )
            whisper_done = True
            self.db.replace_transcript(final)
            await self._write_partial(
                partial_path, self.db.list_cues(stable_only=True)
            )
            unscheduled_batches = self._missing_translation_batches(
                self.db.list_cues(stable_only=True),
                excluded_ids=scheduled_translation_ids,
            )
            if self._combined_translation_progress:
                self._translation_batches_total = (
                    len(translation_tasks) + len(unscheduled_batches)
                )
            # Persist the authoritative transcript before waiting for draft
            # Agent work. If that work is interrupted, resume must not launch
            # Whisper for the same audio again.
            self.db.checkpoint("transcribed", True)
            if translation_tasks:
                if self._combined_translation_progress:
                    await self._report_draft_batch_progress()
                await asyncio.gather(*translation_tasks)
            transcription_succeeded = True
        finally:
            if not transcription_succeeded and translation_tasks:
                for task in translation_tasks:
                    task.cancel()
                await asyncio.gather(*translation_tasks, return_exceptions=True)
        if (
            not self._combined_translation_progress
            or self._translation_batches_completed >= self._translation_batches_total
        ):
            await self._finish_stage("transcribe", "authoritative transcription ready")
        return self.db.list_cues(stable_only=True)

    async def _write_partial(self, path: Path, cues: Sequence[Cue]) -> None:
        mode = self.settings.output_mode if self.settings.output_mode != "all" else "bilingual"
        write_srt_atomic(path, cues, mode)

    async def _translation_stage(self, artifact: MediaArtifact, cues: Sequence[Cue]) -> list[Cue]:
        self.db.set_status(JobStatus.TRANSLATING)
        progress_stage = (
            "transcribe" if self._combined_translation_progress else "translate"
        )
        await self._begin_stage(progress_stage, "translating draft subtitles")
        pipeline = self._translation_pipeline()
        timeline = self.db.list_cues(stable_only=True)
        batches = self._missing_translation_batches(timeline)
        if self._combined_translation_progress:
            self._translation_batches_total = (
                self._translation_batches_completed + len(batches)
            )
        if batches:
            partial_path = self._draft_partial_path(artifact)
            completed_batches = 0

            async def translation_progress(_completed: int, _total: int) -> None:
                nonlocal completed_batches
                completed_batches += 1
                if self._combined_translation_progress:
                    self._translation_batches_completed += 1
                    fraction = (
                        0.85
                        + 0.15
                        * self._translation_batches_completed
                        / self._translation_batches_total
                    )
                else:
                    fraction = completed_batches / len(batches)
                await self._report_progress(
                    progress_stage,
                    fraction,
                    f"Agent draft translation · {completed_batches}/{len(batches)} batches",
                )

            await asyncio.gather(*(
                pipeline.translate_draft(
                    batch,
                    lambda all_cues: self._write_partial(partial_path, all_cues),
                    translation_progress,
                    preceding_context=context,
                    following_context=following,
                )
                for batch, context, following in batches
            ))
            translated_ids = {
                cue.id for batch, _context, _following in batches for cue in batch
            }
            await self._emit_agent_cues(
                cue
                for cue in self.db.list_cues(stable_only=True)
                if cue.id in translated_ids
            )
        remaining = [
            cue.id
            for cue in self.db.list_cues(stable_only=True)
            if not cue.translated
        ]
        if remaining:
            raise RuntimeError(
                f"translation remained empty for cue IDs: {remaining}"
            )
        await self._finish_stage(progress_stage, "draft translation ready")
        return self.db.list_cues(stable_only=True)

    def _missing_translation_batches(
        self,
        cues: Sequence[Cue],
        *,
        excluded_ids: set[str] | None = None,
    ) -> list[tuple[list[Cue], list[Cue], list[Cue]]]:
        """Return contiguous missing-translation batches with their context.

        Each batch carries the cues around it on both sides: a sentence that
        runs past the end of a batch is otherwise invisible to the Agent
        translating it.
        """
        batch_size = self.settings.translation_batch_size
        context_size = self.settings.translation_context
        following_size = self.settings.translation_following_context
        batches: list[tuple[list[Cue], list[Cue], list[Cue]]] = []
        run_start: int | None = None

        def append_run(start: int, end: int) -> None:
            for batch_start in range(start, end, batch_size):
                batch_end = min(batch_start + batch_size, end)
                context_start = max(0, batch_start - context_size)
                batches.append((
                    list(cues[batch_start:batch_end]),
                    list(cues[context_start:batch_start]),
                    list(cues[batch_end:batch_end + following_size]),
                ))

        excluded_ids = excluded_ids or set()
        for index, cue in enumerate(cues):
            missing = (
                cue.id not in excluded_ids
                and not cue.translated
            )
            if missing and run_start is None:
                run_start = index
            elif not missing and run_start is not None:
                append_run(run_start, index)
                run_start = None
        if run_start is not None:
            append_run(run_start, len(cues))
        return batches

    async def _emit_agent_cues(self, cues: Iterable[Cue]) -> None:
        for cue in cues:
            await self.emit("subtitle", "agent-processed subtitle", cue=cue)

    def _draft_partial_path(self, artifact: MediaArtifact) -> Path:
        # Acquired jobs always have a media path. Keep synthetic/recovery
        # artifacts without one inside the job directory rather than writing a
        # generic subtitle snapshot into the caller's current directory.
        base = (
            self.work_dir / "subtitle"
            if artifact.media_path is None and self.settings.output_dir is None
            else default_output_base(artifact, self.settings, self.work_dir)
        )
        return base.parent / f"{base.name}{_DRAFT_PARTIAL_SUFFIX}"

    def _live_partial_path(self, source: MediaSource) -> Path:
        directory = self.work_dir
        base_name = Path(source.value).stem if not source.is_url else "source"
        return directory / f"{base_name}.live.incomplete.srt"

    async def _alignment_stage(self, artifact: MediaArtifact, cues: Sequence[Cue]) -> list[Cue]:
        self.db.set_status(JobStatus.ALIGNING)
        if self.db.get_checkpoint("alignment_complete", False):
            restored = self.db.list_cues(stable_only=True)
            await self._begin_stage("align", "using cached alignment")
            await self._finish_stage("align", "alignment ready")
            return restored
        await self._begin_stage("align", "adjusting subtitle starts against silence")
        language = self.db.get_checkpoint("detected_source_language") or normalize_source_language(
            self.settings.source_language
        )
        backend = make_alignment_backend(
            self.settings.alignment_backend,
            language=language,
            device=self.settings.alignment_device,
            model_name=self.settings.alignment_model,
        )

        async def alignment_progress(completed: int, total: int) -> None:
            fraction = completed / total if total else 1.0
            await self._report_progress(
                "align",
                fraction,
                f"forced alignment · {completed}/{total} cues",
            )

        vad_intervals = self.db.get_checkpoint("whisper_vad_intervals", [])
        cues = adjust_cue_starts_for_long_vad_silences(cues, vad_intervals)
        self.alignment_result = await backend.align(
            artifact.audio_path,
            cues,
            vad_intervals=vad_intervals,
            on_warning=lambda message: self.emit("warning", message),
            on_progress=alignment_progress,
            on_model_failure=self.alignment_model_failure_listener,
        )
        self.alignment_result.cues = await PcmVolumeStartRefiner().refine(
            artifact.audio_path,
            self.alignment_result.cues,
            on_warning=lambda message: self.emit("warning", message),
        )
        self.alignment_result.cues = extend_cue_ends(
            self.alignment_result.cues,
            duration=self._pcm_audio_duration(artifact.audio_path),
        )
        self.db.replace_aligned_timeline(self.alignment_result.cues)
        await self.emit("timeline-replaced", "aligned subtitle timeline replaced")
        # The interactive hand-off copies the draft snapshot over the staged
        # output. Refresh it after timing adjustment so that hand-off cannot
        # restore the pre-alignment timeline.
        await self._write_partial(
            self._draft_partial_path(artifact), self.alignment_result.cues
        )
        await self._finish_stage("align", "alignment complete")
        return self.alignment_result.cues

    async def _post_alignment_translation_stage(
        self,
        artifact: MediaArtifact,
        cues: Sequence[Cue],
    ) -> list[Cue]:
        timeline = self.db.list_cues(stable_only=True)
        batches = self._missing_translation_batches(timeline)
        if not batches:
            if (
                self._progress_plan is not None
                and "post-align-translate" in self._progress_plan.ranges
            ):
                await self._begin_stage(
                    "post-align-translate", "checking forced-alignment translations"
                )
                await self._finish_stage(
                    "post-align-translate", "forced-alignment translations ready"
                )
            return timeline
        self.db.set_status(JobStatus.TRANSLATING)
        await self._begin_stage(
            "post-align-translate",
            "translating sentence cues created by forced alignment",
        )
        pipeline = self._translation_pipeline()
        partial_path = self._draft_partial_path(artifact)
        completed_batches = 0

        async def translation_progress(_completed: int, _total: int) -> None:
            nonlocal completed_batches
            completed_batches += 1
            await self._report_progress(
                "post-align-translate",
                completed_batches / len(batches),
                f"post-alignment translation · {completed_batches}/{len(batches)} batches",
            )

        await asyncio.gather(
            *(
                pipeline.translate_draft(
                    batch,
                    lambda all_cues: self._write_partial(partial_path, all_cues),
                    translation_progress,
                    preceding_context=context,
                    following_context=following,
                    translate_only=True,
                )
                for batch, context, following in batches
            )
        )
        translated_ids = {
            cue.id for batch, _context, _following in batches for cue in batch
        }
        await self._emit_agent_cues(
            cue
            for cue in self.db.list_cues(stable_only=True)
            if cue.id in translated_ids
        )
        remaining = [
            cue.id
            for cue in self.db.list_cues(stable_only=True)
            if not cue.translated
        ]
        if remaining:
            raise RuntimeError(
                f"post-alignment translation remained empty for cue IDs: {remaining}"
            )
        await self._finish_stage(
            "post-align-translate", "post-alignment translation ready"
        )
        return self.db.list_cues(stable_only=True)

    def _publish(self, artifact: MediaArtifact, cues: Sequence[Cue]) -> list[Path]:
        destination_base = default_output_base(artifact, self.settings, self.work_dir)
        self._output_destination_base = destination_base
        staging_base = self.work_dir / destination_base.name
        outputs = publish_outputs(
            staging_base,
            cues,
            self.settings.output_mode,
            source_language=self.settings.source_language,
            target_language=self.settings.target_language,
        )
        self.db.checkpoint("review_outputs", [str(path) for path in outputs])
        self.db.checkpoint("output_destination_base", str(destination_base))
        return outputs

    def _restore_review_outputs(self) -> list[Path]:
        """Restore Agent-editable files without overwriting their reviewed text."""
        saved_outputs = self.db.get_checkpoint("review_outputs")
        if saved_outputs:
            outputs = [Path(path) for path in saved_outputs]
        else:
            # Compatibility for jobs that reached REVIEWING before exact paths
            # were checkpointed. Incomplete snapshots are never final outputs.
            outputs = sorted(
                path
                for path in self.work_dir.glob("*.srt")
                if not path.name.endswith(".incomplete.srt")
            )
        missing = [path for path in outputs if not path.is_file()]
        if not outputs or missing:
            detail = f": {', '.join(str(path) for path in missing)}" if missing else ""
            raise RuntimeError(f"reviewing job is missing staged subtitle files{detail}")
        saved_destination = self.db.get_checkpoint("output_destination_base")
        if saved_destination:
            self._output_destination_base = Path(saved_destination)
        else:
            source = MediaSource.parse(self.input_value)
            media_path = self.db.artifact("source_media")
            audio_path = self.db.artifact("reference_audio") or self.work_dir / "reference.wav"
            artifact = MediaArtifact(
                source,
                audio_path,
                media_path,
                bool(self.settings.download_dir) or not source.is_url,
            )
            self._output_destination_base = default_output_base(
                artifact, self.settings, self.work_dir
            )
        return outputs

    def _validate_review_outputs(self) -> None:
        """Refuse to publish subtitle files the review left structurally broken.

        Cue numbering is repaired in place because it is purely mechanical.
        Everything else is only reported when the pipeline's own rendering of
        the same cues is free of that defect, so a quirk YakiFlow produced
        itself never fails a job at the finish line.
        """
        cues = self.db.list_cues(stable_only=True)
        if not cues:
            return
        tolerated = {
            problem.kind
            for mode in output_modes(self.settings.output_mode)
            for problem in srt_problems(render_srt(cues, mode))
        }
        problems: list[str] = []
        artifacts: list[tuple[str, Sequence[SrtBlock]]] = []
        for staged in self.outputs:
            text = staged.read_text(encoding="utf-8", errors="replace")
            repaired = renumbered_srt(text)
            if repaired is not None:
                staged.write_text(repaired, encoding="utf-8", newline="\n")
                text = repaired
            blocks, _unparsed = parse_srt_blocks(text)
            artifacts.append((staged.name, blocks))
            problems.extend(
                f"{staged.name}: {problem.message}"
                for problem in srt_problems(text)
                if problem.kind not in tolerated
            )
        problems.extend(problem.message for problem in alignment_problems(artifacts))
        if problems:
            detail = "\n".join(f"  - {problem}" for problem in problems)
            raise RuntimeError(
                "reviewed subtitles are not publishable; fix the staged files in "
                f"{self.work_dir} and resume the job:\n{detail}"
            )

    def finalize_artifacts(self) -> list[Path]:
        """Move Agent-editable staging artifacts to their configured targets."""
        self._check_memory_destination_unchanged()
        missing = [staged for staged in self.outputs if not staged.is_file()]
        if missing:
            detail = ", ".join(str(path) for path in missing)
            raise RuntimeError(f"cannot finalize missing staged subtitle files: {detail}")
        self._validate_review_outputs()
        destinations: list[Path] = []
        destination_base = self._output_destination_base
        if destination_base is not None:
            for staged in self.outputs:
                destination = destination_base.parent / staged.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if staged.resolve() != destination.resolve():
                    shutil.move(str(staged), str(destination))
                destinations.append(destination)
            self.outputs = destinations
            # Remove the corresponding draft snapshot from the configured
            # output directory once the finished subtitle has been published.
            draft_name = f"{destination_base.name}{_DRAFT_PARTIAL_SUFFIX}"
            (destination_base.parent / draft_name).unlink(missing_ok=True)
            (self.work_dir / draft_name).unlink(missing_ok=True)
        if self.memory_destination and self.memory_path.is_file():
            self.memory_destination.parent.mkdir(parents=True, exist_ok=True)
            if self.memory_path.resolve() != self.memory_destination.resolve():
                # Subtitle publication above may take long enough for another
                # process to update memory after the initial all-artifact
                # guard. Check again at the actual overwrite boundary.
                self._check_memory_destination_unchanged()
                shutil.move(str(self.memory_path), str(self.memory_destination))
        return self.outputs

    def _check_memory_destination_unchanged(self) -> None:
        if (
            not self.memory_destination
            or not self.memory_path.is_file()
            or self.memory_path.resolve() == self.memory_destination.resolve()
        ):
            return
        current_memory = MemoryFileSnapshot.read(self.memory_destination)
        if current_memory != self._memory_destination_snapshot:
            raise MemoryDestinationConflict(
                self.memory_destination,
                self._memory_destination_snapshot,
                current_memory,
            )

    def accept_memory_destination_change(
        self,
        conflict: MemoryDestinationConflict,
    ) -> None:
        """Record the destination revision after an Agent has merged its diff."""
        if conflict.destination != self.memory_destination:
            raise ValueError("memory conflict belongs to a different destination")
        if conflict.previous != self._memory_destination_snapshot:
            raise ValueError("memory conflict is no longer current")
        self._memory_destination_snapshot = conflict.current
        self.db.checkpoint(
            "memory_destination_snapshot",
            self._memory_destination_snapshot.checkpoint(),
        )

    def finish_review(self) -> None:
        self.db.set_status(JobStatus.COMPLETE)

    def close(self, *, cleanup: bool = False) -> None:
        self.db.close()
        if cleanup and self.temporary_workdir and not self.settings.keep_workdir:
            shutil.rmtree(self.work_dir)
