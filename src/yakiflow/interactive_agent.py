"""The terminal hand-off used for all user-directed subtitle work.

YakiFlow's batch agents return structured data and are intentionally not
interactive.  Review, final polishing, and memory maintenance are different:
they need a conversation with the user and may require several iterations.
This module only starts the configured Agent CLI in the job directory; the
Agent owns that conversation and the resulting file edits.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import sys
import uuid
from pathlib import Path
from typing import Sequence

from .config import Settings
from .database import JobDatabase
from .media_player import open_media
from .models import OutputMode
from .process import CommandRunner, ProcessError, ProcessResult


class AgentFileDisplay:
    """A best-effort live, read-only view of the Agent's staging subtitle."""

    def __init__(self, marker: Path | None = None):
        self.marker = marker

    async def close(self) -> None:
        if self.marker is not None:
            self.marker.touch()


def _preview_command(
    file_path: Path,
    marker: Path,
    media_path: Path | None = None,
    video_open_command: str | None = None,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "yakiflow.srt_preview",
        str(file_path),
        str(marker),
        str(media_path) if media_path else "",
        video_open_command or "",
    ]


def _source_media(work_dir: Path) -> Path | None:
    db = JobDatabase(work_dir / "job.sqlite3")
    try:
        return db.artifact("source_media")
    finally:
        db.close()


def review_open_argv(command: str, file_path: Path, work_dir: Path) -> list[str]:
    """Expand a review-display command without invoking a shell."""
    has_srt = any(
        placeholder in command for placeholder in ("{srt}", "{file}", "{srt_file}")
    )
    memory_file = work_dir / "memory.md"
    replacements = {
        "{srt}": file_path,
        "{file}": file_path,
        "{srt_file}": file_path,
        "{workdir}": work_dir,
        "{memory}": memory_file,
        "{memory_file}": memory_file,
    }
    for placeholder, path in replacements.items():
        command = command.replace(placeholder, shlex.quote(str(path)))
    argv = shlex.split(command)
    if not has_srt:
        argv.append(str(file_path))
    return argv


async def start_agent_file_display(
    settings: Settings,
    work_dir: Path,
    outputs: Sequence[Path],
) -> AgentFileDisplay:
    """Open a live SRT preview using the configured display mode."""
    if not outputs:
        return AgentFileDisplay()
    file_path = outputs[0]
    media_path = _source_media(work_dir)
    marker = work_dir / ".agent-display-done"
    marker.unlink(missing_ok=True)
    if settings.review_display_mode == "open":
        command_text = settings.review_open_command
        if not command_text:
            raise ValueError("review_open_command is required for open display mode")
        command = review_open_argv(command_text, file_path, work_dir)
        await asyncio.create_subprocess_exec(*command, cwd=work_dir)
        return AgentFileDisplay()

    command = ["tmux"]
    # A split pane only makes sense from inside an active multiplexer
    # session. If tmux has no server/socket, leave the Agent terminal alone
    # instead of turning an optional preview into a failed review.
    if command and command[0] == "tmux" and not os.environ.get("TMUX"):
        return AgentFileDisplay()
    wide = shutil.get_terminal_size(fallback=(100, 24)).columns >= 120
    command.extend(("split-window", "-h" if wide else "-v"))
    if not wide:
        command.append("-b")
    command.extend(("--", *_preview_command(
        file_path, marker, media_path, settings.video_open_command
    )))
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=work_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        return AgentFileDisplay()
    await process.wait()
    if process.returncode:
        return AgentFileDisplay()
    return AgentFileDisplay(marker)


def build_interactive_prompt(
    settings: Settings,
    work_dir: Path,
    outputs: Sequence[Path] = (),
    context_files: Sequence[Path] = (),
) -> str:
    output_list = "\n".join(f"- {path}" for path in outputs) or (
        "- inspect the published subtitle files in the work directory"
    )
    memory = work_dir / "memory.md"
    context_section = ""
    if context_files:
        context_list = "\n".join(f"- {path}" for path in context_files)
        context_section = f"""
Additional reference files were copied into the work directory for this review:
{context_list}

Read these files when they are useful as supporting context. They are reference
material only: do not edit or publish them.
"""
    output_mode = OutputMode(settings.output_mode)
    review_scope = {
        OutputMode.SOURCE: f"""The final deliverable is a source-language monolingual SRT ({settings.source_language or 'the detected source language'}).
Fix clear ASR/transcription errors, source-text omissions or duplication,
punctuation, segmentation, formatting, timing, and source-term consistency.
Do not add a translation or a second language line to the final SRT.""",
        OutputMode.TRANSLATED: f"""The final deliverable is a target-language monolingual SRT ({settings.target_language or 'the configured target language'}).
Fix mistranslations, missing or duplicated meaning, unnatural phrasing,
terminology/style inconsistency, target-language punctuation and formatting,
and timing. Use source or Whisper data as evidence when needed, but keep the
final SRT target-language-only; do not add source-language lines.""",
        OutputMode.BILINGUAL: f"""The final deliverable is a bilingual SRT: target language ({settings.target_language or 'the configured target language'}) first and source language ({settings.source_language or 'the detected source language'}) second in each cue.
Fix both clear source transcription problems and target translation problems,
including omissions, duplication, terminology, punctuation, formatting, and
timing. Preserve that target-first/source-second layout and make each pair say
the same thing.""",
        OutputMode.ALL: f"""The final deliverables include source-language monolingual, target-language monolingual, and bilingual SRTs ({settings.source_language or 'the detected source language'} -> {settings.target_language or 'the configured target language'}).
Fix both clear source transcription problems and target translation problems,
including omissions, duplication, terminology, punctuation, formatting, and
timing. Mirror every applicable text and timing correction across all three
artifacts so they remain equivalent; keep the bilingual file target-first and
source-second.""",
    }[output_mode]
    return f"""You are the interactive YakiFlow subtitle editor.

Work directly in {work_dir}. The batch pipeline has finished and published these
subtitle artifacts:
{output_list}

The staged SRT files in this work directory are the files to inspect and edit.
YakiFlow will publish your edits after this session and remove the
destination-side draft snapshot.
{context_section}

The source language is {settings.source_language or 'detected automatically'} and
the target language is {settings.target_language or 'the configured target'}.

Review and editing scope for the configured final output:
{review_scope}

Start immediately with an autonomous review before waiting for user input:
inspect every staged SRT and {memory}, follow the output-specific scope above,
and check the complete timeline for relevant text, timing, formatting, and
cross-cue consistency problems. Report the initial findings in the terminal and
apply only clearly safe mechanical fixes before asking the user questions.
If a cue or timing is genuinely ambiguous, you may occasionally consult the raw
Whisper artifacts in the work directory (for example `whisper-*.json` or the
`process_logs` table in `job.sqlite3`), but do not routinely reread raw output
when the published subtitles already provide enough evidence.

Then work with the user conversationally until they are satisfied. Handle all
of the following in this one session:

1. Review: ask for and understand natural-language subtitle feedback. Inspect
   the staged subtitles and any source/translation evidence applicable to the
   output-specific scope yourself; do not require the user to provide cue IDs
   or line numbers. Whenever you point out or discuss a specific subtitle
   sentence/cue in the terminal, include its SRT start and end timestamp
   (for example, `00:01:23,456 --> 00:01:25,000`).
2. Refinement: apply agreed corrections to the subtitle artifacts, preserving
   SRT structure and any timing or formatting that does not need correction,
   and check the complete timeline for consistency.
3. Memory: proactively extract knowledge that will remain useful for future
   subtitle jobs and is not already in {memory}. Do this from your own review of
   the subtitles and corrections as well as from user comments; do not wait for
   a comment to mention a memory item. Focus on:
   - canonical names, proper nouns, domain terms, acronyms, and their preferred
     target renderings, capitalization, or explicit do-not-translate rules;
   - recurring ASR corrections or domain meanings that can resolve the same
     homophone, spelling, or ambiguity in later material;
   - durable translation and style preferences such as tone/register,
     honorifics and forms of address, locale/orthography, punctuation, and
     number, date, or unit conventions;
   - stable project or series context that affects repeated translation choices,
     such as speaker identities, relationships, or naming conventions.
   Exclude duplicates of existing memory, one-off cue fixes, plot details with
   no likely future use, obvious general language facts, and uncertain guesses.
   Present each new candidate and its reuse rationale to the user, obtain
   explicit approval, and only then write the approved item to {memory}. If
   nothing is genuinely reusable, say so rather than inventing candidates. Do
   not merely describe approved subtitle or memory edits--make them in the files.

You have permission to read and edit the files in the work directory. Keep
generated subtitle outputs and memory internally consistent.
When the user says the session is complete, summarize the changes and exit.
"""


def build_interactive_command(
    settings: Settings,
    prompt: str,
    *,
    additional_dirs: Sequence[Path] = (),
) -> list[str]:
    """Build the configured Agent CLI command in interactive mode."""
    model = settings.final_model or ""
    if settings.translation_backend == "codex":
        command = [
            "codex",
            "--sandbox",
            "workspace-write",
        ]
        for directory in additional_dirs:
            command.extend(("--add-dir", str(directory)))
        if model:
            command.extend(("--model", model))
        command.extend(
            ("--config", f'model_reasoning_effort="{settings.final_effort}"')
        )
        command.extend(settings.final_codex_options)
        command.append(prompt)
        return command
    if settings.translation_backend == "claude":
        command = ["claude"]
        if model:
            command.extend(("--model", model))
        command.extend(("--effort", settings.final_effort))
        command.extend(settings.final_claude_options)
        command.append(prompt)
        return command
    raise ValueError(f"unsupported agent backend: {settings.translation_backend}")


def build_memory_conflict_prompt(
    work_dir: Path,
    destination: Path,
    diff: str,
) -> str:
    memory = work_dir / "memory.md"
    return f"""You are resolving a concurrent YakiFlow translation-memory update.

Work directly in {work_dir}. The staged memory file is {memory}. While this job
was running, the durable destination at {destination} changed. Integrate the
destination-side changes shown below into the staged memory without losing the
valid edits already made in the staged file.

Inspect and edit only the staged memory file; do not edit the destination. Apply
the resolution in the file, preserve valid Markdown structure, avoid duplicate
memory entries, briefly summarize the merge, and exit. If the intent of a
change is genuinely ambiguous, ask the user before choosing.

<destination-memory-diff>
{diff.rstrip()}
</destination-memory-diff>
"""


async def run_memory_conflict_agent(
    settings: Settings,
    work_dir: Path,
    destination: Path,
    diff: str,
    *,
    runner: CommandRunner | None = None,
) -> ProcessResult:
    """Launch an interactive Agent to merge one destination memory revision."""
    runner = runner or CommandRunner()
    prompt = build_memory_conflict_prompt(work_dir, destination, diff)
    command = build_interactive_command(settings, prompt)
    return await runner.run_interactive(command, cwd=work_dir)


async def run_interactive_agent(
    settings: Settings,
    work_dir: Path,
    outputs: Sequence[Path] = (),
    *,
    context_files: Sequence[Path] = (),
    runner: CommandRunner | None = None,
) -> ProcessResult:
    """Hand the terminal to the configured interactive Agent CLI."""
    runner = runner or CommandRunner()
    prompt = build_interactive_prompt(settings, work_dir, outputs, context_files)
    command = build_interactive_command(
        settings, prompt
    )
    if settings.auto_open_video and outputs:
        open_media(
            settings.video_open_command,
            _source_media(work_dir),
            outputs[0],
            cwd=work_dir,
        )
    if (
        settings.review_display_mode == "split"
        and not os.environ.get("TMUX")
    ):
        return await _run_in_new_tmux_session(
            command, work_dir, outputs, _source_media(work_dir),
            settings.video_open_command,
        )
    return await runner.run_interactive(command, cwd=work_dir)


async def _run_in_new_tmux_session(
    command: Sequence[str],
    work_dir: Path,
    outputs: Sequence[Path],
    media_path: Path | None = None,
    video_open_command: str | None = None,
) -> ProcessResult:
    """Run Agent and preview in a tmux session created by YakiFlow itself."""
    session = f"yakiflow-review-{uuid.uuid4().hex[:8]}"
    marker = work_dir / ".agent-display-done"
    status_file = work_dir / ".agent-display-status"
    marker.unlink(missing_ok=True)
    status_file.unlink(missing_ok=True)
    file_path = outputs[0] if outputs else work_dir / "memory.md"
    preview = _preview_command(file_path, marker, media_path, video_open_command)
    wrapped_agent = (
        f"{shlex.join(command)}; status=$?; "
        f"printf '%s' \"$status\" > {shlex.quote(str(status_file))}; "
        f"touch {shlex.quote(str(marker))}; "
        f"tmux kill-session -t {shlex.quote(session)}; exit \"$status\""
    )
    try:
        created = await asyncio.create_subprocess_exec(
            "tmux", "new-session", "-d", "-s", session, "-c", str(work_dir),
            "--", "sh", "-lc", wrapped_agent,
            cwd=work_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await created.wait()
        if created.returncode:
            return await CommandRunner().run_interactive(command, cwd=work_dir)
        wide = shutil.get_terminal_size(fallback=(100, 24)).columns >= 120
        split_args = ["tmux", "split-window", "-h" if wide else "-v"]
        if not wide:
            split_args.append("-b")
        split = await asyncio.create_subprocess_exec(
            *split_args, "-t", session, "-c", str(work_dir),
            "--", *preview,
            cwd=work_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await split.wait()
        attached = await asyncio.create_subprocess_exec(
            "tmux", "attach-session", "-t", session, cwd=work_dir
        )
        await attached.wait()
    except OSError:
        return await CommandRunner().run_interactive(command, cwd=work_dir)
    try:
        returncode = int(status_file.read_text(encoding="utf-8") or "1")
    except (OSError, ValueError):
        returncode = attached.returncode or 1
    result = ProcessResult(tuple(command), returncode, "", "")
    if returncode:
        raise ProcessError(result)
    return result
