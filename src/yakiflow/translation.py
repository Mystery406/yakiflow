from __future__ import annotations

import asyncio
import inspect
import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from .config import Settings
from .database import JobDatabase
from .models import AgentTraceEvent, Cue
from .process import CommandRunner


ProgressCallback = Callable[[int, int], Awaitable[None]]
RetryCallback = Callable[[str], Awaitable[None]]
AgentTraceCallback = Callable[[AgentTraceEvent], Awaitable[None] | None]
_AGENT_OUTPUT_DETAIL_LIMIT = 100_000


@dataclass(slots=True)
class BackendTraceEvent:
    """A small, normalized event emitted by an Agent CLI adapter."""

    kind: str
    message: str
    event_id: str | None = None
    detail: str | None = None


BackendTraceCallback = Callable[[BackendTraceEvent], Awaitable[None] | None]


def _display_value(value: Any, *, limit: int = 2_000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        text = "\n".join(
            f"{key}: {_display_value(item, limit=400)}" for key, item in value.items()
        )
    elif isinstance(value, list):
        text = "\n".join(f"- {_display_value(item, limit=400)}" for item in value)
    else:
        text = str(value)
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n…"


def _structured_agent_message(text: str) -> bool:
    try:
        data = parse_json_response(text)
    except (ValueError, json.JSONDecodeError):
        return False
    return isinstance(data.get("cues"), list)


def _structured_output_detail(data: dict[str, Any]) -> str:
    """Keep the Agent's actual structured response inspectable in the TUI."""
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    if len(text) <= _AGENT_OUTPUT_DETAIL_LIMIT:
        return text
    omitted = len(text) - _AGENT_OUTPUT_DETAIL_LIMIT
    return (
        text[:_AGENT_OUTPUT_DETAIL_LIMIT].rstrip()
        + f"\n… {omitted:,} more characters omitted"
    )


def codex_trace_events(data: dict[str, Any]) -> list[BackendTraceEvent]:
    """Convert one ``codex exec --json`` object into display events."""
    event_type = str(data.get("type", ""))
    if event_type == "error":
        return [BackendTraceEvent("error", str(data.get("message", "Codex error")))]
    if event_type in {"turn.failed", "item.failed"}:
        error = data.get("error") or data.get("message") or "Agent operation failed"
        return [BackendTraceEvent("error", _display_value(error))]
    if not event_type.startswith("item."):
        return []
    item = data.get("item")
    if not isinstance(item, dict):
        return []
    item_type = str(item.get("type", ""))
    event_id = str(item["id"]) if item.get("id") is not None else None
    started = event_type == "item.started"
    completed = event_type == "item.completed"

    if item_type == "agent_message" and completed:
        text = str(item.get("text", "")).strip()
        if text and not _structured_agent_message(text):
            return [BackendTraceEvent("agent_message", text, event_id)]
        return []
    if item_type == "reasoning":
        return [BackendTraceEvent("thinking", "Thinking…", event_id)] if started else []
    if item_type == "command_execution":
        command = _display_value(item.get("command")) or "command"
        if started:
            return [BackendTraceEvent("tool_start", f"Run {command}", event_id)]
        if completed:
            output = item.get("aggregated_output") or item.get("output")
            exit_code = item.get("exit_code")
            suffix = f" · exit {exit_code}" if exit_code is not None else ""
            return [
                BackendTraceEvent(
                    "tool_result", f"Run {command}{suffix}", event_id,
                    _display_value(output) or None,
                )
            ]
    if item_type in {"mcp_tool_call", "tool_call"}:
        server = str(item.get("server", "")).strip()
        tool = str(item.get("tool") or item.get("name") or "tool")
        title = f"{server}.{tool}" if server else tool
        if started:
            return [
                BackendTraceEvent(
                    "tool_start", f"Call {title}", event_id,
                    _display_value(item.get("arguments") or item.get("input")) or None,
                )
            ]
        if completed:
            result = item.get("result") or item.get("output") or item.get("error")
            return [
                BackendTraceEvent(
                    "tool_result", f"Call {title}", event_id,
                    _display_value(result) or None,
                )
            ]
    if item_type == "web_search":
        query = _display_value(item.get("query")) or "web search"
        return [
            BackendTraceEvent(
                "tool_start" if started else "tool_result",
                f"Search web: {query}", event_id,
            )
        ] if started or completed else []
    if item_type == "file_change":
        return [
            BackendTraceEvent(
                "tool_start" if started else "tool_result",
                "Apply file changes", event_id,
                _display_value(item.get("changes")) or None,
            )
        ] if started or completed else []
    return []


def claude_trace_events(data: dict[str, Any]) -> list[BackendTraceEvent]:
    """Convert one Claude Code stream-json object into display events."""
    event_type = str(data.get("type", ""))
    if event_type in {"error", "system"} and data.get("error"):
        return [BackendTraceEvent("error", _display_value(data.get("error")))]
    message = data.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    events: list[BackendTraceEvent] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", ""))
        event_id = str(block["id"]) if block.get("id") is not None else None
        if event_type == "assistant" and block_type == "text":
            text = str(block.get("text", "")).strip()
            if text and not _structured_agent_message(text):
                events.append(BackendTraceEvent("agent_message", text, event_id))
        elif event_type == "assistant" and block_type in {"tool_use", "server_tool_use"}:
            name = str(block.get("name") or "tool")
            events.append(
                BackendTraceEvent(
                    "tool_start", f"Call {name}", event_id,
                    _display_value(block.get("input")) or None,
                )
            )
        elif event_type == "user" and block_type == "tool_result":
            tool_id = str(block.get("tool_use_id") or event_id or "") or None
            events.append(
                BackendTraceEvent(
                    "tool_result", "Tool result", tool_id,
                    _display_value(block.get("content")) or None,
                )
            )
    return events


def _translation_response_schema(
    title: str,
    description: str,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "id": {"type": "string"},
        "source": {"type": "string"},
        "translated": {"type": "string"},
    }
    return {
        "title": title,
        "description": description,
        "type": "object",
        "properties": {
            "cues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": ["id", "source", "translated"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["cues"],
        "additionalProperties": False,
    }


DRAFT_RESPONSE_SCHEMA = _translation_response_schema(
    "DraftTranslationResponse",
    "Corrected source text and a first-pass translation for every input cue.",
)


@dataclass(slots=True)
class TranslatedCueResult:
    source: str
    translated: str


@dataclass(slots=True)
class TranslationBatchResult:
    cues: dict[str, TranslatedCueResult]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationBatchResult:
        mapped = {
            str(item["id"]): TranslatedCueResult(
                str(item.get("source", "")),
                str(item.get("translated", "")),
            )
            for item in data.get("cues", [])
            if isinstance(item, dict) and "id" in item
        }
        return cls(mapped)


class AgentBackend(ABC):
    name: str

    @abstractmethod
    async def invoke_with_trace(
        self,
        prompt: str,
        *,
        model: str,
        effort: str,
        schema: dict[str, Any],
        on_event: BackendTraceCallback | None = None,
    ) -> dict[str, Any]: ...


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            data = json.loads(fence.group(1))
        else:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("agent did not return JSON")
            data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("agent response must be a JSON object")
    # Claude's JSON output wraps the assistant text in `result`.
    if isinstance(data.get("result"), str):
        return parse_json_response(data["result"])
    return data


class CodexBackend(AgentBackend):
    name = "codex"

    def __init__(self, work_dir: Path, runner: CommandRunner | None = None):
        self.work_dir = work_dir
        self.runner = runner or CommandRunner()

    async def invoke_with_trace(
        self,
        prompt: str,
        *,
        model: str,
        effort: str,
        schema: dict[str, Any],
        on_event: BackendTraceCallback | None = None,
    ) -> dict[str, Any]:
        token = uuid.uuid4().hex
        schema_path = self.work_dir / f"agent-schema-{token}.json"
        output_path = self.work_dir / f"agent-output-{token}.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        final_message: str | None = None

        async def line(stream: str, value: str) -> None:
            nonlocal final_message
            if stream != "stdout":
                return
            try:
                data = json.loads(value)
            except json.JSONDecodeError:
                return
            if not isinstance(data, dict):
                return
            item = data.get("item")
            if (
                data.get("type") == "item.completed"
                and isinstance(item, dict)
                and item.get("type") == "agent_message"
                and isinstance(item.get("text"), str)
            ):
                final_message = item["text"]
            if on_event is None:
                return
            for event in codex_trace_events(data):
                maybe = on_event(event)
                if inspect.isawaitable(maybe):
                    await maybe

        try:
            result = await self.runner.run(
                [
                    "codex", "exec", "--ephemeral", "--sandbox", "read-only",
                    "--skip-git-repo-check",
                    "--model", model, "--config", f'model_reasoning_effort="{effort}"',
                    "--json", "--output-schema", schema_path,
                    "--output-last-message", output_path, "-",
                ],
                cwd=self.work_dir,
                stdin=prompt.encode(),
                on_line=line,
            )
            text = (
                output_path.read_text(encoding="utf-8")
                if output_path.exists()
                else final_message or result.stdout
            )
            return parse_json_response(text)
        finally:
            schema_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


class ClaudeBackend(AgentBackend):
    name = "claude"

    def __init__(self, work_dir: Path, runner: CommandRunner | None = None):
        self.work_dir = work_dir
        self.runner = runner or CommandRunner()

    async def invoke_with_trace(
        self,
        prompt: str,
        *,
        model: str,
        effort: str,
        schema: dict[str, Any],
        on_event: BackendTraceCallback | None = None,
    ) -> dict[str, Any]:
        final_response: dict[str, Any] | str | None = None

        async def line(stream: str, value: str) -> None:
            nonlocal final_response
            if stream != "stdout":
                return
            try:
                data = json.loads(value)
            except json.JSONDecodeError:
                return
            if not isinstance(data, dict):
                return
            if data.get("type") == "result":
                candidate = data.get("structured_output")
                if candidate is None:
                    candidate = data.get("result")
                if isinstance(candidate, (dict, str)):
                    final_response = candidate
            if on_event is not None:
                for event in claude_trace_events(data):
                    maybe = on_event(event)
                    if inspect.isawaitable(maybe):
                        await maybe

        result = await self.runner.run(
            [
                "claude", "--print", "--verbose", "--output-format", "stream-json",
                "--json-schema", json.dumps(schema, ensure_ascii=False),
                "--model", model, "--effort", effort,
            ],
            cwd=self.work_dir,
            stdin=prompt.encode(),
            on_line=line,
        )
        if isinstance(final_response, dict):
            return final_response
        if isinstance(final_response, str):
            return parse_json_response(final_response)
        # Preserve a useful error for runners or older Claude versions that do
        # not emit the expected final result event.
        raise ValueError(f"Claude stream did not contain a structured result: {result.stderr[-500:]}")


def make_backend(name: str, work_dir: Path, runner: CommandRunner | None = None) -> AgentBackend:
    if name == "codex":
        return CodexBackend(work_dir, runner)
    if name == "claude":
        return ClaudeBackend(work_dir, runner)
    raise ValueError(f"unsupported agent backend: {name}")


class TranslationPipeline:
    def __init__(
        self,
        settings: Settings,
        backend: AgentBackend,
        db: JobDatabase,
        memory: str,
        on_retry: RetryCallback | None = None,
        on_agent_event: AgentTraceCallback | None = None,
    ):
        self.settings = settings
        self.backend = backend
        self.db = db
        self.memory = memory
        self.on_retry = on_retry
        self.on_agent_event = on_agent_event
        self._write_lock = asyncio.Lock()
        self._agent_semaphore = asyncio.Semaphore(settings.agent_workers)

    async def _emit_agent(
        self,
        operation_id: str,
        kind: str,
        message: str,
        *,
        state: str = "running",
        cues: Sequence[Cue] = (),
        model: str | None = None,
        attempt: int | None = None,
        max_attempts: int | None = None,
        event_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        if self.on_agent_event is None:
            return
        maybe = self.on_agent_event(
            AgentTraceEvent(
                operation_id=operation_id,
                kind=kind,
                message=message,
                state=state,
                cue_ids=tuple(cue.id for cue in cues),
                model=model,
                attempt=attempt,
                max_attempts=max_attempts,
                event_id=event_id,
                detail=detail,
            )
        )
        if inspect.isawaitable(maybe):
            await maybe

    @staticmethod
    def _request_summary(batch: Sequence[Cue], context: Sequence[Cue]) -> str:
        cue_range = (
            f"cue {batch[0].id}" if len(batch) == 1
            else f"cues {batch[0].id}–{batch[-1].id}"
        ) if batch else "no cues"
        context_suffix = f" · Context: {len(context)} cues" if context else ""
        return f"Draft translate {cue_range}{context_suffix}"

    @staticmethod
    def _result_summary(
        batch: Sequence[Cue], result: TranslationBatchResult
    ) -> tuple[str, str | None]:
        corrected = 0
        translated = 0
        shown = 0
        lines: list[str] = []
        for cue in batch:
            values = result.cues.get(cue.id)
            if values is None:
                continue
            source, target = values.source, values.translated
            if source.strip() and source.strip() != cue.source.strip():
                corrected += 1
            if target.strip() and target.strip() != (cue.translated or "").strip():
                translated += 1
            if shown < 20:
                lines.extend([
                    str(cue.id),
                    f"Source: {source.strip() or cue.source}",
                    f"Translation: {target.strip() or cue.translated or ''}",
                    "",
                ])
                shown += 1
        message = f"Processed {len(result.cues)} cues"
        facts = []
        if translated:
            facts.append(f"updated {translated} translations")
        if corrected:
            facts.append(f"corrected {corrected} source lines")
        if facts:
            message += " · " + " · ".join(facts)
        if len(result.cues) > 20:
            lines.append(f"… {len(result.cues) - 20} more cues")
        detail = "\n".join(lines).rstrip() or None
        return message, detail

    def _payload(
        self,
        batch: Sequence[Cue],
        context: Sequence[Cue],
    ) -> dict[str, Any]:
        cue_payloads: list[dict[str, Any]] = []
        for cue in batch:
            cue_payloads.append({
                "id": cue.id,
                "source": cue.source,
                "translated": cue.translated,
            })
        return {
            "source_language": self.settings.source_language,
            "target_language": self.settings.target_language,
            "memory": self.memory,
            "context": [{"id": c.id, "source": c.source, "translated": c.translated} for c in context],
            "cues": cue_payloads,
        }

    def _draft_prompt(
        self,
        batch: Sequence[Cue],
        context: Sequence[Cue],
        *,
        translate_only: bool = False,
    ) -> str:
        instruction = (
            "Translate naturally and do not modify, correct, merge, or split the source text. "
            if translate_only
            else "Correct only clear ASR errors and translate naturally "
        )
        introduction = (
            f"You are YakiFlow's draft subtitle translator. {instruction}"
            f"into the target language ({self.settings.target_language}). "
        )
        return (
            introduction
            + "Preserve every cue ID and cue order. Do not add facts or merge cues. "
            + "Use context only for continuity. Return corrected source text and a "
            "non-empty translation for every cue."
            + "\nINPUT:\n"
            + json.dumps(self._payload(batch, context), ensure_ascii=False)
        )

    async def translate_draft(
        self,
        cues: Sequence[Cue],
        on_batch: Callable[[list[Cue]], Awaitable[None]] | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        preceding_context: Sequence[Cue] = (),
        translate_only: bool = False,
    ) -> list[Cue]:
        batches = [
            list(cues[i:i + self.settings.translation_batch_size])
            for i in range(0, len(cues), self.settings.translation_batch_size)
        ]
        completed = 0

        async def run(index: int, batch: list[Cue]) -> None:
            nonlocal completed
            batch_start = index * self.settings.translation_batch_size
            available_context = [*preceding_context, *cues[:batch_start]]
            context = (
                available_context[-self.settings.translation_context:]
                if self.settings.translation_context
                else []
            )
            async with self._agent_semaphore:
                result = await self._run_draft_batch(
                    batch,
                    context,
                    self.settings.draft_model or "",
                    self.settings.draft_effort,
                    translate_only=translate_only,
                )
            async with self._write_lock:
                current = {cue.id: cue for cue in self.db.list_cues()}
                updated = self._apply(
                    batch,
                    result,
                    current,
                    allow_source_edits=not translate_only,
                )
                self.db.upsert_cues(updated)
                if on_batch:
                    await on_batch(self.db.list_cues(stable_only=True))
                completed += 1
                if on_progress:
                    await on_progress(completed, len(batches))

        tasks = [asyncio.create_task(run(i, batch)) for i, batch in enumerate(batches)]
        succeeded = False
        try:
            await asyncio.gather(*tasks)
            succeeded = True
        finally:
            if not succeeded:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        return self.db.list_cues(stable_only=True)

    async def _run_draft_batch(
        self,
        batch: Sequence[Cue],
        context: Sequence[Cue],
        model: str,
        effort: str,
        *,
        translate_only: bool = False,
    ) -> TranslationBatchResult:
        operation_id = uuid.uuid4().hex
        prompt = self._draft_prompt(
            batch,
            context,
            translate_only=translate_only,
        )
        schema = DRAFT_RESPONSE_SCHEMA
        attempts = self.settings.agent_max_attempts
        summary = self._request_summary(batch, context)
        await self._emit_agent(
            operation_id, "user_message", summary,
            cues=batch, model=model, attempt=1, max_attempts=attempts,
        )
        for attempt in range(1, attempts + 1):
            batch_id = self.db.start_batch(
                [cue.id for cue in batch],
                self.backend.name,
                model,
                {"prompt": prompt, "attempt": attempt, "max_attempts": attempts},
            )
            await self._emit_agent(
                operation_id, "lifecycle", summary,
                state="running", cues=batch, model=model,
                attempt=attempt, max_attempts=attempts,
            )

            async def backend_event(event: BackendTraceEvent) -> None:
                await self._emit_agent(
                    operation_id, event.kind, event.message,
                    state="running", cues=batch, model=model,
                    attempt=attempt, max_attempts=attempts,
                    event_id=event.event_id, detail=event.detail,
                )

            try:
                timeout = self.settings.draft_agent_timeout_seconds
                async with asyncio.timeout(timeout):
                    raw = await self.backend.invoke_with_trace(
                        prompt, model=model, effort=effort, schema=schema,
                        on_event=backend_event,
                    )
                raw_cues = raw.get("cues")
                cue_count = len(raw_cues) if isinstance(raw_cues, list) else 0
                await self._emit_agent(
                    operation_id,
                    "agent_output",
                    f"Structured response · {cue_count} "
                    f"{'cue' if cue_count == 1 else 'cues'}",
                    cues=batch,
                    model=model,
                    attempt=attempt,
                    max_attempts=attempts,
                    detail=_structured_output_detail(raw),
                )
                result = TranslationBatchResult.from_dict(raw)
                expected = {cue.id for cue in batch}
                unknown = set(result.cues) - expected
                if unknown:
                    raise ValueError(f"agent returned unknown cue IDs: {sorted(unknown)}")
                missing = expected - set(result.cues)
                if missing:
                    raise ValueError(f"agent omitted cue IDs: {sorted(missing)}")
                empty = [
                    cue.id
                    for cue in batch
                    if cue.id in result.cues
                    and not cue.translated
                    and not result.cues[cue.id].translated.strip()
                ]
                if empty:
                    raise ValueError(f"agent returned empty translations: {empty}")
                self.db.finish_batch(batch_id, raw)
                result_message, detail = self._result_summary(batch, result)
                await self._emit_agent(
                    operation_id, "result", result_message,
                    state="completed", cues=batch, model=model,
                    attempt=attempt, max_attempts=attempts, detail=detail,
                )
                await self._emit_agent(
                    operation_id, "lifecycle", result_message,
                    state="completed", cues=batch, model=model,
                    attempt=attempt, max_attempts=attempts,
                )
                return result
            except asyncio.CancelledError:
                self.db.finish_batch(batch_id, error="cancelled")
                await self._emit_agent(
                    operation_id, "lifecycle", "Agent operation cancelled",
                    state="cancelled", cues=batch, model=model,
                    attempt=attempt, max_attempts=attempts,
                )
                raise
            except Exception as exc:
                error = (
                    TimeoutError(
                        "agent draft batch timed out after "
                        f"{self.settings.draft_agent_timeout_seconds:g}s"
                    )
                    if isinstance(exc, TimeoutError)
                    else exc
                )
                self.db.finish_batch(batch_id, error=str(error))
                await self._emit_agent(
                    operation_id, "error", str(error),
                    state="retrying" if attempt < attempts else "failed",
                    cues=batch, model=model, attempt=attempt,
                    max_attempts=attempts,
                )
                if attempt == attempts:
                    await self._emit_agent(
                        operation_id, "lifecycle", str(error),
                        state="failed", cues=batch, model=model,
                        attempt=attempt, max_attempts=attempts,
                    )
                    if error is exc:
                        raise
                    raise error from exc
                await self._emit_agent(
                    operation_id, "lifecycle",
                    f"Retrying attempt {attempt + 1}/{attempts}",
                    state="retrying", cues=batch, model=model,
                    attempt=attempt + 1, max_attempts=attempts,
                )
                if self.on_retry:
                    await self.on_retry(
                        f"Agent draft batch failed: {error}; "
                        f"retrying {attempt + 1}/{attempts}"
                    )
                delay = self.settings.agent_retry_delay_seconds
                await asyncio.sleep(min(delay * 2 ** (attempt - 1), 8))
        raise RuntimeError("unreachable Agent retry state")

    @staticmethod
    def _apply(
        batch: Sequence[Cue],
        result: TranslationBatchResult,
        current: dict[str, Cue],
        *,
        allow_source_edits: bool = True,
    ) -> list[Cue]:
        output: list[Cue] = []
        for original in batch:
            cue = current.get(original.id, original)
            values = result.cues.get(cue.id)
            if values is None:
                output.append(cue)
                continue
            source, translated = values.source, values.translated
            metadata = dict(cue.metadata)
            output.append(Cue(
                cue.id, cue.start, cue.end,
                (source.strip() or cue.source) if allow_source_edits else cue.source,
                translated.strip() or cue.translated,
                cue.timing_confidence,
                metadata,
            ))
        return output
