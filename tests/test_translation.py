import asyncio
import json
from pathlib import Path

import pytest

from yakiflow.config import Settings
from yakiflow.database import JobDatabase
from yakiflow.models import AgentTraceEvent, Cue
from yakiflow.process import ProcessResult
from yakiflow.translation import (
    AgentBackend,
    ClaudeBackend,
    CodexBackend,
    TranslationPipeline,
    claude_trace_events,
    codex_trace_events,
)


class RecordingAwaitable:
    def __init__(self, values: list, value: object) -> None:
        self.values = values
        self.value = value

    def __await__(self):
        self.values.append(self.value)
        if False:
            yield
        return None


class FakeBackend(AgentBackend):
    name = "fake"

    def __init__(self):
        self.prompts: list[str] = []
        self.schemas: list[dict] = []

    async def invoke_with_trace(self, prompt, *, model, effort, schema, on_event=None):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        response = {
            "cues": [{"id": "c1", "source": "agent source", "translated": "代理译文"}],
        }
        return response


class SequenceBackend(AgentBackend):
    name = "sequence"

    def __init__(self, translations: list[str]):
        self.translations = translations
        self.calls = 0

    async def invoke_with_trace(self, prompt, *, model, effort, schema, on_event=None):
        translated = self.translations[min(self.calls, len(self.translations) - 1)]
        self.calls += 1
        return {
            "cues": [{"id": "c1", "source": "hello", "translated": translated}],
        }


class TimeoutThenSuccessBackend(AgentBackend):
    name = "timeout"

    def __init__(self):
        self.calls = 0

    async def invoke_with_trace(self, prompt, *, model, effort, schema, on_event=None):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(1)
        return {
            "cues": [{"id": "c1", "source": "hello", "translated": "你好"}],
        }


def test_cli_trace_adapters_normalize_messages_and_tools() -> None:
    started = codex_trace_events({
        "type": "item.started",
        "item": {
            "id": "tool-1",
            "type": "command_execution",
            "command": "rg hello subtitles.srt",
        },
    })
    completed = codex_trace_events({
        "type": "item.completed",
        "item": {
            "id": "tool-1",
            "type": "command_execution",
            "command": "rg hello subtitles.srt",
            "exit_code": 0,
            "aggregated_output": "1:hello",
        },
    })
    hidden_json = codex_trace_events({
        "type": "item.completed",
        "item": {
            "id": "message-1",
            "type": "agent_message",
            "text": '{"cues": []}',
        },
    })
    claude = claude_trace_events({
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "Checking terminology."},
                {"type": "tool_use", "id": "tool-2", "name": "Read", "input": {"file": "memory.md"}},
            ]
        },
    })

    assert started[0].kind == "tool_start"
    assert completed[0].kind == "tool_result"
    assert completed[0].detail == "1:hello"
    assert hidden_json == []
    assert [event.kind for event in claude] == ["agent_message", "tool_start"]
    assert claude[1].detail == "file: memory.md"


def test_cli_backends_stream_trace_while_returning_structured_result(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"cues": {"type": "array"}},
        "required": ["cues"],
    }
    expected = {"cues": [{"id": "c1", "source": "hello", "translated": "你好"}]}

    class CodexRunner:
        async def run(self, args, *, on_line=None, **kwargs):
            assert "--json" in args
            output = Path(args[args.index("--output-last-message") + 1])
            output.write_text(json.dumps(expected))
            assert on_line is not None
            await on_line("stdout", json.dumps({
                "type": "item.started",
                "item": {"id": "tool", "type": "command_execution", "command": "pwd"},
            }))
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    class ClaudeRunner:
        async def run(self, args, *, on_line=None, **kwargs):
            assert args[args.index("--output-format") + 1] == "stream-json"
            assert "--json-schema" in args
            assert on_line is not None
            await on_line("stdout", json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Working on it."}]},
            }))
            await on_line("stdout", json.dumps({
                "type": "result", "structured_output": expected,
            }))
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    async def exercise():
        codex_events = []
        claude_events = []
        codex = await CodexBackend(tmp_path, CodexRunner()).invoke_with_trace(
            "prompt", model="model", effort="medium", schema=schema,
            on_event=lambda event: RecordingAwaitable(codex_events, event),
        )
        claude = await ClaudeBackend(tmp_path, ClaudeRunner()).invoke_with_trace(
            "prompt", model="model", effort="medium", schema=schema,
            on_event=lambda event: RecordingAwaitable(claude_events, event),
        )
        return codex, claude, codex_events, claude_events

    codex, claude, codex_events, claude_events = asyncio.run(exercise())
    assert codex == expected
    assert claude == expected
    assert codex_events[0].kind == "tool_start"
    assert claude_events[0].kind == "agent_message"


def test_draft_backend_commands_append_only_matching_options(tmp_path: Path) -> None:
    schema = {"type": "object"}
    expected = {"cues": []}
    commands: dict[str, list[object]] = {}

    class CodexRunner:
        async def run(self, args, **_kwargs):
            commands["codex"] = list(args)
            output = Path(args[args.index("--output-last-message") + 1])
            output.write_text(json.dumps(expected))
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    class ClaudeRunner:
        async def run(self, args, *, on_line=None, **_kwargs):
            commands["claude"] = list(args)
            assert on_line is not None
            await on_line(
                "stdout",
                json.dumps({"type": "result", "structured_output": expected}),
            )
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    async def exercise() -> None:
        await CodexBackend(
            tmp_path, CodexRunner(), ("-c", "service_tier=fast")
        ).invoke_with_trace(
            "prompt", model="draft-codex", effort="low", schema=schema
        )
        await ClaudeBackend(
            tmp_path, ClaudeRunner(), ("--permission-mode", "plan")
        ).invoke_with_trace(
            "prompt", model="draft-claude", effort="medium", schema=schema
        )

    asyncio.run(exercise())

    codex = commands["codex"]
    assert codex[-3:] == ["-c", "service_tier=fast", "-"]
    assert "--permission-mode" not in codex
    claude = commands["claude"]
    assert claude[-4:] == [
        "--effort",
        "medium",
        "--permission-mode",
        "plan",
    ]
    assert "service_tier=fast" not in claude


def test_pipeline_emits_agent_lifecycle_and_readable_result(tmp_path: Path) -> None:
    settings = Settings(target_language="zh-CN", draft_model="draft")
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "user source")])
    events: list[AgentTraceEvent] = []

    def on_agent_event(event: AgentTraceEvent) -> RecordingAwaitable:
        return RecordingAwaitable(events, event)

    pipeline = TranslationPipeline(
        settings, FakeBackend(), db, "", on_agent_event=on_agent_event
    )
    asyncio.run(pipeline.translate_draft(db.list_cues()))

    assert len({event.operation_id for event in events}) == 1
    assert [event.state for event in events if event.kind == "lifecycle"] == [
        "running",
        "completed",
    ]
    assert events[0].kind == "user_message"
    output = next(event for event in events if event.kind == "agent_output")
    assert output.message == "Structured response · 1 cue"
    assert '"source": "agent source"' in (output.detail or "")
    assert '"translated": "代理译文"' in (output.detail or "")
    result = next(event for event in events if event.kind == "result")
    assert events.index(output) < events.index(result)
    assert "updated 1 translations" in result.message
    assert "Translation: 代理译文" in (result.detail or "")
    db.close()


def test_translation_uses_draft_contract(tmp_path: Path) -> None:
    settings = Settings(
        target_language="zh-CN", translation_backend="codex", whisper_model=tmp_path / "m",
        memory=tmp_path / "memory.md", draft_model="draft", final_model="final",
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "user source")])
    backend = FakeBackend()
    pipeline = TranslationPipeline(settings, backend, db, "")
    result = asyncio.run(pipeline.translate_draft(db.list_cues()))
    assert result[0].source == "agent source"
    assert result[0].translated == "代理译文"
    assert "target language (zh-CN)" in backend.prompts[0]
    payload = json.loads(backend.prompts[0].split("INPUT:\n", 1)[1])
    assert set(payload["cues"][0]) == {"id", "source", "translated"}
    assert set(backend.schemas[0]["properties"]) == {"cues"}
    assert backend.schemas[0]["title"] == "DraftTranslationResponse"
    item_schema = backend.schemas[0]["properties"]["cues"]["items"]
    assert set(item_schema["properties"]) == {"id", "source", "translated"}
    db.close()


def test_post_alignment_translation_ignores_agent_source_edits(tmp_path: Path) -> None:
    settings = Settings(
        target_language="zh-CN",
        translation_backend="codex",
        draft_model="draft",
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "forced-aligned sentence")])
    backend = FakeBackend()
    pipeline = TranslationPipeline(settings, backend, db, "")

    result = asyncio.run(
        pipeline.translate_draft(db.list_cues(), translate_only=True)
    )

    assert result[0].source == "forced-aligned sentence"
    assert result[0].translated == "代理译文"
    assert "do not modify, correct, merge, or split the source text" in backend.prompts[0]
    db.close()


def test_empty_translation_is_retried_and_reported(tmp_path: Path) -> None:
    settings = Settings(
        target_language="zh-CN",
        draft_model="draft",
        agent_max_attempts=3,
        agent_retry_delay_seconds=0,
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "hello")])
    backend = SequenceBackend(["", "你好"])
    retries: list[str] = []

    async def on_retry(message: str) -> None:
        retries.append(message)

    pipeline = TranslationPipeline(settings, backend, db, "", on_retry=on_retry)
    result = asyncio.run(pipeline.translate_draft(db.list_cues()))

    assert backend.calls == 2
    assert result[0].translated == "你好"
    assert len(retries) == 1
    assert "empty translations" in retries[0]
    assert "retrying 2/3" in retries[0]
    statuses = [
        row["status"]
        for row in db.connection.execute(
            "SELECT status FROM translation_batches ORDER BY id"
        )
    ]
    assert statuses == ["failed", "complete"]
    db.close()


def test_timed_out_agent_batch_is_cancelled_and_retried(tmp_path: Path) -> None:
    settings = Settings(
        target_language="zh-CN",
        draft_model="draft",
        draft_agent_timeout_seconds=0.01,
        agent_max_attempts=2,
        agent_retry_delay_seconds=0,
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "hello")])
    backend = TimeoutThenSuccessBackend()
    retries: list[str] = []

    async def on_retry(message: str) -> None:
        retries.append(message)

    pipeline = TranslationPipeline(settings, backend, db, "", on_retry=on_retry)
    result = asyncio.run(pipeline.translate_draft(db.list_cues()))

    assert backend.calls == 2
    assert result[0].translated == "你好"
    assert "timed out after 0.01s" in retries[0]
    db.close()


def test_empty_translation_fails_after_retry_limit(tmp_path: Path) -> None:
    settings = Settings(
        target_language="zh-CN",
        draft_model="draft",
        agent_max_attempts=2,
        agent_retry_delay_seconds=0,
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "hello")])
    backend = SequenceBackend([""])
    pipeline = TranslationPipeline(settings, backend, db, "")

    with pytest.raises(ValueError, match="empty translations"):
        asyncio.run(pipeline.translate_draft(db.list_cues()))

    assert backend.calls == 2
    assert db.list_cues()[0].translated is None
    assert db.connection.execute(
        "SELECT COUNT(*) FROM translation_batches WHERE status = 'failed'"
    ).fetchone()[0] == 2
    db.close()
