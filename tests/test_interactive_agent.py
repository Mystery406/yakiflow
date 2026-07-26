from pathlib import Path

import pytest

from yakiflow.config import Settings
from yakiflow.interactive_agent import (
    build_interactive_command,
    build_interactive_prompt,
    build_memory_conflict_prompt,
)


@pytest.mark.parametrize(
    ("output_mode", "expected", "required_detail"),
    [
        (
            "source",
            "final deliverable is a source-language monolingual SRT",
            "Do not add a translation or a second language line",
        ),
        (
            "translated",
            "final deliverable is a target-language monolingual SRT",
            "do not add source-language lines",
        ),
        (
            "bilingual",
            "final deliverable is a bilingual SRT",
            "Fix both clear source transcription problems and target translation problems",
        ),
        (
            "all",
            "final deliverables include source-language monolingual, target-language monolingual, and bilingual SRTs",
            "Mirror every applicable text and timing correction across all three artifacts",
        ),
    ],
)
def test_interactive_prompt_scopes_review_to_final_output(
    tmp_path: Path,
    output_mode: str,
    expected: str,
    required_detail: str,
) -> None:
    prompt = build_interactive_prompt(
        Settings(
            source_language="en",
            target_language="zh-CN",
            output_mode=output_mode,
        ),
        tmp_path,
        [tmp_path / "movie.srt"],
    )

    normalized = " ".join(prompt.split())
    assert expected in normalized
    assert required_detail in normalized
    assert "source/translation evidence applicable to the output-specific scope" in normalized


def test_interactive_prompt_requires_proactive_novel_memory_candidates(
    tmp_path: Path,
) -> None:
    prompt = build_interactive_prompt(
        Settings(source_language="en", target_language="zh-CN"),
        tmp_path,
    )

    normalized = " ".join(prompt.split())
    assert "from your own review" in normalized
    assert "do not wait for a comment" in normalized
    assert "canonical names, proper nouns, domain terms, acronyms" in normalized
    assert "recurring ASR corrections or domain meanings" in normalized
    assert "durable translation and style preferences" in normalized
    assert "stable project or series context" in normalized
    assert "Exclude duplicates of existing memory" in normalized
    assert "obtain explicit approval" in normalized


def test_interactive_prompt_lists_read_only_context_files(tmp_path: Path) -> None:
    context_files = (
        tmp_path / "context" / "danmaku.xml",
        tmp_path / "context" / "speaker-notes.txt",
    )

    prompt = build_interactive_prompt(
        Settings(source_language="ja", target_language="zh-CN"),
        tmp_path,
        context_files=context_files,
    )

    assert str(context_files[0]) in prompt
    assert str(context_files[1]) in prompt
    assert "supporting context" in prompt
    assert "do not edit or publish them" in prompt


def test_memory_conflict_prompt_scopes_agent_to_staged_memory(tmp_path: Path) -> None:
    destination = tmp_path / "durable" / "memory.md"
    prompt = build_memory_conflict_prompt(
        tmp_path / "work",
        destination,
        "--- previous\n+++ current\n-old\n+new\n",
    )

    assert str(tmp_path / "work" / "memory.md") in prompt
    assert str(destination) in prompt
    assert "-old\n+new" in prompt
    assert "do not edit the destination" in prompt
    assert "avoid duplicate memory entries" in " ".join(prompt.split())


def test_interactive_commands_append_only_matching_final_options() -> None:
    codex = build_interactive_command(
        Settings(
            translation_backend="codex",
            final_model="final-codex",
            final_effort="high",
            final_codex_options=("-c", "service_tier=fast"),
            final_claude_options=("--dangerously-skip-permissions",),
        ),
        "codex prompt",
    )
    claude = build_interactive_command(
        Settings(
            translation_backend="claude",
            final_model="final-claude",
            final_effort="medium",
            final_codex_options=("--search",),
            final_claude_options=("--permission-mode", "plan"),
        ),
        "claude prompt",
    )

    assert codex == [
        "codex",
        "--sandbox",
        "workspace-write",
        "--model",
        "final-codex",
        "--config",
        'model_reasoning_effort="high"',
        "-c",
        "service_tier=fast",
        "codex prompt",
    ]
    assert claude == [
        "claude",
        "--model",
        "final-claude",
        "--effort",
        "medium",
        "--permission-mode",
        "plan",
        "claude prompt",
    ]
