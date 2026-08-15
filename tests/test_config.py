from dataclasses import replace
from pathlib import Path

import pytest

from yakiflow.config import load_settings, validate_run_settings


def test_config_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user = tmp_path / "user.toml"
    project = tmp_path / "project.toml"
    user.write_text('target_language="fr"\nagent_workers=2\noutput_mode="source"\n')
    project.write_text('target_language="ja"\nagent_workers=3\n')
    settings = load_settings(
        {"target_language": "zh-CN", "agent_workers": 7, "translation_backend": "codex"},
        user_file=user,
        project_file=project,
    )
    assert settings.target_language == "zh-CN"
    assert settings.agent_workers == 7
    assert settings.output_mode == "source"
    assert settings.draft_model == "gpt-5.6-terra"
    assert settings.final_model == "gpt-5.6-sol"


def test_auto_open_video_setting_is_loaded(tmp_path: Path) -> None:
    config = tmp_path / "project.toml"
    config.write_text(
        'source_language = "en"\n'
        'target_language = "zh-CN"\n'
        'translation_backend = "codex"\n'
        'auto_open_video = true\n'
    )

    settings = load_settings({}, user_file=Path("/missing"), project_file=config)

    assert settings.auto_open_video is True


def test_yt_dlp_options_are_loaded_as_argv_tokens(tmp_path: Path) -> None:
    config = tmp_path / "project.toml"
    config.write_text(
        'yt_dlp_options = ["--cookies-from-browser", "chrome"]\n'
    )

    settings = load_settings({}, user_file=Path("/missing"), project_file=config)

    assert settings.yt_dlp_options == ("--cookies-from-browser", "chrome")


def test_required_translation_settings() -> None:
    settings = load_settings({}, user_file=Path("/missing"), project_file=Path("/missing"))
    with pytest.raises(ValueError, match="target-language"):
        validate_run_settings(settings)

    settings = load_settings(
        {"target_language": "zh-CN", "translation_backend": "codex"},
        user_file=Path("/missing"), project_file=Path("/missing"),
    )
    with pytest.raises(ValueError, match="source-language"):
        validate_run_settings(settings)


def test_auto_is_an_explicit_source_language() -> None:
    settings = load_settings(
        {"source_language": "auto", "target_language": "zh-CN", "translation_backend": "codex"},
        user_file=Path("/missing"), project_file=Path("/missing"),
    )
    validate_run_settings(settings)


def test_both_review_display_mode_requires_open_command() -> None:
    settings = load_settings(
        {
            "source_language": "en",
            "target_language": "zh-CN",
            "translation_backend": "codex",
            "review_display_mode": "both",
        },
        user_file=Path("/missing"),
        project_file=Path("/missing"),
    )

    with pytest.raises(ValueError, match="review-open-command"):
        validate_run_settings(settings)

    validate_run_settings(replace(settings, review_open_command="editor {srt}"))


def test_whisperx_alignment_settings_are_loaded_and_validated(tmp_path: Path) -> None:
    config = tmp_path / "yakiflow.toml"
    config.write_text(
        'source_language="en-US"\ntarget_language="zh-CN"\n'
        'translation_backend="codex"\nalignment_backend="whisperx"\n'
        'alignment_device="cuda"\nalignment_model="custom/model"\n'
    )
    settings = load_settings({}, user_file=Path("/missing"), project_file=config)

    validate_run_settings(settings)
    assert (settings.alignment_backend, settings.alignment_device) == ("whisperx", "cuda")
    assert settings.alignment_model == "custom/model"


@pytest.mark.parametrize(
    "content",
    [
        '[yakiflow]\ntarget_language = "zh-CN"\n',
        '[yakiflow.translation]\nbackend = "codex"\n',
    ],
)
def test_legacy_yakiflow_tables_are_rejected(
    tmp_path: Path, content: str
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(content)

    with pytest.raises(ValueError, match=r"move its settings.*document root"):
        load_settings({}, user_file=Path("/missing"), project_file=config)


def test_profile_precedence_merges_user_and_project_field_by_field(
    tmp_path: Path,
) -> None:
    user = tmp_path / "user.toml"
    project = tmp_path / "project.toml"
    user.write_text(
        'target_language = "fr"\n'
        'agent_workers = 2\n'
        'draft_codex_options = ["--user-base"]\n'
        '[profiles.stream]\n'
        'stream = true\n'
        'translation_backend = "codex"\n'
        'translation_batch_size = 5\n'
        'agent_workers = 3\n'
        'draft_codex_options = ["-c", "service_tier=fast"]\n'
    )
    project.write_text(
        'target_language = "ja"\n'
        'translation_batch_size = 10\n'
        '[profiles.stream]\n'
        'target_language = "zh-CN"\n'
        'agent_workers = 4\n'
        'final_codex_options = ["--search"]\n'
    )

    settings = load_settings(
        {"agent_workers": 7},
        user_file=user,
        project_file=project,
        profile="stream",
    )

    assert settings.stream is True
    assert settings.translation_backend == "codex"
    assert settings.translation_batch_size == 5
    assert settings.target_language == "zh-CN"
    assert settings.agent_workers == 7
    assert settings.draft_codex_options == ("-c", "service_tier=fast")
    assert settings.final_codex_options == ("--search",)


def test_unselected_profiles_have_no_effect(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        'stream = false\ntranslation_batch_size = 20\n'
        '[profiles.stream]\nstream = true\ntranslation_batch_size = 5\n'
    )

    settings = load_settings(
        {}, user_file=Path("/missing"), project_file=config
    )

    assert settings.stream is False
    assert settings.translation_batch_size == 20


def test_unknown_profile_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[profiles.stream]\nstream = true\n')

    with pytest.raises(ValueError, match="unknown profile 'missing'"):
        load_settings(
            {}, user_file=Path("/missing"), project_file=config, profile="missing"
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('profiles = "stream"\n', "profiles must be a table"),
        ('[profiles]\nstream = true\n', "profile 'stream' must be a table"),
        ('[profiles.stream.nested]\nstream = true\n', "cannot contain nested table"),
        ('[profiles.stream]\ninherits = "base"\n', "cannot inherit"),
    ],
)
def test_malformed_profile_tables_are_rejected(
    tmp_path: Path, content: str, message: str
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(content)

    with pytest.raises(ValueError, match=message):
        load_settings({}, user_file=Path("/missing"), project_file=config)


@pytest.mark.parametrize(
    "value",
    ['"--search"', '["--search", 1]', '[true]'],
)
def test_agent_options_require_arrays_of_strings(
    tmp_path: Path, value: str
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(f"draft_codex_options = {value}\n")

    with pytest.raises(ValueError, match="draft_codex_options must be an array of strings"):
        load_settings({}, user_file=Path("/missing"), project_file=config)


def test_yt_dlp_options_require_arrays_of_strings(tmp_path: Path) -> None:
    config = tmp_path / "project.toml"
    config.write_text('yt_dlp_options = ["--cookies-from-browser", 1]\n')

    with pytest.raises(ValueError, match="yt_dlp_options must be an array of strings"):
        load_settings({}, user_file=Path("/missing"), project_file=config)


def test_agent_retry_settings_are_validated() -> None:
    settings = load_settings(
        {
            "source_language": "auto",
            "target_language": "zh-CN",
            "translation_backend": "codex",
            "agent_max_attempts": 0,
        },
        user_file=Path("/missing"),
        project_file=Path("/missing"),
    )
    with pytest.raises(ValueError, match="attempt count"):
        validate_run_settings(settings)


@pytest.mark.parametrize(
    "field", ["translation_context", "translation_following_context"]
)
def test_negative_translation_context_is_rejected(field: str) -> None:
    settings = load_settings(
        {
            "source_language": "auto",
            "target_language": "zh-CN",
            "translation_backend": "codex",
            field: -1,
        },
        user_file=Path("/missing"),
        project_file=Path("/missing"),
    )

    with pytest.raises(ValueError, match="translation context"):
        validate_run_settings(settings)


def test_path_settings_are_canonicalized_when_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings(
        {
            "whisper_model": "models/whisper.bin",
            "vad_model": "models/vad.bin",
            "download_dir": "downloads",
            "output_dir": "subtitles",
            "memory": "state/memory.md",
            "context_files": ["references/chat.xml", "references/notes.txt"],
            "work_dir": "work",
        },
        user_file=Path("/missing"),
        project_file=Path("/missing"),
    )

    assert settings.whisper_model == tmp_path / "models/whisper.bin"
    assert settings.vad_model == tmp_path / "models/vad.bin"
    assert settings.download_dir == tmp_path / "downloads"
    assert settings.output_dir == tmp_path / "subtitles"
    assert settings.memory == tmp_path / "state/memory.md"
    assert settings.context_files == (
        tmp_path / "references/chat.xml",
        tmp_path / "references/notes.txt",
    )
    assert settings.work_dir == tmp_path / "work"


def test_vad_can_be_disabled_without_unsetting_the_model(tmp_path: Path) -> None:
    project_file = tmp_path / "yakiflow.toml"
    project_file.write_text('vad_model = "/models/vad.bin"\n', encoding="utf-8")

    enabled = load_settings(
        {}, user_file=Path("/missing"), project_file=project_file
    )
    disabled = load_settings(
        {"vad": False}, user_file=Path("/missing"), project_file=project_file
    )

    assert enabled.vad_model == Path("/models/vad.bin")
    assert disabled.vad_model is None


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"stream_chunk_seconds": 0}, "stream chunk duration"),
        ({"stream_chunk_seconds": -1}, "stream chunk duration"),
        ({"stream_context_seconds": -1}, "stream context duration"),
    ],
)
def test_stream_durations_are_validated(values: dict[str, int], message: str) -> None:
    settings = load_settings(
        {
            "source_language": "auto",
            "target_language": "zh-CN",
            "translation_backend": "codex",
            **values,
        },
        user_file=Path("/missing"),
        project_file=Path("/missing"),
    )

    with pytest.raises(ValueError, match=message):
        validate_run_settings(settings)
