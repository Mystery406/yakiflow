from pathlib import Path

from yakiflow.models import Cue
from yakiflow.srt import publish_outputs, render_srt, timestamp


def test_timestamp_rounding_and_bilingual_order() -> None:
    cue = Cue("c1", 1.2345, 62.005, "原文", "Translation")
    text = render_srt([cue], "bilingual")
    assert "00:00:01,234 --> 00:01:02,005" in text
    assert "Translation\n原文" in text


def test_all_outputs(tmp_path: Path) -> None:
    paths = publish_outputs(
        tmp_path / "movie.final",
        [Cue("c1", 0, 1, "a", "b")],
        "all",
        source_language="zh",
        target_language="ja",
    )
    assert {path.name for path in paths} == {
        "movie.final.source.srt", "movie.final.translated.srt", "movie.final.zh-ja.srt"
    }
    assert all(path.exists() for path in paths)


def test_bilingual_output_normalizes_language_tags(tmp_path: Path) -> None:
    paths = publish_outputs(
        tmp_path / "movie.final",
        [Cue("c1", 0, 1, "a", "b")],
        "bilingual",
        source_language="en_US",
        target_language="zh-CN",
    )

    assert paths[0].name == "movie.final.en-us-zh-cn.srt"
