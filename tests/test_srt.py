import stat
from pathlib import Path

from yakiflow.models import Cue
from yakiflow.srt import (
    alignment_problems,
    parse_srt_blocks,
    publish_outputs,
    render_srt,
    renumbered_srt,
    srt_problems,
    write_srt_atomic,
)


def _srt(*blocks: str) -> str:
    return "\n\n".join(blocks) + "\n"


def test_published_subtitles_are_readable_by_other_users(tmp_path: Path) -> None:
    reference = tmp_path / "reference.txt"
    reference.write_text("x")
    published = tmp_path / "movie.srt"
    write_srt_atomic(published, [Cue("1", 0.0, 1.0, "hello")], "source")
    assert stat.S_IMODE(published.stat().st_mode) == stat.S_IMODE(
        reference.stat().st_mode
    )


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


def test_parse_srt_blocks_reads_windows_line_endings_and_reports_bad_blocks() -> None:
    text = (
        "1\r\n00:00:00,000 --> 00:00:01,000\r\nhello\r\n"
        "\r\n"
        "reviewed by Agent\r\n"
    )

    blocks, problems = parse_srt_blocks(text)

    assert [block.text for block in blocks] == [("hello",)]
    assert blocks[0].start == 0.0 and blocks[0].end == 1.0
    assert [problem.kind for problem in problems] == ["unparsable"]


def test_srt_problems_finds_timing_defects() -> None:
    text = _srt(
        "1\n00:00:02,000 --> 00:00:01,000\nbackwards",
        "2\n00:00:01,500 --> 00:00:03,000\nout of order",
        "3\n00:00:02,000 --> 00:00:04,000\noverlapping",
    )

    assert [problem.kind for problem in srt_problems(text)] == [
        "end_before_start",
        "out_of_order",
        "overlap",
    ]


def test_srt_problems_accepts_a_well_formed_file() -> None:
    text = render_srt([Cue("c1", 0, 1, "a", "b"), Cue("c2", 1, 2, "c", "d")], "bilingual")

    assert srt_problems(text) == []


def test_renumbered_srt_repairs_numbering_left_by_a_merge() -> None:
    text = _srt(
        "1\n00:00:00,000 --> 00:00:01,000\nfirst",
        "3\n00:00:01,000 --> 00:00:02,000\nthird",
    )

    repaired = renumbered_srt(text)

    assert repaired is not None
    assert [block.number for block in parse_srt_blocks(repaired)[0]] == ["1", "2"]
    assert renumbered_srt(repaired) is None


def test_alignment_problems_detects_a_merge_applied_to_only_one_artifact() -> None:
    merged, _ = parse_srt_blocks(
        _srt("1\n00:00:00,000 --> 00:00:02,000\nmerged")
    )
    split, _ = parse_srt_blocks(_srt(
        "1\n00:00:00,000 --> 00:00:01,000\nfirst",
        "2\n00:00:01,000 --> 00:00:02,000\nsecond",
    ))
    retimed, _ = parse_srt_blocks(
        _srt("1\n00:00:00,000 --> 00:00:03,000\nmerged")
    )

    assert alignment_problems([("a.srt", merged)]) == []
    assert [
        problem.kind for problem in alignment_problems([("a.srt", merged), ("b.srt", split)])
    ] == ["artifact_mismatch"]
    assert [
        problem.kind
        for problem in alignment_problems([("a.srt", merged), ("b.srt", retimed)])
    ] == ["artifact_mismatch"]


def test_preview_reads_a_file_an_agent_edit_left_broken(tmp_path: Path) -> None:
    """The live preview must survive the defects publish-time validation reports."""
    from yakiflow.srt_preview import read_srt

    broken = tmp_path / "staged.srt"
    broken.write_text(
        _srt(
            "1\n00:00:05,000 --> 00:00:03,000\nends before it starts",
            "1\n00:00:06,000 --> 00:00:07,000\nduplicate number",
        ),
        encoding="utf-8",
    )
    cues = read_srt(broken)
    assert [cue.id for cue in cues] == ["1", "2"]
    assert all(cue.end >= cue.start for cue in cues)
