from pathlib import Path

from yakiflow.database import JobDatabase
from yakiflow.models import Cue


def test_transcript_replacement_preserves_completed_source_corrections(
    tmp_path: Path,
) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    db.upsert_cues([
        Cue("1", 0, 1, "raw draft", None),
        Cue("2", 1, 2, "corrected by Agent", "translated"),
    ])

    db.replace_transcript([
        Cue("1", 0.1, 1.1, "authoritative first", metadata={"final": True}),
        Cue("2", 1.1, 2.1, "raw authoritative", metadata={"final": True}),
    ])

    cues = db.list_cues(stable_only=True)
    assert (cues[0].source, cues[0].translated) == ("authoritative first", None)
    assert (cues[1].source, cues[1].translated) == (
        "corrected by Agent",
        "translated",
    )
    assert (cues[1].start, cues[1].end, cues[1].metadata) == (
        1.1,
        2.1,
        {"final": True},
    )
    db.close()


def test_aligned_timeline_replacement_is_authoritative_and_checkpointed(
    tmp_path: Path,
) -> None:
    db = JobDatabase(tmp_path / "job.sqlite3")
    db.upsert_cues([Cue("old", 0, 1, "old")])
    db.start_batch(["old"], "fake", "model", {"keep": True})

    db.replace_aligned_timeline([
        Cue("1", 0.1, 0.9, "first", metadata={"parent_id": "old"}),
        Cue("2", 0.9, 1.2, "second", metadata={"parent_id": "old"}),
    ])

    assert [cue.id for cue in db.list_cues(stable_only=True)] == ["1", "2"]
    assert db.get_checkpoint("alignment_complete") is True
    assert db.connection.execute("SELECT count(*) FROM translation_batches").fetchone()[0] == 1
    db.close()
