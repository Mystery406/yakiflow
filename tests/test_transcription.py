import asyncio
import http.client
from pathlib import Path
from types import SimpleNamespace

import pytest

import yakiflow.transcription as transcription_module
from yakiflow.config import Settings
from yakiflow.database import JobDatabase
from yakiflow.models import Cue
from yakiflow.transcription import (
    WhisperCliTranscriber,
    WhisperServerTranscriber,
    detected_source_language,
    merge_overlap,
    normalize_source_language,
    parse_json_full,
    parse_vad_segment,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("zh-CN", "zh"), ("en_US", "en"), ("JA", "ja"), ("auto", None), (None, None)],
)
def test_normalize_source_language(value, expected) -> None:
    assert normalize_source_language(value) == expected


def test_detected_language_reads_whisper_cpp_result() -> None:
    assert detected_source_language({"result": {"language": "pt-BR"}}) == "pt"


def test_cli_json_persists_detected_global_language(tmp_path: Path) -> None:
    prefix = tmp_path / "whisper-final"
    prefix.with_suffix(".json").write_text(
        '{"result":{"language":"zh-CN"},"transcription":[]}',
        encoding="utf-8",
    )
    db = JobDatabase(tmp_path / "job.sqlite3")
    transcriber = WhisperCliTranscriber(Settings(), tmp_path, db)

    assert transcriber._load_json(prefix) == []
    assert db.get_checkpoint("detected_source_language") == "zh"
    db.close()


def test_parse_whisper_json_and_overlap_deduplication() -> None:
    data = {
        "transcription": [
            {"timestamps": {"from": "00:00:01,000", "to": "00:00:02,000"}, "text": " hello "},
            {"offsets": {"from": 2500, "to": 4000}, "text": "world"},
        ]
    }
    cues = parse_json_full(data)
    assert [(cue.start, cue.end, cue.source) for cue in cues] == [
        (1, 2, "hello"),
        (2.5, 4, "world"),
    ]
    assert [cue.id for cue in cues] == ["1", "2"]
    merged = merge_overlap([cues[0]], [Cue("other", 1.2, 2.2, "HELLO"), cues[1]])
    assert [cue.source for cue in merged] == ["hello", "world"]
    assert len({cue.id for cue in merged}) == 2


def test_parse_integer_segment_times_as_seconds() -> None:
    cues = parse_json_full({
        "segments": [
            {"start": 5, "end": 7, "text": "integer seconds"},
        ]
    })

    assert [(cue.start, cue.end) for cue in cues] == [(5, 7)]


def test_parse_whisper_vad_segment_uses_original_timeline() -> None:
    line = (
        "whisper_vad_segments_from_probs: VAD segment 7: "
        "start = 41.67, end = 42.27 (duration: 0.60)"
    )

    assert parse_vad_segment(line) == (41.67, 42.27)
    assert parse_vad_segment(line, offset=100) == (141.67, 142.27)
    assert parse_vad_segment("vad time = 10 ms") is None


def test_interrupted_run_keeps_the_whole_vad_timeline(tmp_path: Path) -> None:
    # whisper.cpp prints its VAD pass in one burst, well inside the checkpoint
    # interval, and only then decodes for minutes. An interrupt during that
    # decode must not lose the timeline the incremental persist exists for.
    class BurstThenInterruptRunner:
        @staticmethod
        async def run(args, *, on_line=None, **kwargs):
            for index in range(3):
                await on_line(
                    "stderr",
                    f"whisper_vad_segments_from_probs: VAD segment {index}: "
                    f"start = {index * 10}.00, end = {index * 10 + 5}.00",
                )
            raise asyncio.CancelledError

    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"RIFF-fake")
    db = JobDatabase(tmp_path / "job.sqlite3")
    settings = Settings(vad_model=tmp_path / "vad.bin")
    transcriber = WhisperCliTranscriber(
        settings, tmp_path, db, runner=BurstThenInterruptRunner()
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(transcriber.transcribe(audio))

    assert db.get_checkpoint("whisper_vad_intervals") == [
        [0.0, 5.0], [10.0, 15.0], [20.0, 25.0]
    ]
    db.close()


def test_stream_auto_language_is_sent_on_every_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[bytes] = []
    paths: list[str] = []

    class Response:
        status = 200

        @staticmethod
        def read() -> bytes:
            return b'{"transcription": []}'

    class Connection:
        def __init__(self, host: str, port: int, timeout: int):
            pass

        def request(self, method: str, path: str, *, body: bytes, headers: dict[str, str]) -> None:
            bodies.append(body)
            paths.append(path)

        @staticmethod
        def getresponse() -> Response:
            return Response()

        @staticmethod
        def close() -> None:
            pass

    monkeypatch.setattr(http.client, "HTTPConnection", Connection)
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"RIFF-fake")
    db = JobDatabase(tmp_path / "job.sqlite3")
    server = WhisperServerTranscriber(Settings(source_language="auto"), tmp_path, db)

    server._post_audio(audio)
    server._post_audio(audio)

    language_field = b'name="language"\r\n\r\nauto\r\n'
    assert len(bodies) == 2
    assert all(language_field in body for body in bodies)
    assert len(set(paths)) == 1
    assert paths[0].startswith("/yakiflow-")
    assert paths[0].endswith("/inference")
    db.close()


def test_whisper_server_readiness_retries_until_listener_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    class Writer:
        def __init__(self) -> None:
            self.closed = False
            self.request = b""

        def write(self, value: bytes) -> None:
            self.request += value

        async def drain(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            assert self.closed

    writer = Writer()

    class Reader:
        @staticmethod
        async def readline() -> bytes:
            return b"HTTP/1.1 200 OK\r\n"

    async def open_connection(host: str, port: int):
        nonlocal attempts
        assert host == "127.0.0.1"
        assert port == 8178
        attempts += 1
        if attempts == 1:
            raise ConnectionRefusedError
        return Reader(), writer

    monkeypatch.setattr(asyncio, "open_connection", open_connection)
    db = JobDatabase(tmp_path / "job.sqlite3")
    server = WhisperServerTranscriber(Settings(), tmp_path, db, port=8178)
    server.process = SimpleNamespace(returncode=None)

    asyncio.run(server._wait_until_ready(timeout=1))

    assert attempts == 2
    assert writer.closed
    assert server.request_path.encode() + b"/inference" in writer.request
    db.close()


def test_whisper_server_allocates_ports_from_the_os(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = iter((42101, 42102))
    binds: list[tuple[str, int]] = []

    class Reservation:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def bind(self, address: tuple[str, int]) -> None:
            binds.append(address)

        def getsockname(self) -> tuple[str, int]:
            return "127.0.0.1", next(ports)

    monkeypatch.setattr(
        transcription_module.socket, "socket", lambda *_args: Reservation()
    )

    assert [
        transcription_module._allocate_loopback_port(),
        transcription_module._allocate_loopback_port(),
    ] == [42101, 42102]
    assert binds == [("127.0.0.1", 0), ("127.0.0.1", 0)]


def test_whisper_server_rejects_listener_when_launched_child_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Writer:
        def write(self, _value: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            pass

    process = SimpleNamespace(returncode=None)

    class Reader:
        @staticmethod
        async def readline() -> bytes:
            await asyncio.sleep(0.01)
            process.returncode = 1
            return b"HTTP/1.1 200 OK\r\n"

    async def open_connection(_host: str, _port: int):
        return Reader(), Writer()

    monkeypatch.setattr(asyncio, "open_connection", open_connection)
    db = JobDatabase(tmp_path / "job.sqlite3")
    server = WhisperServerTranscriber(Settings(), tmp_path, db, port=42103)
    server.process = process

    with pytest.raises(RuntimeError, match="exited during startup"):
        asyncio.run(server._wait_until_ready(timeout=1))
    db.close()


def test_whisper_server_rejects_a_listener_without_its_request_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Reader:
        @staticmethod
        async def readline() -> bytes:
            return b"HTTP/1.1 404 Not Found\r\n"

    class Writer:
        def write(self, _value: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            pass

    async def open_connection(_host: str, _port: int):
        return Reader(), Writer()

    monkeypatch.setattr(asyncio, "open_connection", open_connection)
    db = JobDatabase(tmp_path / "job.sqlite3")
    server = WhisperServerTranscriber(Settings(), tmp_path, db, port=42104)
    server.process = SimpleNamespace(returncode=None)

    with pytest.raises(RuntimeError, match="unexpected listener"):
        asyncio.run(server._wait_until_ready(timeout=1))
    db.close()


def test_overlap_deduplication_covers_the_whole_recovery_window() -> None:
    """Dense speech packs more cues into the 5 s overlap than a fixed count."""
    durable = [
        Cue(str(index + 1), index * 0.4, index * 0.4 + 0.4, f"line {index}")
        for index in range(30)
    ]
    resume_at = durable[-1].end - 5.0
    repeated = [cue for cue in durable if cue.end > resume_at]
    assert len(repeated) > 8
    merged = merge_overlap(durable, repeated)
    assert len(merged) == len(durable)
    assert [cue.source for cue in merged] == [cue.source for cue in durable]
