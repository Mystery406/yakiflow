from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

from .config import DEFAULT_MODEL_NAME, default_model_path
from .srt import DEFAULT_FILE_MODE


MODEL_URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{DEFAULT_MODEL_NAME}"
MODEL_SHA256 = "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"


def fetch_model(destination: Path, progress: Callable[[int, int | None], None] | None = None) -> Path:
    destination = destination.expanduser().resolve()
    if destination.exists():
        if _sha256(destination) == MODEL_SHA256:
            return destination
        # Only this one published artifact is ever downloaded, so a different
        # existing file at the destination is somebody's own model. Repair the
        # default cache entry, but never overwrite a custom path.
        if destination != default_model_path().expanduser().resolve():
            raise ValueError(
                f"refusing to overwrite {destination}: it is not the default "
                f"{DEFAULT_MODEL_NAME} model, and 'models fetch' only downloads "
                "that model"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            digest = hashlib.sha256()
            with urllib.request.urlopen(MODEL_URL) as response:
                total_header = response.headers.get("Content-Length")
                total = int(total_header) if total_header else None
                received = 0
                while data := response.read(1024 * 1024):
                    output.write(data)
                    digest.update(data)
                    received += len(data)
                    if progress:
                        progress(received, total)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("downloaded model checksum does not match the published whisper.cpp artifact")
        os.chmod(temp_name, DEFAULT_FILE_MODE)
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

